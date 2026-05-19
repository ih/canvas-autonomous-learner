"""VERIFY through the same recorder pipeline as EXPLORE.

`verify_batch` plans N error-weighted probes (via explorer.pick_probe_*),
spawns the recorder subprocess with a probe_script that forces each
episode's starting position + direction, disconnects / reconnects the
motor bus around the subprocess, then replays each recorded episode
through `episode_canvas.process_recorded_episode` to compute MSE and
write a training-format action canvas.

This is the only remaining verification path. The old live-camera
`verify_once` helper was removed because its before/after frames and
motor-state sampling did not match the canvas-world-model training
distribution (different orientation, different sampling timing, no
motor strip encoding, no action separator), which meant verification
MSE was measuring something other than what the model was trained on.

`wide_verify_batch` extends the same machinery across multiple
operator-arranged scenes: probes are split into chunks, the operator
is prompted to physically rearrange between chunks (via the same
scene-ready dashboard mechanism the advisor's `idle` routing uses),
and the resulting LeRobot dataset paths are returned to the caller
for canvas-build + addition to the held-out wide-verify corpus.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import explorer, scene_prompts
from .episode_canvas import process_recorded_episode
from .gpu_monitor import sample_gpu
from .metrics import ProbeResult, RollingWindow
from .range_tracker import CurriculumState


def quantize_motor(motor_state: np.ndarray, bin_size: float = 10.0) -> str:
    """Coarse bin used as a state-key for histogramming the error landscape."""
    bins = (np.asarray(motor_state, dtype=np.float32) / bin_size).round().astype(int)
    return ",".join(str(b) for b in bins.tolist())


_ACTION_TO_DIRECTION = {1: "positive", 2: "negative", 3: "none"}


def _plan_probe_script(
    cfg,
    window: RollingWindow,
    curriculum: Optional[CurriculumState],
    num_probes: int,
) -> list[tuple[float, str]]:
    """Pick (start_pos, direction) tuples using the same error-weighted
    selection today's live verifier uses, but returning a script the
    recorder subprocess consumes one-per-episode instead of placing the
    arm directly."""
    control_joint = cfg.robot.control_joint

    if curriculum is not None:
        active = tuple(curriculum.active_range)
        active_joint_name = curriculum.active_joint_name
        active_joint_idx = curriculum.active_joint_idx
    else:
        active = (float(cfg.robot.joint_min), float(cfg.robot.joint_max))
        active_joint_name = control_joint
        _ensure_sibling = getattr(cfg.paths, "canvas_robot_control", None)
        import sys
        if _ensure_sibling and str(_ensure_sibling) not in sys.path:
            sys.path.insert(0, str(_ensure_sibling))
        from control.robot_interface import JOINTS  # type: ignore
        active_joint_idx = JOINTS.index(control_joint)

    candidates = list(cfg.actions.candidates)

    scripts: list[tuple[float, str]] = []
    for _ in range(num_probes):
        pos = explorer.pick_probe_state(
            window, active, control_joint_idx=active_joint_idx,
        )
        action = explorer.pick_probe_action(window, candidates)
        direction = _ACTION_TO_DIRECTION.get(int(action), "none")
        scripts.append((float(pos), direction))
    return scripts


def verify_batch(
    cfg,
    hardware,
    window: RollingWindow,
    curriculum: Optional[CurriculumState],
    prev_ckpt: Optional[str],
    cycle: int,
    examples_dir: Path,
    event_log=None,
    num_probes: Optional[int] = None,
) -> list[ProbeResult]:
    """Drive one VERIFY phase through the recorder pipeline.

    Plans `num_probes` error-weighted probe scripts, records them in a
    single subprocess invocation (same pipeline + canvas format as
    EXPLORE), then post-processes each episode with
    `process_recorded_episode` to get an MSE + an action canvas.

    Returns the list of successful probe results in cycle order.
    """
    if num_probes is None:
        num_probes = int(cfg.cadence.probes_per_verify)
    if num_probes <= 0:
        return []

    # Pooled-joint mode: the recorder randomizes the target joint per
    # episode (vary_target_joint=true). Pre-specified (start_pos,
    # direction) tuples don't map cleanly when we don't know which joint
    # the recorder will pick, so we skip the probe_script and let the
    # recorder use its own randomized starts inside each joint's range.
    # The acting joint for each probe is recovered from the motor delta
    # in process_recorded_episode and logged per-probe.
    pooled_verify = bool(
        getattr(getattr(cfg, "explore", None), "vary_target_joint", False)
        and getattr(getattr(cfg, "explore", None), "joints", None)
    )
    if pooled_verify:
        scripts = None
    else:
        scripts = _plan_probe_script(cfg, window, curriculum, num_probes)
    if event_log is not None:
        event_log.log(
            "verify_plan",
            cycle=cycle,
            probe_script=scripts,
            pooled=pooled_verify,
        )

    # Joint-range override: keep the primary free across the curriculum's
    # active range (the probe_script will pin each episode's start within
    # it anyway) but pin non-active curriculum joints to whatever range
    # the curriculum says EXPLORE uses for them.
    joint_range_override: dict = {}
    if curriculum is not None:
        joint_range_override = dict(curriculum.joint_range_override() or {})

    hardware.disconnect()
    try:
        verify_prefix = (
            getattr(getattr(cfg, "verify", None), "repo_id_prefix", None)
            or "auto/autonomous-verify"
        )
        dataset_dir = explorer.collect_batch_continuous(
            cfg,
            num_probes,
            window=window,
            event_log=event_log,
            joint_range_override=joint_range_override,
            probe_script=scripts,
            repo_id_prefix=verify_prefix,
            event_tag="verify_record_start",
        )
    finally:
        try:
            hardware.connect()
            if prev_ckpt is not None:
                hardware.load_predictor(prev_ckpt)
        except Exception as e:
            if event_log is not None:
                event_log.log("verify_reconnect_failed", error=str(e))

    if dataset_dir is None:
        if event_log is not None:
            event_log.log("verify_record_failed", cycle=cycle)
        return []

    predictor = getattr(hardware, "predictor", None)
    if predictor is None:
        if event_log is not None:
            event_log.log("verify_no_predictor", cycle=cycle)
        return []

    # Snapshot VRAM once before the probe loop so the advisor can see
    # how much headroom inference had at this cycle. Cheap: ~20 ms
    # nvidia-smi call, logged at most once per VERIFY.
    if event_log is not None:
        headroom = sample_gpu()
        if headroom is not None:
            event_log.log(
                "verify_gpu_headroom",
                cycle=cycle,
                used_mb=int(headroom["used_mb"]),
                total_mb=int(headroom["total_mb"]),
                used_frac=float(headroom["used_frac"]),
                util_pct=int(headroom["util_pct"]),
            )

    probes: list[ProbeResult] = []
    for probe_idx in range(num_probes):
        try:
            probe = process_recorded_episode(
                cfg,
                cache_dir=Path(dataset_dir),
                episode_index=probe_idx,
                predictor=predictor,
                examples_dir=examples_dir,
                cycle=cycle,
                filename_prefix="p",
            )
        except Exception as e:
            msg = str(e).lower()
            is_oom = (
                "out of memory" in msg
                or "outofmemoryerror" in msg
                or "cuda error: out of memory" in msg
            )
            if event_log is not None:
                event_log.log(
                    "inference_oom" if is_oom else "verify_probe_failed",
                    cycle=cycle, probe_idx=probe_idx, error=str(e),
                )
            continue
        if probe is None:
            continue
        probes.append(probe)
        if event_log is not None:
            event_log.log(
                "probe",
                cycle=cycle,
                action=probe.action,
                mse=probe.mse,
                state_key=probe.state_key,
                motor_state=list(probe.motor_state or []),
                acting_joint_idx=probe.acting_joint_idx,
                target_position=(
                    scripts[probe_idx][0] if scripts is not None else None
                ),
            )
    return probes


# ---------------------------------------------------- wide_verify_batch


def wide_verify_batch(
    cfg,
    hardware,
    window: RollingWindow,
    curriculum: Optional[CurriculumState],
    prev_ckpt: Optional[str],
    cycle: int,
    examples_dir: Path,
    *,
    num_scenes: int,
    num_probes: int,
    runs_dir: Path,
    event_log=None,
    stop: Optional[Callable[[], bool]] = None,
) -> tuple[list[ProbeResult], list[Path]]:
    """Run a wide VERIFY across `num_scenes` operator-arranged scenes.

    Plans `num_probes` probes total, splits them as evenly as possible
    into `num_scenes` chunks, then for each chunk:
      1. Prompts the operator (scene 2+ only — scene 1 reuses the
         scene already set up from the prior cycle) via
         `scene_prompts.prompt_dashboard` so the dashboard's existing
         banner / counter / tab-title alert all fire unchanged.
      2. Records that chunk's probes through the same recorder pipeline
         `verify_batch` uses, but writes to a stable wide-verify
         namespace (`{verify.wide_repo_id_prefix or "auto/wide-verify"}`)
         instead of the rolling-verify namespace, so the LeRobot
         datasets accumulate across cycles instead of being session-
         specific.
      3. Replays each episode through `process_recorded_episode` to
         compute probe MSE.

    Returns `(probes, lerobot_dirs)` — the probe results (which the
    caller adds to its rolling window same as quick-verify) plus the
    list of LeRobot dataset paths the caller should canvas-build and
    register in the wide-verify corpus. If the operator aborts via
    `stop()`, returns whatever was collected before the abort.

    Probe selection is identical to quick-verify (error-weighted via
    `explorer.pick_probe_*`). The same probe pool drives every scene's
    chunk — what makes wide-verify "wide" is the scene diversity of
    the recordings, not the probe-state distribution.
    """
    if num_scenes < 1:
        raise ValueError(f"num_scenes must be >= 1, got {num_scenes}")
    if num_probes < num_scenes:
        # If the caller asked for fewer probes than scenes, shrink the
        # scene count rather than producing empty chunks. Each scene
        # still gets at least one probe.
        num_scenes = max(1, num_probes)

    # Plan the full probe pool upfront, then slice per-scene.
    pooled_verify = bool(
        getattr(getattr(cfg, "explore", None), "vary_target_joint", False)
        and getattr(getattr(cfg, "explore", None), "joints", None)
    )
    if pooled_verify:
        full_scripts: list[tuple[float, str]] | None = None
    else:
        full_scripts = _plan_probe_script(cfg, window, curriculum, num_probes)

    if event_log is not None:
        event_log.log(
            "wide_verify_plan",
            cycle=cycle,
            num_scenes=num_scenes,
            num_probes=num_probes,
            probe_script=full_scripts,
            pooled=pooled_verify,
        )

    # Even split with the remainder distributed to the first chunks so
    # the total sums to num_probes and no chunk is empty.
    base_per_scene = num_probes // num_scenes
    remainder = num_probes - base_per_scene * num_scenes
    chunk_sizes = [
        base_per_scene + (1 if i < remainder else 0) for i in range(num_scenes)
    ]

    # Joint-range override matches verify_batch.
    joint_range_override: dict = {}
    if curriculum is not None:
        joint_range_override = dict(curriculum.joint_range_override() or {})

    wide_prefix = (
        getattr(getattr(cfg, "verify", None), "wide_repo_id_prefix", None)
        or "auto/wide-verify-corpus"
    )

    all_probes: list[ProbeResult] = []
    lerobot_dirs: list[Path] = []
    cursor = 0
    runs_dir = Path(runs_dir)

    for scene_idx in range(1, num_scenes + 1):
        chunk_n = chunk_sizes[scene_idx - 1]
        if chunk_n <= 0:
            continue

        # All scenes — including scene 1 — get a rearrangement prompt.
        # The operator's "current scene" from the prior cycle IS the
        # trained scene; recording wide-verify canvases against it
        # would dilute the held-out cross-scene signal by 1/N. Force
        # mode="rearrange" so scene 1's instruction is identical to
        # scenes 2..N: every canvas in the wide-verify corpus comes
        # from a setup the model has not been trained on.
        ack = scene_prompts.prompt_dashboard(
            runs_dir,
            scene_idx=scene_idx,
            total=num_scenes,
            cycle=cycle,
            event_log=event_log,
            stop=stop,
            mode="rearrange",
        )
        if not ack:
            if event_log is not None:
                event_log.log(
                    "wide_verify_aborted",
                    cycle=cycle, scene_idx=scene_idx,
                    reason="stop_requested_during_scene_change",
                )
            break

        chunk_scripts = (
            None if full_scripts is None
            else full_scripts[cursor:cursor + chunk_n]
        )
        cursor += chunk_n

        if event_log is not None:
            event_log.log(
                "wide_verify_chunk_start",
                cycle=cycle, scene_idx=scene_idx, scene_total=num_scenes,
                chunk_n=chunk_n,
            )

        hardware.disconnect()
        try:
            dataset_dir = explorer.collect_batch_continuous(
                cfg,
                chunk_n,
                window=window,
                event_log=event_log,
                joint_range_override=joint_range_override,
                probe_script=chunk_scripts,
                repo_id_prefix=f"{wide_prefix}/scene{scene_idx}_cycle{cycle}",
                event_tag="wide_verify_chunk_record_start",
            )
        finally:
            try:
                hardware.connect()
                if prev_ckpt is not None:
                    hardware.load_predictor(prev_ckpt)
            except Exception as e:
                if event_log is not None:
                    event_log.log(
                        "wide_verify_reconnect_failed", error=str(e),
                    )

        if dataset_dir is None:
            if event_log is not None:
                event_log.log(
                    "wide_verify_chunk_failed",
                    cycle=cycle, scene_idx=scene_idx,
                )
            continue
        lerobot_dirs.append(Path(dataset_dir))

        predictor = getattr(hardware, "predictor", None)
        if predictor is None:
            if event_log is not None:
                event_log.log(
                    "wide_verify_no_predictor",
                    cycle=cycle, scene_idx=scene_idx,
                )
            continue

        for probe_idx in range(chunk_n):
            try:
                probe = process_recorded_episode(
                    cfg,
                    cache_dir=Path(dataset_dir),
                    episode_index=probe_idx,
                    predictor=predictor,
                    examples_dir=examples_dir,
                    cycle=cycle,
                    filename_prefix=f"wv_s{scene_idx}",
                )
            except Exception as e:
                msg = str(e).lower()
                is_oom = (
                    "out of memory" in msg
                    or "outofmemoryerror" in msg
                    or "cuda error: out of memory" in msg
                )
                if event_log is not None:
                    event_log.log(
                        "inference_oom" if is_oom else "wide_verify_probe_failed",
                        cycle=cycle, scene_idx=scene_idx,
                        probe_idx=probe_idx, error=str(e),
                    )
                continue
            if probe is None:
                continue
            all_probes.append(probe)
            if event_log is not None:
                event_log.log(
                    "probe",
                    cycle=cycle,
                    action=probe.action,
                    mse=probe.mse,
                    state_key=probe.state_key,
                    motor_state=list(probe.motor_state or []),
                    acting_joint_idx=probe.acting_joint_idx,
                    target_position=(
                        chunk_scripts[probe_idx][0]
                        if chunk_scripts is not None else None
                    ),
                    source="wide_verify",
                    scene_idx=scene_idx,
                )

    if event_log is not None:
        event_log.log(
            "wide_verify_done",
            cycle=cycle,
            scenes_completed=len(lerobot_dirs),
            scenes_planned=num_scenes,
            probes_completed=len(all_probes),
            probes_planned=num_probes,
            lerobot_dirs=[str(d) for d in lerobot_dirs],
        )

    return all_probes, lerobot_dirs
