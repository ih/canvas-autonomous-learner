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
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import explorer
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
