"""Action selection + episode recording for EXPLORE bursts.

v0 action-selection policy: uniform over candidates, with optional bias toward
the highest-error action in the current window. Recording is a shell-out to
`run_single_action_record.py` in `robotic-foundation-model-tests` — that script
is deeply entangled with `lerobot-record`'s CLI, so wrapping it as a subprocess
is cleaner than replicating its setup. Output goes to the LeRobot v3.0 cache
under `~/.cache/huggingface/lerobot/<repo_id>` which is what `create_dataset.py`
already expects.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from .metrics import RollingWindow


# The recording subprocess uses LeRobot's tqdm progress bars. When an
# episode finishes, tqdm prints a line like
#   "Episode 47: 100%|##########| 1/1 [00:00<00:00, 24.28it/s]"
# Sometimes these get concatenated with the next episode's "Episode 48: 0%"
# line over a `\r` carriage return. We detect ANY occurrence of
# "Episode <n>: ... 100%" in a stdout chunk and emit a progress event
# for that episode number. Deduped on the most recently emitted number.
_EPISODE_DONE_RE = re.compile(r"Recording episode\s+(\d+)")

# The recorder logs one of these per episode right before execution, e.g.
#   "Next episode: Move shoulder pan positive by 10.0 units"
_EPISODE_ACTION_RE = re.compile(
    r"Next episode:\s*Move\s+(.+?)\s+(positive|negative)\s+by\s+([-\d.]+)\s+units"
)

# The recorder logs `Reset: commanding {'elbow_flex': 67.5, ...}` at the
# start of each episode — a clean snapshot of the commanded joint state.
_RESET_CMD_RE = re.compile(r"Reset(?:\s+retry\s+\d+)?:\s*commanding\s*(\{[^}]*\})")

# Streaming recorder progress lines (record_continuous.py). Each action
# produces one INFO log line like:
#   "2026-04-28 12:34:56,789 INFO root action 1/5 joint=shoulder_pan dir=positive
#    target=30.37 pre_settle=2 action=10 wall=2.45s"
# We anchor on "action N/M joint=" to avoid matching the legacy recorder's
# "action" word.
_STREAM_ACTION_RE = re.compile(
    r"action\s+(\d+)\s*/\s*(\d+)\s+joint=(\S+)\s+dir=(\S+)\s+target=([-\d.]+)"
)
# Verify checkpoint line, every `verify_every` actions:
#   "verify@N: <joint> cmd=X actual=Y err=Z"
_STREAM_VERIFY_RE = re.compile(
    r"verify@(\d+):\s+(\S+)\s+cmd=([-\d.]+)\s+actual=([-\d.]+)\s+err=([-\d.]+)"
)


def pick_probe_action(window: RollingWindow, candidates: list[int]) -> int:
    """Uniform for a cold window; otherwise weight toward high-MSE actions.

    Unseen actions are sampled first so the window has at least one datapoint
    per action before we start biasing.
    """
    if window.is_empty():
        return random.choice(candidates)

    per_action = window.per_action_mean()
    unseen = [a for a in candidates if a not in per_action]
    if unseen:
        return random.choice(unseen)

    mses = [per_action[a] for a in candidates]
    total = sum(mses)
    if total <= 0:
        return random.choice(candidates)
    weights = [m / total for m in mses]
    return random.choices(candidates, weights=weights, k=1)[0]


def pick_probe_state(
    window: RollingWindow,
    active_range: tuple[float, float],
    control_joint_idx: int,
    n_bins: int = 10,
    rng: Optional[random.Random] = None,
) -> float:
    """Pick a target motor position for the next probe, weighted by per-bin MSE.

    Divides `active_range` into `n_bins` equal segments, walks the window's
    probes whose `motor_state[control_joint_idx]` falls inside the active
    range, computes mean MSE per bin, and samples a bin with probability
    proportional to its mean MSE. Unvisited bins get `1.5 × max seen MSE`
    so they're the hottest — the selector attacks coverage holes first.

    Returns a uniform-random position inside the chosen bin. Cold window
    or no in-range probes → uniform sample across the whole active range.
    """
    rng = rng or random
    lo, hi = active_range
    if hi <= lo:
        return lo

    if window.is_empty():
        return rng.uniform(lo, hi)

    bin_width = (hi - lo) / n_bins
    bin_mses: list[list[float]] = [[] for _ in range(n_bins)]
    for r in window.snapshot():
        if r.motor_state is None:
            continue
        if control_joint_idx >= len(r.motor_state):
            continue
        pos = r.motor_state[control_joint_idx]
        if pos < lo or pos > hi:
            continue
        idx = min(n_bins - 1, max(0, int((pos - lo) / bin_width)))
        bin_mses[idx].append(r.mse)

    if all(not b for b in bin_mses):
        return rng.uniform(lo, hi)

    max_seen = max((max(b) for b in bin_mses if b), default=1.0)
    bin_means = [
        (sum(b) / len(b)) if b else max_seen * 1.5
        for b in bin_mses
    ]

    total = sum(bin_means)
    if total <= 0:
        return rng.uniform(lo, hi)
    weights = [m / total for m in bin_means]
    bin_idx = rng.choices(range(n_bins), weights=weights, k=1)[0]
    bin_lo = lo + bin_idx * bin_width
    bin_hi = bin_lo + bin_width
    return rng.uniform(bin_lo, bin_hi)


def plan_explore_sub_bursts(
    window: RollingWindow,
    active_range: tuple[float, float],
    control_joint_idx: int,
    total_episodes: int,
    max_sub_bursts: int = 3,
    min_sub_burst_size: int = 10,
    n_bins: int = 10,
    min_sub_burst_width: float = 0.0,
) -> list[tuple[int, tuple[float, float]]]:
    """Allocate an EXPLORE burst across the top-K highest-error state bins.

    Returns a list of `(episode_count, (lo, hi))` tuples whose episode
    counts sum to at most `total_episodes` (may sum to less after the
    min-sub-burst-size filter). Every sub-range is a bin inside the
    active range, and the sum of assigned episodes respects the
    `min_sub_burst_size` floor per sub-burst.

    `min_sub_burst_width` enforces that no sub-burst range is narrower
    than the given value. Callers pass `2 * position_delta` so that a
    single_action policy always has room for a valid ±delta step inside
    the sub-range — otherwise every episode gets flagged as a no-op and
    `create_dataset.py` crashes with "all actions are no-ops". If the
    active range is narrower than the min width, the function collapses
    to a single sub-burst covering the whole active range.

    Cold-start fallback: if the window has no probes whose starting
    state falls inside the active range (cold window, or right after a
    curriculum expansion), return a single sub-burst covering the
    whole active range. Uniform sampling until we have signal.
    """
    lo, hi = active_range
    if hi <= lo or total_episodes <= 0:
        return []

    if max_sub_bursts <= 1:
        return [(int(total_episodes), (float(lo), float(hi)))]

    # Cap n_bins so every bin is at least `min_sub_burst_width` wide.
    # Integer floor on (range_width / min_width) gives the max #bins that
    # can each honor the width. If we can only fit 1 bin, fall back to a
    # single sub-burst covering the full active range.
    range_width = hi - lo
    if min_sub_burst_width > 0:
        max_bins_by_width = int(range_width / min_sub_burst_width)
        if max_bins_by_width < 2:
            return [(int(total_episodes), (float(lo), float(hi)))]
        n_bins = min(n_bins, max_bins_by_width)
        max_sub_bursts = min(max_sub_bursts, max_bins_by_width)

    in_range = [
        r for r in window.snapshot()
        if r.motor_state is not None
        and control_joint_idx < len(r.motor_state)
        and lo <= r.motor_state[control_joint_idx] <= hi
    ]
    if not in_range:
        return [(int(total_episodes), (float(lo), float(hi)))]

    bin_width = (hi - lo) / n_bins
    bin_mses: list[list[float]] = [[] for _ in range(n_bins)]
    for r in in_range:
        pos = r.motor_state[control_joint_idx]
        idx = min(n_bins - 1, max(0, int((pos - lo) / bin_width)))
        bin_mses[idx].append(r.mse)

    max_seen = max((max(b) for b in bin_mses if b), default=1.0)
    bin_means = [
        (sum(b) / len(b)) if b else max_seen * 1.5
        for b in bin_mses
    ]

    # Top-K bins by mean MSE
    ranked = sorted(enumerate(bin_means), key=lambda x: -x[1])
    k = min(max_sub_bursts, n_bins)
    chosen = ranked[:k]
    total_weight = sum(w for _, w in chosen)
    if total_weight <= 0:
        return [(int(total_episodes), (float(lo), float(hi)))]

    allocations: list[tuple[int, tuple[float, float]]] = []
    allocated = 0
    for idx, (bin_idx, weight) in enumerate(chosen):
        if idx == len(chosen) - 1:
            n_eps = int(total_episodes) - allocated
        else:
            share = int(total_episodes * weight / total_weight)
            n_eps = max(min_sub_burst_size, share)
        allocated += n_eps
        sub_lo = float(lo + bin_idx * bin_width)
        sub_hi = float(sub_lo + bin_width)
        allocations.append((n_eps, (sub_lo, sub_hi)))

    # Fix rounding overshoot: if the min-floor padding pushed us over
    # `total_episodes`, trim the largest sub-burst.
    overshoot = sum(n for n, _ in allocations) - int(total_episodes)
    if overshoot > 0:
        biggest_idx = max(range(len(allocations)), key=lambda i: allocations[i][0])
        n, r = allocations[biggest_idx]
        allocations[biggest_idx] = (max(0, n - overshoot), r)

    # Drop any sub-burst below the floor and redistribute its budget to
    # the nearest remaining sub-burst (or collapse to a single sub-burst
    # if nothing survives).
    surviving = [(n, r) for n, r in allocations if n >= min_sub_burst_size]
    dropped = sum(n for n, _ in allocations if n < min_sub_burst_size)
    if not surviving:
        return [(int(total_episodes), (float(lo), float(hi)))]
    if dropped > 0:
        n, r = surviving[0]
        surviving[0] = (n + dropped, r)
    return surviving


def _cache_path_for_repo_id(repo_id: str) -> Path:
    return Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id


def _build_cameras_arg(cfg) -> str:
    r = cfg.robot
    base_name = cfg.explore.base_camera_name
    wrist_name = cfg.explore.wrist_camera_name
    return (
        "{ "
        f"{base_name}: {{type: opencv, index_or_path: {r.base_camera}, "
        f"width: {r.camera_width}, height: {r.camera_height}, fps: {r.camera_fps}, "
        "warmup_s: 2, rotation: ROTATE_180, backend: DSHOW}, "
        f"{wrist_name}: {{type: opencv, index_or_path: {r.wrist_camera}, "
        f"width: {r.camera_width}, height: {r.camera_height}, fps: {r.camera_fps}, "
        "warmup_s: 2, rotation: ROTATE_180, backend: DSHOW}"
        " }"
    )


def _session_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _joint_ranges_cli_arg(
    joint_range_override: dict[str, tuple[float, float]],
) -> str:
    """Serialize `joint_range_override` to a Hydra-style single flag value.

    LeRobot/Hydra will unwrap a JSON-ish dict literal for `--policy.joint_ranges=...`.
    Example output: `{shoulder_pan.pos: [-20.0, 20.0]}`. The policy's
    `SingleActionConfig.joint_ranges` is a `Dict[str, Tuple[float, float]]`
    and only the joints we pass will be overridden; unspecified joints
    keep `DEFAULT_JOINT_RANGES`.

    Uses a compact JSON-ish form without double quotes around keys (Hydra
    accepts both). If this turns out to misbehave empirically, the
    fallback is writing a temp YAML file and passing `--config_path`.
    """
    items = ", ".join(
        f"{joint}: [{lo}, {hi}]"
        for joint, (lo, hi) in joint_range_override.items()
    )
    return "{" + items + "}"


def collect_batch(
    cfg,
    num_episodes: int,
    window: RollingWindow | None = None,
    event_log=None,
    joint_range_override: dict[str, tuple[float, float]] | None = None,
    randomize_primary_start: bool | None = None,
    probe_script: list[tuple[float, str]] | None = None,
    repo_id_prefix: str | None = None,
    event_tag: str = "explore_start",
) -> Path | None:
    """Run one EXPLORE burst and return the LeRobot v3.0 dataset path.

    Args:
        num_episodes: number of episodes to record.
        joint_range_override: optional dict mapping `"<joint>.pos"` names
            (e.g. `"shoulder_pan.pos"`) to `(lo, hi)` tuples. When set,
            passes `--policy.joint_ranges={...}` so the single_action
            policy constrains its exploration to the narrowed range.
            Used by the curriculum to collect data in the currently-active
            state-space slice, and by the sub-bursting planner to target
            specific hot bins within that slice.
        randomize_primary_start: whether each episode should start from
            a fresh uniform-random position inside `joint_range_override`.
            Default (None): enable iff `joint_range_override` is set —
            the override implies we care about coverage inside the range,
            and randomization is how we get it.

    The robot must NOT be connected from another process — `run_single_action_record.py`
    opens its own FeetechMotorsBus and camera handles, so the orchestrator is
    expected to disconnect its hardware before calling this.
    """
    session = _session_stamp()
    if repo_id_prefix is None:
        repo_id_prefix = getattr(getattr(cfg, "explore", None), "repo_id_prefix", None) \
            or "auto/autonomous-explore"
    repo_id = f"{repo_id_prefix}-{session}"
    dataset_path = _cache_path_for_repo_id(repo_id)
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    rfmt_root = Path(cfg.paths.robotic_foundation_model_tests)
    script = rfmt_root / "scripts" / "run_single_action_record.py"
    if not script.exists():
        raise FileNotFoundError(f"record script not found: {script}")

    python_exe = cfg.paths.python or sys.executable
    cameras_arg = _build_cameras_arg(cfg)

    if randomize_primary_start is None:
        randomize_primary_start = joint_range_override is not None

    cmd = [
        python_exe,
        str(script),
        "--robot.type=so101_follower",
        f"--robot.port={cfg.robot.port}",
        f"--robot.id={cfg.robot.robot_id}",
        f"--robot.cameras={cameras_arg}",
        "--policy.type=single_action",
        f"--policy.joint_name={cfg.explore.policy_joint_name}",
        f"--policy.vary_target_joint={'true' if cfg.explore.vary_target_joint else 'false'}",
        f"--policy.position_delta={cfg.robot.step_size}",
        *(
            # Only emit --policy.secondary_joint_name when the caller
            # explicitly sets it. Needed to avoid collisions when
            # policy_joint_name matches the recorder's default secondary
            # (elbow_flex.pos) — e.g. the locked-val recorder's elbow run.
            [f"--policy.secondary_joint_name={cfg.explore.secondary_joint_name}"]
            if getattr(cfg.explore, "secondary_joint_name", None)
            else []
        ),
        f"--policy.action_duration={cfg.explore.action_duration}",
        f"--policy.start_buffer={getattr(cfg.explore, 'start_buffer', 2.5)}",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.fps={cfg.explore.dataset_fps}",
        "--dataset.push_to_hub=false",
    ]

    # When vary_target_joint=true the policy samples its target from
    # `config.joints`. Thread that list through as a Hydra/draccus list
    # literal: --policy.joints='[shoulder_pan.pos,elbow_flex.pos]'.
    joints_pool = getattr(cfg.explore, "joints", None)
    if joints_pool:
        joints_list = list(joints_pool)
        joints_csv = ",".join(joints_list)
        cmd.append(f"--policy.joints=[{joints_csv}]")

    # Start from any baseline ranges the config wants (e.g. pooled-joint
    # experiments that need safe elbow limits even when the curriculum
    # only names shoulder_pan as "active"). The orchestrator's
    # per-sub-burst override stacks on top so the active joint's narrow
    # bin wins for the acting joint.
    baseline_ranges = getattr(cfg.explore, "joint_ranges", None)
    merged_ranges: dict = {}
    if baseline_ranges:
        if hasattr(baseline_ranges, "__dict__"):
            src = vars(baseline_ranges)
        else:
            src = dict(baseline_ranges)
        for k, v in src.items():
            if v is None:
                continue
            lo, hi = v
            merged_ranges[str(k)] = (float(lo), float(hi))
    if joint_range_override:
        for k, v in joint_range_override.items():
            lo, hi = v
            merged_ranges[str(k)] = (float(lo), float(hi))

    if merged_ranges:
        cmd.append(f"--policy.joint_ranges={_joint_ranges_cli_arg(merged_ranges)}")
        # When a probe_script is supplied, each episode forces its own
        # primary start position, so randomize_primary_start MUST be off —
        # otherwise the policy would discard the forced start.
        effective_randomize = (
            False if probe_script is not None else randomize_primary_start
        )
        cmd.append(
            f"--policy.randomize_primary_start={'true' if effective_randomize else 'false'}"
        )

    import json as _json
    # Force the recorder to use the learner's configured home as its
    # starting-positions baseline. Otherwise the recorder reads live
    # Present_Position after the motor bus hand-off, which is wrong if the
    # arm drooped under gravity during the brief torque release.
    home_ns = getattr(cfg.robot, "home", None)
    if home_ns is not None:
        home_dict = {k: float(v) for k, v in vars(home_ns).items()}
        cmd.append(f"--starting-positions-json={_json.dumps(home_dict)}")

    if probe_script is not None:
        # Queue of [start_pos, direction] tuples consumed one per episode.
        # Used by VERIFY to drive error-weighted probes through the same
        # recorder pipeline as EXPLORE, so training canvases and verify
        # canvases come from the same code path.
        script_list = [[float(p), str(d)] for p, d in probe_script]
        cmd.append(f"--probe-script-json={_json.dumps(script_list)}")

    if event_log is not None:
        event_log.log(
            event_tag,
            repo_id=repo_id,
            episodes=num_episodes,
            joint_range_override=joint_range_override,
            randomize_primary_start=randomize_primary_start,
            probe_script=probe_script,
        )

    # Stream the recorder's stdout in this thread so we can parse
    # "Episode N: 100%" lines and emit `explore_episode_progress` events.
    # Without this, the learner process blocks for ~10-15 minutes during
    # each EXPLORE with zero new events — the dashboard looks frozen.
    # Also pipe "n" into stdin to answer the stale-cache prompt if it
    # ever re-appears.
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(rfmt_root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line buffered
        )
    except (OSError, FileNotFoundError) as e:
        if event_log is not None:
            event_log.log("explore_failed", repo_id=repo_id, error=str(e))
        return None

    # Answer the stale-cache prompt up front; the subprocess will block
    # on stdin until it reads something the first time (or we close it).
    try:
        if proc.stdin is not None:
            proc.stdin.write("n\n")
            proc.stdin.flush()
            proc.stdin.close()
    except (BrokenPipeError, OSError):
        pass

    last_emitted_episode: Optional[int] = None

    def _handle_line(line: str) -> None:
        nonlocal last_emitted_episode
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
        for m in _EPISODE_DONE_RE.finditer(line):
            ep = int(m.group(1))
            if last_emitted_episode is not None and ep <= last_emitted_episode:
                continue
            last_emitted_episode = ep
            if event_log is not None:
                event_log.log(
                    "explore_episode_progress",
                    repo_id=repo_id,
                    episode_index=ep,
                    total_episodes=int(num_episodes),
                )
        for m in _EPISODE_ACTION_RE.finditer(line):
            if event_log is not None:
                event_log.log(
                    "explore_action_taken",
                    repo_id=repo_id,
                    joint=m.group(1).strip().replace(" ", "_"),
                    direction=m.group(2),
                    magnitude=float(m.group(3)),
                )
        for m in _RESET_CMD_RE.finditer(line):
            try:
                import ast
                state = ast.literal_eval(m.group(1))
                if isinstance(state, dict) and event_log is not None:
                    event_log.log(
                        "explore_joint_state",
                        repo_id=repo_id,
                        state={str(k): float(v) for k, v in state.items()},
                    )
            except (ValueError, SyntaxError):
                pass

    try:
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            # Universal newlines splits on \n, \r, and \r\n so we see
            # tqdm updates as they land.
            line = raw_line.rstrip("\r\n")
            if line:
                _handle_line(line)
    finally:
        proc.wait()

    if proc.returncode != 0:
        if event_log is not None:
            event_log.log("explore_failed", repo_id=repo_id, returncode=proc.returncode)
        return None

    if not dataset_path.exists():
        if event_log is not None:
            event_log.log("explore_missing_output", repo_id=repo_id, expected=str(dataset_path))
        return None

    if event_log is not None:
        event_log.log("explore_done", repo_id=repo_id, path=str(dataset_path))
    return dataset_path


def collect_batch_continuous(
    cfg,
    num_episodes: int,
    window: RollingWindow | None = None,
    event_log=None,
    joint_range_override: dict[str, tuple[float, float]] | None = None,
    randomize_primary_start: bool | None = None,
    probe_script: list[tuple[float, str]] | None = None,
    repo_id_prefix: str | None = None,
    event_tag: str = "explore_start",
) -> Path | None:
    """Continuous-stream EXPLORE — same contract as `collect_batch`.

    Shells out to `scripts.streaming.record_continuous` in
    `robotic-foundation-model-tests` instead of the legacy
    `run_single_action_record.py`. Each "episode" is one action_duration
    window with a 200ms pre-action camera-flush settle (matches the
    Phase 1 default). All episodes share one chunked MP4 + parquet,
    bit-compatible with `canvas-world-model/create_dataset.py`.

    Drop-in replacement for `collect_batch`:
      - same signature and return type (`Path | None` to the dataset root)
      - emits the same `explore_start` / `explore_action_taken` /
        `explore_done` / `explore_failed` events (plus a streaming-
        specific `explore_action_progress` per action)
      - hardware MUST be disconnected by the orchestrator before calling

    Differences:
      - `randomize_primary_start` is silently ignored — the streaming
        sequencer produces an organic action stream where each new
        target derives from the *current* motor pose, not a random
        per-episode reset. Coverage of `joint_range_override` happens
        naturally as the sequence walks the range.
      - `probe_script` is unsupported and raises NotImplementedError.
        The verifier path always uses legacy `collect_batch` for forced
        per-episode start positions; only orchestrator EXPLORE bursts
        switch to streaming.
      - `window` is currently unused (the sequencer doesn't yet condition
        on the rolling error window — future enhancement).
    """
    if probe_script is not None:
        raise NotImplementedError(
            "probe_script is not supported by the continuous-stream recorder. "
            "Verifier flows must use collect_batch (legacy) for forced "
            "per-episode start positions."
        )
    del randomize_primary_start  # acknowledged but ignored — see docstring
    del window  # not yet used by the streaming sequencer

    session = _session_stamp()
    if repo_id_prefix is None:
        repo_id_prefix = getattr(getattr(cfg, "explore", None), "repo_id_prefix", None) \
            or "auto/autonomous-explore"
    repo_id = f"{repo_id_prefix}-{session}"
    dataset_path = _cache_path_for_repo_id(repo_id)
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    rfmt_root = Path(cfg.paths.robotic_foundation_model_tests)
    if not (rfmt_root / "scripts" / "streaming" / "record_continuous.py").exists():
        raise FileNotFoundError(
            f"streaming recorder not found under {rfmt_root}/scripts/streaming/. "
            "Phase 1 (rfmt commit d41535a or later) must be installed."
        )

    python_exe = cfg.paths.python or sys.executable

    # Joints pool: prefer cfg.explore.joints if set, else fall back to
    # the single policy_joint_name. Keeps single-joint configs working
    # without forcing them to add a `joints:` list to the YAML.
    joints_pool = getattr(cfg.explore, "joints", None)
    if joints_pool:
        joints = list(joints_pool)
    else:
        joints = [cfg.explore.policy_joint_name]

    # Merge baseline cfg.explore.joint_ranges with the per-burst override
    # the same way collect_batch does — overrides win.
    baseline_ranges = getattr(cfg.explore, "joint_ranges", None)
    merged_ranges: dict[str, tuple[float, float]] = {}
    if baseline_ranges:
        if hasattr(baseline_ranges, "__dict__"):
            src = vars(baseline_ranges)
        else:
            src = dict(baseline_ranges)
        for k, v in src.items():
            if v is None:
                continue
            lo, hi = v
            merged_ranges[str(k)] = (float(lo), float(hi))
    if joint_range_override:
        for k, v in joint_range_override.items():
            lo, hi = v
            merged_ranges[str(k)] = (float(lo), float(hi))

    pre_settle = float(getattr(cfg.explore, "pre_action_settle_duration", 0.2))

    cmd = [
        python_exe,
        "-m", "scripts.streaming.record_continuous",
        f"--robot-port={cfg.robot.port}",
        f"--robot-id={cfg.robot.robot_id}",
        f"--base-camera={cfg.robot.base_camera}",
        f"--wrist-camera={cfg.robot.wrist_camera}",
        f"--base-camera-name={cfg.explore.base_camera_name}",
        f"--wrist-camera-name={cfg.explore.wrist_camera_name}",
        f"--camera-width={cfg.robot.camera_width}",
        f"--camera-height={cfg.robot.camera_height}",
        f"--camera-fps={cfg.robot.camera_fps}",
        f"--num-actions={int(num_episodes)}",
        f"--action-duration={cfg.explore.action_duration}",
        f"--pre-action-settle-duration={pre_settle}",
        f"--fps={cfg.explore.dataset_fps}",
        "--joints", *joints,
        f"--position-delta={cfg.robot.step_size}",
        f"--output-repo-id={repo_id}",
    ]

    for joint, (lo, hi) in merged_ranges.items():
        cmd.extend(["--joint-range", str(joint), str(float(lo)), str(float(hi))])

    # If the config pins joints[0] for vary_target=False (single-joint
    # mode), pass --no-vary-target so the sequencer never wanders.
    if not getattr(cfg.explore, "vary_target_joint", True):
        cmd.append("--no-vary-target")

    # Match legacy: pass cfg.robot.home as starting positions so the
    # subprocess snaps the arm to a known pose before the first action.
    home_ns = getattr(cfg.robot, "home", None)
    if home_ns is not None:
        home_dict = {k: float(v) for k, v in vars(home_ns).items()}
        cmd.append(f"--starting-positions-json={json.dumps(home_dict)}")

    if event_log is not None:
        event_log.log(
            event_tag,
            repo_id=repo_id,
            episodes=int(num_episodes),
            joint_range_override=joint_range_override,
            mode="continuous",
            pre_action_settle_duration=pre_settle,
        )

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(rfmt_root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge so the regex sees logging.INFO lines
            text=True,
            bufsize=1,
        )
    except (OSError, FileNotFoundError) as e:
        if event_log is not None:
            event_log.log("explore_failed", repo_id=repo_id, error=str(e))
        return None

    # Streaming recorder doesn't prompt for input, but close stdin so it
    # doesn't ever block on a stray read.
    try:
        if proc.stdin is not None:
            proc.stdin.close()
    except (BrokenPipeError, OSError):
        pass

    last_emitted_action: Optional[int] = None

    def _handle_line(line: str) -> None:
        nonlocal last_emitted_action
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
        for m in _STREAM_ACTION_RE.finditer(line):
            action_idx = int(m.group(1))
            total = int(m.group(2))
            joint = m.group(3)
            direction = m.group(4)
            target_pos = float(m.group(5))
            if last_emitted_action is None or action_idx > last_emitted_action:
                last_emitted_action = action_idx
                if event_log is not None:
                    # Same shape as legacy `explore_episode_progress` so
                    # the dashboard's progress bar Just Works without
                    # branching on mode.
                    event_log.log(
                        "explore_episode_progress",
                        repo_id=repo_id,
                        episode_index=action_idx - 1,  # 0-indexed
                        total_episodes=total,
                    )
                    event_log.log(
                        "explore_action_taken",
                        repo_id=repo_id,
                        joint=joint,
                        direction=direction,
                        magnitude=float(cfg.robot.step_size),
                    )
                    event_log.log(
                        "explore_action_progress",
                        repo_id=repo_id,
                        action_index=action_idx,
                        total_actions=total,
                        joint=joint,
                        direction=direction,
                        target_pos=target_pos,
                    )
        for m in _STREAM_VERIFY_RE.finditer(line):
            if event_log is not None:
                event_log.log(
                    "explore_verify_checkpoint",
                    repo_id=repo_id,
                    after_action=int(m.group(1)),
                    joint=m.group(2),
                    cmd=float(m.group(3)),
                    actual=float(m.group(4)),
                    err=float(m.group(5)),
                )

    try:
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.rstrip("\r\n")
            if line:
                _handle_line(line)
    finally:
        proc.wait()

    if proc.returncode != 0:
        if event_log is not None:
            event_log.log("explore_failed", repo_id=repo_id, returncode=proc.returncode)
        return None

    if not dataset_path.exists():
        if event_log is not None:
            event_log.log("explore_missing_output", repo_id=repo_id, expected=str(dataset_path))
        return None

    if event_log is not None:
        event_log.log("explore_done", repo_id=repo_id, path=str(dataset_path))
    return dataset_path
