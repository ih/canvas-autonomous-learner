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

    # Cap max_sub_bursts so the episode budget can honor the per-sub-burst
    # floor. Without this, a budget too small relative to the floor lets
    # the inner allocator overrun: the last sub-burst's `n_eps =
    # total_episodes - allocated` goes negative, the negative entry is
    # later dropped by the floor filter, and the surviving entries —
    # already each pinned at the floor — sum to MORE than total_episodes.
    # Cap up front so we either return fewer sub-bursts or a single
    # uniform fallback.
    if min_sub_burst_size > 0:
        max_sub_bursts_by_budget = max(1, int(total_episodes) // int(min_sub_burst_size))
        max_sub_bursts = min(max_sub_bursts, max_sub_bursts_by_budget)
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


def plan_per_joint_sub_bursts(
    per_cell_mse: dict | None,
    joint_pool: list[str],
    joint_ranges: dict[str, tuple[float, float]],
    total_episodes: int,
    max_sub_bursts: int = 3,
    min_sub_burst_size: int = 10,
    min_sub_burst_width: float = 0.0,
    unvisited_bonus: float = 1.5,
) -> list[tuple[int, str, tuple[float, float]]]:
    """Plan per-(joint, position-bin) sub-bursts from a per-cell MSE map.

    Companion to `plan_explore_sub_bursts` for the multi-joint case.
    Reads the per-cell MSE histogram produced by `evaluate.py` (sub-phase 2)
    and returns up to `max_sub_bursts` `(n_eps, joint, (lo, hi))` triples,
    each pinning EXPLORE to a single joint inside one of its hot
    position bins.

    Args:
        per_cell_mse: dict of `joint_name -> [{bin, lo, hi, mean_mse, count}]`
            as written by evaluate.py. None or empty falls back to a
            uniform per-joint allocation across `joint_pool`.
        joint_pool: list of joint names eligible for targeting (e.g.
            ["shoulder_pan", "elbow_flex"]). Joints outside this pool
            are ignored even if they appear in per_cell_mse.
        joint_ranges: per-joint full active range. Used as the fallback
            range when a joint has no measured cells.
        total_episodes: budget across all sub-bursts.
        max_sub_bursts: cap on the number of `(joint, bin)` pairs returned.
        min_sub_burst_size: floor per sub-burst. Capped by total_episodes
            // min_sub_burst_size as in the 1D planner.
        min_sub_burst_width: skip cells narrower than this (typically
            2 × position_delta to ensure a valid step inside the bin).
        unvisited_bonus: weight multiplier on `max(observed_mean_mse) *
            bonus` for joints that appear in joint_pool but have no
            measured cells. Encourages exploration of under-probed joints.

    Returns:
        list of `(n_eps, joint_name, (lo, hi))` tuples summing to at
        most total_episodes (may sum to less after the floor filter).

    Empty histogram fallback:
        - All joints in pool look "unvisited" → uniform per-joint
          allocation across max_sub_bursts (or fewer if budget too small).
        - Each unvisited joint gets a sub-burst spanning its full range.
    """
    if total_episodes <= 0 or not joint_pool:
        return []

    if max_sub_bursts <= 0:
        return []

    if min_sub_burst_size > 0:
        max_sub_bursts_by_budget = max(1, int(total_episodes) // int(min_sub_burst_size))
        max_sub_bursts = min(max_sub_bursts, max_sub_bursts_by_budget)

    # Build a flat list of candidate cells.
    # Visited cells: from per_cell_mse, with their actual mean_mse.
    # Unvisited joints: synthetic cell spanning full range, scored as
    #   max_observed_mean * unvisited_bonus (or 1.0 if nothing observed).
    #
    # Cell ranges are intersected with joint_ranges (the configured safe
    # explore range per joint) so the recorder never gets commanded to
    # drive a motor outside its mechanical safe zone. evaluate.py's
    # per-cell bins are computed over motor_norm_min/max ([-100, 100]
    # by default) which is far wider than the safe range for joints
    # like elbow_flex (configured [50, 90]); without clamping, a hot
    # cell at bin boundary [40, 60] would let the sequencer drive the
    # elbow to 40, below the safe minimum.
    candidates: list[dict] = []  # each: {joint, lo, hi, score}
    pool_set = set(joint_pool)
    cells_by_joint: dict[str, list[dict]] = {}
    if per_cell_mse:
        for joint, cells in per_cell_mse.items():
            if joint not in pool_set:
                continue
            safe_range = joint_ranges.get(joint)
            for c in cells:
                lo = float(c.get("lo"))
                hi = float(c.get("hi"))
                if hi <= lo:
                    continue
                # Intersect with the configured safe range. If the cell
                # falls entirely outside the safe range or has zero
                # remaining width after clamping, drop it.
                if safe_range is not None:
                    safe_lo, safe_hi = float(safe_range[0]), float(safe_range[1])
                    lo = max(lo, safe_lo)
                    hi = min(hi, safe_hi)
                    if hi <= lo:
                        continue
                if min_sub_burst_width > 0 and (hi - lo) < min_sub_burst_width:
                    continue
                cell = {
                    "joint": joint,
                    "lo": lo,
                    "hi": hi,
                    "score": float(c.get("mean_mse", 0.0)),
                }
                candidates.append(cell)
                cells_by_joint.setdefault(joint, []).append(cell)

    max_observed = max((c["score"] for c in candidates), default=0.0)
    unvisited_score = max(max_observed * unvisited_bonus, 1.0)

    for joint in joint_pool:
        if joint in cells_by_joint:
            continue
        full_range = joint_ranges.get(joint)
        if full_range is None:
            continue
        lo, hi = float(full_range[0]), float(full_range[1])
        if hi <= lo:
            continue
        if min_sub_burst_width > 0 and (hi - lo) < min_sub_burst_width:
            continue
        candidates.append({
            "joint": joint, "lo": lo, "hi": hi, "score": unvisited_score,
        })

    if not candidates:
        return []

    # Top-K by score
    candidates.sort(key=lambda c: -c["score"])
    chosen = candidates[:max_sub_bursts]
    total_weight = sum(c["score"] for c in chosen)
    if total_weight <= 0:
        # All zero scores — uniform allocation across chosen.
        n_each = max(min_sub_burst_size, int(total_episodes // len(chosen)))
        out = []
        allocated = 0
        for i, c in enumerate(chosen):
            n = (int(total_episodes) - allocated) if i == len(chosen) - 1 else n_each
            allocated += n
            out.append((max(0, n), c["joint"], (c["lo"], c["hi"])))
        # Filter floor + clip overshoot, mirroring the 1D planner's
        # post-processing.
        return _finalize_per_joint_allocations(out, total_episodes, min_sub_burst_size)

    allocations: list[tuple[int, str, tuple[float, float]]] = []
    allocated = 0
    for idx, c in enumerate(chosen):
        if idx == len(chosen) - 1:
            n_eps = int(total_episodes) - allocated
        else:
            share = int(total_episodes * c["score"] / total_weight)
            n_eps = max(min_sub_burst_size, share)
        allocated += n_eps
        allocations.append((n_eps, c["joint"], (c["lo"], c["hi"])))

    return _finalize_per_joint_allocations(allocations, total_episodes, min_sub_burst_size)


def _finalize_per_joint_allocations(
    allocations: list[tuple[int, str, tuple[float, float]]],
    total_episodes: int,
    min_sub_burst_size: int,
) -> list[tuple[int, str, tuple[float, float]]]:
    """Apply overshoot trim + floor filter + redistribution.

    Mirrors plan_explore_sub_bursts's post-processing exactly so per-joint
    bursts behave the same way as 1D bursts under the same edge cases.
    """
    if not allocations:
        return []

    overshoot = sum(n for n, _, _ in allocations) - int(total_episodes)
    if overshoot > 0:
        biggest_idx = max(range(len(allocations)), key=lambda i: allocations[i][0])
        n, j, r = allocations[biggest_idx]
        allocations[biggest_idx] = (max(0, n - overshoot), j, r)

    surviving = [(n, j, r) for n, j, r in allocations if n >= min_sub_burst_size]
    dropped = sum(n for n, _, _ in allocations if n < min_sub_burst_size)
    if not surviving:
        # Floor filter killed everything — collapse to a single sub-burst
        # on the highest-score candidate covering its full range.
        n, j, r = allocations[0]
        return [(int(total_episodes), j, r)]
    if dropped > 0:
        n, j, r = surviving[0]
        surviving[0] = (n + dropped, j, r)
    return surviving


def _cache_path_for_repo_id(repo_id: str) -> Path:
    return Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id


def _session_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")



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
    force_joint: str | None = None,
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
      - `probe_script`: when set, each entry forces a specific
        (start_pos, direction) for one upcoming action. Used by VERIFY
        to drive error-weighted probes through the same camera-capture
        pipeline as EXPLORE — eliminates the legacy lerobot-record
        DSHOW multi-camera buffer-crosstalk bug that occasionally made
        wrist videos contain base-camera frames during VERIFY.
      - `window` is currently unused (the sequencer doesn't yet condition
        on the rolling error window — future enhancement).
    """
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
    # When `force_joint` is set (per-joint sub-burst targeting), pin
    # the pool to that single joint and disable vary_target.
    if force_joint is not None:
        # Recorder expects motor names with `.pos` suffix for the joints
        # pool. Accept both forms from the caller for convenience.
        joint_with_pos = (
            force_joint if force_joint.endswith(".pos") else f"{force_joint}.pos"
        )
        joints = [joint_with_pos]
    else:
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

    # When force_joint is set, the recorder's --joints is a single-element
    # list. Its CLI parser raises if any --joint-range refers to a joint
    # not in that list. Filter merged_ranges accordingly so we only emit
    # the active joint's range.
    for joint, (lo, hi) in merged_ranges.items():
        if force_joint is not None:
            joint_basename = joint.replace(".pos", "")
            if joint_basename != force_joint:
                continue
        cmd.extend(["--joint-range", str(joint), str(float(lo)), str(float(hi))])

    # If the config pins joints[0] for vary_target=False (single-joint
    # mode), or the caller pinned a force_joint, pass --no-vary-target
    # so the sequencer never wanders.
    if force_joint is not None or not getattr(cfg.explore, "vary_target_joint", True):
        cmd.append("--no-vary-target")

    # Match legacy: pass cfg.robot.home as starting positions so the
    # subprocess snaps the arm to a known pose before the first action.
    home_ns = getattr(cfg.robot, "home", None)
    if home_ns is not None:
        home_dict = {k: float(v) for k, v in vars(home_ns).items()}
        cmd.append(f"--starting-positions-json={json.dumps(home_dict)}")

    # Probe-script mode (used by VERIFY). Each entry forces (start_pos,
    # direction) for one upcoming action; the recorder snaps the active
    # joint to start_pos before that action's frames are captured. Tuple
    # input format `[(pos, direction)]` from the verifier is converted
    # to the recorder's dict schema.
    if probe_script is not None:
        script_dicts = []
        for entry in probe_script:
            if isinstance(entry, dict):
                script_dicts.append({
                    "start_pos": float(entry["start_pos"]),
                    "direction": str(entry["direction"]),
                    **({"joint": str(entry["joint"])} if entry.get("joint") else {}),
                })
            else:
                # Tuple format: (start_pos, direction[, joint?])
                pos = float(entry[0])
                direction = str(entry[1])
                d = {"start_pos": pos, "direction": direction}
                if len(entry) >= 3 and entry[2]:
                    d["joint"] = str(entry[2])
                script_dicts.append(d)
        cmd.append(f"--probe-script-json={json.dumps(script_dicts)}")

    if event_log is not None:
        event_log.log(
            event_tag,
            repo_id=repo_id,
            episodes=int(num_episodes),
            joint_range_override=joint_range_override,
            force_joint=force_joint,
            probe_script=probe_script,
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
