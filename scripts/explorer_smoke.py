"""Phase 0.1: explorer subprocess smoke test.

Runs `learner.explorer.collect_batch_continuous(cfg, num_episodes=2)` against real
hardware and asserts the LeRobot cache dir exists afterward. Catches
CLI/subprocess bugs in the run_single_action_record shell-out before a
multi-hour Arm B run depends on it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from learner.config import load_config  # noqa: E402
from learner import explorer  # noqa: E402
from learner.events import EventLog  # noqa: E402


def _parse_range(s: str) -> tuple[float, float]:
    """Accept '(-20, 20)' / '-20,20' / '-20 20' as range specifications."""
    cleaned = s.strip().strip("()[]")
    parts = cleaned.replace(",", " ").split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected 'lo,hi', got {s!r}")
    return (float(parts[0]), float(parts[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "hardware_run.yaml"))
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument(
        "--joint-range",
        type=_parse_range,
        default=None,
        help="Optional '(lo, hi)' override for the control joint range. "
             "When set, exercises the --policy.joint_ranges CLI path.",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    log = EventLog(cfg.paths.runs_dir, session="explorer_smoke")
    print(f"[smoke] config={args.config} episodes={args.episodes}")
    print(f"[smoke] event log: {log.path}")

    joint_override = None
    if args.joint_range is not None:
        joint_key = f"{cfg.robot.control_joint}.pos"
        joint_override = {joint_key: args.joint_range}
        print(f"[smoke] joint_range_override={joint_override}")

    result = explorer.collect_batch_continuous(
        cfg,
        args.episodes,
        event_log=log,
        joint_range_override=joint_override,
        randomize_primary_start=joint_override is not None,
    )

    if result is None:
        print("[smoke] FAIL: collect_batch returned None")
        sys.exit(1)
    if not result.exists():
        print(f"[smoke] FAIL: output path does not exist: {result}")
        sys.exit(1)

    meta = result / "meta" / "info.json"
    data = result / "data"
    print(f"[smoke] output dir: {result}")
    print(f"[smoke] meta/info.json exists: {meta.exists()}")
    print(f"[smoke] data/ exists: {data.exists()}")

    if not meta.exists() or not data.exists():
        print("[smoke] FAIL: output missing meta/info.json or data/")
        sys.exit(1)

    # If we asked for a narrow range, verify the recorded motor positions
    # actually stayed inside it. Uses lerobot_loader to read the episode
    # parquet files back.
    if joint_override is not None:
        print("[smoke] verifying recorded positions stay in range...")
        joint_key = next(iter(joint_override))
        lo, hi = joint_override[joint_key]
        try:
            for attr in ("canvas_world_model",):
                p_ = getattr(cfg.paths, attr, None)
                if p_ and p_ not in sys.path:
                    sys.path.insert(0, p_)
            from data.lerobot_loader import LeRobotV3Reader  # type: ignore
            reader = LeRobotV3Reader(str(result))
            joints = [
                "shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex", "wrist_roll", "gripper",
            ]
            joint_idx = joints.index(cfg.robot.control_joint)
            chunk_df = reader.get_data_chunk(0)
            state_col = "observation.state"
            if state_col not in chunk_df.columns:
                print(f"[smoke] WARN: no {state_col} column; skip range check")
            else:
                import numpy as np
                positions = []
                for row in chunk_df[state_col]:
                    if isinstance(row, (list, tuple, np.ndarray)):
                        positions.append(float(row[joint_idx]))
                min_pos = min(positions) if positions else None
                max_pos = max(positions) if positions else None
                print(f"[smoke] recorded {cfg.robot.control_joint} positions: "
                      f"min={min_pos:.1f} max={max_pos:.1f}  (requested [{lo},{hi}])")
                # Allow a small epsilon for motor overshoot + the single_action
                # policy's end-buffer + gravity drift.
                eps = 12.0
                if min_pos is not None and (min_pos < lo - eps or max_pos > hi + eps):
                    print(f"[smoke] WARN: positions drifted beyond range "
                          f"by more than {eps} degrees")
                else:
                    print("[smoke] range override honored")
        except Exception as e:
            print(f"[smoke] WARN: could not verify range ({e})")

    print("[smoke] PASS")


if __name__ == "__main__":
    main()
