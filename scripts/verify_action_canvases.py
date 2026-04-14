"""End-to-end sanity check that reproduces training canvases exactly.

For each of the three discrete actions (positive, negative, none) we:

1. Spawn `run_single_action_record.py` with
   `--policy.force_direction=<dir>` and 3 episodes → writes a real
   LeRobot v3.0 dataset to `~/.cache/huggingface/lerobot/auto/sanity-*`.
2. Load each episode with `canvas-world-model/data/lerobot_loader.py`
   → returns the exact `frames` / `actions` / `motor_positions` lists
   that `create_dataset.py` uses.
3. Render a training-format canvas with
   `canvas-world-model/data/canvas_builder.py::build_canvas` using the
   same arguments `create_dataset.py` passes. Byte-identical to what
   the world model was trained on.
4. Arrange the 9 canvases in a 3 × 3 grid (rows = direction, columns
   = episode index) and save to `runs/sanity_grid.png`.

This replaces the earlier `verify_once`-based script whose before/
after selection did not match the training sampling rule.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from learner.config import load_config  # noqa: E402


_DIRECTIONS = [
    ("positive", "POS"),
    ("negative", "NEG"),
    ("none", "HOLD"),
]


def _cameras_arg(cfg) -> str:
    """Mirror learner.explorer._build_cameras_arg exactly so sanity-script
    recordings match the learner's EXPLORE recordings byte-for-byte —
    same rotation, same backend, same warmup, same frame size.
    """
    from learner.explorer import _build_cameras_arg  # noqa: WPS433
    return _build_cameras_arg(cfg)


def _record_episodes(cfg, direction: str, num_episodes: int, out_repo_id: str) -> Path:
    """Spawn run_single_action_record.py to capture `num_episodes` episodes
    with `force_direction=direction`. Returns the LeRobot cache dir.
    """
    rfmt_root = Path(cfg.paths.robotic_foundation_model_tests)
    script = rfmt_root / "scripts" / "run_single_action_record.py"
    python_exe = cfg.paths.python or sys.executable

    cmd = [
        python_exe,
        str(script),
        f"--robot.type=so101_follower",
        f"--robot.port={cfg.robot.port}",
        f"--robot.id={cfg.robot.robot_id}",
        f"--robot.cameras={_cameras_arg(cfg)}",
        "--policy.type=single_action",
        f"--policy.joint_name={cfg.explore.policy_joint_name}",
        f"--policy.vary_target_joint=false",
        f"--policy.position_delta={cfg.robot.step_size}",
        f"--policy.action_duration={cfg.explore.action_duration}",
        f"--policy.start_buffer={getattr(cfg.explore, 'start_buffer', 2.5)}",
        f"--policy.force_direction={direction}",
        f"--dataset.repo_id={out_repo_id}",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.fps={cfg.explore.dataset_fps}",
        "--dataset.push_to_hub=false",
    ]

    # Home override so reset-phase lands at a known pose every episode
    # (same mechanism as learner.explorer.collect_batch).
    home_ns = getattr(cfg.robot, "home", None)
    if home_ns is not None:
        import json as _json
        home_dict = {k: float(v) for k, v in vars(home_ns).items()}
        cmd.append(f"--starting-positions-json={_json.dumps(home_dict)}")

    print(f"[sanity] recording direction={direction} -> {out_repo_id}")
    proc = subprocess.Popen(
        cmd, cwd=str(rfmt_root),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    try:
        if proc.stdin is not None:
            proc.stdin.write("n\n")
            proc.stdin.flush()
            proc.stdin.close()
    except (BrokenPipeError, OSError):
        pass
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\r\n")
        if line:
            print(f"  {line}")
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / out_repo_id
    if not cache_dir.exists():
        raise FileNotFoundError(f"expected lerobot cache at {cache_dir}")
    return cache_dir


def _build_canvases_from_dataset(cfg, cache_dir: Path, num_episodes: int) -> list:
    """Load `num_episodes` episodes from the recorded dataset and return
    one training-format canvas per episode's single action.
    """
    rfmt_root = Path(cfg.paths.robotic_foundation_model_tests)
    cwm_root = Path(cfg.paths.canvas_world_model)
    crc_root = Path(cfg.paths.canvas_robot_control)
    for p in (rfmt_root, cwm_root, crc_root):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)

    from data.lerobot_loader import LeRobotV3Reader, load_episode  # type: ignore
    from data.canvas_builder import build_canvas  # type: ignore
    from control.canvas_utils import FRAME_SIZE  # type: ignore

    reader = LeRobotV3Reader(str(cache_dir))
    cameras = [cfg.explore.base_camera_name, cfg.explore.wrist_camera_name]

    episodes = []
    for idx in range(num_episodes):
        ep = load_episode(
            reader=reader,
            episode_index=idx,
            cameras=cameras,
            stack_mode="vertical",
            frame_size=FRAME_SIZE,
            state_column="observation.state",
        )
        episodes.append(ep)

    # Compute normalization bounds across these episodes (same way
    # create_dataset.py does it per-dataset). Using just these 9 eps
    # means motor strips normalize to their local range — fine for a
    # sanity grid.
    all_motors = []
    for ep in episodes:
        for m in ep.motor_positions or []:
            if m is not None:
                all_motors.append(m)
    if not all_motors:
        raise RuntimeError("no motor positions in recorded episodes")
    stacked = np.stack(all_motors)
    norm_min = stacked.min(axis=0)
    norm_max = stacked.max(axis=0)
    velocities = []
    for ep in episodes:
        mps = ep.motor_positions or []
        for i in range(1, len(mps)):
            if mps[i] is not None and mps[i - 1] is not None:
                velocities.append(mps[i] - mps[i - 1])
    vel_norm_max = (
        np.abs(np.stack(velocities)).max(axis=0)
        if velocities else np.ones(stacked.shape[1])
    )

    from control.canvas_utils import SEPARATOR_WIDTH, MOTOR_STRIP_HEIGHT  # type: ignore
    sep_width = SEPARATOR_WIDTH
    strip_h = MOTOR_STRIP_HEIGHT
    frame_size = (episodes[0].frames[0].shape[0], episodes[0].frames[0].shape[1])

    canvases = []
    for ep in episodes:
        if len(ep.actions) == 0 or len(ep.frames) < 2:
            raise RuntimeError(f"episode {ep.episode_index}: no actionable frames")
        canvas = build_canvas(
            [ep.frames[0], {"action": int(ep.actions[0])}, ep.frames[1]],
            frame_size=frame_size,
            sep_width=sep_width,
            motor_positions=[ep.motor_positions[0], ep.motor_positions[1]],
            motor_strip_height=strip_h,
            motor_norm_min=norm_min,
            motor_norm_max=norm_max,
            motor_vel_norm_max=vel_norm_max,
        )
        canvases.append(canvas)
    return canvases


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "comparison_armB.yaml"))
    p.add_argument("--per-direction", type=int, default=3)
    args = p.parse_args()

    cfg = load_config(args.config)

    rows = []
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for direction, label in _DIRECTIONS:
        repo_id = f"auto/sanity-{direction}-{stamp}"
        cache_dir = _record_episodes(cfg, direction, args.per_direction, repo_id)
        canvases = _build_canvases_from_dataset(cfg, cache_dir, args.per_direction)
        row = np.concatenate(canvases, axis=1)
        # Label column on the left
        label_w = 40
        label_col = np.full((row.shape[0], label_w, 3), 32, dtype=np.uint8)
        cv2.putText(
            label_col, label, (4, row.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
        rows.append(np.concatenate([label_col, row], axis=1))

    # Pad rows to equal width so np.concatenate works.
    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.full((r.shape[0], max_w - r.shape[1], 3), 0, dtype=np.uint8)
            r = np.concatenate([r, pad], axis=1)
        padded.append(r)
    grid = np.concatenate(padded, axis=0)

    out_path = Path(cfg.paths.runs_dir) / f"sanity_grid_{stamp}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_path)
    print(f"[sanity] wrote {out_path} shape={grid.shape}")


if __name__ == "__main__":
    main()
