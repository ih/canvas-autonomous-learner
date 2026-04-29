"""Record the two-joint experiment's locked-val corpus.

Records N shoulder_pan-action episodes and N elbow_flex-action episodes,
then canvas-builds each into a separate dataset under
`paths.locked_val_shoulder` and `paths.locked_val_elbow` (as configured
in `configs/simultaneous.yaml`) using the new acting-joint canvas
encoding and the motor bounds from `cfg.training.motor_bounds`.

Run this ONCE before starting either two-joint arm. Both arms evaluate
their fine-tuned checkpoints against this shared held-out corpus.

Usage:
    python scripts/record_locked_val.py --config configs/simultaneous.yaml

Interruption-safe: if a recording already exists in the HF cache at
`locked_val/shoulder-<stamp>` / `locked_val/elbow-<stamp>`, delete that
subdir to force a re-record; otherwise re-running this script will
produce a NEW stamp and re-record (no auto-reuse).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from copy import copy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from learner import explorer  # noqa: E402
from learner.config import load_config  # noqa: E402


def _override_explore_cfg(cfg, joint_name: str):
    """Shallow-copy cfg and force single-joint recording on `joint_name`.

    The single_action policy's __post_init__ rejects identical primary
    and secondary joints; its default secondary is elbow_flex.pos, so
    when we record elbow we must re-point the secondary elsewhere.
    """
    new_cfg = copy(cfg)
    new_cfg.explore = copy(cfg.explore)
    new_cfg.explore.policy_joint_name = joint_name
    new_cfg.explore.vary_target_joint = False
    new_cfg.explore.joints = None
    # Pick a secondary that's guaranteed different from the primary.
    if joint_name == "elbow_flex.pos":
        new_cfg.explore.secondary_joint_name = "shoulder_pan.pos"
    else:
        new_cfg.explore.secondary_joint_name = "elbow_flex.pos"
    return new_cfg


def _record_one_joint(cfg, *, joint_name, joint_min, joint_max,
                      num_episodes, repo_id_prefix, tag):
    sub_cfg = _override_explore_cfg(cfg, joint_name)
    joint_range_override = {joint_name: (float(joint_min), float(joint_max))}
    print(f"\n[{tag}] Recording {num_episodes} episodes on {joint_name} "
          f"(starts uniform in [{joint_min}, {joint_max}])")
    dataset_path = explorer.collect_batch_continuous(
        sub_cfg,
        num_episodes=num_episodes,
        joint_range_override=joint_range_override,
        repo_id_prefix=repo_id_prefix,
        event_tag="locked_val_record_start",
        randomize_primary_start=True,
    )
    if dataset_path is None:
        raise RuntimeError(f"[{tag}] recorder subprocess failed — see output above.")
    print(f"[{tag}] recorded to: {dataset_path}")
    return Path(dataset_path)


def _canvas_build(cfg, *, lerobot_path, output_dir, motor_bounds, tag):
    cwm = Path(cfg.paths.canvas_world_model)
    python_exe = cfg.paths.python or sys.executable
    if output_dir.exists():
        print(f"[{tag}] wiping stale output dir: {output_dir}")
        shutil.rmtree(output_dir)
    cmd = [
        python_exe,
        "create_dataset.py",
        "--lerobot-path", str(lerobot_path),
        "--output", str(output_dir),
        "--cameras", cfg.explore.base_camera_name, cfg.explore.wrist_camera_name,
        "--stack-cameras", "vertical",
        "--frame-size", "224", "224",
        "--motor-bounds-json", json.dumps(motor_bounds),
    ]
    print(f"[{tag}] canvas-building to: {output_dir}")
    subprocess.run(cmd, cwd=cwm, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="Config YAML (e.g. configs/simultaneous.yaml).")
    p.add_argument("--num-shoulder", type=int, default=30)
    p.add_argument("--num-elbow", type=int, default=30)
    p.add_argument("--shoulder-min", type=float, default=-60.0)
    p.add_argument("--shoulder-max", type=float, default=60.0)
    p.add_argument("--elbow-min", type=float, default=50.0)
    p.add_argument("--elbow-max", type=float, default=90.0)
    p.add_argument("--skip-shoulder", action="store_true",
                   help="Skip the shoulder recording+canvas (use if it already succeeded).")
    p.add_argument("--skip-elbow", action="store_true",
                   help="Skip the elbow recording+canvas (use if it already succeeded).")
    args = p.parse_args()

    cfg = load_config(args.config)

    mb_ns = getattr(getattr(cfg, "training", None), "motor_bounds", None)
    if mb_ns is None:
        print("ERROR: cfg.training.motor_bounds is required.", file=sys.stderr)
        sys.exit(2)
    motor_bounds = {k: list(v) for k, v in vars(mb_ns).items() if v is not None}

    shoulder_out = Path(cfg.paths.locked_val_shoulder)
    elbow_out = Path(cfg.paths.locked_val_elbow)

    # shoulder ---------------------------------------------------------
    if not args.skip_shoulder:
        shoulder_lerobot = _record_one_joint(
            cfg,
            joint_name="shoulder_pan.pos",
            joint_min=args.shoulder_min, joint_max=args.shoulder_max,
            num_episodes=args.num_shoulder,
            repo_id_prefix="locked_val/shoulder", tag="SHOULDER",
        )
        _canvas_build(cfg, lerobot_path=shoulder_lerobot, output_dir=shoulder_out,
                      motor_bounds=motor_bounds, tag="SHOULDER")
    else:
        print("[SHOULDER] skipped (--skip-shoulder)")

    # elbow ------------------------------------------------------------
    if not args.skip_elbow:
        elbow_lerobot = _record_one_joint(
            cfg,
            joint_name="elbow_flex.pos",
            joint_min=args.elbow_min, joint_max=args.elbow_max,
            num_episodes=args.num_elbow,
            repo_id_prefix="locked_val/elbow", tag="ELBOW",
        )
        _canvas_build(cfg, lerobot_path=elbow_lerobot, output_dir=elbow_out,
                      motor_bounds=motor_bounds, tag="ELBOW")
    else:
        print("[ELBOW] skipped (--skip-elbow)")

    print("\nDone.")
    print(f"  shoulder canvas dataset: {shoulder_out}")
    print(f"  elbow canvas dataset:    {elbow_out}")
    print("\nReady to start the simultaneous arm:")
    print(f"  {cfg.paths.python} -m learner --config {args.config}")


if __name__ == "__main__":
    main()
