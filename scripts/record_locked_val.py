"""Phase 0.2: record the locked validation canvas dataset.

Collects N episodes of SO-101 recordings using the same explorer machinery as
Arm B, builds canvases, and copies the canvas dataset into
`../canvas-world-model/local/datasets/<output_name>` so both arms can point
at the same held-out set via `evaluate.py --dataset <output_name>`.

Usage:
    python scripts/record_locked_val.py                      # 50 episodes
    python scripts/record_locked_val.py --episodes 30
    python scripts/record_locked_val.py --name locked_val_v2
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from learner.config import load_config  # noqa: E402
from learner import explorer, trainer_driver  # noqa: E402
from learner.events import EventLog  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "hardware_run.yaml"))
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--name", default="locked_val_v1")
    args = p.parse_args()

    cfg = load_config(args.config)
    log = EventLog(cfg.paths.runs_dir, session=f"record_locked_val_{args.name}")
    print(f"[locked_val] config={args.config}")
    print(f"[locked_val] episodes={args.episodes} name={args.name}")
    print(f"[locked_val] event log: {log.path}")

    # 1. Collect the LeRobot episodes.
    lerobot_dir = explorer.collect_batch(cfg, args.episodes, event_log=log)
    if lerobot_dir is None or not lerobot_dir.exists():
        print(f"[locked_val] FAIL: explorer returned no output")
        sys.exit(1)
    print(f"[locked_val] recorded LeRobot dataset: {lerobot_dir}")

    # 2. Build canvases into a staging directory.
    staging = Path(cfg.paths.canvas_out) / f"{args.name}_staging"
    if staging.exists():
        shutil.rmtree(staging)
    trainer_driver.build_canvases(cfg, lerobot_dir, staging, event_log=log)
    print(f"[locked_val] canvases built at: {staging}")

    # 3. Freeze: copy into the canvas-world-model datasets dir so evaluate.py
    #    can pick it up with a stable path.
    cwm_datasets = Path(cfg.paths.canvas_world_model) / "local" / "datasets"
    cwm_datasets.mkdir(parents=True, exist_ok=True)
    final_dir = cwm_datasets / args.name
    if final_dir.exists():
        print(f"[locked_val] WARN: {final_dir} already exists; removing")
        shutil.rmtree(final_dir)
    shutil.copytree(staging, final_dir)
    print(f"[locked_val] frozen locked val set: {final_dir}")

    meta = final_dir / "dataset_meta.json"
    if not meta.exists():
        print(f"[locked_val] FAIL: {meta} missing")
        sys.exit(1)

    import json
    with open(meta) as f:
        m = json.load(f)
    canvas_count = m.get("canvas_count")
    print(f"[locked_val] canvas_count: {canvas_count}")
    print(f"[locked_val] PASS")


if __name__ == "__main__":
    main()
