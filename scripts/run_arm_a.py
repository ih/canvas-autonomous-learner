"""Phase 0.3: measure the offline baseline (Arm A) on the locked val set.

Loads `diff_iter4_wider/best.pth` (the baseline the learner has to beat),
shells out to `evaluate.py --dataset locked_val_v1`, parses
`val_mse_visual` from the report, and writes `runs/arm_a_result.json`.
No training. Single number, ~2 minutes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from learner.config import load_config  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "comparison_armB.yaml"))
    p.add_argument(
        "--checkpoint",
        default=str(
            REPO_ROOT.parent
            / "canvas-world-model"
            / "local" / "checkpoints" / "diff_iter4_wider" / "best.pth"
        ),
    )
    p.add_argument("--output", default=str(REPO_ROOT / "runs" / "arm_a_result.json"))
    args = p.parse_args()

    cfg = load_config(args.config)
    locked_val = cfg.paths.locked_val_dataset
    if locked_val is None:
        print("[arm_a] FAIL: cfg.paths.locked_val_dataset is null")
        sys.exit(1)
    if not Path(locked_val).exists():
        print(f"[arm_a] FAIL: locked val dataset not found: {locked_val}")
        print("       Run scripts/record_locked_val.py first.")
        sys.exit(1)
    if not Path(args.checkpoint).exists():
        print(f"[arm_a] FAIL: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    eval_out = Path(cfg.paths.runs_dir) / f"arm_a_eval_{time.strftime('%Y%m%d_%H%M%S')}"
    eval_out.mkdir(parents=True, exist_ok=True)
    python_exe = cfg.paths.python or sys.executable

    cmd = [
        python_exe,
        "evaluate.py",
        "--model-type", "diffusion",
        "--checkpoint", str(args.checkpoint),
        "--dataset", str(locked_val),
        "--output-dir", str(eval_out),
        "--no-html",
    ]
    print(f"[arm_a] running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=str(cfg.paths.canvas_world_model), check=True)
    except subprocess.CalledProcessError as e:
        print(f"[arm_a] FAIL: evaluate.py returned {e.returncode}")
        sys.exit(1)

    report = eval_out / "report.json"
    if not report.exists():
        print(f"[arm_a] FAIL: report.json missing at {report}")
        sys.exit(1)

    with open(report) as f:
        data = json.load(f)
    metrics = data.get("metrics", data)
    val_mse_visual = metrics.get("val_mse_visual")
    val_mse = metrics.get("val_mse")

    result = {
        "checkpoint": str(args.checkpoint),
        "locked_val_dataset": str(locked_val),
        "val_mse_visual": val_mse_visual,
        "val_mse": val_mse,
        "eval_output_dir": str(eval_out),
        "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[arm_a] val_mse_visual: {val_mse_visual}")
    print(f"[arm_a] result written: {out_path}")
    print("[arm_a] PASS")


if __name__ == "__main__":
    main()
