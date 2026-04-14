"""Entry point: `python -m learner [--config ...]`."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .orchestrator import main_loop


def _parse_args():
    p = argparse.ArgumentParser(description="canvas-autonomous-learner main loop")
    p.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "configs" / "default.yaml"),
        help="Path to YAML config file",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Use DryRunRobotInterface — no hardware access.",
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Stop after N outer-loop iterations (smoke tests / bounded runs).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    if args.dry_run:
        cfg.dry_run = True
    result = main_loop(cfg, max_iterations=args.max_iterations)
    print(f"\n[learner] done: {result}")


if __name__ == "__main__":
    main()
