"""One-off cold-start training of the 1B model on a list of canvas batch
dirs, then swap the live checkpoint so a subsequent
`python -m learner --config <cfg>` picks it up.

Usage:
    # Cold-start on whatever the registry has accumulated:
    python scripts/cold_start_full_data.py --config configs/sim_1b.yaml

    # Cold-start on every batch_* dir under one or more search roots
    # (lets us train on all sessions at once, not just sim_1b):
    python scripts/cold_start_full_data.py --config configs/sim_1b.yaml \\
        --canvas-search-root runs/sim --canvas-search-root runs/sim_1b \\
        --canvas-search-root runs/sim_smoke

Why a separate script: the orchestrator only does a cold start when
`cycle == 0`. After cycles have run, the only way to force a fresh model
on the full accumulated corpus is to invoke `retrain_cumulative` directly
with `resume_checkpoint=None`, then atomically swap the registry pointer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `learner.*` importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from learner import trainer_driver
from learner.config import load_config
from learner.events import EventLog
from learner.registry import Registry


def _discover_canvas_dirs(roots: list[Path]) -> list[str]:
    """Find every `batch_*` canvas dir under each root that contains at
    least one `canvas_*.png` AND a `dataset_meta.json`. Empty dirs (left
    over from aborted explorer runs) are skipped.
    """
    out: list[str] = []
    for root in roots:
        if not root.exists():
            print(f"[cold_start] WARN search root missing: {root}")
            continue
        for d in sorted(root.rglob("batch_*")):
            if not d.is_dir():
                continue
            meta = d / "dataset_meta.json"
            if not meta.exists():
                continue
            if not any(d.glob("canvas_*.png")):
                continue
            out.append(str(d))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--canvas-search-root",
        action="append",
        default=[],
        help=(
            "Root dir to recursively scan for `batch_*` canvas dirs. "
            "May be passed multiple times. When omitted, falls back to "
            "the registry's accumulated_canvas_dirs."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    registry = Registry(cfg.paths.registry_file)

    if args.canvas_search_root:
        repo_root = Path(__file__).resolve().parent.parent
        roots = []
        for r in args.canvas_search_root:
            p = Path(r)
            if not p.is_absolute():
                p = (repo_root / p).resolve()
            roots.append(p)
        accumulated = _discover_canvas_dirs(roots)
        print(f"[cold_start] discovered {len(accumulated)} canvas dirs across {len(roots)} roots")
        if not accumulated:
            print(f"[cold_start] no canvas dirs found under {roots}")
            sys.exit(2)
    else:
        accumulated = registry.accumulated_canvas_dirs()
        if not accumulated:
            print(f"[cold_start] no accumulated canvas dirs in {cfg.paths.registry_file}")
            sys.exit(2)

    epochs = int(getattr(cfg.cadence, "cold_start_epochs", 30))
    locked_val = getattr(cfg.paths, "locked_val_dataset", None)

    session = registry.session_name() or "cold_start_full_data"
    event_log = EventLog(cfg.paths.runs_dir, session=session)
    event_log.log(
        "cold_start_full_data_begin",
        canvas_dirs=accumulated,
        num_canvas_dirs=len(accumulated),
        epochs=epochs,
        locked_val_dataset=locked_val,
    )

    print(f"[cold_start] {len(accumulated)} canvas dirs, epochs={epochs}")
    for d in accumulated:
        print(f"  - {d}")

    result = trainer_driver.retrain_cumulative(
        cfg,
        accumulated_canvas_dirs=accumulated,
        resume_checkpoint=None,
        epochs=epochs,
        locked_val_dataset=locked_val,
        event_log=event_log,
    )

    if result is None or not isinstance(result, dict) or "checkpoint" not in result:
        event_log.log("cold_start_full_data_failed", result=result)
        print(f"[cold_start] FAILED: {result}")
        sys.exit(1)

    new_ckpt = result["checkpoint"]
    merged = result["merged_dataset"]
    train_val = result.get("train_val_mse")
    locked_val_mse = result.get("locked_val_mse")

    registry.swap(
        new_ckpt,
        merged_canvas_dataset=merged,
        val_mse=train_val,
        notes="cold_start_full_data",
    )

    event_log.log(
        "cold_start_full_data_done",
        checkpoint=new_ckpt,
        merged_dataset=merged,
        train_val_mse=train_val,
        locked_val_mse=locked_val_mse,
    )
    print(f"[cold_start] DONE checkpoint={new_ckpt}")
    print(f"[cold_start]      train_val_mse={train_val}  locked_val_mse={locked_val_mse}")


if __name__ == "__main__":
    main()
