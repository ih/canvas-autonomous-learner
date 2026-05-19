"""Retrospective cold-start vs warm fine-tune comparison.

For each scene boundary in an experiment's history, train a fresh
cold-start model on the cumulative N-scene corpus, evaluate it
against the same locked_val + wide_verify corpus the warm run used,
and emit a paired result so the operator can see whether cold-start
beats warm fine-tune at any/every scene count.

Reads:
  - <runs_dir>/registry.json — current accumulated_canvas_dirs (in
    deterministic append order) + wide_verify_canvas_dirs.
  - <runs_dir>/events_*.jsonl — used to derive `accumulated_dirs[:K]`
    snapshots for each scene_count K, by walking
    `scene_ready_acknowledged` (IDLE-path) events in order and
    counting how many canvas dirs had been appended at each ack point.
    The registry doesn't preserve historical snapshots, but the events
    log is the source of truth for ordering.

For each scene boundary N:
  - Take `accumulated_canvas_dirs[:offset_for_scene_N]`
  - Run `trainer_driver.retrain_cumulative(... resume_checkpoint=None,
    epochs=cold_start_epochs)` — pure cold start
  - Returns `{checkpoint, train_val_mse, locked_val_mse,
    wide_verify_mse, ...}`
  - Append to results

Output: `<runs_dir>/cold_warm_compare.json` with a list of entries:
  {scene_count, total_eps, cold_locked_val_mse, cold_train_val_mse,
   cold_wide_verify_mse, cold_checkpoint}

Pair manually with the live experiment's per-cycle warm results from
the registry's `locked_val_history` (warm eval at each scene_count).
The dashboard chart `wide_verify_vs_scenes` already shows the warm
curve from `checkpoint_swapped` events.

Usage:
    python scripts/cold_warm_compare.py \\
        --runs-dir runs/red_kong_min_data_250m \\
        --config configs/red_kong_min_data_250m.yaml

Optional flags:
    --start-scene N   skip scenes < N (resume a partial sweep)
    --stop-scene N    halt after scene N (early peek)
    --epochs N        override cold_start_epochs (default reads cfg)
    --output PATH     output JSON path (default: <runs_dir>/cold_warm_compare.json)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make `learner.*` importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from learner import trainer_driver
from learner.config import load_config
from learner.events import EventLog
from learner.registry import Registry


def _scene_boundaries(
    runs_dir: Path,
    accumulated_dirs: list[str],
) -> list[dict]:
    """Reconstruct per-scene snapshots of `accumulated_canvas_dirs`.

    Walks the events log looking for `scene_ready_acknowledged` (IDLE-
    path only — wide_verify acks are excluded) and tracks how many
    canvas dirs were registered between successive acks. Each entry
    in the returned list is one scene boundary:

        {
          "scene_idx": int,
          "scene_count": int,        # = scene_idx + 1
          "dirs_through_this_scene": list[str],
          "total_eps": int,           # cumulative through this scene
        }

    If the events log is missing or partial, falls back to "all dirs
    correspond to one scene each" (which matches the deterministic-
    cadence experiment where each cycle adds exactly one canvas dir).
    """
    # Find the latest events file
    candidates = sorted(runs_dir.glob("events_*.jsonl"))
    if not candidates:
        # Fallback: assume one canvas dir per scene
        return [
            {
                "scene_idx": i,
                "scene_count": i + 1,
                "dirs_through_this_scene": list(accumulated_dirs[: i + 1]),
                "total_eps": (i + 1) * 100,  # best-effort guess
            }
            for i in range(len(accumulated_dirs))
        ]

    events_path = candidates[-1]
    events = []
    with open(events_path) as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except Exception:
                continue

    # Walk events: each accumulated-dir append corresponds to a
    # `subprocess_done` event with tag `create_dataset` followed by
    # the registry update. Easier signal: trust the registry's order
    # and find scene_ready_acknowledged events to bucket dirs into
    # scenes. We assume canvas dirs are registered in the cycle that
    # produced them, between scene rotations.
    scene_dir_counts: dict[int, int] = {}
    current_scene = 0
    # Count canvas-build subprocess_done events per scene by walking
    # the timeline.
    for e in events:
        ev = e.get("event")
        if ev == "scene_ready_acknowledged" and e.get("source") != "wide_verify":
            si = e.get("scene_idx")
            if si is not None:
                current_scene = int(si)
        elif ev == "subprocess_done" and e.get("tag") in (
            "create_dataset", "combine_datasets",
        ):
            # only count create_dataset (canvas build) events; combine
            # is internal and doesn't add a new dir
            if e.get("tag") == "create_dataset":
                scene_dir_counts[current_scene] = (
                    scene_dir_counts.get(current_scene, 0) + 1
                )

    # If we couldn't reconstruct from events, fall back to 1:1 mapping.
    if not scene_dir_counts:
        return [
            {
                "scene_idx": i,
                "scene_count": i + 1,
                "dirs_through_this_scene": list(accumulated_dirs[: i + 1]),
                "total_eps": (i + 1) * 100,
            }
            for i in range(len(accumulated_dirs))
        ]

    # Build cumulative offsets per scene
    boundaries: list[dict] = []
    offset = 0
    for scene_idx in sorted(scene_dir_counts):
        count = scene_dir_counts[scene_idx]
        offset += count
        boundaries.append({
            "scene_idx": scene_idx,
            "scene_count": scene_idx + 1,
            "dirs_through_this_scene": list(accumulated_dirs[:offset]),
            # Approximate total_eps = scene_count * 100 (deterministic
            # cadence assumption). Could be refined by parsing events.
            "total_eps": (scene_idx + 1) * 100,
        })
    return boundaries


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--start-scene", type=int, default=0,
                   help="skip scenes < N (resume a partial sweep)")
    p.add_argument("--stop-scene", type=int, default=None,
                   help="halt after scene N (early peek)")
    p.add_argument("--epochs", type=int, default=None,
                   help="override cold_start_epochs")
    p.add_argument("--output", type=Path, default=None,
                   help="output JSON (default <runs_dir>/cold_warm_compare.json)")
    args = p.parse_args()

    runs_dir: Path = args.runs_dir.resolve()
    cfg = load_config(str(args.config))
    registry = Registry(runs_dir / "registry.json")
    accumulated_dirs = registry.accumulated_canvas_dirs()
    wv_dirs = registry.wide_verify_canvas_dirs()
    locked_val = getattr(cfg.paths, "locked_val_dataset", None)
    cold_epochs = args.epochs if args.epochs is not None else int(
        getattr(cfg.cadence, "cold_start_epochs", 30)
    )
    output_path = args.output or (runs_dir / "cold_warm_compare.json")

    boundaries = _scene_boundaries(runs_dir, accumulated_dirs)
    print(f"reconstructed {len(boundaries)} scene boundaries")
    print(f"  total accumulated dirs in registry: {len(accumulated_dirs)}")
    print(f"  wide_verify corpus dirs: {len(wv_dirs)}")
    print(f"  locked_val: {locked_val}")
    print(f"  cold_start_epochs: {cold_epochs}")

    # Resume support: load existing results if present, skip already-done
    # scenes. Lets the script be killed and restarted without losing work.
    existing: list[dict] = []
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text())
            done_scenes = {int(e["scene_count"]) for e in existing}
            print(f"  resuming with {len(existing)} prior results, "
                  f"skipping scene_counts={sorted(done_scenes)}")
        except Exception:
            existing = []
            done_scenes = set()
    else:
        done_scenes = set()

    event_log = EventLog(runs_dir, session=f"cold_warm_compare_{int(time.time())}")
    results = list(existing)

    for b in boundaries:
        sc = b["scene_count"]
        if sc < args.start_scene:
            continue
        if args.stop_scene is not None and sc > args.stop_scene:
            break
        if sc in done_scenes:
            continue

        dirs_subset = b["dirs_through_this_scene"]
        if not dirs_subset:
            print(f"scene {sc}: no canvas dirs — skip")
            continue
        print(f"\n=== scene_count={sc} (scene_idx={b['scene_idx']}) "
              f"using {len(dirs_subset)} canvas dirs, "
              f"epochs={cold_epochs} ===")

        result = trainer_driver.retrain_cumulative(
            cfg,
            accumulated_canvas_dirs=dirs_subset,
            resume_checkpoint=None,  # COLD START
            epochs=cold_epochs,
            locked_val_dataset=locked_val,
            wide_verify_canvas_dirs=wv_dirs,
            event_log=event_log,
        )

        if result is None or not isinstance(result, dict):
            print(f"  retrain failed (result={result}) — recording None and continuing")
            entry = {
                "scene_count": sc,
                "scene_idx": b["scene_idx"],
                "total_eps": b["total_eps"],
                "n_canvas_dirs": len(dirs_subset),
                "cold_locked_val_mse": None,
                "cold_train_val_mse": None,
                "cold_wide_verify_mse": None,
                "cold_checkpoint": None,
                "epochs": cold_epochs,
                "failed": True,
                "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        else:
            entry = {
                "scene_count": sc,
                "scene_idx": b["scene_idx"],
                "total_eps": b["total_eps"],
                "n_canvas_dirs": len(dirs_subset),
                "cold_locked_val_mse": result.get("locked_val_mse"),
                "cold_train_val_mse": result.get("train_val_mse"),
                "cold_wide_verify_mse": result.get("wide_verify_mse"),
                "cold_checkpoint": str(result.get("checkpoint")),
                "epochs": cold_epochs,
                "failed": False,
                "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            print(f"  cold: locked_val={entry['cold_locked_val_mse']} "
                  f"train_val={entry['cold_train_val_mse']} "
                  f"wide_verify={entry['cold_wide_verify_mse']}")

        results.append(entry)
        # Persist incrementally so a kill mid-sweep loses at most one
        # cold-start's worth of compute.
        output_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"  wrote {len(results)} entries to {output_path}")

    print(f"\ndone — {len(results)} entries in {output_path}")
    print("pair with the warm curve from registry.locked_val_history "
          "and dashboard `wide_verify_vs_scenes`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
