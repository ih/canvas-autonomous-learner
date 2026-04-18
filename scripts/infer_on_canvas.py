"""On-demand inference on training canvases for the dashboard.

Given a batch directory (under `<repo>/datasets/canvas/`) and one or
more `canvas_NNNNN.png` filenames, re-runs the current live-checkpoint
predictor on each canvas's source LeRobot episode and writes a fresh
action canvas (`[before | action_sep | ACTUAL | gray_sep | INFERRED]`)
per input.

Invoked by the dashboard's `/api/infer_canvas` endpoint as a subprocess
so the dashboard process stays torch-free.

Output paths are printed as JSON on stdout, one dict per input::

    {"input": "canvas_00000.png", "output": "runs/on_demand_inference/batch_xxx/action_canvas_c000_ui0_20260416_123045.png"}

A failed item has `{"input": ..., "error": "..."}`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from learner.config import load_config  # noqa: E402
from learner.episode_canvas import process_recorded_episode  # noqa: E402
from learner.hardware import Hardware  # noqa: E402
from learner.registry import Registry  # noqa: E402


_CANVAS_RE = re.compile(r"^canvas_(\d+)\.png$", re.IGNORECASE)


def _canvas_index(name: str) -> int | None:
    m = _CANVAS_RE.match(name)
    if m is None:
        return None
    return int(m.group(1))


def _resolve_source(meta: dict, canvas_idx: int) -> tuple[Path | None, int | None, str]:
    """Resolve (lerobot_cache_dir, episode_index_in_that_source) for a
    given canvas_idx.

    Handles three cases:
      - single-source batch: meta has top-level `source_path`.
        `episode_index` comes from meta["episodes"] (canvas_start/end).
      - merged batch: meta has `source_datasets` with per-source
        `{source_path, canvas_offset, canvas_count}`. We find the
        segment containing canvas_idx, then use `canvas_idx -
        canvas_offset` as the local episode index. This assumes
        single-action-per-episode data (the current explore format)
        — i.e. canvas_count == episode_count within each source.
      - neither present: return an error message.

    Returns `(cache_dir or None, episode_index or None, error_msg)`.
    On success `error_msg` is empty.
    """
    # Case 1: single-source batch.
    sp = meta.get("source_path")
    if sp:
        path = Path(sp)
        if not path.exists():
            return None, None, f"source_path does not exist: {sp}"
        episodes = meta.get("episodes") or []
        for ep in episodes:
            start = int(ep.get("canvas_start", -1))
            end = int(ep.get("canvas_end", -2))
            if start <= canvas_idx <= end:
                return path, int(ep["episode_index"]), ""
        return path, None, (
            f"canvas index {canvas_idx} not in any episode range of source"
        )

    # Case 2: merged batch.
    source_datasets = meta.get("source_datasets") or []
    if source_datasets:
        for entry in source_datasets:
            offset = int(entry.get("canvas_offset", 0))
            count = int(entry.get("canvas_count", 0))
            if offset <= canvas_idx < offset + count:
                entry_sp = entry.get("source_path")
                if not entry_sp:
                    return None, None, "matching source_datasets entry has no source_path"
                path = Path(entry_sp)
                if not path.exists():
                    return None, None, f"source_path does not exist: {entry_sp}"
                # Assume 1:1 canvas:episode within the source (single-action
                # explore). `canvas_idx - offset` is the LeRobot episode
                # index inside that source.
                local_idx = canvas_idx - offset
                return path, local_idx, ""
        return None, None, (
            f"canvas index {canvas_idx} not in any source_datasets segment"
        )

    return None, None, "dataset_meta.json has neither source_path nor source_datasets"


def _run_one(
    cfg,
    predictor,
    batch_dir: Path,
    canvas_name: str,
    out_dir: Path,
    meta: dict,
) -> dict:
    out = {"input": canvas_name}
    idx = _canvas_index(canvas_name)
    if idx is None:
        out["error"] = f"not a canvas_NNNNN.png filename: {canvas_name}"
        return out

    cache_dir, episode_index, err = _resolve_source(meta, idx)
    if err or cache_dir is None or episode_index is None:
        out["error"] = err or "could not resolve source lerobot dataset"
        return out

    try:
        probe = process_recorded_episode(
            cfg,
            cache_dir=cache_dir,
            episode_index=episode_index,
            predictor=predictor,
            examples_dir=out_dir,
            cycle=0,
            filename_prefix=f"ui{idx:05d}",
        )
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    if probe is None:
        out["error"] = "process_recorded_episode returned None (no actionable frames)"
        return out

    # process_recorded_episode writes exactly one PNG into out_dir keyed
    # on the call — find the newest one.
    pngs = sorted(out_dir.glob("action_canvas_*.png"), key=lambda p: p.stat().st_mtime)
    if not pngs:
        out["error"] = "no output PNG produced"
        return out
    out["output"] = str(pngs[-1].resolve())
    out["mse"] = float(probe.mse)
    out["action"] = int(probe.action)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--batch-dir", required=True,
                   help="Absolute path to the canvas batch directory")
    p.add_argument("--canvas-name", action="append", required=True,
                   help="canvas_NNNNN.png filename (repeatable)")
    p.add_argument("--out-dir", required=True,
                   help="Directory where output action canvases get written")
    args = p.parse_args()

    cfg = load_config(args.config)
    batch_dir = Path(args.batch_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = batch_dir / "dataset_meta.json"
    if not meta_path.exists():
        print(json.dumps({"fatal": f"no dataset_meta.json in {batch_dir}"}))
        return 2
    with meta_path.open() as f:
        meta = json.load(f)

    # Load live checkpoint from the canvas-autonomous-learner registry.
    registry = Registry(cfg.paths.registry_file)
    ckpt = registry.live_checkpoint()
    if not ckpt or not Path(ckpt).exists():
        print(json.dumps({"fatal": f"no live checkpoint in registry ({ckpt!r})"}))
        return 3

    hw = Hardware(cfg, dry_run=True)  # dry_run=True skips robot bus connect
    hw.load_predictor(ckpt)

    results = []
    for name in args.canvas_name:
        # Give each inference its own sub-output dir so `sorted by mtime`
        # picks up the right file even on the same second.
        per_out = out_dir / f"_{name.replace('.png', '')}"
        per_out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        res = _run_one(cfg, hw.predictor, batch_dir, name, per_out, meta)
        res["elapsed_s"] = time.time() - t0
        results.append(res)
        print(json.dumps(res), flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
