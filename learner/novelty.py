"""Compute novelty metrics for a freshly-collected EXPLORE batch.

After an EXPLORE phase the orchestrator transitions to THINK with a
novelty report attached. The advisor uses it to decide whether the
new data materially changes the training distribution (retrain is
worthwhile) or just repeats known scenes (explore more, or skip
retrain and try a different scene). Cheap to compute — a few seconds.

The report covers:
  - mean-frame MSE vs the most recent prior batch — a scalar "did the
    scene change?" signal. Averaged over a handful of sampled canvases.
  - pixel-range stats for the mean frames (brightness, stddev) so the
    advisor can spot lighting or backdrop drift.
  - representative canvas paths the advisor can Read visually: one new,
    one from the nearest-mean prior batch.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image


def _canvas_paths(canvas_dir: Path, limit: int | None = None) -> list[Path]:
    pngs = sorted(canvas_dir.glob("canvas_*.png"))
    if limit is not None and len(pngs) > limit:
        step = max(1, len(pngs) // limit)
        pngs = pngs[::step][:limit]
    return pngs


def _mean_frame(canvas_dir: Path, sample_n: int = 8) -> Optional[np.ndarray]:
    """Average up to `sample_n` canvas PNGs into a single mean image.
    Returns None if the dir has no canvases.
    """
    paths = _canvas_paths(canvas_dir)
    if not paths:
        return None
    rng = random.Random(hash(canvas_dir.name) & 0xFFFFFFFF)
    sample = rng.sample(paths, min(sample_n, len(paths)))
    acc: Optional[np.ndarray] = None
    n = 0
    for p in sample:
        try:
            arr = np.asarray(Image.open(p), dtype=np.float32)
        except Exception:
            continue
        if arr.ndim != 3:
            continue
        if acc is None:
            acc = np.zeros_like(arr)
        if arr.shape == acc.shape:
            acc += arr
            n += 1
    if n == 0 or acc is None:
        return None
    return (acc / n).astype(np.float32)


def _frame_stats(arr: np.ndarray) -> dict:
    return {
        "mean": float(arr.mean()) / 255.0,
        "std": float(arr.std()) / 255.0,
    }


def compute_novelty_report(
    new_canvas_dirs: Sequence[Path | str],
    prior_canvas_dirs: Sequence[Path | str],
    include_sample_paths: bool = True,
) -> dict:
    """Compare the newly-built canvas datasets against prior accumulated
    datasets; return a summary the advisor can reason about.

    Keys in the returned dict:
      `num_new_dirs`           - count of just-built canvas dirs
      `num_prior_dirs`         - count of accumulated prior dirs
      `mean_frame_mse_vs_prior_latest`
                               - scalar MSE in [0, 1] between the new
                                 batch's mean frame and the most recent
                                 prior batch's mean frame. None if not
                                 computable (no canvases or shape
                                 mismatch). Small = same scene, large =
                                 scene drift or genuinely new poses.
      `new_frame_stats`, `prior_frame_stats`
                               - mean + std of each mean frame, for
                                 detecting lighting / backdrop shifts.
      `sample_canvas_paths`    - list of (tag, path) pairs the advisor
                                 can Read to eyeball differences. Always
                                 at least one from each (if canvases
                                 exist).
    """
    new_dirs = [Path(d) for d in new_canvas_dirs]
    prior_dirs = [Path(d) for d in prior_canvas_dirs]

    report: dict = {
        "num_new_dirs": len(new_dirs),
        "num_prior_dirs": len(prior_dirs),
        "mean_frame_mse_vs_prior_latest": None,
        "new_frame_stats": None,
        "prior_frame_stats": None,
        "sample_canvas_paths": [],
    }

    new_mean = _mean_frame(new_dirs[0]) if new_dirs else None
    prior_mean = _mean_frame(prior_dirs[-1]) if prior_dirs else None

    if new_mean is not None:
        report["new_frame_stats"] = _frame_stats(new_mean)
    if prior_mean is not None:
        report["prior_frame_stats"] = _frame_stats(prior_mean)

    if new_mean is not None and prior_mean is not None and new_mean.shape == prior_mean.shape:
        diff = (new_mean - prior_mean).astype(np.float32) / 255.0
        report["mean_frame_mse_vs_prior_latest"] = float((diff * diff).mean())

    if include_sample_paths:
        pairs: list[tuple[str, str]] = []
        if new_dirs:
            new_pngs = _canvas_paths(new_dirs[0])
            if new_pngs:
                pairs.append(("new_representative", str(new_pngs[len(new_pngs) // 2])))
        if prior_dirs:
            prior_pngs = _canvas_paths(prior_dirs[-1])
            if prior_pngs:
                pairs.append(("prior_representative", str(prior_pngs[len(prior_pngs) // 2])))
        report["sample_canvas_paths"] = pairs

    return report
