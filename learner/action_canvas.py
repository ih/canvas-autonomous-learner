"""Shared helper: compose a `before | predicted | actual` action canvas grid.

An "action canvas" is the visual representation of one action the arm
took: the `before` frame(s), the model's `predicted` post-action frame,
and the `actual` observed post-action frame — stacked into a 2×3 grid
with a header caption. Used by both `learner.verifier.verify_once` (live
camera probes during VERIFY) and `scripts/explore_inference.py` (replay
probes from recorded LeRobot episodes during EXPLORE) so the dashboard's
"recent action canvases" gallery renders identical-looking grids for
both sources and nothing downstream has to know which produced which.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def to_frame_size(frame: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    if frame.shape[0] == h and frame.shape[1] == w:
        return frame
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


def save_action_canvas(
    path: str | Path,
    cams_before: dict,
    pred_base: np.ndarray,
    pred_wrist: np.ndarray,
    actual_base: np.ndarray,
    actual_wrist: np.ndarray,
    mse: float,
    action: int,
    header_prefix: str = "",
) -> None:
    """Compose a 2-row × 3-column action canvas grid and write it as PNG.

    Rows: base camera (top), wrist camera (bottom).
    Columns: before | predicted | actual. A 24-px header strip is
    prepended showing `{header_prefix} action=N mse=X.XXXXX`.
    """
    target_hw = (pred_base.shape[0], pred_base.shape[1])
    before_base = to_frame_size(cams_before["base"], target_hw)
    before_wrist = to_frame_size(cams_before["wrist"], target_hw)
    actual_base = to_frame_size(actual_base, target_hw)
    actual_wrist = to_frame_size(actual_wrist, target_hw)

    row_base = np.concatenate([before_base, pred_base, actual_base], axis=1)
    row_wrist = np.concatenate([before_wrist, pred_wrist, actual_wrist], axis=1)
    grid = np.concatenate([row_base, row_wrist], axis=0)

    header_h = 24
    header = np.full((header_h, grid.shape[1], 3), 32, dtype=np.uint8)
    prefix = f"{header_prefix} " if header_prefix else ""
    label = f"{prefix}action={action}  mse={mse:.5f}  cols: before | pred | actual"
    cv2.putText(
        header, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (255, 255, 255), 1, cv2.LINE_AA,
    )
    out = np.concatenate([header, grid], axis=0)
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr)
