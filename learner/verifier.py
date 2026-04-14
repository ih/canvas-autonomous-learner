"""Predict -> execute -> compare -> MSE.

Reuses exactly the metric that `canvas-world-model/evaluate.py` already computes
offline (last-frame visual MSE), so online thresholds and offline val MSE speak
the same language. Camera frames are resized to the predictor's frame size
before comparison so dimensionality always matches.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Protocol

import cv2
import numpy as np

from .metrics import ProbeResult
from .action_canvas import save_action_canvas as _save_action_canvas_shared


class _HardwareLike(Protocol):
    def observe(self): ...
    def predict(self, ctx, motor, action): ...
    def execute(self, action: int) -> None: ...
    def goto(self, joint: str, target: float) -> None: ...


def quantize_motor(motor_state: np.ndarray, bin_size: float = 10.0) -> str:
    """Coarse bin used as a state-key for histogramming the error landscape."""
    bins = (np.asarray(motor_state, dtype=np.float32) / bin_size).round().astype(int)
    return ",".join(str(b) for b in bins.tolist())


def _to_frame_size(frame: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    if frame.shape[0] == h and frame.shape[1] == w:
        return frame
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


def _mse(pred: np.ndarray, actual: np.ndarray) -> float:
    p = pred.astype(np.float32) / 255.0
    a = actual.astype(np.float32) / 255.0
    return float(((p - a) ** 2).mean())


_save_action_canvas = _save_action_canvas_shared


def verify_once(
    hw: _HardwareLike,
    action: int,
    settle_time: float,
    examples_dir: Optional[Path] = None,
    example_tag: Optional[str] = None,
    target_joint: Optional[str] = None,
    target_position: Optional[float] = None,
) -> ProbeResult:
    """Execute one verify probe.

    If `target_joint` + `target_position` are both set, the probe starts by
    calling `hw.goto(target_joint, target_position)` — used by the curriculum
    to place the arm at an error-weighted starting state before observing.
    Otherwise the probe starts at whatever position the arm is currently
    at (legacy behavior).
    """
    if target_joint is not None and target_position is not None:
        hw.goto(target_joint, float(target_position))

    cams_before, motor_before, ctx = hw.observe()
    pred_base, pred_wrist = hw.predict(ctx, motor_before, action)

    hw.execute(action)
    if settle_time > 0:
        time.sleep(settle_time)

    cams_after, _, _ = hw.observe()

    target_hw = (pred_base.shape[0], pred_base.shape[1])
    actual_base = _to_frame_size(cams_after["base"], target_hw)
    actual_wrist = _to_frame_size(cams_after["wrist"], target_hw)

    mse_base = _mse(pred_base, actual_base)
    mse_wrist = _mse(pred_wrist, actual_wrist)
    mse = (mse_base + mse_wrist) / 2.0

    if examples_dir is not None:
        tag = example_tag or time.strftime("%H%M%S")
        _save_action_canvas(
            Path(examples_dir) / f"action_canvas_{tag}.png",
            cams_before, pred_base, pred_wrist,
            actual_base, actual_wrist, mse, action,
        )

    return ProbeResult(
        state_key=quantize_motor(motor_before),
        action=action,
        mse=mse,
        timestamp=time.time(),
        motor_state=tuple(float(x) for x in motor_before),
    )
