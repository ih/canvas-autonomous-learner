"""Shared helper: load one recorded LeRobot episode, run the world
model on it, compute MSE, and write a training-format action canvas
(`[before | action_sep | actual]` stacked over `[before | action_sep |
inferred]`) to `examples_dir`. Returns a `ProbeResult` on success,
`None` on load failure (transient errors are re-raised so the caller
can retry).

Both `scripts/explore_inference.py` (worker process) and
`learner/verifier.verify_batch` (in-process) call this so every action
canvas on the dashboard goes through the same code path the training
dataset uses — `data.lerobot_loader.load_episode` and
`data.canvas_builder.build_canvas`, byte-identical to
`canvas-world-model/create_dataset.py`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .metrics import ProbeResult


def _quantize_motor(motor_state: np.ndarray, bin_size: float = 10.0) -> str:
    """Coarse state-key binning for the error histogram. Duplicated
    locally to avoid a circular import between episode_canvas and
    verifier (which imports episode_canvas)."""
    bins = (np.asarray(motor_state, dtype=np.float32) / bin_size).round().astype(int)
    return ",".join(str(b) for b in bins.tolist())


_TRANSIENT_MARKERS = (
    "Parquet magic bytes",
    "magic bytes not found",
    "Episodes metadata not found",
)


def _is_transient_load_error(msg: str) -> bool:
    if not msg:
        return False
    if any(marker in msg for marker in _TRANSIENT_MARKERS):
        return True
    return "episode metadata" in msg.lower()


def _ensure_sibling_paths(cfg) -> None:
    for attr in ("canvas_robot_control", "canvas_world_model",
                 "robotic_foundation_model_tests"):
        p = getattr(cfg.paths, attr, None)
        if p and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _canvas_control_joint_idx(cfg) -> int:
    _ensure_sibling_paths(cfg)
    from control.robot_interface import JOINTS  # type: ignore
    return JOINTS.index(cfg.robot.control_joint)


def process_recorded_episode(
    cfg,
    cache_dir: Path,
    episode_index: int,
    predictor,
    examples_dir: Path,
    cycle: int,
    filename_prefix: str,
) -> Optional[ProbeResult]:
    """Load + predict + MSE + canvas for one recorded episode.

    Args:
        cfg: learner config (for camera names, paths, step size).
        cache_dir: LeRobot dataset root for the recorded episodes.
        episode_index: which episode inside `cache_dir` to process.
        predictor: a `WorldModelPredictor`-compatible object with
            `.meta` (dict) and `.predict_batch(...)`.
        examples_dir: destination dir for the written action canvas PNG.
        cycle: current learner cycle (used in the filename).
        filename_prefix: `ep` or `p` — determines whether this reads
            back as an EXPLORE replay or a VERIFY probe in the dashboard
            gallery parser.

    Returns:
        ProbeResult on success, None if the episode had no actionable
        frames or no motor state. Transient load errors (recorder
        mid-flush races) are re-raised so the caller can retry.
    """
    _ensure_sibling_paths(cfg)
    from data.lerobot_loader import LeRobotV3Reader, load_episode  # type: ignore
    from data.canvas_builder import build_canvas  # type: ignore
    from control.canvas_utils import FRAME_SIZE  # type: ignore

    reader = LeRobotV3Reader(str(cache_dir))
    # canvas-world-model's loader accepts either the short camera name
    # or the full "observation.images.<name>" form. Try short first.
    short_cameras = [cfg.explore.base_camera_name, cfg.explore.wrist_camera_name]
    long_cameras = [f"observation.images.{c}" for c in short_cameras]

    episode = None
    last_err = None
    for cam_list in (short_cameras, long_cameras):
        try:
            episode = load_episode(
                reader=reader,
                episode_index=episode_index,
                cameras=cam_list,
                stack_mode="vertical",
                frame_size=FRAME_SIZE,
                state_column="observation.state",
            )
            break
        except Exception as e:
            last_err = e
            continue
    if episode is None:
        msg = str(last_err) if last_err else ""
        if _is_transient_load_error(msg):
            raise last_err
        return None

    if len(episode.actions) == 0 or len(episode.frames) < 2:
        return None
    if not episode.motor_positions or episode.motor_positions[0] is None:
        return None

    meta = getattr(predictor, "meta", None)
    if meta is None:
        return None
    frame_size = tuple(meta.get("frame_size", (448, 224)))
    sep_width = int(meta.get("separator_width", 32))
    strip_h = int(meta.get("motor_strip_height", 16))
    norm_min = np.asarray(meta["motor_norm_min"], dtype=np.float32)
    norm_max = np.asarray(meta["motor_norm_max"], dtype=np.float32)
    vel_norm_max = (
        np.asarray(meta.get("motor_vel_norm_max"), dtype=np.float32)
        if meta.get("motor_vel_norm_max") is not None else None
    )

    control_idx = _canvas_control_joint_idx(cfg)
    step_size = float(cfg.robot.step_size)

    # Single-action episodes have exactly one decision after trimming;
    # use the first one. If the episode has more, we still only render
    # the first so verify canvases are directly comparable to explore.
    i = 0
    action = int(episode.actions[i])
    context_frame = episode.frames[i]
    actual_frame = episode.frames[i + 1]
    motor_state = episode.motor_positions[i]
    motor_next = (
        episode.motor_positions[i + 1]
        if i + 1 < len(episode.motor_positions) else motor_state
    )

    pred_list = predictor.predict_batch(
        context_frame,
        motor_state,
        [action],
        step_size=step_size,
        control_joint_idx=control_idx,
        prediction_depth=1,
    )
    pred_base, pred_wrist = pred_list[0]
    pred_stacked = np.concatenate([pred_base, pred_wrist], axis=0)

    # MSE against actual in the same frame-halves layout.
    pb = pred_base.astype(np.float32) / 255.0
    pw = pred_wrist.astype(np.float32) / 255.0
    ab = actual_frame[: actual_frame.shape[0] // 2].astype(np.float32) / 255.0
    aw = actual_frame[actual_frame.shape[0] // 2:].astype(np.float32) / 255.0
    mse = float(((pb - ab) ** 2).mean() + ((pw - aw) ** 2).mean()) / 2.0

    # One single `build_canvas` call with 3 frames:
    #   [before, actual, predicted]
    # and 2 separators between them:
    #   - first separator: the real discrete action (green/blue/red)
    #   - second separator: a non-standard code (→ gray) distinguishing
    #     "actual observed" from "model inference"
    # Motor positions parallel the 3 frames: motor_before, motor_after
    # for the actual next, motor_after again for the inferred next
    # (the diffusion model doesn't predict motor state separately).
    out_img = build_canvas(
        [
            context_frame,
            {"action": action},
            actual_frame,
            {"action": -1},            # non-standard → gray separator
            pred_stacked,
        ],
        frame_size=frame_size,
        sep_width=sep_width,
        motor_positions=[motor_state, motor_next, motor_next],
        motor_strip_height=strip_h,
        motor_norm_min=norm_min,
        motor_norm_max=norm_max,
        motor_vel_norm_max=vel_norm_max,
    )

    # Label strip above the canvas so the gallery caption shows which
    # frame is which and the MSE of the prediction.
    target_h, target_w = frame_size
    label_h = 16
    label = np.full((label_h, out_img.shape[1], 3), 24, dtype=np.uint8)
    cv2.putText(
        label, "before", (6, 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA,
    )
    actual_x = target_w + sep_width + 2
    cv2.putText(
        label, "ACTUAL", (actual_x, 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 255, 150), 1, cv2.LINE_AA,
    )
    inferred_x = target_w * 2 + sep_width * 2 + 2
    cv2.putText(
        label, f"INFERRED mse={mse:.5f}", (inferred_x, 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 200, 255), 1, cv2.LINE_AA,
    )
    out_img = np.concatenate([label, out_img], axis=0)

    # filename_prefix is "ep" for EXPLORE replays, "p" for VERIFY probes.
    # Dashboard parses these to display "[EXPLORE]" / "[VERIFY]" captions.
    if filename_prefix == "ep":
        tag = f"c{cycle:03d}_ep{episode_index:04d}_d{i:02d}_{time.strftime('%H%M%S')}"
    else:
        tag = f"c{cycle:03d}_p{episode_index}_{time.strftime('%H%M%S')}"
    out_path = examples_dir / f"action_canvas_{tag}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out_img).save(out_path)

    motor_state_arr = np.asarray(motor_state, dtype=np.float32)
    motor_next_arr = np.asarray(motor_next, dtype=np.float32)
    # Infer the acting joint from the motor delta between the action's
    # before and after states. Single_action always moves exactly one
    # joint by ±step_size (≥10 units) while non-acting joints fluctuate
    # by <1 unit of servo noise — so the argmax over the absolute delta
    # reliably picks the acting joint for non-hold actions. For holds
    # (all-zero delta) leave it None.
    acting_joint_idx: Optional[int] = None
    if motor_state_arr.shape == motor_next_arr.shape and motor_state_arr.size > 0:
        delta = np.abs(motor_next_arr - motor_state_arr)
        if float(delta.max()) > 1.0:
            acting_joint_idx = int(np.argmax(delta))

    return ProbeResult(
        state_key=_quantize_motor(motor_state_arr),
        action=action,
        mse=mse,
        timestamp=time.time(),
        motor_state=tuple(float(x) for x in motor_state_arr),
        acting_joint_idx=acting_joint_idx,
    )
