"""Live-inference testing dashboard.

A browser UI for manual QA of the live world-model checkpoint against the
physical SO-101 arm. Serves:

  - A joint dropdown + action (+/-/hold) selector.
  - "Predict": runs the world model on the current observation for the
    chosen (joint, action) and renders a training-format canvas
    `[before | action_sep | predicted]`.
  - "Execute": sends the action to the real robot, then renders an
    `[before | action_sep | actual]` canvas with the same `before` frame
    shared with the predicted canvas.
  - "Relax joints": disables torque so the operator can move the arm
    by hand.
  - "Lock joints": re-enables torque holding the current pose.

Cameras are only read on demand (Predict / Execute click). There is no
continuous MJPEG feed — that was causing camera-enumeration conflicts
when restarting the dashboard.

This dashboard OWNS the robot + cameras for its lifetime. It MUST NOT be
run while `python -m learner` is driving the same arm — COM3 is single-
access on Windows.

Usage:
    python scripts/live_inference_dashboard.py --config configs/default.yaml
    python scripts/live_inference_dashboard.py --config configs/default.yaml --dry-run --port 8766
"""

from __future__ import annotations

import argparse
import http.server
import json
import secrets
import socketserver
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from learner.config import load_config  # noqa: E402
from learner.hardware import Hardware  # noqa: E402
from learner.registry import Registry  # noqa: E402


# ------------------------------------------------------------------- state ---


@dataclass
class DashboardState:
    """In-memory state shared across requests, guarded by `bus_lock`."""

    before_motor: Optional[np.ndarray] = None
    before_ctx: Optional[np.ndarray] = None
    before_ts: Optional[float] = None
    torque_on: bool = True
    canvas_tokens: dict = field(default_factory=dict)  # token -> Path
    active_model: str = "learner"  # "learner" or "baseline"
    active_checkpoint: Optional[str] = None  # path of currently loaded model

    def clear_before(self) -> None:
        self.before_motor = None
        self.before_ctx = None
        self.before_ts = None


# ------------------------------------------------------------- canvas ops ---


def _read_motor_positions(hw: Hardware) -> np.ndarray:
    """Fast motor-only read (skips the camera grabs that hw.observe does).

    Falls back to hw.observe() for DryRunRobotInterface, which exposes its
    positions via the same `get_state()` path as the real robot but has no
    `bus`.
    """
    bus = getattr(hw.robot, "bus", None)
    if bus is not None:
        positions = bus.sync_read("Present_Position")
        from control.robot_interface import JOINTS  # type: ignore
        return np.array([positions[j] for j in JOINTS], dtype=np.float32)
    # Dry run fallback.
    _cams, motor, _ctx = hw.observe()
    return motor


def _wait_until_motion_settled(
    hw: Hardware,
    poll_interval: float = 0.08,
    stable_threshold: float = 0.4,
    stable_window: float = 0.2,
    min_wait: float = 0.25,
    timeout: float = 3.5,
) -> None:
    """Block until the max-joint delta stays below `stable_threshold` degrees
    for `stable_window` seconds, bounded by [min_wait, timeout].

    Caller must hold the bus lock.
    """
    t_start = time.time()
    prev = _read_motor_positions(hw)
    stable_since: Optional[float] = None
    while True:
        elapsed = time.time() - t_start
        if elapsed >= timeout:
            return
        time.sleep(poll_interval)
        now = _read_motor_positions(hw)
        delta = float(np.max(np.abs(now - prev)))
        prev = now
        if elapsed < min_wait:
            continue
        if delta < stable_threshold:
            if stable_since is None:
                stable_since = time.time()
            elif time.time() - stable_since >= stable_window:
                return
        else:
            stable_since = None


def _predict_motor_after(motor_before: np.ndarray, joint_idx: int,
                         action: int, step_size: float,
                         joint_min: float, joint_max: float) -> np.ndarray:
    """Deterministic next-state motor estimate matching
    `RobotInterface.execute_action_on` clamp logic."""
    out = np.asarray(motor_before, dtype=np.float32).copy()
    if action == 1:
        out[joint_idx] = out[joint_idx] + step_size
    elif action == 2:
        out[joint_idx] = out[joint_idx] - step_size
    out[joint_idx] = max(joint_min, min(joint_max, float(out[joint_idx])))
    return out


def _build_two_frame_canvas(
    before_ctx: np.ndarray,
    after_ctx: np.ndarray,
    action: int,
    motor_before: np.ndarray,
    motor_after: np.ndarray,
    meta: dict,
    label: str,
    mse: Optional[float] = None,
) -> np.ndarray:
    """Render `[before | action_sep | after]` in training format with a
    small label strip above so the browser view is self-describing."""
    from data.canvas_builder import build_canvas  # type: ignore

    frame_size = tuple(meta.get("frame_size", (448, 224)))
    sep_width = int(meta.get("separator_width", 32))
    strip_h = int(meta.get("motor_strip_height", 16))
    norm_min = np.asarray(meta["motor_norm_min"], dtype=np.float32)
    norm_max = np.asarray(meta["motor_norm_max"], dtype=np.float32)
    vel_norm_max = (
        np.asarray(meta.get("motor_vel_norm_max"), dtype=np.float32)
        if meta.get("motor_vel_norm_max") is not None else None
    )

    canvas = build_canvas(
        [before_ctx, {"action": action}, after_ctx],
        frame_size=frame_size,
        sep_width=sep_width,
        motor_positions=[motor_before, motor_after],
        motor_strip_height=strip_h,
        motor_norm_min=norm_min,
        motor_norm_max=norm_max,
        motor_vel_norm_max=vel_norm_max,
    )

    target_h, target_w = frame_size
    label_h = 18
    label_img = np.full((label_h, canvas.shape[1], 3), 24, dtype=np.uint8)
    cv2.putText(label_img, "before", (6, 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    right_x = target_w + sep_width + 4
    right_label = label if mse is None else f"{label}  mse={mse:.5f}"
    color = (150, 200, 255) if label.lower().startswith("pred") else (150, 255, 150)
    cv2.putText(label_img, right_label, (right_x, 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return np.concatenate([label_img, canvas], axis=0)


# --------------------------------------------------------------- HTML UI ---


INDEX_HTML = (Path(__file__).parent / "templates" / "live_inference.html").read_text(encoding="utf-8")


# --------------------------------------------------------------- handler ---


def make_handler(
    hw: Hardware,
    cfg,
    registry: Registry,
    bus_lock: threading.Lock,
    state: DashboardState,
    canvas_out_dir: Path,
    joints: list[str],
    mode: str,
    baseline_checkpoint: Optional[str],
):
    from control.robot_interface import JOINTS  # type: ignore

    def _learner_ckpt() -> str:
        """Always read fresh from the registry — the autonomous learner
        may swap this mid-session."""
        ckpt = registry.live_checkpoint() or ""
        return ckpt

    initial_learner_ckpt = _learner_ckpt()
    has_baseline = bool(baseline_checkpoint) and Path(baseline_checkpoint).exists()

    index_html = (
        INDEX_HTML
        .replace("__JOINTS__", json.dumps(joints))
        .replace("__CONTROL_JOINT__", cfg.robot.control_joint)
        .replace("__MODE__", mode)
        .replace("__LEARNER_CKPT__", initial_learner_ckpt)
        .replace("__BASELINE_CKPT__", baseline_checkpoint or "")
        .replace("__HAS_BASELINE__", "true" if has_baseline else "false")
        .replace("__INITIAL_MODEL__", state.active_model)
        .encode("utf-8")
    )
    canvas_out_resolved = canvas_out_dir.resolve()

    def _mint_token(path: Path) -> str:
        tok = secrets.token_urlsafe(12)
        state.canvas_tokens[tok] = path
        return tok

    def _build_and_store(
        before_ctx: np.ndarray,
        after_ctx: np.ndarray,
        action: int,
        motor_before: np.ndarray,
        motor_after: np.ndarray,
        label: str,
        mse: Optional[float],
        prefix: str,
    ) -> str:
        canvas = _build_two_frame_canvas(
            before_ctx, after_ctx, action,
            motor_before, motor_after,
            meta=hw.predictor.meta, label=label, mse=mse,
        )
        canvas_out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S") + f"_{secrets.token_hex(3)}"
        path = canvas_out_dir / f"{prefix}_{ts}.png"
        Image.fromarray(canvas).save(path)
        return _mint_token(path)

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _send_json(self, status: int, obj) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _send_bytes(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _read_json(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return {}
            raw = self.rfile.read(n)
            try:
                return json.loads(raw.decode("utf-8") or "{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                return {}

        # ------------------------------------------------------------- GET

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send_bytes(200, index_html, "text/html; charset=utf-8")
                return
            if path.startswith("/canvas/"):
                token = unquote(path[len("/canvas/"):])
                target = state.canvas_tokens.get(token)
                if target is None:
                    self.send_error(404)
                    return
                try:
                    resolved = target.resolve()
                    resolved.relative_to(canvas_out_resolved)
                except (ValueError, OSError):
                    self.send_error(403)
                    return
                if not resolved.exists():
                    self.send_error(404)
                    return
                self._send_bytes(200, resolved.read_bytes(), "image/png")
                return
            self.send_error(404)

        # ------------------------------------------------------------- POST

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            try:
                if path == "/api/predict":
                    self._send_json(200, self._api_predict(self._read_json()))
                elif path == "/api/execute":
                    self._send_json(200, self._api_execute(self._read_json()))
                elif path == "/api/clear_before":
                    with bus_lock:
                        state.clear_before()
                    self._send_json(200, {"ok": True})
                elif path == "/api/relax":
                    with bus_lock:
                        hw.relax()
                        state.torque_on = False
                    self._send_json(200, {"torque": "off"})
                elif path == "/api/lock":
                    with bus_lock:
                        hw.lock()
                        state.torque_on = True
                        _cams, motor, _ctx = hw.observe()
                    self._send_json(200, {"torque": "on",
                                          "motor_state": motor.tolist()})
                elif path == "/api/set_model":
                    self._send_json(200, self._api_set_model(self._read_json()))
                else:
                    self.send_error(404)
            except Exception as e:
                self._send_json(500, {"error": f"{type(e).__name__}: {e}"})

        # --------------------------------------------------- api implementations

        def _resolve_joint(self, body: dict) -> tuple[str, int]:
            joint = body.get("joint") or cfg.robot.control_joint
            if joint not in JOINTS:
                raise ValueError(f"unknown joint {joint!r}")
            return joint, JOINTS.index(joint)

        def _resolve_action(self, body: dict) -> int:
            action = int(body.get("action", 3))
            if action not in (1, 2, 3):
                raise ValueError(f"action must be 1|2|3, got {action!r}")
            return action

        def _api_set_model(self, body: dict) -> dict:
            choice = str(body.get("model") or "").strip()
            if choice not in ("learner", "baseline"):
                raise ValueError(f"model must be 'learner' or 'baseline', got {choice!r}")
            if choice == "baseline":
                if not has_baseline:
                    raise RuntimeError(
                        "no baseline checkpoint configured — pass "
                        "--baseline-checkpoint PATH on the dashboard CLI."
                    )
                target = baseline_checkpoint
            else:
                target = _learner_ckpt()
                if not target or not Path(target).exists():
                    raise RuntimeError(
                        f"learner checkpoint from registry is missing: {target!r}"
                    )
            # Reload under the bus lock so no Predict/Execute is racing.
            with bus_lock:
                print(f"[set_model] loading {choice}: {target}", flush=True)
                t0 = time.time()
                hw.load_predictor(target)
                elapsed = time.time() - t0
                state.active_model = choice
                state.active_checkpoint = target
                # Any "before" captured for the previous model's prediction
                # is still physically valid but no longer compares apples
                # to apples — drop it so the next Predict starts clean.
                state.clear_before()
                print(f"[set_model] loaded in {elapsed:.1f}s", flush=True)
            return {
                "active_model": state.active_model,
                "active_checkpoint": state.active_checkpoint,
                "load_seconds": elapsed,
            }

        def _capture_before(self) -> None:
            """Populate state.before_* from a fresh observation if not set.

            Arrays are copied so downstream mutation (e.g. the predictor's
            input normalization) can't corrupt the frame Execute reuses.
            """
            if state.before_ctx is not None:
                return
            _cams, motor, ctx = hw.observe()
            state.before_motor = motor.copy()
            state.before_ctx = ctx.copy()
            state.before_ts = time.time()

        def _api_predict(self, body: dict) -> dict:
            action = self._resolve_action(body)
            joint, joint_idx = self._resolve_joint(body)
            with bus_lock:
                self._capture_before()
                motor_before = state.before_motor.copy()
                ctx_before = state.before_ctx.copy()
                pred_list = hw.predictor.predict_batch(
                    ctx_before, motor_before, [action],
                    step_size=cfg.robot.step_size,
                    control_joint_idx=joint_idx,
                    prediction_depth=1,
                )
                pred_base, pred_wrist = pred_list[0]
            pred_stacked = np.concatenate([pred_base, pred_wrist], axis=0)
            motor_after_pred = _predict_motor_after(
                motor_before, joint_idx, action,
                cfg.robot.step_size, cfg.robot.joint_min, cfg.robot.joint_max,
            )
            token = _build_and_store(
                ctx_before, pred_stacked, action,
                motor_before, motor_after_pred,
                label="PREDICTED", mse=None, prefix="predicted",
            )
            return {
                "canvas_token": token,
                "motor_state": motor_before.tolist(),
                "before_captured_at": state.before_ts,
                "joint": joint,
                "action": action,
            }

        def _api_execute(self, body: dict) -> dict:
            action = self._resolve_action(body)
            joint, joint_idx = self._resolve_joint(body)
            with bus_lock:
                # Torque must be on or sync_write(Goal_Position) is a no-op
                # — the most common reason "after" looks identical to
                # "before" is that the user relaxed the arm and forgot to
                # lock it before pressing Execute. Auto-lock at current
                # position so Execute always actually moves.
                if not state.torque_on:
                    hw.lock()
                    state.torque_on = True
                    print("[execute] torque was off; auto-locked at "
                          "current position before executing.",
                          flush=True)

                self._capture_before()
                motor_before = state.before_motor.copy()
                ctx_before = state.before_ctx.copy()

                if joint == cfg.robot.control_joint:
                    hw.execute(action)
                else:
                    hw.execute_on(action, joint)

                # Wait for the motor to actually finish moving. A fixed
                # 0.5s sleep is too short for a 10-degree SO-101 move; the
                # captured "after" frame ends up identical to "before".
                # Poll motor positions and break as soon as two consecutive
                # reads show sub-threshold change. Motor-only reads bypass
                # the multi-grab camera path for fast polling.
                _wait_until_motion_settled(
                    hw,
                    poll_interval=0.08,
                    stable_threshold=0.4,
                    stable_window=0.2,
                    min_wait=0.35,
                    timeout=4.0,
                )
                # Extra flush: DSHOW can keep stale mid-motion frames in
                # the buffer even after the 3-grab flush in get_state().
                # Throw away one full observe so the next one is guaranteed
                # post-motion.
                hw.observe()
                cams_after, motor_after, ctx_after = hw.observe()

                # Diagnostic: if this prints "motor_before == motor_after",
                # the motor didn't move (stuck, torque off on that joint,
                # bus error, etc.) and no post-processing will fix it.
                delta = float(np.max(np.abs(motor_after - motor_before)))
                print(
                    f"[execute] joint={joint} action={action} "
                    f"motor_before={motor_before.tolist()} "
                    f"motor_after={motor_after.tolist()} "
                    f"max_delta={delta:.2f}",
                    flush=True,
                )

                # execution changed physical state — invalidate before
                state.clear_before()
            token = _build_and_store(
                ctx_before, ctx_after, action,
                motor_before, motor_after,
                label="ACTUAL", mse=None, prefix="actual",
            )
            return {
                "canvas_token": token,
                "motor_before": motor_before.tolist(),
                "motor_after": motor_after.tolist(),
                "joint": joint,
                "action": action,
            }

    return Handler


# ----------------------------------------------------------------- server ---


class _ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(cfg_path: Path, port: int, host: str, dry_run: bool,
          baseline_checkpoint: Optional[str] = None) -> None:
    cfg = load_config(cfg_path)
    registry = Registry(cfg.paths.registry_file)
    ckpt = registry.live_checkpoint()
    if not ckpt or not Path(ckpt).exists():
        raise SystemExit(f"no live checkpoint in registry ({ckpt!r}) — "
                         "nothing to run live inference against.")

    if baseline_checkpoint:
        if not Path(baseline_checkpoint).exists():
            raise SystemExit(
                f"--baseline-checkpoint does not exist: {baseline_checkpoint}"
            )

    print(f"live-inference dashboard starting")
    print(f"  config:     {cfg_path}")
    print(f"  dry_run:    {dry_run}")
    print(f"  learner:    {ckpt}")
    if baseline_checkpoint:
        print(f"  baseline:   {baseline_checkpoint}")
    print(f"  control:    {cfg.robot.control_joint} (step={cfg.robot.step_size})")

    hw = Hardware(cfg, dry_run=dry_run)
    print("  connecting robot + cameras...")
    hw.connect()
    print("  loading predictor (learner checkpoint)...")
    hw.load_predictor(ckpt)

    # Runs are timestamped so repeated runs don't stomp each other's canvases.
    from datetime import datetime
    runs_dir = Path(cfg.paths.runs_dir) if hasattr(cfg.paths, "runs_dir") else REPO_ROOT / "runs"
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    canvas_out_dir = runs_dir / "live_inference" / session

    from control.robot_interface import JOINTS  # type: ignore
    joints = list(JOINTS)

    bus_lock = threading.Lock()
    state = DashboardState()
    state.active_model = "learner"
    state.active_checkpoint = ckpt

    mode = "dry-run" if dry_run else "hardware"
    handler = make_handler(
        hw=hw, cfg=cfg, registry=registry,
        bus_lock=bus_lock, state=state,
        canvas_out_dir=canvas_out_dir, joints=joints, mode=mode,
        baseline_checkpoint=baseline_checkpoint,
    )

    with _ThreadedServer((host, port), handler) as httpd:
        url = f"http://{host}:{port}/"
        print(f"dashboard: {url}")
        print("  Ctrl-C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nshutting down...")
        finally:
            try:
                hw.disconnect()
            except Exception:
                pass


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--dry-run", action="store_true",
                   help="Use DryRunRobotInterface (no hardware required).")
    p.add_argument(
        "--baseline-checkpoint",
        default=None,
        help=(
            "Optional path to a second .pth checkpoint (e.g., a pre-learner "
            "model). When set, a model dropdown appears in the dashboard UI "
            "so you can compare predictions between the learner's live "
            "checkpoint and this baseline."
        ),
    )
    args = p.parse_args()
    serve(
        Path(args.config).resolve(), args.port, args.host, args.dry_run,
        baseline_checkpoint=args.baseline_checkpoint,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
