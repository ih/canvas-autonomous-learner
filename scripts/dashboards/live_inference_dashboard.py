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

BC policy mode (--bc-checkpoint): loads a trained BC policy (ResNet-18)
from canvas-bc-distill. The dashboard adds:

  - "BC Predict": captures the current camera frame, runs the BC policy,
    and shows the predicted action + per-class probabilities.
  - "BC Step": predicts an action with the BC policy AND executes it on
    the robot in one click — closed-loop testing.
  - "BC Auto": runs BC Step in a loop at ~1 Hz for hands-free evaluation.

Cameras are only read on demand (Predict / Execute click). There is no
continuous MJPEG feed — that was causing camera-enumeration conflicts
when restarting the dashboard.

This dashboard OWNS the robot + cameras for its lifetime. It MUST NOT be
run while `python -m learner` is driving the same arm — COM3 is single-
access on Windows.

Usage:
    python scripts/live_inference_dashboard.py --config configs/default.yaml
    python scripts/live_inference_dashboard.py --config configs/default.yaml --dry-run --port 8766
    python scripts/live_inference_dashboard.py --config configs/default.yaml \\
        --bc-checkpoint C:/Projects/canvas-bc-distill/checkpoints/bc_red_kong_2k/best.pt
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

# canvas-bc-distill lives next to canvas-autonomous-learner.
BC_DISTILL_ROOT = REPO_ROOT.parent / "canvas-bc-distill"
if BC_DISTILL_ROOT.exists():
    sys.path.insert(0, str(BC_DISTILL_ROOT))

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
    active_model: str = ""  # option key from `_build_model_options`
    active_checkpoint: Optional[str] = None  # path of currently loaded model
    bc_policy: Optional[object] = None  # BCPolicy model, if loaded
    bc_device: str = "cpu"
    bc_auto_running: bool = False

    def clear_before(self) -> None:
        self.before_motor = None
        self.before_ctx = None
        self.before_ts = None


# ----------------------------------------------------------- model options ---


def _best_from_registry(reg_path: Path) -> Optional[dict]:
    """Lowest-val-mse history entry from a learner registry, if any.

    Returns {'path', 'val_mse', 'session'} or None.
    """
    try:
        data = json.loads(reg_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    history = data.get("history") or []
    cands = [
        e for e in history
        if isinstance(e.get("val_mse"), (int, float))
        and e.get("new") and Path(e["new"]).exists()
    ]
    if not cands:
        return None
    best = min(cands, key=lambda e: e["val_mse"])
    return {
        "path": str(Path(best["new"]).resolve()),
        "val_mse": float(best["val_mse"]),
        "session": reg_path.parent.name,
    }


def _best_from_cold_warm_compare(cwc_path: Path) -> Optional[dict]:
    """Best cold-start checkpoint from a cold_warm_compare.json file.

    Picks the entry with the lowest `cold_train_val_mse` whose
    `cold_checkpoint` file still exists on disk. Returns {'path',
    'val_mse', 'locked_val_mse', 'wide_verify_mse', 'scene_count',
    'session'} or None.
    """
    try:
        data = json.loads(cwc_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, list):
        return None
    cands = [
        e for e in data
        if isinstance(e.get("cold_train_val_mse"), (int, float))
        and e.get("cold_checkpoint") and Path(e["cold_checkpoint"]).exists()
    ]
    if not cands:
        return None
    best = min(cands, key=lambda e: e["cold_train_val_mse"])
    return {
        "path": str(Path(best["cold_checkpoint"]).resolve()),
        "val_mse": float(best["cold_train_val_mse"]),
        "locked_val_mse": best.get("cold_locked_val_mse"),
        "wide_verify_mse": best.get("cold_wide_verify_mse"),
        "scene_count": int(best.get("scene_count", 0)),
        "session": cwc_path.parent.name,
    }


def _build_model_options(
    cfg,
    registry: "Registry",
    repo_root: Path,
    extra_baseline: Optional[str] = None,
) -> list[dict]:
    """Build the list of selectable model options.

    Re-evaluated each request so newly-promoted learner checkpoints show up
    immediately. Each entry: {key, label, path, note}. Entries whose
    underlying file is missing are skipped.
    """
    opts: list[dict] = []
    seen: set[str] = set()

    def _add(key: str, label: str, path: Optional[str], note: str) -> None:
        if not path:
            return
        resolved = str(Path(path).resolve())
        if not Path(resolved).exists():
            return
        if resolved in seen:
            return
        seen.add(resolved)
        opts.append({"key": key, "label": label, "path": resolved, "note": note})

    cwm_root = getattr(cfg.paths, "canvas_world_model", None)
    if cwm_root:
        diff4 = (Path(cwm_root) / "local" / "checkpoints"
                 / "diff_iter4_wider" / "best.pth")
        _add(
            "diff_iter4",
            "diff_iter4 (canvas-world-model baseline)",
            str(diff4) if diff4.exists() else None,
            f"600M-class baseline checkpoint: {diff4}",
        )

    # Best 1B: lowest val_mse in runs/sim_1b/registry.json history.
    sim_1b_reg = repo_root / "runs" / "sim_1b" / "registry.json"
    if sim_1b_reg.exists():
        b = _best_from_registry(sim_1b_reg)
        if b is not None:
            _add(
                "best_1b",
                f"best 1B (sim_1b, val_mse={b['val_mse']:.5f})",
                b["path"],
                f"Lowest-val-mse entry from runs/sim_1b/registry.json: "
                f"{b['path']}",
            )

    # Best pan_focus_1b: shoulder_pan-only policy, 1B cold-start.
    pan_focus_reg = repo_root / "runs" / "pan_focus_1b" / "registry.json"
    if pan_focus_reg.exists():
        b = _best_from_registry(pan_focus_reg)
        if b is not None:
            _add(
                "best_pan_focus_1b",
                f"best pan_focus_1b (val_mse={b['val_mse']:.5f})",
                b["path"],
                f"Lowest-val-mse entry from runs/pan_focus_1b/registry.json: "
                f"{b['path']}",
            )

    # Best pan_only_1b: lifelong-mode shoulder-only experiment. Note —
    # locked_val_mse here is auxiliary (not the routing target), so the
    # "best" picked is still the lowest-val_mse checkpoint in registry
    # history; that's the natural pick for the live-inference dashboard
    # which is itself a quality assessment tool, not a routing decision.
    pan_only_reg = repo_root / "runs" / "pan_only_1b" / "registry.json"
    if pan_only_reg.exists():
        b = _best_from_registry(pan_only_reg)
        if b is not None:
            _add(
                "best_pan_only_1b",
                f"best pan_only_1b (val_mse={b['val_mse']:.5f})",
                b["path"],
                f"Lowest-val-mse entry from runs/pan_only_1b/registry.json: "
                f"{b['path']}",
            )

    # Best 1B from red_kong_1b — the actual hardware experiment that
    # plateaued at locked_val ~0.011, train_val ~0.0008. This is the
    # 1B reference for visual-quality comparison against the 250M.
    red_kong_1b_reg = repo_root / "runs" / "red_kong_1b" / "registry.json"
    if red_kong_1b_reg.exists():
        b = _best_from_registry(red_kong_1b_reg)
        if b is not None:
            _add(
                "best_red_kong_1b",
                f"best 1B (red_kong_1b, val_mse={b['val_mse']:.5f})",
                b["path"],
                f"Lowest-val-mse entry from runs/red_kong_1b/registry.json: "
                f"{b['path']}",
            )

    # Best 250M from red_kong_min_data_250m — warm-trajectory best
    # (the live-checkpoint path the orchestrator promoted with the
    # lowest train_val_mse over the 15-cycle min-data sweep).
    rk250_reg = repo_root / "runs" / "red_kong_min_data_250m" / "registry.json"
    if rk250_reg.exists():
        b = _best_from_registry(rk250_reg)
        if b is not None:
            _add(
                "best_red_kong_min_data_250m",
                f"best 250M warm (red_kong_min_data_250m, val_mse={b['val_mse']:.5f})",
                b["path"],
                f"Lowest-val-mse entry from runs/red_kong_min_data_250m/"
                f"registry.json (warm fine-tune trajectory): {b['path']}",
            )

    # Best 250M cold-start from cold_warm_compare.json — the
    # retrospective cold-only sweep. Often beats the warm trajectory
    # at high scene counts (~10% lower locked_val + wide_verify in
    # the 2026-05-05 run).
    cwc_path = (
        repo_root / "runs" / "red_kong_min_data_250m" / "cold_warm_compare.json"
    )
    if cwc_path.exists():
        bc = _best_from_cold_warm_compare(cwc_path)
        if bc is not None:
            wv = bc.get("wide_verify_mse")
            lv = bc.get("locked_val_mse")
            stats = f"val_mse={bc['val_mse']:.5f}"
            if lv is not None:
                stats += f", locked_val={lv:.5f}"
            if wv is not None:
                stats += f", wide_verify={wv:.5f}"
            _add(
                "best_red_kong_min_data_250m_cold",
                f"best 250M cold (scene {bc['scene_count']}, {stats})",
                bc["path"],
                f"Lowest cold_train_val_mse in runs/red_kong_min_data_250m/"
                f"cold_warm_compare.json (scene_count={bc['scene_count']}): "
                f"{bc['path']}",
            )

    # Best across ALL autonomous-learner registries (sim, sim_1b, sim_smoke, ...).
    best_learner: Optional[dict] = None
    for reg in (repo_root / "runs").glob("*/registry.json"):
        b = _best_from_registry(reg)
        if b is None:
            continue
        if best_learner is None or b["val_mse"] < best_learner["val_mse"]:
            best_learner = b
    if best_learner is not None:
        _add(
            "best_learner",
            f"best autonomous-learner ({best_learner['session']}, "
            f"val_mse={best_learner['val_mse']:.5f})",
            best_learner["path"],
            f"Lowest-val-mse across runs/*/registry.json: "
            f"{best_learner['path']}",
        )

    # Currently-promoted live checkpoint from the configured registry —
    # only added if it isn't already present under one of the keys above.
    live = registry.live_checkpoint()
    if live:
        _add(
            "live",
            f"live (current registry: {Path(cfg.paths.registry_file).parent.name})",
            live,
            f"registry.live_checkpoint() from {cfg.paths.registry_file}: {live}",
        )

    if extra_baseline:
        _add(
            "extra_baseline",
            f"extra ({Path(extra_baseline).parent.name}/"
            f"{Path(extra_baseline).name})",
            extra_baseline,
            f"--baseline-checkpoint: {extra_baseline}",
        )

    return opts


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
    repo_root: Path,
):
    from control.robot_interface import JOINTS  # type: ignore

    def _options() -> list[dict]:
        return _build_model_options(cfg, registry, repo_root, baseline_checkpoint)

    def _render_index() -> bytes:
        return (
            INDEX_HTML
            .replace("__JOINTS__", json.dumps(joints))
            .replace("__CONTROL_JOINT__", cfg.robot.control_joint)
            .replace("__MODE__", mode)
            .replace("__MODEL_OPTIONS__", json.dumps(_options()))
            .replace("__INITIAL_MODEL_KEY__", state.active_model)
            .replace("__INITIAL_MODEL_CKPT__", state.active_checkpoint or "")
            .replace("__BC_ENABLED__", "true" if state.bc_policy is not None else "false")
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
                self._send_bytes(200, _render_index(), "text/html; charset=utf-8")
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
                elif path == "/api/bc_predict":
                    self._send_json(200, self._api_bc_predict())
                elif path == "/api/bc_step":
                    self._send_json(200, self._api_bc_step())
                elif path == "/api/bc_auto":
                    self._send_json(200, self._api_bc_auto(self._read_json()))
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
            key = str(body.get("model") or "").strip()
            if not key:
                raise ValueError("missing 'model' key")
            # Re-resolve at click time so "best_1b"/"best_learner"/"live"
            # always reflect the latest registry state.
            options = _options()
            match = next((o for o in options if o["key"] == key), None)
            if match is None:
                available = ", ".join(o["key"] for o in options) or "(none)"
                raise ValueError(
                    f"unknown model {key!r}; available: {available}"
                )
            target = match["path"]
            with bus_lock:
                print(f"[set_model] loading {key}: {target}", flush=True)
                t0 = time.time()
                hw.load_predictor(target)
                elapsed = time.time() - t0
                state.active_model = key
                state.active_checkpoint = target
                # Any "before" captured for the previous model's prediction
                # is still physically valid but no longer compares apples
                # to apples — drop it so the next Predict starts clean.
                state.clear_before()
                print(f"[set_model] loaded in {elapsed:.1f}s", flush=True)
            return {
                "active_model": state.active_model,
                "active_checkpoint": state.active_checkpoint,
                "active_label": match["label"],
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

        # --------------------------------------------- BC policy endpoints

        def _bc_infer(self) -> dict:
            """Capture frame, run BC policy, return action + probabilities."""
            if state.bc_policy is None:
                raise ValueError("no BC policy loaded (start with --bc-checkpoint)")
            import torch
            from canvas_bc_distill.actions import CLASS_TO_ACTION, ACTION_NAMES
            with bus_lock:
                _cams, motor, ctx = hw.observe()
            frame = cv2.resize(ctx, (224, 224), interpolation=cv2.INTER_AREA)
            tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
            tensor = tensor.to(state.bc_device)
            with torch.no_grad():
                logits = state.bc_policy(tensor)
                probs = torch.softmax(logits, dim=1)[0]
            pred_class = int(probs.argmax())
            action = CLASS_TO_ACTION[pred_class]
            prob_dict = {
                ACTION_NAMES[CLASS_TO_ACTION[i]]: round(float(probs[i]), 4)
                for i in range(len(probs))
            }
            return {
                "action": action,
                "action_name": ACTION_NAMES[action],
                "probabilities": prob_dict,
                "confidence": round(float(probs[pred_class]), 4),
                "motor_state": motor.tolist(),
            }

        def _api_bc_predict(self) -> dict:
            return self._bc_infer()

        def _api_bc_step(self) -> dict:
            result = self._bc_infer()
            action = result["action"]
            joint = cfg.robot.control_joint
            joint_idx = JOINTS.index(joint)
            with bus_lock:
                if not state.torque_on:
                    hw.lock()
                    state.torque_on = True
                hw.execute(action)
                _wait_until_motion_settled(hw)
                hw.observe()  # flush stale frame
                _cams, motor_after, ctx_after = hw.observe()
            result["motor_after"] = motor_after.tolist()
            result["joint"] = joint
            return result

        def _api_bc_auto(self, body: dict) -> dict:
            running = body.get("running", False)
            state.bc_auto_running = bool(running)
            return {"bc_auto_running": state.bc_auto_running}

    return Handler


# ----------------------------------------------------------------- server ---


class _ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(cfg_path: Path, port: int, host: str, dry_run: bool,
          baseline_checkpoint: Optional[str] = None,
          bc_checkpoint: Optional[str] = None) -> None:
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
    state.active_checkpoint = str(Path(ckpt).resolve())

    # Pick the option key that matches whichever checkpoint we actually
    # loaded at startup. Fall back to "live" so the dropdown still has
    # something selected even if the loaded path doesn't match a curated
    # entry (e.g. dashboard pointed at an ad-hoc registry).
    initial_options = _build_model_options(
        cfg, registry, REPO_ROOT, baseline_checkpoint
    )
    initial_key = next(
        (o["key"] for o in initial_options
         if o["path"] == state.active_checkpoint),
        "live",
    )
    state.active_model = initial_key
    print(f"  available models:")
    for o in initial_options:
        marker = "*" if o["key"] == initial_key else " "
        print(f"   {marker} {o['key']}: {o['label']}")

    if bc_checkpoint:
        import torch
        from canvas_bc_distill.policy.model import load_for_inference
        bc_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  loading BC policy: {bc_checkpoint} (device={bc_device})")
        state.bc_policy = load_for_inference(bc_checkpoint, device=bc_device)
        state.bc_device = bc_device
        print(f"  BC policy loaded.")

    mode = "dry-run" if dry_run else "hardware"
    handler = make_handler(
        hw=hw, cfg=cfg, registry=registry,
        bus_lock=bus_lock, state=state,
        canvas_out_dir=canvas_out_dir, joints=joints, mode=mode,
        baseline_checkpoint=baseline_checkpoint,
        repo_root=REPO_ROOT,
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
    p.add_argument(
        "--bc-checkpoint",
        default=None,
        help=(
            "Path to a trained BC policy checkpoint (best.pt from "
            "canvas-bc-distill). Enables BC Predict / BC Step / BC Auto "
            "controls in the dashboard."
        ),
    )
    args = p.parse_args()
    if args.bc_checkpoint and not Path(args.bc_checkpoint).exists():
        raise SystemExit(f"--bc-checkpoint does not exist: {args.bc_checkpoint}")
    serve(
        Path(args.config).resolve(), args.port, args.host, args.dry_run,
        baseline_checkpoint=args.baseline_checkpoint,
        bc_checkpoint=args.bc_checkpoint,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
