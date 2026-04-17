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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from learner.config import load_config  # noqa: E402
from learner.hardware import Hardware  # noqa: E402
from learner.registry import Registry  # noqa: E402


# ------------------------------------------------------------------- state ---


@dataclass
class DashboardState:
    """In-memory state shared across requests, guarded by `bus_lock`."""

    before_cameras: Optional[dict] = None
    before_motor: Optional[np.ndarray] = None
    before_ctx: Optional[np.ndarray] = None
    before_ts: Optional[float] = None
    torque_on: bool = True
    canvas_tokens: dict = field(default_factory=dict)  # token -> Path
    active_model: str = "learner"  # "learner" or "baseline"
    active_checkpoint: Optional[str] = None  # path of currently loaded model

    def clear_before(self) -> None:
        self.before_cameras = None
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
    """Block until motor position stops changing.

    Poll positions every `poll_interval` seconds. Consider the arm settled
    once the max-across-joints absolute delta stays under `stable_threshold`
    degrees for `stable_window` seconds. `min_wait` guarantees we don't
    return before the motors have had a chance to start moving. `timeout`
    caps the total wait so a stuck joint doesn't hang the request.

    Assumes the caller holds the bus lock.
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


def _stack_base_wrist(base: np.ndarray, wrist: np.ndarray,
                      frame_size=(224, 224)) -> np.ndarray:
    """Vertically stack base (top) and wrist (bottom) into a single
    training-format context frame shaped (2*H, W, 3)."""
    h, w = frame_size
    base_r = cv2.resize(base, (w, h), interpolation=cv2.INTER_LANCZOS4)
    wrist_r = cv2.resize(wrist, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return np.vstack([base_r, wrist_r])


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


def _save_canvas(canvas: np.ndarray, out_dir: Path, prefix: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S") + f"_{secrets.token_hex(3)}"
    path = out_dir / f"{prefix}_{ts}.png"
    from PIL import Image
    Image.fromarray(canvas).save(path)
    return path


# --------------------------------------------------------------- HTML UI ---


INDEX_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>live inference</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --ink:#e6e8ef; --muted:#8a93a6; --accent:#5aa9ff; --warn:#ff9c5a; --ok:#65d88c; --bad:#ff6464; }
  html, body { background: var(--bg); color: var(--ink); font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; }
  header { padding: 12px 20px; border-bottom: 1px solid #2a2f3b; display: flex; align-items: center; gap: 14px; }
  header h1 { font-size: 15px; font-weight: 600; margin: 0; }
  .pill { font-size: 11px; padding: 3px 9px; border-radius: 12px; background:#2a2f3b; color: var(--muted); }
  .pill.ok { background:#12361e; color: var(--ok); }
  .pill.bad { background:#3a1212; color: var(--bad); }
  main { display: grid; grid-template-columns: minmax(320px, 480px) 1fr; gap: 14px; padding: 14px 20px; align-items: start; }
  .panel { background: var(--panel); border: 1px solid #232735; border-radius: 8px; padding: 12px 14px; }
  .panel h2 { font-size: 11px; margin: 0 0 8px 0; text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted); }
  img.canvas { display: block; max-width: 100%; width: auto; height: auto; max-height: calc((100vh - 140px) / 2); margin: 0 auto; border: 1px solid #2a2f3b; border-radius: 4px; background: #000; object-fit: contain; }
  label { display: block; font-size: 12px; color: var(--muted); margin: 8px 0 4px; }
  select, button { background: #1d2230; color: var(--ink); border: 1px solid #2b3246; padding: 6px 10px; border-radius: 4px; font-size: 13px; font-family: inherit; }
  button { cursor: pointer; }
  button.primary { background: #2e9a4f; color: #fff; border: 0; padding: 8px 16px; font-weight: 600; }
  button.primary:hover { background: #37b35a; }
  button.warn { background: #b35a2e; color: #fff; border: 0; padding: 8px 16px; font-weight: 600; }
  button.warn:hover { background: #cc6933; }
  button.muted { background: #2a2f3b; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-top: 8px; }
  .radio-group { display: flex; gap: 12px; margin-top: 4px; }
  .radio-group label { color: var(--ink); font-size: 13px; display: inline-flex; align-items: center; gap: 4px; margin: 0; }
  .status { font-size: 12px; color: var(--muted); margin-top: 8px; min-height: 16px; }
  .status.err { color: var(--bad); }
  .status.ok { color: var(--ok); }
  .note { font-size: 11px; color: var(--muted); margin-top: 10px; padding: 6px 8px; background: #12151c; border-left: 2px solid #3a4254; border-radius: 2px; }
  .before-ts { font-size: 11px; color: var(--muted); margin-top: 6px; font-family: ui-monospace, Menlo, Consolas, monospace; }
  .canvas-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
  .empty { color: var(--muted); font-style: italic; font-size: 12px; padding: 20px; text-align: center; }
</style></head><body>

<header>
  <h1>live-inference dashboard</h1>
  <span class="pill" id="mode-pill">—</span>
  <span class="pill ok" id="torque-pill">torque: on</span>
  <span class="pill" id="joint-pill">joint: —</span>
  <span class="pill" id="model-pill">model: —</span>
  <span class="pill" id="ckpt-pill">ckpt: —</span>
</header>

<main>
  <div class="panel">
    <h2>Controls</h2>

    <label for="model-select">Model</label>
    <select id="model-select">
      <option value="learner">learner (live from registry)</option>
      <option value="baseline">baseline (pre-learner)</option>
    </select>
    <div class="note" id="model-note"></div>

    <label for="joint-select">Joint</label>
    <select id="joint-select"></select>
    <div class="note" id="joint-note"></div>

    <label>Action</label>
    <div class="radio-group">
      <label><input type="radio" name="action" value="1" checked> move+</label>
      <label><input type="radio" name="action" value="2"> move-</label>
      <label><input type="radio" name="action" value="3"> hold</label>
    </div>

    <div class="row">
      <button class="primary" id="btn-predict">Predict</button>
      <button class="primary" id="btn-execute">Execute on robot</button>
      <button class="muted" id="btn-clear">Clear before</button>
    </div>
    <div class="before-ts" id="before-ts">before: (none captured)</div>
    <div class="status" id="status"></div>

    <label style="margin-top: 18px;">Joint torque</label>
    <div class="row">
      <button class="warn" id="btn-relax">Relax joints</button>
      <button class="primary" id="btn-lock">Lock joints</button>
    </div>
  </div>

  <div>
    <div class="panel" style="margin-bottom: 14px;">
      <div class="canvas-label">Predicted canvas <span style="color:var(--muted);font-transform:none;text-transform:none;">(before | action_sep | predicted)</span></div>
      <div id="predicted-wrap"><div class="empty">(click Predict)</div></div>
    </div>
    <div class="panel">
      <div class="canvas-label">Actual canvas <span style="text-transform:none;">(before | action_sep | actual)</span></div>
      <div id="actual-wrap"><div class="empty">(click Execute)</div></div>
    </div>
  </div>
</main>

<script>
const JOINTS = __JOINTS__;
const CONTROL_JOINT = "__CONTROL_JOINT__";
const MODE = "__MODE__";
const LEARNER_CKPT = "__LEARNER_CKPT__";
const BASELINE_CKPT = "__BASELINE_CKPT__";
const HAS_BASELINE = __HAS_BASELINE__;
const INITIAL_MODEL = "__INITIAL_MODEL__";

document.getElementById("mode-pill").textContent = "mode: " + MODE;
document.getElementById("joint-pill").textContent = "joint: " + CONTROL_JOINT;

function shortCkpt(p) {
  if (!p) return "—";
  const parts = p.replace(/\\/g, "/").split("/");
  return parts.slice(-2).join("/");
}
function updateModelPills(label, ckpt) {
  document.getElementById("model-pill").textContent = "model: " + label;
  document.getElementById("ckpt-pill").textContent = "ckpt: " + shortCkpt(ckpt);
  document.getElementById("model-pill").className = "pill " + (label === "learner" ? "ok" : "");
}
updateModelPills(INITIAL_MODEL, INITIAL_MODEL === "baseline" ? BASELINE_CKPT : LEARNER_CKPT);

const modelSelect = document.getElementById("model-select");
modelSelect.value = INITIAL_MODEL;
if (!HAS_BASELINE) {
  // Disable the baseline option if no baseline path was configured.
  for (const opt of modelSelect.options) {
    if (opt.value === "baseline") { opt.disabled = true; opt.textContent += " (not configured — pass --baseline-checkpoint)"; }
  }
}
function updateModelNote() {
  const note = document.getElementById("model-note");
  const v = modelSelect.value;
  if (v === "learner") {
    note.textContent = "Live checkpoint from the autonomous-learner registry: " + LEARNER_CKPT;
    note.style.borderLeftColor = "#3a4254";
  } else {
    note.textContent = "Pre-learner baseline: " + BASELINE_CKPT;
    note.style.borderLeftColor = "#3a4254";
  }
}
updateModelNote();
modelSelect.addEventListener("change", async () => {
  updateModelNote();
  const chosen = modelSelect.value;
  setStatus("loading " + chosen + " model (this may take 10-30s)...", "");
  try {
    const res = await postJSON("/api/set_model", {model: chosen});
    updateModelPills(res.active_model, res.active_checkpoint);
    setStatus("loaded model: " + res.active_model, "ok");
  } catch (e) {
    setStatus("model switch failed: " + e.message, "err");
    // roll the dropdown back so UI matches server state
    modelSelect.value = INITIAL_MODEL;
    updateModelNote();
  }
});

const jointSelect = document.getElementById("joint-select");
for (const j of JOINTS) {
  const opt = document.createElement("option");
  opt.value = j;
  opt.textContent = j;
  if (j === CONTROL_JOINT) opt.selected = true;
  jointSelect.appendChild(opt);
}

function updateJointNote() {
  const j = jointSelect.value;
  const note = document.getElementById("joint-note");
  if (j === CONTROL_JOINT) {
    note.textContent = "Model was trained on this joint — predictions are in-distribution.";
    note.style.borderLeftColor = "#3a4254";
  } else {
    note.textContent = "Off-joint QA probe: the world model was trained on '" + CONTROL_JOINT + "', so predictions on this joint may be out-of-distribution.";
    note.style.borderLeftColor = "#b35a2e";
  }
}
jointSelect.addEventListener("change", updateJointNote);
updateJointNote();

function currentAction() {
  const el = document.querySelector('input[name=action]:checked');
  return el ? parseInt(el.value, 10) : 3;
}

function setStatus(msg, cls) {
  const el = document.getElementById("status");
  el.textContent = msg;
  el.className = "status " + (cls || "");
}

function setBeforeTs(ts) {
  const el = document.getElementById("before-ts");
  if (ts == null) {
    el.textContent = "before: (none captured)";
  } else {
    const d = new Date(ts * 1000);
    el.textContent = "before captured at: " + d.toLocaleTimeString();
  }
}

function setTorque(on) {
  const pill = document.getElementById("torque-pill");
  pill.textContent = "torque: " + (on ? "on" : "off");
  pill.className = on ? "pill ok" : "pill bad";
}

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body || {}),
  });
  const t = await r.text();
  let j;
  try { j = JSON.parse(t); } catch { j = {error: t}; }
  if (!r.ok) throw new Error(j.error || ("HTTP " + r.status));
  return j;
}

function renderCanvas(wrapId, token, meta) {
  const w = document.getElementById(wrapId);
  if (!token) { w.innerHTML = '<div class="empty">(no canvas)</div>'; return; }
  w.innerHTML = '<img class="canvas" src="/canvas/' + token + '?t=' + Date.now() + '"/>' +
    (meta ? '<div class="before-ts">' + meta + '</div>' : '');
}

document.getElementById("btn-predict").addEventListener("click", async () => {
  setStatus("predicting...", "");
  const btn = document.getElementById("btn-predict");
  btn.disabled = true;
  try {
    const res = await postJSON("/api/predict", {
      action: currentAction(), joint: jointSelect.value,
    });
    setBeforeTs(res.before_captured_at);
    renderCanvas("predicted-wrap", res.canvas_token, "action=" + currentAction() + ", joint=" + jointSelect.value);
    setStatus("predicted.", "ok");
  } catch (e) {
    setStatus("predict failed: " + e.message, "err");
  } finally {
    btn.disabled = false;
  }
});

document.getElementById("btn-execute").addEventListener("click", async () => {
  if (!confirm("Execute on the real robot?")) return;
  setStatus("executing...", "");
  const btn = document.getElementById("btn-execute");
  btn.disabled = true;
  try {
    const res = await postJSON("/api/execute", {
      action: currentAction(), joint: jointSelect.value,
    });
    setBeforeTs(null);  // execute clears the before
    renderCanvas("actual-wrap", res.canvas_token, "action=" + currentAction() + ", joint=" + jointSelect.value);
    setStatus("executed.", "ok");
  } catch (e) {
    setStatus("execute failed: " + e.message, "err");
  } finally {
    btn.disabled = false;
  }
});

document.getElementById("btn-clear").addEventListener("click", async () => {
  try {
    await postJSON("/api/clear_before", {});
    setBeforeTs(null);
    setStatus("cleared before.", "ok");
  } catch (e) {
    setStatus("clear failed: " + e.message, "err");
  }
});

document.getElementById("btn-relax").addEventListener("click", async () => {
  setStatus("relaxing joints...", "");
  try {
    await postJSON("/api/relax", {});
    setTorque(false);
    setStatus("joints relaxed — move by hand.", "ok");
  } catch (e) {
    setStatus("relax failed: " + e.message, "err");
  }
});

document.getElementById("btn-lock").addEventListener("click", async () => {
  setStatus("locking joints...", "");
  try {
    await postJSON("/api/lock", {});
    setTorque(true);
    setStatus("joints locked at current position.", "ok");
  } catch (e) {
    setStatus("lock failed: " + e.message, "err");
  }
});
</script>
</body></html>
"""


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
        path = _save_canvas(canvas, canvas_out_dir, prefix)
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

            All stored arrays are copied so downstream mutation (e.g. the
            predictor's input normalization) can't corrupt the "before"
            frame that Execute will later reuse.
            """
            if state.before_ctx is not None:
                return
            cameras, motor, ctx = hw.observe()
            state.before_cameras = {k: np.asarray(v).copy()
                                    for k, v in cameras.items()}
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
