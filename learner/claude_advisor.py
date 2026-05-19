"""Claude as the orchestrator's THINK phase.

The autonomous learner's THINK branch blocks on a `claude -p "<prompt>"`
subprocess. Claude reads a JSON context snapshot (training curve, locked-
val history, current knobs + curriculum + training hyperparameters, most
recent verify summary, Arm A target) and returns a decision dict:

    {
      "next_state": "verify" | "explore" | "retrain" | "idle" | "terminate",
      "reason": "...",
      "scene_change_description": "<required when next_state=idle>",
      "runtime_overrides": { ... },
      "training_overrides": { ... },
      "curriculum_overrides": { ... },
      "explore_overrides": { ... },
      "from_scratch": false
    }

The orchestrator applies the overrides, then routes on `next_state`. A
dead advisor must never block the state machine, so timeouts and
exceptions fail open to `next_state = default`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional


# ------------------------------------------------------------------- JSON I/O


_JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)


def parse_response(raw: str) -> dict:
    """Extract the first JSON object from a `claude -p` response.

    Tolerates: markdown-fenced code blocks, leading prose, trailing
    commentary, an optional model name banner, ANSI control sequences.
    Returns `{}` if no parseable object is found (caller treats as
    'no decision').
    """
    if not raw:
        return {}
    # Strip ANSI escape sequences (`claude -p` sometimes emits spinner
    # control codes even in non-interactive mode).
    ansi = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
    raw = ansi.sub("", raw)

    # First try a fenced JSON block.
    m = _JSON_BLOCK_RE.search(raw)
    candidates: list[str] = []
    if m:
        candidates.append(m.group(1))

    # Fall back to the first balanced-brace substring that starts with `{`.
    depth = 0
    start = -1
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                candidates.append(raw[start:i + 1])
                start = -1

    for text in candidates:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


# --------------------------------------------------------------- run_advisor


_DECISION_SCHEMA = {
    "type": "object",
    "required": ["next_state"],
    "properties": {
        "next_state": {
            "type": "string",
            "enum": ["verify", "explore", "retrain", "idle", "terminate"],
        },
        "reason": {"type": "string"},
        "scene_change_description": {"type": "string"},
        "runtime_overrides": {"type": "object"},
        "training_overrides": {"type": "object"},
        "curriculum_overrides": {"type": "object"},
        "explore_overrides": {"type": "object"},
        "from_scratch": {"type": "boolean"},
        # Promotes the next VERIFY to a wide (multi-scene) burst, where
        # the operator is prompted via the dashboard between probe
        # chunks. Use when plateau_signal indicates a stuck or high-
        # locked-val plateau and you want a fresh cross-scene
        # generalization snapshot.
        "wide_verify_next": {"type": "boolean"},
        # Optional self-reflection: after reviewing training results,
        # if the advisor noticed the prompt's framing led it astray
        # (or could be sharpened), it can suggest a concrete edit
        # here. Captured in the event log so the operator can review
        # accumulated suggestions and apply prompt edits offline.
        "prompt_improvement_suggestion": {"type": "string"},
    },
    "additionalProperties": False,
}


def run_advisor(
    prompt_text: str,
    *,
    timeout_s: float = 1800.0,
    model: Optional[str] = None,
    effort: Optional[str] = None,
    default_next_state: str = "verify",
    add_dir: Optional[str] = None,
    event_log=None,
) -> dict:
    """Spawn `claude -p` with the given prompt, stream stdout, return the
    parsed decision dict. On timeout, missing binary, non-zero exit, or
    parse failure, log a `claude_advisor_failed` event and return a
    fail-open default that routes to `default_next_state`.

    `model` controls `--model` (e.g. "opus", "sonnet", or a full model
    slug). `effort` controls `--effort` (`low` / `medium` / `high` / `max`)
    which gates the amount of extended thinking Claude uses. A JSON
    schema is passed via `--json-schema` so Claude must produce a valid
    decision object rather than freeform prose.
    """
    t0 = time.time()
    claude_bin = shutil.which("claude") or "claude"

    if shutil.which("claude") is None:
        if event_log is not None:
            event_log.log(
                "claude_advisor_failed",
                reason="claude binary not found on PATH",
            )
        return _fail_open(default_next_state)

    # Build argv WITHOUT the prompt (we pipe it via stdin). Windows
    # cmd.exe caps command-line length at ~8 KB; our prompt with the
    # embedded JSON snapshot routinely runs 8–12 KB, so passing it as
    # an argv gives "The command line is too long" + non-zero exit.
    cmd = [
        claude_bin, "-p",
        # --output-format=json is REQUIRED when using --json-schema.
        # The schema-validated decision lands in the envelope's
        # `structured_output` field; `result` itself is empty.
        "--output-format", "json",
        "--json-schema", json.dumps(_DECISION_SCHEMA),
        # Allow Claude to Read the action canvas PNGs listed in the
        # prompt. Read is the only tool it needs.
        "--allowed-tools", "Read",
    ]
    if add_dir:
        cmd.extend(["--add-dir", add_dir])
    if model:
        cmd.extend(["--model", model])
    if effort:
        cmd.extend(["--effort", effort])

    if event_log is not None:
        event_log.log(
            "claude_advisor_start",
            cmd_head=cmd[:2],
            model=model,
            effort=effort,
            timeout_s=timeout_s,
            prompt_chars=len(prompt_text),
        )

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            # Force UTF-8 in/out so unicode in prompts (e.g. arrows,
            # em-dashes) doesn't crash the advisor on Windows cp1252.
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, FileNotFoundError) as e:
        if event_log is not None:
            event_log.log(
                "claude_advisor_failed", reason=f"spawn: {e}",
            )
        return _fail_open(default_next_state)

    # Feed the prompt via stdin so the argv stays short.
    try:
        if proc.stdin is not None:
            proc.stdin.write(prompt_text)
            proc.stdin.close()
    except (BrokenPipeError, OSError):
        pass

    chunks: list[str] = []
    assert proc.stdout is not None
    try:
        deadline = t0 + timeout_s
        while True:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
            else:
                chunks.append(line)
            if time.time() > deadline:
                if event_log is not None:
                    event_log.log(
                        "claude_advisor_timeout",
                        elapsed_s=time.time() - t0,
                    )
                try:
                    proc.kill()
                except Exception:
                    pass
                return _fail_open(default_next_state, reason="timeout")
    finally:
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass

    raw = "".join(chunks)
    if proc.returncode != 0:
        if event_log is not None:
            event_log.log(
                "claude_advisor_failed",
                reason=f"non-zero exit {proc.returncode}",
                stdout_tail=raw[-500:],
            )
        return _fail_open(default_next_state)

    # --output-format=json wraps the reply in an envelope:
    #   {"type":"result","result":"", "structured_output": {...}, ...}
    # The schema-validated decision lives under `structured_output`.
    decision: dict = {}
    envelope = parse_response(raw)
    if isinstance(envelope, dict):
        if isinstance(envelope.get("structured_output"), dict):
            decision = envelope["structured_output"]
        elif isinstance(envelope.get("result"), str) and envelope["result"]:
            # Fallback for older CLI versions that inline the decision
            # as a JSON string in `result`.
            decision = parse_response(envelope["result"])
        elif envelope.get("next_state"):
            # Bare decision (no envelope) — older behavior.
            decision = envelope

    if not decision or not decision.get("next_state"):
        if event_log is not None:
            event_log.log(
                "claude_advisor_failed",
                reason="no parseable decision in response",
                stdout_tail=raw[-500:],
            )
        return _fail_open(default_next_state)

    if event_log is not None:
        event_log.log(
            "claude_advisor_response",
            elapsed_s=time.time() - t0,
            decision=decision,
        )
    return decision


def _fail_open(default_next_state: str, reason: str = "advisor_failed") -> dict:
    return {
        "next_state": default_next_state,
        "reason": reason,
        "runtime_overrides": {},
        "training_overrides": {},
        "curriculum_overrides": {},
        "explore_overrides": {},
        "from_scratch": False,
    }


# --------------------------------------------------------- context snapshot


def _arm_a_target(runs_dir: Path) -> Optional[float]:
    p = runs_dir / "arm_a_result.json"
    if not p.exists():
        return 0.0375  # documented Arm A baseline
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return 0.0375
    for key in ("arm_a_locked_val_mse", "locked_val_mse", "val_mse_visual"):
        if key in data:
            try:
                return float(data[key])
            except (TypeError, ValueError):
                continue
    return 0.0375


def _read_recent_events(events_path: Path, n: int = 400) -> list[dict]:
    if not events_path.exists():
        return []
    try:
        with events_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []
    out: list[dict] = []
    for line in lines[-n:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _extract_last_training_curve(events: list[dict]) -> dict:
    """Isolate training_progress events from the most recent retrain_start
    forward. Returns {'epochs': [...], 'train_loss': [...], 'val_loss':
    [...], 'best_val': [...], 'train_canvases': int, 'val_canvases': int}.
    """
    start_idx: Optional[int] = None
    for i in range(len(events) - 1, -1, -1):
        if events[i].get("event") == "retrain_start":
            start_idx = i
            break
    if start_idx is None:
        return {}
    tail = events[start_idx:]
    progress = [e for e in tail if e.get("event") == "training_progress"]
    size_ev = next(
        (e for e in tail if e.get("event") == "training_dataset_size"),
        None,
    )
    return {
        "epochs": [int(e.get("epoch", 0)) for e in progress],
        "total_epochs": (progress[-1].get("total_epochs") if progress else None),
        "train_loss": [float(e.get("train_loss", 0)) for e in progress],
        "val_loss": [float(e.get("val_loss", 0)) for e in progress],
        "best_val": [
            (float(e["best_val"]) if e.get("best_val") is not None else None)
            for e in progress
        ],
        "train_canvases": (
            int(size_ev.get("train_canvases", 0)) if size_ev else None
        ),
        "val_canvases": (
            int(size_ev.get("val_canvases", 0)) if size_ev else None
        ),
    }


def _recent_action_canvas_paths(runs_dir: Path, n: int = 5) -> list[str]:
    """Absolute paths to the newest action canvas PNGs across all
    examples_* dirs in `runs_dir`. Latest first.
    """
    if not runs_dir.exists():
        return []
    pngs: list[Path] = []
    for examples_dir in runs_dir.glob("examples_*"):
        if not examples_dir.is_dir():
            continue
        pngs.extend(examples_dir.glob("action_canvas_*.png"))
    if not pngs:
        return []
    pngs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p.resolve()) for p in pngs[:n]]


def _extract_recent_verifies(events: list[dict], n: int = 5) -> list[dict]:
    verifies = [e for e in events if e.get("event") == "verify_summary"]
    return [
        {
            "cycle": v.get("cycle"),
            "mean_err": v.get("mean_err"),
            "n_in_range": v.get("n_in_range"),
            "active_range": v.get("active_range"),
        }
        for v in verifies[-n:]
    ]


def _extract_scene_change_history(events: list[dict]) -> dict:
    """Walk events for scene-change requests + acks across the entire run
    and return a summary the advisor can reason about.

    Returns a dict with:
      - `total_requested`: count of `claude_scene_change_requested` events
      - `total_acknowledged`: count of `scene_ready_acknowledged` events
      - `pending`: True if the latest request has no matching ack yet
      - `recent`: up to 10 most-recent (cycle, description, requested_at,
        acknowledged_at) entries — newest last — so the advisor can avoid
        repeating the same instruction or asking too rapidly.

    The human operator is the ONLY source of physical scene perturbations
    in this run, so the advisor needs the running count to budget how
    often it asks. Repeated "move object N cm to the left" requests
    every cycle waste the operator's attention.
    """
    requests: list[dict] = []
    acks: list[float] = []
    for e in events:
        ev = e.get("event")
        if ev == "claude_scene_change_requested":
            requests.append({
                "cycle": e.get("cycle"),
                "description": e.get("description", ""),
                "requested_at": e.get("t"),
                "acknowledged_at": None,
            })
        elif ev == "scene_ready_acknowledged":
            acks.append(float(e.get("t") or 0.0))
            # Pair with the most recent unacked request.
            for r in reversed(requests):
                if r["acknowledged_at"] is None:
                    r["acknowledged_at"] = e.get("t")
                    break
    pending = bool(requests) and requests[-1]["acknowledged_at"] is None
    return {
        "total_requested": len(requests),
        "total_acknowledged": len(acks),
        "pending": pending,
        "recent": requests[-10:],
    }


def _extract_recent_advisor_decisions(events: list[dict], n: int = 10) -> list[dict]:
    """Surface the advisor's own prior THINK decisions — `claude_think`
    events — so the next THINK can see what it already tried. Each
    `claude_think` carries the full `advice` dict written by the
    orchestrator (see orchestrator.py, the line that logs the event).
    """
    thinks = [e for e in events if e.get("event") == "claude_think"]
    out = []
    for ev in thinks[-n:]:
        advice = ev.get("advice") or {}
        out.append({
            "cycle": ev.get("cycle"),
            "next_state": advice.get("next_state"),
            "reason": advice.get("reason"),
            "runtime_overrides": advice.get("runtime_overrides") or {},
            "training_overrides": advice.get("training_overrides") or {},
            "curriculum_overrides": advice.get("curriculum_overrides") or {},
            "explore_overrides": advice.get("explore_overrides") or {},
            "from_scratch": bool(advice.get("from_scratch")),
        })
    return out


_GPU_EVENT_TYPES = frozenset({
    "gpu_memory_sample",
    "gpu_memory_warn",
    "training_memory_abort",
    "training_stalled",
    "training_throughput_collapse",
    "training_throughput_baseline",
    "subprocess_timeout",
    "retrain_memory_abort_routed_to_think",
    "retrain_stalled_routed_to_think",
    "retrain_throughput_collapse_routed_to_think",
    "retrain_timeout_routed_to_think",
    "inference_oom",
    "verify_gpu_headroom",
})


def _extract_recent_gpu_signals(events: list[dict], n: int = 20) -> list[dict]:
    """Surface GPU-related events (pressure samples, aborts, stalls,
    inference OOM, post-probe VRAM headroom snapshots) so the advisor
    can reason about VRAM state and recent failures.
    """
    out = []
    for ev in events:
        if ev.get("event") in _GPU_EVENT_TYPES:
            entry = {
                "t": ev.get("t"),
                "cycle": ev.get("cycle"),
                "event": ev.get("event"),
            }
            if ev.get("tag") is not None:
                entry["tag"] = ev.get("tag")
            if ev.get("summary") is not None:
                entry["summary"] = ev.get("summary")
            else:
                sample = {}
                for k in ("used_mb", "total_mb", "used_frac", "util_pct"):
                    if ev.get(k) is not None:
                        sample[k] = ev.get(k)
                if sample:
                    entry["summary"] = sample
            if ev.get("seconds_since_last_progress") is not None:
                entry["seconds_since_last_progress"] = ev.get(
                    "seconds_since_last_progress"
                )
            if ev.get("timeout_s") is not None:
                entry["timeout_s"] = ev.get("timeout_s")
            if ev.get("error") is not None:
                entry["error"] = str(ev.get("error"))[:400]
            out.append(entry)
    return out[-n:]


def _curriculum_snapshot(curriculum) -> Optional[dict]:
    if curriculum is None:
        return None
    snap: dict = {
        "stage": curriculum.stage,
        "active_joint": curriculum.active_joint_name,
        "active_range": list(curriculum.active_range),
        "primary": {
            "control_joint": curriculum.primary.control_joint,
            "active": list(curriculum.primary.active),
            "full_min": curriculum.primary.full_min,
            "full_max": curriculum.primary.full_max,
            "stable_cycles": curriculum.primary.stable_cycles,
            "stable_cycles_required": curriculum.primary.stable_cycles_required,
            "expansion_factor": curriculum.primary.expansion_factor,
            "history": list(curriculum.primary.history),
        },
    }
    if curriculum.secondary is not None:
        snap["secondary"] = {
            "control_joint": curriculum.secondary.control_joint,
            "active": list(curriculum.secondary.active),
            "full_min": curriculum.secondary.full_min,
            "full_max": curriculum.secondary.full_max,
            "stable_cycles": curriculum.secondary.stable_cycles,
            "stable_cycles_required": curriculum.secondary.stable_cycles_required,
            "expansion_factor": curriculum.secondary.expansion_factor,
            "pinned_half_width": curriculum.secondary_pinned_half_width,
            "history": list(curriculum.secondary.history),
        }
    return snap


def _training_cfg_snapshot(cfg) -> dict:
    training = getattr(cfg, "training", None)
    if training is None:
        return {}
    fields = [
        "patch_size", "embed_dim", "depth", "num_heads",
        "num_train_timesteps", "beta_schedule", "prediction_type",
        "lr", "weight_decay", "lr_schedule", "warmup_epochs", "min_lr",
        "grad_clip", "batch_size", "seed", "val_ratio",
    ]
    return {k: getattr(training, k, None) for k in fields}


def _cadence_snapshot(cfg) -> dict:
    cadence = getattr(cfg, "cadence", None)
    if cadence is None:
        return {}
    fields = [
        "cold_start_epochs", "ft_epochs", "early_stop_patience",
        "base_explore_batch_size", "explore_batch_size_min",
        "explore_batch_size_max", "max_sub_bursts", "min_sub_burst_size",
    ]
    return {k: getattr(cadence, k, None) for k in fields}


def snapshot_run_context(
    events_path: Path,
    registry,
    cfg,
    knobs,
    curriculum=None,
    *,
    default_next_state: str = "verify",
    consecutive_retrains_without_data: int = 0,
    claude_max_consecutive_retrains: int = 5,
    last_scene_change: Optional[dict] = None,
    pending_explore_overrides: Optional[dict] = None,
    pending_novelty_report: Optional[dict] = None,
) -> dict:
    """Assemble the JSON-serializable context dict `build_think_prompt`
    consumes. Pure function — no side effects.

    When `cfg.cadence.lifelong_mode` is true the snapshot is restructured:
      - `goal.current_locked_val_mse` is dropped (the locked-val set is
        not the optimization target in lifelong mode).
      - `locked_val_history` moves to `auxiliary.locked_val_history` so
        the advisor still has it as a tiebreaker but the prompt's "drive
        this number low" framing no longer points at it.
      - A `plateau_signal` field is added with the structured plateau
        verdict (`improving` / `stuck` / `plateau_low_locked_val` /
        `plateau_high_locked_val` / `insufficient_data`).
    """
    from .plateau import plateau_summary  # local import: avoids cycle

    runs_dir = Path(cfg.paths.runs_dir)
    events = _read_recent_events(events_path, n=600)
    locked_history = registry.locked_val_history()
    recent_canvases = _recent_action_canvas_paths(runs_dir, n=5)
    last_locked_val = None
    for entry in reversed(locked_history):
        if entry.get("accepted") and entry.get("locked_val_mse") is not None:
            last_locked_val = entry["locked_val_mse"]
            break

    lifelong = bool(getattr(getattr(cfg, "cadence", None), "lifelong_mode", True))
    recent_verifies = _extract_recent_verifies(events)
    plateau = plateau_summary(
        locked_history,
        verify_history=recent_verifies,
    )

    if lifelong:
        # In lifelong mode the goal is qualitative (keep the rolling
        # mean_err under tau_low) — we don't pin a numeric target. The
        # locked-val set is auxiliary context only, never the metric to
        # minimize.
        goal: dict = {
            "mode": "lifelong",
            "objective": (
                "Keep VERIFY mean_err below tau_low. When it rises above "
                "tau_high, diagnose and route accordingly. There is no "
                "fixed termination — this loop runs as long as the robot "
                "operates."
            ),
            "tau_low": getattr(knobs, "tau_low", None),
            "tau_high": getattr(knobs, "tau_high", None),
        }
        auxiliary = {
            "locked_val_history": locked_history,
            "current_locked_val_mse": last_locked_val,
            "note": (
                "Locked-val is computed for human inspection only and is "
                "NOT the optimization target in lifelong_mode. Use it as "
                "a tiebreaker when interpreting plateau_signal: a plateau "
                "with low locked_val is acceptable; with high locked_val "
                "it indicates the model is stuck on the current data and "
                "needs new episodes or a scene change."
            ),
        }
    else:
        goal = {
            "arm_a_locked_val_mse": _arm_a_target(runs_dir),
            "current_locked_val_mse": last_locked_val,
        }
        auxiliary = None

    # Wide-verify corpus stats — primary generalization signal in
    # lifelong mode. Empty in legacy mode (wide-verify hasn't run) or
    # in early lifelong cycles.
    wv_history = []
    try:
        wv_history = registry.wide_verify_history()
    except AttributeError:
        # Older registry instances without the wide-verify accumulator.
        pass
    wv_summary = {
        "corpus_size": len(wv_history),
        "history": wv_history[-10:],  # last 10 wide-verify bursts
    }

    snapshot = {
        "goal": goal,
        "cycle": len(locked_history),
        "episodes_collected": registry.episodes_collected(),
        "accumulated_canvas_dirs": len(registry.accumulated_canvas_dirs()),
        "consecutive_guard_rejections": registry.consecutive_guard_rejections(),
        "experiment_status": registry.experiment_status(),
        "wide_verify": wv_summary,
        "curriculum": _curriculum_snapshot(curriculum),
        "knobs": knobs.as_dict(),
        "training_cfg": _training_cfg_snapshot(cfg),
        "cadence_cfg": _cadence_snapshot(cfg),
        "last_training_curve": _extract_last_training_curve(events),
        "recent_verifies": recent_verifies,
        "recent_advisor_decisions": _extract_recent_advisor_decisions(events, n=10),
        "recent_gpu_signals": _extract_recent_gpu_signals(events, n=20),
        "default_next_state": default_next_state,
        "advisor_budget": {
            "consecutive_retrains_without_data": consecutive_retrains_without_data,
            "claude_max_consecutive_retrains": claude_max_consecutive_retrains,
        },
        "last_scene_change": last_scene_change,
        "scene_change_history": _extract_scene_change_history(events),
        "plateau_signal": plateau,
        "pending_explore_overrides": pending_explore_overrides or {},
        "pending_novelty_report": pending_novelty_report or None,
        "recent_action_canvas_paths": recent_canvases,
    }
    if lifelong:
        snapshot["auxiliary"] = auxiliary
    else:
        snapshot["locked_val_history"] = locked_history
    return snapshot


# ------------------------------------------------------------- prompt


_GOAL_BLOCK_LEGACY = """\
## Your goal

Drive the world model's `locked_val_mse` as low as it can reasonably
go. The Arm A offline baseline is the **reference point** — not a
finish line. Beating it is the minimum bar, not the stopping
condition. Keep pushing.

```
arm_a_locked_val_mse: {arm_a}    (reference, not target)
current_locked_val_mse: {current}
```

Lower is always better. Both metrics come from the same held-out
locked_val dataset so they are directly comparable.

**Do not terminate** the run just because the current value beats the
Arm A reference. Only pick `terminate` when one of these is true:

  - The curriculum is at the full primary *and* secondary range, the
    last ~5 locked_val measurements are within 3% of each other, AND
    you have tried at least 3 distinct hyperparameter / architecture
    variations in recent cycles without meaningful improvement. (Real
    plateau with meaningful effort spent, not a first-pass hit.)
  - You have genuinely run out of ideas — list them in your `reason`
    field and demonstrate that each has been tried.

Otherwise: keep iterating. You have full control of the state machine,
the training hyperparameters, the curriculum, and the data collection.
Use it. Try lr schedules, architecture bumps, wider explore ranges,
different batch sizes, longer training, scene changes that expose
novel poses, etc. The whole point of this loop is to let you drive
the model past the baseline, not to stop at parity.
"""


_GOAL_BLOCK_LIFELONG = """\
## Your goal (lifelong mode)

This is a continuous lifelong-learning loop. There is **no fixed
target metric to minimize** and **no automatic termination**. The
operator stops the process when they're done with it; until then your
job is to keep the model continuously useful as the physical scene
evolves.

```
mode:    lifelong
tau_low:  {tau_low}    (rolling VERIFY mean_err target — keep below)
tau_high: {tau_high}   (rolling VERIFY mean_err alarm — act when above)
```

The signals you optimize against come from the model's own collected
data, NOT from a held-out reference set:

  - `recent_verifies[*].mean_err` — rolling MSE on freshly-captured
    probes. PRIMARY quality signal. Keep it below `tau_low`.
  - `wide_verify.history[*]` — held-out CROSS-SCENE generalization
    corpus. Each entry records a wide-verify burst (probes spanning
    multiple operator-arranged scenes). The trainer evaluates each
    candidate checkpoint against the union of these bursts and
    reports `wide_verify_mse` on the retrain — that's the lifelong
    val_guard signal and the strongest "does the model generalize?"
    measurement. Drive `wide_verify.corpus_size` up by occasionally
    setting `wide_verify_next: true` in your decision (see routing).
  - `last_training_curve.best_val` — the trainer's internal val_loss
    on a held-out fraction of the running canvas dataset. SECONDARY
    convergence signal for each retrain.
  - `pending_novelty_report` — whether the most recent EXPLORE batch
    is actually distinct from prior data. Drives explore vs idle vs
    retrain decisions.
  - `plateau_signal.verdict` — structured plateau diagnostic (see
    "Plateau handling" below).
  - `auxiliary.locked_val_history` — held-out locked-val MSE,
    AUXILIARY context only. Use as a tiebreaker when interpreting
    plateaus; do NOT frame your decisions around minimizing it.

`terminate` is essentially never the right routing choice in this
mode. The operator owns lifecycle. Pick it only if the loop is
genuinely impossible to continue (irrecoverable hardware fault, no
viable hyperparameter ever produces a non-degenerate result).

## Plateau handling

`plateau_signal.verdict` is one of:

  - `improving` — VERIFY mean_err is still moving. Keep doing what's
    working. Default to `verify` or continue the current trajectory.
  - `plateau_low_locked_val` — VERIFY has flatlined AND the auxiliary
    locked-val is low. The model has learned the current scene well.
    Routing: `verify` (idle observation) is fine. Don't churn.
  - `plateau_high_locked_val` — VERIFY has flatlined but locked-val
    is high. The model is stuck on its current data. Routing:
    `explore` with widened ranges or a different joint bias, OR
    `idle` for a scene change. Use `pending_novelty_report` to
    choose: low novelty -> `idle`; novelty still useful -> `explore`.
    ALSO: set `wide_verify_next: true` on your next `verify` so the
    next quality check is across multiple scenes — this is when you
    most need a generalization measurement.
  - `stuck` — both signals indicate plateau without locked-val to
    disambiguate. Same response as `plateau_high_locked_val`: route
    to `explore` with novelty in mind, or `idle` for scene change,
    AND set `wide_verify_next: true`.
  - `insufficient_data` — too early to call. Default routing applies.
"""


_THINK_PROMPT_TEMPLATE = """\
The autonomous robot-learning orchestrator has suspended in its THINK
state and needs your decision right now. You are not in a conversation
— this is a one-shot request. Return a single JSON object that matches
the schema below. Do not ask clarifying questions; make your best
judgment from the snapshot.

{goal_block}
## System constraints (hardware)

This learner runs on a single NVIDIA RTX 5090 with **32 GB of VRAM**
(~32607 MB reported by nvidia-smi). The world-model training
subprocess and the probe / canvas pipelines share that budget.

**The previous run saturated GPU memory at ~32 GB.** Stay well under
the cap. Leave headroom for activations, gradients, and checkpoint
loading spikes. Target < 28 GB peak; hard ceiling is 32 GB.

Important — how VRAM failure actually manifests here: PyTorch's
allocator is now capped at 95% of total VRAM, so most over-allocation
will surface as a clean `CUDA out of memory` exception (see
`recent_failures` for the crashed-subprocess details). Some failure
paths bypass the allocator (bnb 8-bit kernels, CUDA context growth);
the orchestrator catches those three additional ways:

- Memory pressure abort. A background thread polls `nvidia-smi`
  every 5 s. Sustained usage above ~96% of total VRAM kills the
  training subprocess and routes back to you with a
  `training_memory_abort` entry in `recent_gpu_signals`.
- Throughput-collapse abort. After an 8-epoch warmup baseline, if
  the median of the last 3 epochs takes ≥5x the baseline (the
  signature of CUDA spilling to host-shared memory), the subprocess
  is killed and you see `training_throughput_collapse` in
  `recent_gpu_signals` with `recent_epoch_s` / `baseline_epoch_s` /
  `ratio` fields.
- Stall abort. If no `training_progress` event arrives for 30
  minutes, the subprocess is killed and you see `training_stalled`
  in `recent_gpu_signals`.

Knobs that grow VRAM, in order of impact:

- `training.batch_size` - roughly linear. Biggest single lever.
- `training.embed_dim`, `training.depth`, `training.num_heads`,
  `training.patch_size` - architecture; also require
  `from_scratch: true`.
- `cadence.cold_start_epochs`, `cadence.ft_epochs` - don't grow peak
  memory directly but extend exposure to transient spikes.

If your last cycle produced a `training_memory_abort`,
`training_throughput_collapse`, or `training_stalled` event, scale
down before retrying. Do not repeat the configuration that just
aborted.

## How you influence the loop

Every field you return is applied in place before the next state runs.
Your reply MUST be a single JSON object. It must contain a `next_state`
field (one of verify, explore, retrain, idle, terminate). All other
fields are optional.

### Available routing

- `verify`   — run a VERIFY burst next. Use this when you want a fresh
               mean_err measurement after changing knobs.
- `explore`  — collect a fresh EXPLORE burst. Use this when the model is
               data-starved or when you've changed the curriculum range.
- `retrain`  — retrain on the EXISTING accumulated data, no new episodes.
               Uses the `training_overrides` you supply. Cap:
               `advisor_budget.claude_max_consecutive_retrains`.
- `idle`     — pause the learner and ask the human to physically rearrange
               the scene. REQUIRES `scene_change_description`. The human
               will hit "Scene ready" on the dashboard and control returns
               to you. **For this experiment, `idle` is the ONLY way the
               physical scene ever changes** — there is no random or
               periodic perturbation. If you never request `idle`, the
               objects will sit in the exact same positions for the
               entire run, and EXPLORE will only ever vary motor
               positions over an unchanging scene. Use `idle` proactively
               when the data is starting to look redundant. Be specific
               in `scene_change_description` ("move the red block ~5cm
               to the right and rotate the cup 90°") so the operator
               knows exactly what to do; vague asks ("change something")
               waste the human's attention.
- `terminate`— stop the run. Use ONLY when you've genuinely exhausted
               your options per the "Your goal" section. In lifelong
               mode, `terminate` is essentially never the right choice
               (the operator owns lifecycle); in legacy mode, beating
               the Arm A reference alone is not sufficient.

In addition to `next_state`, you can set `wide_verify_next: true` in
your decision (any routing choice). When set, the orchestrator
promotes the NEXT VERIFY pass to a multi-scene "wide" verify: probes
are split into chunks across operator-arranged scenes, and each
chunk's recordings are added to the held-out wide-verify corpus the
trainer evaluates against.

**Wide-verify MEASURES the generalization gap; it does NOT close it.**
Every canvas built from a wide-verify burst is permanently held-out
from training (registry enforces a hard disjointness invariant
between `wide_verify_canvas_dirs` and `accumulated_canvas_dirs`). So
when the operator rearranges the scene for a wide-verify chunk, that
scene's data goes to the eval corpus and the model never trains on
it. To actually CLOSE a scene-overfit gap you must route
`idle → explore → retrain` so the new scene feeds the training
corpus. Reserve `wide_verify_next` for AFTER such a cycle, to verify
the gap shrunk — not as the trigger that exposes the model to the
new scene.

Use it to grow `wide_verify.corpus_size` when you need a fresh
generalization snapshot — typically AFTER a retrain that incorporated
new-scene data, or as a periodic check during a plateau. Wide-verify
is operator-intensive (one scene-change prompt per chunk), so don't
fire it every cycle — pace it with `cadence.wide_verify_every` or
discrete advisor decisions.

### Optional: prompt-improvement suggestions

After reviewing the latest cycle's signals — especially after a
RETRAIN whose result you can now compare against your prior
prediction — pause for one beat and ask: **did the rules in this
prompt help me make the right call, or did I have to reason around
them?** If something specific in the prompt led you astray (or could
be sharpened so the next advisor invocation handles a similar
situation better), put it in `prompt_improvement_suggestion` as a
short, concrete suggestion. The operator collects these in the event
log and applies prompt edits offline.

Useful suggestions look like:
  - *"Trigger B (i) uses `VERIFY/best_val > 5×` for scene-overfit, but
    in this run VERIFY is same-static-scene so the ratio understates
    cross-placement gap. When `wide_verify` is empty, also check
    `auxiliary.locked_val_history[-1] / best_val > 4×`."*
  - *"The plateau `threshold: 0.95` (5% over 3 cycles) is too tight
    for the noise in this experiment's VERIFY signal — saw 4
    consecutive cycles within 8% but no plateau verdict. Consider
    widening to 0.90 or making it a per-config knob."*
  - *"The fine-tune-OOM failure mode is described as the symptom
    only; the prompt should also note that train_diffusion now caps
    PyTorch's allocator at 0.95, so a CUDA-OOM crash on `--fine-tune`
    means the model literally doesn't fit, not that it almost fits."*

Skip the field when nothing substantive comes to mind. Don't pad it
with generic "the prompt is fine" boilerplate — empty / omitted is
the right answer most cycles. One concrete edit per cycle, max.

Stay in your lane: this field is a suggestion, not an action. Don't
try to "patch" the prompt by changing your reasoning style this
cycle to compensate. Make the call the current prompt asks you to
make, then note the suggestion separately. The operator will decide
whether to apply it.

### Available overrides

All override objects are optional dicts. Unknown keys are dropped. Values
are clamped to sane minima.

```json
{{
  "next_state": "...",
  "reason": "short string for the event log",
  "scene_change_description": "only when next_state=idle",
  "wide_verify_next": false,
  "runtime_overrides": {{
    "tau_low": 0.04, "tau_high": 0.08, "val_guard": 1.3,
    "probes_per_verify": 8, "base_burst": 30, "max_sub_bursts": 3
  }},
  "training_overrides": {{
    "training.lr": 1e-4, "training.warmup_epochs": 10,
    "cadence.ft_epochs": 200, "cadence.early_stop_patience": 50
  }},
  "curriculum_overrides": {{
    "primary.active": [-60, 60], "primary.stable_cycles": 0,
    "secondary.active": [55, 85], "secondary.pinned_half_width": 5,
    "force_stage_transition": false
  }},
  "explore_overrides": {{
    "num_episodes": 40, "max_sub_bursts": 1, "randomize_primary_start": true
  }},
  "from_scratch": false,
  "prompt_improvement_suggestion": "(optional, omit unless concrete)"
}}
```

`from_scratch` is required when you change architecture fields
(`embed_dim`, `depth`, `num_heads`, `patch_size`) because fine-tune
weights won't load into the new shape. It forces the next retrain to
cold-start on the accumulated data.

**`patch_size` constraint:** the canvas is 464×480 px. A patch_size only
works if it divides BOTH dimensions. Valid values are **{{8, 16}}** (and
toy {{1, 2, 4}} — too small to be useful). `patch_size: 32` is geometrically
invalid (464 / 32 = 14.5) and will be rejected by `apply_cfg_overrides`
with a `claude_override_rejected` event in your next THINK context. Don't
keep proposing it.

## Scene-change budget (human-in-the-loop)

The physical scene only changes when YOU request it via `next_state:
"idle"` — there is no automatic perturbation. The operator reads your
`scene_change_description` from the dashboard, moves objects to match,
then clicks "Scene ready" to release the IDLE gate.

`scene_change_history` in the snapshot below tells you:
  - `total_requested`: how many times you have already asked for a scene
    change in this run.
  - `total_acknowledged`: how many of those the operator has completed.
  - `pending`: True iff the latest request hasn't been acked yet (you
    should never re-issue `idle` while another is pending — but this
    branch of the loop is unreachable when pending=True since the
    orchestrator is still in IDLE waiting).
  - `recent`: descriptions + timestamps of the last 10 requests, so you
    can avoid asking for the same change twice in a row and pace the
    operator reasonably.

**When to spend a scene-change request:**

  Trigger A — DATA REDUNDANCY (joint space exhausted vs current scene):
    - Diagnosed `data-bound` AND most recent EXPLORE batch's
      `pending_novelty_report.mean_frame_mse_vs_prior_latest` is < 1e-3.
      Collected data looks identical to prior data — the joint space has
      been thoroughly explored against the current scene; more episodes
      against the same scene cannot help.

  Trigger B — SCENE-OVERFIT (model memorizing the current scene; this is
  the most common reason to request a scene change in this loop, and the
  one to act on FAST):
    Look at the gap between the trainer's internal validation
    (`last_training_curve.best_val`) and the on-fresh-probes signal
    (`recent_verifies[-1].mean_err`). When the model is generalizing,
    these track each other within 2-3×. When the model is memorizing
    the scene, internal best_val keeps dropping while VERIFY mean_err
    plateaus or REGRESSES.

    Concrete trigger — request `idle` (with a specific
    `scene_change_description`) when ANY of these hold:
      (i) `recent_verifies[-1].mean_err / last_training_curve.best_val`
          ratio exceeds 5×. This is a memorization signal regardless
          of which is moving — the model fits its train distribution
          much better than it predicts fresh observations.
      (ii) `best_val` IMPROVED in the last retrain (`-` delta) while
           VERIFY mean_err REGRESSED across the same retrain
           (positive delta vs the prior cycle's verify). The model got
           better at the data it has seen and worse at fresh data —
           textbook scene-overfit.
      (iii) Two or more consecutive retrains where `best_val` set new
            lows but VERIFY didn't reach a new low. Scene diversity is
            the binding constraint; more compute on the same data
            won't fix it.

    NOTE: Trigger B fires REGARDLESS of `pending_novelty_report`
    novelty. High novelty means the explore sequencer wandered to new
    MOTOR POSES, not new SCENES. A scene-overfit model needs new
    scenes, not new poses against the current scene.

  Trigger C — PLATEAU:
    - Your primary quality signal has plateaued for 3+ cycles AND
      additional EXPLORE has stopped lowering it.
    - `plateau_signal.verdict` is `plateau_high_locked_val` or `stuck`.

**When NOT to ask:**
  - Diagnosis is compute-bound or capacity-bound — change knobs or
    architecture instead. (But: if you've tried 2+ hyperparameter
    variations and the generalization gap from Trigger B is still
    present, the diagnosis is wrong — it's scene-bound, not compute-
    or capacity-bound.)
  - You requested a scene change in the immediately previous cycle.
    Each scene change should be paired with at least one EXPLORE +
    RETRAIN cycle so you can MEASURE the effect on the generalization
    gap before requesting another.

**Default action when Trigger B fires:** `next_state: "idle"` with
`scene_change_description` describing a concrete physical perturbation
("move red block ~5cm to the right and rotate the cup 90°"). The
post-IDLE sequence the orchestrator should run is `idle → explore →
retrain` so the freshly-rearranged scene flows through EXPLORE into
the training corpus. After IDLE acks, your next THINK call should
route `explore` (against the new scene), then `retrain` once that
batch lands.

**Do NOT pair this idle with `wide_verify_next: true`.** Wide-verify
diverts the next VERIFY pass into multi-scene held-out probes, which
means (a) the operator rearranges the scene 3 more times for chunks
that are permanently locked out of training, and (b) the new scene
from this idle never gets an EXPLORE burst against it. That leaves
the scene-overfit gap untouched while burning operator attention. If
you want a generalization measurement, fire `wide_verify_next: true`
on the cycle AFTER the explore-then-retrain completes — that way it
measures the impact of the scene change instead of replacing it.

## Recent action canvases (images)

Before you decide, **read the most recent action canvas images** with
the Read tool. Each canvas is a training-format PNG showing
`[before | action_sep | ACTUAL | gray_sep | INFERRED]` with motor strips
underneath each frame. The ACTUAL frame is what the robot actually did;
the INFERRED frame is what the world model predicted — the visual gap
between them is the real error signal you should reason about.

Paths, newest first:

{canvas_paths}

Look at the actual-vs-inferred gap for hold actions (red separator),
move actions (green = positive, blue = negative). Are there specific
joint positions or action types where the prediction is wildly off?
That tells you where to focus the next EXPLORE, retrain, or tau tweak.

## Current state snapshot

```json
{context_json}
```

## What to think about

1. **Diagnose the binding constraint before choosing a lever.** In one
   sentence each, state whether *capacity*, *compute*, or *data* is the
   primary limit right now, and cite specific evidence from the snapshot:

   - **Capacity-bound** signals: multiple from-scratch retrains on the
     same architecture yielding shrinking gains; train_loss plateau while
     val_loss is still dropping (underfit); same recipe tried 3+ times
     without breakthrough. -> Lever: bump `training.depth` /
     `training.embed_dim` / `training.num_heads` with `from_scratch: true`.
   - **Compute-bound** signals: last training curve's `best_val` was still
     improving at the epoch cutoff; `--early-stop-patience` never
     triggered; the LR schedule ran out of budget before converging. ->
     Lever: more `cadence.cold_start_epochs` / `cadence.ft_epochs`, or
     change `training.lr_schedule`.
   - **Data-bound** signals: growing train/val gap (overfitting); your
     primary quality signal regresses after retraining on unchanged data
     (in lifelong mode this is `recent_verifies[*].mean_err`; in legacy
     mode `current_locked_val_mse`); per-joint breakdown shows one joint
     saturated while another is still learning; recent scene perturbations
     invalidate prior episodes.

     Data-bound has TWO sub-flavors that pick different levers — diagnose
     which one before routing:

       * POSE-bound: `last_training_curve.best_val` and
         `recent_verifies[-1].mean_err` track each other (within 2-3×),
         and both are still moving. The model has the *scene* covered
         but is short on motor-pose coverage. -> Lever: `explore` (new
         episodes against the same scene). Optionally widen joint
         ranges or rebalance toward an under-sampled joint.

       * SCENE-bound (overfit-to-scene): `best_val` keeps dropping
         while VERIFY mean_err plateaus or regresses; their ratio
         exceeds 5×; or `best_val` set new lows but VERIFY didn't.
         The model has memorized the current scene and pose diversity
         alone won't help. -> Lever: `idle` with a specific
         `scene_change_description` AND `wide_verify_next: true`. Do
         NOT route to `explore` — more poses against the same scene
         entrench the memorization further.

     The default for "data-bound" used to be `explore`. With wide-verify
     in the loop, you now have a sharper diagnostic; use it. If the gap
     is >5×, the right answer is almost always `idle`.

   **You MUST name one of the three as the primary constraint in your
   `reason` field** (use the literal tokens "capacity-bound",
   "compute-bound", or "data-bound" somewhere in `reason`).

   **Persistence rule (soft):** there is NO hard limit on how many
   consecutive cycles can name the same constraint. Sustained EXPLORE
   campaigns are correct behavior when the model is genuinely
   data-starved — keep exploring as long as each new batch is delivering
   real gains (improvement in your primary quality signal, novelty MSE
   > threshold, advisor action canvases still showing distribution-shift
   artifacts). Only pivot to a different axis when **all** of these hold
   simultaneously: (a) the same constraint has been named for 5+
   consecutive cycles, (b) the last 3 of those cycles each delivered
   <10% of the prior cycle's improvement on the primary signal, AND
   (c) the most-recent EXPLORE batch showed low novelty (MSE near or
   below the existing threshold), indicating new data isn't actually
   different from old. If gains are accelerating or the novelty signal
   is high, stay on the diagnosis.

2. Look at your **primary quality signal**:
   - In lifelong mode: `recent_verifies[*].mean_err`. Is it trending
     toward / below `tau_low`? Is it spiking above `tau_high`? Cross-
     reference with `plateau_signal.verdict` — that's the structured
     diagnostic that drives "stay" vs "act" decisions.
   - In legacy mode: compare `current_locked_val_mse` to
     `arm_a_locked_val_mse`. Is the gap closing? Has it stalled?
3. Look at `last_training_curve` (`train_loss`, `val_loss`, `best_val`).
   Did the last training run overfit, underfit, or converge cleanly?
4. Look at `recent_verifies`. Is `mean_err` moving in the right direction
   relative to `knobs.tau_low` and `knobs.tau_high`?
5. Look at `curriculum`. Are we stuck at a narrow range? Should we force-
   expand, force a stage transition, or narrow back?
6. Look at `recent_gpu_signals`. If it contains a `training_memory_abort`
   or `training_stalled` event from your last cycle, the orchestrator
   aborted because the configuration exceeded or nearly exceeded 32 GB
   VRAM. Check `summary.used_mb` / `summary.total_mb` to see the peak.
   Reduce `training.batch_size` first, or shrink architecture dims (with
   `from_scratch: true`) before retrying. If there are recurring
   `inference_oom` entries, verify is also memory-bound.
7. Look at `recent_advisor_decisions` — this is YOUR own history across
   prior THINK cycles (reason, next_state, every override you applied).
   Before proposing the same override again, check whether you already
   tried it: if a recent `training_overrides` bump didn't lower val_loss
   or caused an abort, don't repeat it. Look for patterns you're stuck
   in (alternating explore/retrain with no improvement) and break them
   with a genuinely different approach.
8. Your diagnosis from step 1 constrains the routing choice: capacity-bound
   -> `retrain` with a `from_scratch: true` architecture override; compute-
   bound -> `retrain` with longer epochs or different LR schedule; data-
   bound -> `explore` or `idle` for a scene perturbation. Don't pick
   `explore` if you named capacity as binding.
9. Respect `advisor_budget.claude_max_consecutive_retrains` — after that
   many retrains in a row without new data, the orchestrator will force
   a VERIFY anyway.
10. **If `pending_novelty_report` is present**, you are being called
    immediately after an EXPLORE burst. Its fields tell you whether the
    just-collected batch is actually different from prior data:

    - `mean_frame_mse_vs_prior_latest` — scalar in [0, 1]. Very small
      (<1e-3) means the new batch's average canvas looks nearly
      identical to the most recent prior batch's average canvas
      (redundant scene, similar poses). Larger values (>5e-3) indicate
      a real shift — novel poses, scene rearrangement, or lighting
      change. Use this as the PRIMARY cheap signal.
    - `new_frame_stats` / `prior_frame_stats` — brightness + stddev of
      each mean frame. Large mean shift => lighting changed; large
      stddev shift => scene complexity changed.
    - `sample_canvas_paths` — (tag, path) pairs you can `Read` with
      your vision model to confirm visually. One from the new batch,
      one from the nearest prior batch. If the scalar is ambiguous,
      compare them visually before deciding.

    What to do with it:
    - **High novelty** (MSE >5e-3 or clear visual difference): retrain
      is worthwhile. Set `training_overrides` + `from_scratch` based on
      your capacity/compute/data diagnosis (step 1) and proceed. Note —
      a sudden novelty spike usually corresponds to a recent
      `scene_change_history.recent` entry; cross-check timestamps so
      you don't double-count the same scene shift.
    - **Low novelty** (MSE <1e-3, frames look the same): this batch is
      redundant. Consider routing `explore` again with different ranges
      / joint biases to cover new state-space, OR `idle` to request a
      scene perturbation. Retraining on redundant data wastes compute.
    - **Mid novelty**: retrain but keep the override conservative
      (fine-tune, not from-scratch) — save the nuclear option for when
      you have genuinely new data to feed it.

    **Your routing decision after an EXPLORE must predict the
    subsequent retrain's parameters** — the orchestrator transitions
    EXPLORE -> THINK -> RETRAIN without re-entering THINK. So if you
    want a cold-start retrain on the new data, set both `next_state:
    "retrain"` AND `from_scratch: true` AND any `training_overrides`
    you want. If you want to skip the retrain, route to `explore` or
    `idle` instead.

Default next state (what the orchestrator would do if you returned
`{{"next_state":"{default_next}"}}`): **{default_next}**.

Respond now with a single JSON object containing your decision. No
prose, no greeting, no follow-up questions. The schema constraint
will reject anything that isn't valid JSON matching the allowed fields.
"""


def build_think_prompt(context: dict) -> str:
    """Render the prompt for `claude -p`.

    Picks `_GOAL_BLOCK_LIFELONG` when `context["goal"]["mode"] ==
    "lifelong"`, otherwise the legacy locked-val-driven goal block.
    """
    paths = context.get("recent_action_canvas_paths") or []
    if paths:
        canvas_paths = "\n".join(f"- `{p}`" for p in paths)
    else:
        canvas_paths = "- (none yet — first VERIFY/EXPLORE hasn't run)"
    goal = context.get("goal", {}) or {}
    if goal.get("mode") == "lifelong":
        goal_block = _GOAL_BLOCK_LIFELONG.format(
            tau_low=goal.get("tau_low"),
            tau_high=goal.get("tau_high"),
        )
    else:
        goal_block = _GOAL_BLOCK_LEGACY.format(
            arm_a=goal.get("arm_a_locked_val_mse"),
            current=goal.get("current_locked_val_mse"),
        )
    return _THINK_PROMPT_TEMPLATE.format(
        goal_block=goal_block,
        default_next=context.get("default_next_state", "verify"),
        canvas_paths=canvas_paths,
        context_json=json.dumps(context, indent=2, default=str),
    )


# ------------------------------------------------------------- appliers


_KNOWN_TRAINING_FIELDS = {
    "patch_size", "embed_dim", "depth", "num_heads",
    "num_train_timesteps", "beta_schedule", "prediction_type",
    "lr", "weight_decay", "lr_schedule", "warmup_epochs", "min_lr",
    "grad_clip", "batch_size", "seed", "val_ratio",
    # Mixed-precision + memory-fit mechanics — tunable too, paired with
    # `from_scratch: true` whenever the change would alter the on-disk
    # state_dict shape.
    "bf16", "gradient_checkpointing", "use_8bit_adam",
    "gradient_accumulation_steps",
}

# Training fields whose values change the on-disk state_dict shape, so a
# `--fine-tune <prior_ckpt>` is impossible after they're modified. The
# orchestrator consults this set to auto-promote `from_scratch=True` when
# the advisor changed one of these without saying so itself — preventing
# the otherwise-guaranteed CalledProcessError on checkpoint load.
SHAPE_CHANGING_TRAINING_FIELDS = frozenset({
    "patch_size", "embed_dim", "depth", "num_heads",
})

# Canvas geometry — fixed for this project. Both `frame_size` (cameras
# stacked vertically with motor strips) and the patch_size constraint
# derive from these. Any patch_size override must divide BOTH dims.
_CANVAS_H = 464
_CANVAS_W = 480
_VALID_PATCH_SIZES: tuple[int, ...] = tuple(
    p for p in (1, 2, 4, 8, 16, 32, 64)
    if _CANVAS_H % p == 0 and _CANVAS_W % p == 0
)  # → (1, 2, 4, 8, 16) — 32 fails because 464 % 32 == 16

_KNOWN_CADENCE_FIELDS = {
    "cold_start_epochs", "ft_epochs", "early_stop_patience",
}


def apply_cfg_overrides(
    cfg, overrides: dict, event_log=None,
) -> dict:
    """Apply dotted-path overrides to cfg.training.* and cfg.cadence.*.

    Unknown keys are logged + skipped. Numeric values ≤ 0 on
    positive-only fields are clamped. Returns the actually-applied dict.

    Architectural fields (`depth`, `embed_dim`, `num_heads`, `patch_size`,
    `batch_size`, mixed-precision mechanics, ...) are no longer blocked
    here. The advisor is trusted to pair any state_dict-shape-changing
    override with `from_scratch: true` so the next training subprocess
    rebuilds the model fresh instead of loading a now-mismatched checkpoint.
    """
    if not overrides:
        return {}
    applied: dict = {}
    for raw_key, raw_value in overrides.items():
        key = str(raw_key)
        if key.startswith("training."):
            field = key[len("training."):]
            section = "training"
            allowed = _KNOWN_TRAINING_FIELDS
        elif key.startswith("cadence."):
            field = key[len("cadence."):]
            section = "cadence"
            allowed = _KNOWN_CADENCE_FIELDS
        else:
            if event_log is not None:
                event_log.log(
                    "claude_override_unknown", target="cfg", key=key,
                )
            continue
        if field not in allowed:
            if event_log is not None:
                event_log.log(
                    "claude_override_unknown", target=section, key=field,
                )
            continue
        ns = getattr(cfg, section, None)
        if ns is None:
            continue
        current = getattr(ns, field, None)
        try:
            if isinstance(current, bool):
                value: Any = bool(raw_value)
            elif isinstance(current, int):
                value = int(raw_value)
            elif isinstance(current, float):
                value = float(raw_value)
            else:
                value = raw_value
        except (TypeError, ValueError):
            if event_log is not None:
                event_log.log(
                    "claude_override_uncoercible",
                    target=section, key=field, value=raw_value,
                )
            continue
        # Positive-only clamps.
        if field in (
            "lr", "weight_decay", "min_lr", "warmup_epochs",
            "grad_clip", "batch_size", "val_ratio", "patch_size",
            "embed_dim", "depth", "num_heads", "num_train_timesteps",
            "cold_start_epochs", "ft_epochs", "early_stop_patience",
        ):
            if isinstance(value, (int, float)) and value <= 0:
                if event_log is not None:
                    event_log.log(
                        "claude_override_clamped",
                        target=section, key=field,
                        requested=value, clamped_to=1e-9,
                    )
                value = 1e-9 if isinstance(value, float) else 1
        # Geometric validity: patch_size must divide both canvas
        # dimensions (currently 464 high x 480 wide, GCD=16). The
        # advisor has repeatedly picked patch_size=32 (which doesn't
        # divide 464) despite seeing the resulting RuntimeError in
        # event history; reject the override outright rather than let
        # it crash the training subprocess.
        if section == "training" and field == "patch_size":
            if not isinstance(value, int) or value <= 0 or (
                _CANVAS_H % value != 0 or _CANVAS_W % value != 0
            ):
                if event_log is not None:
                    event_log.log(
                        "claude_override_rejected",
                        target=section, key=field,
                        requested=value,
                        reason="patch_size_must_divide_canvas",
                        canvas_h=_CANVAS_H, canvas_w=_CANVAS_W,
                        valid=_VALID_PATCH_SIZES,
                    )
                continue
        setattr(ns, field, value)
        applied[key] = value
    return applied


def apply_curriculum_overrides(
    curriculum, overrides: dict, event_log=None,
) -> dict:
    """Mutate a CurriculumState in place.

    Supported keys: `primary.active`, `primary.stable_cycles`,
    `secondary.active`, `secondary.stable_cycles`, `secondary.pinned_half_width`,
    `force_stage_transition`. Returns the applied dict.
    """
    if not overrides or curriculum is None:
        return {}
    applied: dict = {}
    for raw_key, raw_value in overrides.items():
        key = str(raw_key)
        try:
            if key == "primary.active":
                lo, hi = float(raw_value[0]), float(raw_value[1])
                lo = max(curriculum.primary.full_min, lo)
                hi = min(curriculum.primary.full_max, hi)
                if hi > lo:
                    curriculum.primary.active = (lo, hi)
                    applied[key] = [lo, hi]
            elif key == "primary.stable_cycles":
                curriculum.primary.stable_cycles = max(0, int(raw_value))
                applied[key] = curriculum.primary.stable_cycles
            elif key == "secondary.active":
                if curriculum.secondary is None:
                    continue
                lo, hi = float(raw_value[0]), float(raw_value[1])
                lo = max(curriculum.secondary.full_min, lo)
                hi = min(curriculum.secondary.full_max, hi)
                if hi > lo:
                    curriculum.secondary.active = (lo, hi)
                    applied[key] = [lo, hi]
            elif key == "secondary.stable_cycles":
                if curriculum.secondary is None:
                    continue
                curriculum.secondary.stable_cycles = max(0, int(raw_value))
                applied[key] = curriculum.secondary.stable_cycles
            elif key == "secondary.pinned_half_width":
                hw = max(0.0, float(raw_value))
                curriculum.secondary_pinned_half_width = hw
                applied[key] = hw
            elif key == "force_stage_transition":
                if bool(raw_value) and curriculum.stage == curriculum.STAGE_PRIMARY:
                    curriculum.transition_to_secondary()
                    applied[key] = True
            else:
                if event_log is not None:
                    event_log.log(
                        "claude_override_unknown",
                        target="curriculum", key=key,
                    )
        except (TypeError, ValueError, IndexError) as e:
            if event_log is not None:
                event_log.log(
                    "claude_override_uncoercible",
                    target="curriculum", key=key, value=raw_value,
                    error=str(e),
                )
            continue
    return applied


# ---------------------------------------------------- next-state resolver


def resolve_next_state(
    requested: str,
    default: str,
    consecutive_retrains_without_data: int,
    cap: int,
    *,
    has_scene_description: bool,
    event_log=None,
) -> str:
    """Translate Claude's `next_state` string into one of the canonical
    tokens the orchestrator dispatches on, applying the retrain cap and
    the scene-description requirement. Returns 'verify', 'explore',
    'retrain', 'idle', or 'terminate'. Unknown values fall back to
    `default`.
    """
    canonical = str(requested or default).lower().strip()
    if canonical not in ("verify", "explore", "retrain", "idle", "terminate"):
        if event_log is not None:
            event_log.log(
                "claude_next_state_unknown",
                requested=requested, default=default,
            )
        canonical = str(default).lower().strip()

    if canonical == "retrain" and consecutive_retrains_without_data >= cap:
        if event_log is not None:
            event_log.log(
                "claude_retrain_cap_hit",
                count=consecutive_retrains_without_data, cap=cap,
            )
        canonical = "verify"

    if canonical == "idle" and not has_scene_description:
        if event_log is not None:
            event_log.log(
                "claude_idle_missing_description",
                default=default,
            )
        canonical = str(default).lower().strip()

    return canonical
