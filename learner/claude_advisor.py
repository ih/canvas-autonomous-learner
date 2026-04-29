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
    "subprocess_timeout",
    "retrain_memory_abort_routed_to_think",
    "retrain_stalled_routed_to_think",
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
    consumes. Pure function — no side effects."""
    runs_dir = Path(cfg.paths.runs_dir)
    events = _read_recent_events(events_path, n=600)
    locked_history = registry.locked_val_history()
    recent_canvases = _recent_action_canvas_paths(runs_dir, n=5)
    last_locked_val = None
    for entry in reversed(locked_history):
        if entry.get("accepted") and entry.get("locked_val_mse") is not None:
            last_locked_val = entry["locked_val_mse"]
            break
    return {
        "goal": {
            "arm_a_locked_val_mse": _arm_a_target(runs_dir),
            "current_locked_val_mse": last_locked_val,
        },
        "cycle": len(locked_history),
        "episodes_collected": registry.episodes_collected(),
        "accumulated_canvas_dirs": len(registry.accumulated_canvas_dirs()),
        "consecutive_guard_rejections": registry.consecutive_guard_rejections(),
        "experiment_status": registry.experiment_status(),
        "locked_val_history": locked_history,
        "curriculum": _curriculum_snapshot(curriculum),
        "knobs": knobs.as_dict(),
        "training_cfg": _training_cfg_snapshot(cfg),
        "cadence_cfg": _cadence_snapshot(cfg),
        "last_training_curve": _extract_last_training_curve(events),
        "recent_verifies": _extract_recent_verifies(events),
        "recent_advisor_decisions": _extract_recent_advisor_decisions(events, n=10),
        "recent_gpu_signals": _extract_recent_gpu_signals(events, n=20),
        "default_next_state": default_next_state,
        "advisor_budget": {
            "consecutive_retrains_without_data": consecutive_retrains_without_data,
            "claude_max_consecutive_retrains": claude_max_consecutive_retrains,
        },
        "last_scene_change": last_scene_change,
        "pending_explore_overrides": pending_explore_overrides or {},
        "pending_novelty_report": pending_novelty_report or None,
        "recent_action_canvas_paths": recent_canvases,
    }


# ------------------------------------------------------------- prompt


_THINK_PROMPT_TEMPLATE = """\
The autonomous robot-learning orchestrator has suspended in its THINK
state and needs your decision right now. You are not in a conversation
— this is a one-shot request. Return a single JSON object that matches
the schema below. Do not ask clarifying questions; make your best
judgment from the snapshot.

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

## System constraints (hardware)

This learner runs on a single NVIDIA RTX 5090 with **32 GB of VRAM**
(~32607 MB reported by nvidia-smi). The world-model training
subprocess and the probe / canvas pipelines share that budget.

**The previous run saturated GPU memory at ~32 GB.** Stay well under
the cap. Leave headroom for activations, gradients, and checkpoint
loading spikes. Target < 28 GB peak; hard ceiling is 32 GB.

Important — how VRAM failure actually manifests here: PyTorch does
NOT reliably raise a CUDA OOM exception on this machine. Instead,
once VRAM fills, CUDA spills to host-backed / shared memory and
training throughput collapses. Epochs slow by 10-100x, there is no
crash, the process appears stuck. The orchestrator now detects this
two ways:

- Memory pressure abort. A background thread polls `nvidia-smi`
  every 5 s. Sustained usage above ~93% of total VRAM kills the
  training subprocess and routes back to you with a
  `training_memory_abort` entry in `recent_gpu_signals`.
- Stall abort. If no `training_progress` event arrives for 10
  minutes, the subprocess is killed and you see `training_stalled`
  in `recent_gpu_signals`.

Knobs that grow VRAM, in order of impact:

- `training.batch_size` - roughly linear. Biggest single lever.
- `training.embed_dim`, `training.depth`, `training.num_heads`,
  `training.patch_size` - architecture; also require
  `from_scratch: true`.
- `cadence.cold_start_epochs`, `cadence.ft_epochs` - don't grow peak
  memory directly but extend exposure to transient spikes.

If your last cycle produced a `training_memory_abort` or
`training_stalled` event, scale down before retrying. Do not repeat
the configuration that just aborted.

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
               to you.
- `terminate`— stop the run. Use ONLY when you've genuinely exhausted
               your options per the "Your goal" section. Beating the
               Arm A reference is not sufficient on its own.

### Available overrides

All override objects are optional dicts. Unknown keys are dropped. Values
are clamped to sane minima.

```json
{{
  "next_state": "...",
  "reason": "short string for the event log",
  "scene_change_description": "only when next_state=idle",
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
  "from_scratch": false
}}
```

`from_scratch` is required when you change architecture fields
(`embed_dim`, `depth`, `num_heads`, `patch_size`) because fine-tune
weights won't load into the new shape. It forces the next retrain to
cold-start on the accumulated data.

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
   - **Data-bound** signals: growing train/val gap (overfitting);
     `locked_val_mse` regresses after retraining on unchanged data;
     per-joint breakdown shows one joint saturated while another is still
     learning; recent scene perturbations invalidate prior episodes. ->
     Lever: `explore` (new episodes), rebalance exploration toward the
     lagging joint, or `idle` to request a scene perturbation.

   **You MUST name one of the three as the primary constraint in your
   `reason` field** (use the literal tokens "capacity-bound",
   "compute-bound", or "data-bound" somewhere in `reason`). If the same
   constraint has been named in the last 3 consecutive cycles and each
   delivered <30% of the prior cycle's improvement, that is a signal your
   diagnosis was wrong — pivot to a different axis this cycle even if it
   feels higher-risk.

2. Compare `current_locked_val_mse` to `arm_a_locked_val_mse`. Is the gap
   closing? Has it stalled?
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
      your capacity/compute/data diagnosis (step 1) and proceed.
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
    """Render the prompt for `claude -p`."""
    paths = context.get("recent_action_canvas_paths") or []
    if paths:
        canvas_paths = "\n".join(f"- `{p}`" for p in paths)
    else:
        canvas_paths = "- (none yet — first VERIFY/EXPLORE hasn't run)"
    return _THINK_PROMPT_TEMPLATE.format(
        arm_a=context.get("goal", {}).get("arm_a_locked_val_mse"),
        current=context.get("goal", {}).get("current_locked_val_mse"),
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
}

_KNOWN_CADENCE_FIELDS = {
    "cold_start_epochs", "ft_epochs", "early_stop_patience",
}


# Architecture-binding fields that the advisor MUST NOT change once a
# checkpoint exists. Changing any of these would mismatch the loaded
# state_dict on the next --fine-tune and crash the train subprocess
# (we caught this twice during the depth=14→16 mismatch and again
# during the 1B scale-up). The advisor's prompt encourages bumping
# capacity when val plateaus, but capacity bumps require a cold-start
# rebuild — out of scope for runtime overrides.
_ARCH_FROZEN_TRAINING_FIELDS = frozenset({
    "patch_size", "embed_dim", "depth", "num_heads",
    "num_train_timesteps",
    # Mixed-precision + memory-fit knobs — also pinned because changing
    # them mid-run could OOM or change loss scale unexpectedly.
    "bf16", "gradient_checkpointing", "use_8bit_adam",
    "gradient_accumulation_steps", "batch_size",
})


def apply_cfg_overrides(
    cfg, overrides: dict, event_log=None,
) -> dict:
    """Apply dotted-path overrides to cfg.training.* and cfg.cadence.*.

    Unknown keys are logged + skipped. Architecture-binding training
    fields are silently rejected with a `claude_override_blocked` event
    (see `_ARCH_FROZEN_TRAINING_FIELDS`). Numeric values ≤ 0 on
    positive-only fields are clamped. Returns the actually-applied dict.
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
        # Architectural fields are frozen once a checkpoint exists.
        # The advisor's prompt invites architecture changes ("bump depth
        # from 12 to 16") but those would mismatch the loaded checkpoint
        # at the next --fine-tune. Block them at the apply boundary.
        if section == "training" and field in _ARCH_FROZEN_TRAINING_FIELDS:
            if event_log is not None:
                event_log.log(
                    "claude_override_blocked",
                    target=section, key=field,
                    requested=raw_value,
                    reason="architecture_frozen",
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
