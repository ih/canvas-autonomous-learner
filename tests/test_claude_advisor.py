"""Unit tests for `learner.claude_advisor`.

These are pure-Python tests — no real `claude -p` subprocess is spawned.
The `run_advisor` subprocess path is covered by stubbing the binary with
a trivial Windows command (`cmd /c echo ...`).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from learner import claude_advisor
from learner.runtime_knobs import RuntimeKnobs
from learner.range_tracker import CurriculumState


# ---------------------------------------------------- parse_response


def test_parse_response_markdown_fenced():
    raw = 'Sure thing.\n\n```json\n{"next_state": "retrain", "reason": "try lower lr"}\n```\n'
    obj = claude_advisor.parse_response(raw)
    assert obj == {"next_state": "retrain", "reason": "try lower lr"}


def test_parse_response_bare_json():
    raw = 'The answer is {"next_state": "verify"} and nothing else.'
    obj = claude_advisor.parse_response(raw)
    assert obj == {"next_state": "verify"}


def test_parse_response_leading_prose():
    raw = 'Let me think.\nHere is my decision:\n{"next_state": "explore", "reason": "needs data"}\nThanks.'
    assert claude_advisor.parse_response(raw)["next_state"] == "explore"


def test_parse_response_multiple_blocks_takes_first():
    raw = '```json\n{"next_state": "idle"}\n```\nAlternatively: ```json\n{"next_state": "explore"}\n```'
    obj = claude_advisor.parse_response(raw)
    assert obj["next_state"] == "idle"


def test_parse_response_empty():
    assert claude_advisor.parse_response("") == {}
    assert claude_advisor.parse_response("just prose, no json") == {}


def test_parse_response_strips_ansi():
    raw = "\x1b[2K\x1b[1G{\"next_state\": \"verify\"}\x1b[0m"
    assert claude_advisor.parse_response(raw) == {"next_state": "verify"}


# ---------------------------------------------------- apply_cfg_overrides


def _training_cfg():
    return SimpleNamespace(
        training=SimpleNamespace(
            lr=3e-4, weight_decay=0.01, warmup_epochs=15,
            embed_dim=512, depth=12, num_heads=16,
            patch_size=16, batch_size=4, val_ratio=0.1,
            seed=42, num_train_timesteps=1000, beta_schedule="cosine",
            prediction_type="sample", grad_clip=1.0, min_lr=1e-6,
            lr_schedule="cosine",
        ),
        cadence=SimpleNamespace(
            cold_start_epochs=300, ft_epochs=150, early_stop_patience=30,
        ),
    )


def test_apply_cfg_overrides_training():
    cfg = _training_cfg()
    applied = claude_advisor.apply_cfg_overrides(
        cfg, {"training.lr": 1e-5, "training.warmup_epochs": 5},
    )
    assert applied == {"training.lr": 1e-5, "training.warmup_epochs": 5}
    assert cfg.training.lr == 1e-5
    assert cfg.training.warmup_epochs == 5


def test_apply_cfg_overrides_cadence():
    cfg = _training_cfg()
    applied = claude_advisor.apply_cfg_overrides(
        cfg, {"cadence.ft_epochs": 200, "cadence.early_stop_patience": 50},
    )
    assert cfg.cadence.ft_epochs == 200
    assert cfg.cadence.early_stop_patience == 50


def test_apply_cfg_overrides_drops_unknown():
    cfg = _training_cfg()
    applied = claude_advisor.apply_cfg_overrides(
        cfg, {"training.lr": 1e-4, "training.unknown_field": 999, "other": 1},
    )
    assert "training.lr" in applied
    assert "training.unknown_field" not in applied
    assert "other" not in applied


def test_apply_cfg_overrides_clamps_nonpositive():
    cfg = _training_cfg()
    claude_advisor.apply_cfg_overrides(cfg, {"training.lr": -1.0})
    assert cfg.training.lr > 0


# ---------------------------------------------------- apply_curriculum_overrides


def _stage1_curriculum():
    cfg_range = SimpleNamespace(
        enabled=True,
        primary=SimpleNamespace(
            control_joint="shoulder_pan",
            initial_half_width=15.0,
            full_min=-60.0, full_max=60.0,
            expansion_factor=1.5,
            stable_cycles_required=1,
        ),
        secondary=SimpleNamespace(
            control_joint="elbow_flex",
            initial_half_width=15.0,
            full_min=50.0, full_max=90.0,
            expansion_factor=1.5,
            stable_cycles_required=1,
            pinned_half_width=2.5,
        ),
    )
    return CurriculumState.from_config_or_registry(cfg_range, registry_snapshot=None)


def test_apply_curriculum_primary_active_clamps():
    cur = _stage1_curriculum()
    applied = claude_advisor.apply_curriculum_overrides(
        cur, {"primary.active": [-999, 999]},
    )
    assert applied["primary.active"] == [-60.0, 60.0]
    assert cur.primary.active == (-60.0, 60.0)


def test_apply_curriculum_primary_stable_cycles_reset():
    cur = _stage1_curriculum()
    cur.primary.stable_cycles = 5
    claude_advisor.apply_curriculum_overrides(cur, {"primary.stable_cycles": 0})
    assert cur.primary.stable_cycles == 0


def test_apply_curriculum_force_stage_transition():
    cur = _stage1_curriculum()
    assert cur.stage == CurriculumState.STAGE_PRIMARY
    claude_advisor.apply_curriculum_overrides(cur, {"force_stage_transition": True})
    assert cur.stage == CurriculumState.STAGE_SECONDARY


def test_apply_curriculum_secondary_pinned_half_width():
    cur = _stage1_curriculum()
    claude_advisor.apply_curriculum_overrides(
        cur, {"secondary.pinned_half_width": 5.0},
    )
    assert cur.secondary_pinned_half_width == 5.0


def test_apply_curriculum_drops_unknown():
    cur = _stage1_curriculum()
    applied = claude_advisor.apply_curriculum_overrides(
        cur, {"not_a_key": 123},
    )
    assert applied == {}


# ---------------------------------------------------- resolve_next_state


def test_resolve_next_state_happy():
    assert claude_advisor.resolve_next_state(
        "verify", "verify", 0, 5, has_scene_description=False,
    ) == "verify"
    assert claude_advisor.resolve_next_state(
        "explore", "verify", 0, 5, has_scene_description=False,
    ) == "explore"


def test_resolve_next_state_unknown_falls_back_to_default():
    assert claude_advisor.resolve_next_state(
        "nonsense", "explore", 0, 5, has_scene_description=False,
    ) == "explore"


def test_resolve_next_state_retrain_cap_hit():
    assert claude_advisor.resolve_next_state(
        "retrain", "verify", 5, 5, has_scene_description=False,
    ) == "verify"


def test_resolve_next_state_retrain_under_cap():
    assert claude_advisor.resolve_next_state(
        "retrain", "verify", 2, 5, has_scene_description=False,
    ) == "retrain"


def test_resolve_next_state_idle_requires_description():
    # Missing description → fall back.
    assert claude_advisor.resolve_next_state(
        "idle", "verify", 0, 5, has_scene_description=False,
    ) == "verify"
    # With description → allowed.
    assert claude_advisor.resolve_next_state(
        "idle", "verify", 0, 5, has_scene_description=True,
    ) == "idle"


# ---------------------------------------------------- snapshot_run_context


class _FakeRegistry:
    def __init__(self, history=None):
        self._history = history or []

    def locked_val_history(self):
        return list(self._history)

    def episodes_collected(self):
        return 40

    def accumulated_canvas_dirs(self):
        return ["dir1", "dir2"]

    def experiment_status(self):
        return "running"

    def consecutive_guard_rejections(self):
        return 1


def _cfg_for_snapshot(tmp_path: Path):
    runs = tmp_path / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        paths=SimpleNamespace(runs_dir=str(runs)),
        robot=SimpleNamespace(control_joint="shoulder_pan"),
        thresholds=SimpleNamespace(
            tau_low=0.05, tau_high=0.10, val_guard=1.25,
            max_consecutive_rejections=3,
        ),
        cadence=SimpleNamespace(
            probes_per_verify=6, window_size=24, settle_time=0.5,
            base_explore_batch_size=20,
            explore_batch_size_min=20, explore_batch_size_max=20,
            max_sub_bursts=1, min_sub_burst_size=20,
            safety_cap_episodes=1500, warmup_cycles=2,
            explore_max_retries=2, explore_retry_backoff=10.0,
            cold_start_epochs=300, ft_epochs=150, early_stop_patience=30,
        ),
        training=SimpleNamespace(
            lr=3e-4, weight_decay=0.01, warmup_epochs=15,
            embed_dim=512, depth=12, num_heads=16,
            patch_size=16, batch_size=4, val_ratio=0.1,
            seed=42, num_train_timesteps=1000, beta_schedule="cosine",
            prediction_type="sample", grad_clip=1.0, min_lr=1e-6,
            lr_schedule="cosine",
        ),
    )


def test_snapshot_run_context_has_all_keys(tmp_path):
    cfg = _cfg_for_snapshot(tmp_path)
    knobs = RuntimeKnobs.from_cfg(cfg)

    events_path = Path(cfg.paths.runs_dir) / "events_test.jsonl"
    with events_path.open("w") as f:
        f.write(json.dumps({"t": 1.0, "event": "retrain_start"}) + "\n")
        f.write(json.dumps({"t": 1.1, "event": "training_dataset_size",
                            "train_canvases": 100, "val_canvases": 10}) + "\n")
        f.write(json.dumps({"t": 2.0, "event": "training_progress",
                            "epoch": 1, "total_epochs": 10,
                            "train_loss": 0.5, "val_loss": 0.6,
                            "best_val": 0.6}) + "\n")
        f.write(json.dumps({"t": 3.0, "event": "verify_summary",
                            "cycle": 0, "mean_err": 0.08,
                            "n_in_range": 6, "active_range": [-15, 15]}) + "\n")

    registry = _FakeRegistry(history=[
        {"cycle": 0, "locked_val_mse": 0.12, "accepted": True},
        {"cycle": 1, "locked_val_mse": 0.10, "accepted": True},
    ])

    ctx = claude_advisor.snapshot_run_context(
        events_path, registry, cfg, knobs, curriculum=None,
        default_next_state="verify",
        consecutive_retrains_without_data=2,
        claude_max_consecutive_retrains=5,
    )
    assert "goal" in ctx
    assert ctx["goal"]["current_locked_val_mse"] == 0.10
    assert ctx["goal"]["arm_a_locked_val_mse"] == 0.0375
    assert ctx["cycle"] == 2
    assert ctx["episodes_collected"] == 40
    assert ctx["knobs"]["tau_low"] == 0.05
    assert ctx["training_cfg"]["lr"] == 3e-4
    assert ctx["default_next_state"] == "verify"
    assert ctx["advisor_budget"]["consecutive_retrains_without_data"] == 2
    assert ctx["last_training_curve"]["train_canvases"] == 100
    assert ctx["last_training_curve"]["train_loss"] == [0.5]
    assert len(ctx["recent_verifies"]) == 1
    assert ctx["recent_verifies"][0]["mean_err"] == 0.08


# ---------------------------------------------------- build_think_prompt


def test_build_think_prompt_contains_goal_and_schema():
    ctx = {
        "goal": {
            "arm_a_locked_val_mse": 0.0375,
            "current_locked_val_mse": 0.088,
        },
        "default_next_state": "verify",
    }
    prompt = claude_advisor.build_think_prompt(ctx)
    assert "0.0375" in prompt
    assert "0.088" in prompt
    assert "next_state" in prompt
    assert "verify" in prompt
    assert "explore" in prompt
    assert "retrain" in prompt
    assert "idle" in prompt
    assert "terminate" in prompt
    assert "scene_change_description" in prompt
    # JSON schema example is present.
    assert "runtime_overrides" in prompt
    assert "training_overrides" in prompt
    assert "curriculum_overrides" in prompt


# ---------------------------------------------------- run_advisor (real subprocess)


def test_run_advisor_fails_open_when_claude_missing(monkeypatch):
    # Force shutil.which to return None so run_advisor reports "not on PATH".
    import shutil
    monkeypatch.setattr(shutil, "which", lambda *a, **k: None)
    decision = claude_advisor.run_advisor(
        "ignored", timeout_s=1.0, default_next_state="explore",
    )
    assert decision["next_state"] == "explore"
    assert decision["reason"] != ""


# ---------------------------------------------------- recent advisor + gpu signals


def test_extract_recent_advisor_decisions_shape():
    events = [
        {"t": 1.0, "event": "claude_think", "cycle": 3, "advice": {
            "next_state": "retrain", "reason": "drop batch",
            "training_overrides": {"training.batch_size": 2},
            "from_scratch": True,
        }},
        {"t": 2.0, "event": "verify_summary", "cycle": 4},  # ignored
        {"t": 3.0, "event": "claude_think", "cycle": 4, "advice": {
            "next_state": "explore", "reason": "need data",
        }},
    ]
    out = claude_advisor._extract_recent_advisor_decisions(events, n=10)
    assert len(out) == 2
    assert out[0]["cycle"] == 3
    assert out[0]["next_state"] == "retrain"
    assert out[0]["reason"] == "drop batch"
    assert out[0]["training_overrides"] == {"training.batch_size": 2}
    assert out[0]["from_scratch"] is True
    assert out[1]["cycle"] == 4
    assert out[1]["next_state"] == "explore"
    assert out[1]["from_scratch"] is False
    # Empty-advice safety.
    ev_empty = [{"event": "claude_think", "cycle": 1}]
    out2 = claude_advisor._extract_recent_advisor_decisions(ev_empty)
    assert out2 == [{
        "cycle": 1, "next_state": None, "reason": None,
        "runtime_overrides": {}, "training_overrides": {},
        "curriculum_overrides": {}, "explore_overrides": {},
        "from_scratch": False,
    }]


def test_extract_recent_advisor_decisions_limits_to_n():
    events = [
        {"event": "claude_think", "cycle": i,
         "advice": {"next_state": "verify", "reason": f"r{i}"}}
        for i in range(20)
    ]
    out = claude_advisor._extract_recent_advisor_decisions(events, n=5)
    assert len(out) == 5
    assert [o["cycle"] for o in out] == [15, 16, 17, 18, 19]


def test_extract_recent_gpu_signals_filters_types():
    events = [
        {"t": 1.0, "event": "gpu_memory_sample", "used_mb": 10240,
         "total_mb": 32607, "used_frac": 0.31, "util_pct": 55},
        {"t": 2.0, "event": "verify_summary", "cycle": 1},  # ignored
        {"t": 3.0, "event": "training_memory_abort", "cycle": 2,
         "tag": "train_diffusion",
         "summary": {"used_mb": 30720, "total_mb": 32607, "used_frac": 0.94,
                     "peak_used_mb": 30800}},
        {"t": 4.0, "event": "training_stalled", "cycle": 3,
         "tag": "train_diffusion",
         "seconds_since_last_progress": 630.5,
         "summary": {"used_mb": 29500, "total_mb": 32607}},
        {"t": 5.0, "event": "inference_oom", "cycle": 4,
         "error": "CUDA out of memory. Tried to allocate 1.23 GiB."},
        {"t": 6.0, "event": "verify_gpu_headroom", "cycle": 4,
         "used_mb": 8000, "total_mb": 32607, "used_frac": 0.24, "util_pct": 10},
    ]
    out = claude_advisor._extract_recent_gpu_signals(events, n=20)
    kinds = [o["event"] for o in out]
    assert kinds == [
        "gpu_memory_sample", "training_memory_abort", "training_stalled",
        "inference_oom", "verify_gpu_headroom",
    ]
    # training_memory_abort carries the pre-built summary.
    abort_entry = [o for o in out if o["event"] == "training_memory_abort"][0]
    assert abort_entry["tag"] == "train_diffusion"
    assert abort_entry["summary"]["used_frac"] == 0.94
    # gpu_memory_sample: summary reconstructed from flat fields.
    sample_entry = [o for o in out if o["event"] == "gpu_memory_sample"][0]
    assert sample_entry["summary"] == {
        "used_mb": 10240, "total_mb": 32607, "used_frac": 0.31, "util_pct": 55,
    }
    # training_stalled preserves the gap field.
    stalled_entry = [o for o in out if o["event"] == "training_stalled"][0]
    assert stalled_entry["seconds_since_last_progress"] == 630.5
    # inference_oom preserves the error string (truncated at 400).
    oom_entry = [o for o in out if o["event"] == "inference_oom"][0]
    assert "CUDA out of memory" in oom_entry["error"]


def test_extract_recent_gpu_signals_caps_at_n():
    events = [
        {"event": "gpu_memory_sample", "used_mb": i, "total_mb": 100,
         "used_frac": i / 100, "util_pct": i}
        for i in range(30)
    ]
    out = claude_advisor._extract_recent_gpu_signals(events, n=5)
    assert len(out) == 5
    assert out[0]["summary"]["used_mb"] == 25
    assert out[-1]["summary"]["used_mb"] == 29


def test_snapshot_run_context_surfaces_decisions_and_gpu_signals(tmp_path):
    cfg = _cfg_for_snapshot(tmp_path)
    knobs = RuntimeKnobs.from_cfg(cfg)

    events_path = Path(cfg.paths.runs_dir) / "events_test.jsonl"
    with events_path.open("w") as f:
        f.write(json.dumps({"t": 1.0, "event": "claude_think", "cycle": 0,
                            "advice": {"next_state": "retrain",
                                       "reason": "lower lr",
                                       "training_overrides": {"training.lr": 1e-4}}}) + "\n")
        f.write(json.dumps({"t": 2.0, "event": "training_memory_abort",
                            "cycle": 1, "tag": "train_diffusion",
                            "summary": {"used_mb": 30720, "total_mb": 32607,
                                        "used_frac": 0.94}}) + "\n")
        f.write(json.dumps({"t": 3.0, "event": "claude_think", "cycle": 1,
                            "advice": {"next_state": "retrain",
                                       "reason": "halve batch size",
                                       "training_overrides": {"training.batch_size": 2}}}) + "\n")

    registry = _FakeRegistry(history=[
        {"cycle": 0, "locked_val_mse": 0.12, "accepted": True},
    ])

    ctx = claude_advisor.snapshot_run_context(
        events_path, registry, cfg, knobs, curriculum=None,
    )
    assert "recent_advisor_decisions" in ctx
    assert "recent_gpu_signals" in ctx
    decisions = ctx["recent_advisor_decisions"]
    assert len(decisions) == 2
    assert decisions[0]["reason"] == "lower lr"
    assert decisions[1]["reason"] == "halve batch size"
    gpu = ctx["recent_gpu_signals"]
    assert len(gpu) == 1
    assert gpu[0]["event"] == "training_memory_abort"
    assert gpu[0]["summary"]["used_frac"] == 0.94


def test_build_think_prompt_contains_hardware_section_and_new_items():
    ctx = {
        "goal": {
            "arm_a_locked_val_mse": 0.0375,
            "current_locked_val_mse": 0.088,
        },
        "default_next_state": "verify",
    }
    prompt = claude_advisor.build_think_prompt(ctx)
    assert "System constraints (hardware)" in prompt
    assert "RTX 5090" in prompt
    assert "32 GB" in prompt
    assert "training_memory_abort" in prompt
    assert "training_stalled" in prompt
    assert "recent_gpu_signals" in prompt
    assert "recent_advisor_decisions" in prompt
