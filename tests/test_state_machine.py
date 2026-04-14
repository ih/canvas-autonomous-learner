"""Integration tests for the unified `main_loop`.

Covers the nine invariants from the plan:
1. Warm start: high verify error → EXPLORE → RETRAIN → low error → IDLE
2. Cold start: no live checkpoint → first state is EXPLORE, first retrain
   uses cold_start_epochs, subsequent uses ft_epochs
3. Dynamic burst sizing: burst scales with mean_err / tau_high
4. Range expansion fires after `stable_cycles_required` stable verifies
5. Range expansion is reset by a single bad cycle mid-streak
6. Satisfied-at-max-range termination when stable at the full range
7. Plateau + safety_cap termination
8. val_guard uses locked_val_mse (not train_val)
9. Accumulated canvas dirs grow by one per cycle and are passed in full
   to retrain_cumulative

All tests stub out verifier.verify_once, explorer.collect_batch,
trainer_driver.build_canvases, and trainer_driver.retrain_cumulative so
no hardware or canvas-world-model subprocess is touched.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from learner import orchestrator
from learner.events import EventLog
from learner.metrics import ProbeResult
from learner.registry import Registry


# ----------------------------------------------------------------- fakes

class FakeHardware:
    """Stand-in Hardware that never touches robot code."""

    def __init__(self):
        self.connected = False
        self.loaded_ckpt: Optional[str] = None
        self.disconnect_calls = 0
        self.goto_calls: list[tuple[str, float]] = []

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False
        self.disconnect_calls += 1

    def load_predictor(self, ckpt):
        self.loaded_ckpt = ckpt

    def reload_checkpoint(self, ckpt):
        self.loaded_ckpt = ckpt

    def goto(self, joint: str, target: float, *a, **k):
        self.goto_calls.append((joint, float(target)))


def _make_cfg(
    tmp_path: Path,
    *,
    cold_start: bool = False,
    probes_per_verify: int = 3,
    warmup_cycles: int = 2,
    safety_cap: int = 1500,
    tau_low: float = 0.005,
    tau_high: float = 0.015,
    initial_half_width: float = 20.0,
    expansion_factor: float = 1.5,
    stable_cycles_required: int = 2,
    range_enabled: bool = True,
) -> SimpleNamespace:
    runs_dir = tmp_path / "runs"
    ckpt_dir = tmp_path / "checkpoints"
    canvas_out = tmp_path / "canvas"
    for d in (runs_dir, ckpt_dir, canvas_out):
        d.mkdir(parents=True, exist_ok=True)

    return SimpleNamespace(
        paths=SimpleNamespace(
            canvas_world_model=str(tmp_path / "cwm"),
            canvas_robot_control=str(tmp_path / "crc"),
            robotic_foundation_model_tests=str(tmp_path / "rfmt"),
            base_canvas=None,
            val_dataset=None,
            locked_val_dataset=str(tmp_path / "locked_val_v1"),
            live_checkpoint=None if cold_start else str(tmp_path / "seed.pth"),
            ckpt_dir=str(ckpt_dir),
            canvas_out=str(canvas_out),
            lerobot_out=str(tmp_path / "lerobot"),
            runs_dir=str(runs_dir),
            registry_file=str(ckpt_dir / "registry.json"),
            python="python",
        ),
        robot=SimpleNamespace(
            port="COM3", robot_id="fake", control_joint="shoulder_pan",
            step_size=10.0, joint_min=-60.0, joint_max=60.0,
            base_camera=1, wrist_camera=0,
            camera_width=640, camera_height=480, camera_fps=10,
        ),
        thresholds=SimpleNamespace(
            tau_low=tau_low,
            tau_high=tau_high,
            val_guard=1.25,
            max_consecutive_rejections=3,
        ),
        cadence=SimpleNamespace(
            idle_seconds=0,
            probes_per_verify=probes_per_verify,
            window_size=16,
            settle_time=0,
            base_explore_batch_size=50,
            explore_batch_size_min=10,
            explore_batch_size_max=150,
            max_sub_bursts=1,      # tests assert one sub-burst per cycle for simplicity
            min_sub_burst_size=1,
            cold_start_epochs=100,
            ft_epochs=5,
            safety_cap_episodes=safety_cap,
            warmup_cycles=warmup_cycles,
            explore_max_retries=0,
            explore_retry_backoff=0.0,
        ),
        actions=SimpleNamespace(candidates=[1, 2, 3]),
        explore=SimpleNamespace(
            action_duration=1.0,
            dataset_fps=10,
            policy_joint_name="shoulder_pan.pos",
            vary_target_joint=False,
            base_camera_name="base_0_rgb",
            wrist_camera_name="left_wrist_0_rgb",
        ),
        range=SimpleNamespace(
            enabled=range_enabled,
            control_joint="shoulder_pan",
            initial_half_width=initial_half_width,
            full_min=-60.0,
            full_max=60.0,
            expansion_factor=expansion_factor,
            stable_cycles_required=stable_cycles_required,
        ),
        dry_run=True,
    )


def _probe(action: int, mse: float, shoulder_pan: float = 0.0, t: float = 0.0) -> ProbeResult:
    return ProbeResult(
        state_key="0", action=action, mse=mse, timestamp=t,
        motor_state=(shoulder_pan, 0.0, 0.0, 0.0, 0.0, 0.0),
    )


def _stub_trio(tmp_path: Path, probe_iter, retrain_results=None, retrain_hook=None):
    """Build fake collect_batch / build_canvases / retrain_cumulative
    callables that can be passed as the underscore kwargs of `main_loop`.
    """
    collect_calls: list = []
    build_calls: list = []
    retrain_calls: list = []
    probes = iter(probe_iter)
    results = iter(retrain_results or [])

    def fake_verify(hardware, action, settle_time, examples_dir=None, example_tag=None,
                    target_joint=None, target_position=None):
        return next(probes)

    def fake_collect(cfg_, n, window=None, event_log=None,
                     joint_range_override=None, randomize_primary_start=None):
        d = tmp_path / f"fake_lerobot_{len(collect_calls)}"
        collect_calls.append({
            "n": n, "joint_range_override": joint_range_override,
            "randomize_primary_start": randomize_primary_start,
        })
        return d

    def fake_build(cfg_, new_lerobot_dir, output_dir, event_log=None):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        build_calls.append({
            "new_lerobot_dir": str(new_lerobot_dir),
            "output_dir": str(output_dir),
        })
        return Path(output_dir)

    def fake_retrain(cfg_, accumulated_canvas_dirs, resume_checkpoint, epochs,
                      locked_val_dataset=None, event_log=None):
        retrain_calls.append({
            "accumulated_canvas_dirs": list(accumulated_canvas_dirs),
            "resume_checkpoint": resume_checkpoint,
            "epochs": epochs,
            "locked_val_dataset": locked_val_dataset,
        })
        if retrain_hook is not None:
            return retrain_hook(len(retrain_calls) - 1, tmp_path)
        try:
            r = next(results)
        except StopIteration:
            return None
        return r

    return fake_verify, fake_collect, fake_build, fake_retrain, {
        "collect_calls": collect_calls,
        "build_calls": build_calls,
        "retrain_calls": retrain_calls,
    }


def _default_retrain_result(idx, tmp_path, locked_val=0.01, train_val=0.01):
    ckpt = tmp_path / f"ft_{idx}.pth"
    ckpt.write_text("x")
    return {
        "checkpoint": str(ckpt),
        "merged_dataset": str(tmp_path / f"merged_{idx}"),
        "train_val_mse": train_val,
        "locked_val_mse": locked_val,
    }


# ---------------------------------------------- invariant 1: warm start

def test_warm_start_high_then_low(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, warmup_cycles=0)  # no warmup for clarity
    # First verify burst high → EXPLORE → RETRAIN → second verify burst low → IDLE
    probes = [
        # All starting position 0 so they land in the active range.
        _probe(1, 0.02, 0.0), _probe(2, 0.02, 0.0), _probe(3, 0.02, 0.0),
        _probe(1, 0.001, 0.0), _probe(2, 0.001, 0.0), _probe(3, 0.001, 0.0),
    ]
    retrain_results = [_default_retrain_result(0, tmp_path, locked_val=0.008)]
    fv, fc, fb, fr, calls = _stub_trio(tmp_path, probes, retrain_results)
    monkeypatch.setattr(orchestrator.verifier, "verify_once", fv)

    hw = FakeHardware()
    event_log = EventLog(cfg.paths.runs_dir, session="warm")
    registry = Registry(cfg.paths.registry_file)

    result = orchestrator.main_loop(
        cfg, hardware=hw, registry=registry, event_log=event_log,
        max_iterations=10,
        _collect_batch=fc, _build_canvases=fb, _retrain=fr,
    )

    assert len(calls["collect_calls"]) == 1
    assert len(calls["retrain_calls"]) == 1
    assert registry.live_checkpoint().endswith("ft_0.pth")
    assert hw.loaded_ckpt is not None and hw.loaded_ckpt.endswith("ft_0.pth")
    assert hw.disconnect_calls >= 1  # disconnected around EXPLORE


# ---------------------------------------------- invariant 2: cold start

def test_cold_start_first_state_is_explore(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, cold_start=True, warmup_cycles=1)
    probes = [
        # Verify after cycle 0 retrain — all high so we keep exploring.
        _probe(1, 0.02, 0.0), _probe(2, 0.02, 0.0), _probe(3, 0.02, 0.0),
    ]
    retrain_results = [_default_retrain_result(0, tmp_path, locked_val=0.12)]
    fv, fc, fb, fr, calls = _stub_trio(tmp_path, probes, retrain_results)
    monkeypatch.setattr(orchestrator.verifier, "verify_once", fv)

    hw = FakeHardware()
    event_log = EventLog(cfg.paths.runs_dir, session="cold")
    registry = Registry(cfg.paths.registry_file)

    orchestrator.main_loop(
        cfg, hardware=hw, registry=registry, event_log=event_log,
        max_iterations=4,  # EXPLORE, RETRAIN, VERIFY, EXPLORE
        _collect_batch=fc, _build_canvases=fb, _retrain=fr,
    )

    assert len(calls["retrain_calls"]) >= 1
    # Cycle 0 is cold start → no resume checkpoint, cold_start_epochs
    assert calls["retrain_calls"][0]["resume_checkpoint"] is None
    assert calls["retrain_calls"][0]["epochs"] == cfg.cadence.cold_start_epochs


def test_cold_start_cycle1_uses_ft_epochs_and_resumes(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, cold_start=True, warmup_cycles=2,
                    stable_cycles_required=999)  # never expand
    # Two cycles of high-MSE verifies so both cycle 0 and 1 do EXPLORE+RETRAIN.
    probes = [
        _probe(1, 0.05, 0.0), _probe(2, 0.05, 0.0), _probe(3, 0.05, 0.0),
        _probe(1, 0.05, 0.0), _probe(2, 0.05, 0.0), _probe(3, 0.05, 0.0),
    ]
    retrain_results = [
        _default_retrain_result(0, tmp_path, locked_val=0.12, train_val=0.08),
        _default_retrain_result(1, tmp_path, locked_val=0.10, train_val=0.07),
    ]
    fv, fc, fb, fr, calls = _stub_trio(tmp_path, probes, retrain_results)
    monkeypatch.setattr(orchestrator.verifier, "verify_once", fv)

    hw = FakeHardware()
    orchestrator.main_loop(
        cfg, hardware=hw,
        registry=Registry(cfg.paths.registry_file),
        event_log=EventLog(cfg.paths.runs_dir, session="cold2"),
        max_iterations=8,
        _collect_batch=fc, _build_canvases=fb, _retrain=fr,
    )
    assert len(calls["retrain_calls"]) >= 2
    assert calls["retrain_calls"][0]["resume_checkpoint"] is None
    assert calls["retrain_calls"][0]["epochs"] == cfg.cadence.cold_start_epochs
    # Cycle 1 resumes from cycle 0's output
    assert calls["retrain_calls"][1]["resume_checkpoint"] is not None
    assert calls["retrain_calls"][1]["resume_checkpoint"].endswith("ft_0.pth")
    assert calls["retrain_calls"][1]["epochs"] == cfg.cadence.ft_epochs


# ----------------------------------------- invariant 3: dynamic bursts

def test_dynamic_burst_scales_with_mean_err(tmp_path, monkeypatch):
    # High error → big burst. Low error → base burst.
    cfg = _make_cfg(tmp_path, warmup_cycles=0,
                    tau_high=0.01)  # so mean_err=0.03 is 3x threshold
    probes = [
        _probe(1, 0.03, 0.0), _probe(2, 0.03, 0.0), _probe(3, 0.03, 0.0),
        _probe(1, 0.0001, 0.0), _probe(2, 0.0001, 0.0), _probe(3, 0.0001, 0.0),
    ]
    retrain_results = [_default_retrain_result(0, tmp_path, locked_val=0.005)]
    fv, fc, fb, fr, calls = _stub_trio(tmp_path, probes, retrain_results)
    monkeypatch.setattr(orchestrator.verifier, "verify_once", fv)

    orchestrator.main_loop(
        cfg, hardware=FakeHardware(),
        registry=Registry(cfg.paths.registry_file),
        event_log=EventLog(cfg.paths.runs_dir, session="burst"),
        max_iterations=8,
        _collect_batch=fc, _build_canvases=fb, _retrain=fr,
    )
    # Warm start → first verify (high) → explore with dynamic burst. mean_err
    # is 0.03 = 3x tau_high, so burst = min(hi, 3*base) = 150.
    assert calls["collect_calls"][0]["n"] == 150


# ---------------------------------------- invariant 4: range expansion

def test_range_expands_after_stable_cycles(tmp_path, monkeypatch):
    # With warmup_cycles=1, cycle 0 always explores. After cycle 0's
    # retrain, verify with LOW error → good_cycle → NOT enough yet (need 2).
    # Wait — stable_cycles_required=2, but we also need enough cycles done.
    # So: cycle 0 warmup explore → verify low → good_cycle (1) → IDLE
    # cycle 1 (no warmup) → verify low → good_cycle (2) → EXPAND → explore.
    cfg = _make_cfg(tmp_path, warmup_cycles=1,
                    stable_cycles_required=2, initial_half_width=10.0)
    probes = [
        _probe(1, 0.001, 0.0), _probe(2, 0.001, 0.0), _probe(3, 0.001, 0.0),  # post cycle 0
        _probe(1, 0.001, 5.0), _probe(2, 0.001, -5.0), _probe(3, 0.001, 0.0),  # post cycle 1
        # After expansion, cycle 1 EXPLOREs, which triggers another retrain
        # and verify. We provide more probes to keep the iter from being
        # exhausted mid-run.
        _probe(1, 0.001, 0.0), _probe(2, 0.001, 0.0), _probe(3, 0.001, 0.0),
    ]
    retrain_results = [
        _default_retrain_result(i, tmp_path, locked_val=0.005, train_val=0.005)
        for i in range(3)
    ]
    fv, fc, fb, fr, calls = _stub_trio(tmp_path, probes, retrain_results)
    monkeypatch.setattr(orchestrator.verifier, "verify_once", fv)

    registry = Registry(cfg.paths.registry_file)
    orchestrator.main_loop(
        cfg, hardware=FakeHardware(), registry=registry,
        event_log=EventLog(cfg.paths.runs_dir, session="expand"),
        max_iterations=12,
        _collect_batch=fc, _build_canvases=fb, _retrain=fr,
    )

    # primary range_history should have at least one entry after expansion
    snap = registry.range_snapshot()
    history = snap.get("range_primary_history") or snap.get("range_history") or []
    assert len(history) >= 1
    assert history[0]["old_range"] == [-10.0, 10.0]
    assert history[0]["new_range"] == [-15.0, 15.0]  # 10 * 1.5 = 15


# ---------------------------------------- invariant 5: bad cycle resets

def test_bad_cycle_resets_stable_streak(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, warmup_cycles=0,
                    stable_cycles_required=2, initial_half_width=10.0)
    probes = [
        # cycle 0: LOW → good_cycle(1) → not yet expand
        _probe(1, 0.001, 0.0), _probe(2, 0.001, 0.0), _probe(3, 0.001, 0.0),
        # cycle 1: HIGH → bad_cycle → streak reset
        _probe(1, 0.05, 0.0), _probe(2, 0.05, 0.0), _probe(3, 0.05, 0.0),
        # cycle 1 explores → retrain → verify low → good_cycle(1)
        _probe(1, 0.001, 0.0), _probe(2, 0.001, 0.0), _probe(3, 0.001, 0.0),
    ]
    retrain_results = [_default_retrain_result(i, tmp_path, locked_val=0.005) for i in range(3)]
    fv, fc, fb, fr, calls = _stub_trio(tmp_path, probes, retrain_results)
    monkeypatch.setattr(orchestrator.verifier, "verify_once", fv)

    registry = Registry(cfg.paths.registry_file)
    orchestrator.main_loop(
        cfg, hardware=FakeHardware(), registry=registry,
        event_log=EventLog(cfg.paths.runs_dir, session="badcycle"),
        max_iterations=10,
        _collect_batch=fc, _build_canvases=fb, _retrain=fr,
    )
    snap = registry.range_snapshot()
    # No expansion should have happened
    assert len(snap["range_history"]) == 0
    assert snap["range_stable_cycles"] <= 1


# -------------------------------------- invariant 6: satisfied at max

def test_satisfied_at_full_range(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, warmup_cycles=0, stable_cycles_required=2,
                    initial_half_width=60.0)  # already at full range
    probes = [
        _probe(1, 0.001, 0.0), _probe(2, 0.001, 0.0), _probe(3, 0.001, 0.0),
        _probe(1, 0.001, 10.0), _probe(2, 0.001, -10.0), _probe(3, 0.001, 20.0),
    ]
    fv, fc, fb, fr, calls = _stub_trio(tmp_path, probes, retrain_results=[])
    monkeypatch.setattr(orchestrator.verifier, "verify_once", fv)

    result = orchestrator.main_loop(
        cfg, hardware=FakeHardware(),
        registry=Registry(cfg.paths.registry_file),
        event_log=EventLog(cfg.paths.runs_dir, session="satisfied"),
        max_iterations=10,
        _collect_batch=fc, _build_canvases=fb, _retrain=fr,
    )
    # Single-joint curriculum (legacy flat range block, no secondary) →
    # "satisfied at full curriculum" still fires at max + stable.
    assert result["reason"] == "satisfied_at_full_curriculum"
    # Should never have explored
    assert len(calls["collect_calls"]) == 0


# --------------------------------------------- invariant 7: safety cap

def test_safety_cap_fires(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, warmup_cycles=10, safety_cap=100,
                    stable_cycles_required=999)  # force explore every cycle
    # Many high-MSE probes, keep exploring
    probes = [_probe(1, 0.05, 0.0) for _ in range(30)]
    retrain_results = [_default_retrain_result(i, tmp_path, locked_val=0.10) for i in range(30)]
    fv, fc, fb, fr, calls = _stub_trio(tmp_path, probes, retrain_results)
    monkeypatch.setattr(orchestrator.verifier, "verify_once", fv)

    result = orchestrator.main_loop(
        cfg, hardware=FakeHardware(),
        registry=Registry(cfg.paths.registry_file),
        event_log=EventLog(cfg.paths.runs_dir, session="cap"),
        max_iterations=100,
        _collect_batch=fc, _build_canvases=fb, _retrain=fr,
    )
    assert result["reason"] == "safety_cap_hit"
    assert result["total_eps"] >= cfg.cadence.safety_cap_episodes


# ------------------------------------------ invariant 8: locked-val guard

def test_val_guard_uses_locked_val_mse(tmp_path, monkeypatch):
    # Cycle 0 accepted (cycle < warmup). Cycle 1: train_val ticks UP but
    # locked_val ticks DOWN — the guard must accept on locked_val.
    cfg = _make_cfg(tmp_path, warmup_cycles=1, stable_cycles_required=999)
    probes = [
        _probe(1, 0.05, 0.0), _probe(2, 0.05, 0.0), _probe(3, 0.05, 0.0),
        _probe(1, 0.05, 0.0), _probe(2, 0.05, 0.0), _probe(3, 0.05, 0.0),
    ]
    retrain_results = [
        _default_retrain_result(0, tmp_path, locked_val=0.10, train_val=0.02),
        _default_retrain_result(1, tmp_path, locked_val=0.08, train_val=0.05),
        # cycle 1's train_val=0.05 > 0.02*1.25=0.025 would fail a train-guard,
        # but locked_val=0.08 < 0.10*1.25=0.125 so the locked guard accepts.
    ]
    fv, fc, fb, fr, calls = _stub_trio(tmp_path, probes, retrain_results)
    monkeypatch.setattr(orchestrator.verifier, "verify_once", fv)

    registry = Registry(cfg.paths.registry_file)
    orchestrator.main_loop(
        cfg, hardware=FakeHardware(), registry=registry,
        event_log=EventLog(cfg.paths.runs_dir, session="guard"),
        max_iterations=10,
        _collect_batch=fc, _build_canvases=fb, _retrain=fr,
    )
    history = registry.locked_val_history()
    assert len(history) >= 2
    # Cycle 1's retrain should be ACCEPTED because locked_val improved
    assert history[1]["accepted"] is True


# --------------------------------- invariant 9: accumulated dirs grow

def test_accumulated_canvas_dirs_grow_and_passed_in_full(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, cold_start=True, warmup_cycles=3,
                    stable_cycles_required=999)
    probes = [_probe(1, 0.05, 0.0) for _ in range(30)]
    retrain_results = [_default_retrain_result(i, tmp_path, locked_val=0.08) for i in range(3)]
    fv, fc, fb, fr, calls = _stub_trio(tmp_path, probes, retrain_results)
    monkeypatch.setattr(orchestrator.verifier, "verify_once", fv)

    orchestrator.main_loop(
        cfg, hardware=FakeHardware(),
        registry=Registry(cfg.paths.registry_file),
        event_log=EventLog(cfg.paths.runs_dir, session="accum"),
        max_iterations=12,
        _collect_batch=fc, _build_canvases=fb, _retrain=fr,
    )
    # Each retrain_cumulative call's `accumulated_canvas_dirs` should be
    # one entry longer than the previous.
    for i, call in enumerate(calls["retrain_calls"]):
        assert len(call["accumulated_canvas_dirs"]) == i + 1
        if i > 0:
            prev = calls["retrain_calls"][i - 1]["accumulated_canvas_dirs"]
            assert call["accumulated_canvas_dirs"][: len(prev)] == prev
