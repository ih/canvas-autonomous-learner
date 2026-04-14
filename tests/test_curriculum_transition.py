"""Sequential (Option B) curriculum transition tests.

Stage 1 (primary = shoulder_pan) expands until at_max + stable, then
transitions to Stage 2 (secondary = elbow_flex) which has its own
curriculum, and the experiment terminates when the secondary is
at_max + stable. These tests use the same synthetic-trajectory
pattern as `test_state_machine.py` so no hardware is required.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from learner import orchestrator
from learner.events import EventLog
from learner.metrics import ProbeResult
from learner.range_tracker import CurriculumState
from learner.registry import Registry


class FakeHardware:
    def __init__(self):
        self.connected = False
        self.loaded_ckpt = None
        self.disconnect_calls = 0
        self.goto_calls: list[tuple[str, float]] = []

    def connect(self): self.connected = True
    def disconnect(self):
        self.connected = False
        self.disconnect_calls += 1
    def load_predictor(self, ckpt): self.loaded_ckpt = ckpt
    def reload_checkpoint(self, ckpt): self.loaded_ckpt = ckpt
    def goto(self, joint, target, *a, **k):
        self.goto_calls.append((joint, float(target)))


def _probe(action, mse, shoulder_pan=0.0, elbow_flex=70.0):
    return ProbeResult(
        state_key="0", action=action, mse=mse, timestamp=0.0,
        motor_state=(shoulder_pan, 0.0, elbow_flex, 0.0, 0.0, 0.0),
    )


def _make_cfg(tmp_path: Path, *,
              primary_initial_hw: float = 20.0,
              primary_expansion: float = 3.0,   # big so few steps to max
              primary_stable: int = 1,
              secondary_initial_hw: float = 10.0,
              secondary_expansion: float = 2.0,
              secondary_stable: int = 1,
              secondary_pinned_hw: float = 2.0) -> SimpleNamespace:
    runs = tmp_path / "runs"; runs.mkdir()
    ckpt = tmp_path / "ckpts"; ckpt.mkdir()
    canvas = tmp_path / "canvas"; canvas.mkdir()

    return SimpleNamespace(
        paths=SimpleNamespace(
            canvas_world_model=str(tmp_path / "cwm"),
            canvas_robot_control=str(tmp_path / "crc"),
            robotic_foundation_model_tests=str(tmp_path / "rfmt"),
            base_canvas=None, val_dataset=None,
            locked_val_dataset=str(tmp_path / "locked_val"),
            live_checkpoint=None,   # cold start
            ckpt_dir=str(ckpt), canvas_out=str(canvas),
            lerobot_out=str(tmp_path / "lerobot"),
            runs_dir=str(runs),
            registry_file=str(ckpt / "registry.json"),
            python="python",
        ),
        robot=SimpleNamespace(
            port="COM3", robot_id="fake", control_joint="shoulder_pan",
            step_size=10.0, joint_min=-60.0, joint_max=60.0,
            base_camera=1, wrist_camera=0,
            camera_width=640, camera_height=480, camera_fps=10,
        ),
        thresholds=SimpleNamespace(
            tau_low=0.005, tau_high=0.015,
            val_guard=1.25, max_consecutive_rejections=3,
        ),
        cadence=SimpleNamespace(
            idle_seconds=0, probes_per_verify=3, window_size=16,
            settle_time=0, base_explore_batch_size=50,
            explore_batch_size_min=10, explore_batch_size_max=100,
            max_sub_bursts=1, min_sub_burst_size=1,
            cold_start_epochs=100, ft_epochs=5,
            safety_cap_episodes=5000, warmup_cycles=0,
            explore_max_retries=0, explore_retry_backoff=0.0,
        ),
        actions=SimpleNamespace(candidates=[1, 2, 3]),
        explore=SimpleNamespace(
            action_duration=1.0, dataset_fps=10,
            policy_joint_name="shoulder_pan.pos",
            vary_target_joint=False,
            base_camera_name="base_0_rgb", wrist_camera_name="left_wrist_0_rgb",
        ),
        range=SimpleNamespace(
            enabled=True,
            primary=SimpleNamespace(
                control_joint="shoulder_pan",
                initial_half_width=primary_initial_hw,
                full_min=-60.0, full_max=60.0,
                expansion_factor=primary_expansion,
                stable_cycles_required=primary_stable,
            ),
            secondary=SimpleNamespace(
                control_joint="elbow_flex",
                initial_half_width=secondary_initial_hw,
                full_min=50.0, full_max=90.0,
                expansion_factor=secondary_expansion,
                stable_cycles_required=secondary_stable,
                pinned_half_width=secondary_pinned_hw,
            ),
        ),
        dry_run=True,
    )


def _stub_trio(tmp_path: Path, probe_iter, retrain_results):
    collect_calls: list = []
    retrain_calls: list = []
    probes = iter(probe_iter)
    results = iter(retrain_results)

    def fake_verify(hw, action, settle_time, examples_dir=None, example_tag=None,
                    target_joint=None, target_position=None):
        return next(probes)

    def fake_collect(cfg_, n, window=None, event_log=None,
                     joint_range_override=None, randomize_primary_start=None):
        d = tmp_path / f"fake_lerobot_{len(collect_calls)}"
        collect_calls.append({
            "n": n, "joint_range_override": joint_range_override,
        })
        return d

    def fake_build(cfg_, new_lerobot_dir, output_dir, event_log=None):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "dataset_meta.json").write_text("{}")
        return Path(output_dir)

    def fake_retrain(cfg_, accumulated_canvas_dirs, resume_checkpoint, epochs,
                      locked_val_dataset=None, event_log=None):
        retrain_calls.append({"resume": resume_checkpoint, "epochs": epochs})
        try:
            return next(results)
        except StopIteration:
            return None

    return fake_verify, fake_collect, fake_build, fake_retrain, {
        "collect_calls": collect_calls,
        "retrain_calls": retrain_calls,
    }


def _retrain_result(idx, tmp_path, locked_val=0.003, train_val=0.003):
    ckpt = tmp_path / f"ft_{idx}.pth"
    ckpt.write_text("x")
    return {
        "checkpoint": str(ckpt),
        "merged_dataset": str(tmp_path / f"merged_{idx}"),
        "train_val_mse": train_val,
        "locked_val_mse": locked_val,
    }


def test_stage1_to_stage2_transition(tmp_path, monkeypatch):
    """Stage 1 starts, primary expands to full, transitions to stage 2,
    stage 2 expands to full, experiment terminates with
    satisfied_at_full_curriculum."""
    # Initial primary half_width=20 → [-20,20]. Expansion factor 3 →
    # [-60, 60] in one step (clamped). stable_cycles_required=1.
    # So primary satisfies after 1 good cycle → expand to full → 1 more good
    # cycle → transition to secondary.
    cfg = _make_cfg(tmp_path,
                    primary_initial_hw=20.0, primary_expansion=3.0,
                    primary_stable=1,
                    secondary_initial_hw=10.0, secondary_expansion=3.0,
                    secondary_stable=1)

    # All low-MSE probes — every verify passes. But the probes' motor_state
    # must be in the ACTIVE curriculum joint's current range to count.
    # Stage 1: probes land at shoulder=0 (inside any shoulder range)
    # Stage 2: probes land at elbow=70 (center of [50,90]) inside any elbow range
    probes = []
    for _ in range(30):
        probes.append(_probe(1, 0.001, shoulder_pan=0.0, elbow_flex=70.0))
    retrain_results = [_retrain_result(i, tmp_path) for i in range(10)]
    fv, fc, fb, fr, calls = _stub_trio(tmp_path, probes, retrain_results)
    monkeypatch.setattr(orchestrator.verifier, "verify_once", fv)

    registry = Registry(cfg.paths.registry_file)
    hw = FakeHardware()
    result = orchestrator.main_loop(
        cfg, hardware=hw, registry=registry,
        event_log=EventLog(cfg.paths.runs_dir, session="transition"),
        max_iterations=40,
        _collect_batch=fc, _build_canvases=fb, _retrain=fr,
    )

    assert result["reason"] == "satisfied_at_full_curriculum"
    assert result["final_stage"] == CurriculumState.STAGE_SECONDARY

    # Registry should have BOTH primary and secondary snapshots populated.
    snap = registry.range_snapshot()
    assert snap.get("curriculum_stage") == CurriculumState.STAGE_SECONDARY
    assert snap.get("range_primary_active") == [-60.0, 60.0]
    # Secondary reached full [50, 90]
    assert snap.get("range_secondary_active") == [50.0, 90.0]
    # Primary has at least one expansion logged
    assert len(snap.get("range_primary_history") or []) >= 1
    # Secondary has at least one expansion logged
    assert len(snap.get("range_secondary_history") or []) >= 1


def test_stage1_explore_pins_secondary_at_center(tmp_path, monkeypatch):
    """During stage 1, EXPLORE sub-bursts should include BOTH a primary
    range override (curriculum-driven) AND a secondary range override
    pinned to a narrow center band around elbow_flex's full-range midpoint.
    """
    cfg = _make_cfg(tmp_path,
                    primary_initial_hw=15.0, primary_expansion=1.5,
                    primary_stable=999,   # never transition
                    secondary_pinned_hw=2.5)
    # Warm start so the first iteration is IDLE → VERIFY (not a cold
    # EXPLORE we don't care about for this test).
    cfg.paths.live_checkpoint = str(tmp_path / "seed.pth")

    # One high-MSE verify burst → EXPLORE
    probes = [_probe(1, 0.05, 0.0), _probe(2, 0.05, 0.0), _probe(3, 0.05, 0.0)]
    retrain_results = [_retrain_result(0, tmp_path, locked_val=0.05)]
    fv, fc, fb, fr, calls = _stub_trio(tmp_path, probes, retrain_results)
    monkeypatch.setattr(orchestrator.verifier, "verify_once", fv)

    orchestrator.main_loop(
        cfg, hardware=FakeHardware(),
        registry=Registry(cfg.paths.registry_file),
        event_log=EventLog(cfg.paths.runs_dir, session="stage1"),
        max_iterations=6,
        _collect_batch=fc, _build_canvases=fb, _retrain=fr,
    )

    assert len(calls["collect_calls"]) >= 1
    override = calls["collect_calls"][0]["joint_range_override"]
    assert "shoulder_pan.pos" in override
    assert "elbow_flex.pos" in override
    # shoulder_pan pinned to the primary tracker's narrow initial range
    assert override["shoulder_pan.pos"] == (-15.0, 15.0)
    # elbow_flex pinned to a tight center band [70-2.5, 70+2.5] = [67.5, 72.5]
    lo, hi = override["elbow_flex.pos"]
    assert abs(lo - 67.5) < 1e-9 and abs(hi - 72.5) < 1e-9


def test_stage2_explore_pins_primary_at_full_and_curriculum_on_secondary(tmp_path, monkeypatch):
    """In stage 2, EXPLORE overrides should pin shoulder_pan at its FULL
    range and let elbow_flex be the curriculum-active joint.
    """
    # Pre-seed the registry in stage 2 so the orchestrator starts there.
    cfg = _make_cfg(tmp_path,
                    primary_initial_hw=60.0,   # already at full
                    primary_stable=1,
                    secondary_initial_hw=10.0,
                    secondary_stable=999)       # never advance in secondary

    reg_path = Path(cfg.paths.registry_file)
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    reg_path.write_text(json.dumps({
        "live_checkpoint": None, "base_canvas_dataset": None,
        "merged_canvas_dataset": None, "baseline_val_mse": None,
        "last_retrain": None, "history": [],
        "episodes_collected": 0, "accumulated_canvas_dirs": [],
        "locked_val_history": [], "experiment_status": "unstarted",
        "consecutive_guard_rejections": 0,
        "curriculum_stage": CurriculumState.STAGE_SECONDARY,
        "range_primary_active": [-60.0, 60.0],
        "range_primary_stable_cycles": 1,
        "range_primary_history": [],
        "range_secondary_active": [60.0, 80.0],
        "range_secondary_stable_cycles": 0,
        "range_secondary_history": [],
    }, indent=2))

    # Warm start (live_checkpoint None but skip cold start by setting it on cfg.paths)
    cfg.paths.live_checkpoint = str(tmp_path / "seed.pth")

    # One high-MSE verify burst → EXPLORE. Probes must land inside the
    # current stage-2 elbow range [60, 80] to count toward mean_err.
    probes = [
        _probe(1, 0.05, shoulder_pan=0.0, elbow_flex=70.0),
        _probe(2, 0.05, shoulder_pan=0.0, elbow_flex=65.0),
        _probe(3, 0.05, shoulder_pan=0.0, elbow_flex=75.0),
    ]
    retrain_results = [_retrain_result(0, tmp_path, locked_val=0.05)]
    fv, fc, fb, fr, calls = _stub_trio(tmp_path, probes, retrain_results)
    monkeypatch.setattr(orchestrator.verifier, "verify_once", fv)

    hw = FakeHardware()
    orchestrator.main_loop(
        cfg, hardware=hw,
        registry=Registry(cfg.paths.registry_file),
        event_log=EventLog(cfg.paths.runs_dir, session="stage2"),
        max_iterations=8,
        _collect_batch=fc, _build_canvases=fb, _retrain=fr,
    )

    assert len(calls["collect_calls"]) >= 1
    override = calls["collect_calls"][0]["joint_range_override"]
    assert "shoulder_pan.pos" in override
    assert "elbow_flex.pos" in override
    # shoulder_pan pinned to full range
    assert override["shoulder_pan.pos"] == (-60.0, 60.0)
    # elbow_flex narrowed to the stage-2 curriculum range [60, 80]
    assert override["elbow_flex.pos"] == (60.0, 80.0)

    # Hardware.goto should have been called on shoulder_pan (the
    # primary-pin in stage 2 verify) at some point before the collect.
    shoulder_gotos = [c for c in hw.goto_calls if c[0] == "shoulder_pan"]
    assert len(shoulder_gotos) >= 3  # at least one per verify probe
