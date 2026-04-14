"""Unit tests for the pure-function parts of `learner.verifier`.

`verify_batch` itself spawns a recorder subprocess and isn't exercised
here — that's covered indirectly by `test_state_machine` and
`test_curriculum_transition` via their `fake_verify_batch` stubs. The
logic under test below is the probe-script planner, which chooses
`(start_pos, direction)` tuples from the rolling window using the
error-weighted explorer selectors.
"""

from types import SimpleNamespace
from pathlib import Path

import numpy as np

from learner.metrics import ProbeResult, RollingWindow
from learner.verifier import _ACTION_TO_DIRECTION, _plan_probe_script, quantize_motor
from learner.range_tracker import CurriculumState


def _make_cfg(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        robot=SimpleNamespace(
            control_joint="shoulder_pan",
            joint_min=-60.0, joint_max=60.0,
        ),
        actions=SimpleNamespace(candidates=[1, 2, 3]),
        paths=SimpleNamespace(canvas_robot_control=None),
    )


def _make_curriculum_stage1() -> CurriculumState:
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


def test_plan_probe_script_length_and_shape(tmp_path):
    cfg = _make_cfg(tmp_path)
    curriculum = _make_curriculum_stage1()
    window = RollingWindow(size=24)
    scripts = _plan_probe_script(cfg, window, curriculum, num_probes=6)
    assert len(scripts) == 6
    for pos, direction in scripts:
        assert isinstance(pos, float)
        assert direction in ("positive", "negative", "none")
        # Within curriculum active range for the primary joint
        lo, hi = curriculum.active_range
        assert lo <= pos <= hi


def test_plan_probe_script_directions_come_from_action_candidates(tmp_path):
    cfg = _make_cfg(tmp_path)
    curriculum = _make_curriculum_stage1()
    window = RollingWindow(size=24)
    scripts = _plan_probe_script(cfg, window, curriculum, num_probes=30)
    directions = {d for _, d in scripts}
    # Every direction produced must be a valid single_action direction.
    assert directions <= {"positive", "negative", "none"}
    # The mapping covers every action candidate in [1, 2, 3].
    assert set(_ACTION_TO_DIRECTION.values()) == {"positive", "negative", "none"}


def test_plan_probe_script_empty_when_zero_probes(tmp_path):
    cfg = _make_cfg(tmp_path)
    curriculum = _make_curriculum_stage1()
    window = RollingWindow(size=24)
    assert _plan_probe_script(cfg, window, curriculum, num_probes=0) == []


def test_quantize_motor_stable():
    a = np.array([0.1, 4.0, -3.0, 11.0, -13.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 3.5, -4.0, 10.8, -12.9, 0.4], dtype=np.float32)
    assert quantize_motor(a) == quantize_motor(b)
    assert quantize_motor(a) == "0,0,0,1,-1,0"
