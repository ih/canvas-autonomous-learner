"""Dry-run coverage of Hardware.goto().

We don't want these tests to import the real `control.robot_interface`
(the sibling repo), so they construct a `FakeBus`/`FakeDryRobot` that
mimics the two code paths Hardware.goto cares about: the DryRun branch
(updates `_positions[joint]`) and the real branch (sync_read /
sync_write on a bus).

The tests stub out the Hardware constructor's sibling-repo imports so
no actual hardware or canvas-robot-control code is touched.
"""

import time
from types import SimpleNamespace

import pytest


JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


class _FakeDryRobot:
    def __init__(self):
        self._positions = {j: 0.0 for j in JOINTS}

    def connect(self): pass
    def disconnect(self): pass
    def get_state(self): raise NotImplementedError
    def execute_action(self, a): raise NotImplementedError


class _FakeBus:
    """Records every sync_write, and reports Present_Position tracking
    the most recent goal (simulating an instantly-reaching motor)."""

    def __init__(self):
        self.positions = {j: 0.0 for j in JOINTS}
        self.writes: list[dict] = []

    def sync_read(self, key):
        assert key == "Present_Position"
        return dict(self.positions)

    def sync_write(self, key, goal):
        assert key == "Goal_Position"
        self.writes.append(dict(goal))
        # Simulate instantaneous motion for test determinism.
        for j, v in goal.items():
            self.positions[j] = v


class _FakeRealRobot:
    def __init__(self):
        self.bus = _FakeBus()

    def connect(self): pass
    def disconnect(self): pass


def _make_hardware(dry_run: bool):
    """Build a Hardware instance without running its real constructor.

    The constructor does sibling-repo imports and instantiates
    `RobotInterface` or `DryRunRobotInterface` — we want neither in a
    unit test. So we allocate an empty shell and populate the fields
    Hardware.goto actually touches: `cfg`, `robot`, `dry_run`, `_JOINTS`.
    """
    from learner.hardware import Hardware  # local import, picks up Edits

    hw = Hardware.__new__(Hardware)
    hw.cfg = SimpleNamespace(
        robot=SimpleNamespace(
            joint_min=-60.0, joint_max=60.0, control_joint="shoulder_pan",
        ),
    )
    hw._JOINTS = JOINTS
    hw.dry_run = dry_run
    hw.robot = _FakeDryRobot() if dry_run else _FakeRealRobot()
    hw.predictor = None
    return hw


def test_goto_dry_run_sets_position_instantly():
    hw = _make_hardware(dry_run=True)
    hw.goto("shoulder_pan", 15.0, settle_time=0.0)
    assert hw.robot._positions["shoulder_pan"] == 15.0
    # Other joints untouched
    assert hw.robot._positions["elbow_flex"] == 0.0


def test_goto_dry_run_clamps_to_joint_bounds():
    hw = _make_hardware(dry_run=True)
    hw.goto("shoulder_pan", 9999.0, settle_time=0.0)
    assert hw.robot._positions["shoulder_pan"] == 60.0  # joint_max
    hw.goto("shoulder_pan", -9999.0, settle_time=0.0)
    assert hw.robot._positions["shoulder_pan"] == -60.0  # joint_min


def test_goto_real_writes_full_goal_and_holds_other_joints():
    hw = _make_hardware(dry_run=False)
    # Seed the bus so other joints have distinct positions to verify
    # they're held unchanged in the goal dict.
    hw.robot.bus.positions = {
        "shoulder_pan": 5.0,
        "shoulder_lift": -42.0,
        "elbow_flex": 63.0,
        "wrist_flex": 11.0,
        "wrist_roll": -7.0,
        "gripper": 0.5,
    }
    hw.goto("shoulder_pan", -25.0, settle_time=0.0, timeout=0.1)
    assert len(hw.robot.bus.writes) == 1
    goal = hw.robot.bus.writes[0]
    assert goal["shoulder_pan"] == -25.0
    assert goal["shoulder_lift"] == -42.0
    assert goal["elbow_flex"] == 63.0
    assert goal["wrist_flex"] == 11.0


def test_goto_real_clamps_to_joint_bounds():
    hw = _make_hardware(dry_run=False)
    hw.goto("shoulder_pan", 9999.0, settle_time=0.0, timeout=0.1)
    assert hw.robot.bus.writes[-1]["shoulder_pan"] == 60.0


def test_goto_real_polls_until_within_tolerance():
    hw = _make_hardware(dry_run=False)
    # FakeBus reaches goal instantly, so the poll loop should exit on
    # the first iteration and not hit the timeout.
    t0 = time.perf_counter()
    hw.goto("shoulder_pan", 10.0, settle_time=0.0, tolerance=2.0, timeout=2.0)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, f"goto took {elapsed:.2f}s, expected instant"
    assert hw.robot.bus.positions["shoulder_pan"] == 10.0
