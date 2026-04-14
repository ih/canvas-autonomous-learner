"""Unit tests for RangeTracker (curriculum state-space expansion)."""

import pytest

from learner.range_tracker import RangeTracker


def _make(initial_half_width=20.0, full_min=-60.0, full_max=60.0,
          expansion_factor=1.5, stable_cycles_required=2):
    return RangeTracker.from_config(
        control_joint="shoulder_pan",
        control_joint_idx=0,
        full_min=full_min,
        full_max=full_max,
        initial_half_width=initial_half_width,
        expansion_factor=expansion_factor,
        stable_cycles_required=stable_cycles_required,
    )


def test_from_config_starts_symmetric_around_center():
    r = _make(initial_half_width=20.0, full_min=-60.0, full_max=60.0)
    assert r.active == (-20.0, 20.0)
    assert not r.at_max()
    assert r.stable_cycles == 0


def test_from_config_clamps_half_width_to_full_range():
    r = _make(initial_half_width=9999.0, full_min=-60.0, full_max=60.0)
    assert r.active == (-60.0, 60.0)
    assert r.at_max()


def test_good_cycle_accumulates_stable_cycles():
    r = _make(stable_cycles_required=3)
    r.good_cycle(); r.good_cycle()
    assert r.stable_cycles == 2
    assert not r.stable()
    r.good_cycle()
    assert r.stable()
    assert r.should_expand()


def test_bad_cycle_resets_stable_cycles():
    r = _make(stable_cycles_required=2)
    r.good_cycle()
    r.good_cycle()
    assert r.should_expand()
    r.bad_cycle()
    assert r.stable_cycles == 0
    assert not r.should_expand()


def test_expand_widens_and_resets_stable_cycles():
    r = _make(initial_half_width=20.0, expansion_factor=1.5)
    r.good_cycle(); r.good_cycle()
    new_range = r.expand(cycle=3, total_eps=200)
    assert new_range == (-30.0, 30.0)
    assert r.active == (-30.0, 30.0)
    assert r.stable_cycles == 0
    assert len(r.history) == 1
    h = r.history[0]
    assert h["cycle"] == 3
    assert h["total_eps"] == 200
    assert h["old_range"] == [-20.0, 20.0]
    assert h["new_range"] == [-30.0, 30.0]


def test_expand_clamps_to_full_range():
    r = _make(initial_half_width=45.0, expansion_factor=1.5, full_min=-60.0, full_max=60.0)
    # half=45 × 1.5 = 67.5 → clamp to 60
    r.expand(cycle=1, total_eps=50)
    assert r.active == (-60.0, 60.0)
    assert r.at_max()


def test_expand_is_no_op_at_max():
    r = _make(initial_half_width=60.0, full_min=-60.0, full_max=60.0)
    assert r.at_max()
    before = r.active
    out = r.expand(cycle=99, total_eps=999)
    assert out == before
    assert r.active == before
    assert r.history == []    # no entry recorded


def test_should_expand_false_at_max_even_when_stable():
    r = _make(initial_half_width=60.0, stable_cycles_required=1)
    r.good_cycle()
    assert r.stable()
    assert r.at_max()
    assert not r.should_expand()


def test_to_registry_state_roundtrip():
    r = _make()
    r.good_cycle()
    r.expand(cycle=5, total_eps=250)
    r.good_cycle()
    snapshot = r.to_registry_state()
    assert snapshot["range_active"] == [-30.0, 30.0]
    assert snapshot["range_stable_cycles"] == 1
    assert len(snapshot["range_history"]) == 1

    # Restore from the snapshot
    restored = RangeTracker.from_config_or_registry(
        control_joint="shoulder_pan",
        control_joint_idx=0,
        full_min=-60.0,
        full_max=60.0,
        initial_half_width=20.0,
        expansion_factor=1.5,
        stable_cycles_required=2,
        registry_state=snapshot,
    )
    assert restored.active == (-30.0, 30.0)
    assert restored.stable_cycles == 1
    assert len(restored.history) == 1


def test_from_registry_state_ignores_invalid_saved_range():
    # lo > hi is malformed; fall back to initial
    snapshot = {"range_active": [10.0, 5.0]}
    r = RangeTracker.from_config_or_registry(
        control_joint="shoulder_pan",
        control_joint_idx=0,
        full_min=-60.0, full_max=60.0,
        initial_half_width=20.0, expansion_factor=1.5,
        stable_cycles_required=2,
        registry_state=snapshot,
    )
    assert r.active == (-20.0, 20.0)
