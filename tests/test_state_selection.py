"""Unit tests for pick_probe_state (VERIFY error-driven state selection)."""

import random

from learner.explorer import pick_probe_state
from learner.metrics import ProbeResult, RollingWindow


def _ps(action, mse, shoulder_pan):
    return ProbeResult(
        state_key="0",
        action=action,
        mse=mse,
        timestamp=0.0,
        motor_state=(shoulder_pan, 0.0, 0.0, 0.0, 0.0, 0.0),
    )


def test_empty_window_samples_uniformly_in_range():
    w = RollingWindow(size=10)
    rng = random.Random(0)
    for _ in range(200):
        pos = pick_probe_state(w, (-20.0, 20.0), control_joint_idx=0, rng=rng)
        assert -20.0 <= pos <= 20.0


def test_hot_bin_attracts_samples():
    # Populate EVERY bin so we test the exploitation phase (unvisited bins
    # would otherwise dominate via the coverage-first 1.5x prior — see
    # test_unvisited_bins_get_high_priority below for that behavior).
    w = RollingWindow(size=40)
    # n_bins=4 over [-20, 20] → bins at [-20,-10), [-10,0), [0,10), [10,20].
    # Put one clearly-hottest probe in bin 3 and cool probes elsewhere.
    for _ in range(5):
        w.add(_ps(1, 0.002, -15.0))  # bin 0
        w.add(_ps(1, 0.002, -5.0))   # bin 1
        w.add(_ps(1, 0.002, 5.0))    # bin 2
        w.add(_ps(1, 0.500, 15.0))   # bin 3 — HOT

    rng = random.Random(42)
    samples = [
        pick_probe_state(w, (-20.0, 20.0), control_joint_idx=0, n_bins=4, rng=rng)
        for _ in range(2000)
    ]
    hot_bin_count = sum(1 for s in samples if 10.0 <= s <= 20.0)
    # Weight of hot bin = 0.5 / (0.002*3 + 0.5) = ~0.988
    assert hot_bin_count > 1800, f"expected hot-bin dominance, got {hot_bin_count}/2000"


def test_out_of_range_probes_ignored():
    w = RollingWindow(size=10)
    # A probe far outside the active range with massive MSE must not
    # affect sampling inside [-20, 20].
    w.add(_ps(1, 99.0, 55.0))
    w.add(_ps(1, 0.01, 0.0))
    rng = random.Random(1)
    samples = [
        pick_probe_state(w, (-20.0, 20.0), control_joint_idx=0, rng=rng)
        for _ in range(100)
    ]
    for s in samples:
        assert -20.0 <= s <= 20.0


def test_unvisited_bins_get_high_priority():
    w = RollingWindow(size=10)
    # Three probes all in a single bin (say 0-5), leaving the rest empty.
    for _ in range(3):
        w.add(_ps(1, 0.001, 2.5))
    rng = random.Random(7)
    samples = [
        pick_probe_state(w, (-20.0, 20.0), control_joint_idx=0, n_bins=4, rng=rng)
        for _ in range(500)
    ]
    # Bin 2 (index 2, range 0-10) has the visited probes with low MSE.
    # The other 3 bins are unvisited → max_seen × 1.5 >> 0.001. So most
    # samples should land OUTSIDE the visited bin.
    in_visited = sum(1 for s in samples if 0.0 <= s <= 10.0)
    assert in_visited < 150, f"expected coverage push, got {in_visited}/500 in visited bin"


def test_degenerate_range_returns_lo():
    w = RollingWindow(size=4)
    assert pick_probe_state(w, (10.0, 10.0), control_joint_idx=0) == 10.0
    assert pick_probe_state(w, (10.0, 5.0), control_joint_idx=0) == 10.0


def test_probe_without_motor_state_is_ignored():
    w = RollingWindow(size=10)
    # Legacy probe w/o motor_state + one real probe
    w.add(ProbeResult(state_key="0", action=1, mse=0.5, timestamp=0.0))
    w.add(_ps(1, 0.01, 5.0))
    rng = random.Random(0)
    for _ in range(50):
        pos = pick_probe_state(w, (-10.0, 10.0), control_joint_idx=0, rng=rng)
        assert -10.0 <= pos <= 10.0
