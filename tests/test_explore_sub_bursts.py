"""Unit tests for plan_explore_sub_bursts."""

from learner.explorer import plan_explore_sub_bursts
from learner.metrics import ProbeResult, RollingWindow


def _ps(mse, shoulder_pan, action=1):
    return ProbeResult(
        state_key="0",
        action=action,
        mse=mse,
        timestamp=0.0,
        motor_state=(shoulder_pan, 0.0, 0.0, 0.0, 0.0, 0.0),
    )


def test_empty_window_returns_single_uniform_sub_burst():
    w = RollingWindow(size=10)
    out = plan_explore_sub_bursts(
        w, (-20.0, 20.0), control_joint_idx=0, total_episodes=50,
        max_sub_bursts=3, min_sub_burst_size=10,
    )
    assert out == [(50, (-20.0, 20.0))]


def test_no_in_range_probes_falls_back_to_uniform():
    w = RollingWindow(size=10)
    # Probes are out of range
    w.add(_ps(0.5, -55.0))
    w.add(_ps(0.5, 50.0))
    out = plan_explore_sub_bursts(
        w, (-20.0, 20.0), control_joint_idx=0, total_episodes=30,
    )
    assert out == [(30, (-20.0, 20.0))]


def test_max_sub_bursts_one_is_single_burst():
    w = RollingWindow(size=10)
    w.add(_ps(0.3, 5.0))
    out = plan_explore_sub_bursts(
        w, (-20.0, 20.0), control_joint_idx=0, total_episodes=50,
        max_sub_bursts=1, min_sub_burst_size=5,
    )
    assert len(out) == 1
    assert out[0] == (50, (-20.0, 20.0))


def test_hot_bin_gets_biggest_allocation():
    # Populate every bin with small MSE + one bin with high MSE so that
    # we test exploitation, not coverage-first (unvisited bins otherwise
    # get 1.5x the max seen MSE as their prior — see sub-burst planner).
    # n_bins=10 over range 40 → bin width 4. Use n_bins=4 here for bigger
    # bins so population is feasible with few samples.
    w = RollingWindow(size=40)
    # Bins (n_bins=4): [-20,-10), [-10,0), [0,10), [10,20]
    for _ in range(4):
        w.add(_ps(0.05, -15.0))  # bin 0
        w.add(_ps(0.05, -5.0))   # bin 1
        w.add(_ps(0.50, 5.0))    # bin 2 — HOT
        w.add(_ps(0.05, 15.0))   # bin 3
    out = plan_explore_sub_bursts(
        w, (-20.0, 20.0), control_joint_idx=0, total_episodes=100,
        max_sub_bursts=3, min_sub_burst_size=10, n_bins=4,
    )
    total = sum(n for n, _ in out)
    assert total == 100
    # Biggest sub-burst should be the hot bin [0, 10).
    biggest = max(out, key=lambda x: x[0])
    biggest_lo, biggest_hi = biggest[1]
    assert 0.0 <= biggest_lo < biggest_hi <= 10.0, f"hot bin not biggest: {out}"


def test_episode_sum_matches_total_with_floor():
    w = RollingWindow(size=20)
    for _ in range(5):
        w.add(_ps(0.4, 5.0))
        w.add(_ps(0.3, -5.0))
        w.add(_ps(0.2, 15.0))
    out = plan_explore_sub_bursts(
        w, (-20.0, 20.0), control_joint_idx=0, total_episodes=50,
        max_sub_bursts=3, min_sub_burst_size=10, n_bins=8,
    )
    total = sum(n for n, _ in out)
    assert total == 50
    for n, _ in out:
        assert n >= 10


def test_sub_bursts_never_exceed_active_range():
    w = RollingWindow(size=10)
    for _ in range(5):
        w.add(_ps(0.5, 3.0))
        w.add(_ps(0.3, -8.0))
    out = plan_explore_sub_bursts(
        w, (-20.0, 20.0), control_joint_idx=0, total_episodes=60,
        max_sub_bursts=3, min_sub_burst_size=10,
    )
    for _, (lo, hi) in out:
        assert -20.0 <= lo < hi <= 20.0


def test_total_equal_to_floor_returns_single_burst():
    """Regression: when total_episodes == min_sub_burst_size, the planner
    used to return multiple sub-bursts that summed to MORE than the
    requested budget (negative last-allocation got dropped, leaving the
    floor-pinned survivors over-committed). Cap max_sub_bursts up front
    by the budget/floor ratio so we collapse cleanly to one sub-burst.
    """
    w = RollingWindow(size=20)
    for _ in range(5):
        w.add(_ps(0.5, 5.0))
        w.add(_ps(0.3, -5.0))
        w.add(_ps(0.2, 15.0))
    out = plan_explore_sub_bursts(
        w, (-20.0, 20.0), control_joint_idx=0, total_episodes=20,
        max_sub_bursts=3, min_sub_burst_size=20,
    )
    assert len(out) == 1
    assert sum(n for n, _ in out) == 20


def test_total_one_floor_below_max_subbursts_caps_correctly():
    """total=40, floor=20, max=3 → only 2 sub-bursts of 20 fit."""
    w = RollingWindow(size=20)
    for _ in range(5):
        w.add(_ps(0.5, 5.0))
        w.add(_ps(0.3, -5.0))
        w.add(_ps(0.2, 15.0))
    out = plan_explore_sub_bursts(
        w, (-20.0, 20.0), control_joint_idx=0, total_episodes=40,
        max_sub_bursts=3, min_sub_burst_size=20,
    )
    assert len(out) <= 2
    assert sum(n for n, _ in out) == 40
    for n, _ in out:
        assert n >= 20


def test_small_total_collapses_to_fewer_bursts():
    w = RollingWindow(size=20)
    for _ in range(5):
        w.add(_ps(0.5, 5.0))
        w.add(_ps(0.3, -5.0))
        w.add(_ps(0.2, 15.0))
    # Only 20 eps total + min-burst=10 → at most 2 sub-bursts survive
    out = plan_explore_sub_bursts(
        w, (-20.0, 20.0), control_joint_idx=0, total_episodes=20,
        max_sub_bursts=3, min_sub_burst_size=10,
    )
    assert len(out) <= 2
    assert sum(n for n, _ in out) == 20


def test_min_sub_burst_width_prevents_too_narrow_bins():
    """Regression: the sanity run hit a ValueError "all actions are no-ops"
    when sub-bursts were narrower than 2 × position_delta. Verify the
    min_sub_burst_width param collapses to a single burst when the
    active range is too narrow to support the requested split.
    """
    w = RollingWindow(size=40)
    # n_bins=4 over [-20, 20] = bins of width 10. With position_delta=10
    # each bin needs width >= 20 to allow a valid ±10 step. Only 2 bins fit.
    for _ in range(4):
        w.add(_ps(0.05, -15.0))
        w.add(_ps(0.50, 5.0))
        w.add(_ps(0.05, 15.0))
    out = plan_explore_sub_bursts(
        w, (-20.0, 20.0), control_joint_idx=0, total_episodes=50,
        max_sub_bursts=3, min_sub_burst_size=10, n_bins=4,
        min_sub_burst_width=20.0,
    )
    # 40-unit range / 20-unit min width = 2 bins max → at most 2 sub-bursts
    assert len(out) <= 2
    for _, (lo, hi) in out:
        assert (hi - lo) >= 20.0 - 1e-6, f"sub-burst too narrow: [{lo},{hi}]"
    assert sum(n for n, _ in out) == 50


def test_min_sub_burst_width_collapses_to_single_burst_if_range_too_narrow():
    """Tight active range (e.g. curriculum stage 1 at [-10, 10]) has no
    room for ANY split given position_delta=10 (min width 20). Should
    return exactly one sub-burst spanning the whole range.
    """
    w = RollingWindow(size=20)
    # Populate bins to exercise exploitation path (not fallback)
    for _ in range(5):
        w.add(_ps(0.1, -8.0))
        w.add(_ps(0.5, 0.0))
        w.add(_ps(0.3, 8.0))
    out = plan_explore_sub_bursts(
        w, (-10.0, 10.0), control_joint_idx=0, total_episodes=30,
        max_sub_bursts=3, min_sub_burst_size=5, n_bins=10,
        min_sub_burst_width=20.0,
    )
    assert out == [(30, (-10.0, 10.0))]


def test_min_sub_burst_width_zero_behaves_as_before():
    # Without the width constraint (the default), existing behavior holds.
    w = RollingWindow(size=20)
    for _ in range(5):
        w.add(_ps(0.5, 0.0))
    out = plan_explore_sub_bursts(
        w, (-10.0, 10.0), control_joint_idx=0, total_episodes=30,
        max_sub_bursts=3, min_sub_burst_size=5, n_bins=4,
    )
    # With width=0 we're allowed to split into narrower bins
    assert sum(n for n, _ in out) == 30
