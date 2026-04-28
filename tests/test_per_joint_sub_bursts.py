"""Unit tests for plan_per_joint_sub_bursts (per-cell error-driven EXPLORE)."""

from learner.explorer import plan_per_joint_sub_bursts


# Helpers
_FULL_RANGES = {
    "shoulder_pan": (-60.0, 60.0),
    "elbow_flex": (50.0, 90.0),
}


def _cell(bin_idx: int, lo: float, hi: float, mean_mse: float, count: int = 5):
    return {"bin": bin_idx, "lo": lo, "hi": hi, "mean_mse": mean_mse, "count": count}


def test_empty_histogram_falls_back_to_uniform_per_joint():
    """No measured cells → each pool joint gets a sub-burst on its full range."""
    out = plan_per_joint_sub_bursts(
        per_cell_mse=None,
        joint_pool=["shoulder_pan", "elbow_flex"],
        joint_ranges=_FULL_RANGES,
        total_episodes=40,
        max_sub_bursts=3,
        min_sub_burst_size=10,
    )
    joints = sorted(j for _, j, _ in out)
    assert joints == ["elbow_flex", "shoulder_pan"]
    assert sum(n for n, _, _ in out) == 40


def test_hot_cell_gets_largest_allocation():
    """A clearly hot cell on one joint should attract more episodes than a
    less-hot cell on another joint."""
    per_cell = {
        "shoulder_pan": [
            _cell(2, -20.0, 0.0, mean_mse=0.30),
            _cell(3, 0.0, 20.0, mean_mse=0.05),
        ],
        "elbow_flex": [
            _cell(0, 50.0, 60.0, mean_mse=0.10),
        ],
    }
    out = plan_per_joint_sub_bursts(
        per_cell_mse=per_cell,
        joint_pool=["shoulder_pan", "elbow_flex"],
        joint_ranges=_FULL_RANGES,
        total_episodes=60,
        max_sub_bursts=3,
        min_sub_burst_size=10,
    )
    by_score = sorted(out, key=lambda r: -r[0])
    # The hottest cell (shoulder_pan, 0.30) gets the biggest burst.
    assert by_score[0][1] == "shoulder_pan"
    assert by_score[0][2] == (-20.0, 0.0)
    assert sum(n for n, _, _ in out) == 60


def test_unvisited_joint_gets_exploration_bonus():
    """A joint with measured cells alongside one without should still see
    the unvisited joint targeted (via the max_observed * bonus heuristic)."""
    per_cell = {
        "shoulder_pan": [_cell(0, -60.0, -40.0, mean_mse=0.20)],
        # elbow_flex absent → unvisited
    }
    out = plan_per_joint_sub_bursts(
        per_cell_mse=per_cell,
        joint_pool=["shoulder_pan", "elbow_flex"],
        joint_ranges=_FULL_RANGES,
        total_episodes=60,
        max_sub_bursts=3,
        min_sub_burst_size=10,
        unvisited_bonus=1.5,  # 0.20 * 1.5 = 0.30, beats observed 0.20
    )
    joints = [j for _, j, _ in out]
    assert "elbow_flex" in joints
    # Elbow gets the largest allocation because it's the highest-scored.
    by_score = sorted(out, key=lambda r: -r[0])
    assert by_score[0][1] == "elbow_flex"


def test_pool_filter_excludes_unwanted_joints():
    """A joint with cells but not in the pool should be ignored."""
    per_cell = {
        "shoulder_pan": [_cell(0, -60.0, -40.0, mean_mse=0.30)],
        "wrist_flex": [_cell(0, -30.0, -10.0, mean_mse=0.50)],
    }
    out = plan_per_joint_sub_bursts(
        per_cell_mse=per_cell,
        joint_pool=["shoulder_pan", "elbow_flex"],
        joint_ranges=_FULL_RANGES,
        total_episodes=40,
        max_sub_bursts=3,
        min_sub_burst_size=10,
    )
    for _, j, _ in out:
        assert j in ("shoulder_pan", "elbow_flex")


def test_total_equal_to_floor_returns_single_burst():
    """Regression: same edge case as the 1D planner — total==floor must
    not over-commit to multiple sub-bursts."""
    per_cell = {
        "shoulder_pan": [
            _cell(0, -60.0, -40.0, 0.30),
            _cell(2, -20.0, 0.0, 0.20),
        ],
    }
    out = plan_per_joint_sub_bursts(
        per_cell_mse=per_cell,
        joint_pool=["shoulder_pan", "elbow_flex"],
        joint_ranges=_FULL_RANGES,
        total_episodes=20,
        max_sub_bursts=3,
        min_sub_burst_size=20,
    )
    assert len(out) == 1
    assert sum(n for n, _, _ in out) == 20


def test_min_sub_burst_width_skips_narrow_cells():
    """Cells narrower than min_sub_burst_width are dropped before allocation."""
    per_cell = {
        "shoulder_pan": [
            _cell(0, -10.0, -5.0, mean_mse=0.50),    # 5 wide — skipped
            _cell(2, -20.0, 0.0, mean_mse=0.10),     # 20 wide — kept
        ],
    }
    out = plan_per_joint_sub_bursts(
        per_cell_mse=per_cell,
        joint_pool=["shoulder_pan", "elbow_flex"],
        joint_ranges=_FULL_RANGES,
        total_episodes=40,
        max_sub_bursts=3,
        min_sub_burst_size=10,
        min_sub_burst_width=20.0,
    )
    # The narrow shoulder cell shouldn't appear; elbow's full range (40
    # wide) should appear via the unvisited-bonus path.
    for _, j, (lo, hi) in out:
        if j == "shoulder_pan":
            assert hi - lo >= 20.0


def test_sum_never_exceeds_total():
    """Across many random budgets, sum(n) <= total_episodes always."""
    import random
    rng = random.Random(7)
    per_cell = {
        "shoulder_pan": [
            _cell(i, -60.0 + i * 12, -60.0 + (i + 1) * 12, mean_mse=rng.random())
            for i in range(10)
        ],
        "elbow_flex": [
            _cell(i, 50.0 + i * 4, 50.0 + (i + 1) * 4, mean_mse=rng.random())
            for i in range(10)
        ],
    }
    for _ in range(50):
        budget = rng.randint(10, 200)
        floor = rng.choice([5, 10, 20, 30])
        max_k = rng.randint(1, 5)
        out = plan_per_joint_sub_bursts(
            per_cell_mse=per_cell,
            joint_pool=["shoulder_pan", "elbow_flex"],
            joint_ranges=_FULL_RANGES,
            total_episodes=budget,
            max_sub_bursts=max_k,
            min_sub_burst_size=floor,
        )
        assert sum(n for n, _, _ in out) <= budget
