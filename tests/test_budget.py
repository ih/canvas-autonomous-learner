"""Unit tests for dynamic_explore_batch_size."""

from learner.budget import dynamic_explore_batch_size


def test_cold_start_returns_hi():
    assert dynamic_explore_batch_size(
        mean_err=None, tau_high=0.06, base=50, lo=10, hi=150
    ) == 150


def test_at_threshold_returns_base():
    assert dynamic_explore_batch_size(
        mean_err=0.06, tau_high=0.06, base=50, lo=10, hi=150
    ) == 50


def test_below_threshold_still_returns_base():
    # Scale clamps to >= 1.0 → base * 1.0 = base (clamped to lo if base < lo)
    assert dynamic_explore_batch_size(
        mean_err=0.03, tau_high=0.06, base=50, lo=10, hi=150
    ) == 50


def test_two_x_threshold_returns_two_x_base():
    assert dynamic_explore_batch_size(
        mean_err=0.12, tau_high=0.06, base=50, lo=10, hi=150
    ) == 100


def test_three_x_threshold_capped_at_three_x_base_then_clamped_to_hi():
    # base 50 * 3 = 150, exactly at hi
    assert dynamic_explore_batch_size(
        mean_err=0.18, tau_high=0.06, base=50, lo=10, hi=150
    ) == 150


def test_extreme_err_still_clamped_to_hi():
    assert dynamic_explore_batch_size(
        mean_err=10.0, tau_high=0.06, base=50, lo=10, hi=150
    ) == 150


def test_lo_clamp_honored_when_base_too_small():
    # base=3 is below lo=10 → clamp up
    assert dynamic_explore_batch_size(
        mean_err=0.06, tau_high=0.06, base=3, lo=10, hi=150
    ) == 10


def test_hi_clamp_honored_when_base_too_large():
    # base=200 is above hi=150 → clamp down even at scale 1.0
    assert dynamic_explore_batch_size(
        mean_err=0.06, tau_high=0.06, base=200, lo=10, hi=150
    ) == 150


def test_degenerate_tau_high_zero_returns_hi():
    # When tau_high is 0 or negative, we can't compute a ratio. Treat as
    # "always bad" → max burst.
    assert dynamic_explore_batch_size(
        mean_err=0.5, tau_high=0.0, base=50, lo=10, hi=150
    ) == 150


def test_returns_int():
    out = dynamic_explore_batch_size(
        mean_err=0.08, tau_high=0.06, base=50, lo=10, hi=150
    )
    assert isinstance(out, int)
