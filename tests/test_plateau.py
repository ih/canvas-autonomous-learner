"""Unit tests for the plateau detector."""

from learner.plateau import plateau_reached


def _h(cycle, mse, accepted=True):
    return {
        "cycle": cycle,
        "total_eps": (cycle + 1) * 50,
        "locked_val_mse": mse,
        "train_val_mse": mse,
        "accepted": accepted,
    }


def test_empty_history_no_plateau():
    assert plateau_reached([]) is False


def test_fewer_than_min_cycles_no_plateau():
    history = [_h(i, 0.1 - i * 0.001) for i in range(4)]
    assert plateau_reached(history, min_cycles=5) is False


def test_strictly_decreasing_not_plateau():
    history = [_h(i, 0.1 - i * 0.01) for i in range(6)]
    # last 3: 0.06, 0.05, 0.04 -> ratio 0.04/0.06 = 0.667 < 0.95
    assert plateau_reached(history) is False


def test_flat_is_plateau():
    history = [_h(i, 0.01) for i in range(6)]
    assert plateau_reached(history) is True


def test_within_5_percent_is_plateau():
    # min=0.0096, max=0.0100, ratio=0.96 > 0.95 → plateau
    msesequence = [0.10, 0.05, 0.02, 0.0100, 0.0098, 0.0096]
    history = [_h(i, m) for i, m in enumerate(msesequence)]
    assert plateau_reached(history) is True


def test_outside_5_percent_not_plateau():
    # min=0.008, max=0.010, ratio=0.80 < 0.95 → still improving
    msesequence = [0.10, 0.05, 0.02, 0.010, 0.009, 0.008]
    history = [_h(i, m) for i, m in enumerate(msesequence)]
    assert plateau_reached(history) is False


def test_rejected_cycles_excluded_from_window():
    # 6 accepted-looking cycles, but 2 of the last 3 are actually rejected.
    # After filtering, only 4 cycles count — below min_cycles=5.
    history = [
        _h(0, 0.10), _h(1, 0.05), _h(2, 0.02),
        _h(3, 0.012), _h(4, 0.012, accepted=False), _h(5, 0.012, accepted=False),
    ]
    assert plateau_reached(history) is False


def test_none_locked_val_excluded():
    history = [
        _h(0, 0.10), _h(1, 0.05), _h(2, 0.02),
        _h(3, None), _h(4, 0.012), _h(5, 0.012),
    ]
    # 5 accepted cycles with non-None MSE: 0.10, 0.05, 0.02, 0.012, 0.012.
    # Last 3: 0.02, 0.012, 0.012 -> ratio 0.012/0.02 = 0.6 < 0.95.
    assert plateau_reached(history) is False


def test_noisy_values_not_plateau_if_spread_too_wide():
    # min=0.008, max=0.012, ratio=0.667 → not plateau
    msesequence = [0.10, 0.05, 0.02, 0.012, 0.008, 0.010]
    history = [_h(i, m) for i, m in enumerate(msesequence)]
    assert plateau_reached(history) is False


def test_custom_threshold():
    msesequence = [0.10, 0.05, 0.02, 0.010, 0.009, 0.008]
    history = [_h(i, m) for i, m in enumerate(msesequence)]
    # With threshold=0.75, ratio=0.008/0.010=0.8 > 0.75 → plateau
    assert plateau_reached(history, threshold=0.75) is True
    # With default 0.95 → not plateau
    assert plateau_reached(history) is False
