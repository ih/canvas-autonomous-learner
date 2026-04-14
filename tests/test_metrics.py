"""Rolling window behavior: mean, eviction, per-action mean."""

from learner.metrics import ProbeResult, RollingWindow


def _p(action, mse, t=0.0):
    return ProbeResult(state_key="0", action=action, mse=mse, timestamp=t)


def test_empty_window_mean_is_zero():
    w = RollingWindow(size=4)
    assert w.is_empty()
    assert w.mean() == 0.0


def test_mean_after_adds():
    w = RollingWindow(size=4)
    for mse in (1.0, 2.0, 3.0):
        w.add(_p(1, mse))
    assert w.mean() == 2.0


def test_window_evicts_oldest():
    w = RollingWindow(size=2)
    w.add(_p(1, 10.0))
    w.add(_p(2, 20.0))
    w.add(_p(3, 30.0))
    assert len(w) == 2
    assert w.mean() == 25.0


def test_per_action_mean():
    w = RollingWindow(size=10)
    w.add(_p(1, 0.1))
    w.add(_p(1, 0.3))
    w.add(_p(2, 0.5))
    per = w.per_action_mean()
    assert per[1] == 0.2
    assert per[2] == 0.5


def test_clear():
    w = RollingWindow(size=4)
    w.add(_p(1, 1.0))
    w.clear()
    assert w.is_empty()


def _ps(action, mse, motor_state, t=0.0):
    return ProbeResult(
        state_key="0", action=action, mse=mse, timestamp=t, motor_state=motor_state
    )


def test_mean_in_range_filters_by_joint_position():
    w = RollingWindow(size=10)
    # Joint 0 values: [-25, -10, 5, 15, 40]. Active range [-20, 20] should
    # include -10, 5, 15 (mses 0.2, 0.3, 0.4 → mean 0.3).
    w.add(_ps(1, 0.1, (-25.0, 0, 0, 0, 0, 0)))
    w.add(_ps(2, 0.2, (-10.0, 0, 0, 0, 0, 0)))
    w.add(_ps(1, 0.3, (5.0, 0, 0, 0, 0, 0)))
    w.add(_ps(3, 0.4, (15.0, 0, 0, 0, 0, 0)))
    w.add(_ps(2, 0.9, (40.0, 0, 0, 0, 0, 0)))
    assert abs(w.mean_in_range((-20.0, 20.0), joint_idx=0) - 0.3) < 1e-9
    assert w.count_in_range((-20.0, 20.0), joint_idx=0) == 3


def test_mean_in_range_ignores_probes_without_motor_state():
    w = RollingWindow(size=10)
    w.add(_p(1, 0.5))  # no motor_state at all
    w.add(_ps(2, 0.1, (0.0, 0, 0, 0, 0, 0)))
    assert abs(w.mean_in_range((-10.0, 10.0), joint_idx=0) - 0.1) < 1e-9
    assert w.count_in_range((-10.0, 10.0), joint_idx=0) == 1


def test_mean_in_range_empty_returns_zero():
    w = RollingWindow(size=4)
    assert w.mean_in_range((-10.0, 10.0), joint_idx=0) == 0.0
    assert w.count_in_range((-10.0, 10.0), joint_idx=0) == 0


def test_probe_result_motor_state_optional():
    # Backward compat: existing code paths construct ProbeResult without
    # motor_state and should continue to work.
    p = ProbeResult(state_key="0", action=1, mse=0.1, timestamp=0.0)
    assert p.motor_state is None
