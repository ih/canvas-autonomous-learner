"""Unit tests for `learner.runtime_knobs.RuntimeKnobs`."""

from types import SimpleNamespace

from learner.runtime_knobs import RuntimeKnobs


def _cfg():
    return SimpleNamespace(
        thresholds=SimpleNamespace(
            tau_low=0.05, tau_high=0.10, val_guard=1.25,
            max_consecutive_rejections=3,
        ),
        cadence=SimpleNamespace(
            probes_per_verify=6, window_size=24, settle_time=0.5,
            base_explore_batch_size=20,
            explore_batch_size_min=20, explore_batch_size_max=20,
            max_sub_bursts=1, min_sub_burst_size=20,
            safety_cap_episodes=1500, warmup_cycles=2,
            explore_max_retries=2, explore_retry_backoff=10.0,
        ),
    )


def test_from_cfg_populates_every_field():
    cfg = _cfg()
    knobs = RuntimeKnobs.from_cfg(cfg)
    assert knobs.tau_low == 0.05
    assert knobs.tau_high == 0.10
    assert knobs.val_guard == 1.25
    assert knobs.max_consecutive_rejections == 3
    assert knobs.probes_per_verify == 6
    assert knobs.window_size == 24
    assert knobs.settle_time == 0.5
    assert knobs.base_burst == 20
    assert knobs.burst_min == 20
    assert knobs.burst_max == 20
    assert knobs.max_sub_bursts == 1
    assert knobs.min_sub_burst_size == 20
    assert knobs.safety_cap == 1500
    assert knobs.warmup_cycles == 2
    assert knobs.explore_max_retries == 2
    assert knobs.explore_retry_backoff == 10.0


def test_apply_overrides_happy_path():
    knobs = RuntimeKnobs.from_cfg(_cfg())
    applied = knobs.apply_overrides({
        "tau_low": 0.04, "tau_high": 0.08,
        "probes_per_verify": 8, "base_burst": 30,
    })
    assert applied == {
        "tau_low": 0.04, "tau_high": 0.08,
        "probes_per_verify": 8, "base_burst": 30,
    }
    assert knobs.tau_low == 0.04
    assert knobs.probes_per_verify == 8


def test_apply_overrides_accepts_knobs_prefix():
    knobs = RuntimeKnobs.from_cfg(_cfg())
    applied = knobs.apply_overrides({"knobs.tau_low": 0.02})
    assert applied == {"tau_low": 0.02}
    assert knobs.tau_low == 0.02


def test_apply_overrides_drops_unknown():
    knobs = RuntimeKnobs.from_cfg(_cfg())
    applied = knobs.apply_overrides({"not_a_knob": 123, "tau_low": 0.03})
    assert applied == {"tau_low": 0.03}


def test_apply_overrides_clamps_non_positive_tau():
    knobs = RuntimeKnobs.from_cfg(_cfg())
    applied = knobs.apply_overrides({"tau_low": 0.0, "tau_high": -0.5})
    # Both are clamped to the positive minimum (1e-6).
    assert applied["tau_low"] > 0
    assert applied["tau_high"] > 0
    assert knobs.tau_low > 0
    assert knobs.tau_high > 0


def test_apply_overrides_coerces_ints():
    knobs = RuntimeKnobs.from_cfg(_cfg())
    # Passing a float where an int is expected should coerce.
    applied = knobs.apply_overrides({"probes_per_verify": 10.0})
    assert applied == {"probes_per_verify": 10}
    assert isinstance(knobs.probes_per_verify, int)


def test_apply_overrides_drops_uncoercible():
    knobs = RuntimeKnobs.from_cfg(_cfg())
    applied = knobs.apply_overrides({"tau_low": "not-a-number"})
    assert applied == {}
    # Unchanged.
    assert knobs.tau_low == 0.05


def test_as_dict_round_trip():
    knobs = RuntimeKnobs.from_cfg(_cfg())
    d = knobs.as_dict()
    assert d["tau_low"] == 0.05
    assert "probes_per_verify" in d
    assert len(d) > 10
