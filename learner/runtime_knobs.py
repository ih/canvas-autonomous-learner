"""Mutable runtime knobs the THINK phase can patch mid-run.

The orchestrator's `main_loop` used to hoist every scalar threshold /
cadence value into a local variable at function entry. That made them
unreachable once the loop was running, so a live parameter change required
a full restart. `RuntimeKnobs` packages them into a single dataclass the
Claude advisor can mutate in place via `apply_overrides` — the loop
references `knobs.tau_low` (etc) everywhere and picks up the change on
the very next branch.

Clamping is built in: obviously-bad values (non-positive taus, zero
counts, negative timers) are replaced with minima before being applied,
and a `claude_override_clamped` event is logged.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


# Field-level clamps. Each entry: (min, max) — either bound may be None.
_CLAMPS: dict[str, tuple[float | None, float | None]] = {
    "tau_low": (1e-6, None),
    "tau_high": (1e-6, None),
    "val_guard": (1.0, None),
    "max_consecutive_rejections": (1, None),
    "probes_per_verify": (1, None),
    "window_size": (1, None),
    "settle_time": (0.0, None),
    "base_burst": (1, None),
    "burst_min": (1, None),
    "burst_max": (1, None),
    "max_sub_bursts": (1, None),
    "min_sub_burst_size": (1, None),
    "safety_cap": (1, None),
    "warmup_cycles": (0, None),
    "explore_max_retries": (0, None),
    "explore_retry_backoff": (0.0, None),
}


@dataclass
class RuntimeKnobs:
    """Every scalar the main loop references after startup.

    Construct once with `from_cfg(cfg)`, then reference `knobs.<field>`
    everywhere — no more hoisted locals.
    """
    tau_low: float
    tau_high: float
    val_guard: float
    max_consecutive_rejections: int
    probes_per_verify: int
    window_size: int
    settle_time: float
    base_burst: int
    burst_min: int
    burst_max: int
    max_sub_bursts: int
    min_sub_burst_size: int
    safety_cap: int
    warmup_cycles: int
    explore_max_retries: int
    explore_retry_backoff: float

    @classmethod
    def from_cfg(cls, cfg) -> "RuntimeKnobs":
        return cls(
            tau_low=float(cfg.thresholds.tau_low),
            tau_high=float(cfg.thresholds.tau_high),
            val_guard=float(cfg.thresholds.val_guard),
            max_consecutive_rejections=int(
                getattr(cfg.thresholds, "max_consecutive_rejections", 3)
            ),
            probes_per_verify=int(getattr(cfg.cadence, "probes_per_verify", 4)),
            window_size=int(getattr(cfg.cadence, "window_size", 16)),
            settle_time=float(getattr(cfg.cadence, "settle_time", 0.5)),
            base_burst=int(getattr(cfg.cadence, "base_explore_batch_size", 50)),
            burst_min=int(getattr(cfg.cadence, "explore_batch_size_min", 10)),
            burst_max=int(getattr(cfg.cadence, "explore_batch_size_max", 150)),
            max_sub_bursts=int(getattr(cfg.cadence, "max_sub_bursts", 3)),
            min_sub_burst_size=int(getattr(cfg.cadence, "min_sub_burst_size", 10)),
            safety_cap=int(getattr(cfg.cadence, "safety_cap_episodes", 10**9)),
            warmup_cycles=int(getattr(cfg.cadence, "warmup_cycles", 2)),
            explore_max_retries=int(getattr(cfg.cadence, "explore_max_retries", 2)),
            explore_retry_backoff=float(
                getattr(cfg.cadence, "explore_retry_backoff", 10.0)
            ),
        )

    def apply_overrides(
        self, overrides: dict[str, Any], event_log=None,
    ) -> dict[str, Any]:
        """Apply a flat dict of field→value overrides in place.

        Field keys can be bare (`tau_low`) or prefixed (`knobs.tau_low`).
        Unknown keys and uncoercible values are dropped and logged. Values
        are clamped to sensible minima. Returns the dict of what was
        actually applied.
        """
        if not overrides:
            return {}
        known = {f.name for f in fields(self)}
        applied: dict[str, Any] = {}
        for raw_key, raw_value in overrides.items():
            key = str(raw_key).removeprefix("knobs.")
            if key not in known:
                if event_log is not None:
                    event_log.log(
                        "claude_override_unknown", target="knobs", key=raw_key,
                    )
                continue
            current = getattr(self, key)
            try:
                if isinstance(current, int) and not isinstance(current, bool):
                    value: Any = int(raw_value)
                elif isinstance(current, float):
                    value = float(raw_value)
                else:
                    value = raw_value
            except (TypeError, ValueError):
                if event_log is not None:
                    event_log.log(
                        "claude_override_uncoercible",
                        target="knobs", key=key, value=raw_value,
                    )
                continue
            clamp = _CLAMPS.get(key)
            if clamp is not None:
                lo, hi = clamp
                if lo is not None and value < lo:
                    if event_log is not None:
                        event_log.log(
                            "claude_override_clamped",
                            target="knobs", key=key,
                            requested=value, clamped_to=lo,
                        )
                    value = lo
                if hi is not None and value > hi:
                    if event_log is not None:
                        event_log.log(
                            "claude_override_clamped",
                            target="knobs", key=key,
                            requested=value, clamped_to=hi,
                        )
                    value = hi
            setattr(self, key, value)
            applied[key] = value
        return applied

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}
