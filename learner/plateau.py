"""Plateau detection for the comparison experiment's cold-start loop.

The rule: require at least `min_cycles` completed retrain cycles, then
declare plateau if the ratio `min(last 3 locked_val_mse) / max(last 3)`
exceeds `threshold` — i.e., the last three measurements are within
`(1 - threshold) * 100 %` of each other. Default 0.95 → within 5%.

Only accepted cycles (cycles where the new checkpoint was promoted) count
toward the window: rejected cycles reuse the previous checkpoint's locked
val MSE, which would artificially inflate the plateau signal.
"""

from __future__ import annotations

from typing import Iterable


def plateau_reached(
    history: Iterable[dict],
    min_cycles: int = 5,
    window: int = 3,
    threshold: float = 0.95,
) -> bool:
    """Return True if the locked-val trajectory has plateaued.

    `history` is a list of dicts shaped `{cycle, total_eps, locked_val_mse,
    train_val_mse, accepted}` — the same shape `Registry.append_locked_val`
    writes. Only entries with `accepted=True` and a non-None
    `locked_val_mse` contribute to the plateau window.
    """
    accepted = [
        h for h in history
        if h.get("accepted") and h.get("locked_val_mse") is not None
    ]
    if len(accepted) < min_cycles:
        return False
    last = accepted[-window:]
    if len(last) < window:
        return False
    values = [float(h["locked_val_mse"]) for h in last]
    lo = min(values)
    hi = max(values)
    if hi <= 0:
        return True  # degenerate; treat as plateau to avoid infinite loop
    return (lo / hi) > threshold
