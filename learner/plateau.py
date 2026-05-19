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


def plateau_summary(
    locked_val_history: Iterable[dict],
    verify_history: Iterable[dict] | None = None,
    min_cycles: int = 5,
    window: int = 3,
    threshold: float = 0.95,
) -> dict:
    """Structured plateau diagnostic for the advisor's snapshot.

    Unlike `plateau_reached` (a bool used for auto-termination), this
    returns a dict the advisor can interpret along with other signals:

        {
          "locked_val_plateau": bool | None,    # None = not enough data
          "verify_plateau":     bool | None,
          "locked_val_recent":  [float, ...],   # last `window` values
          "verify_recent":      [float, ...],
          "cycles_evaluated":   int,
          "verdict":            "stuck" | "improving" | "insufficient_data"
                                | "plateau_low_locked_val"  (acceptable plateau)
                                | "plateau_high_locked_val" (need new data)
        }

    The advisor's prompt uses `verdict` to decide:
      - `stuck` / `plateau_high_locked_val` → route to explore (widen
        ranges) or idle (request scene change).
      - `plateau_low_locked_val` → fine to stay in VERIFY; the model has
        learned the current scene.
      - `improving` → keep doing what's working.
      - `insufficient_data` → too early to call.

    Both histories are optional. `locked_val_history` matches
    `Registry.append_locked_val` shape; `verify_history` is a list of
    `verify_summary` event dicts (or anything carrying `mean_err`).
    """
    locked_accepted = [
        h for h in (locked_val_history or [])
        if h.get("accepted") and h.get("locked_val_mse") is not None
    ]
    locked_recent = [
        float(h["locked_val_mse"]) for h in locked_accepted[-window:]
    ]
    verify_recent = [
        float(v.get("mean_err"))
        for v in (verify_history or [])[-window:]
        if v.get("mean_err") is not None
    ]

    def _is_plateau(values: list[float]) -> bool | None:
        if len(values) < window:
            return None
        lo = min(values)
        hi = max(values)
        if hi <= 0:
            return True
        return (lo / hi) > threshold

    locked_plateau = _is_plateau(locked_recent)
    verify_plateau = _is_plateau(verify_recent)

    # Verdict prefers the verify signal (collected-data, lifelong-aligned)
    # and uses locked-val as the tiebreaker described in the docstring.
    if len(locked_accepted) < min_cycles and verify_plateau is None:
        verdict = "insufficient_data"
    elif verify_plateau is False:
        # Verify mean_err still moving — model is genuinely improving
        # against fresh data, regardless of locked-val.
        verdict = "improving"
    elif verify_plateau is True:
        # Probes have flatlined. Locked-val tells us if the flat is good
        # by comparing current vs the first-ever measurement: if the
        # model has dropped its locked-val substantially over the run,
        # the plateau is at a "good" place; if locked-val never really
        # improved, the plateau is "high" and the model is stuck.
        if locked_accepted and len(locked_accepted) >= 2:
            first_lv = float(locked_accepted[0]["locked_val_mse"])
            last_lv = float(locked_accepted[-1]["locked_val_mse"])
            if first_lv > 0 and last_lv < 0.5 * first_lv:
                verdict = "plateau_low_locked_val"
            else:
                verdict = "plateau_high_locked_val"
        else:
            # No locked-val signal at all — caller is in pure-lifelong
            # mode without a held-out set. A plateau in mean_err alone
            # isn't enough to declare "stuck"; the advisor still has
            # train/val curves and novelty to consult.
            verdict = "stuck"
    elif locked_plateau is True:
        # Verify hasn't plateaued (or no data) but locked-val has —
        # likely scene-shift compensating across retrains. Treat as
        # ambiguous, lean toward "stuck" so the advisor can act.
        verdict = "stuck"
    else:
        verdict = "insufficient_data"

    return {
        "locked_val_plateau": locked_plateau,
        "verify_plateau": verify_plateau,
        "locked_val_recent": locked_recent,
        "verify_recent": verify_recent,
        "cycles_evaluated": len(locked_accepted),
        "verdict": verdict,
    }
