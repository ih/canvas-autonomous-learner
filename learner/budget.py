"""Dynamic explore-burst sizing.

Scales the number of episodes per EXPLORE call by how far above `tau_high`
the recent probe MSE sits. This is part of the autonomous learner's value
proposition: when the model is bad, collect a lot of data fast; when it's
marginal, collect a little; when it's cold-start (no window yet), collect
the maximum because we have no signal to economize on.

Kept in its own tiny module (vs. inlined into the orchestrator) so the
logic is unit-testable in isolation without spinning up a state machine.
"""

from __future__ import annotations


def dynamic_explore_batch_size(
    mean_err: float | None,
    tau_high: float,
    base: int,
    lo: int,
    hi: int,
) -> int:
    """Return the number of episodes to collect in the next EXPLORE burst.

    Rules:
    - `mean_err is None` → `hi` (cold start or just after a range expansion;
      we have no signal to thresh on, so collect the maximum).
    - `mean_err <= tau_high` → `base` (clamped to `[lo, hi]`).
    - `mean_err >= 3 × tau_high` → `3 × base` (clamped to `[lo, hi]`).
    - In between: linear scaling `base * (mean_err / tau_high)`.

    The final result is always an integer clamped to `[lo, hi]`.
    """
    if mean_err is None:
        return int(max(lo, min(hi, hi)))  # == hi, but kept explicit
    # Degenerate: tau_high very small or zero → treat as "always bad".
    if tau_high <= 0:
        return int(max(lo, min(hi, hi)))
    scale = max(1.0, min(3.0, mean_err / tau_high))
    return int(max(lo, min(hi, base * scale)))
