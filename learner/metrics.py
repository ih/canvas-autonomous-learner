"""Rolling error window + per-action histograms.

Keeps the last N probe results in memory. `mean()` returns the mean MSE across
the window, which is what the state machine thresholds against. Per-action MSE
lets the explorer bias toward whichever actions the model is currently worst at.
Per-state (raw `motor_state`) lets the explorer bias probe starting positions
and EXPLORE sub-bursts toward high-error regions of the joint space.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Optional


@dataclass
class ProbeResult:
    state_key: str                         # quantized motor bins, comma-joined
    action: int
    mse: float
    timestamp: float
    # Raw motor state at the start of the probe. Optional for backward
    # compatibility with older code paths (and for tests) — the new
    # error-driven selectors need it; the legacy `mean()` / `per_action_mean`
    # accessors don't. `None` means "no state information available".
    motor_state: Optional[tuple[float, ...]] = None


class RollingWindow:
    def __init__(self, size: int):
        if size <= 0:
            raise ValueError("window size must be positive")
        self.size = size
        self._buf: deque[ProbeResult] = deque(maxlen=size)

    def add(self, result: ProbeResult) -> None:
        self._buf.append(result)

    def extend(self, results: Iterable[ProbeResult]) -> None:
        for r in results:
            self.add(r)

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)

    def is_empty(self) -> bool:
        return len(self._buf) == 0

    def mean(self) -> float:
        if not self._buf:
            return 0.0
        return sum(r.mse for r in self._buf) / len(self._buf)

    def per_action_mean(self) -> dict[int, float]:
        sums: dict[int, list[float]] = {}
        for r in self._buf:
            sums.setdefault(r.action, []).append(r.mse)
        return {a: sum(v) / len(v) for a, v in sums.items()}

    def mean_in_range(
        self,
        active_range: tuple[float, float],
        joint_idx: int,
    ) -> float:
        """Mean MSE over probes whose starting `motor_state[joint_idx]` is
        inside `active_range`.

        Probes that are missing `motor_state` or have a joint position
        outside the range are excluded from the average entirely — so
        callers get "how well are we doing in the currently-active region"
        cleanly separated from drift outside it.

        Returns 0.0 when no probe qualifies (same semantics as `mean()` on
        an empty window — the state machine caller checks `is_empty()` or
        the absolute count separately when it matters).
        """
        lo, hi = active_range
        vals = [
            r.mse
            for r in self._buf
            if r.motor_state is not None
            and 0 <= joint_idx < len(r.motor_state)
            and lo <= r.motor_state[joint_idx] <= hi
        ]
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    def count_in_range(
        self,
        active_range: tuple[float, float],
        joint_idx: int,
    ) -> int:
        """How many probes in the window have a start state inside the
        active range. Useful to decide whether `mean_in_range` is meaningful
        yet (typically < `probes_per_verify` means the active range just
        expanded and we don't have enough in-range samples to trust the
        mean).
        """
        lo, hi = active_range
        return sum(
            1
            for r in self._buf
            if r.motor_state is not None
            and 0 <= joint_idx < len(r.motor_state)
            and lo <= r.motor_state[joint_idx] <= hi
        )

    def snapshot(self) -> list[ProbeResult]:
        return list(self._buf)
