"""Progressive state-space curriculum for the autonomous learner.

The `RangeTracker` owns the "active range" for the control joint — the
slice of the full joint space the learner is currently focused on. It
starts narrow (e.g. shoulder_pan ∈ [−20°, +20°]) and expands outward
only after the learner has consistently predicted well inside the
current range for `stable_cycles_required` VERIFY passes in a row.
This is a curriculum: learn the easiest region first, then gradually
take on more, while keeping old data in the accumulated training set
so earlier regions stay covered.

All state the state machine needs to resume a curriculum after a crash
lives in the registry (`range_active`, `range_stable_cycles`,
`range_history`), so `from_config_or_registry` handles both fresh runs
and mid-experiment restarts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RangeTracker:
    control_joint: str
    control_joint_idx: int
    full_min: float
    full_max: float
    expansion_factor: float
    stable_cycles_required: int
    active: tuple[float, float]
    stable_cycles: int = 0
    # History of expansions, newest last. Each entry:
    # {"cycle": int, "total_eps": int, "old_range": (lo, hi), "new_range": (lo, hi)}
    history: list[dict] = field(default_factory=list)

    # ----------------------------------------------------------------- build

    @classmethod
    def from_config(
        cls,
        *,
        control_joint: str,
        control_joint_idx: int,
        full_min: float,
        full_max: float,
        initial_half_width: float,
        expansion_factor: float,
        stable_cycles_required: int,
    ) -> "RangeTracker":
        center = (full_min + full_max) / 2.0
        half = float(min(initial_half_width, (full_max - full_min) / 2.0))
        return cls(
            control_joint=control_joint,
            control_joint_idx=control_joint_idx,
            full_min=float(full_min),
            full_max=float(full_max),
            expansion_factor=float(expansion_factor),
            stable_cycles_required=int(stable_cycles_required),
            active=(
                max(full_min, center - half),
                min(full_max, center + half),
            ),
        )

    @classmethod
    def from_config_or_registry(
        cls,
        *,
        control_joint: str,
        control_joint_idx: int,
        full_min: float,
        full_max: float,
        initial_half_width: float,
        expansion_factor: float,
        stable_cycles_required: int,
        registry_state: Optional[dict] = None,
    ) -> "RangeTracker":
        """Build from config, optionally restoring `active` / `stable_cycles`
        / `history` from a registry snapshot. Useful for resuming mid-run.
        """
        tracker = cls.from_config(
            control_joint=control_joint,
            control_joint_idx=control_joint_idx,
            full_min=full_min,
            full_max=full_max,
            initial_half_width=initial_half_width,
            expansion_factor=expansion_factor,
            stable_cycles_required=stable_cycles_required,
        )
        if not registry_state:
            return tracker

        saved_active = registry_state.get("range_active")
        if saved_active and len(saved_active) == 2:
            lo = max(full_min, float(saved_active[0]))
            hi = min(full_max, float(saved_active[1]))
            if lo < hi:
                tracker.active = (lo, hi)
        tracker.stable_cycles = int(registry_state.get("range_stable_cycles", 0))
        history = registry_state.get("range_history") or []
        if isinstance(history, list):
            tracker.history = [dict(h) for h in history]
        return tracker

    # --------------------------------------------------------------- queries

    def at_max(self) -> bool:
        lo, hi = self.active
        return (lo <= self.full_min + 1e-3) and (hi >= self.full_max - 1e-3)

    def stable(self) -> bool:
        return self.stable_cycles >= self.stable_cycles_required

    def should_expand(self) -> bool:
        return self.stable() and not self.at_max()

    # -------------------------------------------------------------- updates

    def good_cycle(self) -> None:
        self.stable_cycles += 1

    def bad_cycle(self) -> None:
        self.stable_cycles = 0

    def expand(self, *, cycle: int, total_eps: int) -> tuple[float, float]:
        """Widen the active range by `expansion_factor`, clamped to the
        full joint bounds. Appends an entry to `history` and resets
        `stable_cycles` to 0 (since the newly-added region needs its own
        fresh set of good verifies before the next expansion).

        Returns the new `(lo, hi)`.
        """
        if self.at_max():
            return self.active

        old_range = self.active
        center = (self.full_min + self.full_max) / 2.0
        cur_half = (self.active[1] - self.active[0]) / 2.0
        new_half = cur_half * self.expansion_factor
        full_half = (self.full_max - self.full_min) / 2.0
        new_half = min(new_half, full_half)
        self.active = (
            max(self.full_min, center - new_half),
            min(self.full_max, center + new_half),
        )
        self.stable_cycles = 0
        self.history.append({
            "cycle": int(cycle),
            "total_eps": int(total_eps),
            "old_range": [float(old_range[0]), float(old_range[1])],
            "new_range": [float(self.active[0]), float(self.active[1])],
        })
        return self.active

    # ------------------------------------------------------------ serialize

    def to_registry_state(self) -> dict:
        """Snapshot for `Registry` persistence. The registry's write path
        will merge these fields into the top-level JSON object.
        """
        return {
            "range_active": [float(self.active[0]), float(self.active[1])],
            "range_stable_cycles": int(self.stable_cycles),
            "range_history": [dict(h) for h in self.history],
        }

    def to_namespaced_snapshot(self, prefix: str) -> dict:
        """Like to_registry_state but with a caller-chosen prefix so two
        trackers (primary + secondary) can coexist in one registry JSON.
        """
        return {
            f"{prefix}_active": [float(self.active[0]), float(self.active[1])],
            f"{prefix}_stable_cycles": int(self.stable_cycles),
            f"{prefix}_history": [dict(h) for h in self.history],
        }

    @classmethod
    def from_namespaced_snapshot(
        cls,
        snapshot: dict,
        prefix: str,
        *,
        control_joint: str,
        control_joint_idx: int,
        full_min: float,
        full_max: float,
        initial_half_width: float,
        expansion_factor: float,
        stable_cycles_required: int,
    ) -> "RangeTracker":
        """Mirror of `from_config_or_registry` that reads prefix-keyed fields."""
        tracker = cls.from_config(
            control_joint=control_joint,
            control_joint_idx=control_joint_idx,
            full_min=full_min,
            full_max=full_max,
            initial_half_width=initial_half_width,
            expansion_factor=expansion_factor,
            stable_cycles_required=stable_cycles_required,
        )
        saved_active = snapshot.get(f"{prefix}_active")
        if saved_active and len(saved_active) == 2:
            lo = max(full_min, float(saved_active[0]))
            hi = min(full_max, float(saved_active[1]))
            if lo < hi:
                tracker.active = (lo, hi)
        tracker.stable_cycles = int(snapshot.get(f"{prefix}_stable_cycles", 0))
        hist = snapshot.get(f"{prefix}_history") or []
        if isinstance(hist, list):
            tracker.history = [dict(h) for h in hist]
        return tracker


# ------------------------------------------------------- curriculum state

# SO-101 joint order — matches the `JOINTS` list in canvas-robot-control.
# Kept as a static lookup here so CurriculumState doesn't need a live
# Hardware instance to resolve joint names during tests.
_JOINT_ORDER = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


def _joint_idx(name: str) -> int:
    return _JOINT_ORDER.index(name)


class CurriculumState:
    """Sequential two-stage curriculum: primary joint first, then secondary.

    Stage 1 ("primary"): widen `primary_tracker.active` from its initial
    narrow band to the full primary joint range. The secondary joint's
    range is pinned to a tight center band (`secondary_pinned_range`) so
    the model only has to learn shoulder_pan under a nearly-constant
    elbow_flex scene.

    Transition: once `primary_tracker.at_max() and .stable()`, the state
    machine flips to stage 2 and the secondary curriculum begins.

    Stage 2 ("secondary"): primary is pinned at its full range; widen
    `secondary_tracker.active` from its initial narrow band to the full
    secondary joint range. Sub-burst planning and probe-state picking
    both operate on the secondary joint axis now.

    Termination: once `secondary_tracker.at_max() and .stable()`, the
    learner is "satisfied at full curriculum".

    If no secondary is configured, stage 2 is skipped — the curriculum
    collapses to a single-joint curriculum and terminates when primary
    is satisfied at full range.
    """

    STAGE_PRIMARY = "primary"
    STAGE_SECONDARY = "secondary"

    def __init__(
        self,
        primary: RangeTracker,
        secondary_config: Optional[dict] = None,
        stage: str = STAGE_PRIMARY,
        secondary: Optional[RangeTracker] = None,
        secondary_pinned_half_width: float = 3.0,
    ):
        self.primary = primary
        self.secondary_config = secondary_config  # dict or None
        self.stage = stage
        self.secondary = secondary
        # Width of the pinned secondary range during stage 1.
        self.secondary_pinned_half_width = float(secondary_pinned_half_width)

    # ----------------------------------------------------------- factory

    @classmethod
    def from_config_or_registry(
        cls,
        cfg_range,
        registry_snapshot: Optional[dict] = None,
    ) -> Optional["CurriculumState"]:
        """Build a CurriculumState from a config-range namespace.

        The config can be either the legacy flat form or the new nested
        form. Legacy flat form is treated as primary-only (no secondary),
        preserving backward compatibility with tests that pass a flat
        `range` block.
        """
        if cfg_range is None or not getattr(cfg_range, "enabled", True):
            return None

        registry_snapshot = registry_snapshot or {}

        # Detect nested vs flat form
        primary_cfg = getattr(cfg_range, "primary", None)
        secondary_cfg = getattr(cfg_range, "secondary", None)
        is_nested = primary_cfg is not None

        if is_nested:
            primary = RangeTracker.from_namespaced_snapshot(
                registry_snapshot,
                prefix="range_primary",
                control_joint=primary_cfg.control_joint,
                control_joint_idx=_joint_idx(primary_cfg.control_joint),
                full_min=float(primary_cfg.full_min),
                full_max=float(primary_cfg.full_max),
                initial_half_width=float(primary_cfg.initial_half_width),
                expansion_factor=float(primary_cfg.expansion_factor),
                stable_cycles_required=int(primary_cfg.stable_cycles_required),
            )
            # Freeze the secondary config as a dict for later instantiation.
            sec_cfg_dict = None
            secondary: Optional[RangeTracker] = None
            secondary_pinned_hw = 3.0
            if secondary_cfg is not None:
                sec_cfg_dict = {
                    "control_joint": secondary_cfg.control_joint,
                    "full_min": float(secondary_cfg.full_min),
                    "full_max": float(secondary_cfg.full_max),
                    "initial_half_width": float(secondary_cfg.initial_half_width),
                    "expansion_factor": float(secondary_cfg.expansion_factor),
                    "stable_cycles_required": int(secondary_cfg.stable_cycles_required),
                }
                secondary_pinned_hw = float(getattr(secondary_cfg, "pinned_half_width", 3.0))
                # Restore secondary tracker from registry if it already exists
                if registry_snapshot.get("range_secondary_active") is not None:
                    secondary = RangeTracker.from_namespaced_snapshot(
                        registry_snapshot,
                        prefix="range_secondary",
                        control_joint=sec_cfg_dict["control_joint"],
                        control_joint_idx=_joint_idx(sec_cfg_dict["control_joint"]),
                        full_min=sec_cfg_dict["full_min"],
                        full_max=sec_cfg_dict["full_max"],
                        initial_half_width=sec_cfg_dict["initial_half_width"],
                        expansion_factor=sec_cfg_dict["expansion_factor"],
                        stable_cycles_required=sec_cfg_dict["stable_cycles_required"],
                    )
        else:
            # Legacy flat form: one tracker, no secondary.
            primary = RangeTracker.from_config_or_registry(
                control_joint=cfg_range.control_joint,
                control_joint_idx=_joint_idx(cfg_range.control_joint),
                full_min=float(cfg_range.full_min),
                full_max=float(cfg_range.full_max),
                initial_half_width=float(cfg_range.initial_half_width),
                expansion_factor=float(cfg_range.expansion_factor),
                stable_cycles_required=int(cfg_range.stable_cycles_required),
                registry_state=registry_snapshot,
            )
            sec_cfg_dict = None
            secondary = None
            secondary_pinned_hw = 3.0

        stage = registry_snapshot.get("curriculum_stage", cls.STAGE_PRIMARY)
        if stage not in (cls.STAGE_PRIMARY, cls.STAGE_SECONDARY):
            stage = cls.STAGE_PRIMARY

        return cls(
            primary=primary,
            secondary_config=sec_cfg_dict,
            stage=stage,
            secondary=secondary,
            secondary_pinned_half_width=secondary_pinned_hw,
        )

    # ---------------------------------------------------------- queries

    def active_tracker(self) -> RangeTracker:
        """Return whichever tracker is currently being expanded."""
        if self.stage == self.STAGE_SECONDARY and self.secondary is not None:
            return self.secondary
        return self.primary

    @property
    def active_joint_name(self) -> str:
        return self.active_tracker().control_joint

    @property
    def active_joint_idx(self) -> int:
        return self.active_tracker().control_joint_idx

    @property
    def active_range(self) -> tuple[float, float]:
        return self.active_tracker().active

    def is_done(self) -> bool:
        """True when the full curriculum has been satisfied."""
        if self.stage == self.STAGE_PRIMARY:
            if self.secondary_config is None:
                # Single-joint curriculum: done when primary is at_max + stable.
                return self.primary.at_max() and self.primary.stable()
            # Two-joint: stage 1 alone doesn't mean "done"; we need stage 2.
            return False
        # Stage 2 done when secondary tracker is at_max + stable.
        return (
            self.secondary is not None
            and self.secondary.at_max()
            and self.secondary.stable()
        )

    def should_transition_to_secondary(self) -> bool:
        return (
            self.stage == self.STAGE_PRIMARY
            and self.secondary_config is not None
            and self.primary.at_max()
            and self.primary.stable()
        )

    # ---------------------------------------------------------- updates

    def good_cycle(self) -> None:
        self.active_tracker().good_cycle()

    def bad_cycle(self) -> None:
        self.active_tracker().bad_cycle()

    def should_expand(self) -> bool:
        return self.active_tracker().should_expand()

    def expand(self, *, cycle: int, total_eps: int) -> tuple[float, float]:
        return self.active_tracker().expand(cycle=cycle, total_eps=total_eps)

    def transition_to_secondary(self) -> RangeTracker:
        """Flip from stage 1 to stage 2. Creates the secondary tracker
        from the stored config. Idempotent — calling a second time is
        a no-op beyond returning the existing tracker.
        """
        if self.stage == self.STAGE_SECONDARY and self.secondary is not None:
            return self.secondary
        if self.secondary_config is None:
            raise RuntimeError("cannot transition: no secondary config")
        c = self.secondary_config
        self.secondary = RangeTracker.from_config(
            control_joint=c["control_joint"],
            control_joint_idx=_joint_idx(c["control_joint"]),
            full_min=c["full_min"],
            full_max=c["full_max"],
            initial_half_width=c["initial_half_width"],
            expansion_factor=c["expansion_factor"],
            stable_cycles_required=c["stable_cycles_required"],
        )
        self.stage = self.STAGE_SECONDARY
        return self.secondary

    # ------------------------------------------------ override builders

    def joint_range_override(self) -> dict:
        """Build the `joint_range_override` dict to pass to `collect_batch`.

        Stage 1: primary = primary_tracker.active (narrow-to-wide),
                 secondary = tight pinned band around secondary center.
        Stage 2: primary = full primary range,
                 secondary = secondary_tracker.active (narrow-to-wide).

        If no secondary is configured, only the primary is set.
        """
        out: dict[str, tuple[float, float]] = {}
        primary_key = f"{self.primary.control_joint}.pos"

        if self.stage == self.STAGE_PRIMARY:
            out[primary_key] = (
                float(self.primary.active[0]),
                float(self.primary.active[1]),
            )
            if self.secondary_config is not None:
                # Pin secondary to a tight band around its range center.
                c = self.secondary_config
                sec_key = f"{c['control_joint']}.pos"
                center = (c["full_min"] + c["full_max"]) / 2.0
                half = self.secondary_pinned_half_width
                sec_lo = max(c["full_min"], center - half)
                sec_hi = min(c["full_max"], center + half)
                out[sec_key] = (sec_lo, sec_hi)
        else:
            # Stage 2: primary at full range, secondary curriculum-driven.
            out[primary_key] = (
                float(self.primary.full_min),
                float(self.primary.full_max),
            )
            if self.secondary is not None:
                sec_key = f"{self.secondary.control_joint}.pos"
                out[sec_key] = (
                    float(self.secondary.active[0]),
                    float(self.secondary.active[1]),
                )
        return out

    # ---------------------------------------------------- serialization

    def to_registry_snapshot(self) -> dict:
        """All the state needed to resume a curriculum mid-run."""
        snap: dict = {"curriculum_stage": self.stage}
        snap.update(self.primary.to_namespaced_snapshot("range_primary"))
        if self.secondary is not None:
            snap.update(self.secondary.to_namespaced_snapshot("range_secondary"))
        return snap
