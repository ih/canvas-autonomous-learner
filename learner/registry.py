"""Single source of truth for the live checkpoint + live canvas dataset.

Writes are atomic (temp + rename) so a crash mid-retrain can't leave the
pointer half-swapped. Every swap appends to `history[]` so the operator can
audit what was promoted, when, and with what val MSE.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


class Registry:
    def __init__(self, path: str | Path, initial: dict[str, Any] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write(initial or {
                "live_checkpoint": None,
                "base_canvas_dataset": None,
                "merged_canvas_dataset": None,
                "baseline_val_mse": None,
                "last_retrain": None,
                "history": [],
                # Cold start + plateau stop
                "episodes_collected": 0,
                "accumulated_canvas_dirs": [],
                # Parallel accumulator for wide-verify probe canvases.
                # MUST stay strictly disjoint from `accumulated_canvas_dirs`
                # (which feeds training) so the wide-verify corpus
                # remains held-out for generalization eval. Each entry:
                # {"path": str, "n_probes": int, "scenes": int,
                #  "cycle": int, "t": iso8601-string}.
                "wide_verify_canvas_dirs": [],
                "wide_verify_history": [],
                "locked_val_history": [],
                "experiment_status": "unstarted",
                "consecutive_guard_rejections": 0,
                # Progressive state-space curriculum
                "range_active": None,
                "range_stable_cycles": 0,
                "range_history": [],
                # Pinned session name (set on first startup so dashboard
                # history survives learner restarts) and any pending
                # advisor-issued verb awaiting orchestrator action.
                "session_name": None,
                "pending_advisor_decision": None,
                # Scene-config tracking for the min-data experiment.
                # `scene_idx` is the index of the scene CURRENTLY in
                # front of the robot — increments by 1 on every IDLE-
                # path `scene_ready_acknowledged`. `eps_per_scene` is a
                # rollup mapping `str(scene_idx) -> cumulative episodes
                # collected on that scene` (str keys because JSON
                # mandates string-keyed dicts). Together these let the
                # dashboard plot `wide_verify_mse vs num_scenes` and
                # `vs eps_per_scene` directly. `scene_idx == 0` is the
                # operator's initial setup at experiment_start.
                "scene_idx": 0,
                "eps_per_scene": {},
            })

    # --------------------------------------------------------------- internals

    def _read(self) -> dict[str, Any]:
        with open(self.path) as f:
            return json.load(f)

    def _write(self, data: dict[str, Any]) -> None:
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=".registry_", suffix=".json", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self.path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    # ------------------------------------------------------------------ public

    def load(self) -> dict[str, Any]:
        return self._read()

    def live_checkpoint(self) -> str | None:
        return self._read().get("live_checkpoint")

    def baseline_val_mse(self) -> float | None:
        return self._read().get("baseline_val_mse")

    def set_baseline(
        self,
        live_checkpoint: str,
        base_canvas_dataset: str,
        baseline_val_mse: float | None,
    ) -> None:
        data = self._read()
        data["live_checkpoint"] = str(live_checkpoint)
        data["base_canvas_dataset"] = str(base_canvas_dataset)
        data["merged_canvas_dataset"] = str(base_canvas_dataset)
        data["baseline_val_mse"] = (
            float(baseline_val_mse) if baseline_val_mse is not None else None
        )
        data["last_retrain"] = None
        self._write(data)

    def set_baseline_val_mse(self, val_mse: float) -> None:
        data = self._read()
        data["baseline_val_mse"] = float(val_mse)
        self._write(data)

    # ------------------------------------------------ comparison experiment

    def episodes_collected(self) -> int:
        return int(self._read().get("episodes_collected", 0))

    def accumulated_canvas_dirs(self) -> list[str]:
        return list(self._read().get("accumulated_canvas_dirs", []))

    def locked_val_history(self) -> list[dict]:
        return list(self._read().get("locked_val_history", []))

    def experiment_status(self) -> str:
        return str(self._read().get("experiment_status", "unstarted"))

    def consecutive_guard_rejections(self) -> int:
        return int(self._read().get("consecutive_guard_rejections", 0))

    def append_canvas_dir(
        self,
        path: str | Path,
        episodes_added: int,
        scene_idx: int | None = None,
    ) -> None:
        """Add a training canvas dir + bump cumulative episode count.

        `scene_idx`: the scene the episodes were collected on. When
        provided, increments `eps_per_scene[str(scene_idx)]` by
        `episodes_added`. Defaults to the registry's current
        `scene_idx` for backward compatibility — pass an explicit value
        when historical episodes are added (e.g. from a prior session).
        """
        data = self._read()
        path_str = str(path)
        # Leakage invariant: training data must never overlap the
        # wide-verify (held-out) corpus. Symmetric check to
        # append_wide_verify_dir.
        wv_dirs = data.get("wide_verify_canvas_dirs", []) or []
        if path_str in wv_dirs:
            raise ValueError(
                f"refusing to add training canvas dir {path_str!r}: it "
                "is already in wide_verify_canvas_dirs (held-out set). "
                "Training corpus MUST stay disjoint from wide-verify."
            )
        dirs = list(data.get("accumulated_canvas_dirs", []))
        dirs.append(path_str)
        data["accumulated_canvas_dirs"] = dirs
        data["episodes_collected"] = int(data.get("episodes_collected", 0)) + int(episodes_added)
        idx = int(scene_idx) if scene_idx is not None else int(data.get("scene_idx", 0))
        eps_per_scene = dict(data.get("eps_per_scene", {}) or {})
        key = str(idx)
        eps_per_scene[key] = int(eps_per_scene.get(key, 0)) + int(episodes_added)
        data["eps_per_scene"] = eps_per_scene
        self._write(data)

    def scene_idx(self) -> int:
        return int(self._read().get("scene_idx", 0))

    def eps_per_scene(self) -> dict[str, int]:
        return dict(self._read().get("eps_per_scene", {}) or {})

    def bump_scene_idx(self) -> int:
        """Increment scene_idx by 1, persist, return the new value.

        Called by the orchestrator on every IDLE-path
        `scene_ready_acknowledged` so subsequent canvas-dir appends are
        tagged with the new scene's index. Wide-verify scene changes
        do NOT bump this counter — wide-verify scenes are held-out and
        live in their own corpus.
        """
        data = self._read()
        new_idx = int(data.get("scene_idx", 0)) + 1
        data["scene_idx"] = new_idx
        self._write(data)
        return new_idx

    # ---------------------------------------- wide-verify (held-out corpus)

    def wide_verify_canvas_dirs(self) -> list[str]:
        """Paths to every canvas dir from prior wide-verify bursts.

        These canvases are NEVER fed to training — they are the
        held-out generalization corpus the trainer evaluates against
        each retrain. The hard rule: a path that ever appears here
        must never appear in `accumulated_canvas_dirs`.
        """
        return list(self._read().get("wide_verify_canvas_dirs", []))

    def append_wide_verify_dir(
        self,
        path: str | Path,
        *,
        n_probes: int,
        scenes: int,
        cycle: int,
    ) -> None:
        """Record a new wide-verify canvas dir + its provenance.

        Raises ValueError if the path already appears in
        `accumulated_canvas_dirs` — the leakage invariant. Caller
        should never construct a wide-verify path that overlaps the
        training accumulator, but checking here makes the invariant
        impossible to violate accidentally.
        """
        data = self._read()
        path_str = str(path)
        train_dirs = data.get("accumulated_canvas_dirs", []) or []
        if path_str in train_dirs:
            raise ValueError(
                f"refusing to add wide-verify dir {path_str!r}: it is "
                "already in accumulated_canvas_dirs (training set). "
                "Wide-verify corpus MUST stay disjoint from training."
            )
        dirs = list(data.get("wide_verify_canvas_dirs", []))
        dirs.append(path_str)
        data["wide_verify_canvas_dirs"] = dirs
        history = list(data.get("wide_verify_history", []))
        history.append({
            "path": path_str,
            "n_probes": int(n_probes),
            "scenes": int(scenes),
            "cycle": int(cycle),
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        data["wide_verify_history"] = history
        self._write(data)

    def wide_verify_history(self) -> list[dict]:
        """Provenance entries (path, n_probes, scenes, cycle, t) per
        wide-verify burst. Newest last."""
        return list(self._read().get("wide_verify_history", []))

    def append_locked_val(
        self,
        cycle: int,
        total_eps: int,
        locked_val_mse: float | None,
        train_val_mse: float | None,
        accepted: bool,
    ) -> None:
        data = self._read()
        history = list(data.get("locked_val_history", []))
        history.append({
            "cycle": int(cycle),
            "total_eps": int(total_eps),
            "locked_val_mse": float(locked_val_mse) if locked_val_mse is not None else None,
            "train_val_mse": float(train_val_mse) if train_val_mse is not None else None,
            "accepted": bool(accepted),
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        data["locked_val_history"] = history
        self._write(data)

    def set_experiment_status(self, status: str) -> None:
        data = self._read()
        data["experiment_status"] = str(status)
        self._write(data)

    def bump_guard_rejections(self) -> int:
        data = self._read()
        n = int(data.get("consecutive_guard_rejections", 0)) + 1
        data["consecutive_guard_rejections"] = n
        self._write(data)
        return n

    def reset_guard_rejections(self) -> None:
        data = self._read()
        data["consecutive_guard_rejections"] = 0
        self._write(data)

    # ------------------------------------------------- session + advisor state

    def session_name(self) -> str | None:
        return self._read().get("session_name")

    def set_session_name(self, name: str) -> None:
        data = self._read()
        data["session_name"] = str(name)
        self._write(data)

    def pending_advisor_decision(self) -> dict | None:
        v = self._read().get("pending_advisor_decision")
        return dict(v) if isinstance(v, dict) else None

    def set_pending_advisor_decision(self, decision: dict | None) -> None:
        data = self._read()
        data["pending_advisor_decision"] = dict(decision) if decision else None
        self._write(data)

    # ---------------------------------------------- range curriculum state

    def range_snapshot(self) -> dict:
        """Return every curriculum-related field from the registry.

        Covers both the legacy single-tracker fields (for backward compat
        with old sessions and legacy flat `range:` configs) and the new
        two-stage namespaced fields (`curriculum_stage`, `range_primary_*`,
        `range_secondary_*`) that `CurriculumState.from_config_or_registry`
        knows how to rehydrate.
        """
        data = self._read()
        out: dict = {}
        for key, value in data.items():
            if key.startswith("range_") or key.startswith("curriculum_"):
                out[key] = value
        # Preserve legacy flat-field semantics for callers that still read
        # the un-namespaced names directly.
        out.setdefault("range_active", data.get("range_active"))
        out.setdefault("range_stable_cycles", int(data.get("range_stable_cycles", 0)))
        out.setdefault("range_history", list(data.get("range_history", [])))
        return out

    def save_range_state(self, snapshot: dict) -> None:
        """Merge `{range_active, range_stable_cycles, range_history}` from
        a RangeTracker snapshot into the registry.
        """
        data = self._read()
        # Persist ALL curriculum-related keys from the snapshot. Includes
        # the legacy single-tracker fields (`range_active`, `range_stable_cycles`,
        # `range_history`) AND the new namespaced two-stage fields
        # (`curriculum_stage`, `range_primary_*`, `range_secondary_*`).
        allowed_prefixes = ("range_", "curriculum_")
        for key, value in snapshot.items():
            if any(key.startswith(p) for p in allowed_prefixes):
                data[key] = value
        self._write(data)

    # ----------------------------------------------------------------- swap

    def swap(
        self,
        new_checkpoint: str,
        merged_canvas_dataset: str | None = None,
        val_mse: float | None = None,
        notes: str | None = None,
    ) -> None:
        data = self._read()
        previous = data.get("live_checkpoint")
        data["live_checkpoint"] = str(new_checkpoint)
        if merged_canvas_dataset is not None:
            data["merged_canvas_dataset"] = str(merged_canvas_dataset)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        data["last_retrain"] = stamp
        data.setdefault("history", []).append({
            "t": stamp,
            "previous": previous,
            "new": str(new_checkpoint),
            "val_mse": val_mse,
            "notes": notes,
        })
        self._write(data)
