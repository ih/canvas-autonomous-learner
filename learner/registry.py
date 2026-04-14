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
                "locked_val_history": [],
                "experiment_status": "unstarted",
                "consecutive_guard_rejections": 0,
                # Progressive state-space curriculum
                "range_active": None,
                "range_stable_cycles": 0,
                "range_history": [],
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

    def append_canvas_dir(self, path: str | Path, episodes_added: int) -> None:
        data = self._read()
        dirs = list(data.get("accumulated_canvas_dirs", []))
        dirs.append(str(path))
        data["accumulated_canvas_dirs"] = dirs
        data["episodes_collected"] = int(data.get("episodes_collected", 0)) + int(episodes_added)
        self._write(data)

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
