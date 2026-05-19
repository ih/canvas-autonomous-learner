"""Background episode recorder running concurrently with RETRAIN.

While the trainer subprocess holds the GPU for 1-3.7h at 1B params, the
robot hardware is otherwise idle. This module runs a fixed-budget
streaming recording in a daemon thread so the next EXPLORE phase has
fresh episodes already in the queue, breaking the data-starvation
spiral the May 2-4 session showed (745 eps over 18 cycles, ~47
eps/burst, 1-3.7h retrains in between).

Hardware contract: caller MUST call hardware.disconnect() before
entering the context, and hardware.connect() (+ load_predictor) after
exit. The recorder spawns a subprocess that takes exclusive control of
the SO-101 + cameras until it finishes. Mirrors the foreground EXPLORE
disconnect/reconnect pattern.

Sizing: the recorder runs to a fixed `target_episodes` budget set by
caller. Keep this conservative — `target_episodes ≈ 0.5 * expected
retrain wall-time / sec_per_episode` — so the recorder finishes well
before the trainer. We do NOT kill the recorder mid-run on context
exit because the streaming recorder writes parquet/MP4 chunks
incrementally and a SIGTERM mid-chunk may leave a partial file.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from . import explorer


class BackgroundRecorder:
    """Context manager wrapping a fixed-budget background record run.

    Usage:
        # caller already disconnected hardware
        with BackgroundRecorder(cfg, target_eps=600, event_log=log) as rec:
            result = trainer_driver.retrain(...)  # blocks for hours
        # context exit waits for the recorder to finish naturally
        for d, n in zip(rec.dataset_dirs(), rec.episode_counts()):
            ...
        # caller reconnects hardware

    `target_episodes <= 0` makes the context a no-op (records nothing,
    returns empty lists from `dataset_dirs()`). Useful for callers that
    want to gate on a config flag without branching the surrounding
    code.
    """

    def __init__(
        self,
        cfg,
        target_episodes: int,
        event_log,
        repo_id_prefix: Optional[str] = None,
        joint_range_override: Optional[dict] = None,
        force_joint: Optional[str] = None,
    ):
        self._cfg = cfg
        self._target_episodes = max(0, int(target_episodes))
        self._event_log = event_log
        self._repo_id_prefix = repo_id_prefix
        self._joint_range_override = joint_range_override
        self._force_joint = force_joint
        self._thread: Optional[threading.Thread] = None
        self._dataset_dir: Optional[Path] = None
        self._exception: Optional[BaseException] = None

    def __enter__(self) -> "BackgroundRecorder":
        if self._target_episodes <= 0:
            return self
        if self._event_log is not None:
            self._event_log.log(
                "background_recorder_start",
                target_episodes=self._target_episodes,
                repo_id_prefix=self._repo_id_prefix,
                joint_range_override=self._joint_range_override,
                force_joint=self._force_joint,
            )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            self._dataset_dir = explorer.collect_batch_continuous(
                self._cfg,
                self._target_episodes,
                window=None,
                event_log=self._event_log,
                joint_range_override=self._joint_range_override,
                repo_id_prefix=self._repo_id_prefix,
                event_tag="background_explore_start",
                force_joint=self._force_joint,
            )
        except BaseException as e:
            self._exception = e

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._thread is None:
            return False
        # Always join to completion. The recorder MUST finish writing its
        # parquet/MP4 chunks cleanly before we hand the hardware back to
        # the orchestrator. If trainer finishes first, we sit here briefly
        # — sizing target_episodes conservatively keeps that wait small.
        self._thread.join()
        if self._event_log is not None:
            self._event_log.log(
                "background_recorder_done",
                dataset_dir=str(self._dataset_dir) if self._dataset_dir else None,
                episodes=self._target_episodes if self._dataset_dir else 0,
                exception=(
                    f"{type(self._exception).__name__}: {self._exception}"
                    if self._exception else None
                ),
            )
        # Don't suppress exceptions raised inside the with-block.
        return False

    def dataset_dirs(self) -> list[Path]:
        """Return list of dataset dirs produced (0 or 1 entry).

        Empty list if the recorder failed, was disabled (target<=0), or
        produced no output.
        """
        return [self._dataset_dir] if self._dataset_dir is not None else []

    def episode_counts(self) -> list[int]:
        """Episode counts paired with `dataset_dirs()`.

        Best-effort: returns the target budget if the recorder succeeded,
        otherwise an empty list. The streaming recorder writes the full
        budget before returning, so this matches the actual count
        modulo extremely rare partial-write edge cases.
        """
        return [self._target_episodes] if self._dataset_dir is not None else []

    def exception(self) -> Optional[BaseException]:
        """Return the exception raised by the recorder thread (if any)."""
        return self._exception
