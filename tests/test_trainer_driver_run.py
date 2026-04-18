"""Unit tests for the stall/pressure/timeout paths in `trainer_driver._run`.

Uses a minimal Python one-liner subprocess (via the same interpreter
that's running pytest) rather than any real training code. The
subprocess's sole job is to emit a known pattern of output so we can
exercise the polling loop deterministically.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from learner import trainer_driver
from learner.events import EventLog


# ---------------------------------------------------------- stall detection


def test_run_raises_stalled_when_no_progress_lines(tmp_path):
    """A subprocess that prints non-progress lines slowly should trigger
    SubprocessStalled, because `_EPOCH_RE` never matches."""
    log = EventLog(tmp_path, session="stall_test")
    # Print one non-progress line then sleep well past the stall timeout.
    cmd = [
        sys.executable, "-u", "-c",
        "import time; print('starting'); time.sleep(10); print('done')",
    ]
    t0 = time.time()
    with pytest.raises(trainer_driver.SubprocessStalled) as ei:
        trainer_driver._run(
            cmd,
            event_log=log,
            tag="train_diffusion",
            stall_timeout_s=2.0,
        )
    elapsed = time.time() - t0
    # Should bail shortly after the stall timeout, not wait for the full sleep.
    assert elapsed < 8.0
    assert ei.value.tag == "train_diffusion"
    assert ei.value.seconds_since_last_progress >= 2.0

    # Event log should carry a training_stalled entry with last_lines.
    events_path = Path(tmp_path) / "events_stall_test.jsonl"
    text = events_path.read_text()
    assert "training_stalled" in text
    assert "starting" in text  # from the rolling last_lines buffer


# --------------------------------------------------------- hard timeout


def test_run_raises_timeout_when_total_wall_time_exceeded(tmp_path):
    log = EventLog(tmp_path, session="timeout_test")
    # Emit progress-looking lines so stall detection does NOT fire, but
    # total runtime exceeds the hard timeout.
    cmd = [
        sys.executable, "-u", "-c",
        (
            "import time\n"
            "for i in range(100):\n"
            "    print(f'Epoch {i}/100: train_loss=0.1, val_loss=0.1')\n"
            "    time.sleep(1.0)\n"
        ),
    ]
    t0 = time.time()
    with pytest.raises(trainer_driver.SubprocessTimeout) as ei:
        trainer_driver._run(
            cmd,
            event_log=log,
            tag="train_diffusion",
            stall_timeout_s=30.0,
            hard_timeout_s=3.0,
        )
    elapsed = time.time() - t0
    assert elapsed < 9.0  # some slack for Python startup + poll ticks
    assert ei.value.timeout_s == 3.0

    events_path = Path(tmp_path) / "events_timeout_test.jsonl"
    text = events_path.read_text()
    assert "subprocess_timeout" in text


# --------------------------------------------------------- memory-pressure abort


def test_run_raises_memory_abort_when_monitor_pressure(tmp_path):
    """Inject a monitor stub whose `is_under_pressure()` always returns
    True. A long-running subprocess should be killed and
    SubprocessMemoryAbort raised.
    """
    log = EventLog(tmp_path, session="mem_test")

    class _AlwaysHot:
        def is_under_pressure(self, _threshold=None):
            return True

        def summary(self):
            return {
                "used_mb": 31000, "total_mb": 32607,
                "used_frac": 0.95, "peak_used_mb": 31000, "samples": 6,
                "util_pct": 100,
            }

    cmd = [sys.executable, "-u", "-c", "import time; time.sleep(60)"]
    t0 = time.time()
    with pytest.raises(trainer_driver.SubprocessMemoryAbort) as ei:
        trainer_driver._run(
            cmd,
            event_log=log,
            tag="train_diffusion",
            monitor=_AlwaysHot(),
            stall_timeout_s=30.0,
            hard_timeout_s=30.0,
        )
    elapsed = time.time() - t0
    assert elapsed < 8.0
    assert ei.value.tag == "train_diffusion"
    assert ei.value.summary["used_frac"] == 0.95

    events_path = Path(tmp_path) / "events_mem_test.jsonl"
    text = events_path.read_text()
    assert "training_memory_abort" in text


# ------------------------------------------------------ happy path still works


def test_run_clean_exit_logs_subprocess_done(tmp_path):
    log = EventLog(tmp_path, session="ok_test")
    cmd = [sys.executable, "-c", "print('hello'); print('world')"]
    trainer_driver._run(cmd, event_log=log, tag="evaluate")
    events_path = Path(tmp_path) / "events_ok_test.jsonl"
    text = events_path.read_text()
    assert "subprocess_done" in text
