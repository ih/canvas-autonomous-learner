"""Unit tests for `learner.gpu_monitor`.

No live `nvidia-smi` call — `sample_gpu` is parse-tested with fake
output, and `GpuMonitor` is seeded via its `ingest()` method so the
sustained-pressure check is deterministic.
"""

from __future__ import annotations

import time

import pytest

from learner import gpu_monitor


# -------------------------------------------------- _parse_nvidia_smi_output


def test_parse_nvidia_smi_output_happy_path():
    raw = "2446, 32607, 3\n"
    out = gpu_monitor._parse_nvidia_smi_output(raw)
    assert out == {
        "used_mb": 2446,
        "total_mb": 32607,
        "util_pct": 3,
        "used_frac": 2446 / 32607,
    }


def test_parse_nvidia_smi_output_rejects_garbage():
    assert gpu_monitor._parse_nvidia_smi_output("") is None
    assert gpu_monitor._parse_nvidia_smi_output("not csv") is None
    assert gpu_monitor._parse_nvidia_smi_output("abc, def, ghi") is None
    # Zero total VRAM is nonsense — reject.
    assert gpu_monitor._parse_nvidia_smi_output("100, 0, 0") is None


def test_parse_nvidia_smi_output_takes_first_line():
    raw = "1000, 8000, 10\n2000, 8000, 20\n"
    out = gpu_monitor._parse_nvidia_smi_output(raw)
    assert out["used_mb"] == 1000


# ---------------------------------------------------- sample_gpu (via stub)


def test_sample_gpu_returns_none_when_nvidia_smi_missing(monkeypatch):
    def _raise(*_a, **_kw):
        raise FileNotFoundError("nvidia-smi not on PATH")
    monkeypatch.setattr(gpu_monitor.subprocess, "run", _raise)
    assert gpu_monitor.sample_gpu() is None


def test_sample_gpu_returns_none_on_nonzero_exit(monkeypatch):
    class _Result:
        returncode = 1
        stdout = ""
    monkeypatch.setattr(gpu_monitor.subprocess, "run", lambda *a, **k: _Result())
    assert gpu_monitor.sample_gpu() is None


def test_sample_gpu_parses_live_style_output(monkeypatch):
    class _Result:
        returncode = 0
        stdout = "12345, 24576, 42\n"
    monkeypatch.setattr(gpu_monitor.subprocess, "run", lambda *a, **k: _Result())
    out = gpu_monitor.sample_gpu()
    assert out == {
        "used_mb": 12345,
        "total_mb": 24576,
        "util_pct": 42,
        "used_frac": 12345 / 24576,
    }


# ---------------------------------------------------- GpuMonitor (pure logic)


def _mk(used_mb: int, total_mb: int = 32607) -> dict:
    return {
        "used_mb": used_mb,
        "total_mb": total_mb,
        "util_pct": 50,
        "used_frac": used_mb / total_mb,
    }


def test_monitor_is_under_pressure_requires_sustained_breach():
    m = gpu_monitor.GpuMonitor(sustained_seconds=30.0, abort_frac=0.93)
    now = time.time()
    # Single spike in the window: NOT pressure.
    m.ingest(_mk(30720), now=now - 20)        # 94%
    m.ingest(_mk(10000), now=now - 10)        # 31%
    assert not m.is_under_pressure()
    # Two sustained breaches: IS pressure.
    m2 = gpu_monitor.GpuMonitor(sustained_seconds=30.0, abort_frac=0.93)
    m2.ingest(_mk(30720), now=now - 20)
    m2.ingest(_mk(30800), now=now - 10)
    assert m2.is_under_pressure()


def test_monitor_is_under_pressure_needs_at_least_two_samples():
    m = gpu_monitor.GpuMonitor(sustained_seconds=30.0, abort_frac=0.93)
    m.ingest(_mk(30720), now=time.time() - 5)
    # One sample is not "sustained" — don't trip.
    assert not m.is_under_pressure()


def test_monitor_is_under_pressure_ignores_stale_samples():
    m = gpu_monitor.GpuMonitor(sustained_seconds=30.0, abort_frac=0.93)
    now = time.time()
    # Old samples above threshold, recent samples below — not pressure.
    m.ingest(_mk(30720), now=now - 300)
    m.ingest(_mk(30800), now=now - 290)
    m.ingest(_mk(5000), now=now - 10)
    m.ingest(_mk(6000), now=now - 5)
    assert not m.is_under_pressure()


def test_monitor_peak_used_mb_and_summary():
    m = gpu_monitor.GpuMonitor()
    now = time.time()
    m.ingest(_mk(10000), now=now - 10)
    m.ingest(_mk(25000), now=now - 5)
    m.ingest(_mk(15000), now=now)
    assert m.peak_used_mb == 25000
    summary = m.summary()
    assert summary["used_mb"] == 15000
    assert summary["total_mb"] == 32607
    assert summary["peak_used_mb"] == 25000
    assert summary["samples"] == 3


def test_monitor_emits_warn_once():
    class _Log:
        def __init__(self):
            self.events = []

        def log(self, event, **fields):
            self.events.append((event, fields))

    log = _Log()
    m = gpu_monitor.GpuMonitor(warn_frac=0.85, event_log=log)
    m.ingest(_mk(5000))   # under threshold — no warn
    assert not any(e for e, _ in log.events if e == "gpu_memory_warn")
    m.ingest(_mk(30000))  # above — warn fires
    assert sum(1 for e, _ in log.events if e == "gpu_memory_warn") == 1
    m.ingest(_mk(31000))  # still high — no duplicate warn
    assert sum(1 for e, _ in log.events if e == "gpu_memory_warn") == 1


def test_monitor_emits_bucketed_samples():
    class _Log:
        def __init__(self):
            self.samples = []

        def log(self, event, **fields):
            if event == "gpu_memory_sample":
                self.samples.append(fields["used_mb"])

    log = _Log()
    m = gpu_monitor.GpuMonitor(event_log=log)
    m.ingest(_mk(100))      # bucket 0  — logs
    m.ingest(_mk(500))      # bucket 0  — same bucket, no log
    m.ingest(_mk(2500))     # bucket 1  — logs
    m.ingest(_mk(2800))     # bucket 1  — no log
    m.ingest(_mk(4200))     # bucket 2  — logs
    assert log.samples == [100, 2500, 4200]


def test_monitor_current_empty_returns_none():
    m = gpu_monitor.GpuMonitor()
    assert m.current() is None
    assert m.peak_used_mb == 0
    assert not m.is_under_pressure()
