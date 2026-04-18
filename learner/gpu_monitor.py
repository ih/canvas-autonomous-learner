"""GPU VRAM monitoring via `nvidia-smi` polling.

The learner process itself does not import torch — training runs in a
subprocess. This module shells out to `nvidia-smi` to sample VRAM
usage, maintains a rolling window of samples, and exposes a
sustained-pressure check so the trainer driver can abort a training
subprocess that is about to saturate the card.

Why sustained-pressure, not exception-catching: on this host (RTX
5090, 32 GB) PyTorch does not reliably raise `CUDA out of memory`.
Once VRAM fills, CUDA spills to host-backed / shared memory and
training throughput collapses by 10-100x with no exception. The
orchestrator needs a proactive signal.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections import deque
from typing import Optional

_NVIDIA_SMI_CMD = [
    "nvidia-smi",
    "--query-gpu=memory.used,memory.total,utilization.gpu",
    "--format=csv,noheader,nounits",
]


def sample_gpu() -> Optional[dict]:
    """One-shot VRAM sample via `nvidia-smi`.

    Returns `{"used_mb": int, "total_mb": int, "util_pct": int,
    "used_frac": float}` on success, or `None` on any failure (missing
    binary, non-zero exit, parse error). Never raises.
    """
    try:
        result = subprocess.run(
            _NVIDIA_SMI_CMD,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    return _parse_nvidia_smi_output(result.stdout)


def _parse_nvidia_smi_output(raw: str) -> Optional[dict]:
    """Parse a single line of `nvidia-smi --format=csv,noheader,nounits`
    output. Returns None on any parse failure.
    """
    if not raw:
        return None
    line = raw.strip().splitlines()[0] if raw.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return None
    try:
        used_mb = int(parts[0])
        total_mb = int(parts[1])
        util_pct = int(parts[2])
    except ValueError:
        return None
    if total_mb <= 0:
        return None
    return {
        "used_mb": used_mb,
        "total_mb": total_mb,
        "util_pct": util_pct,
        "used_frac": used_mb / total_mb,
    }


class GpuMonitor:
    """Background thread that polls `sample_gpu()` at a fixed interval.

    Thread-safe for the subset of operations the caller needs:
    `.current()`, `.peak_used_mb`, `.is_under_pressure()`, `.summary()`.
    The internal deque uses `append`/`popleft` which are atomic in
    CPython; a Lock guards the compound reads.
    """

    def __init__(
        self,
        *,
        sample_interval_s: float = 5.0,
        window_samples: int = 300,
        warn_frac: float = 0.85,
        abort_frac: float = 0.93,
        sustained_seconds: float = 30.0,
        event_log=None,
        sampler=sample_gpu,
    ):
        self.sample_interval_s = float(sample_interval_s)
        self.warn_frac = float(warn_frac)
        self.abort_frac = float(abort_frac)
        self.sustained_seconds = float(sustained_seconds)
        self.event_log = event_log
        self._sampler = sampler
        self._samples: deque[tuple[float, dict]] = deque(maxlen=int(window_samples))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._warned = False
        self._last_bucket_mb: Optional[int] = None

    # --------------------------------------------------------- lifecycle

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="GpuMonitor", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.sample_interval_s + 2.0)
            self._thread = None

    def __enter__(self) -> "GpuMonitor":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()

    # ------------------------------------------------------------- probe

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = self._sampler()
            if sample is not None:
                self.ingest(sample)
            # Sleep in small slices so stop() is responsive.
            slept = 0.0
            while slept < self.sample_interval_s and not self._stop.is_set():
                time.sleep(min(0.5, self.sample_interval_s - slept))
                slept += 0.5

    def ingest(self, sample: dict, *, now: Optional[float] = None) -> None:
        """Insert a sample. Public for tests to seed the window without a
        live nvidia-smi.
        """
        t = time.time() if now is None else now
        with self._lock:
            self._samples.append((t, sample))
        self._maybe_emit_events(sample)

    def _maybe_emit_events(self, sample: dict) -> None:
        if self.event_log is None:
            return
        used_mb = int(sample["used_mb"])
        # Bucket-quantized sample event: emit when usage crosses a 2 GB
        # boundary so the log doesn't balloon.
        bucket = used_mb // 2048
        if self._last_bucket_mb is None or bucket != self._last_bucket_mb:
            self._last_bucket_mb = bucket
            self.event_log.log(
                "gpu_memory_sample",
                used_mb=used_mb,
                total_mb=int(sample["total_mb"]),
                util_pct=int(sample["util_pct"]),
                used_frac=float(sample["used_frac"]),
            )
        # One-shot warn event.
        if not self._warned and sample["used_frac"] >= self.warn_frac:
            self._warned = True
            self.event_log.log(
                "gpu_memory_warn",
                used_mb=used_mb,
                total_mb=int(sample["total_mb"]),
                used_frac=float(sample["used_frac"]),
                warn_frac=self.warn_frac,
            )

    # ------------------------------------------------------------ queries

    def current(self) -> Optional[dict]:
        with self._lock:
            if not self._samples:
                return None
            return dict(self._samples[-1][1])

    @property
    def peak_used_mb(self) -> int:
        with self._lock:
            if not self._samples:
                return 0
            return max(int(s[1]["used_mb"]) for s in self._samples)

    def is_under_pressure(self, threshold_frac: Optional[float] = None) -> bool:
        """True iff every sample in the last `sustained_seconds` exceeds
        the threshold. A single spike does not trigger.
        """
        threshold = self.abort_frac if threshold_frac is None else float(threshold_frac)
        cutoff = time.time() - self.sustained_seconds
        with self._lock:
            recent = [s for (t, s) in self._samples if t >= cutoff]
        if not recent:
            return False
        # Require at least 2 samples in the window so a single breach
        # doesn't trip the abort.
        if len(recent) < 2:
            return False
        return all(float(s["used_frac"]) >= threshold for s in recent)

    def summary(self) -> dict:
        cur = self.current() or {}
        return {
            "used_mb": int(cur.get("used_mb", 0)),
            "total_mb": int(cur.get("total_mb", 0)),
            "util_pct": int(cur.get("util_pct", 0)),
            "used_frac": float(cur.get("used_frac", 0.0)),
            "peak_used_mb": self.peak_used_mb,
            "samples": len(self._samples),
        }
