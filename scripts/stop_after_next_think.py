"""Watch the learner's event log; when the next `claude_think` event
fires (newer than `--threshold-t`), gracefully stop the learner.

On Windows, sends CTRL_BREAK / CTRL_C via AttachConsole + GenerateConsoleCtrlEvent
so the orchestrator's existing SIGINT handler triggers a clean shutdown
(loop exits at next stop() check, train_diffusion subprocess is killed
via the existing _kill_subprocess path, cameras released cleanly).

Usage:
    python scripts/stop_after_next_think.py \\
        --events-path runs/red_kong_1b/events_<sess>.jsonl \\
        --learner-pid 19276 \\
        --threshold-t 1777784590.67
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import sys
import time
from pathlib import Path


def _send_ctrl_break_via_attach(pid: int) -> bool:
    """Attach to target's console and generate a Ctrl+Break event.

    Returns True on success. The orchestrator's SIGTERM/SIGINT handler
    catches Ctrl+Break as SIGBREAK on Windows, which Python's signal
    module then routes — the orchestrator's handler list at
    `_install_signal_handlers` covers SIGINT and SIGTERM, but not
    SIGBREAK by default. So we ALSO try a direct os.kill with
    CTRL_C_EVENT, which arrives as SIGINT.
    """
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # Free our own console first (a process can only attach to one).
    kernel32.FreeConsole()
    if not kernel32.AttachConsole(pid):
        err = ctypes.get_last_error()
        print(f"[stop-watcher] AttachConsole({pid}) failed errno={err}",
              file=sys.stderr)
        return False
    # Suppress the Ctrl+C handler in OUR process so we don't kill ourselves.
    kernel32.SetConsoleCtrlHandler(None, True)
    # GenerateConsoleCtrlEvent(CTRL_C_EVENT=0, group=0). Group 0 = all
    # processes attached to this console (i.e. the learner).
    ok = bool(kernel32.GenerateConsoleCtrlEvent(0, 0))
    if not ok:
        err = ctypes.get_last_error()
        print(f"[stop-watcher] GenerateConsoleCtrlEvent failed errno={err}",
              file=sys.stderr)
    # Detach so we can re-attach if we need to retry.
    kernel32.FreeConsole()
    kernel32.SetConsoleCtrlHandler(None, False)
    return ok


def _try_oskill_ctrl_c(pid: int) -> bool:
    """Fallback: Python's os.kill with CTRL_C_EVENT. Only works if the
    target was launched in our process group (CREATE_NEW_PROCESS_GROUP).
    Usually fails for processes started from a different shell.
    """
    try:
        os.kill(pid, signal.CTRL_C_EVENT)
        return True
    except (OSError, SystemError) as e:
        print(f"[stop-watcher] os.kill CTRL_C_EVENT failed: {e}",
              file=sys.stderr)
        return False


def _process_alive(pid: int) -> bool:
    """Cheap liveness check via OpenProcess."""
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return False
    exit_code = ctypes.c_ulong()
    kernel32.GetExitCodeProcess(h, ctypes.byref(exit_code))
    kernel32.CloseHandle(h)
    STILL_ACTIVE = 259
    return exit_code.value == STILL_ACTIVE


def _wait_for_next_think(
    events_path: Path,
    threshold_t: float,
    poll_s: float = 5.0,
    log_path: Path | None = None,
) -> dict | None:
    """Poll the events file until a `claude_think` event with t >
    threshold_t appears. Returns the parsed event dict, or None on
    KeyboardInterrupt.
    """
    last_size = 0
    while True:
        try:
            stat = events_path.stat()
        except FileNotFoundError:
            time.sleep(poll_s)
            continue
        if stat.st_size <= last_size:
            time.sleep(poll_s)
            continue
        # Read the whole file (small) — events are append-only and the
        # file rotates per session, so size should stay manageable.
        with open(events_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    ev.get("event") == "claude_think"
                    and ev.get("t", 0) > threshold_t
                ):
                    return ev
        last_size = stat.st_size
        if log_path is not None:
            with open(log_path, "a") as lf:
                lf.write(
                    f"[{time.strftime('%H:%M:%S')}] no new think yet "
                    f"(file size {stat.st_size}, threshold {threshold_t})\n"
                )
        time.sleep(poll_s)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--events-path", required=True, type=Path)
    p.add_argument("--learner-pid", required=True, type=int)
    p.add_argument("--threshold-t", required=True, type=float,
                   help="Latest claude_think timestamp BEFORE we started "
                        "watching. We trigger on the NEXT one after this.")
    p.add_argument("--log-path", type=Path, default=None,
                   help="Optional path to write progress messages.")
    args = p.parse_args()

    log = args.log_path
    def _log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line)
        if log is not None:
            with open(log, "a") as f:
                f.write(line + "\n")

    _log(f"watching {args.events_path}")
    _log(f"learner pid={args.learner_pid}, threshold_t={args.threshold_t}")

    if not _process_alive(args.learner_pid):
        _log(f"learner pid {args.learner_pid} is not alive — exiting")
        return 1

    next_think = _wait_for_next_think(
        args.events_path, args.threshold_t, log_path=log,
    )
    if next_think is None:
        _log("watcher interrupted before next think fired")
        return 130
    _log(f"next claude_think detected: cycle={next_think.get('cycle')} "
         f"t={next_think.get('t')}")

    # Try the cross-console-group AttachConsole approach first; that's
    # the only one that reliably works for processes spawned by a
    # different PowerShell session.
    ok = _send_ctrl_break_via_attach(args.learner_pid)
    if not ok:
        _log("AttachConsole failed; trying os.kill CTRL_C_EVENT")
        ok = _try_oskill_ctrl_c(args.learner_pid)

    if not ok:
        _log("all signal mechanisms failed — manual stop required")
        return 2

    # Wait up to 60s for the learner to exit.
    for _ in range(60):
        time.sleep(1.0)
        if not _process_alive(args.learner_pid):
            _log(f"learner exited gracefully")
            return 0
    _log("learner still alive 60s after signal — may need manual taskkill")
    return 3


if __name__ == "__main__":
    sys.exit(main())
