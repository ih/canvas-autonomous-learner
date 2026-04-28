"""Entry point: `python -m learner [--config ...]`."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .config import load_config
from .orchestrator import main_loop


def _parse_args():
    p = argparse.ArgumentParser(description="canvas-autonomous-learner main loop")
    p.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "configs" / "default.yaml"),
        help="Path to YAML config file",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Use DryRunRobotInterface — no hardware access.",
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Stop after N outer-loop iterations (smoke tests / bounded runs).",
    )
    p.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Skip auto-spawning the metrics dashboard (which in turn "
             "auto-spawns the canvas-inference worker). Default behavior "
             "is to spawn both so a real run is observable out of the box.",
    )
    p.add_argument(
        "--dashboard-port",
        type=int,
        default=8765,
        help="Port the auto-spawned dashboard listens on (default 8765).",
    )
    p.add_argument(
        "--dashboard-host",
        default="0.0.0.0",
        help="Host the auto-spawned dashboard binds to (default 0.0.0.0 "
             "so other machines on the LAN can reach it; use 127.0.0.1 "
             "to restrict to localhost).",
    )
    return p.parse_args()


def _spawn_dashboard(
    config_path: Path,
    runs_dir: Path,
    port: int,
    host: str,
) -> Optional[subprocess.Popen]:
    """Spawn scripts/dashboard.py as a child subprocess.

    The dashboard itself spawns the canvas-inference worker
    (explore_inference.py) so the full observability stack comes up
    with the learner in one command. Binds to `host` (default
    0.0.0.0) so other machines on the LAN can reach the dashboard —
    there is no authentication, so keep the machine on a trusted
    network or override to 127.0.0.1.
    """
    script = Path(__file__).resolve().parent.parent / "scripts" / "dashboard.py"
    if not script.exists():
        print(f"[learner] skip dashboard: {script} not found")
        return None
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                str(script),
                "--runs-dir", str(runs_dir),
                "--port", str(port),
                "--host", str(host),
                "--config", str(config_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        # Try to resolve a LAN-reachable URL for the print line. When
        # host is 0.0.0.0 we show the machine's primary non-loopback
        # IPv4 so the user can copy-paste into a browser on another
        # device; fall back to the given host string otherwise.
        shown_host = host
        if host in ("0.0.0.0", "::"):
            try:
                import socket
                shown_host = socket.gethostbyname(socket.gethostname())
            except Exception:
                shown_host = host
        print(f"[learner] dashboard pid={proc.pid} @ http://{shown_host}:{port}/  "
              f"(bound={host}, config={config_path}, runs_dir={runs_dir})")
        return proc
    except Exception as e:
        print(f"[learner] failed to spawn dashboard: {e}")
        return None


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    if args.dry_run:
        cfg.dry_run = True

    # Auto-spawn dashboard + canvas-inference unless explicitly disabled
    # or this is a dry run. The dashboard is user-observability only;
    # a dry run typically doesn't need it.
    dashboard_proc: Optional[subprocess.Popen] = None
    if not args.no_dashboard and not args.dry_run:
        dashboard_proc = _spawn_dashboard(
            Path(args.config).resolve(),
            Path(cfg.paths.runs_dir),
            args.dashboard_port,
            args.dashboard_host,
        )

    try:
        result = main_loop(cfg, max_iterations=args.max_iterations)
        print(f"\n[learner] done: {result}")
    finally:
        if dashboard_proc is not None and dashboard_proc.poll() is None:
            print(f"[learner] terminating dashboard pid={dashboard_proc.pid}")
            dashboard_proc.terminate()
            try:
                dashboard_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                dashboard_proc.kill()


if __name__ == "__main__":
    main()
