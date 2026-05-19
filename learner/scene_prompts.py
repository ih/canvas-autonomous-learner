"""Shared scene-change prompting for human-in-the-loop recording.

Both the locked-val recorder (offline, terminal-driven) and the
wide-verify mode of the live learner (online, dashboard-driven) need to
ask the operator to physically rearrange objects between recording
chunks. This module provides one source of truth for:

  - The instruction text the operator reads (so terminal and dashboard
    say the same thing).
  - The two delivery modes:
      * `prompt_terminal()` — blocks on stdin until the operator hits
        Enter. Used by `record_locked_val_multi_scene.py`.
      * `prompt_dashboard()` — emits a `claude_scene_change_requested`
        event identical to the one the advisor's `idle` routing emits,
        then polls `runs_dir/scene_ready.flag` exactly the way the
        orchestrator's IDLE branch already does. Used by the live
        wide-verify path so the existing dashboard banner + button
        wiring works unchanged.

Keeping these unified means: change the wording in one place, both
surfaces update; change the dashboard flag protocol once, both paths
follow it.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Optional


def scene_instruction_text(scene_idx: int, total: int, mode: str = "rearrange") -> str:
    """Return the human-facing instruction for a given scene index.

    `mode` is "initial" for the very first scene of a recording session
    (operator sets up baseline objects), "rearrange" for every
    subsequent scene (operator perturbs from the prior state).

    The text is identical between terminal and dashboard delivery so
    the operator sees consistent instructions regardless of where the
    prompt fires.
    """
    if mode not in ("initial", "rearrange"):
        raise ValueError(f"mode must be 'initial' or 'rearrange', got {mode!r}")

    if mode == "initial":
        return (
            f"SCENE 1 / {total} — initial setup\n\n"
            "Place objects in front of the robot in their starting "
            "configuration. This is the first of several distinct scenes; "
            "you'll be asked to rearrange before each one."
        )
    return (
        f"SCENE {scene_idx} / {total} — REARRANGE the scene\n\n"
        "Move objects into a configuration that is VISIBLY different from "
        "the previous scene. Suggestions:\n"
        "  - shift block positions by 5+ cm\n"
        "  - rotate or swap object orientations\n"
        "  - add, remove, or substitute an object\n\n"
        "The goal is variation across scenes — pick whatever physical "
        "changes you'd expect the model to see during real deployment."
    )


def prompt_terminal(
    scene_idx: int,
    total: int,
    *,
    mode: Optional[str] = None,
) -> None:
    """Block on stdin until the operator confirms the scene is ready.

    Used by offline scripts (record_locked_val_multi_scene.py) that
    own the terminal. Loud separators so the prompt is unmissable.

    `mode` defaults to "initial" for scene 1, "rearrange" otherwise.
    Pass explicitly to override — wide-verify seeding should pass
    "rearrange" for scene 1 so the operator perturbs the scene the
    model was trained on instead of recording held-out canvases on
    the same physical setup.
    """
    if mode is None:
        mode = "initial" if scene_idx == 1 else "rearrange"
    text = scene_instruction_text(scene_idx, total, mode=mode)
    bar = "=" * 64
    print()
    print(bar)
    for line in text.splitlines():
        print(f"  {line}")
    print(bar)
    try:
        input("Press Enter when the scene is ready (Ctrl-C to abort)... ")
    except (EOFError, KeyboardInterrupt):
        print("\nAborted before scene was confirmed.", file=sys.stderr)
        raise SystemExit(130)


def prompt_dashboard(
    runs_dir: Path,
    scene_idx: int,
    total: int,
    *,
    cycle: int,
    event_log,
    stop: Callable[[], bool] | None = None,
    heartbeat_s: float = 30.0,
    poll_interval_s: float = 1.0,
    mode: Optional[str] = None,
) -> bool:
    """Emit a scene-change request event and block until the operator
    acknowledges via the dashboard's "Scene ready" button.

    Acknowledgement protocol matches the orchestrator's IDLE branch:
      - request: emit `claude_scene_change_requested` with the scene
        instruction as `description`. The dashboard's banner / counter
        / tab-title alert all key off this event so they fire
        unchanged.
      - acknowledgement: poll for `runs_dir/scene_ready.flag` once per
        second; when it appears, delete it (so it can't double-fire),
        emit `scene_ready_acknowledged`, return True.
      - shutdown: if `stop()` returns True during the poll, return
        False without consuming the flag.

    `mode` defaults to "initial" for scene 1, "rearrange" otherwise.
    Pass explicitly if a wide-verify scene 1 should still ask for a
    rearrangement (e.g., the operator already set up scene 1 implicitly
    by being in a non-fresh learner cycle).
    """
    if mode is None:
        mode = "initial" if scene_idx == 1 else "rearrange"
    description = scene_instruction_text(scene_idx, total, mode=mode)

    flag = Path(runs_dir) / "scene_ready.flag"
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    # Consume any stale flag left over from a prior request — otherwise
    # the next pump tick would immediately succeed without the operator
    # actually rearranging anything.
    try:
        if flag.exists():
            flag.unlink()
    except OSError:
        pass

    if event_log is not None:
        event_log.log(
            "claude_scene_change_requested",
            cycle=cycle,
            description=description,
            scene_idx=scene_idx,
            scene_total=total,
            source="wide_verify",
        )

    requested_at = time.time()
    last_heartbeat = requested_at
    while True:
        if stop is not None and stop():
            return False
        if flag.exists():
            try:
                flag.unlink()
            except OSError:
                pass
            ack_t = time.time()
            if event_log is not None:
                event_log.log(
                    "scene_ready_acknowledged",
                    cycle=cycle,
                    description=description,
                    requested_at=requested_at,
                    acknowledged_at=ack_t,
                    scene_idx=scene_idx,
                    scene_total=total,
                    source="wide_verify",
                )
            return True
        now = time.time()
        if now - last_heartbeat >= heartbeat_s and event_log is not None:
            event_log.log(
                "idle_waiting_for_scene_ready",
                elapsed_s=round(now - requested_at, 1),
                source="wide_verify",
            )
            last_heartbeat = now
        time.sleep(poll_interval_s)
