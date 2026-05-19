"""Unit tests for the shared scene-prompts helper."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from learner import scene_prompts


# ----------------------------------------------- scene_instruction_text


def test_initial_text_mentions_setup_not_rearrange():
    text = scene_prompts.scene_instruction_text(1, 5, mode="initial")
    assert "SCENE 1 / 5" in text
    assert "initial setup" in text
    # Initial mode is for the FIRST scene — no "REARRANGE" copy.
    assert "REARRANGE" not in text


def test_rearrange_text_mentions_visible_difference():
    text = scene_prompts.scene_instruction_text(3, 5, mode="rearrange")
    assert "SCENE 3 / 5" in text
    assert "REARRANGE" in text
    assert "VISIBLY different" in text


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        scene_prompts.scene_instruction_text(1, 5, mode="bogus")


def test_prompt_dashboard_scene1_rearrange_override(tmp_path: Path):
    """When wide-verify forces scene 1 into rearrange mode, the
    dashboard request description must carry REARRANGE copy — proving
    the operator is being asked to perturb, not to record on the
    trained scene as-is."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    log = _RecordingEventLog()

    def drop_flag_after():
        time.sleep(0.1)
        (runs_dir / "scene_ready.flag").write_text("1")

    t = threading.Thread(target=drop_flag_after, daemon=True)
    t.start()
    ok = scene_prompts.prompt_dashboard(
        runs_dir, scene_idx=1, total=3,
        cycle=0, event_log=log, poll_interval_s=0.05,
        mode="rearrange",
    )
    t.join()
    assert ok is True
    requests = [e for e in log.events if e["event"] == "claude_scene_change_requested"]
    assert len(requests) == 1
    assert "REARRANGE" in requests[0]["description"]
    assert "initial" not in requests[0]["description"].lower()


# ------------------------------------------------------ prompt_dashboard


class _RecordingEventLog:
    """Mirrors the real EventLog's contract: every entry gets a `t`
    timestamp added automatically, so callers can compare event
    timestamps the same way they do against real event-stream files."""

    def __init__(self):
        self.events: list[dict] = []

    def log(self, event: str, **fields) -> None:
        self.events.append({"event": event, "t": time.time(), **fields})


def test_prompt_dashboard_emits_request_and_blocks_until_flag(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    log = _RecordingEventLog()

    # Spawn a worker that drops the scene-ready flag after a short delay.
    def drop_flag_after():
        time.sleep(0.15)
        (runs_dir / "scene_ready.flag").write_text("1")

    t = threading.Thread(target=drop_flag_after, daemon=True)
    t.start()

    ok = scene_prompts.prompt_dashboard(
        runs_dir, scene_idx=2, total=4,
        cycle=7, event_log=log, poll_interval_s=0.05,
    )
    t.join()
    assert ok is True

    # Must emit exactly one request and one acknowledgement, in order,
    # both tagged with source=wide_verify so the dashboard's existing
    # scene-change-counts logic registers them.
    requests = [e for e in log.events if e["event"] == "claude_scene_change_requested"]
    acks = [e for e in log.events if e["event"] == "scene_ready_acknowledged"]
    assert len(requests) == 1
    assert len(acks) == 1
    assert requests[0]["cycle"] == 7
    assert requests[0]["scene_idx"] == 2
    assert requests[0]["scene_total"] == 4
    assert requests[0]["source"] == "wide_verify"
    assert "REARRANGE" in requests[0]["description"]
    assert acks[0]["scene_idx"] == 2
    # ack timestamp comes after request timestamp in event order.
    assert acks[0]["acknowledged_at"] >= requests[0]["t"]

    # Flag was consumed so it can't re-fire on the next prompt.
    assert not (runs_dir / "scene_ready.flag").exists()


def test_prompt_dashboard_consumes_stale_flag_before_polling(tmp_path: Path):
    # If a stale flag is left over from a previous request, the prompt
    # MUST delete it BEFORE emitting its own request — otherwise the
    # next poll tick would immediately succeed without the operator
    # actually rearranging anything.
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    flag = runs_dir / "scene_ready.flag"
    flag.write_text("stale")

    log = _RecordingEventLog()

    # Start the prompt in a thread and verify it doesn't immediately
    # return — it should be blocking on the new flag.
    result = {}
    def run_prompt():
        result["ok"] = scene_prompts.prompt_dashboard(
            runs_dir, scene_idx=1, total=3,
            cycle=0, event_log=log, poll_interval_s=0.05,
        )
    t = threading.Thread(target=run_prompt, daemon=True)
    t.start()
    time.sleep(0.2)  # let the prompt run a few poll ticks
    assert t.is_alive(), (
        "prompt_dashboard returned without operator action — stale flag "
        "was not consumed before polling."
    )
    # Now drop a fresh flag and let it complete.
    flag.write_text("fresh")
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert result["ok"] is True


def test_prompt_dashboard_returns_false_on_stop(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    log = _RecordingEventLog()

    stop_after = {"stop": False}
    def stop():
        return stop_after["stop"]

    def trigger_stop():
        time.sleep(0.15)
        stop_after["stop"] = True

    t = threading.Thread(target=trigger_stop, daemon=True)
    t.start()

    ok = scene_prompts.prompt_dashboard(
        runs_dir, scene_idx=1, total=3,
        cycle=0, event_log=log, stop=stop, poll_interval_s=0.05,
    )
    t.join()
    assert ok is False
    # The request was still emitted — operator may need to know an
    # interrupted wide-verify happened.
    requests = [e for e in log.events if e["event"] == "claude_scene_change_requested"]
    assert len(requests) == 1
    # No ack emitted on shutdown.
    acks = [e for e in log.events if e["event"] == "scene_ready_acknowledged"]
    assert len(acks) == 0
