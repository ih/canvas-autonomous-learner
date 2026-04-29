"""Smoke tests for dashboard.build_state_payload.

Builds a synthetic events.jsonl in a temp runs dir and asserts the
payload shape — no HTTP server, no sockets, just the aggregation logic
that the dashboard exposes at `/api/state`.
"""

import json
import sys
from pathlib import Path

import pytest

# Pretend scripts/ is on sys.path for the test
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import dashboard  # noqa: E402


def _write_events(runs_dir: Path, session: str, events: list[dict]) -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"events_{session}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def test_empty_runs_returns_null_session(tmp_path):
    payload = dashboard.build_state_payload(tmp_path)
    assert payload == {"session": None, "probes": [], "events": []}


def test_payload_has_new_top_level_fields(tmp_path):
    runs = tmp_path / "runs"
    events = [
        {"t": 1.0, "event": "experiment_start", "tau_low": 0.02, "tau_high": 0.06,
         "active_range": [-20.0, 20.0]},
        {"t": 1.1, "event": "range_init", "full": [-60.0, 60.0], "active": [-20.0, 20.0]},
        {"t": 1.2, "event": "state", "state": "IDLE", "iteration": 1, "cycle": 0, "total_eps": 0},
        {"t": 2.0, "event": "state", "state": "EXPLORE", "iteration": 2, "cycle": 0, "total_eps": 0},
        {"t": 2.1, "event": "cycle_start", "cycle": 0, "total_eps": 0, "active_range": [-20, 20]},
        {"t": 2.2, "event": "explore_sub_bursts_planned", "cycle": 0, "burst": 50,
         "sub_bursts": [{"n_eps": 30, "range": [-10, 0]}, {"n_eps": 20, "range": [0, 10]}]},
        {"t": 2.3, "event": "subprocess_start", "tag": "train_diffusion"},
        {"t": 2.9, "event": "retrain_start", "cycle": 0, "epochs": 100},
        {"t": 5.0, "event": "retrain_done", "checkpoint": "ft.pth",
         "train_val_mse": 0.02, "locked_val_mse": 0.10},
        {"t": 5.1, "event": "locked_val_measured", "cycle": 0, "total_eps": 50,
         "locked_val_mse": 0.10, "train_val_mse": 0.02, "accepted": True},
        {"t": 5.2, "event": "cycle_start", "cycle": 1, "total_eps": 50,
         "active_range": [-20, 20]},
        {"t": 5.5, "event": "range_expanded", "cycle": 1, "total_eps": 50,
         "old_range": [-20, 20], "new_range": [-30, 30]},
        {"t": 6.0, "event": "probe", "cycle": 1, "action": 1, "mse": 0.03,
         "motor_state": [5.0, 0, 0, 0, 0, 0], "state_key": "0"},
        {"t": 6.1, "event": "probe", "cycle": 1, "action": 2, "mse": 0.04,
         "motor_state": [-12.0, 0, 0, 0, 0, 0], "state_key": "0"},
        {"t": 6.2, "event": "probe", "cycle": 1, "action": 3, "mse": 0.02,
         "motor_state": [18.0, 0, 0, 0, 0, 0], "state_key": "0"},
    ]
    _write_events(runs, "20260414_120000", events)

    payload = dashboard.build_state_payload(runs)

    assert payload["session"] == "20260414_120000"
    # Experiment aggregates
    assert payload["cycle_count"] == 2  # cycles 0 and 1 started
    assert payload["retrain_count"] == 1
    assert payload["total_eps"] == 50
    assert payload["active_range"] == [-30, 30]  # last range_expanded wins
    assert payload["range_full"] == [-60.0, 60.0]
    assert payload["thresholds"]["tau_low"] == 0.02
    assert payload["thresholds"]["tau_high"] == 0.06
    assert payload["last_retrain_duration_s"] == pytest.approx(2.1, rel=1e-3)

    # Locked val trajectory
    assert len(payload["locked_val_history"]) == 1
    assert payload["locked_val_history"][0]["locked_val"] == 0.10

    # Range history
    assert len(payload["range_history"]) == 1
    assert payload["range_history"][0]["new_range"] == [-30, 30]

    # Sub-bursts from most recent EXPLORE
    assert len(payload["last_explore_sub_bursts"]) == 2
    assert payload["last_explore_sub_bursts"][0]["n_eps"] == 30

    # Heatmap built from the 3 probes (12 bins over [-60, 60] → bin width 10)
    hm = payload["heatmap"]
    assert len(hm["state_bins"]) == 13
    assert len(hm["cells"]) == 3  # 3 distinct (bin, action) pairs
    assert hm["max_mse"] > 0
    # Coverage histogram counts
    cov = payload["coverage"]
    assert sum(cov["counts"]) == 3

    # Current phase should reflect the most recent subprocess_start tag
    # (train_diffusion), since there's no later shutdown/explore_start.
    # Events after subprocess_start are probes + range events, which
    # don't change the phase → still train_diffusion.
    assert payload["current_phase"] == "train_diffusion"


def test_phase_pill_explore(tmp_path):
    runs = tmp_path / "runs"
    events = [
        {"t": 1.0, "event": "state", "state": "EXPLORE", "iteration": 1},
        {"t": 1.1, "event": "explore_start", "repo_id": "auto/foo", "episodes": 50},
    ]
    _write_events(runs, "x", events)
    payload = dashboard.build_state_payload(runs)
    assert payload["current_phase"] == "explore"


def test_arm_a_result_loaded_when_present(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "arm_a_result.json").write_text(json.dumps({
        "val_mse_visual": 0.0375,
    }))
    _write_events(runs, "s", [{"t": 0, "event": "state", "state": "IDLE"}])
    payload = dashboard.build_state_payload(runs)
    assert payload["arm_a_locked_val_mse"] == 0.0375


def test_heatmap_skips_probes_without_motor_state(tmp_path):
    runs = tmp_path / "runs"
    events = [
        {"t": 1.0, "event": "state", "state": "IDLE"},
        # Legacy probe without motor_state — ignored
        {"t": 2.0, "event": "probe", "action": 1, "mse": 0.5, "state_key": "0"},
        # New-format probe
        {"t": 2.1, "event": "probe", "action": 2, "mse": 0.1,
         "motor_state": [15.0, 0, 0, 0, 0, 0], "state_key": "0"},
    ]
    _write_events(runs, "legacy", events)
    payload = dashboard.build_state_payload(runs)
    # Only the new-format probe contributes to the heatmap
    assert sum(c["n"] for c in payload["heatmap"]["cells"]) == 1


# --------------------------------------------- training progress + probe gallery

def test_training_progress_in_flight(tmp_path):
    runs = tmp_path / "runs"
    events = [
        {"t": 1.0, "event": "experiment_start"},
        {"t": 2.0, "event": "cycle_start", "cycle": 0, "total_eps": 0},
        {"t": 3.0, "event": "retrain_start", "cycle": 0, "epochs": 300,
         "from_scratch": True, "num_accumulated_dirs": 1, "total_eps": 50},
        {"t": 3.1, "event": "subprocess_start", "tag": "train_diffusion"},
        {"t": 3.2, "event": "training_dataset_size",
         "train_canvases": 45, "val_canvases": 5},
        {"t": 3.3, "event": "training_progress",
         "epoch": 1, "total_epochs": 300, "train_loss": 0.98, "val_loss": 0.95},
        {"t": 3.4, "event": "training_progress",
         "epoch": 2, "total_epochs": 300, "train_loss": 0.91, "val_loss": 0.88},
        {"t": 3.5, "event": "training_progress",
         "epoch": 3, "total_epochs": 300, "train_loss": 0.85, "val_loss": 0.82,
         "lr": 3e-4, "best_val": 0.82},
        # No retrain_done yet → in progress
    ]
    _write_events(runs, "train_live", events)
    payload = dashboard.build_state_payload(runs)

    t = payload["training"]
    assert t["in_progress"] is True
    assert t["cycle"] == 0
    assert t["epochs_target"] == 300
    assert t["from_scratch"] is True
    assert t["num_accumulated_dirs"] == 1
    assert t["dataset_size"] == {"train_canvases": 45, "val_canvases": 5}
    assert len(t["progress"]) == 3
    assert t["progress"][0]["epoch"] == 1
    assert t["progress"][0]["train_loss"] == pytest.approx(0.98)
    assert t["progress"][-1]["epoch"] == 3
    assert t["progress"][-1]["best_val"] == pytest.approx(0.82)


def test_training_progress_completed(tmp_path):
    runs = tmp_path / "runs"
    events = [
        {"t": 1.0, "event": "retrain_start", "cycle": 0, "epochs": 100,
         "from_scratch": True},
        {"t": 1.1, "event": "training_progress",
         "epoch": 1, "total_epochs": 100, "train_loss": 0.5, "val_loss": 0.55},
        {"t": 1.2, "event": "training_progress",
         "epoch": 2, "total_epochs": 100, "train_loss": 0.4, "val_loss": 0.45},
        {"t": 2.0, "event": "retrain_done", "checkpoint": "ft.pth",
         "train_val_mse": 0.04, "locked_val_mse": 0.10},
        {"t": 2.1, "event": "locked_val_measured", "cycle": 0, "total_eps": 50,
         "locked_val_mse": 0.10, "train_val_mse": 0.04, "accepted": True},
    ]
    _write_events(runs, "train_done", events)
    payload = dashboard.build_state_payload(runs)

    t = payload["training"]
    assert t["in_progress"] is False
    assert len(t["progress"]) == 2


def test_training_progress_only_reflects_most_recent_retrain(tmp_path):
    runs = tmp_path / "runs"
    events = [
        # Cycle 0 retrain (completed, 2 epochs)
        {"t": 1.0, "event": "retrain_start", "cycle": 0, "epochs": 100},
        {"t": 1.1, "event": "training_progress",
         "epoch": 1, "total_epochs": 100, "train_loss": 0.5, "val_loss": 0.55},
        {"t": 1.2, "event": "training_progress",
         "epoch": 2, "total_epochs": 100, "train_loss": 0.4, "val_loss": 0.45},
        {"t": 1.3, "event": "retrain_done", "checkpoint": "ft0.pth",
         "train_val_mse": 0.04, "locked_val_mse": 0.10},
        # Cycle 1 retrain (in flight, 1 epoch so far)
        {"t": 2.0, "event": "retrain_start", "cycle": 1, "epochs": 50,
         "from_scratch": False},
        {"t": 2.1, "event": "training_progress",
         "epoch": 1, "total_epochs": 50, "train_loss": 0.3, "val_loss": 0.35},
    ]
    _write_events(runs, "train_two", events)
    payload = dashboard.build_state_payload(runs)

    t = payload["training"]
    # Progress reflects ONLY the most recent retrain, not cycle 0's
    assert t["cycle"] == 1
    assert t["epochs_target"] == 50
    assert t["in_progress"] is True
    assert len(t["progress"]) == 1
    assert t["progress"][0]["train_loss"] == pytest.approx(0.3)


def test_latest_action_canvases_returns_last_n(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    ex_dir = runs / "examples_testsession"
    ex_dir.mkdir()
    # Mix of new + legacy filenames to exercise back-compat glob.
    names = [f"action_canvas_00{i}.png" for i in range(5)] + [
        "probe_005.png", "probe_006.png",
    ]
    for i, name in enumerate(names):
        p = ex_dir / name
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([i]) * 8)
        atime = 1000000.0 + i
        import os
        os.utime(p, (atime, atime))

    _write_events(runs, "testsession", [
        {"t": 0, "event": "state", "state": "IDLE"},
    ])
    payload = dashboard.build_state_payload(runs)
    # Newest first; legacy `probe_*` files are still included.
    assert payload["latest_action_canvases"] == [
        "probe_006.png", "probe_005.png",
        "action_canvas_004.png", "action_canvas_003.png", "action_canvas_002.png",
    ]
    assert payload["latest_action_canvas"] == "probe_006.png"


def test_probe_counts_per_cycle_and_total(tmp_path):
    runs = tmp_path / "runs"
    events = [
        {"t": 1.0, "event": "cycle_start", "cycle": 0, "total_eps": 0},
        {"t": 1.1, "event": "probe", "cycle": 0, "action": 1, "mse": 0.1,
         "motor_state": [0, 0, 0, 0, 0, 0], "state_key": "0"},
        {"t": 1.2, "event": "probe", "cycle": 0, "action": 2, "mse": 0.1,
         "motor_state": [5, 0, 0, 0, 0, 0], "state_key": "0"},
        {"t": 2.0, "event": "cycle_start", "cycle": 1, "total_eps": 50},
        {"t": 2.1, "event": "probe", "cycle": 1, "action": 1, "mse": 0.05,
         "motor_state": [10, 0, 0, 0, 0, 0], "state_key": "0"},
    ]
    _write_events(runs, "counts", events)
    payload = dashboard.build_state_payload(runs)
    # Current cycle is 1 (cycle_count=2, current=2-1=1)
    assert payload["probes_total"] == 3
    assert payload["probes_this_cycle"] == 1  # only the probe in cycle 1


def test_explore_progress_in_flight(tmp_path):
    runs = tmp_path / "runs"
    events = [
        {"t": 1.0, "event": "experiment_start"},
        {"t": 2.0, "event": "explore_start", "repo_id": "auto/demo", "episodes": 60},
        {"t": 2.5, "event": "explore_episode_progress", "repo_id": "auto/demo",
         "episode_index": 0, "total_episodes": 60},
        {"t": 3.0, "event": "explore_episode_progress", "repo_id": "auto/demo",
         "episode_index": 1, "total_episodes": 60},
        {"t": 3.5, "event": "explore_episode_progress", "repo_id": "auto/demo",
         "episode_index": 5, "total_episodes": 60},
    ]
    _write_events(runs, "explore_live", events)
    payload = dashboard.build_state_payload(runs)
    ex = payload["explore"]
    assert ex["in_progress"] is True
    assert ex["episode_index"] == 5
    assert ex["total_episodes"] == 60
    assert ex["repo_id"] == "auto/demo"


def test_explore_progress_completed(tmp_path):
    runs = tmp_path / "runs"
    events = [
        {"t": 2.0, "event": "explore_start", "repo_id": "auto/demo", "episodes": 60},
        {"t": 3.0, "event": "explore_episode_progress",
         "episode_index": 59, "total_episodes": 60},
        {"t": 4.0, "event": "explore_done", "repo_id": "auto/demo"},
    ]
    _write_events(runs, "explore_done", events)
    payload = dashboard.build_state_payload(runs)
    ex = payload["explore"]
    assert ex["in_progress"] is False
    assert ex["episode_index"] == 59


def test_live_episode_counter_during_explore(tmp_path):
    """Regression: while the learner is blocked inside run_single_action_record,
    the dashboard's total_eps / episodes_this_cycle counters should tick up
    via explore_episode_progress events, not wait for the next cycle_start.
    """
    runs = tmp_path / "runs"
    events = [
        {"t": 1.0, "event": "experiment_start"},
        # Cycle 0 starts at total_eps = 0
        {"t": 2.0, "event": "cycle_start", "cycle": 0, "total_eps": 0},
        {"t": 3.0, "event": "explore_start", "repo_id": "auto/demo-0", "episodes": 60},
        # 7 episodes done so far in this sub-burst
        {"t": 3.1, "event": "explore_episode_progress", "episode_index": 0, "total_episodes": 60},
        {"t": 3.2, "event": "explore_episode_progress", "episode_index": 3, "total_episodes": 60},
        {"t": 3.3, "event": "explore_episode_progress", "episode_index": 6, "total_episodes": 60},
    ]
    _write_events(runs, "live_counter", events)
    payload = dashboard.build_state_payload(runs)
    assert payload["episodes_this_cycle"] == 7  # 6+1 (index → count)
    assert payload["total_eps"] == 7             # baseline 0 + 7


def test_live_episode_counter_across_sub_bursts(tmp_path):
    """Multiple sub-bursts within one cycle: each completed sub-burst's
    n_eps should roll into the accumulator and the next sub-burst's
    progress should restart from 0.
    """
    runs = tmp_path / "runs"
    events = [
        {"t": 1.0, "event": "cycle_start", "cycle": 2, "total_eps": 100},
        # First sub-burst: 30 eps, all completed
        {"t": 1.1, "event": "explore_start", "repo_id": "auto/a", "episodes": 30},
        {"t": 1.2, "event": "explore_episode_progress", "episode_index": 29},
        {"t": 1.3, "event": "explore_done", "repo_id": "auto/a"},
        # Second sub-burst: 20 eps, currently at index 4 (so 5 done)
        {"t": 2.0, "event": "explore_start", "repo_id": "auto/b", "episodes": 20},
        {"t": 2.1, "event": "explore_episode_progress", "episode_index": 4},
    ]
    _write_events(runs, "multi_sub", events)
    payload = dashboard.build_state_payload(runs)
    # Cycle baseline 100 + 30 completed + 5 in-flight = 135
    assert payload["total_eps"] == 135
    assert payload["episodes_this_cycle"] == 35


def test_live_episode_counter_resets_per_cycle(tmp_path):
    runs = tmp_path / "runs"
    events = [
        {"t": 1.0, "event": "cycle_start", "cycle": 0, "total_eps": 0},
        {"t": 1.1, "event": "explore_start", "repo_id": "a", "episodes": 30},
        {"t": 1.2, "event": "explore_episode_progress", "episode_index": 29},
        {"t": 1.3, "event": "explore_done", "repo_id": "a"},
        # cycle_start wipes the per-cycle counters
        {"t": 2.0, "event": "cycle_start", "cycle": 1, "total_eps": 30},
        {"t": 2.1, "event": "explore_start", "repo_id": "b", "episodes": 40},
        {"t": 2.2, "event": "explore_episode_progress", "episode_index": 2},
    ]
    _write_events(runs, "reset", events)
    payload = dashboard.build_state_payload(runs)
    # Baseline 30 + 3 in-flight from new sub-burst = 33
    assert payload["total_eps"] == 33
    assert payload["episodes_this_cycle"] == 3


def test_streaming_action_regex_matches_recorder_log():
    """Regression: the regex should match `action N/M joint=X dir=Y target=Z`
    lines emitted by record_continuous.py's logger per action. Replaces
    the old _EPISODE_DONE_RE test that matched the now-retired legacy
    `Recording episode N` line.
    """
    from learner.explorer import _STREAM_ACTION_RE
    samples = [
        # Standard streaming action line
        (
            "2026-04-29 08:12:34 INFO root action 1/60 joint=shoulder_pan dir=positive "
            "target=10.50 pre_settle=2 action=10 wall=2.30s",
            [(1, 60, "shoulder_pan", "positive")],
        ),
        # Different joint + direction
        (
            "action 47/60 joint=elbow_flex dir=negative target=70.00 "
            "pre_settle=2 action=10 wall=2.10s",
            [(47, 60, "elbow_flex", "negative")],
        ),
        # Unrelated line → no match
        ("INFO 2026-04-13 ls\\utils.py: Start recording", []),
        # Legacy `Recording episode N` should NOT match the new regex
        ("INFO 2026-04-14 ls\\utils.py:227 Recording episode 0", []),
    ]
    for line, expected in samples:
        matches = [
            (int(m.group(1)), int(m.group(2)), m.group(3), m.group(4))
            for m in _STREAM_ACTION_RE.finditer(line)
        ]
        assert matches == expected, f"{line!r}: got {matches}, want {expected}"
