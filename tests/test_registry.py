"""Registry persistence: atomic writes, swap history."""

import json

from learner.registry import Registry


def test_registry_seed_and_swap(tmp_path):
    reg_path = tmp_path / "registry.json"
    reg = Registry(reg_path)

    reg.set_baseline(
        live_checkpoint="ckpt_v0.pth",
        base_canvas_dataset="canvas_base",
        baseline_val_mse=0.005,
    )
    assert reg.live_checkpoint() == "ckpt_v0.pth"
    assert reg.baseline_val_mse() == 0.005

    reg.swap(
        new_checkpoint="ckpt_v1.pth",
        merged_canvas_dataset="canvas_merged_1",
        val_mse=0.004,
    )
    data = json.loads(reg_path.read_text())
    assert data["live_checkpoint"] == "ckpt_v1.pth"
    assert data["merged_canvas_dataset"] == "canvas_merged_1"
    assert len(data["history"]) == 1
    assert data["history"][0]["previous"] == "ckpt_v0.pth"
    assert data["history"][0]["new"] == "ckpt_v1.pth"
    assert data["history"][0]["val_mse"] == 0.004


def test_registry_reload_preserves_state(tmp_path):
    reg_path = tmp_path / "registry.json"
    r1 = Registry(reg_path)
    r1.set_baseline("ckpt.pth", "base", 0.01)
    r1.swap("ckpt2.pth", val_mse=0.009)
    r2 = Registry(reg_path)
    assert r2.live_checkpoint() == "ckpt2.pth"
    assert len(r2.load()["history"]) == 1


# ----------------------------------------- wide-verify (held-out) corpus


def test_wide_verify_dirs_start_empty(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    assert reg.wide_verify_canvas_dirs() == []
    assert reg.wide_verify_history() == []


def test_append_wide_verify_dir_records_provenance(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    reg.append_wide_verify_dir(
        "/runs/exp/wide_verify/burst_1", n_probes=12, scenes=3, cycle=4,
    )
    assert reg.wide_verify_canvas_dirs() == ["/runs/exp/wide_verify/burst_1"]
    history = reg.wide_verify_history()
    assert len(history) == 1
    entry = history[0]
    assert entry["path"] == "/runs/exp/wide_verify/burst_1"
    assert entry["n_probes"] == 12
    assert entry["scenes"] == 3
    assert entry["cycle"] == 4
    assert "t" in entry  # timestamp populated


def test_wide_verify_appends_grow_over_time(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    for i in range(3):
        reg.append_wide_verify_dir(
            f"/runs/exp/wide_verify/burst_{i}",
            n_probes=10, scenes=2, cycle=i,
        )
    dirs = reg.wide_verify_canvas_dirs()
    assert len(dirs) == 3
    assert dirs[0].endswith("burst_0")
    assert dirs[2].endswith("burst_2")


def test_wide_verify_dir_rejected_if_in_training_accumulator(tmp_path):
    """Leakage invariant: a path that's already in the training set
    cannot be added to the held-out wide-verify corpus."""
    reg = Registry(tmp_path / "registry.json")
    reg.append_canvas_dir("/runs/exp/canvas/explore_1", episodes_added=20)
    import pytest
    with pytest.raises(ValueError, match="disjoint from training"):
        reg.append_wide_verify_dir(
            "/runs/exp/canvas/explore_1",
            n_probes=10, scenes=2, cycle=1,
        )


def test_training_dir_rejected_if_in_wide_verify(tmp_path):
    """Symmetric leakage invariant: a path that's already in the
    held-out corpus cannot become training data."""
    reg = Registry(tmp_path / "registry.json")
    reg.append_wide_verify_dir(
        "/runs/exp/wide_verify/burst_1",
        n_probes=10, scenes=2, cycle=0,
    )
    import pytest
    with pytest.raises(ValueError, match="disjoint from wide-verify"):
        reg.append_canvas_dir(
            "/runs/exp/wide_verify/burst_1", episodes_added=20,
        )


def test_wide_verify_state_survives_registry_reload(tmp_path):
    reg_path = tmp_path / "registry.json"
    r1 = Registry(reg_path)
    r1.append_wide_verify_dir(
        "/runs/exp/wide_verify/burst_1",
        n_probes=15, scenes=3, cycle=2,
    )
    r2 = Registry(reg_path)
    assert r2.wide_verify_canvas_dirs() == ["/runs/exp/wide_verify/burst_1"]
    assert r2.wide_verify_history()[0]["n_probes"] == 15
