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
