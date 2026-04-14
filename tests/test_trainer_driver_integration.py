"""Subprocess-wiring integration tests for trainer_driver.

These tests actually shell out to `canvas-world-model/create_dataset.py` and
`combine_datasets.py`. They verify the CLI commands we construct match the
real scripts' interfaces — the kind of bug unit tests can't catch. They're
slow (seconds to tens of seconds) and require a populated local LeRobot cache,
so they're marked `slow` and skipped if prerequisites are missing.

Run with:
    pytest tests/test_trainer_driver_integration.py -m slow
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from learner import trainer_driver
from learner.events import EventLog


CWM = Path("C:/Projects/canvas-world-model")
LEROBOT_CACHE = (
    Path.home() / ".cache" / "huggingface" / "lerobot"
    / "irvinh" / "single-action-shoulder-pan-10"
)
BASE_CANVAS = CWM / "local/datasets/single-action-shoulder-pan-700-combined"


def _cfg_for(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        paths=SimpleNamespace(
            canvas_world_model=str(CWM),
            canvas_out=str(tmp_path / "canvas"),
            ckpt_dir=str(tmp_path / "checkpoints"),
            runs_dir=str(tmp_path / "runs"),
            python="C:/Projects/pythonenv-lerobot/Scripts/python.exe",
        ),
        cadence=SimpleNamespace(ft_epochs=1),
        thresholds=SimpleNamespace(val_guard=1.25),
    )


pytestmark = pytest.mark.slow


@pytest.mark.skipif(not LEROBOT_CACHE.exists(), reason="lerobot cache missing")
def test_build_canvases_from_real_lerobot_dataset(tmp_path):
    """create_dataset.py is the first step of the retrain pipeline —
    verify our subprocess wiring actually produces canvases.
    """
    cfg = _cfg_for(tmp_path)
    out = tmp_path / "canvas" / "new_batch"
    log = EventLog(cfg.paths.runs_dir, session="itest_build")

    trainer_driver.build_canvases(
        cfg,
        new_lerobot_dir=LEROBOT_CACHE,
        output_dir=out,
        event_log=log,
    )

    assert out.exists()
    pngs = list(out.glob("canvas_*.png"))
    assert len(pngs) > 0, f"expected at least one canvas in {out}"
    assert (out / "dataset_meta.json").exists()


@pytest.mark.skipif(
    not LEROBOT_CACHE.exists() or not BASE_CANVAS.exists(),
    reason="lerobot cache or base canvas missing",
)
def test_build_and_combine_end_to_end(tmp_path):
    """Chains build_canvases -> combine_datasets to validate that our merge
    step accepts the output of the build step as input.
    """
    cfg = _cfg_for(tmp_path)
    new_batch = tmp_path / "canvas" / "new_batch"
    merged = tmp_path / "canvas" / "merged"
    log = EventLog(cfg.paths.runs_dir, session="itest_merge")

    trainer_driver.build_canvases(
        cfg, new_lerobot_dir=LEROBOT_CACHE, output_dir=new_batch, event_log=log
    )
    trainer_driver.combine_datasets(
        cfg,
        inputs=[BASE_CANVAS, new_batch],
        output_dir=merged,
        event_log=log,
    )

    assert merged.exists()
    assert (merged / "dataset_meta.json").exists()
    merged_pngs = list(merged.glob("canvas_*.png"))
    base_pngs = list(BASE_CANVAS.glob("canvas_*.png"))
    new_pngs = list(new_batch.glob("canvas_*.png"))
    # Combined dataset should have at least as many canvases as the base alone.
    assert len(merged_pngs) >= len(base_pngs)
    assert len(merged_pngs) >= len(new_pngs)
