"""Config loading + typed access.

YAML -> nested SimpleNamespace so callers can write `cfg.thresholds.tau_high`.
Paths are resolved relative to the config file's parent directory so the
repo can be relocated without editing the YAML.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml


_PATH_KEYS = {
    "canvas_world_model",
    "canvas_robot_control",
    "robotic_foundation_model_tests",
    "base_canvas",
    "val_dataset",
    "locked_val_dataset",
    "locked_val_shoulder",
    "locked_val_elbow",
    "live_checkpoint",
    "ckpt_dir",
    "canvas_out",
    "lerobot_out",
    "runs_dir",
    "registry_file",
}


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def _resolve_paths(raw: dict, base: Path) -> dict:
    paths = raw.get("paths", {})
    for key, value in list(paths.items()):
        if key in _PATH_KEYS and isinstance(value, str):
            p = Path(value)
            if not p.is_absolute():
                p = (base / p).resolve()
            paths[key] = str(p)
    raw["paths"] = paths
    return raw


def _repo_root(config_path: Path) -> Path:
    """Repo root is the config file's parent if it's named `configs/`,
    else the config file's own parent. Keeps relative paths in YAML aligned
    with how users think about the project layout.
    """
    parent = config_path.parent
    if parent.name == "configs":
        return parent.parent
    return parent


def load_config(config_path: str | Path) -> SimpleNamespace:
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    raw = _resolve_paths(raw, _repo_root(config_path))
    cfg = _to_ns(raw)
    cfg._config_path = str(config_path)
    return cfg
