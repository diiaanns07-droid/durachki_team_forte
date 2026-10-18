"""Загрузка конфига ml.scene: config.yaml + env-override (глубокое слияние)."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Optional

_DEFAULT_PATH = Path(__file__).with_name("config.yaml")
_CACHE: dict[str, dict[str, Any]] = {}


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must be a mapping")
    return data


def load_config(override: Optional[dict[str, Any]] = None, path: Optional[str] = None) -> dict[str, Any]:
    """Базовый config.yaml ← файл из SENTINEL_SCENE_CONFIG (или ``path``) ← ``override``."""
    extra_path = path or os.environ.get("SENTINEL_SCENE_CONFIG")
    key = str(extra_path or "")
    if key not in _CACHE:
        cfg = _read_yaml(_DEFAULT_PATH)
        if extra_path:
            cfg = deep_merge(cfg, _read_yaml(Path(extra_path)))
        _CACHE[key] = cfg
    cfg = copy.deepcopy(_CACHE[key])
    if override:
        cfg = deep_merge(cfg, override)
    return cfg


def is_mock() -> bool:
    return os.environ.get("SENTINEL_MOCK", "").strip().lower() in {"1", "true", "yes", "on"}
