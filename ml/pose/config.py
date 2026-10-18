"""Загрузка ml/pose/config.yaml (+ deep-merge переопределений)."""
from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

CONFIG_PATH = Path(__file__).with_name("config.yaml")


def deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Рекурсивно сливает override поверх base (base не мутируется)."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, Mapping) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml  # лениво

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: ожидался YAML-словарь")
    return data


@lru_cache(maxsize=8)
def _load_cached(extra_path: Optional[str]) -> dict[str, Any]:
    cfg = _read_yaml(CONFIG_PATH)
    if extra_path:
        cfg = deep_merge(cfg, _read_yaml(Path(extra_path)))
    return cfg


def load_config(overrides: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Конфиг модуля: config.yaml ⊕ $SENTINEL_POSE_CONFIG ⊕ overrides (dict)."""
    cfg = copy.deepcopy(_load_cached(os.environ.get("SENTINEL_POSE_CONFIG") or None))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def is_mock_mode() -> bool:
    """SENTINEL_MOCK=1 или SENTINEL_POSE_BACKEND=mock — работаем без весов."""
    if os.environ.get("SENTINEL_POSE_BACKEND", "").strip().lower() == "mock":
        return True
    return os.environ.get("SENTINEL_MOCK", "").strip().lower() in {"1", "true", "yes", "on"}


BACKEND_NAMES = ("rtmpose_onnx", "mediapipe", "mock")


def backend_override() -> Optional[str]:
    """$SENTINEL_POSE_BACKEND (rtmpose_onnx | mediapipe | mock) — один бэкенд вместо backend.order."""
    name = os.environ.get("SENTINEL_POSE_BACKEND", "").strip().lower()
    return name if name in BACKEND_NAMES else None


def enabled_mode(cfg: Mapping[str, Any]) -> str:
    """Главный флаг позы: 'auto' | 'true' | 'false' ($SENTINEL_POSE_ENABLED перекрывает config)."""
    raw: Any = os.environ.get("SENTINEL_POSE_ENABLED")
    if raw is None or str(raw).strip() == "":
        raw = cfg.get("enabled", "auto")
    val = str(raw).strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return "true"
    if val in {"0", "false", "no", "off"}:
        return "false"
    return "auto"
