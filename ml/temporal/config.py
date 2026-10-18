"""Конфигурация ml/temporal: config.yaml + переопределения из env. Без побочных эффектов при импорте."""
from __future__ import annotations

import copy
import functools
import hashlib
import json
import os
from pathlib import Path
from typing import Any

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parents[1]

TYPE_IDS = ["V01", "V02", "V03", "V04", "V05", "V06", "V07", "V08", "V99"]


def is_mock() -> bool:
    return os.environ.get("SENTINEL_MOCK", "").strip().lower() in {"1", "true", "yes", "on"}


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path) -> dict:
    import yaml  # лениво

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@functools.lru_cache(maxsize=8)
def _load_cached(override_path: str) -> dict:
    cfg = _read_yaml(PKG_DIR / "config.yaml")
    if override_path:
        cfg = _deep_merge(cfg, _read_yaml(Path(override_path)))
    return cfg


def load_config(overrides: dict | None = None) -> dict:
    """Вернуть копию конфига. ``overrides`` мерджится поверх (для тестов и CLI)."""
    cfg = copy.deepcopy(_load_cached(os.environ.get("SENTINEL_TEMPORAL_CONFIG", "")))
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def _resolve(p: str | os.PathLike) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


def cache_dir(cfg: dict) -> Path:
    base = os.environ.get("SENTINEL_CACHE_DIR") or cfg["paths"]["cache_dir"]
    return _resolve(base) / cfg["paths"].get("features_subdir", "features")


def weights_dir(cfg: dict) -> Path:
    env = os.environ.get("SENTINEL_TEMPORAL_WEIGHTS")
    if env:
        return _resolve(env)
    base = os.environ.get("SENTINEL_WEIGHTS_DIR")
    if base:
        return _resolve(base) / "temporal"
    return _resolve(cfg["paths"]["weights_dir"])


def reports_dir(cfg: dict) -> Path:
    return _resolve(cfg["paths"]["reports_dir"])


def outputs_dir(cfg: dict) -> Path:
    return _resolve(cfg["paths"]["outputs_dir"])


def load_prompts(cfg: dict) -> dict:
    return _read_yaml(_resolve_prompts(cfg))


def _resolve_prompts(cfg: dict) -> Path:
    p = Path(cfg.get("prompts_file", "prompts.yaml"))
    return p if p.is_absolute() else PKG_DIR / p


def stable_hash(obj: Any, n: int = 12) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:n]


def feature_signature(cfg: dict, backend: str) -> dict:
    """Параметры, от которых зависят кэшированные эмбеддинги изображений."""
    f = cfg["features"]
    from .features import FEATURE_VERSION, hf_revision

    return {
        "backend": backend,
        "model_id": f["model_id"] if backend != "mock" else f"mock{f['mock_dim']}",
        "revision": hf_revision(f["model_id"], cfg) if backend != "mock" else None,  # снапшот весов в HF-кэше
        "focus": f["focus"],
        "zone_views": list(f.get("zone_views") or []),
        "max_frames": int(f.get("max_frames", 1200)),  # меняет шаг сэмплинга длинных видео
        "v": FEATURE_VERSION,
    }
