"""Конфиг ml/ingest: встроенные значения по умолчанию + YAML (+ env override).

Порядок: DEFAULTS <- ml/ingest/config.yaml (или $SENTINEL_INGEST_CONFIG) <- overrides.
Без PyYAML или без файла модуль работает на DEFAULTS.
"""
from __future__ import annotations

import copy
import os
import threading
from pathlib import Path
from typing import Any, Mapping, Optional

CONFIG_PATH = Path(__file__).with_name("config.yaml")
ENV_CONFIG = "SENTINEL_INGEST_CONFIG"

DEFAULTS: dict[str, Any] = {
    "sampling": {"default_fps": 5.0, "max_samples": 6000},
    "decode": {
        "backend": "auto",
        "threads": 0,
        "skip_nonref": False,
        "adaptive_skip": True,
        "budget_s": 20.0,
        "max_decode_errors": 200,
        "min_coverage": 0.5,
        "warn_coverage": 0.98,
    },
    "storage": {
        "mode": "jpeg",
        "jpeg_quality": 92,
        "max_side": 1280,
        "max_ram_mb": 512,
        "lru_size": 96,
        "lru_max_mb": 256,
        "copy_on_access": True,
        "encode_workers": 4,
    },
    "frame_at": {"cache_size": 32, "cache_mb": 128, "forward_window_s": 1.5,
                 "max_offset_s": 0.25, "gap_factor": 3.0, "min_gap_s": 0.25},
    "mock": {"allow_synthetic_uri": False},
    "coarse_fine": {
        "enabled": False,
        "coarse_step_s": 1.0,
        "coarse_frames": 0,
        "dense_fps": 5.0,
        "max_frames_per_window": 24,
        "max_windows": 6,
        "window_pad_s": 1.5,
        "merge_gap_s": 1.0,
        "min_window_s": 1.0,
        "budget_frames": 0,
        "coarse_share": 0.4,
        "activity_top_k": 3,
        "activity_window_s": 4.0,
    },
    "quality": {
        "analysis_short_side": 160,
        "blur_ksize": 5,
        "diff_threshold": 18,
        "motion_norm": 0.06,
        "dark_level": 16,
        "bright_level": 245,
        "blurry_laplacian": 40.0,
        "dark_mean": 0.15,
        "bright_mean": 0.88,
        "frozen_motion": 0.0005,
    },
    "activity": {
        "smooth_s": 0.6,
        "threshold": 0.12,
        "pad_s": 1.0,
        "min_gap_s": 1.0,
        "idle_every_n": 5,
        "active_every_n": 1,
    },
    "probe": {"min_duration_s": 1.0, "max_duration_s": 900.0, "ffprobe_timeout_s": 20.0},
}

_lock = threading.Lock()
_cache: dict[str, dict[str, Any]] = {}


def deep_merge(base: Mapping[str, Any], override: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Рекурсивное слияние словарей (override побеждает), без мутации аргументов."""
    out = copy.deepcopy(dict(base))
    for key, value in (override or {}).items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # lazy
    except ImportError:  # pragma: no cover - PyYAML есть почти везде
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except (OSError, ValueError):
        return {}
    except Exception:  # yaml.YAMLError — битый конфиг не должен ронять пайплайн
        return {}
    return data if isinstance(data, dict) else {}


def load_config(overrides: Optional[Mapping[str, Any]] = None, path: Optional[str] = None) -> dict[str, Any]:
    """Вернуть полный конфиг (копию). ``overrides`` — частичный словарь той же формы."""
    cfg_path = Path(path or os.environ.get(ENV_CONFIG) or CONFIG_PATH)
    key = str(cfg_path)
    with _lock:
        if key not in _cache:
            _cache[key] = deep_merge(DEFAULTS, _read_yaml(cfg_path) if cfg_path.is_file() else {})
        base = _cache[key]
    return deep_merge(base, overrides)


def clear_cache() -> None:
    with _lock:
        _cache.clear()
