"""Загрузка ml/detect/config.yaml (+ необязательный override через env SENTINEL_DETECT_CONFIG)."""
from __future__ import annotations

import copy
import functools
import os
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).with_name("config.yaml")


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

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: ожидался YAML-словарь")
    return data


@functools.lru_cache(maxsize=8)
def _load_cached(override_path: str) -> dict[str, Any]:
    cfg = _read_yaml(CONFIG_PATH)
    if override_path:
        cfg = deep_merge(cfg, _read_yaml(Path(override_path)))
    return cfg


def load_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Копия конфига: YAML → SENTINEL_DETECT_CONFIG → SENTINEL_DETECT_EXTRAS → overrides (тесты/тюнинг)."""
    cfg = copy.deepcopy(_load_cached(os.environ.get("SENTINEL_DETECT_CONFIG", "")))
    cfg = apply_env_extras(cfg)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def label_threshold(table: dict[str, float], label: str) -> float:
    return float(table.get(label, table.get("default", 0.5)))


def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


EXTRAS = ("tta", "tiles", "open_vocab", "finetuned", "rare_open_vocab")


def apply_env_extras(cfg: dict[str, Any]) -> dict[str, Any]:
    """SENTINEL_DETECT_EXTRAS=tta,tiles,open_vocab,finetuned — включить опциональные стадии без правки YAML."""
    raw = os.environ.get("SENTINEL_DETECT_EXTRAS", "")
    flags = {x.strip().lower() for x in raw.split(",") if x.strip()}
    if not flags:
        return cfg
    unknown = flags - set(EXTRAS) - {"all"}
    if unknown:
        raise ValueError(f"SENTINEL_DETECT_EXTRAS: неизвестные флаги {sorted(unknown)}; допустимы {EXTRAS}")
    if "all" in flags:
        flags = set(EXTRAS)
    cfg = copy.deepcopy(cfg)
    if "tta" in flags:
        cfg["tta"]["hflip"] = True
    if "tiles" in flags:
        cfg["tiling"]["enabled"] = True
    if "open_vocab" in flags:
        cfg["open_vocab"]["enabled"] = True
    if "rare_open_vocab" in flags:
        cfg["inventory"]["mode"] = "rare_open_vocab"
    if "finetuned" in flags and "onnx_finetuned" not in cfg["closed_set"]["backends"]:
        cfg["closed_set"]["backends"] = ["onnx_finetuned", *cfg["closed_set"]["backends"]]
    return cfg
