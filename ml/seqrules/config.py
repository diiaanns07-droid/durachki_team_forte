"""Конфиг ml.seqrules: config.yaml ← SENTINEL_SEQRULES_CONFIG (файл, глубокое слияние) ← overrides."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def deep_merge(base: Mapping, override: Optional[Mapping]) -> dict:
    out = copy.deepcopy(dict(base))
    for k, v in (override or {}).items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: str | Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: ожидался словарь YAML")
    return data


def load_config(overrides: Optional[Mapping] = None) -> dict:
    cfg = _read_yaml(CONFIG_PATH)
    env = os.environ.get("SENTINEL_SEQRULES_CONFIG", "").strip()
    if env:
        cfg = deep_merge(cfg, _read_yaml(env))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def config_hash(cfg: Mapping) -> str:
    raw = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
