"""Загрузка конфига оркестратора: config.yaml ← SENTINEL_PIPELINE_CONFIG ← env ← overrides."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping, Optional

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent.parent
DEFAULT_CONFIG_PATH = PIPELINE_DIR / "config.yaml"

STAGE_ORDER = ["ingest", "detect", "track", "pose", "scene", "rules", "seqrules", "temporal", "refine",
               "vlm_verify", "vlm_discover", "fusion"]
ALWAYS_ON = ("ingest", "fusion")

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _env_bool(name: str) -> Optional[bool]:
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return None
    low = val.strip().lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    return None


def deep_merge(base: dict, override: Mapping) -> dict:
    """Рекурсивно сливает override в копию base (словари сливаются, остальное заменяется)."""
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, Mapping) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _read_yaml(path: Path) -> dict:
    import yaml  # лениво: импорт пакета не должен тянуть зависимости

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: ожидался YAML-словарь")
    return data


def _apply_env(cfg: dict) -> dict:
    mock = _env_bool("SENTINEL_MOCK")
    if mock is not None:
        cfg["mock"] = mock
    device = os.environ.get("SENTINEL_DEVICE")
    if device:
        cfg.setdefault("gpu", {})["device"] = device.strip().lower()
    policy = os.environ.get("SENTINEL_VLM_POLICY")
    if policy:
        cfg.setdefault("gpu", {})["vlm_policy"] = policy.strip().lower()
    if _env_bool("SENTINEL_NO_VLM"):
        cfg.setdefault("stages", {}).update({"vlm_verify": False, "vlm_discover": False})
    fps = os.environ.get("SENTINEL_SAMPLE_FPS")
    if fps:
        try:
            cfg.setdefault("sampling", {})["base_fps"] = float(fps)
        except ValueError:
            pass
    cache = os.environ.get("SENTINEL_CACHE")
    if cache:
        low = cache.strip().lower()
        if low in _FALSE:
            cfg.setdefault("cache", {})["mode"] = "off"
        elif low in ("read_write", "from_cache", "off"):
            cfg.setdefault("cache", {})["mode"] = low
    return cfg


def load_config(path: Optional[str | os.PathLike] = None,
                overrides: Optional[Mapping[str, Any]] = None) -> dict:
    """Возвращает итоговый конфиг (dict). Не кэшируется: дёшево и всегда актуально."""
    cfg = _read_yaml(DEFAULT_CONFIG_PATH)
    extra = path or os.environ.get("SENTINEL_PIPELINE_CONFIG")
    if extra:
        cfg = deep_merge(cfg, _read_yaml(Path(extra)))
    cfg = _apply_env(cfg)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    for stage in ALWAYS_ON:
        cfg.setdefault("stages", {})[stage] = True
    return cfg


def cfg_get(cfg: Mapping, dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def cache_root() -> Path:
    return Path(os.environ.get("SENTINEL_CACHE_DIR") or (REPO_ROOT / "cache"))


def data_root() -> Path:
    return Path(os.environ.get("SENTINEL_DATA_DIR") or (REPO_ROOT / "data"))
