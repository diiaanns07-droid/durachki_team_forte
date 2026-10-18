"""Загрузка config.yaml + env-переопределения. Без побочных эффектов при импорте."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Optional

_HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = _HERE / "config.yaml"
PROMPTS_DIR = _HERE / "prompts"

_cache: dict[str, dict[str, Any]] = {}


def _truthy(v: Optional[str]) -> bool:
    return str(v or "").strip().lower() in {"1", "true", "yes", "on"}


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml  # лениво

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must be a mapping")
    return data


def load_config(path: Optional[str] = None, overrides: Optional[dict[str, Any]] = None,
                use_env: bool = True) -> dict[str, Any]:
    """Читает YAML (кэшируется), применяет env и явные overrides. Возвращает копию."""
    p = Path(path or (os.environ.get("SENTINEL_VLM_CONFIG") if use_env else None) or DEFAULT_CONFIG_PATH)
    key = str(p.resolve())
    if key not in _cache:
        base = _read_yaml(DEFAULT_CONFIG_PATH)
        if p.resolve() != DEFAULT_CONFIG_PATH:
            base = deep_merge(base, _read_yaml(p))  # частичный yaml поверх дефолтного
        _cache[key] = base
    cfg = copy.deepcopy(_cache[key])
    if use_env:
        cfg = apply_env(cfg)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def apply_env(cfg: dict[str, Any]) -> dict[str, Any]:
    env = os.environ
    if env.get("SENTINEL_VLM_BACKEND"):
        cfg["backend"] = env["SENTINEL_VLM_BACKEND"].strip()
    if _truthy(env.get("SENTINEL_MOCK")):  # глобальный mock сильнее: гарантирует прогон без весов (CI)
        cfg["backend"] = "mock"
    if env.get("SENTINEL_VLM_PRESET"):
        cfg["preset"] = env["SENTINEL_VLM_PRESET"].strip()
    if env.get("SENTINEL_VLM_LANG"):
        cfg["lang"] = env["SENTINEL_VLM_LANG"].strip()
    if env.get("SENTINEL_VLM_URL"):
        cfg.setdefault("llamacpp_server", {})["url"] = env["SENTINEL_VLM_URL"].strip()
    if env.get("SENTINEL_VLM_MOCK_MODE"):
        cfg.setdefault("mock", {})["mode"] = env["SENTINEL_VLM_MOCK_MODE"].strip()
    if env.get("SENTINEL_VLM_MODEL"):
        # путь к локальной папке / HF id (transformers) или имя .gguf (llama.cpp)
        preset = cfg.setdefault("presets", {}).setdefault(cfg.get("preset", "custom"), {})
        model = env["SENTINEL_VLM_MODEL"].strip()
        if model.lower().endswith(".gguf"):
            preset.setdefault("llamacpp", {})["model_file"] = model
        else:
            preset.setdefault("transformers", {})["model"] = model
    return cfg


def preset_cfg(cfg: dict[str, Any], section: str) -> dict[str, Any]:
    """Параметры активного пресета для секции (transformers | llamacpp)."""
    preset = (cfg.get("presets") or {}).get(cfg.get("preset", ""), {}) or {}
    return dict(preset.get(section) or {})


def clear_cache() -> None:
    _cache.clear()
