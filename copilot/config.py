"""Загрузка конфигурации Sentinel Copilot (YAML + env-переопределения). Без побочных эффектов при импорте."""
from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "config.yaml"

# env-переменная -> (секция, ключ, приведение типа)
_ENV_OVERRIDES: dict[str, tuple[str, str, Any]] = {
    "COPILOT_LLM_BACKEND": ("llm", "backend", str),
    "COPILOT_LORA_PATH": ("llm", "lora_path", str),
    "COPILOT_GPU_LAYERS": ("llm", "gpu_layers", str),
    "COPILOT_SERVER_URL": ("llm", "server_url", str),
    "COPILOT_TIMEOUT_S": ("llm", "timeout_s", float),
    "COPILOT_N_CTX": ("llm", "n_ctx", int),
    "COPILOT_IDLE_UNLOAD_S": ("llm", "idle_unload_s", float),
    "COPILOT_EMBED_BACKEND": ("embed", "backend", str),
    "COPILOT_EMBED_MODEL": ("embed", "model", str),
    "COPILOT_EMBED_DEVICE": ("embed", "device", str),
    "COPILOT_CACHE_DIR": ("embed", "cache_dir", str),
    "COPILOT_TOP_K": ("rag", "top_k", int),
    "COPILOT_VLM_MODULE": ("vlm_service", "module", str),
}


def is_mock() -> bool:
    """Глобальный мок-режим всего пайплайна (SENTINEL_MOCK=1)."""
    return os.environ.get("SENTINEL_MOCK", "").strip().lower() in {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml  # лёгкая зависимость, но всё равно лениво

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must be a mapping")
    return data


def load_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Конфиг: config.yaml -> файл из COPILOT_CONFIG -> env -> overrides (dict)."""
    cfg = _read_yaml(DEFAULT_CONFIG_PATH)
    extra_path = os.environ.get("COPILOT_CONFIG")
    if extra_path:
        cfg = _deep_merge(cfg, _read_yaml(Path(extra_path)))
    for env_name, (section, key, cast) in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_name)
        if raw is None or raw == "":
            continue
        try:
            cfg.setdefault(section, {})[key] = cast(raw)
        except (TypeError, ValueError):
            continue
    model_path = os.environ.get("COPILOT_MODEL_PATH")
    if model_path:
        cfg.setdefault("llm", {})["model_candidates"] = [model_path]
    norms_dir = os.environ.get("COPILOT_NORMS_DIR")
    if norms_dir:
        cfg.setdefault("corpus", {})["catalog_path"] = str(Path(norms_dir) / "catalog.yaml")
        cfg["corpus"]["sources_glob"] = str(Path(norms_dir) / "sources" / "*.md")
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


@lru_cache(maxsize=1)
def default_config() -> dict[str, Any]:
    return load_config()


def resolve_path(path: str | os.PathLike[str] | None) -> Path | None:
    """Относительные пути — от корня репозитория."""
    if path is None or str(path) == "":
        return None
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def copilot_enabled(cfg: dict[str, Any]) -> bool:
    """COPILOT_ENABLED (env) > enabled (конфиг: true | false | mock)."""
    env = os.environ.get("COPILOT_ENABLED")
    if env is not None and env.strip() != "":
        return env.strip().lower() in {"1", "true", "yes", "on"}
    mode = str(cfg.get("enabled", "mock")).strip().lower()
    if mode in {"true", "1", "yes", "on"}:
        return True
    if mode == "mock":
        return is_mock()
    return False


def fixture_allowed(cfg: dict[str, Any]) -> bool:
    mode = str(cfg.get("corpus", {}).get("fixture_fallback", "auto")).lower()
    if mode in {"true", "1", "yes", "always"}:
        return True
    if mode in {"false", "0", "no", "never"}:
        return False
    return is_mock() or _env_flag("COPILOT_ALLOW_FIXTURE")
