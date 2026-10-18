"""Загрузка deploy/config/deploy.yaml (+ env SENTINEL_DEPLOY_CONFIG). Без побочных эффектов при импорте."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "deploy" / "config" / "deploy.yaml"


def load(path: str | os.PathLike | None = None) -> dict[str, Any]:
    import yaml  # PyYAML есть в lock L1

    p = Path(path or os.environ.get("SENTINEL_DEPLOY_CONFIG") or DEFAULT_CONFIG)
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_path"] = str(p)
    return cfg


def repo_path(value: str | os.PathLike, root: Path = REPO_ROOT) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def env_active(cond: str, environ: dict | None = None) -> bool:
    """'SENTINEL_NO_VLM=1' → True, если переменная равна значению (без учёта регистра)."""
    environ = os.environ if environ is None else environ
    key, _, val = cond.partition("=")
    return (environ.get(key.strip()) or "").strip().lower() == val.strip().lower()


def required_models(cfg: dict[str, Any], environ: dict | None = None) -> tuple[list[dict], list[dict]]:
    """→ (required, optional) с учётом unless_env: выключенная стадия не требует свою модель."""
    models = cfg.get("models") or {}
    req, opt = [], list(models.get("optional") or [])
    for m in models.get("required") or []:
        if any(env_active(c, environ) for c in m.get("unless_env") or []):
            opt.append({**m, "demoted_by_env": True})
        else:
            req.append(m)
    return req, opt
