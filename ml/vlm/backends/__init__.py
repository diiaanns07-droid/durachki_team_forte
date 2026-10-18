"""Реестр VLM-бэкендов. Импорт тяжёлых библиотек — только при создании/загрузке конкретного бэкенда."""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..config import preset_cfg
from .base import BackendUnavailable, GenResult, VLMBackend

log = logging.getLogger(__name__)

BACKENDS = ("mock", "transformers", "llamacpp_server", "llamacpp_python")


def create_backend(name: str, cfg: dict[str, Any]) -> VLMBackend:
    name = (name or "").strip().lower()
    if name == "mock":
        from .mock import MockBackend

        m = cfg.get("mock") or {}
        return MockBackend(mode=str(m.get("mode", "neutral")), latency_s=float(m.get("latency_s", 0.0)))
    if name == "transformers":
        from .transformers_backend import TransformersBackend

        kw = dict(cfg.get("transformers") or {})
        kw.update(preset_cfg(cfg, "transformers"))
        if not kw.get("model"):
            raise BackendUnavailable("transformers: model not configured in preset")
        return TransformersBackend(**kw)
    if name == "llamacpp_server":
        from .llamacpp import LlamaCppServerBackend

        kw = dict(cfg.get("llamacpp_server") or {})
        kw.update({k: v for k, v in preset_cfg(cfg, "llamacpp").items() if k == "model_file"})
        kw["jpeg_quality"] = (cfg.get("frames") or {}).get("jpeg_quality", 85)
        return LlamaCppServerBackend(**kw)
    if name == "llamacpp_python":
        from .llamacpp import LlamaCppPythonBackend

        kw = dict(cfg.get("llamacpp_python") or {})
        kw.update(preset_cfg(cfg, "llamacpp"))
        return LlamaCppPythonBackend(**kw)
    raise BackendUnavailable(f"unknown VLM backend '{name}'")


def resolve_backend(cfg: dict[str, Any]) -> tuple[Optional[VLMBackend], str]:
    """backend: auto|off|<name>. -> (бэкенд или None, причина недоступности)."""
    name = str(cfg.get("backend", "auto")).strip().lower()
    if name in ("off", "none", "disabled", ""):
        return None, "vlm disabled (backend=off)"
    order = list(cfg.get("auto_order") or ["llamacpp_server", "transformers"]) if name == "auto" else [name]
    reasons = []
    for cand in order:
        try:
            be = create_backend(cand, cfg)
            ok, why = be.available()
        except Exception as e:  # noqa: BLE001
            ok, why, be = False, f"{type(e).__name__}: {e}", None
        if ok and be is not None:
            return be, ""
        reasons.append(f"{cand}: {why}")
    return None, "; ".join(reasons)


__all__ = ["BACKENDS", "BackendUnavailable", "GenResult", "VLMBackend", "create_backend", "resolve_backend"]
