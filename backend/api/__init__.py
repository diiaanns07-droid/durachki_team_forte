"""Kassa Sentinel HTTP API (A02). Публично: create_app(), app (в backend.api.main)."""
from __future__ import annotations

from typing import Any

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:  # ленивый импорт: `import backend.api` ничего не тянет
    if name == "create_app":
        from .main import create_app
        return create_app
    raise AttributeError(name)
