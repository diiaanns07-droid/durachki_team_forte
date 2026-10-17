"""Контракт v1: canonical из contracts/ (A01), иначе локальная fallback-копия (удалит интегратор)."""
from __future__ import annotations

try:
    from contracts.sentinel_contracts import *  # noqa: F401,F403
except ImportError:  # пока ветка A01 не смерджена
    from ._contract_fallback import *  # noqa: F401,F403
