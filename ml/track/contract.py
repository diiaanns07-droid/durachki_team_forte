"""Единая точка импорта контракта для ml.track.

Пока ветка A01 не смерджена, используется локальная копия `_contract_fallback.py`.
Интегратор может удалить fallback — этот модуль продолжит работать.
"""
from __future__ import annotations

try:
    from contracts.sentinel_contracts import *  # noqa: F401,F403
    from contracts.sentinel_contracts import BBox, Detection, LABELS, SceneState, SIGNALS, Track, Zone  # noqa: F401
except ImportError:  # пока ветка A01 не смерджена
    from contracts.sentinel_contracts import *  # noqa: F401,F403
    from contracts.sentinel_contracts import BBox, Detection, LABELS, SceneState, SIGNALS, Track, Zone  # noqa: F401

ZONE_NAMES = ("cashier_area", "desk", "safe", "door", "client_area", "cash_tray")
