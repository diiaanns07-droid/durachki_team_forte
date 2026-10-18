"""Единая точка импорта контракта для ml.scene (fallback — пока ветка A01 не смерджена)."""
from __future__ import annotations

try:
    from contracts.sentinel_contracts import *  # noqa: F401,F403
    from contracts.sentinel_contracts import LABELS, SIGNALS, Detection, SceneState, Track, Zone  # noqa: F401
except ImportError:  # пока ветка A01 не смерджена
    from contracts.sentinel_contracts import *  # noqa: F401,F403
    from contracts.sentinel_contracts import LABELS, SIGNALS, Detection, SceneState, Track, Zone  # noqa: F401
