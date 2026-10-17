"""Kassa Sentinel — общий контракт между модулями (владелец A01).

Реэкспорт замороженного контракта v1 из ``contracts.sentinel_contracts``.
Сериализация в API — ``contracts.serialize``, JSON Schema ответа — ``contracts/api_schema.json``,
валидатор — ``contracts.validate``. Импорт пакета не тянет тяжёлых зависимостей.
"""
from __future__ import annotations

from .sentinel_contracts import (  # noqa: F401
    LABELS,
    SIGNALS,
    AnalysisResult,
    BBox,
    Detection,
    Event,
    EvidenceFrame,
    PoseFrame,
    Role,
    RuleRef,
    SceneState,
    Track,
    VideoClip,
    Violation,
    Zone,
)

CONTRACT_VERSION = "1"

__all__ = [
    "AnalysisResult",
    "BBox",
    "CONTRACT_VERSION",
    "Detection",
    "Event",
    "EvidenceFrame",
    "LABELS",
    "PoseFrame",
    "Role",
    "RuleRef",
    "SIGNALS",
    "SceneState",
    "Track",
    "VideoClip",
    "Violation",
    "Zone",
]
