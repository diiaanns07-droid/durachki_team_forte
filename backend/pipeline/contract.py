"""Единая точка импорта замороженного контракта для оркестратора (канон — contracts/, владелец A01)."""
from __future__ import annotations

from contracts.sentinel_contracts import *  # noqa: F401,F403
from contracts.sentinel_contracts import (  # noqa: F401  (явно, для линтеров/IDE)
    AnalysisResult, BBox, Detection, Event, EvidenceFrame, LABELS, PoseFrame, RuleRef,
    SceneState, SIGNALS, Track, Violation, Zone,
)

CONTRACT_SOURCE = "contracts.sentinel_contracts"
