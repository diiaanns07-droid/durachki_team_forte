"""Единая точка импорта контракта для ml.fusion.

Пока ветка A01 не смерджена, используем локальную копию ``_contract_fallback.py``.
Интегратор удалит fallback — этот файл продолжит работать без изменений.
"""
from __future__ import annotations

try:
    from contracts.sentinel_contracts import (  # noqa: F401
        LABELS,
        SIGNALS,
        AnalysisResult,
        BBox,
        Event,
        EvidenceFrame,
        RuleRef,
        Violation,
    )
except ImportError:  # пока ветка A01 не смерджена
    from contracts.sentinel_contracts import (  # type: ignore[no-redef]  # noqa: F401
        LABELS,
        SIGNALS,
        AnalysisResult,
        BBox,
        Event,
        EvidenceFrame,
        RuleRef,
        Violation,
    )

__all__ = [
    "LABELS",
    "SIGNALS",
    "AnalysisResult",
    "BBox",
    "Event",
    "EvidenceFrame",
    "RuleRef",
    "Violation",
]
