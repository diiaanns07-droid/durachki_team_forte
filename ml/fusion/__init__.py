"""ml.fusion (A12) — слияние событий, вердикт, кадры-доказательства, описания.

Публичная функция (сигнатура заморожена контрактом v1)::

    fuse(clip, events, risk, catalog, job_id) -> AnalysisResult

Дополнительно (для A03/A02/A15): ``apply_completeness(result, stages)`` — технический failed при
упавших обязательных стадиях; ``result_to_jury(result)`` — строгий ответ жюри (через A01, если есть);
``result_to_ui_report(result)`` — полный отчёт для UI (подсказки, оверлеи, статус норм).
"""
from __future__ import annotations

from .completeness import apply_completeness
from .core import FUSION_VERSION, fuse, fuse_scores
from .serialize import result_to_jury, result_to_ui_report

__all__ = ["FUSION_VERSION", "apply_completeness", "fuse", "fuse_scores", "result_to_jury",
           "result_to_ui_report"]
