"""A09 — движок правил Kassa Sentinel: ``SceneState`` → ``Event`` (source="rules").

Публичная функция контракта v1::

    from ml.rules import run_rules
    events = run_rules(states, tracks, zones, catalog)

Каждое событие несёт ``extra["assessment"]``: ``candidate_violation`` (к вердикту допускаются
только они), ``review_observation`` (наблюдение для проверяющего, без юридического
утверждения) или ``insufficient_context`` (этап не наблюдался полностью).
Импорт пакета лёгкий: numpy + stdlib; YAML читается лениво при первом вызове.
"""
from __future__ import annotations

from .catalog import FALLBACK_CATALOG, norm_status, resolve_catalog
from .config import ConfigError, RulesConfig, load_rules_config
from .engine import (
    ASSESSMENTS,
    CANDIDATE,
    INSUFFICIENT,
    REVIEW,
    SIGNAL_RU,
    RulesReport,
    assessment_of,
    evaluate,
    run_rules,
    split_by_assessment,
)
from .expr import ExprError, compile_expr

__all__ = [
    "ASSESSMENTS",
    "CANDIDATE",
    "ConfigError",
    "ExprError",
    "FALLBACK_CATALOG",
    "INSUFFICIENT",
    "REVIEW",
    "RulesConfig",
    "RulesReport",
    "SIGNAL_RU",
    "assessment_of",
    "compile_expr",
    "evaluate",
    "load_rules_config",
    "norm_status",
    "resolve_catalog",
    "run_rules",
    "split_by_assessment",
]
