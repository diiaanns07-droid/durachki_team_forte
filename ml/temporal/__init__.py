"""Kassa Sentinel — ml/temporal (A10): обучаемая временная модель на замороженных признаках SigLIP2.

Публично (контракт): ``score(clip) -> (risk_curve, events)``; без обученного чекпоинта — TemporalUnavailable
(не «пустой = чисто»). ``propose(clip) -> list[Event]`` — 3–8 кандидатов для VLM/правил. Доп.: ``warmup()``,
``status()``, ``model_info()`` (идентичность для кэша пайплайна). Temporal — генератор риска/кандидатов,
не самостоятельное доказательство нарушения.
Импорт пакета лёгкий: torch/transformers/cv2 подгружаются лениво внутри функций.
"""
from __future__ import annotations

from typing import Any, Optional

from .errors import TemporalUnavailable  # noqa: F401  (лёгкий модуль)


def score(clip: Any, zones: Any = None, budget_s: Optional[float] = None
          ) -> tuple[list[tuple[float, float]], list[Any]]:
    from .infer import score as _score

    return _score(clip, zones=zones, budget_s=budget_s)


def propose(clip: Any, zones: Any = None, budget_s: Optional[float] = None, n_max: Optional[int] = None,
            mode: Optional[str] = None) -> list[Any]:
    """3–8 моментов-кандидатов (Event, extra.role="candidate") для VLM и правил."""
    from .infer import propose as _propose

    return _propose(clip, zones=zones, budget_s=budget_s, n_max=n_max, mode=mode)


def warmup() -> None:
    from .infer import warmup as _warmup

    _warmup()


def status() -> dict:
    from .infer import status as _status

    return _status()


def model_info() -> dict:
    """Хук для ключа кэша пайплайна (backend.pipeline.stages.module_identity): подписи файлов чекпоинта,
    конфиг, версия признаков, статус trained/no_checkpoint/oof."""
    from .infer import model_info as _model_info

    return _model_info()


__all__ = ["score", "propose", "warmup", "status", "model_info", "TemporalUnavailable"]
