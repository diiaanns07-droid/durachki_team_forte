"""Sentinel Copilot (A14) — опциональный локальный ассистент проверяющего: RAG по нормам + локальная LLM.

Импорт пакета лёгкий: модели, корпус, torch/llama.cpp/fastapi не загружаются (всё лениво).

    from copilot import answer, report, status
    answer(job_result, "Почему это нарушение?", history=[])  # -> {"answer", "citations", "grounded", ...}
    report(job_result)                                        # -> markdown «заключение для проверяющего»
"""
from __future__ import annotations

from typing import Any

__all__ = ["answer", "report", "suggestions", "status", "warmup", "release", "make_router"]


def answer(job_result: Any, question: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    from .service import answer as _answer

    return _answer(job_result, question, history or [])


def report(job_result: Any, fmt: str = "md") -> str:
    from .report import report as _report

    return _report(job_result, fmt=fmt)


def suggestions(job_result: Any, limit: int = 4) -> dict[str, Any]:
    from .suggestions import suggestions as _suggestions

    return _suggestions(job_result, limit)


def status() -> dict[str, Any]:
    from .service import status as _status

    return _status()


def warmup() -> bool:
    from .service import warmup as _warmup

    return _warmup()


def release() -> None:
    from .service import release as _release

    _release()


def make_router(*args: Any, **kwargs: Any) -> Any:
    from .router import make_router as _make_router

    return _make_router(*args, **kwargs)
