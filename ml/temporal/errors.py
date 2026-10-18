"""Исключения ml.temporal (лёгкий модуль: импорт пакета не тянет тяжёлых зависимостей)."""
from __future__ import annotations


class TemporalUnavailable(RuntimeError):
    """Временная модель не выполнила анализ (нет чекпоинта, несовместимость, ошибка, слишком короткое видео).
    Пустой результат в этом случае НЕ означает «нарушений нет»."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
