"""Ошибки модуля ingest.

У каждой ошибки есть машинный ``code`` (для API/UI, A02 может маппить в HTTP 422)
и человекочитаемое русское сообщение. Все наследуются от :class:`IngestError`.
"""
from __future__ import annotations

from typing import Any, Optional


class IngestError(Exception):
    """Базовая ошибка загрузки/проверки видео."""

    code: str = "ingest_error"

    def __init__(
        self,
        message: str,
        *,
        path: Optional[str] = None,
        code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.path = path
        if code:
            self.code = code
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "path": self.path, "details": self.details}


class VideoNotFoundError(IngestError):
    code = "file_not_found"


class VideoProbeError(IngestError):
    """Файл не удаётся прочитать как видео (битый, не видео, неподдерживаемый контейнер)."""

    code = "probe_failed"


class NoVideoStreamError(IngestError):
    code = "no_video_stream"


class InvalidDurationError(IngestError):
    code = "duration_out_of_range"


class DecodeError(IngestError):
    """Контейнер открылся, но кадры не декодируются."""

    code = "decode_failed"


class FrameGapError(IngestError):
    """В запрошенный момент нет кадра: вне видео или внутри пробела покрытия (strict-режим)."""

    code = "frame_gap"
