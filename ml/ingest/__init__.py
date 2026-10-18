"""ml.ingest — декодирование видео, сэмплинг, качество кадров, аугментации ракурса (A04).

Публичный API (тяжёлые зависимости импортируются лениво, импорт пакета без побочных эффектов):

    from ml.ingest import load_video
    clip = load_video("X-1.mov")            # sample_fps=5.0 по контракту
    clip.times, clip.frame(i), clip.frame_at(t), clip.quality
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "load_video",
    "Clip",
    "probe",
    "validate",
    "IngestError",
    "VideoNotFoundError",
    "VideoProbeError",
    "NoVideoStreamError",
    "InvalidDurationError",
    "DecodeError",
    "FrameGapError",
    "FrameRef",
]

_LAZY = {
    "load_video": ("video", "load_video"),
    "Clip": ("video", "Clip"),
    "probe": ("probe", "probe"),
    "validate": ("probe", "validate"),
    "IngestError": ("errors", "IngestError"),
    "VideoNotFoundError": ("errors", "VideoNotFoundError"),
    "VideoProbeError": ("errors", "VideoProbeError"),
    "NoVideoStreamError": ("errors", "NoVideoStreamError"),
    "InvalidDurationError": ("errors", "InvalidDurationError"),
    "DecodeError": ("errors", "DecodeError"),
    "FrameGapError": ("errors", "FrameGapError"),
    "FrameRef": ("video", "FrameRef"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        value = getattr(importlib.import_module(f"{__name__}.{module_name}"), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
