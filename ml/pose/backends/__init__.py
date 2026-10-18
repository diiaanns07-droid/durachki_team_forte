"""Бэкенды позы. По умолчанию один реальный — RTMPose ONNX; mock — только явно (SENTINEL_MOCK=1)."""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from .base import PoseBackend, PoseBackendError, WeightsNotFound

log = logging.getLogger("ml.pose")
_CACHE: dict[str, PoseBackend] = {}
_LOCK = threading.Lock()


def _make(name: str, cfg: dict[str, Any]) -> PoseBackend:
    if name == "rtmpose_onnx":
        from .rtmpose_onnx import RTMPoseOnnx
        return RTMPoseOnnx(cfg)
    if name == "mediapipe":
        from .mediapipe_backend import MediaPipePose
        return MediaPipePose(cfg)
    if name == "mock":
        from .mock import MockPose
        return MockPose(cfg)
    raise PoseBackendError(f"неизвестный бэкенд позы: {name}")


def backend_order(cfg: dict[str, Any], mock: bool = False) -> list[str]:
    from ..config import backend_override

    if mock:
        return ["mock"]
    override = backend_override()
    if override:
        return [override]
    return list(cfg.get("backend", {}).get("order", ["rtmpose_onnx"]))


def get_backend(cfg: dict[str, Any], mock: bool = False, errors_out: Optional[list[BaseException]] = None
                ) -> tuple[Optional[PoseBackend], list[str]]:
    """Тёплый (закэшированный) бэкенд + список ошибок попыток. None — ни один не поднялся.

    errors_out (если передан) получает сами исключения — чтобы отличить «нет весов» от сбоя.
    """
    order = backend_order(cfg, mock)
    errors: list[str] = []
    with _LOCK:
        for name in order:
            if name in _CACHE:
                return _CACHE[name], errors
            try:
                be = _make(name, cfg)
            except Exception as exc:  # noqa: BLE001 — сбой бэкенда не должен ронять пайплайн
                msg = f"{name}: {type(exc).__name__}: {exc}"
                log.warning("pose backend unavailable — %s", msg)
                errors.append(msg)
                if errors_out is not None:
                    errors_out.append(exc)
                continue
            _CACHE[name] = be
            return be, errors
    return None, errors


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()
