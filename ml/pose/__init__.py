"""ml.pose — поза COCO-17 и поза-сигналы Kassa Sentinel (агент A07).

Публичные функции (сигнатуры контракта v1):
    estimate(clip, tracks) -> list[PoseFrame]
    pose_signals(poses, tracks, zones) -> list[SceneState]
Дополнительно: warmup() -> bool, status() -> dict (последний запуск estimate: stage_status, причина, coverage),
availability() -> dict (доступен ли реальный бэкенд сейчас, без загрузки моделей),
model_info() -> dict (хук A03: identity/ключ кэша стадии позы — веса, флаги, бэкенд).

Тяжёлые зависимости (numpy/cv2/onnxruntime/mediapipe) импортируются лениво — импорт пакета без побочных
эффектов. Импортируемость пакета НЕ означает, что поза доступна: см. availability().
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from ._contract import PoseFrame, PoseFrameX, PoseSceneState, SceneState, Track, Zone
from .status import status

__version__ = "2.1.0"
__all__ = ["estimate", "pose_signals", "warmup", "status", "availability", "model_info", "PoseFrameX",
           "PoseSceneState", "__version__"]


def estimate(clip: Any, tracks: Sequence[Track]) -> list[PoseFrame]:
    """Top-down поза по bbox-ам person-треков. Нет бэкенда/весов — [] и причина в status() (без исключений)."""
    from .estimator import estimate_impl

    return list(estimate_impl(clip, tracks))


def pose_signals(poses: Sequence[PoseFrame], tracks: Sequence[Track], zones: Sequence[Zone]) -> list[SceneState]:
    """Поза → SceneState (signals только из SIGNALS; нет ключа = unknown; ≥ 0.5 = активен)."""
    from .signals import compute_pose_signals

    return list(compute_pose_signals(poses, tracks, zones))


def warmup() -> bool:
    """Загрузить и прогреть бэкенд позы. False — выключен/недоступен (пайплайн продолжает без позы)."""
    from .estimator import warmup_impl

    return warmup_impl()


def availability(cfg: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Доступен ли реальный бэкенд прямо сейчас (веса на диске + пакет), без загрузки моделей."""
    from ._availability import availability as _availability

    return _availability(cfg)


def model_info() -> dict[str, Any]:
    """Хук A03 (module_identity): checkpoint, available, stage_status, флаги — в identity и ключ кэша."""
    from ._availability import model_info as _model_info

    return _model_info()
