"""Импорт замороженного контракта + расширения модуля ml.pose.

Контракт не меняется: расширенные классы — наследники контрактных dataclass-ов,
т.е. ``isinstance(x, PoseFrame)`` / ``isinstance(x, SceneState)`` остаются истинными.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from contracts.sentinel_contracts import *  # noqa: F401,F403
    from contracts.sentinel_contracts import PoseFrame, SceneState, SIGNALS  # noqa: F401
except ImportError:  # пока ветка A01 не смерджена
    from contracts.sentinel_contracts import *  # noqa: F401,F403
    from contracts.sentinel_contracts import PoseFrame, SceneState, SIGNALS  # noqa: F401


@dataclass
class PoseFrameX(PoseFrame):
    """PoseFrame + служебные поля.

    extra (всё опционально):
      bbox        — bbox человека (норм.), по которому считалась поза
      source      — бэкенд: rtmpose-onnx | rtmlib | mediapipe | mock
      frame_wh    — (w, h) кадра в пикселях (для изотропной геометрии)
      role        — роль трека на момент кадра
      smoothed    — True, если к точкам применён One-Euro фильтр
      raw_keypoints — точки до сглаживания (если smoothing.keep_raw)
    """

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PoseSceneState(SceneState):
    """SceneState + extra.

    signals — только имена из SIGNALS (0..1; >= 0.5 ⇔ сигнал активен после гистерезиса).
    extra   — bending_to_safe, sitting, standing, posture_conf, by_track, track_ids, boxes, ...
    """

    extra: dict[str, Any] = field(default_factory=dict)
