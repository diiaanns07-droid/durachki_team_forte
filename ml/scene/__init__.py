"""ml.scene — состояния сцены (A08): сейф, дверь, наличные, присутствие людей.

Публичная функция (сигнатура заморожена контрактом v1):
    scene_states(clip, dets, tracks, zones, pose_states) -> list[SceneState]
Импорт пакета лёгкий: numpy/cv2 подгружаются внутри функций.
"""
from __future__ import annotations

from typing import Any, Sequence


def scene_states(clip: Any, dets: Sequence[Any], tracks: Sequence[Any], zones: Sequence[Any], pose_states: Sequence[Any]) -> list[Any]:
    from .core import scene_states as _impl

    return _impl(clip, dets, tracks, zones, pose_states)


__all__ = ["scene_states"]
