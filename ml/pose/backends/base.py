"""Интерфейс бэкенда позы: top-down по bbox-ам, выход строго COCO-17 в пикселях кадра."""
from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np


class PoseBackendError(RuntimeError):
    """Бэкенд не может быть инициализирован (нет весов / библиотеки / провайдера)."""


class WeightsNotFound(PoseBackendError):
    """Веса бэкенда не найдены локально (не сбой: в режиме enabled=auto стадия просто пропускается)."""


class PoseBackend(Protocol):
    name: str
    batch_size: int

    def infer(self, items: Sequence[tuple[np.ndarray, np.ndarray]], meta: Any = None) -> list[tuple[np.ndarray, np.ndarray]]:
        """items: [(frame_bgr uint8 HxWx3, box_xyxy_px float(4,))] →
        [(keypoints_px float32 (17, 2), scores float32 (17,) в 0..1)] — порядок COCO-17."""
        ...
