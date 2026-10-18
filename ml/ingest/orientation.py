"""Ориентация кадра: display matrix / тег rotate -> (поворот по часовой, зеркало).

Логика повторяет autorotate ffmpeg (fftools/ffmpeg_filter.c), чтобы кадры совпадали
с тем, что показывает плеер. Соглашение модуля: ``rotation_cw`` — на сколько градусов
по часовой стрелке повернуть сохранённый кадр для показа (iPhone-портрет: 90),
затем, если ``hflip``, отразить по горизонтали.

Связь с источниками:
  * тег ``rotate`` (старые mov/mp4): rotation_cw = rotate;
  * ffprobe/PyAV ``rotation`` (av_display_rotation_get, против часовой): rotation_cw = -rotation;
  * OpenCV ``CAP_PROP_ORIENTATION_META``: уже по часовой.
"""
from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass
from typing import Any, Optional, Sequence


def normalize_rotation(deg: float) -> int:
    """Округлить угол до ближайших 90° и привести к {0, 90, 180, 270}."""
    if deg is None or (isinstance(deg, float) and math.isnan(deg)):
        return 0
    return int(round(float(deg) / 90.0)) % 4 * 90


@dataclass(frozen=True)
class Orientation:
    rotation_cw: int = 0  # 0 | 90 | 180 | 270
    hflip: bool = False   # после поворота
    source: str = "none"  # откуда взято: displaymatrix | rotate_tag | ffprobe | opencv | none

    @property
    def is_identity(self) -> bool:
        return self.rotation_cw == 0 and not self.hflip

    @property
    def swaps_axes(self) -> bool:
        return self.rotation_cw in (90, 270)

    def display_size(self, width: int, height: int) -> tuple[int, int]:
        """Размер после поворота для кадра width x height (как хранится в потоке)."""
        return (height, width) if self.swaps_axes else (width, height)

    def apply(self, img: Any) -> Any:
        """Повернуть/отразить BGR-кадр (np.ndarray HxWxC или HxW)."""
        if self.is_identity:
            return img
        try:
            import cv2  # lazy

            codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
            if self.rotation_cw in codes:
                img = cv2.rotate(img, codes[self.rotation_cw])
            if self.hflip:
                img = cv2.flip(img, 1)
            return img
        except ImportError:  # pragma: no cover - cv2 есть в requirements
            import numpy as np

            if self.rotation_cw:
                img = np.rot90(img, k=-(self.rotation_cw // 90))
            if self.hflip:
                img = img[:, ::-1]
            return np.ascontiguousarray(img)

    def to_dict(self) -> dict[str, Any]:
        return {"rotation_cw": self.rotation_cw, "hflip": self.hflip, "source": self.source}


IDENTITY = Orientation()


def display_rotation_ccw(matrix: Sequence[float]) -> float:
    """Аналог av_display_rotation_get: угол (против часовой) из 3x3 display matrix (9 int32)."""
    conv = [float(v) / 65536.0 for v in matrix]
    s0 = math.hypot(conv[0], conv[3])
    s1 = math.hypot(conv[1], conv[4])
    if s0 == 0.0 or s1 == 0.0:
        return float("nan")
    return -math.degrees(math.atan2(conv[1] / s1, conv[0] / s0))


def from_display_matrix(matrix: Sequence[int], source: str = "displaymatrix") -> Orientation:
    """Ориентация по display matrix — как ffmpeg autorotate (включая варианты с зеркалом)."""
    if len(matrix) < 9:
        return IDENTITY
    ccw = display_rotation_ccw(matrix)
    if math.isnan(ccw):
        return IDENTITY
    theta = -round(ccw)
    theta -= 360 * math.floor(theta / 360 + 0.9 / 360)
    m = list(matrix)
    if abs(theta - 90) < 1.0:
        # transpose=clock | cclock_flip (= transpose = rot90cw + hflip)
        return Orientation(90, m[3] > 0, source)
    if abs(theta - 180) < 1.0:
        h, v = m[0] < 0, m[4] < 0
        if h and v:
            return Orientation(180, False, source)
        if h:
            return Orientation(0, True, source)
        if v:
            return Orientation(180, True, source)
        return Orientation(180, False, source)
    if abs(theta - 270) < 1.0:
        # transpose=cclock | clock_flip (= rot270cw + hflip)
        return Orientation(270, m[3] < 0, source)
    if abs(theta) < 1.0:
        return Orientation(180, True, source) if m[4] < 0 else Orientation(0, False, source)
    # произвольный угол (почти не встречается) — округляем до 90°
    return Orientation(normalize_rotation(theta), False, source)


def matrix_from_bytes(raw: bytes) -> Optional[list[int]]:
    """9 x int32 (little-endian) из side data AV_FRAME_DATA_DISPLAYMATRIX."""
    if raw is None or len(raw) < 36:
        return None
    return list(struct.unpack("<9i", bytes(raw[:36])))


_HEX_ROW = re.compile(r"^\s*[0-9a-fA-F]{8}:\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*$")


def matrix_from_ffprobe(text: str) -> Optional[list[int]]:
    """Распарсить поле ``displaymatrix`` из JSON ffprobe (3 строки по 3 числа)."""
    if not text:
        return None
    values: list[int] = []
    for line in str(text).splitlines():
        match = _HEX_ROW.match(line)
        if match:
            values.extend(int(g) for g in match.groups())
    return values if len(values) == 9 else None


def from_rotation_cw(deg: Optional[float], source: str) -> Orientation:
    rot = normalize_rotation(deg or 0)
    return Orientation(rot, False, source if rot else "none")
