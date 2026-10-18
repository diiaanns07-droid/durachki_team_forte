"""Геометрия в нормированных координатах (0..1): полигоны зон, боксы, членство в зоне.

Всё на numpy, без shapely. Координаты никогда не захардкожены — только из Zone/Detection.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

BBox = tuple[float, float, float, float]
Polygon = Sequence[tuple[float, float]]


def clip01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


def points_in_polygon(xs: np.ndarray, ys: np.ndarray, poly: Polygon) -> np.ndarray:
    """Ray casting, векторизовано по точкам. Возвращает bool-маску той же формы, что xs."""
    pts = np.asarray(poly, dtype=np.float64)
    inside = np.zeros(np.shape(xs), dtype=bool)
    if pts.ndim != 2 or len(pts) < 3:
        return inside
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        cond = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        inside ^= cond & (x < x_cross)
        j = i
    return inside


def point_in_polygon(x: float, y: float, poly: Polygon) -> bool:
    return bool(points_in_polygon(np.array([x]), np.array([y]), poly)[0])


def polygon_bbox(poly: Polygon) -> Optional[BBox]:
    pts = np.asarray(poly, dtype=np.float64)
    if pts.ndim != 2 or len(pts) < 3:
        return None
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    if x2 - x1 <= 1e-6 or y2 - y1 <= 1e-6:
        return None
    return (clip01(x1), clip01(y1), clip01(x2), clip01(y2))


def bbox_to_polygon(b: BBox) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = b
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def expand_bbox(b: BBox, frac: float = 0.0, pad: float = 0.0) -> BBox:
    """Расширить бокс на frac своей стороны + абсолютный pad (норм.), обрезать к кадру."""
    x1, y1, x2, y2 = b
    w, h = x2 - x1, y2 - y1
    dx, dy = w * frac + pad, h * frac + pad
    return (clip01(x1 - dx), clip01(y1 - dy), clip01(x2 + dx), clip01(y2 + dy))


def bbox_area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def bbox_intersection(a: BBox, b: BBox) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(a: BBox, b: BBox) -> float:
    inter = bbox_intersection(a, b)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def bbox_gap(a: BBox, b: BBox) -> float:
    """Евклидов зазор между боксами (0 — пересекаются/касаются)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return float(np.hypot(dx, dy))


def bbox_center(b: BBox) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def anchor_point(b: BBox, anchor_y: float = 0.6) -> tuple[float, float]:
    """Опорная точка человека: по x — центр, по y — доля высоты от верха бокса."""
    return ((b[0] + b[2]) / 2.0, b[1] + anchor_y * (b[3] - b[1]))


_GRID = np.linspace(0.07, 0.93, 7)


def bbox_fraction_in_polygon(b: BBox, poly: Polygon) -> float:
    """Доля площади бокса внутри полигона (оценка по сетке 7×7)."""
    x1, y1, x2, y2 = b
    if x2 <= x1 or y2 <= y1:
        return 0.0
    gx, gy = np.meshgrid(x1 + _GRID * (x2 - x1), y1 + _GRID * (y2 - y1))
    return float(points_in_polygon(gx, gy, poly).mean())


def zone_membership(b: BBox, poly: Polygon, anchor_y: float = 0.6, overlap_full: float = 0.5) -> float:
    """Мягкое членство человека в зоне 0..1: опорная точка внутри → 1, иначе доля бокса / overlap_full."""
    if poly is None or len(poly) < 3:
        return 0.0
    ax, ay = anchor_point(b, anchor_y)
    if point_in_polygon(ax, ay, poly):
        return 1.0
    frac = bbox_fraction_in_polygon(b, poly)
    return clip01(frac / max(overlap_full, 1e-6))


def dilate_polygon(poly: Polygon, pad: float) -> list[tuple[float, float]]:
    """Грубая дилатация: раздуваем от центроида на pad (норм.) и обрезаем к кадру."""
    pts = np.asarray(poly, dtype=np.float64)
    if len(pts) < 3:
        return [tuple(map(float, p)) for p in pts]
    c = pts.mean(axis=0)
    vec = pts - c
    norm = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9
    out = np.clip(pts + vec / norm * pad, 0.0, 1.0)
    return [(float(x), float(y)) for x, y in out]


def union_bbox(boxes: Iterable[BBox]) -> Optional[BBox]:
    arr = np.asarray(list(boxes), dtype=np.float64)
    if arr.size == 0:
        return None
    return (float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 2].max()), float(arr[:, 3].max()))


def median_bbox(boxes: Iterable[BBox]) -> Optional[BBox]:
    arr = np.asarray(list(boxes), dtype=np.float64)
    if arr.size == 0:
        return None
    m = np.median(arr, axis=0)
    return (float(m[0]), float(m[1]), float(m[2]), float(m[3]))


def bbox_to_px(b: BBox, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = int(np.floor(clip01(b[0]) * w))
    y1 = int(np.floor(clip01(b[1]) * h))
    x2 = int(np.ceil(clip01(b[2]) * w))
    y2 = int(np.ceil(clip01(b[3]) * h))
    return x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)
