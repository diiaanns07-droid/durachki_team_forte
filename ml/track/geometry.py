"""Геометрия в нормированных координатах 0..1 (только numpy/math).

Расстояния считаются «изотропно»: x умножается на aspect = width/height,
чтобы 1 единица по x и по y соответствовала одинаковому числу пикселей.
Выпуклые оболочки и масштабирование относительно центроида — аффинно-эквивариантны,
поэтому зоны «едут» вместе с камерой без захардкоженных координат.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np

Point = tuple[float, float]


def box_center(b: Sequence[float]) -> Point:
    return ((float(b[0]) + float(b[2])) * 0.5, (float(b[1]) + float(b[3])) * 0.5)


def anchor_point(b: Sequence[float], anchor: str = "center") -> Point:
    x1, y1, x2, y2 = (float(v) for v in b[:4])
    if anchor == "bottom":
        return ((x1 + x2) * 0.5, y2)
    if anchor == "torso":
        return ((x1 + x2) * 0.5, y1 + 0.6 * (y2 - y1))
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def box_diag(b: Sequence[float], aspect: float = 1.0) -> float:
    return math.hypot((float(b[2]) - float(b[0])) * aspect, float(b[3]) - float(b[1]))


def iso_dist(p: Sequence[float], q: Sequence[float], aspect: float = 1.0) -> float:
    return math.hypot((float(p[0]) - float(q[0])) * aspect, float(p[1]) - float(q[1]))


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def box_gap(a: Sequence[float], b: Sequence[float], aspect: float = 1.0) -> float:
    """Расстояние между краями боксов (0, если пересекаются)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2])) * aspect
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy)


def scale_box(b: Sequence[float], s: float) -> tuple[float, float, float, float]:
    cx, cy = box_center(b)
    hw, hh = (b[2] - b[0]) * 0.5 * s, (b[3] - b[1]) * 0.5 * s
    return (cx - hw, cy - hh, cx + hw, cy + hh)


def box_to_polygon(b: Sequence[float]) -> list[Point]:
    x1, y1, x2, y2 = (float(v) for v in b[:4])
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def convex_hull(points: Iterable[Sequence[float]]) -> list[Point]:
    """Andrew monotone chain, против часовой стрелки (в осях y-вниз — по часовой)."""
    pts = sorted({(round(float(p[0]), 6), round(float(p[1]), 6)) for p in points})
    if len(pts) <= 2:
        return list(pts)

    def cross(o: Point, a: Point, b: Point) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[Point] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def polygon_area(poly: Sequence[Sequence[float]]) -> float:
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def polygon_centroid(poly: Sequence[Sequence[float]]) -> Point:
    n = len(poly)
    if n == 0:
        return (0.5, 0.5)
    a = 0.0
    cx = cy = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        c = x1 * y2 - x2 * y1
        a += c
        cx += (x1 + x2) * c
        cy += (y1 + y2) * c
    if abs(a) < 1e-12:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return (float(sum(xs) / n), float(sum(ys) / n))
    a *= 0.5
    return (float(cx / (6 * a)), float(cy / (6 * a)))


def scale_polygon(poly: Sequence[Sequence[float]], s: float,
                  center: Optional[Sequence[float]] = None) -> list[Point]:
    c = center if center is not None else polygon_centroid(poly)
    return [(c[0] + (p[0] - c[0]) * s, c[1] + (p[1] - c[1]) * s) for p in poly]


def reflect_polygon(poly: Sequence[Sequence[float]], c: Sequence[float]) -> list[Point]:
    """Центральная симметрия относительно точки c (аффинно-эквивариантна)."""
    return [(2 * c[0] - p[0], 2 * c[1] - p[1]) for p in poly]


def polygon_bbox(poly: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def point_in_polygon(pt: Sequence[float], poly: Sequence[Sequence[float]]) -> bool:
    n = len(poly)
    if n < 3:
        return False
    x, y = float(pt[0]), float(pt[1])
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            xin = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < xin:
                inside = not inside
        j = i
    return inside


def _dist_point_segment(p: Point, a: Point, b: Point, aspect: float) -> float:
    px, py = p[0] * aspect, p[1]
    ax, ay = a[0] * aspect, a[1]
    bx, by = b[0] * aspect, b[1]
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = 0.0 if L <= 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def dist_point_polygon(pt: Sequence[float], poly: Sequence[Sequence[float]], aspect: float = 1.0) -> float:
    """0 внутри полигона, иначе изотропное расстояние до ближайшего ребра."""
    if not poly:
        return float("inf")
    p = (float(pt[0]), float(pt[1]))
    if len(poly) >= 3 and point_in_polygon(p, poly):
        return 0.0
    if len(poly) == 1:
        return iso_dist(p, poly[0], aspect)
    n = len(poly)
    return min(_dist_point_segment(p, tuple(poly[i]), tuple(poly[(i + 1) % n]), aspect) for i in range(n))


def clip_polygon_unit(poly: Sequence[Sequence[float]]) -> list[Point]:
    """Sutherland–Hodgman: обрезка полигона квадратом [0,1]^2."""
    out = [(float(p[0]), float(p[1])) for p in poly]
    for axis, bound, keep_ge in ((0, 0.0, True), (0, 1.0, False), (1, 0.0, True), (1, 1.0, False)):
        if not out:
            break
        inp, out = out, []

        def inside(p: Point, axis: int = axis, bound: float = bound, keep_ge: bool = keep_ge) -> bool:
            return p[axis] >= bound if keep_ge else p[axis] <= bound

        def inter(p: Point, q: Point, axis: int = axis, bound: float = bound) -> Point:
            t = (bound - p[axis]) / ((q[axis] - p[axis]) or 1e-12)
            r = (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))
            return (bound, r[1]) if axis == 0 else (r[0], bound)

        for i, cur in enumerate(inp):
            prev = inp[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(inter(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(inter(prev, cur))
    return out


def reduce_vertices(poly: Sequence[Sequence[float]], max_v: int = 12) -> list[Point]:
    """Убираем вершины с минимальным вкладом в площадь, пока их не станет <= max_v."""
    pts = [tuple(map(float, p)) for p in poly]
    while len(pts) > max(3, max_v):
        best_i, best_a = 0, float("inf")
        n = len(pts)
        for i in range(n):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
            area = abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) * 0.5
            if area < best_a:
                best_i, best_a = i, area
        pts.pop(best_i)
    return pts


def ensure_min_size(poly: Sequence[Sequence[float]], min_w: float, min_h: float) -> list[Point]:
    if len(poly) < 3:
        return [tuple(map(float, p)) for p in poly]
    x1, y1, x2, y2 = polygon_bbox(poly)
    w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    s = max(1.0, min_w / w, min_h / h)
    return scale_polygon(poly, s) if s > 1.0 else [tuple(map(float, p)) for p in poly]


def round_polygon(poly: Sequence[Sequence[float]], nd: int = 4) -> list[tuple[float, float]]:
    return [(round(float(p[0]), nd), round(float(p[1]), nd)) for p in poly]


def median_box(boxes: np.ndarray) -> tuple[float, float, float, float]:
    m = np.median(np.asarray(boxes, float).reshape(-1, 4), axis=0)
    return (float(m[0]), float(m[1]), float(m[2]), float(m[3]))
