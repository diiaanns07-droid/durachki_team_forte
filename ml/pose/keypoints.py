"""COCO-17: индексы, массивы и геометрия позы (numpy, изотропные координаты)."""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16
NUM_KP = 17
HEAD = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)
WRISTS = (L_WRIST, R_WRIST)
SHOULDERS = (L_SHOULDER, R_SHOULDER)
HIPS = (L_HIP, R_HIP)

COCO17_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def kp_array(keypoints: Sequence[Sequence[float]]) -> np.ndarray:
    """list[(x, y, score)] → float32 (17, 3). Недостающие/битые точки (не 3 числа, NaN/inf) — score 0 (unknown),
    score обрезается в 0..1. Чужая схема (2 числа на точку, 33 точки) не превращается в «уверенные» точки."""
    arr = np.zeros((NUM_KP, 3), dtype=np.float32)
    if keypoints is None:
        return arr
    try:
        items = list(keypoints)[:NUM_KP]
    except TypeError:
        return arr
    for k, kp in enumerate(items):
        try:
            x, y, sc = (float(v) for v in kp)
        except (TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y) and np.isfinite(sc):
            arr[k] = (x, y, min(max(sc, 0.0), 1.0))
    return arr


def to_iso(kps: np.ndarray, aspect_wh: float) -> np.ndarray:
    """Нормированные (x, y) → изотропные (x * W/H, y): расстояния в долях высоты кадра."""
    out = kps.copy()
    out[..., 0] = out[..., 0] * aspect_wh
    return out


def bbox_to_iso(bbox: Sequence[float], aspect_wh: float) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return np.array([x1 * aspect_wh, y1, x2 * aspect_wh, y2], dtype=np.float32)


def point_box_dist(p: np.ndarray, box: np.ndarray) -> float:
    """Расстояние от точки до прямоугольника (0 — внутри)."""
    dx = max(box[0] - p[0], 0.0, p[0] - box[2])
    dy = max(box[1] - p[1], 0.0, p[1] - box[3])
    return float(np.hypot(dx, dy))


def point_in_polygon(p: Sequence[float], poly: Sequence[Sequence[float]]) -> bool:
    """Ray casting; poly — список (x, y)."""
    x, y = float(p[0]), float(p[1])
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def point_polygon_dist(p: Sequence[float], poly: Sequence[Sequence[float]]) -> float:
    """Евклидово расстояние до полигона (0 — внутри). Координаты в тех же единицах, что и p."""
    if point_in_polygon(p, poly):
        return 0.0
    px, py = float(p[0]), float(p[1])
    best = float("inf")
    n = len(poly)
    for i in range(n):
        ax, ay = float(poly[i][0]), float(poly[i][1])
        bx, by = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        abx, aby = bx - ax, by - ay
        denom = abx * abx + aby * aby
        u = 0.0 if denom <= 0 else max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / denom))
        cx, cy = ax + u * abx, ay + u * aby
        best = min(best, float(np.hypot(px - cx, py - cy)))
    return best


def polygon_iso(poly: Sequence[Sequence[float]], aspect_wh: float) -> list[tuple[float, float]]:
    return [(float(x) * aspect_wh, float(y)) for x, y in poly]


def polygon_centroid(poly: Sequence[Sequence[float]]) -> np.ndarray:
    pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    return pts.mean(axis=0) if len(pts) else np.zeros(2)


def bbox_polygon_overlap(bbox: Sequence[float], poly: Sequence[Sequence[float]], grid: int = 5) -> float:
    """Доля сетки точек bbox (grid×grid), попадающих в полигон — дешёвая оценка пересечения."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if x2 <= x1 or y2 <= y1 or len(poly) < 3:
        return 0.0
    xs = np.linspace(x1, x2, grid)
    ys = np.linspace(y1, y2, grid)
    hits = sum(point_in_polygon((x, y), poly) for x in xs for y in ys)
    return hits / float(grid * grid)


class PersonGeom:
    """Геометрия одного человека в одном кадре (изотропные координаты, масштаб s)."""

    __slots__ = ("p", "s", "vis", "min_score")

    def __init__(self, kps_iso: np.ndarray, scale: float, min_score: float) -> None:
        self.p = kps_iso
        self.s = max(float(scale), 1e-6)
        self.min_score = min_score
        self.vis = kps_iso[:, 2] >= min_score

    def ok(self, idx: int) -> bool:
        return bool(self.vis[idx])

    def xy(self, idx: int) -> np.ndarray:
        return self.p[idx, :2]

    def score(self, idx: int) -> float:
        return float(self.p[idx, 2])

    def mean_of(self, idxs: Sequence[int]) -> Optional[np.ndarray]:
        pts = [self.p[i, :2] for i in idxs if self.vis[i]]
        if not pts:
            return None
        return np.mean(pts, axis=0)

    def head(self) -> Optional[np.ndarray]:
        return self.mean_of(HEAD)

    def mouth(self) -> Optional[np.ndarray]:
        """Рот ≈ нос (с видом сверху точнее не получить); иначе центр головы."""
        if self.vis[NOSE]:
            return self.p[NOSE, :2]
        return self.head()

    def ears(self) -> list[np.ndarray]:
        pts = [self.p[i, :2] for i in (L_EAR, R_EAR) if self.vis[i]]
        if not pts:
            h = self.head()
            if h is not None:
                pts = [h]
        return pts

    def shoulder_mid(self) -> Optional[np.ndarray]:
        if self.vis[L_SHOULDER] and self.vis[R_SHOULDER]:
            return (self.p[L_SHOULDER, :2] + self.p[R_SHOULDER, :2]) / 2.0
        return self.mean_of(SHOULDERS) if any(self.vis[list(SHOULDERS)]) else None

    def hip_mid(self) -> Optional[np.ndarray]:
        return self.mean_of(HIPS)

    def body_center(self) -> Optional[np.ndarray]:
        c = self.shoulder_mid()
        if c is None:
            c = self.head()
        if c is None:
            c = self.mean_of(range(NUM_KP))
        return c

    def wrists(self) -> list[tuple[int, np.ndarray]]:
        return [(i, self.p[i, :2]) for i in WRISTS if self.vis[i]]

    def dist(self, a: np.ndarray, b: np.ndarray) -> float:
        """Расстояние в единицах масштаба тела."""
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b))) / self.s

    def facing(self, nose_weight: float, min_nose_offset: float) -> tuple[Optional[np.ndarray], float]:
        """Направление «груди» на плоскости изображения (единичный вектор, надёжность 0..1).

        Основной признак — перпендикуляр к линии плеч, знак — по разметке левое/правое плечо:
        d = (-v_y, v_x), v = L_shoulder - R_shoulder (камера смотрит сверху/сверху-спереди, без зеркала).
        Нос смещён от плеч ещё и из-за высоты головы (наклонная камера), поэтому его вклад nose_weight
        по умолчанию 0; при nose_weight > 0 он подмешивается (для почти вертикальной камеры).
        Плечи «схлопнуты» (человек боком к наклонной камере) → надёжность падает.
        """
        if not (self.vis[L_SHOULDER] and self.vis[R_SHOULDER]):
            return None, 0.0
        v = self.p[L_SHOULDER, :2] - self.p[R_SHOULDER, :2]
        n = float(np.linalg.norm(v))
        if n < 1e-6:
            return None, 0.0
        vec = np.array([-v[1], v[0]], dtype=np.float64) / n
        conf = min(self.score(L_SHOULDER), self.score(R_SHOULDER)) * min(1.0, (n / self.s) / 0.6)
        mid = self.shoulder_mid()
        if nose_weight > 0 and self.vis[NOSE] and mid is not None:
            off = self.p[NOSE, :2] - mid
            on = float(np.linalg.norm(off))
            if on / self.s >= min_nose_offset:
                vec = (1.0 - nose_weight) * vec + nose_weight * off / on
                nv = float(np.linalg.norm(vec))
                if nv < 1e-6:
                    return None, 0.0
                vec = vec / nv
        return vec, float(min(1.0, conf))


def track_scale(kps_iso_seq: Sequence[np.ndarray], bboxes_iso: Sequence[Optional[np.ndarray]],
                cfg: dict[str, Any], min_score: float) -> float:
    """Робастный масштаб тела трека: перцентиль ширины плеч → по ушам → по bbox."""
    pct = float(cfg.get("percentile", 75))
    min_valid = int(cfg.get("min_valid_frames", 3))
    shoulder_w, ear_w, box_s = [], [], []
    for k in kps_iso_seq:
        if k[L_SHOULDER, 2] >= min_score and k[R_SHOULDER, 2] >= min_score:
            shoulder_w.append(float(np.linalg.norm(k[L_SHOULDER, :2] - k[R_SHOULDER, :2])))
        if k[L_EAR, 2] >= min_score and k[R_EAR, 2] >= min_score:
            ear_w.append(float(np.linalg.norm(k[L_EAR, :2] - k[R_EAR, :2])))
    for b in bboxes_iso:
        if b is not None:
            w, h = float(b[2] - b[0]), float(b[3] - b[1])
            if w > 0 and h > 0:
                box_s.append(float(np.sqrt(w * h)))
    box_scale = float(cfg.get("bbox_frac", 0.35)) * float(np.median(box_s)) if box_s else None
    if len(shoulder_w) >= min_valid:
        s = float(np.percentile(shoulder_w, pct))
        if box_scale is None or s >= 0.3 * box_scale:  # защита от вырожденных (профиль) плеч
            return s
    if len(ear_w) >= min_valid:
        return float(cfg.get("head_to_shoulder", 2.6)) * float(np.percentile(ear_w, pct))
    if box_scale is not None:
        return box_scale
    return 0.08
