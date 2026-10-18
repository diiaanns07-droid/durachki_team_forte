"""Индексы наблюдений: кадры (PTS), треки людей, позы, предметы, зоны.

Три состояния на каждом шаге: observed / absent / unknown.
- unknown: кадр не обработан детектором, человека не видно, сустав ниже порога видимости — вывода нет;
- absent: кадр обработан, нужные суставы видны, искомого нет рядом (это не «доказано отсутствие» в
  юридическом смысле, а «модель не увидела при хорошей видимости»);
- observed: есть детекция модели с её score и происхождением.
Потеря детекции человека НЕ означает, что человек ушёл: это unknown, пока он не виден вне зоны.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

HEAD = (0, 1, 2, 3, 4)            # COCO-17: нос, глаза, уши
WRISTS = {"left_wrist": 9, "right_wrist": 10}


def get(o: Any, k: str, default: Any = None) -> Any:
    if isinstance(o, Mapping):
        return o.get(k, default)
    return getattr(o, k, default)


def r4(x: float) -> float:
    return round(float(x), 4)


def box_list(b: Any) -> list[float]:
    return [round(float(v), 5) for v in b]


@dataclass
class Frames:
    times: np.ndarray            # PTS сэмплированных кадров, с
    processed: np.ndarray        # детектор отработал на кадре (есть хотя бы одна детекция любого класса)
    width: float                 # px после поворота — для изотропных расстояний
    height: float
    source: str = "clip"

    @property
    def dt(self) -> float:
        return float(np.median(np.diff(self.times))) if len(self.times) > 1 else 0.2


def frames_from(clip: Any, dets: Sequence[Any]) -> Frames:
    det_t = np.unique(np.round(np.asarray([float(get(d, "t")) for d in dets], float), 4)) if dets else np.zeros(0)
    times = list(get(clip, "times", None) or []) if clip is not None else []
    src = "clip"
    if not times:
        times, src = list(det_t), "detections"
    tt = np.round(np.asarray(times, float), 4)
    processed = np.isin(tt, det_t) if len(det_t) else np.zeros(len(tt), bool)
    w = float(get(clip, "width", 0) or 0) if clip is not None else 0.0
    h = float(get(clip, "height", 0) or 0) if clip is not None else 0.0
    return Frames(tt, processed, w or 1.0, h or 1.0, src)


class _TimeIndex:
    """Ближайшее наблюдение по времени в пределах допуска (без интерполяции)."""

    def __init__(self, items: Iterable[tuple[float, Any]]):
        rows = sorted(items, key=lambda x: x[0])
        self.t = [float(t) for t, _ in rows]
        self.v = [v for _, v in rows]

    def at(self, t: float, tol: float) -> Optional[Any]:
        if not self.t:
            return None
        i = bisect.bisect_left(self.t, t)
        best, bd = None, tol + 1e-9
        for j in (i - 1, i):
            if 0 <= j < len(self.t) and abs(self.t[j] - t) <= bd:
                best, bd = j, abs(self.t[j] - t)
        return None if best is None else self.v[best]

    @property
    def span(self) -> tuple[float, float]:
        return (self.t[0], self.t[-1]) if self.t else (0.0, -1.0)


@dataclass
class PersonTrack:
    tid: int
    role: str
    role_conf: float
    boxes: _TimeIndex

    def box_at(self, t: float, tol: float) -> Optional[np.ndarray]:
        return self.boxes.at(t, tol)

    def spans(self, t: float) -> bool:
        a, b = self.boxes.span
        return a - 1e-6 <= t <= b + 1e-6


def person_tracks(tracks: Sequence[Any]) -> dict[int, PersonTrack]:
    out: dict[int, PersonTrack] = {}
    for tr in tracks or []:
        if str(get(tr, "label", "")) != "person":
            continue
        tid = int(get(tr, "track_id"))
        rows = [(float(t), np.asarray(b, np.float32)) for t, b in (get(tr, "boxes", []) or [])]
        out[tid] = PersonTrack(tid, str(get(tr, "role", "unknown") or "unknown"), float(get(tr, "role_conf", 0.0) or 0.0),
                               _TimeIndex(rows))
    return out


class PoseIndex:
    def __init__(self, poses: Sequence[Any]):
        by: dict[int, list[tuple[float, np.ndarray]]] = {}
        self.n = 0
        for p in poses or []:
            tid = get(p, "track_id")
            if tid is None:
                continue
            kp = np.asarray(get(p, "keypoints"), np.float32).reshape(-1, 3)
            if kp.shape[0] < 17:
                continue
            by.setdefault(int(tid), []).append((float(get(p, "t")), kp))
            self.n += 1
        self.idx = {tid: _TimeIndex(rows) for tid, rows in by.items()}

    def at(self, tid: int, t: float, tol: float) -> Optional[np.ndarray]:
        ix = self.idx.get(int(tid))
        return None if ix is None else ix.at(t, tol)

    @property
    def tracks(self) -> list[int]:
        return sorted(self.idx)


@dataclass
class ObjDet:
    t: float
    label: str
    box: np.ndarray
    score: float
    source: str


class ObjIndex:
    def __init__(self, dets: Sequence[Any], labels: Sequence[str], min_score: float):
        labels = set(labels)
        by: dict[float, list[ObjDet]] = {}
        self.n = 0
        self.sources: dict[str, int] = {}
        for d in dets or []:
            lab = str(get(d, "label", ""))
            sc = float(get(d, "score", 0.0) or 0.0)
            if lab not in labels or sc < min_score:
                continue
            t = r4(get(d, "t"))
            src = str(get(d, "source", "") or "")
            by.setdefault(t, []).append(ObjDet(t, lab, np.asarray(get(d, "bbox"), np.float32), sc, src))
            self.n += 1
            key = src.split(":")[0] or "unknown"
            self.sources[key] = self.sources.get(key, 0) + 1
        self.ix = _TimeIndex(by.items())

    def at(self, t: float, tol: float) -> list[ObjDet]:
        return self.ix.at(t, tol) or []


# ---------------------------------------------------------------- геометрия (нормированные координаты кадра)

def zone_polygon(zones: Sequence[Any], name: str) -> Optional[np.ndarray]:
    for z in zones or []:
        if str(get(z, "name", "")) == name:
            poly = np.asarray(get(z, "polygon"), np.float32).reshape(-1, 2)
            return poly if len(poly) >= 3 else None
    return None


def points_in_polygon(px: np.ndarray, py: np.ndarray, poly: np.ndarray) -> np.ndarray:
    x, y = poly[:, 0], poly[:, 1]
    xj, yj = np.roll(x, 1), np.roll(y, 1)
    inside = np.zeros(np.shape(px), bool)
    for k in range(len(poly)):
        cond = (y[k] > py) != (yj[k] > py)
        denom = yj[k] - y[k] if abs(yj[k] - y[k]) > 1e-12 else 1e-12
        xin = (xj[k] - x[k]) * (py - y[k]) / denom + x[k]
        inside ^= cond & (px < xin)
    return inside


def lower_half_in_zone(box: np.ndarray, poly: Optional[np.ndarray], k: int = 6) -> float:
    """Доля сетки точек нижней половины bbox внутри зоны (ноги/стул — где человек находится)."""
    if poly is None:
        return 0.0
    x1, y1, x2, y2 = (float(v) for v in box)
    ym = (y1 + y2) / 2.0
    g = (np.arange(k, dtype=np.float32) + 0.5) / k
    px, py = np.meshgrid(x1 + (x2 - x1) * g, ym + (y2 - ym) * g)
    return float(points_in_polygon(px, py, poly).mean())


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def center(b: np.ndarray) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0], np.float32)


def iso_dist(p: np.ndarray, q: np.ndarray, fr: Frames) -> float:
    """Расстояние в пикселях (нормированные x и y имеют разный масштаб у портретного кадра)."""
    return float(np.hypot((p[0] - q[0]) * fr.width, (p[1] - q[1]) * fr.height))


def box_height_px(b: np.ndarray, fr: Frames) -> float:
    return max(1e-6, float(b[3] - b[1]) * fr.height)


@dataclass
class Joints:
    wrists: list[tuple[str, np.ndarray, float]] = field(default_factory=list)
    head: Optional[tuple[np.ndarray, float]] = None

    @property
    def any(self) -> bool:
        return bool(self.wrists) or self.head is not None


def joints(kp: Optional[np.ndarray], wrist_min: float, head_min: float) -> Joints:
    if kp is None:
        return Joints()
    out = Joints()
    for name, i in WRISTS.items():
        if float(kp[i, 2]) >= wrist_min:
            out.wrists.append((name, kp[i, :2].copy(), float(kp[i, 2])))
    hv = [i for i in HEAD if float(kp[i, 2]) >= head_min]
    if hv:
        out.head = (kp[hv, :2].mean(axis=0), float(np.mean(kp[hv, 2])))
    return out
