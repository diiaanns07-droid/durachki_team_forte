"""Синтетические последовательности COCO-17 для тестов (без видео и весов)."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ml.pose._contract import PoseFrame, Track, Zone
from ml.pose.backends.mock import BASE_SCORE, TEMPLATE

FPS = 5.0
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10


def times(duration: float, fps: float = FPS) -> list[float]:
    return [round(i / fps, 4) for i in range(int(round(duration * fps)))]


def kps_in_box(box, overrides: Optional[dict[int, tuple[float, float]]] = None,
               scores: Optional[dict[int, float]] = None, uv: Optional[np.ndarray] = None,
               jitter: float = 0.0, rng: Optional[np.random.Generator] = None) -> list[tuple[float, float, float]]:
    """Шаблон «лицом к камере» внутри bbox (норм.); overrides — (u, v) в координатах bbox."""
    base = (TEMPLATE if uv is None else uv).copy()
    sc = BASE_SCORE.copy()
    for k, (u, v) in (overrides or {}).items():
        base[k] = (u, v)
    for k, s in (scores or {}).items():
        sc[k] = s
    x1, y1, x2, y2 = box
    out = []
    for k in range(17):
        x = x1 + base[k, 0] * (x2 - x1)
        y = y1 + base[k, 1] * (y2 - y1)
        if jitter and rng is not None:
            x += float(rng.normal(0, jitter))
            y += float(rng.normal(0, jitter))
        out.append((float(x), float(y), float(sc[k])))
    return out


def person_track(tid: int, role: str, box, ts: list[float]) -> Track:
    return Track(track_id=tid, label="person", role=role, role_conf=0.9, boxes=[(t, tuple(box)) for t in ts])


def object_track(tid: int, label: str, box, ts: list[float]) -> Track:
    return Track(track_id=tid, label=label, boxes=[(t, tuple(box)) for t in ts])


def poses_for(tid: int, box, ts: list[float],
              fn: Callable[[float], tuple[Optional[dict], Optional[dict]]] = lambda t: (None, None),
              uv: Optional[np.ndarray] = None, jitter: float = 0.0, seed: int = 0) -> list[PoseFrame]:
    rng = np.random.default_rng(seed)
    out = []
    for t in ts:
        ov, sc = fn(t)
        out.append(PoseFrame(t=t, track_id=tid, keypoints=kps_in_box(box, ov, sc, uv=uv, jitter=jitter, rng=rng)))
    return out


def desk_zone() -> Zone:
    return Zone(name="desk", polygon=[(0.15, 0.62), (0.85, 0.62), (0.85, 0.80), (0.15, 0.80)])


def values(states, name: str, t0: float, t1: float) -> list[Optional[float]]:
    return [s.signals.get(name) for s in states if t0 - 1e-6 <= s.t <= t1 + 1e-6]


def at(states, t: float):
    return min(states, key=lambda s: abs(s.t - t))
