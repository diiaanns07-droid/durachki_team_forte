"""Внутреннее представление трека в numpy (для зон/ролей/событий)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .geometry import anchor_point


@dataclass
class TrackView:
    tid: int
    label: str
    times: np.ndarray        # (N,)
    boxes: np.ndarray        # (N,4)
    scores: np.ndarray       # (N,)
    pts: np.ndarray          # (N,2) точки-якоря
    w: np.ndarray            # (N,) вес наблюдения во времени, с
    diag: float              # медианная изотропная диагональ бокса

    @property
    def t_start(self) -> float:
        return float(self.times[0])

    @property
    def t_end(self) -> float:
        return float(self.times[-1])

    @property
    def presence(self) -> float:
        return float(self.w.sum())

    @property
    def median_pt(self) -> tuple[float, float]:
        m = np.median(self.pts, axis=0)
        return (float(m[0]), float(m[1]))

    def overlap_s(self, other: "TrackView") -> float:
        return max(0.0, min(self.t_end, other.t_end) - max(self.t_start, other.t_start))


def make_view(tid: int, label: str, times: Sequence[float], boxes: Sequence[Sequence[float]],
              scores: Sequence[float], anchor: str, aspect: float, frame_dt: float) -> TrackView:
    t = np.asarray(times, float)
    order = np.argsort(t, kind="stable")
    t = t[order]
    b = np.asarray(boxes, float).reshape(-1, 4)[order]
    s = np.asarray(scores, float)[order] if len(scores) else np.ones(len(t))
    pts = np.array([anchor_point(x, anchor) for x in b]).reshape(-1, 2)
    if len(t) > 1:
        w = np.minimum(np.diff(t), 2.0 * frame_dt)
        w = np.concatenate([w, [frame_dt]])
    else:
        w = np.array([frame_dt])
    diag = float(np.median(np.hypot((b[:, 2] - b[:, 0]) * aspect, b[:, 3] - b[:, 1]))) if len(b) else 0.0
    return TrackView(tid=int(tid), label=str(label), times=t, boxes=b, scores=s, pts=pts, w=w, diag=max(diag, 1e-4))


def view_from_track(tr: Any, anchor: str, aspect: float, frame_dt: float) -> TrackView:
    """Из контрактного Track (или dict) — для events/viz."""
    get = (lambda k, d=None: tr.get(k, d)) if isinstance(tr, dict) else (lambda k, d=None: getattr(tr, k, d))
    bx = get("boxes", []) or []
    times = [float(x[0]) for x in bx]
    boxes = [tuple(x[1]) for x in bx]
    return make_view(int(get("track_id", 0)), str(get("label", "")), times, boxes, [1.0] * len(times),
                     anchor, aspect, frame_dt)
