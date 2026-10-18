"""ByteTrack (Zhang et al., 2022, идея — MIT) — своя компактная реализация на numpy.

Отличия от оригинала под нашу задачу:
* время — в секундах (кадры сэмплируются 5 fps, детекция может идти через кадр);
* стоимость 1-го этапа = 1 - [(1-w)·IoU + w·близость центров] — при 5 fps IoU
  у идущего человека между кадрами бывает 0, близость центров спасает;
* фильтр Калмана с постоянной скоростью и её затуханием (tau), dt-зависимый.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence

import numpy as np


def xyxy_to_cxcywh(b: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in b[:4])
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5, max(x2 - x1, 1e-4), max(y2 - y1, 1e-4)])


def cxcywh_to_xyxy(s: Sequence[float]) -> np.ndarray:
    cx, cy, w, h = (float(v) for v in s[:4])
    return np.array([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5])


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-12)


def fused_cost(tb: np.ndarray, db: np.ndarray, aspect: float, dist_weight: float,
               sigma: float, gate: float, max_shift: Optional[np.ndarray] = None) -> np.ndarray:
    """1 - [(1-w)·IoU + w·близость центров]; при IoU=0 — запрет дальше gate (или max_shift[i]) диагоналей."""
    tb = np.asarray(tb, float).reshape(-1, 4)
    db = np.asarray(db, float).reshape(-1, 4)
    if len(tb) == 0 or len(db) == 0:
        return np.zeros((len(tb), len(db)))
    ov = iou_matrix(tb, db)
    tc = np.stack([(tb[:, 0] + tb[:, 2]) * 0.5 * aspect, (tb[:, 1] + tb[:, 3]) * 0.5], 1)
    dc = np.stack([(db[:, 0] + db[:, 2]) * 0.5 * aspect, (db[:, 1] + db[:, 3]) * 0.5], 1)
    d = np.linalg.norm(tc[:, None, :] - dc[None, :, :], axis=2)
    td = np.hypot((tb[:, 2] - tb[:, 0]) * aspect, tb[:, 3] - tb[:, 1])
    dd = np.hypot((db[:, 2] - db[:, 0]) * aspect, db[:, 3] - db[:, 1])
    dn = d / (0.5 * (td[:, None] + dd[None, :]) + 1e-9)
    sim = (1.0 - dist_weight) * ov + dist_weight * np.exp(-0.5 * (dn / max(sigma, 1e-6)) ** 2)
    cost = 1.0 - sim
    cost[(ov <= 0) & (dn > gate)] = 1e6
    if max_shift is not None:  # H-SPEED: правдоподобная скорость с момента последнего наблюдения трека
        cost[(ov <= 0) & (dn > np.asarray(max_shift, float)[:, None])] = 1e6
    return cost


def linear_assignment(cost: np.ndarray, thresh: float,
                      accept: Optional[np.ndarray] = None) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Венгерский алгоритм по `cost`; пара допустима, если accept[i, j] <= thresh (по умолчанию accept = cost)."""
    n, m = cost.shape if cost.ndim == 2 else (0, 0)
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    acc = cost if accept is None else accept
    matches: list[tuple[int, int]] = []
    try:
        from scipy.optimize import linear_sum_assignment
        c = np.where(acc > thresh, float(np.max(cost)) + thresh + 1e3, cost)
        r, k = linear_sum_assignment(c)
        matches = [(int(i), int(j)) for i, j in zip(r, k) if acc[i, j] <= thresh]
    except ImportError:  # жадный fallback без scipy
        pairs = sorted((float(cost[i, j]), i, j) for i in range(n) for j in range(m) if acc[i, j] <= thresh)
        used_i: set[int] = set()
        used_j: set[int] = set()
        for _, i, j in pairs:
            if i in used_i or j in used_j:
                continue
            used_i.add(i)
            used_j.add(j)
            matches.append((i, j))
    mi = {i for i, _ in matches}
    mj = {j for _, j in matches}
    return matches, [i for i in range(n) if i not in mi], [j for j in range(m) if j not in mj]


class KalmanBox:
    """Состояние [cx, cy, w, h, vx, vy, vw, vh]; шумы масштабируются высотой бокса."""

    def __init__(self, box: Sequence[float], tau: float = 1.0, q_pos: float = 0.5,
                 q_vel: float = 1.0, r: float = 0.05) -> None:
        z = xyxy_to_cxcywh(box)
        self.x = np.concatenate([z, np.zeros(4)])
        s = max(z[3], 1e-3)
        self.P = np.diag([(0.1 * s) ** 2] * 4 + [(0.5 * s) ** 2] * 4)
        self.tau, self.q_pos, self.q_vel, self.r = tau, q_pos, q_vel, r

    def predict(self, dt: float) -> None:
        if dt <= 0:
            return
        if self.tau > 0:
            e = math.exp(-dt / self.tau)
            g = self.tau * (1.0 - e)
        else:
            e, g = 1.0, dt
        F = np.eye(8)
        for i in range(4):
            F[i, i + 4] = g
            F[i + 4, i + 4] = e
        s = max(self.x[3], 1e-3)
        q = (np.array([self.q_pos * s] * 4 + [self.q_vel * s] * 4) ** 2) * dt
        self.x = F @ self.x
        self.x[2] = max(self.x[2], 1e-4)
        self.x[3] = max(self.x[3], 1e-4)
        self.P = F @ self.P @ F.T + np.diag(q)

    def update(self, box: Sequence[float]) -> None:
        z = xyxy_to_cxcywh(box)
        H = np.zeros((4, 8))
        H[:4, :4] = np.eye(4)
        s = max(z[3], 1e-3)
        R = np.diag([(self.r * s) ** 2] * 4)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ H) @ self.P

    def box(self) -> np.ndarray:
        return cxcywh_to_xyxy(self.x[:4])


@dataclass
class Tracklet:
    """Непрерывный кусок трека (результат ByteTrack до склейки)."""
    tid: int
    group: str
    times: list[float] = field(default_factory=list)
    boxes: list[np.ndarray] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    refs: list[Any] = field(default_factory=list)
    label: str = ""

    @property
    def t_start(self) -> float:
        return self.times[0]

    @property
    def t_end(self) -> float:
        return self.times[-1]

    def median_box(self) -> np.ndarray:
        return np.median(np.asarray(self.boxes, float), axis=0)

    def add(self, t: float, det: Any) -> None:
        self.times.append(float(t))
        self.boxes.append(np.asarray(det.box, float))
        self.scores.append(float(det.score))
        self.labels.append(str(det.label))
        self.refs.append(det.ref)


class _STrack:
    def __init__(self, tid: int, det: Any, t: float, cfg: dict, group: str, confirmed: bool) -> None:
        self.kf = KalmanBox(det.box, cfg["kf_tau_s"], cfg["kf_q_pos"], cfg["kf_q_vel"], cfg["kf_r"])
        self.t_kf = float(t)
        self.t_last = float(t)
        self.hits = 1
        self.confirmed = confirmed
        self.lost = False
        self.data = Tracklet(tid=tid, group=group)
        self.data.add(t, det)

    def predict_to(self, t: float) -> None:
        dt = float(t) - self.t_kf
        if dt > 0:
            self.kf.predict(dt)
            self.t_kf = float(t)

    def update(self, det: Any, t: float, min_hits: int) -> None:
        self.kf.update(det.box)
        self.t_last = float(t)
        self.hits += 1
        self.lost = False
        if self.hits >= min_hits:
            self.confirmed = True
        self.data.add(t, det)


class ByteTracker:
    """Трекер одной группы классов. `update(t, dets)` — по каждому обработанному кадру по порядку."""

    def __init__(self, cfg: dict, aspect: float, group: str, id_iter: Iterator[int]) -> None:
        self.cfg = cfg
        self.aspect = aspect
        self.group = group
        self.ids = id_iter
        self.alive: list[_STrack] = []
        self.done: list[_STrack] = []
        self.t_first: Optional[float] = None

    def update(self, t: float, dets: Sequence[Any]) -> None:
        c = self.cfg
        if self.t_first is None:
            self.t_first = float(t)
        for tr in self.alive:
            tr.predict_to(t)
        dets = [d for d in dets if d.score >= c["low_thr"]]
        high = [d for d in dets if d.score >= c["high_thr"]]
        low = [d for d in dets if d.score < c["high_thr"]]

        tb = np.array([tr.kf.box() for tr in self.alive]).reshape(-1, 4)
        hb = np.array([d.box for d in high]).reshape(-1, 4)
        shift = None
        if c.get("speed_gate", False):  # H-SPEED (ВЫКЛ): трек не «прыгает» на соседа без перекрытия боксов
            shift = np.array([c["speed_gate_base_diag"] + c["speed_gate_diag_s"] * max(float(t) - tr.t_last, 0.0)
                              for tr in self.alive])
        cost = fused_cost(tb, hb, self.aspect, c["dist_weight"], c["dist_sigma_diag"], c["gate_diag"], shift)
        # Штраф потерянным трекам — только для выбора пар (порог — по исходной стоимости): при почти равной
        # стоимости детекция достаётся видимому треку. Иначе в точке пересечения потерянный трек перекрытого
        # человека забирал детекцию проходящего впереди (подмена ID). 0 — прежнее поведение.
        pen = np.array([float(c.get("lost_cost_penalty", 0.0)) if tr.lost else 0.0 for tr in self.alive])
        m1, ut, ud = linear_assignment(cost + pen[:, None], c["match_thr"], accept=cost)
        for i, j in m1:
            self.alive[i].update(high[j], t, c["min_hits"])
        high_left = [high[j] for j in ud]

        rem = [self.alive[i] for i in ut if not self.alive[i].lost]
        lost_unmatched = [self.alive[i] for i in ut if self.alive[i].lost]
        rb = np.array([tr.kf.box() for tr in rem]).reshape(-1, 4)
        lb = np.array([d.box for d in low]).reshape(-1, 4)
        m2, ut2, _ = linear_assignment(1.0 - iou_matrix(rb, lb), c["low_match_thr"])
        for i, j in m2:
            rem[i].update(low[j], t, c["min_hits"])
        for tr in [rem[i] for i in ut2] + lost_unmatched:
            tr.lost = True

        first = abs(float(t) - self.t_first) < 1e-9
        for d in high_left:  # несопоставленные уверенные детекции -> новые треки
            if d.score >= c["new_track_thr"]:
                self.alive.append(_STrack(next(self.ids), d, t, c, self.group, confirmed=first))

        keep: list[_STrack] = []
        for tr in self.alive:
            age = float(t) - tr.t_last
            buf = c["track_buffer_s"] if tr.confirmed else c["tentative_buffer_s"]
            if tr.lost and age > buf:
                self.done.append(tr)
            else:
                keep.append(tr)
        self.alive = keep

    def finish(self) -> list[Tracklet]:
        out = []
        for tr in self.done + self.alive:
            if tr.confirmed and len(tr.data.times) >= max(1, int(self.cfg["min_track_obs"])):
                out.append(tr.data)
        self.done, self.alive = [], []
        out.sort(key=lambda d: (d.t_start, d.tid))
        return out
