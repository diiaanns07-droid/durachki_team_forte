"""Привязка детекций и треков к сэмплам clip.times; люди на каждом сэмпле с вероятностями ролей."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from .geometry import BBox, bbox_gap


@dataclass
class PersonObs:
    bbox: BBox
    score: float
    track_id: Optional[int] = None
    p_cashier: float = 0.0
    p_collector: float = 0.0
    explicit_role: bool = False


@dataclass
class Timeline:
    times: np.ndarray
    dets: list[list[Any]]                 # детекции, привязанные к сэмплу (с учётом every_n)
    persons: list[list[PersonObs]] = field(default_factory=list)
    det_stride: int = 1
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.times)

    def dets_of(self, i: int, labels: Sequence[str], min_score: float = 0.0) -> list[Any]:
        return [d for d in self.dets[i] if d.label in labels and float(d.score) >= min_score]


def _nearest_idx(times: np.ndarray, t: float) -> int:
    j = int(np.searchsorted(times, t))
    if j <= 0:
        return 0
    if j >= len(times):
        return len(times) - 1
    return j if abs(times[j] - t) < abs(times[j - 1] - t) else j - 1


def sample_dt(times: np.ndarray) -> float:
    if len(times) < 2:
        return 0.2
    return float(np.median(np.diff(times)))


def assign_dets(times: np.ndarray, dets: Sequence[Any], hold_s: float) -> tuple[list[list[Any]], int]:
    """Детекции → ближайший сэмпл. Если детектор шёл через every_n кадров — заполняем пропуски
    ближайшим обработанным кадром (оценка шага по моде разностей)."""
    n = len(times)
    out: list[list[Any]] = [[] for _ in range(n)]
    if n == 0:
        return out, 1
    tol = max(hold_s, sample_dt(times) * 0.51)
    for d in dets or []:
        try:
            j = _nearest_idx(times, float(d.t))
        except (TypeError, ValueError):
            continue
        if abs(times[j] - float(d.t)) <= tol:
            out[j].append(d)
    processed = [i for i in range(n) if out[i]]
    stride = 1
    if len(processed) >= 3:
        diffs = Counter(np.diff(processed).tolist())
        stride = int(max(1, diffs.most_common(1)[0][0]))
    if stride > 1:
        proc = np.array(processed)
        filled: list[list[Any]] = [list(x) for x in out]
        for i in range(n):
            if out[i]:
                continue
            k = int(np.argmin(np.abs(proc - i)))
            if abs(int(proc[k]) - i) <= stride // 2 + (stride % 2):
                filled[i] = out[int(proc[k])]
        out = filled
    return out, stride


def _track_arrays(track: Any) -> tuple[np.ndarray, np.ndarray]:
    boxes = sorted(track.boxes or [], key=lambda tb: float(tb[0]))
    if not boxes:
        return np.zeros(0), np.zeros((0, 4))
    ts = np.array([float(tb[0]) for tb in boxes])
    bb = np.array([tuple(map(float, tb[1])) for tb in boxes])
    return ts, bb


def _explicit_roles(track: Any, min_conf: float) -> Optional[tuple[float, float]]:
    role = getattr(track, "role", "unknown") or "unknown"
    conf = float(getattr(track, "role_conf", 0.0) or 0.0)
    if role == "unknown" or conf <= 0:
        return None
    q = float(np.clip(conf, 0.0, 1.0))
    rest = (1.0 - q) / 2.0
    pc = q if role == "cashier" else rest
    pcol = q if role == "collector" else rest
    if q < min_conf:
        return None
    return pc, pcol


def _member(b: BBox, poly: Any, anchor_y: float, ofull: float) -> float:
    from .presence import member

    return member(b, [poly], anchor_y, ofull)


def build_persons(
    tl: Timeline,
    tracks: Sequence[Any],
    workplace: Optional[list[tuple[float, float]]],
    cfg: dict[str, Any],
    hold_s: float,
) -> None:
    """Заполняет tl.persons: кто виден на каждом сэмпле, P(кассир), P(инкассатор)."""
    pcfg = cfg["presence"]
    hcfg = pcfg["heuristic"]
    min_score = float(pcfg["person_min_score"])
    anchor_y = float(pcfg["anchor_y"])
    ofull = float(pcfg["overlap_full"])
    n = len(tl)
    dt = sample_dt(tl.times)
    tl.persons = [[] for _ in range(n)]
    person_tracks = [tr for tr in tracks or [] if getattr(tr, "label", "person") == "person" and tr.boxes]

    det_score: dict[tuple[int, int], float] = {}
    for i in range(n):
        for d in tl.dets[i]:
            if d.label == "person" and d.track_id is not None:
                det_score[(i, int(d.track_id))] = max(det_score.get((i, int(d.track_id)), 0.0), float(d.score))

    bag_boxes = [[tuple(d.bbox) for d in tl.dets_of(i, ("cash_bag",), 0.2)] for i in range(n)]

    if person_tracks:
        for tr in person_tracks:
            ts, bb = _track_arrays(tr)
            vis: list[tuple[int, BBox]] = []
            # hold заполняет только разрывы ВНУТРИ трека: за последним боксом трек не «продлевается»
            # (иначе уход/вход сдвигаются на hold_s и портят таймкод)
            edge = dt * 0.5 + 1e-6
            for i, t in enumerate(tl.times):
                if t < ts[0] - edge or t > ts[-1] + edge:
                    continue
                j = _nearest_idx(ts, float(t))
                if abs(ts[j] - t) <= hold_s:
                    vis.append((i, tuple(bb[j])))
            if not vis:
                continue
            explicit = _explicit_roles(tr, float(pcfg["role_min_conf"]))
            # эвристика по треку целиком (офлайн): доля времени в зоне кассира, длительность, сумка рядом
            if workplace:
                frac_ws = float(np.mean([_member(b, workplace, anchor_y, ofull) for _, b in vis]))
            else:
                frac_ws = 0.5
            dur = len(vis) * dt
            pc_h = float(hcfg["cashier_base"]) + float(hcfg["cashier_gain"]) * frac_ws * min(1.0, dur / max(float(hcfg["cashier_min_track_s"]), 1e-6))
            pc_h = float(np.clip(pc_h, 0.0, 0.9))
            bag_hits = sum(1 for i, b in vis if any(bbox_gap(b, bg) <= float(hcfg["collector_bag_dist"]) for bg in bag_boxes[i]))
            frac_bag = bag_hits / len(vis)
            pcol_h = float(min(0.85, 0.85 * frac_bag / max(float(hcfg["collector_bag_frac"]), 1e-6)) * (1.0 - pc_h))
            role = getattr(tr, "role", "unknown") or "unknown"
            if explicit is not None:
                pc, pcol = explicit
            elif role != "unknown" and float(getattr(tr, "role_conf", 0.0) or 0.0) > 0:
                # неуверенная роль трекера — смешиваем с эвристикой
                q = float(tr.role_conf)
                pc = 0.5 * (q if role == "cashier" else (1 - q) / 2) + 0.5 * pc_h
                pcol = 0.5 * (q if role == "collector" else (1 - q) / 2) + 0.5 * pcol_h
            else:
                pc, pcol = pc_h, pcol_h
            for i, b in vis:
                sc = det_score.get((i, int(tr.track_id)), 0.8)
                if sc < min_score:
                    continue
                tl.persons[i].append(PersonObs(bbox=b, score=sc, track_id=int(tr.track_id), p_cashier=pc, p_collector=pcol, explicit_role=explicit is not None))
        return

    # нет треков — люди из детекций по кадрам (роль по зоне: кандидат в кассиры — тот, кто глубже в зоне)
    if tracks is not None and len(tracks) == 0 and any(d.label == "person" for ds in tl.dets for d in ds):
        tl.warnings.append("no person tracks: roles from per-frame zone heuristic")
    for i in range(n):
        for d in tl.dets[i]:
            if d.label != "person" or float(d.score) < min_score:
                continue
            b = tuple(map(float, d.bbox))
            m = _member(b, workplace, anchor_y, ofull) if workplace else 0.5
            pc = float(np.clip(float(hcfg["cashier_base"]) + float(hcfg["cashier_gain"]) * m, 0.0, 0.9))
            near_bag = any(bbox_gap(b, bg) <= float(hcfg["collector_bag_dist"]) for bg in bag_boxes[i])
            pcol = 0.6 * (1.0 - pc) if near_bag else 0.0
            tl.persons[i].append(PersonObs(bbox=b, score=float(d.score), track_id=d.track_id, p_cashier=pc, p_collector=pcol))


def build_timeline(times: Sequence[float], dets: Sequence[Any], cfg: dict[str, Any]) -> Timeline:
    t = np.asarray(list(times), dtype=np.float64)
    hold = float(cfg["time"]["hold_s"])
    per, stride = assign_dets(t, dets, hold)
    tl = Timeline(times=t, dets=per, det_stride=stride)
    if stride > 1:
        tl.warnings.append(f"detections every {stride} samples: gaps filled by nearest processed frame")
    return tl


def group_by_track(dets: Sequence[Any]) -> dict[int, list[Any]]:
    g: dict[int, list[Any]] = defaultdict(list)
    for d in dets:
        if d.track_id is not None:
            g[int(d.track_id)].append(d)
    return g
