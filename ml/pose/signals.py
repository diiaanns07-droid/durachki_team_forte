"""pose_signals: позы → SceneState с семантикой unknown.

Ключ сигнала присутствует в ``signals`` только если сигнал определим в момент t:
  * значение ≥ 0.5 — сигнал активен (после гистерезиса и требований к длительности);
  * значение < 0.5 — наблюдаемое «нет» (нужные точки видны, предмет-подтверждение есть или не нужен);
  * ключа нет — unknown (окклюзия, низкая уверенность, нет нужного предмета, нет кассира в кадре).

Жест без предмета не превращается в предметный вывод: «рука у уха» без cell_phone, «кисть ко рту» без
cup/bottle/food, «кисти не видны» — пишутся только в ``extra["gestures"]`` как подсказка, не как сигнал.
Предметы берутся ТОЛЬКО из аргумента ``tracks`` (треки с label из LABELS, например cell_phone, cup).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from ._contract import PoseFrame, PoseSceneState, Track, Zone
from .config import load_config
from .filters import gated_values, ramp, runs_from_mask, smooth_keypoint_track
from .keypoints import (
    L_KNEE, L_SHOULDER, L_WRIST, R_KNEE, R_SHOULDER, R_WRIST, WRISTS,
    PersonGeom, bbox_to_iso, kp_array, point_box_dist, point_in_polygon, point_polygon_dist,
    polygon_centroid, polygon_iso, to_iso, track_scale,
)

POSE_SIGNALS = ("phone_to_ear", "phone_in_hand", "eating_drinking", "hands_hidden",
                "cashier_facing_away", "handover_motion")
STAFF_SIGNALS = ("phone_to_ear", "phone_in_hand", "eating_drinking", "hands_hidden", "cashier_facing_away")
_TKEY = 1000.0  # ключ времени: round(t * _TKEY)


def _tkey(t: float) -> int:
    return int(round(float(t) * _TKEY))


@dataclass
class _Series:
    """Временной ряд позы одного трека."""

    tid: int
    role: str
    times: np.ndarray
    kps: np.ndarray                          # (T, 17, 3) нормированные
    bboxes: list[Optional[tuple[float, float, float, float]]]
    iso: np.ndarray = field(default_factory=lambda: np.zeros((0, 17, 3)))
    scale: float = 0.08
    geoms: list[PersonGeom] = field(default_factory=list)
    values: dict[str, list[Optional[float]]] = field(default_factory=dict)
    gestures: dict[str, list[Optional[float]]] = field(default_factory=dict)
    extras: dict[str, list[Optional[float]]] = field(default_factory=dict)
    runs: dict[str, list[tuple[float, float, float]]] = field(default_factory=dict)  # (t_start, t_end, t_peak)
    aspect: float = 0.5472
    motion: list[Optional[dict[str, Any]]] = field(default_factory=list)


class _ObjectIndex:
    """Объект-треки (не person) из аргумента tracks: поиск bbox-ов по метке и времени."""

    def __init__(self, tracks: Sequence[Track], aspect: float) -> None:
        self.by_label: dict[str, tuple[np.ndarray, list[np.ndarray]]] = {}
        tmp: dict[str, list[tuple[float, np.ndarray]]] = defaultdict(list)
        for tr in tracks or []:
            if tr.label == "person":
                continue
            for t, bb in tr.boxes:
                tmp[tr.label].append((float(t), bbox_to_iso(bb, aspect)))
        for label, items in tmp.items():
            items.sort(key=lambda x: x[0])
            self.by_label[label] = (np.array([x[0] for x in items]), [x[1] for x in items])

    def has_any(self, labels: Sequence[str]) -> bool:
        return any(lb in self.by_label for lb in labels)

    def boxes(self, labels: Sequence[str], t0: float, t1: float) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for lb in labels:
            if lb not in self.by_label:
                continue
            ts, bxs = self.by_label[lb]
            i0 = int(np.searchsorted(ts, t0, side="left"))
            i1 = int(np.searchsorted(ts, t1, side="right"))
            out.extend(bxs[i0:i1])
        return out


def _aspect_from_poses(poses: Sequence[PoseFrame], default: float) -> float:
    for p in poses:
        wh = (getattr(p, "extra", None) or {}).get("frame_wh")
        if wh and len(wh) == 2 and wh[0] and wh[1]:
            return float(wh[0]) / float(wh[1])
    return float(default)


def _box_at(track: Optional[Track], t: float, tol: float) -> Optional[tuple[float, float, float, float]]:
    if track is None or not track.boxes:
        return None
    best, best_dt = None, tol
    for bt, bb in track.boxes:
        dt = abs(float(bt) - t)
        if dt <= best_dt:
            best, best_dt = bb, dt
    return tuple(float(v) for v in best) if best is not None else None


def _assign_pseudo_ids(poses: Sequence[PoseFrame], aspect: float) -> list[int]:
    """track_id=None → псевдо-id (отрицательные) жадной ассоциацией по центру точек."""
    ids: list[int] = []
    active: dict[int, tuple[float, np.ndarray]] = {}
    next_id = -1
    for p in poses:
        if p.track_id is not None:
            ids.append(int(p.track_id))
            continue
        k = kp_array(p.keypoints)
        vis = k[:, 2] > 0.1
        c = (to_iso(k[vis], aspect)[:, :2].mean(axis=0) if vis.any() else np.array([0.5 * aspect, 0.5]))
        best, best_d = None, 0.08
        for pid, (lt, lc) in active.items():
            if 0 < p.t - lt <= 1.0:
                d = float(np.linalg.norm(c - lc))
                if d < best_d:
                    best, best_d = pid, d
        if best is None:
            best = next_id
            next_id -= 1
        active[best] = (float(p.t), c)
        ids.append(best)
    return ids


def _build_series(poses: Sequence[PoseFrame], tracks: Sequence[Track], aspect: float,
                  cfg: dict[str, Any]) -> list[_Series]:
    person_tracks = {int(t.track_id): t for t in tracks or [] if t.label == "person"}
    order = sorted(range(len(poses)), key=lambda i: (poses[i].t, -1 if poses[i].track_id is None else poses[i].track_id))
    poses_sorted = [poses[i] for i in order]
    ids = _assign_pseudo_ids(poses_sorted, aspect)
    tol = float(cfg["selection"].get("box_time_tolerance_s", 0.25))
    groups: dict[int, list[PoseFrame]] = defaultdict(list)
    for tid, p in zip(ids, poses_sorted):
        groups[tid].append(p)
    smooth_cfg = cfg.get("smoothing", {}) or {}
    sc = cfg["signals"]
    out: list[_Series] = []
    for tid, plist in groups.items():
        # дубликаты по времени (один трек, один t) — берём с большей средней уверенностью
        uniq: dict[int, PoseFrame] = {}
        for p in plist:
            key = _tkey(p.t)
            if key not in uniq or kp_array(p.keypoints)[:, 2].mean() > kp_array(uniq[key].keypoints)[:, 2].mean():
                uniq[key] = p
        plist = [uniq[k] for k in sorted(uniq)]
        times = np.array([float(p.t) for p in plist], dtype=np.float64)
        kps = np.stack([kp_array(p.keypoints) for p in plist]) if plist else np.zeros((0, 17, 3), np.float32)
        already = all(bool((getattr(p, "extra", None) or {}).get("smoothed")) for p in plist)
        if smooth_cfg.get("enabled", True) and not already and len(plist) > 1:
            kps = smooth_keypoint_track(times, kps, smooth_cfg)
        track = person_tracks.get(tid)
        bboxes = []
        for p in plist:
            bb = (getattr(p, "extra", None) or {}).get("bbox")
            bboxes.append(tuple(float(v) for v in bb) if bb else _box_at(track, float(p.t), tol))
        s = _Series(tid=tid, role=(track.role if track is not None else "unknown"), times=times, kps=kps, bboxes=bboxes,
                    aspect=aspect)
        s.iso = to_iso(kps, aspect)
        min_score = float(sc.get("kp_min_score", 0.3))
        s.scale = track_scale(list(s.iso), [bbox_to_iso(b, aspect) if b else None for b in bboxes],
                              sc.get("scale", {}) or {}, min_score)
        s.geoms = [PersonGeom(s.iso[j], s.scale, min_score) for j in range(len(times))]
        out.append(s)
    return out


# ---------------------------------------------------------------- per-signal evidence


def _apply_gate(series: _Series, name: str, ev: Sequence[Optional[float]], p: dict[str, Any]) -> None:
    """Гистерезис по известным кадрам; unknown внутри активного отрезка (≤ max_gap) — активны."""
    n = len(series.times)
    known = np.array([e is not None for e in ev], dtype=bool)
    out: list[Optional[float]] = [None] * n
    if known.any():
        idx = np.flatnonzero(known)
        vals, mask = gated_values(series.times[idx], [ev[i] for i in idx], p)
        for k, i in enumerate(idx):
            out[i] = float(vals[k])
        runs = []
        for a, b in runs_from_mask(mask):
            t_a, t_b = series.times[idx[a]], series.times[idx[b]]
            fill = float(vals[a:b + 1].mean())
            seg = [ev[idx[k]] for k in range(a, b + 1)]
            t_peak = float(series.times[idx[a + int(np.argmax(seg))]])
            runs.append((float(t_a), float(t_b), t_peak))
            for j in range(n):
                if out[j] is None and t_a <= series.times[j] <= t_b:
                    out[j] = fill
        series.runs[name] = runs
    series.values[name] = out


def _phone_to_ear(series: _Series, objs: _ObjectIndex, cfg: dict[str, Any]) -> None:
    sc = cfg["signals"]
    p = sc["phone_to_ear"]
    tol = float(sc.get("obj_time_tolerance_s", 0.5))
    n = len(series.times)
    gesture: list[Optional[float]] = [None] * n
    n_vis = [0] * n
    phone_near = [False] * n
    for j, g in enumerate(series.geoms):
        heads = g.ears()
        wrists = g.wrists()
        n_vis[j] = len(wrists)
        if not heads or not wrists:
            continue
        mouth = g.mouth()
        vals = []
        for _, w in wrists:
            d_ear = min(g.dist(w, e) for e in heads)
            e = ramp(d_ear, float(p["near"]), float(p["far"]))
            if mouth is not None and g.dist(w, mouth) < float(p["mouth_pref_ratio"]) * d_ear:
                e *= float(p["mouth_penalty"])
            vals.append(e)
        best = max(vals)
        if len(vals) == 2 and min(vals) > 0.5:
            best *= float(p["both_hands_penalty"])
        gesture[j] = best
        t = float(series.times[j])
        head = g.head()
        for box in objs.boxes(sc.get("phone_labels", ["cell_phone"]), t - tol, t + tol):
            cands = [w for _, w in wrists] + ([head] if head is not None else [])
            if cands and min(point_box_dist(c, box) for c in cands) / g.s <= float(p["phone_near"]):
                phone_near[j] = True
                break
    # отрезки жеста «рука у головы» (гистерезис + ≥ min_duration_s по известным кадрам)
    known = [i for i in range(n) if gesture[i] is not None]
    out: list[Optional[float]] = [None] * n
    gest_out: list[Optional[float]] = [None] * n
    if known:
        times_k = series.times[known]
        vals, mask = gated_values(times_k, [gesture[i] for i in known], p)
        pos = {j: k for k, j in enumerate(known)}
        in_run = np.zeros(n, dtype=bool)
        for a, b in runs_from_mask(mask):
            run_idx = [known[k] for k in range(a, b + 1)]
            t_a, t_b = series.times[run_idx[0]], series.times[run_idx[-1]]
            span = [j for j in range(n) if t_a <= series.times[j] <= t_b]
            confirmed = any(phone_near[j] for j in span)
            fill = float(vals[a:b + 1].mean())
            if confirmed:
                seg = [gesture[known[k]] for k in range(a, b + 1)]
                series.runs.setdefault("phone_to_ear", []).append(
                    (float(t_a), float(t_b), float(series.times[known[a + int(np.argmax(seg))]])))
            for j in span:
                in_run[j] = True
                gest_out[j] = fill
                if confirmed:  # предмет подтверждён — сигнал; иначе только жест (unknown)
                    out[j] = float(vals[pos[j]]) if j in pos else fill
        for k, i in enumerate(known):
            if not in_run[i]:
                gest_out[i] = float(vals[k])
                if n_vis[i] == 2 and gesture[i] <= float(p["gesture_neg"]):
                    out[i] = float(vals[k])  # обе кисти видны и далеко от головы — наблюдаемое «нет»
    series.values["phone_to_ear"] = out
    series.gestures["hand_at_head"] = gest_out


def _phone_in_hand(series: _Series, objs: _ObjectIndex, cfg: dict[str, Any]) -> None:
    sc = cfg["signals"]
    p = sc["phone_in_hand"]
    tol = float(sc.get("obj_time_tolerance_s", 0.5))
    ev: list[Optional[float]] = [None] * len(series.times)
    for j, g in enumerate(series.geoms):
        t = float(series.times[j])
        phones = objs.boxes(sc.get("phone_labels", ["cell_phone"]), t - tol, t + tol)
        if not phones:
            continue  # нет телефона в кадре — unknown (детектор мог не увидеть)
        wrists = g.wrists()
        center = g.body_center()
        far_from_body = center is not None and all(
            point_box_dist(center, b) / g.s >= float(p["body_far"]) for b in phones)
        if wrists:
            d = min(point_box_dist(w, b) for _, w in wrists for b in phones) / g.s
            e = ramp(d, float(p["near"]), float(p["far"]))
            if e >= float(p["thr_off"]) or len(wrists) == 2 or far_from_body:
                ev[j] = e
        elif far_from_body:
            ev[j] = 0.0
    _apply_gate(series, "phone_in_hand", ev, p)


def _eating(series: _Series, objs: _ObjectIndex, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    sc = cfg["signals"]
    p = sc["eating_drinking"]
    tol = float(sc.get("obj_time_tolerance_s", 0.5))
    near, far = float(p["near"]), float(p["far"])
    mid = 0.5 * (near + far)
    n = len(series.times)
    cycles: list[dict[str, Any]] = []
    for widx in WRISTS:
        last_far: Optional[tuple[float, np.ndarray, Optional[np.ndarray]]] = None
        cur: Optional[dict[str, Any]] = None
        last_t = None
        for j, g in enumerate(series.geoms):
            t = float(series.times[j])
            mouth = g.mouth()
            if not g.ok(widx) or mouth is None:
                continue
            if last_t is not None and t - last_t > float(p["max_approach_s"]):
                last_far, cur = None, None  # длинный пропуск — сброс
            last_t = t
            w = g.xy(widx)
            d = g.dist(w, mouth)
            head = g.head()
            if cur is None:
                if d >= far:
                    last_far = (t, w.copy(), None if head is None else head.copy())
                elif d <= near and last_far is not None and t - last_far[0] <= float(p["max_approach_s"]):
                    travel_w = g.dist(w, last_far[1])
                    travel_h = g.dist(head, last_far[2]) if (head is not None and last_far[2] is not None) else 0.0
                    if travel_w >= float(p["min_wrist_travel"]) and travel_w >= float(p["wrist_vs_head_ratio"]) * travel_h:
                        cur = {"wrist": widx, "t_start": last_far[0], "t_enter": t, "t_last_near": t}
            else:
                if d <= mid:
                    cur["t_last_near"] = t
                    if d <= near:
                        cur["t_last_near"] = t
                if d > mid:
                    dwell = cur["t_last_near"] - cur["t_enter"]
                    if float(p["min_dwell_s"]) - 1e-9 <= dwell + _period(series.times) <= float(p["max_dwell_s"]) + 1e-9:
                        cur["t_end"] = t
                        cycles.append(cur)
                    cur = None
                    last_far = (t, w.copy(), None if head is None else head.copy()) if d >= far else None
    # предмет (cup/bottle/food) должен быть У РТА, пока кисть у рта (dwell), а не у кисти в начале движения:
    # стакан на столе у руки + «поправила волосы» / «почесала ухо» не дают питья
    food_labels = sc.get("food_labels", ["cup", "bottle", "food"])
    for c in cycles:
        c["object"] = False
        for j, g in enumerate(series.geoms):
            t = float(series.times[j])
            if not (c["t_enter"] - tol <= t <= c["t_last_near"] + tol):
                continue
            m = g.mouth()
            if m is None:
                continue
            pts = [m]
            if g.ok(c["wrist"]) and g.dist(g.xy(c["wrist"]), m) <= mid:
                pts.append(g.xy(c["wrist"]))
            for box in objs.boxes(food_labels, t - tol, t + tol):
                if pts and min(point_box_dist(q, box) for q in pts) / g.s <= float(p["object_near"]):
                    c["object"] = True
                    break
            if c["object"]:
                break
    cycles.sort(key=lambda c: c["t_enter"])
    clusters: list[list[dict[str, Any]]] = []
    for c in cycles:
        if clusters and c["t_enter"] - clusters[-1][-1]["t_enter"] <= float(p["window_s"]):
            clusters[-1].append(c)
        else:
            clusters.append([c])
    ev: list[Optional[float]] = [None] * n
    gest: list[Optional[float]] = [None] * n
    covered = np.zeros(n, dtype=bool)
    for cl in clusters:
        t0 = min(c["t_start"] for c in cl)
        t1 = max(c["t_end"] for c in cl)
        k = len(cl)
        has_obj = any(c["object"] for c in cl)
        val = None
        if has_obj:
            val = float(p["evidence_single_object"]) if k == 1 else min(
                1.0, float(p["evidence_multi_object"]) + float(p["evidence_per_extra_cycle"]) * (k - 2))
        g_val = min(1.0, 0.3 * k)
        for j in range(n):
            if t0 <= series.times[j] <= t1:
                covered[j] = True
                ev[j] = val  # без предмета — None (unknown): «кисть ко рту» ≠ еда
                gest[j] = max(gest[j] or 0.0, g_val)
    for j, g in enumerate(series.geoms):
        if covered[j]:
            continue
        m = g.mouth()
        wr = g.wrists()
        if m is not None and len(wr) == 2 and min(g.dist(w, m) for _, w in wr) > near:
            ev[j] = 0.0  # рот и обе кисти видны, кисти не у рта — наблюдаемое «нет»
    _apply_gate(series, "eating_drinking", ev, p)
    fixed = []
    for t_a, t_b, t_pk in series.runs.get("eating_drinking", []):
        enters = [c["t_enter"] for c in cycles if c["object"] and t_a - 1e-6 <= c["t_enter"] <= t_b + 1e-6]
        fixed.append((t_a, t_b, float(min(enters)) if enters else t_pk))
    if fixed:
        series.runs["eating_drinking"] = fixed
    series.gestures["hand_to_mouth"] = gest
    return cycles


def _period(times: np.ndarray) -> float:
    if len(times) < 2:
        return 0.2
    d = np.diff(times)
    d = d[d > 1e-6]
    return float(np.median(d)) if len(d) else 0.2


def _hands_hidden(series: _Series, cfg: dict[str, Any]) -> None:
    p = cfg["signals"]["hands_hidden"]
    vis_thr = float(p["wrist_visible_score"])
    torso_thr = float(p["torso_min_score"])
    ev: list[Optional[float]] = [None] * len(series.times)
    gest: list[Optional[float]] = [None] * len(series.times)
    for j, g in enumerate(series.geoms):
        ws = (g.score(L_WRIST), g.score(R_WRIST))
        torso = min(g.score(L_SHOULDER), g.score(R_SHOULDER)) >= torso_thr
        if min(ws) >= vis_thr:
            ev[j] = 0.0  # обе кисти видны — наблюдаемое «руки не скрыты»
            gest[j] = 0.0
        elif torso and max(ws) < vis_thr:
            gest[j] = 1.0  # кисти не видны при видимом корпусе: подсказка, НЕ доказательство сокрытия
            if bool(p.get("allow_positive_from_pose", False)):
                ev[j] = 1.0
    _apply_gate(series, "hands_hidden", ev, {"thr_on": 0.6, "thr_off": 0.4, **p})
    series.gestures["hands_not_visible"] = gest


def _zones_by_name(zones: Sequence[Zone], aspect: float) -> dict[str, list[list[tuple[float, float]]]]:
    out: dict[str, list[list[tuple[float, float]]]] = defaultdict(list)
    for z in zones or []:
        if len(z.polygon) >= 3:
            out[z.name].append(polygon_iso(z.polygon, aspect))
    return out


def _facing_away(series: _Series, zones_iso: dict[str, list], cfg: dict[str, Any]) -> None:
    p = cfg["signals"]["cashier_facing_away"]
    refs = zones_iso.get("desk", []) or zones_iso.get("cash_tray", [])
    doors = zones_iso.get("door", [])
    ev: list[Optional[float]] = [None] * len(series.times)
    if not refs and bool(p.get("require_desk_zone", True)):
        series.values["cashier_facing_away"] = ev
        return
    for j, g in enumerate(series.geoms):
        f, fc = g.facing(float(p["nose_weight"]), float(p["min_nose_offset"]))
        c = g.body_center()
        if f is None or c is None or fc < float(p["min_facing_conf"]) or not refs:
            continue
        ref = min((polygon_centroid(r) for r in refs), key=lambda q: float(np.linalg.norm(q - c)))
        v = ref - c
        if float(np.linalg.norm(v)) / g.s < 0.2:
            continue
        cos = float(np.dot(f, v / np.linalg.norm(v)))
        e = ramp(cos, float(p["away_cos"]), float(p["toward_cos"]))
        for d in doors:
            vd = polygon_centroid(d) - c
            if float(np.linalg.norm(vd)) > 1e-6:
                cos_d = float(np.dot(f, vd / np.linalg.norm(vd)))
                e = max(e, 0.8 * ramp(cos_d, 1.0, float(p["door_cos"])))
        ev[j] = e
    _apply_gate(series, "cashier_facing_away", ev, p)


def _bending_and_posture(series: _Series, zones_iso: dict[str, list], cfg: dict[str, Any],
                         bending: bool = True) -> None:
    sc = cfg["signals"]
    pb = sc["bending_to_safe"]
    pp = sc["posture"]
    safes = zones_iso.get("safe", []) if bending else []
    n = len(series.times)
    head_hip = []
    for g in series.geoms:
        h, hm = g.head(), g.hip_mid()
        head_hip.append(g.dist(h, hm) if (h is not None and hm is not None) else None)
    valid = [v for v in head_hip if v is not None]
    med = float(np.median(valid)) if len(valid) >= 3 else None
    heights = [b[3] - b[1] for b in series.bboxes if b]
    h90 = float(np.percentile(heights, 90)) if len(heights) >= 3 else None
    ev: list[Optional[float]] = [None] * n
    sit: list[Optional[float]] = [None] * n
    for j, g in enumerate(series.geoms):
        if safes:
            pts = [w for _, w in g.wrists()]
            head = g.head()
            if pts or head is not None:
                margin = float(pb["zone_margin"])
                w_in = any(point_polygon_dist(w, s) <= margin for w in pts for s in safes)
                h_in = head is not None and any(point_polygon_dist(head, s) <= margin for s in safes)
                near_v = 1.0 if w_in else (0.7 if h_in else 0.0)
                bend = 0.5
                if med and head_hip[j] is not None:
                    bend = ramp(head_hip[j] / med, float(pb["bend_ratio"]), 1.0)
                ev[j] = near_v * (float(pb["base"]) + (1.0 - float(pb["base"])) * bend)
        # поза сидя/стоя (эвристика)
        sm, hm = g.shoulder_mid(), g.hip_mid()
        knees = [g.xy(k) for k in (L_KNEE, R_KNEE) if g.ok(k)]
        val, conf = None, 0.0
        if sm is not None and hm is not None and knees:
            torso = sm - hm
            thigh = np.mean(knees, axis=0) - hm
            nt, nh = float(np.linalg.norm(torso)), float(np.linalg.norm(thigh))
            if nt > 1e-6 and nh > 1e-6:
                ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(torso, -thigh)) / (nt * nh)))))
                val = ramp(ang, float(pp["sit_angle_hi_deg"]), float(pp["sit_angle_lo_deg"]))
                conf = 0.7
        elif h90 and series.bboxes[j]:
            b = series.bboxes[j]
            r = (b[3] - b[1]) / h90
            thr = float(pp["height_ratio_sit"])
            val = ramp(r, thr - 0.15, thr + 0.1)
            conf = 0.4
        if val is not None and conf >= float(pp["min_conf"]):
            sit[j] = val
    if safes:
        _apply_gate(series, "_bending_to_safe", ev, pb)
        series.extras["bending_to_safe"] = series.values.pop("_bending_to_safe")
    else:
        series.extras["bending_to_safe"] = [None] * n
    series.extras["sitting"] = sit


def _handover(series_list: list[_Series], zones_iso: dict[str, list], cfg: dict[str, Any]
              ) -> dict[tuple[int, int], tuple[np.ndarray, list[Optional[float]], list[tuple[float, float, float]]]]:
    p = cfg["signals"]["handover_motion"]
    zone_polys = [poly for name in p.get("zone_names", []) for poly in zones_iso.get(name, [])]
    by_time: dict[int, list[tuple[_Series, int]]] = defaultdict(list)
    for s in series_list:
        for j, t in enumerate(s.times):
            by_time[_tkey(t)].append((s, j))
    pair_ev: dict[tuple[int, int], dict[int, Optional[float]]] = defaultdict(dict)
    for key, items in by_time.items():
        for ia in range(len(items)):
            for ib in range(ia + 1, len(items)):
                (sa, ja), (sb, jb) = items[ia], items[ib]
                if sa.tid == sb.tid:
                    continue
                ga, gb = sa.geoms[ja], sb.geoms[jb]
                pair = (min(sa.tid, sb.tid), max(sa.tid, sb.tid))
                ca, cb = ga.body_center(), gb.body_center()
                wa, wb = ga.wrists(), gb.wrists()
                if ca is None or cb is None or not wa or not wb:
                    pair_ev[pair][key] = None
                    continue
                s_mean = 0.5 * (ga.s + gb.s)
                if float(np.linalg.norm(ca - cb)) / s_mean < float(p["same_person_dist"]):
                    continue  # вероятно дубликат одного человека
                best, mid = math.inf, None
                for _, xa in wa:
                    if ga.dist(xa, ca) < float(p["min_reach"]):
                        continue
                    for _, xb in wb:
                        if gb.dist(xb, cb) < float(p["min_reach"]):
                            continue
                        d = float(np.linalg.norm(xa - xb)) / s_mean
                        if d < best:
                            best, mid = d, 0.5 * (xa + xb)
                e = 0.0 if not math.isfinite(best) else ramp(best, float(p["near"]), float(p["far"]))
                if e > 0 and zone_polys and mid is not None:
                    inside = any(point_polygon_dist(mid, z) <= float(p["zone_margin"]) for z in zone_polys)
                    if not inside:
                        e *= float(p["outside_zone_factor"])
                if e >= float(p["thr_off"]) or (len(wa) == 2 and len(wb) == 2):
                    pair_ev[pair][key] = e
                else:
                    pair_ev[pair][key] = None  # не все кисти видны — «нет передачи» не утверждаем
    out: dict[tuple[int, int], tuple[np.ndarray, list[Optional[float]], list[tuple[float, float, float]]]] = {}
    for pair, d in pair_ev.items():
        keys = sorted(d)
        fake = _Series(tid=-999, role="unknown", times=np.array([k / _TKEY for k in keys]),
                       kps=np.zeros((0, 17, 3)), bboxes=[])
        _apply_gate(fake, "handover_motion", [d[k] for k in keys], p)
        out[pair] = (np.array(keys), fake.values["handover_motion"], fake.runs.get("handover_motion", []))
    return out


# ---------------------------------------------------------------- public


def _body_point(series: _Series, j: int) -> Optional[np.ndarray]:
    """Центр корпуса (плечи/таз из позы); иначе центр bbox трека (изотропные координаты)."""
    g = series.geoms[j]
    sm, hm = g.shoulder_mid(), g.hip_mid()
    if sm is not None and hm is not None:
        return (sm + hm) / 2.0
    if sm is not None:
        return sm
    b = series.bboxes[j] if j < len(series.bboxes) else None
    if b:
        bi = bbox_to_iso(b, series.aspect)
        return np.array([(bi[0] + bi[2]) / 2.0, (bi[1] + bi[3]) / 2.0])
    return None


def _motion(series: _Series, zones_iso: dict[str, list], cfg: dict[str, Any]) -> None:
    """sitting / standing / walking + направление движения (в масштабах тела, без пикселей и цвета).

    walking — скорость центра корпуса в окне window_s ≥ порогов (гистерезис, ≥ min_duration_s);
    иначе поза сидя/стоя (эвристика); нет данных — unknown. Направление и «к какой зоне / от какой»
    считаются по смещению за весь отрезок ходьбы.
    """
    p = cfg["signals"]["motion"]
    n = len(series.times)
    pts = [_body_point(series, j) for j in range(n)]
    half = float(p["window_s"]) / 2.0
    speed: list[Optional[float]] = [None] * n
    vel: list[Optional[np.ndarray]] = [None] * n
    tt = np.asarray(series.times, dtype=np.float64)
    lo = np.searchsorted(tt, tt - half - 1e-9, side="left")
    hi = np.searchsorted(tt, tt + half + 1e-9, side="right") - 1
    nxt, prv = [-1] * (n + 1), [-1] * n
    for k in range(n - 1, -1, -1):
        nxt[k] = k if pts[k] is not None else nxt[k + 1]
    for k in range(n):
        prv[k] = k if pts[k] is not None else (prv[k - 1] if k else -1)
    for j in range(n):
        a, b = nxt[int(lo[j])], prv[int(hi[j])]
        if a < 0 or b < 0 or b <= a:
            continue
        dt = float(series.times[b] - series.times[a])
        if dt < 0.6 * 2 * half:
            continue
        v = (pts[b] - pts[a]) / dt / series.scale
        vel[j], speed[j] = v, float(np.linalg.norm(v))
    ev = [None if sp is None else ramp(sp, float(p["walk_speed_full"]), float(p["walk_speed_min"])) for sp in speed]
    _apply_gate(series, "_walking", ev, p)
    walking = series.values.pop("_walking")
    runs = series.runs.pop("_walking", [])
    zone_names = list(p.get("zone_names", []))
    run_info = []
    for t_a, t_b, _ in runs:
        ks = [k for k in range(n) if t_a - 1e-6 <= series.times[k] <= t_b + 1e-6 and pts[k] is not None]
        info: dict[str, Any] = {"t_start": round(t_a, 3), "t_end": round(t_b, 3), "direction": None,
                                "toward": None, "away_from": None}
        if len(ks) >= 2:
            d = pts[ks[-1]] - pts[ks[0]]
            nd = float(np.linalg.norm(d))
            if nd > 1e-9:
                u = d / nd
                info["direction"] = (round(float(u[0]), 3), round(float(u[1]), 3))
                best_to, best_from = (None, float(p["toward_cos"])), (None, -float(p["toward_cos"]))
                for zn in zone_names:
                    for poly in zones_iso.get(zn, []):
                        vz = polygon_centroid(poly) - pts[ks[0]]
                        if float(np.linalg.norm(vz)) < 1e-6:
                            continue
                        cos = float(np.dot(u, vz / np.linalg.norm(vz)))
                        if cos >= best_to[1]:
                            best_to = (zn, cos)
                        if cos <= best_from[1]:
                            best_from = (zn, cos)
                info["toward"], info["away_from"] = best_to[0], best_from[0]
        run_info.append(info)
    sit = series.extras.get("sitting", [None] * n)
    out: list[Optional[dict[str, Any]]] = []
    for j in range(n):
        t = float(series.times[j])
        w = walking[j]
        run = next((r for r in run_info if r["t_start"] - 1e-6 <= t <= r["t_end"] + 1e-6), None)
        if w is not None and w >= 0.5:
            state = "walking"
        elif sit[j] is not None and sit[j] >= float(p["sit_hi"]):
            state = "sitting"
        elif sit[j] is not None and sit[j] <= float(p["sit_lo"]):
            state = "standing"
        else:
            state = "unknown"
        if state == "unknown" and speed[j] is None and pts[j] is None:
            out.append(None)
            continue
        out.append({"state": state, "speed": None if speed[j] is None else round(speed[j], 3),
                    "run": run if state == "walking" else None})
    series.motion = out


def _item_placements(series_list: list[_Series], tracks: Sequence[Track], zones_iso: dict[str, list],
                     aspect: float, cfg: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
    """Предмет положен на стол: объект-трек «лёг» (неподвижен ≥ persist_s) в зоне desk, перед этим к нему
    подходила кисть человека, затем ушла. None — неопределимо (нет зоны desk или объект-треков).

    Не событие: предмет лежал с начала видео; новый трек на месте старого (смена id); кисти рядом не было.
    """
    p = cfg["signals"]["item_placed_on_desk"]
    desks = [poly for nm in p.get("zone_names", ["desk"]) for poly in zones_iso.get(nm, [])]
    labels = set(p.get("labels", []))
    objs = [tr for tr in tracks or [] if tr.label in labels and tr.boxes]
    if not desks or not objs:
        return None
    t_video0 = min([float(s.times[0]) for s in series_list if len(s.times)] +
                   [float(min(t for t, _ in tr.boxes)) for tr in tracks or [] if tr.boxes] or [0.0])
    persist, stable = float(p["persist_s"]), float(p["stable_dist"])
    events: list[dict[str, Any]] = []
    for tr in objs:
        items = sorted(((float(t), bbox_to_iso(b, aspect)) for t, b in tr.boxes), key=lambda x: x[0])
        ts = np.array([x[0] for x in items])
        cs = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for _, b in items])
        settle = None
        for k in range(len(items)):
            if ts[-1] < ts[k] + persist - 0.25:
                break
            win = (ts >= ts[k]) & (ts <= ts[k] + persist)
            if np.all(np.linalg.norm(cs[win] - cs[k], axis=1) <= stable) and any(
                    point_polygon_dist(cs[k], d) <= float(p["zone_margin"]) for d in desks):
                settle = k
                break
        if settle is None:
            continue
        t_s, box_s = float(ts[settle]), items[settle][1]
        if settle == 0 and t_s - t_video0 <= 0.5:
            continue  # лежал с начала видео
        reid = False
        for other in objs:
            if other is tr or other.label != tr.label:
                continue
            for t_o, b_o in other.boxes:
                bo = bbox_to_iso(b_o, aspect)
                c_o = np.array([(bo[0] + bo[2]) / 2, (bo[1] + bo[3]) / 2])
                if t_s - float(p["reid_gap_s"]) <= float(t_o) < t_s and np.linalg.norm(c_o - cs[settle]) <= 2 * stable:
                    reid = True
                    break
            if reid:
                break
        if reid:
            continue  # тот же предмет под новым id
        best = None
        for s in series_list:
            t_near = None
            for j, g in enumerate(s.geoms):
                t = float(s.times[j])
                if not (t_s - float(p["pre_s"]) <= t <= t_s + float(p["post_s"])):
                    continue
                wr = g.wrists()
                if wr and min(point_box_dist(w, box_s) for _, w in wr) / g.s <= float(p["near"]):
                    t_near = t
                    break
            if t_near is None:
                continue
            released = False
            for j, g in enumerate(s.geoms):
                t = float(s.times[j])
                if t_near < t <= t_s + float(p["release_s"]):
                    wr = g.wrists()
                    if wr and min(point_box_dist(w, box_s) for _, w in wr) / g.s >= float(p["far"]):
                        released = True
                        break
            if released and (best is None or t_near < best[0]):
                best = (t_near, s)
        if best is None:
            continue
        t_on, who = best
        x1, y1, x2, y2 = (float(v) for v in items[settle][1])  # iso → нормированный bbox
        events.append({"t_start": round(t_on, 3), "t_settle": round(t_s, 3),
                       "t_end": round(t_s + float(p["hold_s"]), 3), "person_track_id": who.tid, "role": who.role,
                       "object_track_id": int(tr.track_id), "label": tr.label,
                       "bbox": (x1 / aspect, y1, x2 / aspect, y2)})
    return events


def _run_at(runs: Sequence[tuple[float, float, float]], t: float) -> Optional[dict[str, float]]:
    for t_a, t_b, t_pk in runs:
        if t_a - 1e-6 <= t <= t_b + 1e-6:
            return {"t_start": round(t_a, 3), "t_end": round(t_b, 3), "t_peak": round(t_pk, 3)}
    return None


def compute_pose_signals(poses: Sequence[PoseFrame], tracks: Sequence[Track], zones: Sequence[Zone],
                         cfg: Optional[dict[str, Any]] = None) -> list[PoseSceneState]:
    """Реализация pose_signals с явным конфигом (для тестов/тюнинга)."""
    cfg = cfg or load_config()
    sc = cfg["signals"]
    enabled = [nm for nm in POSE_SIGNALS if nm in set(sc.get("enabled", POSE_SIGNALS))]
    extras_on = sc.get("extras", {}) or {}
    poses = list(poses or [])
    tracks = list(tracks or [])
    aspect = _aspect_from_poses(poses, float(sc.get("default_aspect_wh", 0.5472)))
    objs = _ObjectIndex(tracks, aspect)
    zones_iso = _zones_by_name(zones, aspect)
    series_list = _build_series(poses, tracks, aspect, cfg)
    staff_roles = set(sc.get("staff_roles", ["cashier"]))
    for s in series_list:
        if "phone_to_ear" in enabled:
            _phone_to_ear(s, objs, cfg)
        if "phone_in_hand" in enabled:
            _phone_in_hand(s, objs, cfg)
        if "eating_drinking" in enabled:
            _eating(s, objs, cfg)
        if "hands_hidden" in enabled:
            _hands_hidden(s, cfg)
        if "cashier_facing_away" in enabled:
            _facing_away(s, zones_iso, cfg)
        _bending_and_posture(s, zones_iso, cfg, bending=bool(extras_on.get("bending_to_safe", False)))
        if extras_on.get("motion", False):
            _motion(s, zones_iso, cfg)
    pairs = _handover(series_list, zones_iso, cfg) if "handover_motion" in enabled else {}
    roles = {s.tid: s.role for s in series_list}
    staff_enabled = [nm for nm in STAFF_SIGNALS if nm in enabled]

    # все моменты времени: позы + наблюдения person-треков (там, где позы нет, всё unknown)
    tkeys = {_tkey(p.t) for p in poses}
    for tr in tracks:
        if tr.label == "person":
            tkeys.update(_tkey(t) for t, _ in tr.boxes)
    index: dict[int, list[tuple[_Series, int]]] = defaultdict(list)
    for s in series_list:
        for j, t in enumerate(s.times):
            index[_tkey(t)].append((s, j))
    pair_index: dict[int, list[tuple[tuple[int, int], Optional[float]]]] = defaultdict(list)
    pair_runs = {pair: runs for pair, (_, _, runs) in pairs.items()}
    for pair, (keys, vals, _) in pairs.items():
        for k, v in zip(keys, vals):
            pair_index[int(k)].append((pair, v))

    from .status import status as _pose_status

    pose_status = _pose_status().get("state", "unknown")
    placements = (_item_placements(series_list, tracks, zones_iso, aspect, cfg)
                  if extras_on.get("item_placed_on_desk", False) else None)
    states: list[PoseSceneState] = []
    for key in sorted(tkeys):
        t = key / _TKEY
        signals: dict[str, float] = {}
        extra: dict[str, Any] = {"n_posed": len(index.get(key, [])), "pose_status": pose_status,
                                 "by_track": {}, "gestures": {}, "track_ids": {}, "boxes": {}, "runs": {},
                                 "bending_to_safe": None, "sitting": None, "standing": None,
                                 "handover_pairs": [], "enabled": list(enabled)}
        staff_items = [(s, j) for s, j in index.get(key, []) if s.role in staff_roles]
        extra["staff_track_ids"] = [s.tid for s, _ in staff_items]
        for s, j in index.get(key, []):
            entry = {"role": s.role,
                     "signals": {nm: s.values[nm][j] for nm in staff_enabled if s.values.get(nm) and s.values[nm][j] is not None},
                     "gestures": {gn: gv[j] for gn, gv in s.gestures.items() if gv[j] is not None},
                     "bending_to_safe": s.extras.get("bending_to_safe", [None] * (j + 1))[j],
                     "sitting": s.extras.get("sitting", [None] * (j + 1))[j],
                     "bbox": s.bboxes[j]}
            extra["by_track"][s.tid] = entry
        for nm in staff_enabled:
            best, best_s = None, None
            for s, j in staff_items:
                v = s.values.get(nm, [None] * (j + 1))[j]
                if v is not None and (best is None or v > best):
                    best, best_s = v, (s, j)
            if best is not None:
                signals[nm] = round(float(best), 4)
                if best >= 0.5 and best_s is not None:
                    extra["track_ids"][nm] = [best_s[0].tid]
                    if best_s[0].bboxes[best_s[1]]:
                        extra["boxes"][nm] = best_s[0].bboxes[best_s[1]]
                    run = _run_at(best_s[0].runs.get(nm, []), t)
                    if run:
                        extra["runs"][nm] = run  # t_start — начало перехода (таймкод события), t_peak — максимум
        for gn in ("hand_at_head", "hand_to_mouth", "hands_not_visible"):
            vals = [s.gestures.get(gn, [None] * (j + 1))[j] for s, j in staff_items]
            vals = [v for v in vals if v is not None]
            if vals:
                extra["gestures"][gn] = round(float(max(vals)), 4)
        bvals = [s.extras["bending_to_safe"][j] for s, j in staff_items if s.extras["bending_to_safe"][j] is not None]
        if bvals:
            extra["bending_to_safe"] = round(float(max(bvals)), 4)
        svals = [s.extras["sitting"][j] for s, j in staff_items if s.extras["sitting"][j] is not None]
        if svals:
            extra["sitting"] = round(float(svals[0]), 4)
            extra["standing"] = round(1.0 - float(svals[0]), 4)
        hv = [(pair, v) for pair, v in pair_index.get(key, []) if v is not None]
        if hv:
            pair, v = max(hv, key=lambda x: x[1])
            signals["handover_motion"] = round(float(v), 4)
            for pr, vv in hv:
                if vv >= 0.5:
                    extra["handover_pairs"].append([pr[0], pr[1], roles.get(pr[0], "unknown"),
                                                    roles.get(pr[1], "unknown"), round(float(vv), 4)])
            if v >= 0.5:
                extra["track_ids"]["handover_motion"] = [pair[0], pair[1]]
                run = _run_at(pair_runs.get(pair, []), t)
                if run:
                    extra["runs"]["handover_motion"] = run
        if extras_on.get("motion", False):
            main = staff_items[0] if staff_items else None
            extra["motion"] = (main[0].motion[main[1]] if main is not None and main[0].motion else None)
            for s, j in index.get(key, []):
                if s.motion:
                    extra["by_track"][s.tid]["motion"] = s.motion[j]
        if extras_on.get("item_placed_on_desk", False):
            if placements is None:
                extra["item_placed_on_desk"] = None  # нет зоны desk или объект-треков — unknown
            else:
                act = [e for e in placements if e["t_start"] - 1e-6 <= t <= e["t_end"] + 1e-6]
                # «не клали» утверждаем, только если в момент t есть хоть одна поза (кисти видели); иначе unknown
                extra["item_placed_on_desk"] = 1.0 if act else (0.0 if index.get(key) else None)
                extra["placements"] = act
        extra["unknown"] = [nm for nm in enabled if nm not in signals]
        states.append(PoseSceneState(t=t, signals=signals, extra=extra))
    return states


def pose_signals(poses: Sequence[PoseFrame], tracks: Sequence[Track], zones: Sequence[Zone]) -> list[PoseSceneState]:
    """Публичная функция контракта: ml.pose.pose_signals(poses, tracks, zones) -> list[SceneState]."""
    return compute_pose_signals(poses, tracks, zones)
