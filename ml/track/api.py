"""ml.track.track(clip, dets) -> (tracks, zones): трекинг, склейка, авто-зоны, роли.

`track_full()` возвращает то же + `info` (provenance ролей и зон, журнал склеек, тайминги,
предупреждения) — для A03/A08/A09 и отладки. Сигнатура `track()` заморожена контрактом.
"""
from __future__ import annotations

import itertools
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from . import roles as roles_mod
from . import zones as zones_mod
from .appearance import tracklet_appearance
from .bytetrack import ByteTracker, Tracklet
from .config import load_config
from .contract import Track, Zone
from .stitch import merge_chain, stitch_persons, stitch_static
from .views import TrackView, make_view
from .visibility import loss_intervals

log = logging.getLogger("ml.track")


@dataclass
class TrackResult:
    tracks: list
    zones: list
    info: dict = field(default_factory=dict)


@dataclass
class _Det:
    t: float
    label: str
    box: np.ndarray
    score: float
    ref: Any


def _get(o: Any, k: str, default: Any = None) -> Any:
    if isinstance(o, dict):
        return o.get(k, default)
    return getattr(o, k, default)


def _normalize_dets(dets: Optional[Sequence[Any]]) -> tuple[list[_Det], int]:
    out: list[_Det] = []
    bad = 0
    for d in dets or []:
        try:
            t = float(_get(d, "t"))
            label = str(_get(d, "label"))
            b = np.clip(np.asarray(_get(d, "bbox"), float).reshape(-1)[:4], 0.0, 1.0)
            s = float(_get(d, "score", 1.0) if _get(d, "score", 1.0) is not None else 1.0)
        except Exception:
            bad += 1
            continue
        if len(b) != 4 or not np.all(np.isfinite(b)) or not math.isfinite(t) or b[2] - b[0] <= 1e-4 or b[3] - b[1] <= 1e-4:
            bad += 1
            continue
        out.append(_Det(t, label, b, s, d))
    return out, bad


def _clip_meta(clip: Any, cfg: dict) -> tuple[float, Optional[np.ndarray], Optional[float]]:
    w, h = _get(clip, "width"), _get(clip, "height")
    try:
        aspect = float(w) / float(h) if w and h else float(cfg["default_aspect"])
    except Exception:
        aspect = float(cfg["default_aspect"])
    times = _get(clip, "times")
    try:
        times = np.asarray(list(times), float) if times is not None and len(times) else None
    except Exception:
        times = None
    dur = _get(clip, "duration")
    try:
        dur = float(dur) if dur is not None else None
    except Exception:
        dur = None
    return aspect, times, dur


def _build_frames(times: Optional[np.ndarray], dets: list[_Det]) -> tuple[list[float], dict[int, list[_Det]]]:
    """Кадры, реально прошедшие через детектор (учёт every_n), и детекции по кадрам."""
    if not dets:
        return [], {}
    if times is not None and len(times) > 1:
        idx = np.clip(np.searchsorted(times, [d.t for d in dets]), 0, len(times) - 1)
        prev = np.clip(idx - 1, 0, len(times) - 1)
        tt = np.array([d.t for d in dets])
        idx = np.where(np.abs(times[prev] - tt) < np.abs(times[idx] - tt), prev, idx)
        tol = max(0.5 * float(np.median(np.diff(times))), 0.02)
        if np.all(np.abs(times[idx] - tt) <= tol):
            uniq = np.unique(idx)
            step = 1
            if len(uniq) > 1:
                vals, cnt = np.unique(np.diff(uniq), return_counts=True)
                step = max(1, int(vals[int(np.argmax(cnt))]))
            proc = set(range(int(uniq[0]), int(uniq[-1]) + 1, step)) | set(int(u) for u in uniq)
            frames = sorted(proc)
            by: dict[int, list[_Det]] = defaultdict(list)
            for d, i in zip(dets, idx):
                by[int(i)].append(d)
            return [float(times[i]) for i in frames], {k: by.get(i, []) for k, i in enumerate(frames)}
    keys = sorted({round(d.t, 4) for d in dets})
    pos = {k: i for i, k in enumerate(keys)}
    by2: dict[int, list[_Det]] = defaultdict(list)
    for d in dets:
        by2[pos[round(d.t, 4)]].append(d)
    return [float(k) for k in keys], dict(by2)


def _group_of(label: str, cfg: dict) -> str:
    if label in cfg["person_labels"]:
        return "person"
    for g, labels in cfg["tracker"]["label_groups"].items():
        if label in labels:
            return g
    return label


def _resolve_label(tr: Tracklet, cfg: dict) -> str:
    tc = cfg["tracker"]
    if tr.group == "person":
        return "person"
    if tr.group in tc["group_label"]:
        return tc["group_label"][tr.group]
    labels = tr.labels
    pref = tc["prefer_label"].get(tr.group)
    if pref and labels and labels.count(pref["label"]) / len(labels) >= pref["min_frac"]:
        return pref["label"]
    vals, cnt = np.unique(np.asarray(labels), return_counts=True)
    return str(vals[int(np.argmax(cnt))]) if len(vals) else tr.group


def _fill_gaps(tr: Tracklet, frame_times: np.ndarray, max_gap: float) -> list[tuple[float, np.ndarray, bool]]:
    rows = [(t, b, True) for t, b in zip(tr.times, tr.boxes)]
    if max_gap <= 0 or len(rows) < 2:
        return rows
    out = [rows[0]]
    for (ta, ba, _), (tb, bb, ob) in zip(rows[:-1], rows[1:]):
        if 0 < tb - ta <= max_gap:
            for t in frame_times[(frame_times > ta + 1e-6) & (frame_times < tb - 1e-6)]:
                a = (t - ta) / (tb - ta)
                out.append((float(t), ba + a * (bb - ba), False))
        out.append((tb, bb, ob))
    return out


def _merge_static_times(obj_times: list, tracklets: dict) -> list:
    """Времена наблюдений по группам статичных предметов (мерцание одного трека не важно)."""
    out = []
    for g, tls in tracklets.items():
        if g == "person" or not tls:
            continue
        ts = np.sort(np.concatenate([np.asarray(t.times, float) for t in tls]))
        out.append(ts)
    return out or obj_times


def track(clip: Any, dets: Sequence[Any]) -> tuple[list[Track], list[Zone]]:
    """Публичная функция контракта: треки (люди + предметы) с ролями и авто-зоны."""
    r = track_full(clip, dets)
    return r.tracks, r.zones


def track_full(clip: Any, dets: Sequence[Any], cfg: Optional[dict] = None,
               assign_ids: bool = True) -> TrackResult:
    t_begin = time.monotonic()
    cfg = cfg or load_config()
    tc, sc = cfg["tracker"], cfg["stitch"]
    info: dict[str, Any] = {"version": "a06-1", "warnings": [], "timings": {}}
    aspect, times, duration = _clip_meta(clip, cfg)
    nd, bad = _normalize_dets(dets)
    if bad:
        info["warnings"].append(f"skipped_invalid_detections:{bad}")
    if not nd:
        info["warnings"].append("no_detections")
        # пустые детекции != «людей нет»: покрытие сообщаем, решение — за потребителем
        info["coverage"] = {"frames_total": int(len(times)) if times is not None else None, "frames_processed": 0,
                            "frames_with_detections": 0, "span_frac": 0.0}
        info["timings"]["total_s"] = round(time.monotonic() - t_begin, 4)
        return TrackResult([], [], info)

    frame_times, by_frame = _build_frames(times, nd)
    ft = np.asarray(frame_times, float)
    frame_dt = float(np.median(np.diff(ft))) if len(ft) > 1 else 0.2
    t0 = float(times[0]) if times is not None and len(times) else float(ft[0])
    t1 = float(times[-1]) if times is not None and len(times) else float(ft[-1])
    t1 = max(t1, t0)
    # Нормировки (доли присутствия, «статичный предмет», conf зон) — по реально сэмплированному отрезку
    # clip.times, а не по объявленной длительности: в окне refine (WindowClip A03) duration = весь ролик,
    # а times — только окно, и зоны получали заниженный conf.
    declared = duration
    if times is not None and len(times) > 1:
        duration = max(t1 - t0 + float(np.median(np.diff(times))), frame_dt)
    elif duration is None or duration <= 0:
        duration = max(t1 - t0 + frame_dt, frame_dt)
    info.update(aspect=round(aspect, 4), frame_dt=round(frame_dt, 4), n_frames_processed=len(ft),
                n_detections=len(nd))
    n_total = int(len(times)) if times is not None else None
    info["coverage"] = {
        "frames_total": n_total,                       # кадров сэмплировано (clip.times)
        "frames_processed": len(ft),                   # кадров, прошедших через детектор (с учётом every_n)
        "frames_with_detections": int(sum(1 for v in by_frame.values() if v)),
        "t_first_detection": round(float(ft[0]), 4), "t_last_detection": round(float(ft[-1]), 4),
        "t_window": [round(t0, 4), round(t1, 4)],      # отрезок clip.times: весь ролик или окно refine
        "declared_duration": None if declared is None else round(float(declared), 4),
        "span_frac": None if n_total is None or duration <= 0 else round(min(1.0, (ft[-1] - ft[0] + frame_dt) / duration), 4),
    }

    # 1) ByteTrack по группам классов
    ids = itertools.count(1)
    trackers: dict[str, ByteTracker] = {}
    for k, t in enumerate(frame_times):
        groups: dict[str, list[_Det]] = defaultdict(list)
        for d in by_frame.get(k, []):
            groups[_group_of(d.label, cfg)].append(d)
        for g in set(groups) | set(trackers):
            if g not in trackers:
                trackers[g] = ByteTracker(tc, aspect, g, ids)
            trackers[g].update(t, groups.get(g, []))
    tracklets: dict[str, list[Tracklet]] = {g: tr.finish() for g, tr in trackers.items()}
    info["timings"]["bytetrack_s"] = round(time.monotonic() - t_begin, 4)

    # 1b) Потеря видимости: пропали ВСЕ статичные предметы (без яркости)
    obj_times = []
    for g, tls in tracklets.items():
        if g == "person":
            continue
        for tl in tls:
            obj_times.append(np.asarray(tl.times, float))
    vloss, vstatus = loss_intervals(_merge_static_times(obj_times, tracklets), ft, 1.5 * frame_dt,
                                    cfg["visibility"]["min_loss_s"])
    info["coverage"]["visibility_loss"] = [list(x) for x in vloss]
    info["coverage"]["visibility_check"] = vstatus
    if vloss:
        info["warnings"].append(f"visibility_loss:{len(vloss)}")

    # 2) Внешность людей (одежда) и склейка
    persons = tracklets.pop("person", [])
    t_app = time.monotonic()
    app, app_info = tracklet_appearance(clip, persons, cfg["appearance"], times)
    info["appearance"] = app_info
    info["timings"]["appearance_s"] = round(time.monotonic() - t_app, 4)
    bridges = vloss if cfg["visibility"]["bridge_tracks"] else []
    chains, stitch_log = stitch_persons(persons, app, sc, aspect, bridges=bridges, vcfg=cfg["visibility"])
    info["stitch"] = stitch_log
    person_tl = []
    for ch in chains:
        m = merge_chain(ch)
        hs = [app[c.tid] for c in ch if c.tid in app]
        if hs:
            mm = np.mean(np.stack(hs), axis=0)
            app[m.tid] = mm / (mm.sum() + 1e-9)
        person_tl.append(m)
    obj_tl = []
    for g, tls in tracklets.items():
        for ch in stitch_static(tls, sc):
            obj_tl.append(merge_chain(ch))

    # 3) Финальные ID: люди 1..N по времени появления, затем предметы
    person_tl.sort(key=lambda x: (x.t_start, x.tid))
    obj_tl.sort(key=lambda x: (x.group, x.t_start, x.tid))
    final: list[tuple[int, Tracklet, str]] = []
    old2new: dict[int, int] = {}
    for i, tl in enumerate(person_tl + obj_tl, start=1):
        old2new[tl.tid] = i
        final.append((i, tl, _resolve_label(tl, cfg)))
    app = {old2new[k]: v for k, v in app.items() if k in old2new}
    for e in stitch_log:
        e["track_id"] = old2new.get(e["from"])

    anchor = cfg["anchor"]
    pviews: list[TrackView] = []
    oviews: list[TrackView] = []
    for nid, tl, label in final:
        v = make_view(nid, label, tl.times, tl.boxes, tl.scores, anchor, aspect, frame_dt)
        (pviews if tl.group == "person" else oviews).append(v)

    # 4) Независимые зоны (из предметов) -> кассир -> производные зоны -> роли
    t_z = time.monotonic()
    zinfo: dict[str, dict] = {}
    desk, zinfo["desk"] = zones_mod.desk_zone(oviews, pviews, cfg, duration)
    safe, zinfo["safe"] = zones_mod.safe_zone(oviews, cfg, duration)
    counters = [zones_mod.median_box(v.boxes) for v in oviews if v.label == "money_counter"
                and v.presence >= cfg["zones"]["min_object_presence_s"]]
    cashier = roles_mod.pick_cashier(pviews, desk, safe, cfg, aspect, t0, t1, duration, counters)
    cviews = [v for v in pviews if v.tid == cashier.get("primary")]
    others = [v for v in pviews if v.tid != cashier.get("primary")]
    if desk is None:
        desk, zinfo["desk"] = zones_mod.desk_fallback(cviews, others, cfg)
    desk_ind = desk if zinfo["desk"].get("independent") else None
    seat = np.concatenate([roles_mod.seated_mask(v, desk_ind, cfg, aspect) for v in cviews]) if cviews else None
    ca, zinfo["cashier_area"] = zones_mod.cashier_area_zone(
        cviews, desk_ind, cfg, aspect, float(cashier.get("conf", 0.0)), bool(cashier.get("independent")), seat)
    cl, zinfo["client_area"] = zones_mod.client_area_zone(others, desk, ca, cfg, aspect,
                                                          bool(zinfo["desk"].get("independent")))
    seg_gap = max(tc["track_buffer_s"], 3.0 * frame_dt)
    door, zinfo["door"] = zones_mod.door_zone(oviews, pviews, cfg, aspect, t0, t1, seg_gap, duration)
    tray, zinfo["cash_tray"] = zones_mod.cash_tray_zone(oviews, cfg, aspect)
    zones = [z for z in (ca, desk, safe, door, cl, tray) if z is not None]
    zd = {z.name: z for z in zones}
    role_info = roles_mod.assign_roles(pviews, oviews, zd, cashier, app, cfg, aspect, t0, t1)
    info["timings"]["zones_roles_s"] = round(time.monotonic() - t_z, 4)
    info["zones"] = zinfo
    info["cashier_selection"] = {k: v for k, v in cashier.items() if k not in ("features", "side")}
    info["roles"] = {}
    feats = cashier.get("features", {})
    for tid, r in role_info.items():
        r = dict(r)
        r["features"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in feats.get(tid, {}).items()}
        info["roles"][tid] = r

    # 5) Контрактные Track + проставление track_id в исходные детекции
    out_tracks: list[Track] = []
    fill = float(tc.get("fill_gaps_s", 0.0) or 0.0)
    for nid, tl, label in final:
        rows = _fill_gaps(tl, ft, fill)
        boxes = [(round(float(t), 4), tuple(round(float(x), 5) for x in b)) for t, b, _ in rows]
        if tl.group == "person":
            r = role_info.get(nid, {"role": "unknown", "conf": 0.0})
            role, conf = r["role"], float(r["conf"])
        else:
            role, conf = "unknown", 0.0
        out_tracks.append(Track(track_id=nid, label=label, role=role, role_conf=round(conf, 3), boxes=boxes))
        if assign_ids:
            for ref in tl.refs:
                try:
                    if isinstance(ref, dict):
                        ref["track_id"] = nid
                    else:
                        setattr(ref, "track_id", nid)
                except Exception:
                    pass
    info["n_tracks"] = {"person": len(pviews), "object": len(oviews)}
    info["timings"]["total_s"] = round(time.monotonic() - t_begin, 4)
    return TrackResult(out_tracks, zones, info)
