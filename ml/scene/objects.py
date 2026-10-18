"""Покадровые свидетельства по объектам: детекции сейфа/двери, проходы через дверь, наличные, сумка.

Принцип: отсутствие детекции НЕ является свидетельством отсутствия объекта (conf = 0), кроме
случаев, когда объект хорошо виден (ROI не перекрыт, кадр качественный) и несколько независимых
признаков молчат одновременно — тогда «нет» с умеренной уверенностью.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from . import roi as R
from .geometry import BBox, bbox_area, bbox_center, bbox_intersection, expand_bbox, median_bbox, point_in_polygon, polygon_bbox
from .presence import member
from .smoothing import logit
from .timeline import Timeline

Polys = list[list[tuple[float, float]]]


def object_roi(zones: dict[str, Polys], tl: Timeline, zone_name: str, labels: tuple[str, ...], min_score: float = 0.3) -> tuple[Optional[BBox], str]:
    """ROI объекта: из зоны трекера (A06), иначе — медианный бокс уверенных детекций."""
    polys = zones.get(zone_name) or []
    boxes = [polygon_bbox(p) for p in polys]
    boxes = [b for b in boxes if b is not None]
    if boxes:
        return max(boxes, key=bbox_area), "zone"
    dets = [tuple(d.bbox) for ds in tl.dets for d in ds if d.label in labels and float(d.score) >= min_score]
    if len(dets) >= 3:
        return median_bbox(dets), "det_roi"
    return None, "none"


def _overlaps(b: BBox, roi_box: Optional[BBox]) -> bool:
    if roi_box is None:
        return True
    big = expand_bbox(roi_box, 0.25)
    cx, cy = bbox_center(b)
    return (big[0] <= cx <= big[2] and big[1] <= cy <= big[3]) or bbox_intersection(b, roi_box) > 0.3 * max(bbox_area(b), 1e-9)


def det_evidence(tl: Timeline, roi_box: Optional[BBox], open_label: str, closed_label: str, cfg: dict[str, Any], quality: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Лог-шансы «открыто» по детекциям; без детекций — conf 0 (неизвестно, а не «закрыто»)."""
    n = len(tl)
    lo = np.zeros(n)
    c = np.zeros(n)
    ms = float(cfg["min_det_score"])
    for i in range(n):
        so = max((float(d.score) for d in tl.dets[i] if d.label == open_label and float(d.score) >= ms and _overlaps(tuple(d.bbox), roi_box)), default=0.0)
        sc = max((float(d.score) for d in tl.dets[i] if d.label == closed_label and float(d.score) >= ms and _overlaps(tuple(d.bbox), roi_box)), default=0.0)
        if so <= 0 and sc <= 0:
            continue
        if so >= sc:
            lo[i] = float(logit(np.clip(0.5 + 0.5 * so, 0.05, 0.97)))
            c[i] = min(1.0, so * 1.25) * max(float(quality[i]), 0.3)
        else:
            lo[i] = float(logit(np.clip(0.5 - 0.5 * sc, 0.05, 0.95)))
            c[i] = float(cfg["det_closed_c"]) * min(1.0, sc * 1.25) * max(float(quality[i]), 0.3)
    return lo, c


def door_traffic(tl: Timeline, zones: dict[str, Polys], cfg_door: dict[str, Any], cfg_presence: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Свидетельства «дверь открыта» от людей: в зоне двери и пересечение границы cashier_area у двери."""
    n = len(tl)
    p = np.full(n, 0.5)
    c = np.zeros(n)
    door = zones.get("door")
    if not door:
        return p, c, []
    ay, of = float(cfg_presence["anchor_y"]), float(cfg_presence["overlap_full"])
    cash = zones.get("cashier_area")
    tracks: dict[int, list[tuple[int, BBox]]] = {}
    for i in range(n):
        for o in tl.persons[i]:
            if member(o.bbox, door, ay, of) >= 0.5:
                p[i] = max(p[i], float(cfg_door["traffic_in_zone_p"]))
                c[i] = max(c[i], float(cfg_door["traffic_in_zone_c"]))
            if o.track_id is not None:
                tracks.setdefault(int(o.track_id), []).append((i, o.bbox))
    crossings: list[float] = []
    if cash:
        win = float(cfg_door["traffic_cross_window_s"])
        for obs in tracks.values():
            # смена устойчивого состояния «вне cashier_area (m<0.3)» ↔ «внутри (m≥0.7)»,
            # если между этими наблюдениями трек касался зоны двери
            state: Optional[str] = None
            door_seen = False
            for i, b in obs:
                m, dm = member(b, cash, ay, of), member(b, door, ay, of)
                door_seen = door_seen or dm > 0.0
                s_ = "in" if m >= 0.7 else ("out" if m < 0.3 else None)
                if s_ is None:
                    continue
                if state is not None and s_ != state and door_seen:
                    tc = float(tl.times[i])
                    crossings.append(tc)
                    sel = np.abs(tl.times - tc) <= win
                    p[sel] = np.maximum(p[sel], float(cfg_door["traffic_cross_p"]))
                    c[sel] = np.maximum(c[sel], 1.0)
                state = s_
                door_seen = dm > 0.0
    return p, c, crossings


def cash_det_cue(tl: Timeline, zones: dict[str, Polys], cfg: dict[str, Any], cfg_presence: dict[str, Any]) -> tuple[np.ndarray, list[Optional[BBox]], bool]:
    """Сила детекций наличных по зонам (контракт: cash_visible — наличные на столе).

    Приоритет: стол/лоток (1.0) → клиентская зона (0: купюры в руках клиента — не «на столе») →
    зона кассира (0.8) → прочее (0). Без зон стола/лотка/кассира место не проверить — localized=False,
    и вызывающий код не даёт такому «да» стать известным."""
    n = len(tl)
    zw = cfg["zone_weights"]
    s = np.zeros(n)
    best: list[Optional[BBox]] = [None] * n
    localized = any(zones.get(z) for z in ("desk", "cash_tray", "cashier_area"))
    client = zones.get("client_area")
    for i in range(n):
        for d in tl.dets_of(i, ("cash",), float(cfg["min_det_score"])):
            b = tuple(d.bbox)
            if localized:
                cx, cy = bbox_center(b)
                if any(zones.get(z) and member(b, zones[z], 0.5, 0.5) >= 0.5 for z in ("desk", "cash_tray")):
                    w = max(float(zw[z]) for z in ("desk", "cash_tray") if zones.get(z) and member(b, zones[z], 0.5, 0.5) >= 0.5)
                elif client and any(point_in_polygon(cx, cy, p) for p in client):
                    w = float(zw.get("client_area", 0.0))
                elif zones.get("cashier_area") and member(b, zones["cashier_area"], 0.5, 0.5) >= 0.5:
                    w = float(zw["cashier_area"])
                else:
                    w = float(zw.get("other", 0.0))
            else:
                w = 1.0
            if float(d.score) * w > s[i]:
                s[i] = float(d.score) * w
                best[i] = b
    return s, best, bool(localized)


MASK_LABELS = ("person", "laptop", "cell_phone", "money_counter", "cup", "bottle", "keys", "document", "cash_bag", "bag", "food")


def cash_color_frame(fr: np.ndarray, desk: BBox, boxes: list[BBox], shift: tuple[float, float], cfg: dict[str, Any], pad: float) -> tuple[float, float]:
    """Запасной цвет/форма-классификатор купюр на ROI стола (один кадр): пастельные прямоугольные пятна
    (не белая бумага, не тёмная техника); люди и известные предметы маскируются. → (доля площади, видимость стола)."""
    import cv2

    if fr is None or fr.ndim != 3:
        return 0.0, 0.0
    long_side = int(cfg.get("desk_size", 160))
    crop = R.crop(fr, desk, shift)
    h, w = crop.shape[:2]
    sc = long_side / float(max(h, w))
    wh = (max(8, int(w * sc)), max(8, int(h * sc)))
    crop = cv2.resize(crop, wh, interpolation=cv2.INTER_AREA)
    valid = R.mask_from_boxes(desk, boxes, wh, pad)
    vis = float(valid.mean())
    if vis < 0.2:
        return 0.0, vis
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hch, sch, vch = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    colored = (sch >= int(cfg["color_sat_lo"])) & (sch <= int(cfg["color_sat_hi"])) & (vch >= int(cfg["color_val_lo"]))
    bg = valid > 0
    if (bg & colored).sum() > 0.5 * bg.sum():  # стол сам цветной — доминирующий оттенок исключаем
        hist = np.bincount(hch[bg & colored].ravel(), minlength=180)
        dom = int(np.argmax(hist))
        dist = np.minimum(np.abs(hch - dom), 180 - np.abs(hch - dom))
        colored &= dist >= int(cfg["color_hue_bg_dist"])
    m = (colored & bg).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ncomp, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    tot = float(wh[0] * wh[1])
    acc = 0.0
    for k in range(1, ncomp):
        a = float(stats[k, cv2.CC_STAT_AREA])
        frac = a / tot
        if not (float(cfg["blob_min_frac"]) <= frac <= float(cfg["blob_max_frac"])):
            continue
        ys, xs = np.nonzero(lab == k)
        rect = cv2.minAreaRect(np.stack([xs, ys], axis=1).astype(np.float32))
        rw, rh = rect[1]
        if rw < 2 or rh < 2:
            continue
        if a / (rw * rh) < float(cfg["blob_min_rect"]) or max(rw, rh) / min(rw, rh) > float(cfg["blob_aspect_max"]):
            continue
        acc += frac
    return acc, vis


def cash_color_finalize(per_frame: dict[int, tuple[float, float]], n: int, quality: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Покадровые (площадь, видимость) → p, conf, видимость; пропуски (stride) — ближайшим обработанным."""
    area, vis, seen = np.zeros(n), np.ones(n), np.zeros(n, dtype=bool)
    for i, (a, v) in per_frame.items():
        area[i], vis[i], seen[i] = a, v, True
    if seen.any() and not seen.all():
        si = np.flatnonzero(seen)
        for i in np.flatnonzero(~seen):
            j = si[np.argmin(np.abs(si - i))]
            if abs(j - i) <= 3:
                area[i], vis[i], seen[i] = area[j], vis[j], True
    p = np.where(area > 0, 0.5 + 0.4 * (1.0 - np.exp(-area / float(cfg["area_scale"]))), 0.35)
    c = np.where(seen, np.where(area > 0, float(cfg["color_c"]), 0.25) * vis * quality, 0.0)
    return p, c, vis


def green_frac_frame(fr: np.ndarray, region: Optional[BBox], cfg: dict[str, Any]) -> float:
    """Доля насыщенно-зелёных пикселей (инкассаторская сумка) в области (один кадр, грубая подвыборка)."""
    import cv2

    if fr is None or fr.ndim != 3:
        return 0.0
    img = R.crop(fr, region) if region is not None else fr
    step = max(1, img.shape[1] // 120)
    img = np.ascontiguousarray(img[::step, ::step])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lo_h, hi_h = cfg["green_hue"]
    g = (hsv[..., 0] >= lo_h) & (hsv[..., 0] <= hi_h) & (hsv[..., 1] >= int(cfg["green_sat_lo"])) & (hsv[..., 2] >= int(cfg["green_val_lo"]))
    return float(g.mean())


def green_bag_finalize(per_frame: dict[int, float], n: int, quality: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Слабый запасной признак зелёной сумки: есть блоб → p=0.7 с малой уверенностью; нет — молчим (не «нет сумки»)."""
    p = np.full(n, 0.5)
    c = np.zeros(n)
    for i, frac in per_frame.items():
        if frac >= float(cfg["green_min_frac"]):
            p[i] = 0.7
            c[i] = float(cfg["green_c"]) * float(quality[i])
    return p, c
