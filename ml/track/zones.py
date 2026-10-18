"""Авто-зоны без захардкоженных координат.

Независимые опоры (не зависят от ролей людей):
  desk      — медианные боксы laptop / money_counter / статичных телефона, ключей, документов;
  safe      — медианный бокс safe/safe_open;
  door      — детекции door/door_open + кластер точек появления/исчезновения людей;
  cash_tray — кластер детекций cash (fallback — окрестность money_counter, низкий conf).
Производные (зависят от выбранного кассира, см. roles.pick_cashier):
  cashier_area — где кассир проводит время у стола;
  client_area  — люди по другую сторону стола, иначе зеркало cashier_area через центр стола.
Если опор нет — зона либо отсутствует, либо получает низкий conf (fallback_zone_conf_cap).
Зона «весь кадр по умолчанию» не создаётся никогда.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .contract import Zone
from .geometry import (box_center, box_to_polygon, clip_polygon_unit, convex_hull, dist_point_polygon,
                       ensure_min_size, iso_dist, median_box, point_in_polygon, polygon_area,
                       polygon_centroid, reduce_vertices, reflect_polygon, round_polygon, scale_box,
                       scale_polygon)
from .views import TrackView


def make_zone(name: str, poly: Sequence[Sequence[float]], conf: float, max_v: int = 12) -> Optional[Zone]:
    p = clip_polygon_unit(poly)
    if len(p) < 3:
        return None
    p = reduce_vertices(p, max_v)
    if polygon_area(p) < 1e-6:
        return None
    return Zone(name=name, polygon=round_polygon(p), conf=round(float(min(1.0, max(0.0, conf))), 3))


def person_size(views: Sequence[TrackView]) -> Optional[tuple[float, float]]:
    if not views:
        return None
    b = np.concatenate([v.boxes for v in views])
    return (float(np.median(b[:, 2] - b[:, 0])), float(np.median(b[:, 3] - b[:, 1])))


def is_static(v: TrackView, zc: dict, duration: float) -> bool:
    if duration > 0 and v.presence < zc["static_min_presence_frac"] * duration:
        return False
    med = np.median(v.pts, axis=0)
    d = np.hypot(v.pts[:, 0] - med[0], v.pts[:, 1] - med[1])
    return float(np.median(d)) <= zc["static_max_motion_diag"] * max(v.diag, 1e-4)


def desk_zone(objs: Sequence[TrackView], persons: Sequence[TrackView], cfg: dict,
              duration: float) -> tuple[Optional[Zone], dict]:
    zc = cfg["zones"]
    anchors = []
    for v in objs:
        if v.presence < zc["min_object_presence_s"]:
            continue
        if v.label in zc["desk_labels"] or (v.label in zc["desk_static_labels"] and is_static(v, zc, duration)):
            anchors.append(v)
    if not anchors:
        return None, {"method": "none", "reason": "no_desk_objects"}
    pts: list = []
    for v in anchors:
        pts += box_to_polygon(median_box(v.boxes))
    poly = scale_polygon(convex_hull(pts), zc["desk_expand"])
    ps = person_size(persons)
    if ps:
        poly = ensure_min_size(poly, zc["desk_min_size_person"] * ps[0], zc["desk_min_size_person"] * ps[1])
    labels = sorted({v.label for v in anchors})
    pres = float(np.mean([min(1.0, v.presence / max(duration, 1e-6)) for v in anchors])) if duration > 0 else 1.0
    score = float(np.mean([v.scores.mean() for v in anchors]))
    conf = min(1.0, 0.35 + 0.2 * len(labels) + 0.15 * pres) * min(1.0, max(0.5, score + 0.25))
    z = make_zone("desk", poly, conf, zc["max_vertices"])
    return z, {"method": "objects", "evidence": labels, "tracks": [v.tid for v in anchors], "independent": True}


def safe_zone(objs: Sequence[TrackView], cfg: dict, duration: float) -> tuple[Optional[Zone], dict]:
    zc = cfg["zones"]
    vs = [v for v in objs if v.label == "safe" and v.presence >= zc["min_object_presence_s"]]
    if not vs:
        return None, {"method": "none", "reason": "no_safe_detections"}
    best = max(vs, key=lambda v: v.presence * float(v.scores.mean()))
    mb = median_box(best.boxes)
    pres = min(1.0, best.presence / max(duration, 1e-6)) if duration > 0 else 1.0
    conf = min(1.0, 0.4 + 0.6 * pres) * min(1.0, float(best.scores.mean()) + 0.3)
    z = make_zone("safe", box_to_polygon(scale_box(mb, zc["safe_expand"])), conf, zc["max_vertices"])
    return z, {"method": "objects", "evidence": ["safe"], "tracks": [best.tid], "independent": True}


def _segments(v: TrackView, max_gap: float) -> list[tuple[int, int]]:
    if len(v.times) == 0:
        return []
    cuts = np.where(np.diff(v.times) > max_gap)[0]
    starts = [0] + [int(c) + 1 for c in cuts]
    ends = [int(c) for c in cuts] + [len(v.times) - 1]
    return list(zip(starts, ends))


def boundary_points(persons: Sequence[TrackView], t0: float, t1: float, early_s: float,
                    seg_gap: float) -> list[tuple[float, float, str, int, float]]:
    """Точки появления (birth) / исчезновения (death) людей внутри ролика."""
    out = []
    for v in persons:
        for s, e in _segments(v, seg_gap):
            if v.times[s] > t0 + early_s:
                out.append((float(v.pts[s, 0]), float(v.pts[s, 1]), "birth", v.tid, float(v.times[s])))
            if v.times[e] < t1 - early_s:
                out.append((float(v.pts[e, 0]), float(v.pts[e, 1]), "death", v.tid, float(v.times[e])))
    return out


def door_zone(objs: Sequence[TrackView], persons: Sequence[TrackView], cfg: dict, aspect: float,
              t0: float, t1: float, seg_gap: float, duration: float) -> tuple[Optional[Zone], dict]:
    zc, rc = cfg["zones"], cfg["roles"]
    pts = boundary_points(persons, t0, t1, rc["early_s"], seg_gap)
    ps = person_size(persons)
    pdiag = float(np.median([v.diag for v in persons])) if persons else 0.2
    r = zc["door_cluster_diag"] * pdiag
    cluster: list = []
    if pts:
        arr = np.array([(p[0], p[1]) for p in pts])
        dd = np.hypot((arr[:, None, 0] - arr[None, :, 0]) * aspect, arr[:, None, 1] - arr[None, :, 1])
        nb = (dd <= r).sum(1)
        i = int(np.argmax(nb))
        cluster = [pts[j] for j in np.where(dd[i] <= r)[0]]
    vs = [v for v in objs if v.label == "door" and v.presence >= zc["min_object_presence_s"]]
    info: dict[str, Any] = {"boundary_points": len(pts), "cluster_points": len(cluster)}
    if vs:
        best = max(vs, key=lambda v: v.presence * float(v.scores.mean()))
        mb = median_box(best.boxes)
        pres = min(1.0, best.presence / max(duration, 1e-6)) if duration > 0 else 1.0
        conf = (0.35 + 0.45 * pres) * min(1.0, float(best.scores.mean()) + 0.3)
        poly = box_to_polygon(scale_box(mb, zc["door_expand"]))
        if cluster:
            c = np.mean([(p[0], p[1]) for p in cluster], axis=0)
            if dist_point_polygon(c, poly, aspect) <= r:
                conf += 0.2
                info["agrees_with_entries"] = True
        info.update(method="objects", evidence=["door"], tracks=[best.tid], independent=True)
        return make_zone("door", poly, conf, zc["max_vertices"]), info
    if len(cluster) >= zc["door_min_points"] and ps:
        hw, hh = ps[0] * 0.5, ps[1] * 0.5
        cp = []
        for x, y, *_ in cluster:
            cp += [(x - hw, y - hh), (x + hw, y - hh), (x + hw, y + hh), (x - hw, y + hh)]
        poly = scale_polygon(convex_hull(cp), zc["door_expand"])
        conf = min(0.7, 0.2 + 0.1 * len(cluster))
        info.update(method="entry_exit_cluster", independent=True)
        return make_zone("door", poly, conf, zc["max_vertices"]), info
    info.update(method="none", reason="no_door_detections_and_few_entries")
    return None, info


def cash_tray_zone(objs: Sequence[TrackView], cfg: dict, aspect: float) -> tuple[Optional[Zone], dict]:
    zc = cfg["zones"]
    cash = [v for v in objs if v.label == "cash"]
    if cash:
        boxes = np.concatenate([v.boxes for v in cash])
        cen = np.stack([(boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5], 1)
        diag = float(np.median(np.hypot((boxes[:, 2] - boxes[:, 0]) * aspect, boxes[:, 3] - boxes[:, 1])))
        r = zc["cash_cluster_diag"] * max(diag, 1e-3)
        dd = np.hypot((cen[:, None, 0] - cen[None, :, 0]) * aspect, cen[:, None, 1] - cen[None, :, 1])
        i = int(np.argmax((dd <= r).sum(1)))
        sel = boxes[dd[i] <= r]
        b = (float(np.percentile(sel[:, 0], 10)), float(np.percentile(sel[:, 1], 10)),
             float(np.percentile(sel[:, 2], 90)), float(np.percentile(sel[:, 3], 90)))
        conf = min(1.0, 0.3 + 0.05 * len(sel))
        z = make_zone("cash_tray", box_to_polygon(scale_box(b, zc["cash_expand"])), conf, zc["max_vertices"])
        return z, {"method": "cash_cluster", "evidence": ["cash"], "n": int(len(sel)), "independent": True}
    mc = [v for v in objs if v.label == "money_counter"]
    if mc:
        best = max(mc, key=lambda v: v.presence)
        z = make_zone("cash_tray", box_to_polygon(scale_box(median_box(best.boxes), 1.5)),
                      min(0.2, zc["fallback_zone_conf_cap"]), zc["max_vertices"])
        return z, {"method": "money_counter_fallback", "independent": True}
    return None, {"method": "none", "reason": "no_cash_or_counter"}


def _shrunk_corners(boxes: np.ndarray, s: float) -> list:
    out: list = []
    for b in boxes:
        out += box_to_polygon(scale_box(b, s))
    return out


def cashier_area_zone(cviews: Sequence[TrackView], desk: Optional[Zone], cfg: dict, aspect: float,
                      cashier_conf: float, independent: bool,
                      seated: Optional[np.ndarray] = None) -> tuple[Optional[Zone], dict]:
    zc = cfg["zones"]
    if not cviews:
        return None, {"method": "none", "reason": "no_cashier"}
    pts = np.concatenate([v.pts for v in cviews])
    boxes = np.concatenate([v.boxes for v in cviews])
    w = np.concatenate([v.w for v in cviews])
    diag = float(np.median([v.diag for v in cviews]))
    keep = np.ones(len(pts), bool)
    if seated is not None and int(np.sum(seated)) >= 3:
        keep = np.asarray(seated, bool)  # рабочее место = где кассир СИДИТ у стола, а не где стоял, уйдя
    elif desk is not None:
        dist = np.array([dist_point_polygon(p, desk.polygon, aspect) for p in pts]) / diag
        near = dist <= zc["cashier_max_dist_diag"]
        if near.sum() >= 3:
            keep = near
    med = np.median(pts[keep], axis=0)
    d = np.hypot((pts[:, 0] - med[0]) * aspect, pts[:, 1] - med[1]) / diag
    thr = min(float(np.percentile(d[keep], zc["cashier_keep_pct"])), zc["cashier_max_dist_diag"])
    keep &= d <= max(thr, 0.25)
    if keep.sum() == 0:
        keep = d <= np.min(d) + 1e-9
    hull_pts = _shrunk_corners(boxes[keep], zc["cashier_shrink"])
    band = False
    if desk is not None and zc.get("cashier_band", True):
        # «сторона стола»: отпечаток стола, сдвинутый туда, где сидит кассир (аффинно-эквивариантно)
        dc = polygon_centroid(desk.polygon)
        cc = np.median(pts[keep], axis=0)
        sh = (float(cc[0] - dc[0]), float(cc[1] - dc[1]))
        hull_pts += [(x + sh[0], y + sh[1]) for x, y in desk.polygon]
        band = True
    poly = scale_polygon(convex_hull(hull_pts), zc["cashier_expand"])
    conf = float(cashier_conf) * min(1.0, float(w[keep].sum()) / 5.0)
    if not independent:
        conf = min(conf, zc["fallback_zone_conf_cap"])
    z = make_zone("cashier_area", poly, conf, zc["max_vertices"])
    return z, {"method": "cashier_dwell+desk_band" if band else "cashier_dwell", "tracks": [v.tid for v in cviews], "kept_obs": int(keep.sum()),
               "independent": False, "cashier_support": "desk_objects" if independent else "fallback"}


def client_area_zone(others: Sequence[TrackView], desk: Optional[Zone], cashier_area: Optional[Zone],
                     cfg: dict, aspect: float, desk_independent: bool = True) -> tuple[Optional[Zone], dict]:
    zc = cfg["zones"]
    cap = zc["fallback_zone_conf_cap"]
    if desk is not None and cashier_area is not None:
        dc = polygon_centroid(desk.polygon)
        cc = polygon_centroid(cashier_area.polygon)
        u = np.array([(cc[0] - dc[0]) * aspect, cc[1] - dc[1]])
        nu = float(np.linalg.norm(u))
        if others and nu > 1e-6:
            u /= nu
            sel_boxes = []
            for v in others:
                for p, b in zip(v.pts, v.boxes):
                    proj = (p[0] - dc[0]) * aspect * u[0] + (p[1] - dc[1]) * u[1]
                    if proj < -0.1 * v.diag and dist_point_polygon(p, desk.polygon, aspect) <= zc["client_max_dist_diag"] * v.diag:
                        if not point_in_polygon(p, cashier_area.polygon):
                            sel_boxes.append(b)
            if len(sel_boxes) >= zc["client_min_points"]:
                poly = scale_polygon(convex_hull(_shrunk_corners(np.array(sel_boxes), zc["cashier_shrink"])),
                                     zc["client_expand"])
                conf = min(0.85, 0.4 + 0.02 * len(sel_boxes)) * max(desk.conf, 0.5)
                if not desk_independent:
                    conf = min(conf, cap)
                return make_zone("client_area", poly, conf, zc["max_vertices"]), {
                    "method": "visitors_opposite_side", "n_obs": len(sel_boxes), "independent": False}
        poly = reflect_polygon(cashier_area.polygon, dc)
        return make_zone("client_area", poly, min(cap, desk.conf * cashier_area.conf), zc["max_vertices"]), {
            "method": "mirror_of_cashier_area", "independent": False}
    return None, {"method": "none", "reason": "no_desk_or_cashier_area"}


def desk_fallback(cviews: Sequence[TrackView], others: Sequence[TrackView], cfg: dict) -> tuple[Optional[Zone], dict]:
    """Без объектов стола: между кассиром и остальными людьми, низкий conf."""
    zc = cfg["zones"]
    if not cviews:
        return None, {"method": "none", "reason": "no_desk_objects_no_cashier"}
    cb = np.concatenate([v.boxes for v in cviews])
    mb = median_box(cb)
    c = box_center(mb)
    if others:
        op = np.concatenate([v.pts for v in others])
        o = np.median(op, axis=0)
        mid = (c[0] + 0.5 * (o[0] - c[0]), c[1] + 0.5 * (o[1] - c[1]))
        hw, hh = (mb[2] - mb[0]) * 0.5, (mb[3] - mb[1]) * 0.5
        poly = box_to_polygon((mid[0] - hw, mid[1] - hh, mid[0] + hw, mid[1] + hh))
        method = "between_cashier_and_others"
    else:
        poly = box_to_polygon(scale_box(mb, 1.3))
        method = "around_cashier"
    return make_zone("desk", poly, min(0.2, zc["fallback_zone_conf_cap"]), zc["max_vertices"]), {
        "method": method, "independent": False}


def zone_dict(zones: Sequence[Zone]) -> dict[str, Zone]:
    return {z.name: z for z in zones}


def iso_point_dist(p: Sequence[float], q: Sequence[float], aspect: float) -> float:
    return iso_dist(p, q, aspect)
