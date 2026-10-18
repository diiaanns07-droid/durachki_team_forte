"""Проверяемые предикаты для A09 (фаза 2): трёхзначные (true/false/unknown/occluded), с интервалами.

Аддитивный слой поверх SIGNALS: legacy-сигналы фазы 1 не меняются, а предикаты реализуют
спецификацию поведения команды:

* клиент напротив кассира (опорная точка в client_area) — норма, «посторонние» предикаты не растут;
* уход кассира — норма, нарушение-кандидат только если в окне [t_ухода, t_ухода+5 с] видно, что на столе
  остались наличные / инкассаторская сумка или сейф открыт; в темноте берётся последнее известное
  состояние (заморозка), без него — unknown(dark);
* работа при открытом сейфе, вход при открытых ценностях, мобильный (не стационарный) телефон в руке,
  еда/питьё только при видимом предмете, личные вещи на столе, сумка без присмотра;
* роль — по поведению и зоне за весь ролик (кто работает со стороны кассы), а не по порядку появления.

Все пороги — в config.yaml → predicates. Ни один предикат не берёт unknown как 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from . import tri
from .geometry import BBox, anchor_point, bbox_area, bbox_center, bbox_fraction_in_polygon, bbox_gap, bbox_intersection, expand_bbox, point_in_polygon
from .geometry import polygon_bbox as bbox_from_poly
from .tri import F, O, T, U

Polys = list[list[tuple[float, float]]]

PREDICATES = [
    "lights_off",
    "cashier_left_workplace",
    "cash_left_after_leave",
    "bag_left_after_leave",
    "safe_open_after_leave",
    "personal_item_on_desk",
    "outsider_cashier_side",
    "second_person_client_side",
    "entry_while_valuables_open",
    "safe_open_while_working",
    "mobile_phone_in_hand",
    "eating_drinking_with_object",
    "eating_at_workplace_with_valuables",
    "bag_left_unattended",
]
# предикат → кандидат типа из norms/catalog.yaml (решает A09/A12; здесь — подсказка)
TYPE_HINT = {
    "cash_left_after_leave": "V02",
    "bag_left_after_leave": "V07",
    "safe_open_after_leave": "V03",
    "outsider_cashier_side": "V01",
    "second_person_client_side": "V01",
    "entry_while_valuables_open": "V04",
    "safe_open_while_working": "V03",
    "mobile_phone_in_hand": "V05",
    "eating_at_workplace_with_valuables": "V06",
    "personal_item_on_desk": "V09",   # по спецификации команды; если ключа нет в norms/catalog.yaml — A12 сводит к V99
    "bag_left_unattended": "V07",
}


@dataclass
class PredContext:
    times: np.ndarray
    tl: Any                                   # Timeline (persons, dets)
    zones: dict[str, Polys]
    workplace: Optional[Polys]
    sig_tri: dict[str, np.ndarray]            # tri-состояния SIGNALS после сглаживания
    sig_vis: dict[str, np.ndarray]
    quality: np.ndarray
    brightness: Optional[np.ndarray]          # медиана яркости кадра (nan — кадра нет)
    det_ran: np.ndarray
    pres_extra: dict[str, np.ndarray]
    safe_roi: Optional[BBox]
    desk_box: Optional[BBox]
    cfg: dict[str, Any]
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PredResult:
    tri: dict[str, np.ndarray]
    why: dict[str, list[str]]
    events: list[dict[str, Any]]


def _in(polys: Optional[Polys], x: float, y: float) -> bool:
    return bool(polys) and any(point_in_polygon(x, y, p) for p in polys)


def _why_fill(x: np.ndarray, reason: list[str], default: str) -> list[str]:
    return [("" if v in (T, F) else (reason[i] or default)) for i, v in enumerate(x)]


# ----------------------------------------------------------------------------------------- сцена
def lights_off(ctx: PredContext) -> tuple[np.ndarray, list[str]]:
    """Свет выключен: медианная яркость кадра << обычной для ролика (и абсолютно низкая). Гистерезис по времени."""
    c = ctx.cfg["lights"]
    n = len(ctx.times)
    b = ctx.brightness
    if b is None or not np.isfinite(b).any():
        return np.full(n, U, np.int8), ["frames_unavailable"] * n
    ok = np.isfinite(b)
    ref = float(np.nanpercentile(b, 75))
    thr = max(float(c["abs_dark"]), float(c["rel_dark"]) * ref)
    raw = np.where(ok, np.where(b < thr, T, F), U).astype(np.int8)
    if float(np.nanmax(b)) < float(c["abs_dark"]):  # весь ролик тёмный — опоры нет, это не «выключили свет»
        raw = np.where(ok, U, U).astype(np.int8)
    out = tri.sustain(raw, ctx.times, float(c["min_s"]), float(c["max_gap_s"]))
    # короткие «тёмные» участки — не выключение света, а шум/перекрытие камеры: считаем «светло»
    out = np.where((raw == T) & (out == U), F, out).astype(np.int8)
    return out, ["" if v in (T, F) else "frames_unavailable" for v in out]


# ----------------------------------------------------------------------------------------- люди
@dataclass
class _TrackRole:
    staff: bool = False
    non_staff: bool = False
    collector: bool = False
    dominant: bool = False


def _sides(ctx: PredContext) -> list[list[tuple[Any, str]]]:
    """Сторона каждого человека по опорной точке: client > desk > cashier > door > other."""
    pc = ctx.cfg["presence"]
    ay = float(pc["anchor_y"])
    z = ctx.zones
    near_safe = expand_bbox(ctx.safe_roi, float(ctx.cfg["predicates"]["near_safe_pad"])) if ctx.safe_roi else None
    out: list[list[tuple[Any, str]]] = []
    naive = bool(ctx.cfg.get("ablation", {}).get("naive"))
    for ps in ctx.tl.persons:
        row = []
        for o in ps:
            x, y = anchor_point(o.bbox, ay)
            if naive and ctx.workplace and any(bbox_fraction_in_polygon(o.bbox, p) >= 0.3 for p in ctx.workplace):
                side = "cashier"  # АБЛЯЦИЯ: сторона по перекрытию бокса, а не по опорной точке
            elif _in(z.get("client_area"), x, y):
                side = "client"
            elif _in(z.get("desk"), x, y):
                side = "desk"
            elif _in(ctx.workplace, x, y) or (near_safe is not None and near_safe[0] <= x <= near_safe[2] and near_safe[1] <= y <= near_safe[3]):
                side = "cashier"
            elif _in(z.get("door"), x, y):
                side = "door"
            else:
                side = "other"
            row.append((o, side))
        out.append(row)
    return out


def _roles(ctx: PredContext, sides: list[list[tuple[Any, str]]]) -> dict[int, _TrackRole]:
    """Роль трека по поведению за весь ролик: доминирующий «работник со стороны кассы» — сотрудник;
    пришедший со стороны клиента — посторонний (если не доминирующий); явные роли A06 — приоритетнее."""
    pr = ctx.cfg["predicates"]
    dt = float(np.median(np.diff(ctx.times))) if len(ctx.times) > 1 else 0.2
    side_time: dict[int, float] = {}
    first_cashier: dict[int, int] = {}
    client_before: dict[int, float] = {}
    info: dict[int, Any] = {}
    for i, row in enumerate(sides):
        for o, side in row:
            if o.track_id is None:
                continue
            tid = int(o.track_id)
            info.setdefault(tid, o)
            if side == "cashier":
                side_time[tid] = side_time.get(tid, 0.0) + dt
                first_cashier.setdefault(tid, i)
            elif side == "client" and tid not in first_cashier:
                client_before[tid] = client_before.get(tid, 0.0) + dt
    dom = max(side_time, key=lambda k: side_time[k]) if side_time else None
    if dom is not None and side_time[dom] < float(pr["dominant_min_s"]):
        dom = None
    roles: dict[int, _TrackRole] = {}
    for tid, o in info.items():
        r = _TrackRole(dominant=tid == dom)
        role_vis = max(0.0, 1.0 - o.p_cashier - o.p_collector)
        if o.explicit_role:
            r.collector = o.p_collector >= 0.5
            r.staff = o.p_cashier >= 0.5 or r.collector
            r.non_staff = role_vis >= 0.5
        else:
            r.collector = o.p_collector >= 0.5
            came_from_client = client_before.get(tid, 0.0) >= float(pr["client_history_s"])
            # без явной роли сотрудник — только доминирующий «работник со стороны кассы» или инкассатор;
            # второй человек без истории в client_area — неизвестная роль (не «посторонний» и не «сотрудник»)
            r.staff = r.dominant or r.collector
            r.non_staff = came_from_client and not r.staff
            if ctx.cfg.get("ablation", {}).get("naive"):
                r.non_staff = not r.staff  # АБЛЯЦИЯ: «не кассир» = посторонний
        roles[tid] = r
    return roles


def outsider_preds(ctx: PredContext, sides: list[list[tuple[Any, str]]], roles: dict[int, _TrackRole], lights: np.ndarray) -> dict[str, tuple[np.ndarray, list[str]]]:
    pr = ctx.cfg["predicates"]
    n = len(ctx.times)
    q_ok = (ctx.quality >= float(pr["min_quality"])) & ctx.det_ran & (lights != T)
    out = np.full(n, U, np.int8)
    why = [""] * n
    second = np.full(n, U, np.int8)
    for i, row in enumerate(sides):
        cash_side = [(o, roles.get(int(o.track_id)) if o.track_id is not None else None) for o, s in row if s == "cashier"]
        if any(r is not None and r.non_staff and not r.collector for _, r in cash_side):
            out[i] = T
        elif not ctx.workplace:
            why[i] = "no_zone"
        elif not q_ok[i]:
            why[i] = "dark" if lights[i] == T else ("no_detector" if not ctx.det_ran[i] else "low_quality")
        elif all(r is not None and (r.staff or r.collector) for _, r in cash_side):
            out[i] = F  # на стороне кассы только сотрудники или никого; клиент напротив не учитывается
        else:
            why[i] = "role_unknown"
        n_client = sum(1 for _, s in row if s == "client")
        if not q_ok[i] or not ctx.zones.get("client_area"):
            continue
        second[i] = T if n_client >= 2 else F
    out = tri.sustain(out, ctx.times, float(pr["outsider_min_s"]), float(pr["max_gap_s"]))
    service = ctx.sig_tri.get("cashier_in_workplace", np.full(n, U, np.int8))
    second = tri.sustain(tri.t_and(second, service), ctx.times, float(pr["second_person_min_s"]), float(pr["max_gap_s"]))
    sec_why = _why_fill(second, ["no_zone" if not ctx.zones.get("client_area") else ""] * n, "no_evidence")
    return {"outsider_cashier_side": (out, _why_fill(out, why, "no_evidence")), "second_person_client_side": (second, sec_why)}


def entry_events(ctx: PredContext, sides: list[list[tuple[Any, str]]], roles: dict[int, _TrackRole]) -> list[dict[str, Any]]:
    """Вход на сторону кассы: ≥ before_s вне её (клиент/дверь/прочее), затем ≥ after_s на ней.
    Доминирующий кассир и инкассатор входом постороннего не считаются."""
    pr = ctx.cfg["predicates"]
    seq: dict[int, list[tuple[int, str]]] = {}
    for i, row in enumerate(sides):
        for o, s in row:
            if o.track_id is not None:
                seq.setdefault(int(o.track_id), []).append((i, s))
    t = ctx.times
    evs = []
    for tid, obs in seq.items():
        r = roles.get(tid, _TrackRole())
        if r.dominant or r.collector or (r.staff and not r.non_staff):
            continue
        for k in range(1, len(obs)):
            i, s = obs[k]
            if s != "cashier" or obs[k - 1][1] == "cashier":
                continue
            before = [j for j, sj in obs[:k] if sj in ("client", "door", "other") and t[i] - t[j] <= float(pr["entry_before_s"]) + 1e-9]
            after = [j for j, sj in obs[k:] if sj == "cashier" and t[j] - t[i] <= float(pr["entry_after_s"]) + 1e-9]
            span_after = t[after[-1]] - t[i] if after else 0.0
            if before and span_after >= float(pr["entry_after_s"]) - 0.21:
                evs.append({"type": "entry", "track_id": tid, "t": float(t[i]), "i": int(i), "non_staff": bool(r.non_staff)})
                break
    return evs


def entry_pred(ctx: PredContext, evs: list[dict[str, Any]], lights: np.ndarray, cash_desk: Optional[np.ndarray] = None) -> tuple[np.ndarray, list[str]]:
    pr = ctx.cfg["predicates"]
    n = len(ctx.times)
    ok = ctx.det_ran & (lights != T) & (ctx.quality >= float(pr["min_quality"]))
    out = np.where(ok, F, U).astype(np.int8)
    why = ["" if ok[i] else ("dark" if lights[i] == T else "no_detector") for i in range(n)]
    vo = tri.t_or(ctx.sig_tri["safe_open"], cash_desk if cash_desk is not None else ctx.sig_tri["cash_visible"])
    win = float(pr["entry_mark_s"])
    for e in evs:
        i = e["i"]
        lo, hi = max(0, i - 2), min(n, i + 3)
        st = T if (vo[lo:hi] == T).any() else (F if (vo[lo:hi] == F).all() else U)
        e["valuables_open"] = tri.NAMES[int(st)]
        sel = (ctx.times >= ctx.times[i]) & (ctx.times <= ctx.times[i] + win)
        if st == T and e["non_staff"]:
            out[sel] = T
        elif st != F:
            # ценности открыты, но роль вошедшего неизвестна (мог вернуться сотрудник) — unknown, не true
            out[sel] = np.where(out[sel] == T, T, U)
            for j in np.flatnonzero(sel):
                why[j] = "role_unknown" if st == T else "valuables_unknown"
    return out, _why_fill(out, why, "no_evidence")


# ----------------------------------------------------------------------------------------- уход
def leave_events(ctx: PredContext) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """cashier_left_workplace (tri) и события ухода: переход «кассир на месте (виден)» → «подтверждённо нет»."""
    pr = ctx.cfg["predicates"]
    n = len(ctx.times)
    occ = ctx.pres_extra.get("cashier_in_obs", np.zeros(n)) > 0
    gone = ctx.pres_extra.get("cashier_gone_known", np.zeros(n)) > 0
    st = np.where(occ, F, np.where(gone, T, U)).astype(np.int8)
    st = tri.sustain(st, ctx.times, float(pr["leave_min_s"]), float(pr["max_gap_s"]))
    evs: list[dict[str, Any]] = []
    for i0, i1 in tri.runs(st == T):
        prev_occ = np.flatnonzero(occ[:i0])
        if len(prev_occ) == 0:
            continue  # ухода не видели (место пустое с начала) — это не «уход»
        j = int(prev_occ[-1])
        onset = j + 1 if j + 1 < n else j
        ret = np.flatnonzero(occ[i1 + 1 :])
        t_ret = float(ctx.times[i1 + 1 + ret[0]]) if len(ret) else None
        i_desk = _left_desk_index(ctx, onset)
        ev = {"type": "leave", "t": float(ctx.times[onset]), "i": onset, "t_confirmed": float(ctx.times[i0]), "i_end": int(i1),
              "t_return": t_ret, "t_left_desk": float(ctx.times[i_desk]), "onset_policy": str(pr.get("leave_onset", "exit"))}
        if pr.get("leave_onset", "exit") == "left_desk" and i_desk < onset:
            ev["i"] = i_desk  # [H5] таймкод — момент, когда отошла от стола, а не когда вышла из зоны кассы
        evs.append(ev)
        st[ev["i"]:i0] = np.where(st[ev["i"]:i0] == F, F, T)
    return st, evs


def _left_desk_index(ctx: PredContext, onset: int) -> int:
    """Первый сэмпл после того, как «кассироподобный» человек последний раз был у стола (до выхода из зоны)."""
    desk = ctx.zones.get("desk")
    if not desk:
        return onset
    pad = float(ctx.cfg["predicates"].get("desk_near_pad", 0.12))
    ay = float(ctx.cfg["presence"]["anchor_y"])
    boxes = [b for b in (bbox_from_poly(p) for p in desk) if b is not None]
    near = [expand_bbox(b, 0.0, pad) for b in boxes]
    last = None
    for i in range(onset - 1, -1, -1):
        for o in ctx.tl.persons[i]:
            if o.p_cashier < 0.4:
                continue
            x, y = anchor_point(o.bbox, ay)
            if any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in near):
                last = i
                break
        if last is not None:
            break
    return onset if last is None else min(onset, last + 1)


def _bag_on_desk(ctx: PredContext) -> np.ndarray:
    """Сумка лежит на столе/в зоне кассы (детекция cash_bag, никто не держит): tri; нет детекции — unknown."""
    pr = ctx.cfg["predicates"]
    n = len(ctx.times)
    out = np.full(n, U, np.int8)
    zones = [p for zn in ("desk", "cash_tray") for p in (ctx.zones.get(zn) or [])] or (ctx.workplace or [])
    for i in range(n):
        bags = ctx.tl.dets_of(i, ("cash_bag",), float(pr["bag_min_score"]))
        if not bags:
            if ctx.sig_tri["cash_bag_visible"][i] == F:
                out[i] = F
            continue
        on = False
        for d in bags:
            cx, cy = bbox_center(tuple(d.bbox))
            held = any(bbox_gap(tuple(d.bbox), o.bbox) <= float(pr["reach"]) for o in ctx.tl.persons[i])
            if (not zones or any(point_in_polygon(cx, cy, p) for p in zones)) and not held:
                on = True
        out[i] = T if on else F
    return out


def _cash_on_desk(ctx: PredContext) -> np.ndarray:
    """Наличные именно НА СТОЛЕ (desk/cash_tray): детекция cash с центром в зоне стола (±1 сэмпл) — true;
    legacy cash_visible = false (стол виден, детектор работал, цветовой признак молчит) — false; иначе unknown/occluded.
    Купюры в руках клиента у стойки сюда не попадают. Без зоны стола — legacy cash_visible."""
    pr = ctx.cfg["predicates"]
    n = len(ctx.times)
    zones = [p for zn in ("desk", "cash_tray") for p in (ctx.zones.get(zn) or [])]
    legacy = ctx.sig_tri["cash_visible"]
    if not zones:
        return legacy.copy()
    hit = np.zeros(n, bool)
    for i in range(n):
        for d in ctx.tl.dets_of(i, ("cash",), float(pr["object_min_score"])):
            cx, cy = bbox_center(tuple(d.bbox))
            if any(point_in_polygon(cx, cy, p) for p in zones):
                hit[max(0, i - 1) : i + 2] = True
    out = np.where(hit, T, np.where(legacy == F, F, np.where(legacy == O, O, U))).astype(np.int8)
    return out


def after_leave(ctx: PredContext, leaves: list[dict[str, Any]], left: np.ndarray, lights: np.ndarray, obj: np.ndarray, vis: Optional[np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Решение по окну [t_ухода, t_ухода + window_s]: true — объект виден в ≥ k_true сэмплах; false — виден,
    что его нет, в ≥ k_false сэмплах и ни разу «да»; иначе unknown. В темноте — последнее известное
    состояние внутри окна до наступления темноты (заморозка)."""
    pr = ctx.cfg["predicates"]
    n = len(ctx.times)
    out = np.where(left == F, F, U).astype(np.int8)    # кассир на месте — «после ухода» неприменимо → false
    why = ["" if left[i] == F else ("no_leave_observed" if left[i] == T else "cashier_presence_unknown") for i in range(n)]
    vis_thr = float(pr["vis_thr"])
    for ev in leaves:
        i0 = ev["i"]
        i_end = ev["i_end"]
        t_end_win = ctx.times[i0] + float(pr["after_leave_window_s"])
        win = [i for i in range(i0, i_end + 1) if ctx.times[i] <= t_end_win + 1e-9]
        states: list[int] = []
        last_known: Optional[int] = None
        n_dark = n_occ = 0
        for i in win:
            v = int(obj[i])
            if lights[i] == T:
                n_dark += 1
                if last_known is not None:
                    states.append(last_known)  # заморозка последнего известного состояния
                continue
            if v in (T, F):
                last_known = v
                states.append(v)
            elif v == O or (vis is not None and vis[i] < vis_thr):
                n_occ += 1
        k_t, k_f = int(pr["k_true"]), int(pr["k_false"])
        if states.count(T) >= k_t:
            dec, reason = T, ""
        elif states.count(F) >= k_f and states.count(T) == 0:
            dec, reason = F, ""
        else:
            dec = U
            reason = "dark" if n_dark >= max(1, len(win) // 2) else ("occluded" if n_occ >= max(1, len(win) // 2) else "no_evidence")
        out[i0 : i_end + 1] = dec
        for j in range(i0, i_end + 1):
            why[j] = reason
        ev["_last"] = (dec, reason)
    return out, _why_fill(out, why, "no_evidence")


# ----------------------------------------------------------------------------------------- предметы
def _static_regions(ctx: PredContext, label: str) -> list[tuple[float, float]]:
    """Неподвижные позиции предмета (например, стационарный телефон): центр детекций в ≥ static_frac сэмплов."""
    pr = ctx.cfg["predicates"]
    cents = [(i, bbox_center(tuple(d.bbox))) for i in range(len(ctx.times)) for d in ctx.tl.dets_of(i, (label,), 0.3)]
    if not cents:
        return []
    r = float(pr["static_radius"])
    regions: list[tuple[float, float]] = []
    pts = np.array([c for _, c in cents])
    frames = np.array([i for i, _ in cents])
    used = np.zeros(len(pts), bool)
    n = len(ctx.times)
    for k in range(len(pts)):
        if used[k]:
            continue
        near = np.hypot(pts[:, 0] - pts[k, 0], pts[:, 1] - pts[k, 1]) <= r
        used |= near
        if len(set(frames[near].tolist())) >= float(pr["static_frac"]) * n:
            regions.append((float(np.median(pts[near, 0])), float(np.median(pts[near, 1]))))
    return regions


def _object_near_staff(ctx: PredContext, labels: tuple[str, ...], roles: dict[int, _TrackRole], static: list[tuple[float, float]]) -> np.ndarray:
    pr = ctx.cfg["predicates"]
    n = len(ctx.times)
    hit = np.zeros(n, bool)
    r = float(pr["static_radius"])
    for i in range(n):
        staff = [o for o in ctx.tl.persons[i] if o.track_id is None or roles.get(int(o.track_id), _TrackRole()).staff or o.p_cashier >= 0.5]
        for d in ctx.tl.dets_of(i, labels, float(pr["object_min_score"])):
            b = tuple(d.bbox)
            cx, cy = bbox_center(b)
            if any(np.hypot(cx - sx, cy - sy) <= r for sx, sy in static):
                continue  # стационарный предмет (телефон на столе) — не «в руке»
            if any(bbox_intersection(b, o.bbox) >= 0.5 * max(bbox_area(b), 1e-9) for o in staff):
                hit[i] = True
    # ±object_window_s
    w = float(pr["object_window_s"])
    out = hit.copy()
    for i in np.flatnonzero(hit):
        out[np.abs(ctx.times - ctx.times[i]) <= w] = True
    return out


def object_in_hand(ctx: PredContext, pose_key: str, labels: tuple[str, ...], roles: dict[int, _TrackRole], det_only: bool) -> tuple[np.ndarray, list[str]]:
    """Поза говорит «в руке/у лица» И рядом с кассиром виден предмет (не стационарный).
    Рука у лица без предмета — не сигнал (unknown, причина no_object_evidence)."""
    pr = ctx.cfg["predicates"]
    n = len(ctx.times)
    static = _static_regions(ctx, "cell_phone") if "cell_phone" in labels else []
    near = _object_near_staff(ctx, labels, roles, static)
    pose = ctx.sig_tri.get(pose_key, np.full(n, U, np.int8))
    out = np.full(n, U, np.int8)
    why = ["no_pose"] * n
    for i in range(n):
        if pose[i] == T:
            out[i] = T if near[i] else U
            why[i] = "" if near[i] else "no_object_evidence"
        elif pose[i] == F:
            out[i] = F
            why[i] = ""
        elif det_only and near[i]:
            out[i] = T
            why[i] = ""
    out = tri.sustain(out, ctx.times, float(pr["object_min_s"]), float(pr["max_gap_s"]))
    return out, _why_fill(out, why, "no_evidence")


def personal_items(ctx: PredContext, lights: np.ndarray) -> tuple[np.ndarray, list[str]]:
    pr = ctx.cfg["predicates"]
    n = len(ctx.times)
    desk = ctx.zones.get("desk")
    if not desk:
        return np.full(n, U, np.int8), ["no_zone"] * n
    labels = tuple(pr["personal_labels"])
    vis = ctx.sig_vis.get("cash_visible", np.ones(n))
    out = np.full(n, U, np.int8)
    why = [""] * n
    for i in range(n):
        items = [d for d in ctx.tl.dets_of(i, labels, float(pr["object_min_score"])) if any(point_in_polygon(*bbox_center(tuple(d.bbox)), p) for p in desk)]
        if items:
            out[i] = T
        elif lights[i] == T:
            why[i] = "dark"
        elif not ctx.det_ran[i]:
            why[i] = "no_detector"
        elif vis[i] < float(pr["vis_thr"]):
            out[i] = O
            why[i] = "occluded"
        elif ctx.quality[i] >= float(pr["min_quality"]):
            out[i] = F
    out = tri.sustain(out, ctx.times, float(pr["personal_min_s"]), float(pr["max_gap_s"]))
    return out, _why_fill(out, why, "no_evidence")


def bag_unattended(ctx: PredContext, lights: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Сумка видна и никого рядом ≥ min_s → true; сумку держат → false; сумки не видно → unknown (могла уйти в сейф)."""
    pr = ctx.cfg["predicates"]
    n = len(ctx.times)
    out = np.full(n, U, np.int8)
    why = ["bag_not_visible"] * n
    for i in range(n):
        bags = ctx.tl.dets_of(i, ("cash_bag",), float(pr["bag_min_score"]))
        if not bags or lights[i] == T:
            why[i] = "dark" if lights[i] == T else "bag_not_visible"
            continue
        alone = any(all(bbox_gap(tuple(d.bbox), o.bbox) > float(pr["reach"]) for o in ctx.tl.persons[i]) for d in bags)
        out[i] = T if alone else F
        why[i] = ""
    out = tri.sustain(out, ctx.times, float(pr["bag_alone_min_s"]), float(pr["max_gap_s"]))
    return out, _why_fill(out, why, "no_evidence")


# ----------------------------------------------------------------------------------------- сборка
def compute(ctx: PredContext) -> PredResult:
    pr = ctx.cfg["predicates"]
    n = len(ctx.times)
    res: dict[str, np.ndarray] = {}
    whys: dict[str, list[str]] = {}

    lo, lo_why = lights_off(ctx)
    res["lights_off"], whys["lights_off"] = lo, lo_why

    left, leaves = leave_events(ctx)
    res["cashier_left_workplace"] = left
    whys["cashier_left_workplace"] = _why_fill(left, ["track_lost" if ctx.pres_extra.get("workplace_lost", np.zeros(n))[i] > 0 else "" for i in range(n)], "no_evidence")

    st = ctx.sig_tri
    cash_desk = _cash_on_desk(ctx)
    for name, obj, vis in (
        ("cash_left_after_leave", cash_desk, ctx.sig_vis.get("cash_visible")),
        ("bag_left_after_leave", _bag_on_desk(ctx), None),
        ("safe_open_after_leave", st["safe_open"], ctx.sig_vis.get("safe_open")),
    ):
        res[name], whys[name] = after_leave(ctx, leaves, left, lo, obj, vis)
        for ev in leaves:
            ev.setdefault("decisions", {})[name] = tri.NAMES[int(ev["_last"][0])]
    for ev in leaves:
        ev.pop("_last", None)

    res["personal_item_on_desk"], whys["personal_item_on_desk"] = personal_items(ctx, lo)

    sides = _sides(ctx)
    roles = _roles(ctx, sides)
    for name, (x, w) in outsider_preds(ctx, sides, roles, lo).items():
        res[name], whys[name] = x, w
    entries = entry_events(ctx, sides, roles)
    res["entry_while_valuables_open"], whys["entry_while_valuables_open"] = entry_pred(ctx, entries, lo, cash_desk)

    safe_vis = ctx.sig_vis.get("safe_open", np.ones(n))
    nobody_at_safe = np.where(safe_vis >= float(pr["vis_thr_safe_free"]), T, F).astype(np.int8)
    sow = tri.t_and(st["safe_open"], st["cashier_in_workplace"], nobody_at_safe)
    res["safe_open_while_working"] = tri.sustain(sow, ctx.times, float(pr["safe_working_min_s"]), float(pr["max_gap_s"]))
    whys["safe_open_while_working"] = _why_fill(res["safe_open_while_working"], [""] * n, "no_evidence")

    det_only = bool(pr.get("phone_det_only", False))
    res["mobile_phone_in_hand"], whys["mobile_phone_in_hand"] = object_in_hand(ctx, "phone_in_hand", ("cell_phone",), roles, det_only)
    res["eating_drinking_with_object"], whys["eating_drinking_with_object"] = object_in_hand(ctx, "eating_drinking", ("cup", "bottle", "food"), roles, False)
    valuables = tri.t_or(cash_desk, st["safe_open"])
    ew = tri.t_and(res["eating_drinking_with_object"], st["cashier_in_workplace"], valuables)
    res["eating_at_workplace_with_valuables"] = ew
    whys["eating_at_workplace_with_valuables"] = _why_fill(ew, [""] * n, "no_evidence")

    res["bag_left_unattended"], whys["bag_left_unattended"] = bag_unattended(ctx, lo)

    events = [dict(e) for e in leaves] + [dict(e) for e in entries]
    for e in events:
        e.pop("i", None)
        e.pop("i_end", None)
    return PredResult(tri=res, why=whys, events=sorted(events, key=lambda e: e["t"]))
