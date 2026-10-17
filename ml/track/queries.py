"""Явные функции-события для правил (A09): ЗНАЧЕНИЕ + УВЕРЕННОСТЬ + ИСТОЧНИК.

    q = TrackQueries(tracks, zones, info)          # info = track_full().info (необязательно)
    q.cashier_left_workplace(t)   q.cashier_left_room(t)   q.person_entered(t, role)
    q.n_outsiders(t)              q.outsider_at_cashier_side(t)   q.events()

Правила честности:
* value=None — «неизвестно» (не видели: окклюзия, пропуск детектора, потеря видимости,
  зона не определена). Это НЕ False и не «нарушения нет».
* unknown-роль не превращается в «посторонний»: такие люди идут в detail["unknown"].
* Вход — только появление у двери/края кадра. Появление в глубине кадра после окклюзии,
  пропуска детектора или выключенного света — не вход (устранённый источник ложных V01/V04).
* t_event — момент НАЧАЛА (переход: вышла из зоны, появился в двери).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .config import load_config
from .events import box_at, gaps, inside, zone_by_name
from .geometry import anchor_point, point_in_polygon, scale_polygon
from .visibility import in_intervals, loss_from_tracks


@dataclass
class Answer:
    value: Any                    # bool | int | None (None = неизвестно)
    conf: float                   # уверенность в value, 0..1
    source: str                   # откуда вывод (provenance)
    track_ids: list[int] = field(default_factory=list)
    t_event: Optional[float] = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackEvent:
    kind: str                     # person_entered | cashier_left_workplace | cashier_returned_workplace |
                                  # cashier_left_room | outsider_at_cashier_side | unknown_at_cashier_side
    t: float                      # начало (момент перехода)
    t_end: Optional[float]
    conf: float
    source: str
    track_ids: list[int]
    role: Optional[str] = None
    facts: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _get(o: Any, k: str, default: Any = None) -> Any:
    return o.get(k, default) if isinstance(o, dict) else getattr(o, k, default)


def _rows(tr: Any) -> list[tuple[float, tuple]]:
    return sorted(((float(t), tuple(b)) for t, b in (_get(tr, "boxes", []) or [])), key=lambda r: r[0])


def _near_edge(b: Sequence[float], m: float) -> bool:
    return b[0] <= m or b[1] <= m or b[2] >= 1.0 - m or b[3] >= 1.0 - m


class TrackQueries:
    def __init__(self, tracks: Iterable[Any], zones: Iterable[Any], info: Optional[dict] = None,
                 cfg: Optional[dict] = None, *, t_end: Optional[float] = None) -> None:
        full = cfg or load_config()
        self.cfg = full["queries"]
        self.anchor = full.get("anchor", "center")
        qc = self.cfg
        self.tracks = list(tracks or [])
        self.persons = [t for t in self.tracks if _get(t, "label") == "person"]
        self.zl = list(zones or [])
        info = info or {}
        all_t = [r[0] for t in self.tracks for r in _rows(t)]
        self.frame_dt = float(info.get("frame_dt") or (np.median(np.diff(sorted(set(all_t)))) if len(set(all_t)) > 1 else 0.2))
        self.max_dt = max(qc["max_dt"], 1.5 * self.frame_dt)
        cov = info.get("coverage") or {}
        self.t0 = float(cov.get("t_first_detection", min(all_t) if all_t else 0.0))
        self.t1 = float(t_end if t_end is not None else cov.get("t_last_detection", max(all_t) if all_t else 0.0))
        if "visibility_loss" in cov:
            self.vis = [tuple(x) for x in cov.get("visibility_loss") or []]
            self.vis_status = cov.get("visibility_check", "objects")
        else:
            self.vis, self.vis_status = loss_from_tracks(self.tracks, self.frame_dt, qc["min_loss_s"])
        wp = zone_by_name(self.zl, "cashier_area")
        self.workplace = wp if wp is not None and float(_get(wp, "conf", 0)) >= qc["min_zone_conf"] else None
        self.door = zone_by_name(self.zl, "door")
        safe = zone_by_name(self.zl, "safe")
        self.safe_poly = scale_polygon(_get(safe, "polygon"), qc["safe_expand"]) if safe is not None else None
        self.cashiers = [t for t in self.persons if _get(t, "role") == "cashier"
                         and float(_get(t, "role_conf", 0) or 0) >= qc["min_role_conf"]]
        smooth = qc.get("smoothing", "off") == "hysteresis"
        self.min_state_s = qc["smooth_min_state_s"] if smooth else qc["min_state_s"]
        self.merge_gap_s = qc["smooth_merge_gap_s"] if smooth else qc["merge_gap_s"]
        self.grid = self._grid()
        self._wp_states = self._workplace_states()
        self._away = self._episodes(self._wp_states, "away")
        self._entries = self._entry_events()

    # ---------- служебное ----------
    def _grid(self) -> np.ndarray:
        if self.t1 <= self.t0:
            return np.array([self.t0])
        return np.round(np.arange(self.t0, self.t1 + 1e-6, max(self.frame_dt, 1e-3)), 4)

    def _role(self, tr: Any) -> str:
        r = _get(tr, "role", "unknown") or "unknown"
        if r != "unknown" and float(_get(tr, "role_conf", 0) or 0) < self.cfg["min_role_conf"]:
            return "unknown"  # низкая уверенность передаётся дальше как unknown
        return r

    def _low_quality(self, tr: Any) -> bool:
        rows = _rows(tr)
        return len(rows) < 2 or (rows[-1][0] - rows[0][0] + self.frame_dt) < self.cfg["low_quality_min_s"]

    def _in_wp(self, b: Sequence[float]) -> bool:
        return self.workplace is not None and inside(b, self.workplace, self.anchor)

    def _workplace_states(self) -> list[str]:
        """Состояние кассира на сетке: at | away | unknown | not_yet."""
        out = []
        seen = False
        was_at = False  # «ушёл с рабочего места» возможно только после того, как там был
        for t in self.grid:
            if in_intervals(t, self.vis):
                out.append("unknown" if was_at else "not_yet")
                continue
            boxes = [b for b in (box_at(c, t, self.max_dt) for c in self.cashiers) if b is not None]
            if boxes:
                seen = True
                at = any(self._in_wp(b) for b in boxes)
                was_at = was_at or at
                out.append("at" if at else ("away" if was_at else "not_yet"))
                continue
            if not seen or not was_at:
                out.append("not_yet")
                continue
            st = "unknown"
            for c in self.cashiers:
                prev = [r for r in _rows(c) if r[0] <= t]
                if not prev:
                    continue
                g = gaps({"track_id": _get(c, "track_id", 0), "boxes": prev}, self.zl, min_gap_s=0.0,
                         edge_margin=self.cfg["edge_margin"], t_end=float(t))
                kind = g[-1]["kind"] if g else "lost_in_view"
                if kind in ("left_via_door", "left_view_edge") or not self._in_wp(prev[-1][1]):
                    st = "away"  # ушёл из поля зрения / был вне рабочего места, когда пропал
                else:
                    st = "unknown" if st != "away" else st  # пропал на рабочем месте: окклюзия ≠ уход
            out.append(st)
        return out

    def _episodes(self, states: Sequence[str], target: str) -> list[tuple[float, float]]:
        """Эпизоды state==target с гистерезисом min_state_s; unknown не разрывает текущее состояние."""
        g = self.grid
        cur = None
        runs: list[list[float]] = []
        cand = None
        cand_t = 0.0
        for t, s in zip(g, states):
            if s in ("unknown", "not_yet"):
                continue
            val = s == target
            if cur is None:
                cur = val
                if val:
                    runs.append([float(t), float(t)])
                continue
            if val == cur:
                cand = None
                if val:
                    runs[-1][1] = float(t)
                continue
            if cand is None:
                cand, cand_t = val, float(t)
            if float(t) - cand_t + self.frame_dt >= self.min_state_s:
                cur = val
                cand = None
                if val:
                    runs.append([cand_t, float(t)])
        merged: list[list[float]] = []
        for r in runs:
            if merged and r[0] - merged[-1][1] <= self.merge_gap_s:
                merged[-1][1] = r[1]
            else:
                merged.append(list(r))
        return [(a, b) for a, b in merged]

    def _entry_events(self) -> list[TrackEvent]:
        qc = self.cfg
        door_poly = scale_polygon(_get(self.door, "polygon"), qc["door_expand"]) if self.door is not None else None
        out: list[TrackEvent] = []
        for tr in self.persons:
            rows = _rows(tr)
            if not rows or self._low_quality(tr):
                continue
            tid = int(_get(tr, "track_id"))
            starts = [0] + [i + 1 for i in range(len(rows) - 1) if rows[i + 1][0] - rows[i][0] > max(qc["min_gap_s"], 2 * self.frame_dt)]
            for k, i in enumerate(starts):
                t, b = rows[i]
                if t <= self.t0 + qc["start_grace_s"]:
                    continue  # был в кадре с начала — не «вошёл»
                if in_intervals(t, self.vis, pad=qc["vis_grace_s"]):
                    continue  # вернулась видимость — это не вход
                if k > 0:  # повторное появление того же трека: вход, только если перед этим вышел
                    prev_b = rows[i - 1][1]
                    left = (door_poly is not None and point_in_polygon(anchor_point(prev_b, self.anchor), door_poly)) \
                        or _near_edge(prev_b, qc["edge_margin"])
                    if not left:
                        continue
                p = anchor_point(b, self.anchor)
                if door_poly is not None and point_in_polygon(p, door_poly):
                    src, conf = "appeared_in_door_zone", 0.8
                elif _near_edge(b, qc["edge_margin"]):
                    src, conf = "appeared_at_frame_edge", 0.6
                elif not qc["entry_requires_boundary"]:
                    src, conf = "appeared_in_view", 0.3
                else:
                    continue  # появился в глубине кадра: окклюзия/пропуск детектора, не вход
                role = self._role(tr)
                out.append(TrackEvent("person_entered", t, None, round(conf * (0.5 + 0.5 * self.door_conf()), 3), src,
                                      [tid], role, [f"Человек (трек #{tid}, роль: {role}) появился в кадре в {t:.1f} с"
                                                    + (" в зоне двери" if src == "appeared_in_door_zone" else " у края кадра")]))
        out.sort(key=lambda e: (e.t, e.track_ids))
        return out

    def door_conf(self) -> float:
        return float(_get(self.door, "conf", 0.0)) if self.door is not None else 0.0

    def _base_conf(self) -> float:
        rc = max([float(_get(c, "role_conf", 0) or 0) for c in self.cashiers] or [0.0])
        zc = float(_get(self.workplace, "conf", 0.0)) if self.workplace is not None else 0.0
        return round(rc * (0.5 + 0.5 * zc), 3)

    # ---------- публичные функции ----------
    def cashier_left_workplace(self, t: float) -> Answer:
        if not self.cashiers:
            return Answer(None, 0.0, "no_cashier_identified")
        if self.workplace is None:
            return Answer(None, 0.0, "workplace_zone_undefined")
        ids = [int(_get(c, "track_id")) for c in self.cashiers]
        for a, b in self._away:
            if a <= t <= b + self.frame_dt * 0.5:
                return Answer(True, self._base_conf(), "observed_away_from_workplace", ids, a, {"t_end": b})
        i = int(np.argmin(np.abs(self.grid - t)))
        st = self._wp_states[i]
        if st == "at":
            return Answer(False, self._base_conf(), "observed_at_workplace", ids)
        if st == "not_yet":
            return Answer(None, 0.0, "cashier_not_yet_seen", ids)
        if st == "away":  # короче гистерезиса — дребезг
            return Answer(False, 0.3, "brief_absence_below_min_state", ids)
        return Answer(None, 0.2, "visibility_loss" if in_intervals(t, self.vis) else "lost_in_view_occlusion_or_miss", ids)

    def cashier_left_room(self, t: float) -> Answer:
        if not self.cashiers:
            return Answer(None, 0.0, "no_cashier_identified")
        if in_intervals(t, self.vis):
            return Answer(None, 0.0, "visibility_loss")
        ids = [int(_get(c, "track_id")) for c in self.cashiers]
        if any(box_at(c, t, self.max_dt) is not None for c in self.cashiers):
            return Answer(False, self._base_conf(), "cashier_observed", ids)
        last = None
        for c in self.cashiers:
            prev = [r for r in _rows(c) if r[0] <= t]
            if prev and (last is None or prev[-1][0] > last[1][-1][0]):
                last = (c, prev)
        if last is None:
            return Answer(None, 0.0, "cashier_not_yet_seen", ids)
        g = gaps({"track_id": _get(last[0], "track_id", 0), "boxes": last[1]}, self.zl, min_gap_s=0.0,
                 edge_margin=self.cfg["edge_margin"], t_end=float(t))
        kind = g[-1]["kind"] if g else "lost_in_view"
        tl = last[1][-1][0]
        rc = float(_get(last[0], "role_conf", 0) or 0)
        if kind == "left_via_door":
            return Answer(True, round(0.8 * rc * (0.5 + 0.5 * self.door_conf()), 3), "last_seen_in_door_zone", ids, tl)
        if kind == "left_view_edge":
            return Answer(True, round(0.5 * rc, 3), "last_seen_at_frame_edge", ids, tl,
                          {"note": "вышел из поля зрения; выход из помещения не подтверждён"})
        return Answer(None, 0.2, "lost_in_view_occlusion_or_miss", ids, tl)

    def person_entered(self, t: float, role: Optional[str] = None, window_s: Optional[float] = None) -> Answer:
        w = self.cfg["entry_window_s"] if window_s is None else window_s
        if in_intervals(t, self.vis):
            return Answer(None, 0.0, "visibility_loss")
        hits = [e for e in self._entries if t - w <= e.t <= t and (role is None or e.role == role)]
        if hits:
            best = max(hits, key=lambda e: e.conf)
            return Answer(True, best.conf, best.source, sorted({i for e in hits for i in e.track_ids}), min(e.t for e in hits))
        return Answer(False, 0.6 if self.door is not None else 0.4, "no_entry_observed")

    def _people_at(self, t: float) -> list[tuple[Any, tuple, str]]:
        out = []
        for tr in self.persons:
            b = box_at(tr, t, self.max_dt)
            if b is not None:
                out.append((tr, b, self._role(tr)))
        return out

    def n_outsiders(self, t: float) -> Answer:
        if in_intervals(t, self.vis):
            return Answer(None, 0.0, "visibility_loss")
        vis_ids, unk_ids, lq = [], [], 0
        for tr, _b, role in self._people_at(t):
            if self._low_quality(tr):
                lq += 1
                continue
            if role == "visitor":
                vis_ids.append(int(_get(tr, "track_id")))
            elif role == "unknown":
                unk_ids.append(int(_get(tr, "track_id")))
        conf = 0.9 if not unk_ids else 0.5
        return Answer(len(vis_ids), conf, "observed_tracks_by_role", vis_ids + unk_ids, None,
                      {"visitor": len(vis_ids), "unknown": len(unk_ids), "low_quality": lq,
                       "upper_bound": len(vis_ids) + len(unk_ids)})

    def outsider_at_cashier_side(self, t: float) -> Answer:
        if in_intervals(t, self.vis):
            return Answer(None, 0.0, "visibility_loss")
        if self.workplace is None and self.safe_poly is None:
            return Answer(None, 0.0, "cashier_side_undefined")
        zc = max(float(_get(self.workplace, "conf", 0.0)) if self.workplace is not None else 0.0, 0.5 if self.safe_poly else 0.0)
        vis_hits, unk_hits, coll = [], [], []
        for tr, b, role in self._people_at(t):
            if role == "cashier" or self._low_quality(tr):
                continue
            p = anchor_point(b, self.anchor)
            at_side = (self.workplace is not None and point_in_polygon(p, _get(self.workplace, "polygon"))) or \
                      (self.safe_poly is not None and point_in_polygon(p, self.safe_poly))
            if not at_side:
                continue
            tid = int(_get(tr, "track_id"))
            if role == "visitor":
                vis_hits.append((tid, float(_get(tr, "role_conf", 0) or 0)))
            elif role == "collector":
                coll.append(tid)
            else:
                unk_hits.append(tid)
        detail = {"visitor": [h[0] for h in vis_hits], "unknown": unk_hits, "collector": coll}
        if vis_hits:
            return Answer(True, round(max(h[1] for h in vis_hits) * zc, 3), "visitor_at_cashier_side",
                          [h[0] for h in vis_hits] + unk_hits, None, detail)
        if unk_hits:
            return Answer(None, 0.5, "unknown_role_at_cashier_side", unk_hits, None, detail)
        return Answer(False, round(zc, 3), "no_outsider_at_cashier_side", coll, None, detail)

    def events(self) -> list[TrackEvent]:
        """Все переходы (onset) с уверенностью и источником; дребезг объединяется (merge_gap_s)."""
        ev: list[TrackEvent] = list(self._entries)
        ids = [int(_get(c, "track_id")) for c in self.cashiers]
        if self.cashiers and self.workplace is not None:
            bc = self._base_conf()
            for a, b in self._away:
                ev.append(TrackEvent("cashier_left_workplace", a, b, bc, "observed_away_from_workplace", ids, "cashier",
                                     [f"Кассир вне зоны рабочего места с {a:.1f} с"]))
                if b < self.t1 - self.frame_dt:
                    ev.append(TrackEvent("cashier_returned_workplace", b, None, bc, "observed_at_workplace", ids, "cashier"))
        for c in self.cashiers:
            for g in gaps(c, self.zl, min_gap_s=self.cfg["min_gap_s"], edge_margin=self.cfg["edge_margin"], t_end=self.t1):
                if g["kind"] in ("left_via_door", "left_view_edge") and not in_intervals(g["t_lost"], self.vis, 0.5):
                    a = self.cashier_left_room(min(g["t_lost"] + 2 * self.max_dt, self.t1))
                    if a.value:
                        ev.append(TrackEvent("cashier_left_room", g["t_lost"], g["t_found"], a.conf, a.source, ids, "cashier"))
        # посторонний / unknown на стороне кассы — эпизоды
        states = []
        for t in self.grid:
            a = self.outsider_at_cashier_side(float(t))
            states.append("at" if a.value is True else ("unk" if a.source == "unknown_role_at_cashier_side" else
                                                         ("unknown" if a.value is None else "no")))
        for kind, target in (("outsider_at_cashier_side", "at"), ("unknown_at_cashier_side", "unk")):
            st = [("away" if s == target else ("unknown" if s == "unknown" else "at")) for s in states]
            for a, b in self._episodes(st, "away"):
                ans = self.outsider_at_cashier_side(a)
                ev.append(TrackEvent(kind, a, b, ans.conf, ans.source, ans.track_ids,
                                     "visitor" if target == "at" else "unknown"))
        ev.sort(key=lambda e: (e.t, e.kind))
        return ev
