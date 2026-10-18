"""Утилиты для A08/A09 поверх контрактных Track/Zone (чистые функции, без моделей).

Важно: исчезновение трека из детекций НЕ равно уходу из помещения.
`gaps()` классифицирует каждый пропуск:
  lost_in_view    — пропал в глубине кадра: окклюзия / пропуск детектора, НЕ уход;
  left_view_edge  — последний раз виден у края кадра: вероятно вышел из поля зрения
                    (из помещения — не обязательно);
  left_via_door   — последний раз виден в зоне door.
`entries_exits()` по умолчанию НЕ создаёт exit из-за пропуска детекций (только по
наблюдаемому пересечению границы зоны); это можно включить параметром gap_as_exit_s.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .contract import SceneState
from .geometry import anchor_point, box_gap, box_diag, point_in_polygon

EventTuple = tuple[float, int, str, str]


def _get(o: Any, k: str, default: Any = None) -> Any:
    return o.get(k, default) if isinstance(o, dict) else getattr(o, k, default)


def _rows(track: Any) -> list[tuple[float, tuple]]:
    return sorted(((float(t), tuple(b)) for t, b in (_get(track, "boxes", []) or [])), key=lambda r: r[0])


def zone_by_name(zones: Iterable[Any], name: str) -> Optional[Any]:
    for z in zones or []:
        if _get(z, "name") == name:
            return z
    return None


def box_at(track: Any, t: float, max_dt: float = 0.5, interpolate: bool = False) -> Optional[tuple]:
    """Наблюдённый бокс трека в момент t (ближайший в пределах max_dt), иначе None."""
    rows = _rows(track)
    if not rows:
        return None
    ts = np.array([r[0] for r in rows])
    i = int(np.searchsorted(ts, t))
    cands = [j for j in (i - 1, i) if 0 <= j < len(rows)]
    j = min(cands, key=lambda k: abs(ts[k] - t))
    if abs(ts[j] - t) <= max_dt:
        return rows[j][1]
    if interpolate and 0 < i < len(rows) and ts[i] - ts[i - 1] <= 2 * max_dt:
        a = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
        return tuple(float(x) + a * (float(y) - float(x)) for x, y in zip(rows[i - 1][1], rows[i][1]))
    return None


def inside(bbox: Sequence[float], zone: Any, anchor: str = "center") -> bool:
    return point_in_polygon(anchor_point(bbox, anchor), _get(zone, "polygon", []) or [])


def entries_exits(tracks: Iterable[Any], zones: Iterable[Any], *, anchor: str = "center",
                  min_dwell_s: float = 0.4, enter_at_start: bool = True,
                  gap_as_exit_s: Optional[float] = None, labels: Sequence[str] = ("person",)) -> list[EventTuple]:
    """(t, track_id, zone, "enter"|"exit") с гистерезисом min_dwell_s.

    enter_at_start — трек появился сразу внутри зоны -> enter в момент появления.
    gap_as_exit_s  — None (по умолчанию): пропуск детекций не считается выходом.
    """
    out: list[EventTuple] = []
    zones = list(zones or [])
    for tr in tracks or []:
        if labels and _get(tr, "label") not in labels:
            continue
        rows = _rows(tr)
        if not rows:
            continue
        tid = int(_get(tr, "track_id"))
        for z in zones:
            name = str(_get(z, "name"))
            state: Optional[bool] = None
            cand: Optional[bool] = None
            cand_t = 0.0
            prev_t: Optional[float] = None
            for t, b in rows:
                ins = inside(b, z, anchor)
                if gap_as_exit_s is not None and prev_t is not None and t - prev_t > gap_as_exit_s and state:
                    out.append((prev_t, tid, name, "exit"))
                    state, cand = False, None
                prev_t = t
                if state is None:
                    state = ins
                    if ins and enter_at_start:
                        out.append((t, tid, name, "enter"))
                    continue
                if ins == state:
                    cand = None
                    continue
                if cand is None or cand != ins:
                    cand, cand_t = ins, t
                if t - cand_t >= min_dwell_s - 1e-9:
                    state = ins
                    out.append((cand_t, tid, name, "enter" if ins else "exit"))
                    cand = None
            if cand is not None and state is not None and cand != state and rows[-1][0] - cand_t >= min_dwell_s - 1e-9:
                out.append((cand_t, tid, name, "enter" if cand else "exit"))
    out.sort(key=lambda e: (e[0], e[1], e[2], e[3]))
    return out


def dwell(track: Any, zone: Any, t0: Optional[float] = None, t1: Optional[float] = None, *,
          max_dt: float = 1.0, anchor: str = "center") -> float:
    """Секунды наблюдённого присутствия трека в зоне (разрывы > max_dt не засчитываются)."""
    rows = [r for r in _rows(track) if (t0 is None or r[0] >= t0) and (t1 is None or r[0] <= t1)]
    if not rows:
        return 0.0
    ts = [r[0] for r in rows]
    dts = [min(b - a, max_dt) for a, b in zip(ts[:-1], ts[1:])]
    if dts:
        dts.append(float(np.median(dts)))
    else:
        dts = [min(0.2, max_dt)]
    return float(sum(dt for (t, b), dt in zip(rows, dts) if inside(b, zone, anchor)))


def proximity(track_a: Any, track_b: Any, t: float, *, max_dt: float = 0.6, aspect: float = 1.0,
              scale: float = 1.0) -> float:
    """Близость двух треков в момент t: 1 — касаются/пересекаются, 0 — далеко или кого-то нет."""
    a = box_at(track_a, t, max_dt)
    b = box_at(track_b, t, max_dt)
    if a is None or b is None:
        return 0.0
    g = box_gap(a, b, aspect)
    d = 0.5 * (box_diag(a, aspect) + box_diag(b, aspect)) * scale
    return float(max(0.0, 1.0 - g / max(d, 1e-6)))


def _near_edge(b: Sequence[float], m: float) -> bool:
    return b[0] <= m or b[1] <= m or b[2] >= 1.0 - m or b[3] >= 1.0 - m


def gaps(track: Any, zones: Optional[Iterable[Any]] = None, *, min_gap_s: float = 0.6,
         edge_margin: float = 0.06, t_end: Optional[float] = None) -> list[dict[str, Any]]:
    """Пропуски трека с классификацией (см. docstring модуля).

    Если t_end задан и трек закончился раньше — последний элемент с t_found=None.
    """
    rows = _rows(track)
    if not rows:
        return []
    zl = list(zones or [])
    door = zone_by_name(zl, "door")
    tid = int(_get(track, "track_id", 0))

    def classify(b: Sequence[float]) -> tuple[str, Optional[str]]:
        last_zone = None
        for z in zl:
            if inside(b, z):
                last_zone = str(_get(z, "name"))
                break
        if door is not None and inside(b, door):
            return "left_via_door", last_zone
        if _near_edge(b, edge_margin):
            return "left_view_edge", last_zone
        return "lost_in_view", last_zone

    out = []
    for (ta, ba), (tb, _) in zip(rows[:-1], rows[1:]):
        if tb - ta >= min_gap_s:
            kind, lz = classify(ba)
            out.append({"track_id": tid, "t_lost": ta, "t_found": tb, "duration_s": round(tb - ta, 3),
                        "kind": kind, "last_zone": lz})
    if t_end is not None and t_end - rows[-1][0] >= min_gap_s:
        kind, lz = classify(rows[-1][1])
        out.append({"track_id": tid, "t_lost": rows[-1][0], "t_found": None,
                    "duration_s": round(t_end - rows[-1][0], 3), "kind": kind, "last_zone": lz})
    return out


def presence_state(track: Any, t: float, zones: Optional[Iterable[Any]] = None, *, max_dt: float = 0.6,
                   edge_margin: float = 0.06) -> str:
    """observed | not_yet | lost_in_view | left_view_edge | left_via_door."""
    rows = _rows(track)
    if not rows:
        return "not_yet"
    if box_at(track, t, max_dt) is not None:
        return "observed"
    if t < rows[0][0]:
        return "not_yet"
    prev = [r for r in rows if r[0] <= t]
    g = gaps({"track_id": _get(track, "track_id", 0), "boxes": prev}, zones, min_gap_s=0.0,
             edge_margin=edge_margin, t_end=t)
    return g[-1]["kind"] if g else "lost_in_view"


_OUTSIDER_WEIGHT = {"visitor": 1.0, "unknown": 0.5, "collector": 0.0, "cashier": 0.0}


def person_signals(tracks: Iterable[Any], zones: Iterable[Any], times: Sequence[float], *,
                   max_dt: float = 0.6, extended: bool = False) -> list[SceneState]:
    """Сигналы о людях: cashier_present, cashier_in_workplace, n_persons,
    outsider_in_cashier_area, collector_present (имена из SIGNALS).

    Кассир не виден, но пропал «в глубине кадра» -> 0.5 (неизвестно), а не 0.
    outsider_in_cashier_area: visitor -> role_conf, unknown -> 0.5·role_conf..0.5, collector -> 0
    (non-cashier ≠ неуполномоченный; окончательное решение — за правилами A09).
    extended=True добавляет служебные ключи с префиксом x_ (не из SIGNALS).
    """
    persons = [tr for tr in tracks or [] if _get(tr, "label") == "person"]
    zl = list(zones or [])
    ca = zone_by_name(zl, "cashier_area") or zone_by_name(zl, "desk")
    out = []
    for t in times:
        t = float(t)
        n = 0
        cp = cw = out_ca = coll = 0.0
        x_lost = 0.0
        for tr in persons:
            role = _get(tr, "role", "unknown")
            rc = float(_get(tr, "role_conf", 0.0) or 0.0)
            b = box_at(tr, t, max_dt)
            if b is None:
                if role == "cashier":
                    st = presence_state(tr, t, zl, max_dt=max_dt)
                    if st == "lost_in_view":
                        cp = max(cp, 0.5)
                        x_lost = 1.0
                        last = box_at(tr, t, max_dt=1e9)
                        if ca is not None and last is not None and inside(last, ca):
                            cw = max(cw, 0.5)
                continue
            n += 1
            if role == "cashier":
                cp = max(cp, 1.0)
                if ca is not None and inside(b, ca):
                    cw = 1.0
            elif ca is not None and inside(b, ca):
                w = _OUTSIDER_WEIGHT.get(role, 0.5)
                out_ca = max(out_ca, w * (rc if role == "visitor" else 1.0))
            if role == "collector":
                coll = max(coll, min(1.0, rc / 0.5) if rc < 0.5 else 1.0)
        sig = {"cashier_present": cp, "cashier_in_workplace": cw, "n_persons": float(n),
               "outsider_in_cashier_area": round(out_ca, 3), "collector_present": round(coll, 3)}
        if extended:
            sig["x_cashier_lost_in_view"] = x_lost
        out.append(SceneState(t=t, signals=sig))
    return out
