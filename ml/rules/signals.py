"""Подготовка рядов сигналов для правил.

* слияние ``SceneState`` из разных источников (pose_signals + scene_states) по времени;
* общая временная ось, forward-fill пропусков (разные модули могут работать на разной
  частоте), значения по умолчанию;
* производные сигналы из треков и зон — **только** если сигнала нет ни в одном состоянии
  (страховка на случай, когда A07/A08 недоступны или в mock-режиме);
* геометрия: точка в полигоне, бокс трека на момент ``t`` (интерполяция).

Координаты зон не захардкожены: всё берётся из ``Zone``/``Track``, пришедших от A06.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "SignalSeries",
    "build_series",
    "derive_from_tracks",
    "point_in_polygon",
    "track_box_at",
    "polygon_bbox",
]


@dataclass
class SignalSeries:
    times: np.ndarray                                   # (N,), отсортировано
    signals: dict[str, np.ndarray]                      # имя → (N,)
    present: set[str] = field(default_factory=set)      # сигналы, пришедшие в states
    coverage: dict[str, float] = field(default_factory=dict)   # доля кадров с реальным значением
    derived: dict[str, str] = field(default_factory=dict)      # имя → источник ("tracks")
    absent: set[str] = field(default_factory=set)       # сигналы, которых нет ни в одном состоянии (неизвестны)
    dt: float = 0.2                                     # медианный шаг сэмплирования

    def known(self, name: str) -> np.ndarray:
        """Маска кадров, где значение сигнала известно (после forward-fill)."""
        arr = self.signals.get(name)
        if arr is None:
            return np.zeros(self.n, dtype=bool)
        return np.isfinite(arr)

    @property
    def n(self) -> int:
        return int(self.times.shape[0])

    def value_at(self, name: str, t: float) -> float:
        arr = self.signals.get(name)
        if arr is None or self.n == 0:
            return float("nan")
        i = int(np.clip(np.searchsorted(self.times, t), 0, self.n - 1))
        if i > 0 and abs(self.times[i - 1] - t) <= abs(self.times[i] - t):
            i -= 1
        return float(arr[i])


def _state_t(state: Any) -> Optional[float]:
    t = getattr(state, "t", None)
    if t is None and isinstance(state, Mapping):
        t = state.get("t")
    try:
        t = float(t)
    except (TypeError, ValueError):
        return None
    return t if math.isfinite(t) else None


def _state_signals(state: Any) -> Mapping[str, Any]:
    sig = getattr(state, "signals", None)
    if sig is None and isinstance(state, Mapping):
        sig = state.get("signals")
        if sig is None:
            sig = {k: v for k, v in state.items() if k != "t"}
    return sig or {}


def _state_get(state: Any, name: str) -> Any:
    if isinstance(state, Mapping):
        return state.get(name)
    return getattr(state, name, None)


def _to_float(v: Any) -> float:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def build_series(
    states: Sequence[Any],
    needed: Iterable[str] = (),
    *,
    defaults: Optional[Mapping[str, float]] = None,
    ffill_max_s: float = 1.5,
    merge_policy: str = "max",
    time_round: float = 1e-3,
    known_min_conf: float = 0.5,
    observed_one_sided: bool = True,
) -> SignalSeries:
    """Свести список состояний в выровненные ряды.

    Отсутствие сигнала ≠ отрицание: кадры без значения (и пропуски длиннее ``ffill_max_s``)
    остаются ``NaN`` = «неизвестно». ``defaults`` — только явные, задокументированные
    исключения (например, модуль, который по договорённости не шлёт нулевой сигнал).

    Совместимость с ``SceneStateEx`` (A08) и его JSON (``state_to_dict``): если у состояния есть
    ``known[name]`` — значение берётся только при ``True``; иначе при наличии ``conf`` — только при
    ``conf[name] ≥ known_min_conf``; у расширенного состояния сигнал без сведений о подтверждении —
    неизвестен. ``observed`` (признаки вне SIGNALS) добавляются как сигналы; при
    ``observed_one_sided`` значение < 0.5 считается «не наблюдалось» (= неизвестно), а не «нет».
    ``quality`` → сигнал ``frame_quality``.
    """
    defaults = dict(defaults or {})
    buckets: dict[float, dict[str, float]] = {}
    for st in states or ():
        t = _state_t(st)
        if t is None:
            continue
        key = round(round(t / time_round) * time_round, 6)
        dst = buckets.setdefault(key, {})
        known = _state_get(st, "known")
        conf = _state_get(st, "conf")
        known = known if isinstance(known, Mapping) else None
        conf = conf if isinstance(conf, Mapping) else None
        values: list[tuple[str, float]] = []
        for name, raw in _state_signals(st).items():
            name = str(name)
            if known is not None and name in known:
                if not bool(known[name]):
                    continue  # не подтверждено наблюдением → неизвестно
            elif conf is not None and name in conf:
                if _to_float(conf[name]) < known_min_conf:
                    continue
            elif known is not None or conf is not None:
                continue  # расширенное состояние без сведений о подтверждении → неизвестно
            values.append((name, _to_float(raw)))
        observed = _state_get(st, "observed")
        if isinstance(observed, Mapping):
            for name, raw in observed.items():
                v = _to_float(raw)
                if observed_one_sided and not name.endswith("_conf") and v < 0.5:
                    continue  # «не наблюдали» ≠ «нет»
                values.append((str(name), v))
        quality = _state_get(st, "quality")
        if quality is not None and not isinstance(quality, (Mapping, list, tuple)):
            values.append(("frame_quality", _to_float(quality)))
        for name, v in values:
            if not math.isfinite(v):
                continue
            if name in dst and merge_policy == "max":
                dst[name] = max(dst[name], v)
            elif name in dst and merge_policy == "min":
                dst[name] = min(dst[name], v)
            else:
                dst[name] = v  # "last" или первое появление
    times = np.asarray(sorted(buckets), dtype=float)
    n = times.shape[0]
    dt = float(np.median(np.diff(times))) if n > 1 else 0.2
    if not math.isfinite(dt) or dt <= 0:
        dt = 0.2
    present: set[str] = set()
    for d in buckets.values():
        present.update(d)
    names = sorted(present | set(needed))
    signals: dict[str, np.ndarray] = {}
    coverage: dict[str, float] = {}
    absent: set[str] = set()
    rows = [buckets[k] for k in sorted(buckets)]
    for name in names:
        raw = np.array([row.get(name, np.nan) for row in rows], dtype=float) if n else np.zeros(0)
        have = np.isfinite(raw)
        coverage[name] = float(have.mean()) if n else 0.0
        fill = float(defaults[name]) if name in defaults else np.nan
        if not have.any():
            if name not in defaults:
                absent.add(name)
            signals[name] = np.full(n, fill)
            continue
        out = raw.copy()
        last_v, last_t = np.nan, -np.inf
        for i in range(n):
            if have[i]:
                last_v, last_t = raw[i], times[i]
            elif math.isfinite(last_v) and times[i] - last_t <= ffill_max_s + 1e-9:
                out[i] = last_v
            else:
                out[i] = fill
        signals[name] = out
    return SignalSeries(times=times, signals=signals, present=present, coverage=coverage,
                        absent=absent, dt=dt)


# --------------------------------------------------------------------------- geometry

def point_in_polygon(x: float, y: float, polygon: Sequence[Sequence[float]]) -> bool:
    """Ray casting; полигон в нормированных координатах."""
    pts = [(float(p[0]), float(p[1])) for p in polygon or ()]
    if len(pts) < 3:
        return False
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def polygon_bbox(polygon: Sequence[Sequence[float]]) -> Optional[tuple[float, float, float, float]]:
    pts = [(float(p[0]), float(p[1])) for p in polygon or ()]
    if not pts:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (max(0.0, min(xs)), max(0.0, min(ys)), min(1.0, max(xs)), min(1.0, max(ys)))


def _track_boxes(track: Any) -> list[tuple[float, tuple[float, float, float, float]]]:
    boxes = getattr(track, "boxes", None)
    if boxes is None and isinstance(track, Mapping):
        boxes = track.get("boxes")
    out = []
    for item in boxes or ():
        try:
            t, b = item[0], item[1]
            out.append((float(t), (float(b[0]), float(b[1]), float(b[2]), float(b[3]))))
        except (TypeError, ValueError, IndexError):
            continue
    out.sort(key=lambda x: x[0])
    return out


def track_box_at(track: Any, t: float, tol_s: float = 1.0) -> Optional[tuple[float, float, float, float]]:
    """Бокс трека на момент ``t``: линейная интерполяция между соседями или ближайший в пределах ``tol_s``."""
    boxes = _track_boxes(track)
    if not boxes:
        return None
    ts = [b[0] for b in boxes]
    i = int(np.searchsorted(ts, t))
    if 0 < i < len(boxes):
        (t0, b0), (t1, b1) = boxes[i - 1], boxes[i]
        if t1 - t0 <= 2 * tol_s and t0 <= t <= t1:
            a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return tuple(float(b0[k] + a * (b1[k] - b0[k])) for k in range(4))  # type: ignore[return-value]
    best = min(boxes, key=lambda b: abs(b[0] - t))
    if abs(best[0] - t) <= tol_s:
        return best[1]
    return None


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def derive_from_tracks(
    times: np.ndarray,
    tracks: Sequence[Any],
    zones: Sequence[Any],
    *,
    presence_tol_s: float = 0.6,
    desk_margin: float = 0.08,
) -> dict[str, np.ndarray]:
    """Грубые сигналы из треков/зон (используются только при отсутствии сигнала в states)."""
    n = int(times.shape[0])
    if n == 0 or not tracks:
        return {}
    zone_polys: dict[str, list] = {}
    for z in zones or ():
        name = str(_attr(z, "name", ""))
        poly = _attr(z, "polygon", None)
        if name and poly:
            zone_polys[name] = list(poly)

    def in_zone(box: Sequence[float], zname: str, margin: float = 0.0) -> bool:
        poly = zone_polys.get(zname)
        if not poly:
            return False
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        if point_in_polygon(cx, cy, poly):
            return True
        if margin > 0:
            bb = polygon_bbox(poly)
            if bb and bb[0] - margin <= cx <= bb[2] + margin and bb[1] - margin <= cy <= bb[3] + margin:
                return True
        return False

    out: dict[str, np.ndarray] = {}
    n_persons = np.zeros(n)
    cashier = np.zeros(n)
    cashier_wp = np.zeros(n)
    collector = np.zeros(n)
    outsider = np.zeros(n)
    obj_presence: dict[str, np.ndarray] = {}
    desk_bag = np.zeros(n) if "desk" in zone_polys else None  # личная сумка (label bag) на столе — V09
    # клиент у окна со своей стороны: односторонний сигнал — 1, где клиент виден в client_area, иначе
    # неизвестно (NaN): «трекер клиента не видит» ≠ «клиента нет» (окно может быть вне кадра)
    client = np.full(n, np.nan) if "client_area" in zone_polys else None
    has_person = False
    workplace_zone = "desk" if "desk" in zone_polys else ("cashier_area" if "cashier_area" in zone_polys else None)
    for tr in tracks:
        label = str(_attr(tr, "label", ""))
        role = str(_attr(tr, "role", "unknown") or "unknown")
        role_conf = float(_attr(tr, "role_conf", 0.0) or 0.0)
        boxes = _track_boxes(tr)
        if not boxes:
            continue
        bt = np.asarray([b[0] for b in boxes])
        idx = np.clip(np.searchsorted(bt, times), 0, len(bt) - 1)
        idx_prev = np.clip(idx - 1, 0, len(bt) - 1)
        use_prev = np.abs(bt[idx_prev] - times) < np.abs(bt[idx] - times)
        nearest = np.where(use_prev, idx_prev, idx)
        # присутствие — только в пределах жизни трека (без растягивания до первого/после последнего
        # бокса: иначе начало действия «съезжает» раньше на presence_tol_s); внутри — разрывы ≤ tol
        alive = (times >= bt[0] - 1e-6) & (times <= bt[-1] + 1e-6)
        present = alive & (np.abs(bt[nearest] - times) <= presence_tol_s)
        if label == "person":
            has_person = True
            n_persons += present
            conf = role_conf if role_conf > 0 else 0.7
            for i in np.nonzero(present)[0]:
                box = boxes[int(nearest[i])][1]
                if role == "cashier":
                    cashier[i] = max(cashier[i], conf)
                    margin = desk_margin if workplace_zone == "desk" else 0.0
                    if workplace_zone and in_zone(box, workplace_zone, margin):
                        cashier_wp[i] = max(cashier_wp[i], conf)
                elif role == "collector":
                    collector[i] = max(collector[i], conf)
                elif role == "visitor" and in_zone(box, "cashier_area"):
                    outsider[i] = max(outsider[i], conf)
                if role in ("visitor", "unknown") and client is not None and in_zone(box, "client_area") \
                        and not in_zone(box, "cashier_area"):
                    client[i] = conf if np.isnan(client[i]) else max(client[i], conf)
        else:
            arr = obj_presence.setdefault(label, np.zeros(n))
            np.maximum(arr, present.astype(float), out=arr)
            if label == "bag" and desk_bag is not None:
                for i in np.nonzero(present)[0]:
                    if in_zone(boxes[int(nearest[i])][1], "desk", desk_margin):
                        desk_bag[i] = 1.0
    if has_person:
        out["n_persons"] = n_persons
        out["cashier_present"] = cashier
        out["collector_present"] = collector
        if workplace_zone:
            out["cashier_in_workplace"] = cashier_wp
        if "cashier_area" in zone_polys:
            out["outsider_in_cashier_area"] = outsider
    label_to_signal = {"safe_open": "safe_open", "door_open": "door_open", "cash": "cash_visible",
                       "cash_bag": "cash_bag_visible"}
    for label, sig in label_to_signal.items():
        if label in obj_presence:
            out[sig] = obj_presence[label]
    if desk_bag is not None:
        out["personal_items_on_desk"] = desk_bag
    if client is not None and has_person:
        out["client_present"] = client
    return out
