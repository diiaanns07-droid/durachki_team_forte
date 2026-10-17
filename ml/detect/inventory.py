"""desk_inventory / safe_door_state — состояние стола и сейфа в окне вокруг t.

Вход для правила «ушла и оставила» (A09): после ухода кассира — что осталось на столе и открыт ли сейф.

Принципы:
- unknown ≠ «нет». Если предмет не найден, state="unknown": детектор не утверждает отсутствие.
  Для сейфа «closed» — только при положительном свидетельстве закрытой дверцы.
- Обычная сумка или книга не становится «инкассаторской сумкой». cash_bag выдаётся только от модели,
  дообученной на проверенной разметке, или (mode=rare_open_vocab) при контрастной проверке
  «инкассаторская сумка» > «обычная сумка» минимум на min_frames_present кадрах. Иначе это bag —
  сумка неизвестного типа.
- Мобильный ≠ стационарный. В rare_open_vocab сравниваются «mobile phone» и «desk telephone»; если
  перевешивает стационарный, детекция phone гасится.
- Никаких пиксельных координат, цвета или яркости: только нормированные зоны A06 и детекции моделей.

Абляция (ровно одна): inventory.mode = baseline | rare_open_vocab (по умолчанию baseline).
"""
from __future__ import annotations

import threading
import weakref
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

try:
    from contracts.sentinel_contracts import BBox  # noqa
except ImportError:  # пока ветка A01 не смерджена
    from contracts.sentinel_contracts import BBox  # noqa

from .boxes import RawDets, iou_matrix
from .config import label_threshold, load_config

KINDS = ("cash", "cash_bag", "cup", "food", "phone", "personal_item", "bag")
OWL_ONLY = ("cash", "cash_bag", "personal_item")   # в baseline (COCO) не наблюдаемы
GROUPS = ("cash", "cash_bag", "generic_bag", "personal_item", "phone", "landline", "cup", "food",
          "safe_open", "safe_closed")


@dataclass
class ItemState:
    kind: str
    state: str                      # "present" | "unknown"
    conf: float = 0.0
    t: Optional[float] = None       # время кадра с лучшим свидетельством
    bbox: Optional[BBox] = None
    frames_seen: int = 0
    frames_checked: int = 0
    sources: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class DeskInventory:
    t: float
    window: tuple[float, float]
    mode: str
    zone: Optional[str]                     # "desk+cash_tray" | "desk" | None (зоны нет → весь кадр)
    items: dict[str, ItemState]
    person_occlusion: float = 0.0           # средняя доля зоны стола, закрытая людьми
    frames_checked: int = 0
    warnings: list[str] = field(default_factory=list)

    def present(self, kind: str) -> bool:
        return self.items.get(kind, ItemState(kind, "unknown")).state == "present"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SafeDoorState:
    t: float
    window: tuple[float, float]
    mode: str
    zone: Optional[str]
    state: str                               # "open" | "closed" | "unknown"
    conf: float = 0.0
    bbox: Optional[BBox] = None
    votes: dict[str, int] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------- зоны (нормированные полигоны A06)

def zone_polys(zones: Any, names: Sequence[str]) -> list[np.ndarray]:
    """Полигоны нужных зон из list[Zone] | list[dict] | {name: polygon | [polygons]}."""
    if not zones:
        return []
    out: list[np.ndarray] = []

    def add(poly: Any) -> None:
        arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        if len(arr) >= 3:
            out.append(arr)

    if isinstance(zones, Mapping):
        for n in names:
            v = zones.get(n)
            if v is None:
                continue
            if len(v) and isinstance(v[0], (list, tuple)) and len(v[0]) and isinstance(v[0][0], (list, tuple)):
                for p in v:
                    add(p)
            else:
                add(v)
        return out
    for z in zones:
        name = z.get("name") if isinstance(z, Mapping) else getattr(z, "name", None)
        poly = z.get("polygon") if isinstance(z, Mapping) else getattr(z, "polygon", None)
        if name in names and poly is not None:
            add(poly)
    return out


def _inside(px: np.ndarray, py: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Векторный ray casting: точки (px, py) внутри полигона."""
    x, y = poly[:, 0], poly[:, 1]
    xj, yj = np.roll(x, 1), np.roll(y, 1)
    inside = np.zeros(px.shape, dtype=bool)
    for k in range(len(poly)):
        cond = (y[k] > py) != (yj[k] > py)
        denom = np.where(np.abs(yj[k] - y[k]) < 1e-12, 1e-12, yj[k] - y[k])
        xin = (xj[k] - x[k]) * (py - y[k]) / denom + x[k]
        inside ^= cond & (px < xin)
    return inside


def _inside_any(px: np.ndarray, py: np.ndarray, polys: Sequence[np.ndarray]) -> np.ndarray:
    m = np.zeros(px.shape, dtype=bool)
    for p in polys:
        m |= _inside(px, py, p)
    return m


def box_zone_fraction(boxes: np.ndarray, polys: Sequence[np.ndarray], k: int = 7) -> np.ndarray:
    """Доля площади каждого бокса внутри объединения полигонов (сетка k×k точек). Без зон → 1."""
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    if not polys:
        return np.ones(len(boxes), np.float32)
    if len(boxes) == 0:
        return np.zeros(0, np.float32)
    g = (np.arange(k, dtype=np.float32) + 0.5) / k
    px = boxes[:, 0:1, None] + (boxes[:, 2:3, None] - boxes[:, 0:1, None]) * g[None, None, :]   # (N,1,k)
    py = boxes[:, 1:2, None] + (boxes[:, 3:4, None] - boxes[:, 1:2, None]) * g[None, :, None]   # (N,k,1)
    px, py = np.broadcast_arrays(px, py)                                                          # (N,k,k)
    return _inside_any(px, py, polys).reshape(len(boxes), -1).mean(axis=1).astype(np.float32)


def zone_occlusion(polys: Sequence[np.ndarray], person_boxes: np.ndarray, k: int = 24) -> float:
    """Доля площади зоны, закрытая боксами людей (для предупреждения «отсутствие не проверяемо»)."""
    pb = np.asarray(person_boxes, np.float32).reshape(-1, 4)
    if not polys or len(pb) == 0:
        return 0.0
    allp = np.concatenate(polys)
    x0, y0 = allp.min(0)
    x1, y1 = allp.max(0)
    g = (np.arange(k, dtype=np.float32) + 0.5) / k
    px, py = np.meshgrid(x0 + (x1 - x0) * g, y0 + (y1 - y0) * g)
    inz = _inside_any(px, py, polys)
    if not inz.any():
        return 0.0
    cov = np.zeros_like(inz)
    for b in pb:
        cov |= (px >= b[0]) & (px <= b[2]) & (py >= b[1]) & (py <= b[3])
    return float((cov & inz).sum() / inz.sum())


# ---------------------------------------------------------------- кадры окна и кэш

def window_times(clip: Any, t: float, window_s: float, n: int) -> list[float]:
    dur = float(getattr(clip, "duration", 0.0) or 0.0)
    times = list(getattr(clip, "times", []) or [])
    hi = dur if dur > 0 else (float(times[-1]) if times else float(t))
    if n <= 1 or window_s <= 0:
        ts = [float(t)]
    else:
        ts = list(np.linspace(t - window_s / 2.0, t + window_s / 2.0, int(n)))
    out: list[float] = []
    for x in ts:
        x = float(min(max(x, 0.0), hi))
        if not out or abs(out[-1] - x) > 1e-6:
            out.append(round(x, 4))
    return out


def get_frame(clip: Any, t: float) -> Optional[np.ndarray]:
    try:
        if hasattr(clip, "frame_at"):
            f = clip.frame_at(t)
        else:
            times = list(clip.times)
            f = clip.frame(int(np.argmin([abs(x - t) for x in times])))
    except Exception:  # noqa: BLE001 — битый кадр = нет свидетельства, а не ошибка
        return None
    if f is None or not hasattr(f, "shape") or f.ndim != 3 or f.size == 0:
        return None
    return np.ascontiguousarray(f)


class _LRU:
    """Кэш результатов по (клип, t). Запись валидна, только пока жив ТОТ ЖЕ объект клипа (weakref):
    id() освобождённого клипа может достаться новому видео — без проверки это дало бы чужие детекции."""

    def __init__(self, cap: int = 256):
        self.cap = cap
        self.d: OrderedDict[tuple, tuple[Any, Any]] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: tuple, owner: Any) -> Any:
        with self.lock:
            hit = self.d.get(key)
            if hit is None:
                return None
            ref, val = hit
            if ref is None or ref() is not owner:
                del self.d[key]
                return None
            self.d.move_to_end(key)
            return val

    def put(self, key: tuple, owner: Any, val: Any) -> None:
        try:
            ref = weakref.ref(owner)
        except TypeError:   # объект без поддержки weakref — не кэшируем
            return
        with self.lock:
            self.d[key] = (ref, val)
            self.d.move_to_end(key)
            while len(self.d) > self.cap:
                self.d.popitem(last=False)


_CACHE = _LRU()


def clear_cache() -> None:
    with _CACHE.lock:
        _CACHE.d.clear()


def _clip_key(clip: Any) -> tuple:
    return (str(getattr(clip, "path", "")), id(clip))


def _closed_dets(clip: Any, t: float, frame: np.ndarray, closed: Sequence[Any]) -> RawDets:
    key = ("closed", _clip_key(clip), round(t, 4), tuple(getattr(b, "name", "") for b in closed))
    hit = _CACHE.get(key, clip)
    if hit is not None:
        return hit
    d = RawDets.concat([b.predict([frame])[0] for b in closed])
    _CACHE.put(key, clip, d)
    return d


def _owl_scores(clip: Any, t: float, frame: np.ndarray, scorer: Any) -> tuple[np.ndarray, np.ndarray]:
    key = ("owl", _clip_key(clip), round(t, 4), getattr(scorer, "name", ""), tuple(scorer.texts))
    hit = _CACHE.get(key, clip)
    if hit is not None:
        return hit
    r = scorer.score([frame])[0]
    _CACHE.put(key, clip, r)
    return r


def _group_scores(probs: np.ndarray, texts: Sequence[str], queries: Mapping[str, Sequence[str]]) -> dict[str, np.ndarray]:
    idx = {tx: i for i, tx in enumerate(texts)}
    out = {}
    for g in GROUPS:
        cols = [idx[q] for q in queries.get(g, []) if q in idx]
        out[g] = probs[:, cols].max(axis=1) if cols and len(probs) else np.zeros(len(probs), np.float32)
    return out


# ---------------------------------------------------------------- бэкенды

def _setup(config: Optional[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], list[Any], Any, list[str]]:
    from .backends import get_backends, get_inventory_scorer

    cfg = load_config(config)
    inv = cfg["inventory"]
    be = get_backends(cfg)   # DetectorUnavailable пробрасывается: production сообщает unavailable
    warnings: list[str] = list(be.warnings)
    scorer = None
    if inv.get("mode", "baseline") == "rare_open_vocab":
        scorer = get_inventory_scorer(cfg, warnings)
    return cfg, inv, list(be.closed), scorer, warnings


def _effective_mode(inv: Mapping[str, Any], scorer: Any) -> str:
    return "rare_open_vocab" if inv.get("mode") == "rare_open_vocab" and scorer is not None else "baseline"


def _owl_frame_times(ts: Sequence[float], t: float, max_frames: int) -> set[float]:
    return set(sorted(ts, key=lambda x: (abs(x - t), x))[: max(0, int(max_frames))])


# ---------------------------------------------------------------- desk_inventory

def _offer(frame_best: dict[str, tuple[float, tuple, str]], kind: str, score: float, box: Any, src: str) -> None:
    """Лучшее свидетельство вида на кадре (max score)."""
    if kind not in frame_best or score > frame_best[kind][0]:
        frame_best[kind] = (float(score), tuple(round(float(v), 5) for v in box), src)


def _pick(frame_best: dict[str, tuple[float, tuple, str]], mask: np.ndarray, inz: np.ndarray, sc: np.ndarray,
          boxes: np.ndarray, kind: str, src: str) -> None:
    """Лучший OWL-патч в зоне, прошедший условие mask, → свидетельство вида kind."""
    m = mask & inz
    if m.any():
        j = int(np.argmax(np.where(m, sc, -1.0)))
        _offer(frame_best, kind, sc[j], boxes[j], src)


def desk_inventory(clip: Any, t: float, zones: Any, *, config: Optional[dict[str, Any]] = None) -> DeskInventory:
    """Что лежит на кассовом столе (зоны desk ∪ cash_tray) в окне вокруг t: present / unknown по видам KINDS."""
    from .pipeline import _RUN_LOCK

    with _RUN_LOCK:
        return _desk_inventory(clip, float(t), zones, config)


def _desk_inventory(clip: Any, t: float, zones: Any, config: Optional[dict[str, Any]]) -> DeskInventory:
    cfg, inv, closed, scorer, warnings = _setup(config)
    mode = _effective_mode(inv, scorer)
    polys = zone_polys(zones, inv.get("desk_zones", ["desk", "cash_tray"]))
    zone_name = "+".join(n for n in inv.get("desk_zones", []) if zone_polys(zones, [n])) or None
    if not polys:
        warnings.append("inventory: зона стола не задана — проверен весь кадр; «на столе» не подтверждено")
    ts = window_times(clip, t, float(inv.get("window_s", 1.0)), int(inv.get("n_frames", 3)))
    owl_ts = _owl_frame_times(ts, t, inv["rare_open_vocab"].get("max_frames", 2)) if mode == "rare_open_vocab" else set()
    thr = inv.get("thr", {})
    rov = inv["rare_open_vocab"]
    rthr, margin = rov.get("thr", {}), float(rov.get("margin", 0.05))
    min_ov = float(inv.get("zone_overlap", 0.5))
    from_labels, from_ft = inv.get("from_labels", {}), inv.get("from_finetuned", {})
    person_thr = label_threshold(cfg["closed_set"].get("score_thr", {}), "person")

    best: dict[str, list[tuple[float, float, tuple, str]]] = {k: [] for k in KINDS}  # (score, t, bbox, source)
    checked = {k: 0 for k in KINDS}
    notes: dict[str, list[str]] = {k: [] for k in KINDS}
    occl: list[float] = []
    phone_cands_by_t: dict[float, list[tuple[float, np.ndarray, str]]] = {}
    owl_phone_by_t: dict[float, tuple[float, tuple, str]] = {}
    landline_all: list[np.ndarray] = []
    n_ok = 0
    has_ft = any(str(getattr(b, "name", "")).startswith("finetuned") for b in closed)
    for tk in ts:
        frame = get_frame(clip, tk)
        if frame is None:
            warnings.append(f"inventory: кадр t={tk:.2f}s не декодирован — пропущен")
            continue
        n_ok += 1
        frame_best: dict[str, tuple[float, tuple, str]] = {}

        d = _closed_dets(clip, tk, frame, closed) if closed else RawDets()
        frac = box_zone_fraction(d.boxes, polys)
        persons = [i for i, lab in enumerate(d.labels) if lab == "person" and d.scores[i] >= person_thr]
        occl.append(zone_occlusion(polys, d.boxes[persons]) if persons else 0.0)
        phone_cands: list[tuple[float, np.ndarray, str]] = []
        for i, lab in enumerate(d.labels):
            src = d.sources[i]
            kind = from_ft.get(lab) if src.startswith("finetuned") else None
            if kind is None and lab in from_labels:
                kind = from_labels[lab]
            if kind is None or kind not in KINDS or frac[i] < min_ov:
                continue
            if d.scores[i] < float(thr.get(kind, 0.35)):
                continue
            if kind == "phone":
                phone_cands.append((float(d.scores[i]), d.boxes[i], src))
                continue
            _offer(frame_best, kind, d.scores[i], d.boxes[i], src)
        for k in ("cup", "food", "phone", "bag"):
            checked[k] += 1
        if has_ft:
            checked["cash"] += 1
            checked["cash_bag"] += 1

        phone_cands_by_t[tk] = phone_cands
        if tk in owl_ts:
            boxes, probs = _owl_scores(clip, tk, frame, scorer)
            G = _group_scores(probs, scorer.texts, rov.get("queries", {}))
            inz = box_zone_fraction(boxes, polys) >= min_ov if len(boxes) else np.zeros(0, bool)
            src = getattr(scorer, "name", "owlv2")

            _pick(frame_best, G["cash"] >= rthr.get("cash", 0.25), inz, G["cash"], boxes, "cash", f"{src}:cash")
            confirmed = (G["cash_bag"] >= rthr.get("cash_bag", 0.3)) & (G["cash_bag"] - G["generic_bag"] >= margin)
            _pick(frame_best, confirmed, inz, G["cash_bag"], boxes, "cash_bag", f"{src}:cash_bag>generic_bag")
            unconfirmed = (G["cash_bag"] >= rthr.get("cash_bag", 0.3)) & ~confirmed
            if (unconfirmed & inz).any():
                notes["cash_bag"].append("есть сумка, похожая на инкассаторскую, но контраст с обычной сумкой не пройден")
                _pick(frame_best, unconfirmed, inz, G["cash_bag"], boxes, "bag", f"{src}:bag?cash_bag_unconfirmed")
            generic = (G["generic_bag"] >= rthr.get("generic_bag", 0.3)) & (G["generic_bag"] - G["cash_bag"] >= margin)
            _pick(frame_best, generic, inz, G["generic_bag"], boxes, "personal_item", f"{src}:generic_bag")
            _pick(frame_best, generic, inz, G["generic_bag"], boxes, "bag", f"{src}:generic_bag")
            _pick(frame_best, G["personal_item"] >= rthr.get("personal_item", 0.3), inz, G["personal_item"], boxes,
                  "personal_item", f"{src}:jacket/coat")
            mobile = (G["phone"] >= rthr.get("phone", 0.25)) & (G["phone"] - G["landline"] >= margin) & inz
            if mobile.any():
                j = int(np.argmax(np.where(mobile, G["phone"], -1.0)))
                owl_phone_by_t[tk] = (float(G["phone"][j]), tuple(round(float(v), 5) for v in boxes[j]),
                                      f"{src}:mobile>landline")
            land = (G["landline"] >= rthr.get("landline", 0.25)) & (G["landline"] - G["phone"] >= margin) & inz
            if land.any():
                landline_all.append(boxes[land])
            _pick(frame_best, G["cup"] >= rthr.get("cup", 0.25), inz, G["cup"], boxes, "cup", f"{src}:cup")
            _pick(frame_best, G["food"] >= rthr.get("food", 0.3), inz, G["food"], boxes, "food", f"{src}:food")
            for k in OWL_ONLY:
                checked[k] += 1
        for kind, (sc, box, src) in frame_best.items():
            best[kind].append((sc, tk, box, src))

    # телефон — после всех кадров: стационарный телефон, найденный OWL хоть на одном кадре окна, гасит COCO
    # «cell phone» в том же месте на ВСЕХ кадрах окна (стационарный телефон неподвижен)
    landline = np.concatenate(landline_all) if landline_all else np.zeros((0, 4), np.float32)
    for tk, cands in phone_cands_by_t.items():
        fb: Optional[tuple[float, tuple, str]] = owl_phone_by_t.get(tk)
        for sc, box, src in cands:
            if len(landline) and iou_matrix(box[None], landline).max() >= 0.3:
                notes["phone"].append("на месте детекции «cell phone» перевешивает «стационарный телефон» — погашено")
                continue
            if fb is None or sc > fb[0]:
                fb = (float(sc), tuple(round(float(v), 5) for v in box), src)
        if fb is not None:
            best["phone"].append((fb[0], tk, fb[1], fb[2]))

    need_default = max(1, int(inv.get("min_frames_present", 2)))
    items: dict[str, ItemState] = {}
    for k in KINDS:
        hits, n_chk = best[k], checked[k]
        if n_chk == 0:
            why = ("не наблюдаемо в режиме baseline: в COCO нет такого класса (нужен rare_open_vocab "
                   "или модель, дообученная на проверенной разметке)") if k in OWL_ONLY else "нет проверенных кадров"
            items[k] = ItemState(k, "unknown", note=why)
            continue
        need = min(need_default, n_chk)
        seen = len(hits)
        note = "; ".join(dict.fromkeys(notes[k]))
        if seen >= need:
            sc, tb, box, _src = max(hits, key=lambda h: h[0])
            conf = float(np.mean([h[0] for h in hits])) * seen / n_chk
            items[k] = ItemState(k, "present", round(conf, 4), tb, box, seen, n_chk,
                                 sorted({h[3] for h in hits}), note)
        else:
            extra = f"найдено на {seen}/{n_chk} кадрах — недостаточно (нужно {need})" if seen else "не найдено"
            items[k] = ItemState(k, "unknown", 0.0, None, None, seen, n_chk, sorted({h[3] for h in hits}),
                                 "; ".join(x for x in (extra, note) if x))
    if items["bag"].state == "present" and items["cash_bag"].state != "present":
        items["bag"].note = "; ".join(x for x in (items["bag"].note, "тип сумки не подтверждён (не считать инкассаторской)") if x)
    if items["phone"].state == "present" and mode == "baseline":
        items["phone"].note = "; ".join(x for x in (items["phone"].note, "мобильный/стационарный в baseline не различаются") if x)
    po = float(np.mean(occl)) if occl else 0.0
    if po >= float(inv.get("occlusion_warn", 0.6)):
        warnings.append(f"inventory: зона стола закрыта людьми на {po:.0%} — отсутствие предметов не проверяемо")
    if n_ok == 0:
        warnings.append("inventory: ни один кадр окна не декодирован — всё unknown")
    return DeskInventory(t=float(t), window=(ts[0], ts[-1]), mode=mode, zone=zone_name, items=items,
                         person_occlusion=round(po, 4), frames_checked=n_ok, warnings=warnings)


# ---------------------------------------------------------------- safe_door_state

def safe_door_state(clip: Any, t: float, zones: Any, *, config: Optional[dict[str, Any]] = None) -> SafeDoorState:
    """Дверца сейфа в окне вокруг t: open / closed / unknown (closed — только при положительном свидетельстве)."""
    from .pipeline import _RUN_LOCK

    with _RUN_LOCK:
        return _safe_door_state(clip, float(t), zones, config)


def _safe_door_state(clip: Any, t: float, zones: Any, config: Optional[dict[str, Any]]) -> SafeDoorState:
    cfg, inv, closed, scorer, warnings = _setup(config)
    mode = _effective_mode(inv, scorer)
    polys = zone_polys(zones, inv.get("safe_zones", ["safe"]))
    zone_name = "safe" if polys else None
    if not polys:
        warnings.append("safe: зона сейфа не задана — проверен весь кадр")
    ts = window_times(clip, t, float(inv.get("window_s", 1.0)), int(inv.get("n_frames", 3)))
    owl_ts = _owl_frame_times(ts, t, inv["rare_open_vocab"].get("max_frames", 2)) if mode == "rare_open_vocab" else set()
    rov = inv["rare_open_vocab"]
    rthr, margin = rov.get("thr", {}), float(rov.get("margin", 0.05))
    min_ov = float(inv.get("zone_overlap", 0.5))
    ft = [b for b in closed if str(getattr(b, "name", "")).startswith("finetuned")]
    thr_ft = float(inv.get("thr", {}).get("safe_open", 0.35))
    evidence: list[dict[str, Any]] = []
    for tk in ts:
        if not ft and tk not in owl_ts:
            continue
        frame = get_frame(clip, tk)
        if frame is None:
            warnings.append(f"safe: кадр t={tk:.2f}s не декодирован — пропущен")
            continue
        ev: dict[str, Any] = {"t": tk, "state": "unknown", "open": 0.0, "closed": 0.0, "bbox": None, "source": ""}
        if ft:
            d = _closed_dets(clip, tk, frame, ft)
            frac = box_zone_fraction(d.boxes, polys)
            cand = [i for i, lab in enumerate(d.labels) if lab == "safe_open" and frac[i] >= min_ov and d.scores[i] >= thr_ft]
            if cand:
                i = max(cand, key=lambda j: d.scores[j])
                ev.update(state="open", open=float(d.scores[i]), bbox=tuple(float(v) for v in d.boxes[i]), source=d.sources[i])
        if tk in owl_ts and ev["state"] == "unknown":
            boxes, probs = _owl_scores(clip, tk, frame, scorer)
            G = _group_scores(probs, scorer.texts, rov.get("queries", {}))
            inz = box_zone_fraction(boxes, polys) >= min_ov if len(boxes) else np.zeros(0, bool)
            if inz.any():
                so, scl = G["safe_open"], G["safe_closed"]
                j = int(np.argmax(np.where(inz, np.maximum(so, scl), -1.0)))
                ev.update(open=float(so[j]), closed=float(scl[j]), bbox=tuple(round(float(v), 5) for v in boxes[j]),
                          source=getattr(scorer, "name", "owlv2"))
                if so[j] >= rthr.get("safe_open", 0.2) and so[j] - scl[j] >= margin:
                    ev["state"] = "open"
                elif scl[j] >= rthr.get("safe_closed", 0.2) and scl[j] - so[j] >= margin:
                    ev["state"] = "closed"
        evidence.append(ev)
    votes = {s: sum(1 for e in evidence if e["state"] == s) for s in ("open", "closed", "unknown")}
    n_chk = len(evidence)
    state, conf, bbox = "unknown", 0.0, None
    if n_chk == 0:
        warnings.append("safe: в режиме baseline состояние сейфа не наблюдаемо (нет класса в COCO) — "
                        "нужен rare_open_vocab или модель, дообученная на проверенной разметке (safe_open)")
    else:
        need = min(max(1, int(inv.get("min_frames_present", 2))), n_chk)
        for s in ("open", "closed"):
            other = "closed" if s == "open" else "open"
            if votes[s] >= need and votes[s] > votes[other]:
                win = [e for e in evidence if e["state"] == s]
                state = s
                conf = round(float(np.mean([e[s] for e in win])) * votes[s] / n_chk, 4)
                bbox = max(win, key=lambda e: e[s])["bbox"]
        if state == "unknown" and votes["open"] and votes["closed"]:
            warnings.append("safe: противоречивые кадры (open и closed) — unknown")
    return SafeDoorState(t=float(t), window=(ts[0], ts[-1]), mode=mode, zone=zone_name, state=state, conf=conf,
                         bbox=bbox, votes=votes, evidence=evidence, warnings=warnings)


def as_jsonable(x: Any) -> Any:
    """DeskInventory/SafeDoorState → dict (для логов/кэша A03)."""
    return x.as_dict() if hasattr(x, "as_dict") else x


__all__: Iterable[str] = ("KINDS", "ItemState", "DeskInventory", "SafeDoorState", "desk_inventory",
                          "safe_door_state", "zone_polys", "box_zone_fraction", "zone_occlusion", "clear_cache")
