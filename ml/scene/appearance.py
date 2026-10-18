"""Двухсостоятельная модель вида ROI (сейф / дверь) без допущения «первый кадр = закрыто».

1. Эталон A — маскированная медиана первых N чистых кадров (без людей у ROI, хорошее качество).
   Его состояние НЕИЗВЕСТНО: это просто «как выглядело в начале».
2. Изменение вида d(t) = 1 − CS-SSIM(ROI_t, A) на z-нормированных кропах (инвариантно к общему
   освещению/экспозиции; люди маскируются, сильная окклюзия → кадр не используется).
3. Если есть устойчивый (≥ min_state_s) и внутренне согласованный альтернативный вид B —
   считаем, что объект менял состояние. Короткие/несогласованные изменения (тень, проход, шум)
   состоянием не считаются.
4. Какое из состояний «открыто», решают абсолютные свидетельства: детекции safe_open/door_open
   против safe/door в кадрах каждого состояния, проходы людей через дверь, и слабый априор вида
   (у открытого сейфа больше тёмного проёма/кромок, дверца выступает за контур).
   Нет свидетельств → уверенность 0 (неизвестно), а не «закрыто».
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from . import roi as R
from .geometry import BBox, expand_bbox
from .smoothing import logit, sigmoid


@dataclass
class AppearanceResult:
    p_open: np.ndarray                 # покадровая P(open) от вида+детекций (до HMM)
    conf: np.ndarray                   # 0..1 сила свидетельства
    visibility: np.ndarray             # 1 − окклюзия ROI
    change: np.ndarray                 # d(t) относительно эталона A (nan — кадр не использован)
    state_b: np.ndarray                # P(вид B) по кадрам
    p_a_open: float = 0.5              # P(эталон A = открыто)
    label_conf: float = 0.0            # |2·P − 1| — насколько понятно, какое состояние «открыто»
    has_two_states: bool = False
    roi: Optional[BBox] = None
    provenance: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)


def roi_boxes(roi_box: BBox, cfg_roi: dict[str, Any]) -> tuple[BBox, BBox]:
    """Внутренний ROI (объект + margin) и внешний (кольцо для «кромки вне контура»)."""
    return expand_bbox(roi_box, float(cfg_roi["margin"])), expand_bbox(roi_box, float(cfg_roi["ring"]))


def extract_u8(frame: np.ndarray, roi_box: BBox, shift: tuple[float, float], cfg_roi: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Сжатые серые кропы (uint8, size×size, размытие 3×3) — всё, что нужно анализу вида; кадр читается один раз."""
    import cv2

    size = int(cfg_roi["size"])
    inner, outer = roi_boxes(roi_box, cfg_roi)
    ci = cv2.GaussianBlur(R.to_gray(R.crop(frame, inner, shift, (size, size))), (3, 3), 0)
    co = cv2.GaussianBlur(R.to_gray(R.crop(frame, outer, shift, (size, size))), (3, 3), 0)
    return np.clip(ci, 0, 255).astype(np.uint8), np.clip(co, 0, 255).astype(np.uint8)


class _Crops:
    """Ленивый доступ к подготовленным кропам: z-норма и маски людей считаются по запросу (экономия памяти)."""

    def __init__(self, store: dict[int, tuple[np.ndarray, np.ndarray]], inner: BBox, outer: BBox, person_boxes: list[list[BBox]], size: int, pad: float) -> None:
        self.store, self.inner, self.outer, self.boxes, self.size, self.pad = store, inner, outer, person_boxes, size, pad
        self.keys: set[int] = set()
        self._ring = R.ring_mask(outer, inner, (size, size))

    def get(self, i: int) -> Optional[tuple[np.ndarray, ...]]:
        c = self.store.get(i)
        if c is None:
            return None
        ci, co = c[0].astype(np.float32), c[1].astype(np.float32)
        wh = (self.size, self.size)
        mi = R.mask_from_boxes(self.inner, self.boxes[i], wh, self.pad)
        mo = R.mask_from_boxes(self.outer, self.boxes[i], wh, self.pad) * self._ring
        return R.znorm(ci, mi), mi, R.znorm(co, mo), mo, ci, co

    def __contains__(self, i: int) -> bool:
        return i in self.keys

    def __getitem__(self, i: int) -> tuple[np.ndarray, ...]:
        c = self.get(i)
        if c is None:
            raise KeyError(i)
        return c

    def items(self):  # noqa: ANN201
        for i in sorted(self.keys):
            yield i, self[i]


def analyze_roi(
    store: dict[int, tuple[np.ndarray, np.ndarray]],
    idxs: list[int],
    n: int,
    times: np.ndarray,
    roi_box: BBox,
    person_boxes: list[list[BBox]],
    quality: np.ndarray,
    det_logodds: np.ndarray,
    det_conf: np.ndarray,
    cfg_roi: dict[str, Any],
    cfg_obj: dict[str, Any],
    extra_open_votes: Optional[np.ndarray] = None,
    use_shape_prior: bool = True,
    naive_reference: bool = False,
    frame_fn: Optional[Callable[[int], Optional[np.ndarray]]] = None,
    shifts: Optional[list[tuple[float, float]]] = None,
    namer: Optional[Callable[[list[np.ndarray], str], list[float]]] = None,
    obj: str = "safe",
) -> AppearanceResult:
    """store — uint8-кропы (extract_u8) по индексам idxs (stride); результат — на все n сэмплов."""
    size = int(cfg_roi["size"])
    pad = float(cfg_roi["person_pad"])
    inner, outer = roi_boxes(roi_box, cfg_roi)
    occ_max = float(cfg_roi["occlusion_max"])

    vis = np.ones(n)
    for i in range(n):
        vis[i] = 1.0 - R.occlusion_fraction(inner, person_boxes[i], pad)

    # --- эталон A: первые N чистых кадров (сначала в первых ref_search_s секундах) ---
    ref_max_occ = float(cfg_roi["ref_max_occlusion"])
    n_ref = int(cfg_roi["ref_frames"])
    clean = [i for i in idxs if i in store and 1.0 - vis[i] <= ref_max_occ and quality[i] >= 0.4]
    early = [i for i in clean if times[i] <= float(cfg_roi["ref_search_s"])]
    ref_idx = (early if len(early) >= max(3, n_ref // 2) else clean)[:n_ref]

    crops = _Crops(store, inner, outer, person_boxes, size, pad)
    res = AppearanceResult(p_open=np.full(n, 0.5), conf=np.zeros(n), visibility=vis, change=np.full(n, np.nan), state_b=np.zeros(n), roi=roi_box)
    if len(ref_idx) < 3:
        res.info["reason"] = "no clean reference frames"
        res.provenance.append("no_reference")
        return _with_dets(res, det_logodds, det_conf, cfg_obj)

    ref_c = [crops.get(i) for i in ref_idx]
    ref_c = [c for c in ref_c if c is not None]
    if len(ref_c) < 3:
        res.info["reason"] = "frames unavailable"
        return _with_dets(res, det_logodds, det_conf, cfg_obj)
    ref_a = R.masked_median([c[0] for c in ref_c], [c[1] for c in ref_c])

    cur = crops
    d_a = np.full(n, np.nan)
    win = int(cfg_roi["ssim_win"])
    for i in idxs:
        if 1.0 - vis[i] > occ_max:
            continue
        c = crops.get(i)
        if c is None:
            continue
        cur.keys.add(i)
        d_a[i] = 1.0 - R.cs_ssim(c[0], ref_a, c[1], win)
    res.change = d_a

    center, scale = float(cfg_obj["roi_center"]), float(cfg_obj["roi_scale"])
    d_hi = center + scale
    min_q = float(cfg_obj.get("min_quality", 0.35))
    good = np.array([i in cur and np.isfinite(d_a[i]) and quality[i] >= min_q for i in range(n)])
    min_state_s = float(cfg_obj.get("min_state_s", 1.0))
    km = None
    if str(cfg_obj.get("cluster_mode", "ref_first")) == "kmeans2":
        # [H1] два кластера вида по ВСЕМУ ролику (симметрично, без «эталона начала»)
        km = _kmeans2(cur, [i for i in range(n) if good[i]], times, min_state_s, d_hi, win)
        if km is not None:
            ref_a = km["ref_a"]
            for i in list(cur.keys):
                c = cur[i]
                d_a[i] = 1.0 - R.cs_ssim(c[0], ref_a, c[1], win)
            good = np.array([i in cur and np.isfinite(d_a[i]) and quality[i] >= min_q for i in range(n)])
            res.info["cluster_mode"] = "kmeans2"
            res.info["kmeans_sep"] = km["sep"]

    # --- альтернативный вид B: устойчивые участки с d > d_hi ---
    b_cand = good & (np.nan_to_num(d_a, nan=0.0) > d_hi)
    b_idx = _long_runs(b_cand, times, min_state_s, max_gap_s=0.6)
    has_b = len(b_idx) >= 3
    ref_b = None
    if km is not None:
        has_b, ref_b = km["has_b"], km["ref_b"]
    elif has_b:
        b_c = [cur[i] for i in b_idx]
        ref_b = R.masked_median([c[0] for c in b_c], [c[1] for c in b_c])
        d_bb = np.array([1.0 - R.cs_ssim(c[0], ref_b, c[1], win) for c in b_c])
        del b_c
        if not np.isfinite(d_bb).any() or np.nanmedian(d_bb) > d_hi:
            has_b = False  # B несогласован (люди/шум/тени) — не считаем это состоянием
            res.provenance.append("inconsistent_change")
    res.has_two_states = bool(has_b)

    p_b = np.zeros(n)
    a_conf = np.zeros(n)
    for i in range(n):
        if not good[i]:
            continue
        da = d_a[i]
        if has_b and ref_b is not None:
            ci_ = cur[i]
            db = 1.0 - R.cs_ssim(ci_[0], ref_b, ci_[1], win)
            if not np.isfinite(db):
                continue
            p_b[i] = float(sigmoid((da - db) / max(scale, 1e-3)))
            close = min(da, db)
        else:
            p_b[i] = 0.0
            close = da
        fit = 1.0 if close < d_hi else 0.3   # ни на A, ни на B не похоже — слабое свидетельство
        a_conf[i] = fit * vis[i] * float(quality[i])
    res.state_b = p_b
    feats = {k: np.full(n, np.nan) for k in ("dark_rel", "edge", "ring_vert", "change_b")}
    rel = float(cfg_obj.get("dark_rel", 0.55))
    for i, c in cur.items():
        ci, mi, co, mo = c[4], c[1], c[5], c[3]
        ring_vals = co[mo > 0] if (mo > 0).sum() > 16 else co.ravel()
        vals = ci[mi > 0] if (mi > 0).sum() > 16 else ci.ravel()
        feats["dark_rel"][i] = float(np.mean(vals < rel * float(np.median(ring_vals))))
        feats["edge"][i] = R.edge_density(c[0], mi)
        feats["ring_vert"][i] = R.vertical_edge_energy(c[2], mo)
        if has_b and ref_b is not None:
            feats["change_b"][i] = 1.0 - R.cs_ssim(c[0], ref_b, mi, win)
    res.info["features"] = feats

    # --- какое состояние «открыто»: абсолютные свидетельства ---
    in_a = good & (p_b < 0.5)
    in_b = good & (p_b >= 0.5)
    lo_a = _state_det_logodds(det_logodds, det_conf, in_a)
    lo_b = _state_det_logodds(det_logodds, det_conf, in_b) if has_b else 0.0
    lo = lo_a - lo_b
    prov = []
    if lo_a != 0.0 or lo_b != 0.0:
        prov.append("det")
    if extra_open_votes is not None:
        va = float(np.sum(extra_open_votes[in_a])) if in_a.any() else 0.0
        vb = float(np.sum(extra_open_votes[in_b])) if has_b and in_b.any() else 0.0
        if va or vb:
            lo += float(np.clip(va - vb, -3.0, 3.0))
            prov.append("traffic")
    if has_b and use_shape_prior:
        prior = _shape_prior(cur, in_a, in_b, cfg_obj)
        if prior != 0.0:
            lo -= prior  # prior > 0 → B больше похоже на открытое → A закрыто
            prov.append("geom")
        res.info["shape_prior_b_open"] = prior
    if namer is not None and frame_fn is not None:
        lo_n = _name_clusters(namer, obj, frame_fn, shifts, inner, cur, d_a, p_b, in_a, in_b, has_b, ref_b, win, cfg_obj, res)
        if lo_n != 0.0:
            lo += lo_n
            prov.append("namer")
    p_a_open = float(sigmoid(lo))
    if naive_reference:  # АБЛЯЦИЯ: «первые кадры = закрыто» без подтверждения
        p_a_open = 0.0
        prov = ["naive_reference"]
    res.p_a_open = p_a_open
    res.label_conf = float(abs(2.0 * p_a_open - 1.0))
    p_app = p_a_open * (1.0 - p_b) + (1.0 - p_a_open) * p_b
    res.p_open = p_app
    res.conf = a_conf * res.label_conf
    res.provenance.extend(["roi"] + prov)
    res.info.update({"ref_idx": ref_idx, "n_state_b": int(in_b.sum()), "p_a_open": p_a_open})
    return _with_dets(res, det_logodds, det_conf, cfg_obj)


def _kmeans2(cur: Any, ids: list[int], times: np.ndarray, min_state_s: float, d_hi: float, win: int) -> Optional[dict[str, Any]]:
    """2-means по дескрипторам вида (z-норм ROI 16×16, косинус). A — кластер самого раннего кадра.
    Второй вид признаётся состоянием, только если эталоны кластеров различаются (1−CS-SSIM > d_hi)
    и у B есть устойчивые участки ≥ min_state_s; иначе — одно состояние (эталон — медиана всех кадров)."""
    import cv2

    if len(ids) < 6:
        return None
    step = max(1, len(ids) // 400)
    sel = ids[::step]
    X = []
    for i in sel:
        c = cur[i]
        z = (c[0] * c[1]).astype(np.float32)
        d = cv2.resize(z, (16, 16), interpolation=cv2.INTER_AREA).ravel()
        X.append(d / max(float(np.linalg.norm(d)), 1e-6))
    X = np.asarray(X)
    c0 = X[int(np.argmax(X @ X.mean(axis=0)))]
    c1 = X[int(np.argmin(X @ c0))]
    lab = np.zeros(len(X), bool)
    for _ in range(15):
        new = (X @ c1) > (X @ c0)
        if new.all() or not new.any():
            break
        lab = new
        c0 = X[~lab].mean(axis=0)
        c1 = X[lab].mean(axis=0)
        c0 /= max(float(np.linalg.norm(c0)), 1e-6)
        c1 /= max(float(np.linalg.norm(c1)), 1e-6)
    if lab[0]:
        lab = ~lab  # A — вид в начале (только имя, не «закрыто»)

    def med(members: list[int]) -> np.ndarray:
        pick = members[:: max(1, len(members) // 30)]
        cs = [cur[i] for i in pick]
        return R.masked_median([c[0] for c in cs], [c[1] for c in cs])

    a_ids = [sel[k] for k in np.flatnonzero(~lab)]
    b_ids = [sel[k] for k in np.flatnonzero(lab)]
    ref_a = med(a_ids)
    if not b_ids:
        return {"ref_a": med(ids), "ref_b": None, "has_b": False, "sep": 0.0}
    ref_b = med(b_ids)
    sep = float(1.0 - R.cs_ssim(ref_a, ref_b, None, win))
    mask_b = np.zeros(int(max(ids)) + 1, bool)
    mask_b[b_ids] = True
    b_runs = _long_runs(mask_b, times[: len(mask_b)], min_state_s, max_gap_s=0.6 + step * 0.2)
    if sep <= d_hi or len(b_runs) < 3:
        return {"ref_a": med(ids), "ref_b": None, "has_b": False, "sep": round(sep, 3)}
    return {"ref_a": ref_a, "ref_b": ref_b, "has_b": True, "sep": round(sep, 3)}


def _name_clusters(namer: Callable[[list[np.ndarray], str], list[float]], obj: str, frame_fn: Callable[[int], Optional[np.ndarray]],
                   shifts: Optional[list[tuple[float, float]]], inner: BBox, cur: Any, d_a: np.ndarray, p_b: np.ndarray,
                   in_a: np.ndarray, in_b: np.ndarray, has_b: bool, ref_b: Optional[np.ndarray], win: int,
                   cfg_obj: dict[str, Any], res: AppearanceResult) -> float:
    """[H2] Лог-шансы «A открыто» от open-vocab запросов по k кропам-представителям каждого кластера."""
    import cv2

    nc = cfg_obj.get("namer") or {}
    k = int(nc.get("k", 3))

    def reps(mask: np.ndarray, dist: np.ndarray) -> list[int]:
        ids = [i for i in np.flatnonzero(mask) if np.isfinite(dist[i])]
        return sorted(ids, key=lambda i: dist[i])[:k]

    ra = reps(in_a, d_a)
    rb: list[int] = []
    if has_b and ref_b is not None:
        d_b = np.full(len(d_a), np.nan)
        for i in np.flatnonzero(in_b):
            c = cur[i]
            d_b[i] = 1.0 - R.cs_ssim(c[0], ref_b, c[1], win)
        rb = reps(in_b, d_b)
    crops: list[np.ndarray] = []
    for i in ra + rb:
        fr = frame_fn(int(i))
        if fr is None or fr.ndim != 3:
            return 0.0
        sh = shifts[int(i)] if shifts else (0.0, 0.0)
        crops.append(cv2.cvtColor(R.crop(fr, inner, sh, (224, 224)), cv2.COLOR_BGR2RGB))
    if not ra:
        return 0.0
    try:
        probs = namer(crops, obj)
    except Exception as e:  # noqa: BLE001 — именование необязательно
        res.info["namer_error"] = str(e)[:200]
        return 0.0
    pa = float(np.clip(np.mean(probs[: len(ra)]), 0.02, 0.98))
    cap = float(nc.get("cap", 3.0))
    if rb:
        pb = float(np.clip(np.mean(probs[len(ra):]), 0.02, 0.98))
        lo = float(nc.get("weight", 1.5)) * (float(logit(pa)) - float(logit(pb)))
        res.info["namer_p_b"] = round(pb, 3)
    else:
        lo = float(nc.get("single_state_weight", 1.0)) * float(logit(pa))
    res.info["namer_p_a"] = round(pa, 3)
    return float(np.clip(lo, -cap, cap))


def _with_dets(res: AppearanceResult, det_lo: np.ndarray, det_c: np.ndarray, cfg_obj: dict[str, Any]) -> AppearanceResult:
    """Покадровое слияние (логит-сумма) вида и прямых детекций."""
    w_roi, w_det = float(cfg_obj["w_roi"]), float(cfg_obj["w_det"])
    z = w_roi * res.conf * logit(res.p_open) + w_det * det_c * det_lo
    c = np.clip(w_roi * res.conf + w_det * det_c, 0.0, 1.0)
    res.p_open = sigmoid(z)
    res.conf = c
    if np.any(det_c > 0) and "det" not in res.provenance:
        res.provenance.append("det")
    return res


def _state_det_logodds(det_lo: np.ndarray, det_c: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    w = det_c[mask]
    if w.sum() <= 1e-6:
        return 0.0
    mean = float(np.sum(w * det_lo[mask]) / w.sum())
    support = min(1.0, float(w.sum()) / 3.0)
    return float(np.clip(mean * support * 2.0, -4.0, 4.0))


def _shape_prior(cur: Any, in_a: np.ndarray, in_b: np.ndarray, cfg: dict[str, Any]) -> float:
    """Слабый априор: «открытый» вид темнее (проём), с большим числом кромок и с вертикальной кромкой вне контура."""
    ia = [i for i in np.flatnonzero(in_a) if i in cur]
    ib = [i for i in np.flatnonzero(in_b) if i in cur]
    if len(ia) < 2 or len(ib) < 2:
        return 0.0

    rel = float(cfg.get("dark_rel", 0.55))

    def dark(c: tuple[np.ndarray, ...]) -> float:
        # тёмный проём: пиксели ROI темнее dark_rel × медианы окружения (инвариантно к общему освещению)
        ci, mi, co, mo = c[4], c[1], c[5], c[3]
        ring_vals = co[mo > 0] if (mo > 0).sum() > 16 else co.ravel()
        thr = rel * float(np.median(ring_vals))
        vals = ci[mi > 0] if (mi > 0).sum() > 16 else ci.ravel()
        return float(np.mean(vals < thr))

    def feats(ids: list[int]) -> tuple[float, float, float]:
        dk, edge, ring = [], [], []
        for i in ids:
            c = cur[i]
            dk.append(dark(c))
            edge.append(R.edge_density(c[0], c[1]))
            ring.append(R.vertical_edge_energy(c[2], c[3]))
        return float(np.median(dk)), float(np.median(edge)), float(np.median(ring))

    da, ea, ra = feats(ia)
    db, eb, rb = feats(ib)
    # симметричная нормировка: перестановка A↔B меняет только знак априора
    s = float(cfg.get("dark_gain", 6.0)) * (db - da)
    s += float(cfg.get("edge_gain", 0.5)) * (eb - ea) / max(ea, eb, 1e-3)
    s += float(cfg.get("w_geom", 0.3)) * (rb - ra) / max(ra, rb, 1e-3)
    cap = float(cfg.get("shape_prior_cap", 1.5))
    return float(np.clip(s, -cap, cap))


def _long_runs(mask: np.ndarray, times: np.ndarray, min_s: float, max_gap_s: float) -> list[int]:
    out: list[int] = []
    run: list[int] = []
    last_t: Optional[float] = None
    for i in range(len(mask)):
        if not mask[i]:
            continue
        if run and last_t is not None and times[i] - last_t > max_gap_s + 1e-9:
            if times[run[-1]] - times[run[0]] >= min_s - 1e-9:
                out.extend(run)
            run = []
        run.append(i)
        last_t = float(times[i])
    if run and times[run[-1]] - times[run[0]] >= min_s - 1e-9:
        out.extend(run)
    return out
