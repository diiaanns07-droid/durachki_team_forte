"""scene_states: все SIGNALS на каждый сэмпл clip.times, сглаженные HMM, с conf/known/visibility/provenance.

Публичные функции:
    scene_states(clip, dets, tracks, zones, pose_states) -> list[SceneStateEx]   (SceneStateEx ⊂ SceneState)
    scene_states_with_trace(...) -> (states, trace)                              (для отладки/обучения/графиков)
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, Sequence

import numpy as np

from . import roi as R
from . import tri
from ._contract import SIGNALS
from .appearance import analyze_roi, extract_u8
from .config import is_mock, load_config
from .geometry import BBox, dilate_polygon, expand_bbox, polygon_bbox
from .learned import feature_matrix, load_model, predict
from .objects import (
    MASK_LABELS,
    cash_color_finalize,
    cash_color_frame,
    cash_det_cue,
    det_evidence,
    door_traffic,
    green_bag_finalize,
    green_frac_frame,
    object_roi,
)
from .presence import Sig, presence_signals
from .smoothing import hold_conf, logit, median_filter, runs_above, sigmoid, smooth
from .timeline import build_persons, build_timeline, sample_dt
from .types import KNOWN_CONF, SceneStateEx

log = logging.getLogger("ml.scene")

POSE_KEYS_DEFAULT = ["phone_in_hand", "phone_to_ear", "eating_drinking", "hands_hidden", "cashier_facing_away", "handover_motion"]


def scene_states(clip: Any, dets: Sequence[Any], tracks: Sequence[Any], zones: Sequence[Any], pose_states: Sequence[Any]) -> list[SceneStateEx]:
    states, _ = scene_states_with_trace(clip, dets, tracks, zones, pose_states)
    return states


def _zones_by_name(zones: Sequence[Any]) -> dict[str, list[list[tuple[float, float]]]]:
    out: dict[str, list[list[tuple[float, float]]]] = {}
    for z in zones or []:
        poly = [(float(x), float(y)) for x, y in (z.polygon or [])]
        if len(poly) >= 3:
            out.setdefault(str(z.name), []).append(poly)
    return out


def _frame_reader(clip: Any) -> Callable[[int], Optional[np.ndarray]]:
    cache: dict[int, Optional[np.ndarray]] = {}
    failed = {"n": 0}

    def get(i: int) -> Optional[np.ndarray]:
        if i in cache:
            return cache[i]
        fr = None
        if failed["n"] < 5:
            try:
                fr = clip.frame(i)
                if fr is not None:
                    fr = np.asarray(fr)
                    if fr.ndim not in (2, 3) or fr.size == 0:
                        fr = None
            except Exception as e:  # noqa: BLE001 — кадр недоступен → признаки кадра пропускаются
                failed["n"] += 1
                log.debug("frame %d unavailable: %s", i, e)
        if len(cache) > 8:  # кадры читаются одним проходом; кэш нужен только для первых (эталон сдвига)
            cache.clear()
        cache[i] = fr
        return fr

    return get


def _clip_quality(clip: Any, n: int) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Качество и motion из clip.quality (A04), если формат распознан: list[float] | list[dict] | dict[str, list]."""
    q = getattr(clip, "quality", None)
    if q is None:
        return None, None

    def arr(v: Any) -> Optional[np.ndarray]:
        try:
            a = np.asarray(v, dtype=np.float64).ravel()
        except (TypeError, ValueError):
            return None
        return a if len(a) == n and np.isfinite(a).all() else None

    qual = mot = None
    if isinstance(q, dict):
        for k in ("quality", "score", "q", "per_frame"):
            if k in q and arr(q[k]) is not None:
                qual = arr(q[k])
                break
        for k in ("motion_energy", "motion"):
            if k in q and arr(q[k]) is not None:
                mot = arr(q[k])
                break
    elif isinstance(q, (list, tuple)) and len(q) == n:
        if all(isinstance(x, dict) for x in q):
            qual = arr([x.get("quality", x.get("score", np.nan)) for x in q])
            mot = arr([x.get("motion_energy", x.get("motion", np.nan)) for x in q])
        else:
            qual = arr(q)
    if qual is not None:
        qual = np.clip(qual, 0.0, 1.0)
    if mot is not None:
        mot = np.clip(mot, 0.0, 1.0)
    return qual, mot


def _fill_nearest(vals: np.ndarray, have: np.ndarray, default: float) -> np.ndarray:
    out = vals.copy()
    if not have.any():
        out[:] = default
        return out
    hi = np.flatnonzero(have)
    for i in np.flatnonzero(~have):
        out[i] = vals[hi[np.argmin(np.abs(hi - i))]]
    return out


def scene_states_with_trace(
    clip: Any,
    dets: Sequence[Any],
    tracks: Sequence[Any],
    zones: Sequence[Any],
    pose_states: Sequence[Any],
    config: Optional[dict[str, Any]] = None,
) -> tuple[list[SceneStateEx], dict[str, Any]]:
    t0 = time.perf_counter()
    cfg = load_config(config)
    t_attr = getattr(clip, "times", None)  # может быть list или np.ndarray
    times = np.asarray([] if t_attr is None else list(t_attr), dtype=np.float64)
    n = len(times)
    trace: dict[str, Any] = {"warnings": [], "rois": {}, "raw": {}, "conf_raw": {}, "provenance": {}, "info": {}, "features": {}}
    if n == 0:
        return [], trace
    dets = [] if dets is None else list(dets)
    tracks = [] if tracks is None else list(tracks)
    zones = [] if zones is None else list(zones)
    pose_states = [] if pose_states is None else list(pose_states)
    zb = _zones_by_name(zones)
    rc = cfg["roi"]

    # --- рабочее место: cashier_area, иначе стол с дилатацией ---
    workplace = None
    for zn in cfg["presence"]["workplace_zones"]:
        if zb.get(zn):
            workplace = zb[zn]
            break
    if workplace is None and zb.get("desk"):
        workplace = [dilate_polygon(p, float(cfg["presence"]["desk_dilate"])) for p in zb["desk"]]
        trace["warnings"].append("no cashier_area zone: workplace = dilated desk")
    if workplace is None:
        trace["warnings"].append("no workplace zone: cashier_in_workplace/outsider are unknown")

    tl = build_timeline(times, dets, cfg)
    build_persons(tl, tracks, workplace[0] if workplace else None, cfg, float(cfg["time"]["hold_s"]))
    trace["warnings"].extend(tl.warnings)
    person_boxes: list[list[BBox]] = [[o.bbox for o in tl.persons[i]] for i in range(n)]

    # --- ROI объектов определяются ДО чтения кадров (из зон A06 или устойчивых детекций) ---
    obj_specs = (
        ("safe_open", "safe", "safe_open", "safe", cfg["safe"]),
        ("door_open", "door", "door_open", "door", cfg["door"]),
    )
    roi_of: dict[str, Optional[BBox]] = {}
    for name, zone_name, open_lbl, closed_lbl, _ in obj_specs:
        roi_box, roi_src = object_roi(zb, tl, zone_name, (closed_lbl, open_lbl))
        roi_of[name] = roi_box
        trace["rois"][zone_name] = {"bbox": roi_box, "source": roi_src}
    cc, bc = cfg["cash"], cfg["cash_bag"]
    desk_box = None
    for zn in ("desk", "cash_tray"):
        if zb.get(zn):
            desk_box = polygon_bbox(zb[zn][0])
            break
    green_region = expand_bbox(polygon_bbox(workplace[0]) or (0, 0, 1, 1), 0.2) if workplace else None

    # --- один проход по кадрам: качество, сдвиг камеры, motion, кропы ROI, цветовые признаки ---
    stride = max(int(rc["stride"]), int(np.ceil(n / max(int(rc["max_frames"]), 1))))
    idxs = list(range(0, n, stride))
    frames = _frame_reader(clip)
    q_clip, m_clip = _clip_quality(clip, n)
    have = np.zeros(n, dtype=bool)
    q_own = np.zeros(n)
    mot = np.zeros(n)
    bright = np.full(n, np.nan)
    shifts_have: dict[int, tuple[float, float]] = {}
    stores: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {k: {} for k, v in roi_of.items() if v is not None}
    cash_pf: dict[int, tuple[float, float]] = {}
    green_pf: dict[int, float] = {}
    ref_small: Optional[np.ndarray] = None
    prev_small: Optional[np.ndarray] = None
    aw = int(rc["align_width"])
    pad = float(rc["person_pad"])
    try:
        import cv2  # noqa: F401

        import_ok = True
    except ImportError:
        import_ok = False
        trace["warnings"].append("opencv unavailable: frame-based cues disabled")
    if import_ok:
        first = [frames(i) for i in idxs[: min(5, len(idxs))]]
        first_g = [R.resize_w(R.to_gray(f), aw) for f in first if f is not None]
        if first_g and all(g.shape == first_g[0].shape for g in first_g):
            ref_small = np.median(np.stack(first_g), axis=0).astype(np.float32)
        for i in idxs:
            fr = frames(i)
            if fr is None:
                continue
            g = R.resize_w(R.to_gray(fr), aw)
            have[i] = True
            q_own[i] = R.frame_quality(g)["quality"]
            bright[i] = float(np.median(g))
            if prev_small is not None and prev_small.shape == g.shape:
                mot[i] = float(np.mean(np.abs(g - prev_small)))
            prev_small = g
            sx = sy = 0.0
            if rc.get("align", True) and ref_small is not None and ref_small.shape == g.shape:
                max_px = float(rc["align_max_shift"]) * max(g.shape)
                dx, dy = R.phase_shift(ref_small, g, max_px, float(rc["align_min_response"]))
                sx, sy = dx / g.shape[1], dy / g.shape[0]
            shifts_have[i] = (sx, sy)
            for name, store in stores.items():
                store[i] = extract_u8(fr, roi_of[name], (sx, sy), rc)  # type: ignore[arg-type]
            if cc.get("color", True) and desk_box is not None:
                boxes = [o.bbox for o in tl.persons[i]] + [tuple(d.bbox) for d in tl.dets[i] if d.label in MASK_LABELS]
                cash_pf[i] = cash_color_frame(fr, desk_box, boxes, (sx, sy), cc, pad)
            if bc.get("color", True):
                green_pf[i] = green_frac_frame(fr, green_region, bc)
    if not have.any():
        trace["warnings"].append("frames unavailable: ROI/colour cues disabled, detections only")
    shifts_list = [shifts_have.get(i, (0.0, 0.0)) for i in range(n)]
    quality = q_clip if q_clip is not None else _fill_nearest(q_own, have, 1.0 if not have.any() else 0.5)
    trace["info"]["quality_source"] = "clip.quality" if q_clip is not None else ("frames" if have.any() else "default")
    det_ran = np.full(n, bool(dets))
    if not dets:
        trace["warnings"].append("no detections at all: absence claims are unknown")

    raw: dict[str, Sig] = {}
    vis: dict[str, np.ndarray] = {}
    weighted: set[str] = set()      # p уже взвешено уверенностью (p=0.5 ⇔ нет свидетельств)
    model = load_model(cfg)
    if model:
        trace["info"]["learned_signals"] = sorted(model)

    # --- присутствие ---
    pres, pres_extra = presence_signals(tl, zb, workplace, quality, det_ran, cfg)
    raw.update(pres)

    # --- сейф и дверь: вид ROI (без допущения «эталон = закрыто») + детекции (+ проходы для двери) ---
    for name, _zone, open_lbl, closed_lbl, ocfg in obj_specs:
        roi_box = roi_of[name]
        det_lo, det_c = det_evidence(tl, roi_box, open_lbl, closed_lbl, ocfg, quality)
        if roi_box is not None and stores.get(name):
            nm = None
            backend = str((ocfg.get("namer") or {}).get("backend", "none"))
            if backend != "none":
                from .namer import get_namer

                nm = get_namer(cfg.get("namer", {}), backend)
                if nm is None:
                    trace["warnings"].append(f"{name}: namer '{backend}' unavailable — naming by detections/shape only")
            res = analyze_roi(stores[name], [i for i in idxs if have[i]], n, times, roi_box, person_boxes, quality, det_lo, det_c, rc, ocfg,
                              use_shape_prior=(name == "safe_open"), naive_reference=bool(cfg.get("ablation", {}).get("naive")),
                              frame_fn=frames if nm is not None else None, shifts=shifts_list, namer=nm, obj=_zone)
            p, c, v = res.p_open, res.conf, res.visibility
            prov = list(dict.fromkeys(res.provenance))
            trace["info"][name] = {"p_a_open": res.p_a_open, "label_conf": res.label_conf, "two_states": res.has_two_states, **{k: v2 for k, v2 in res.info.items() if k not in ("ref_idx", "features")}}
            trace["raw"][name + "_change"] = np.nan_to_num(res.change, nan=-1.0)
            feats = dict(res.info.get("features", {}))
            feats.update({"change": res.change, "state_b": res.state_b})
        else:
            z = float(ocfg["w_det"]) * det_c * det_lo
            p, c = sigmoid(z), np.clip(float(ocfg["w_det"]) * det_c, 0.0, 1.0)
            v = np.ones(n) if roi_box is None else np.array([1.0 - R.occlusion_fraction(expand_bbox(roi_box, 0.1), person_boxes[i]) for i in range(n)])
            prov = ["det"] if (det_c > 0).any() else ["none"]
            feats = {}
        stores.pop(name, None)  # кропы больше не нужны — освобождаем память
        feats.update({"visibility": v, "quality": quality, "det_lo": det_lo, "det_c": det_c})
        if name == "door_open":
            tp, tc, crossings = door_traffic(tl, zb, cfg["door"], cfg["presence"])
            z = logit(p) + float(cfg["door"]["w_traffic"]) * tc * logit(tp)
            p, c = sigmoid(z), np.clip(c + float(cfg["door"]["w_traffic"]) * tc, 0.0, 1.0)
            trace["info"]["door_crossings"] = crossings
            feats.update({"traffic_lo": np.where(tc > 0, logit(tp), 0.0), "traffic_c": tc})
            if (tc > 0).any():
                prov.append("traffic")
        p, c = _apply_learned(name, p, c, feats, v, quality, model, cfg, prov)
        trace["features"][name] = feats
        raw[name] = Sig(p, c, prov)
        vis[name] = v
        weighted.add(name)

    # --- наличные: детекции в зоне стола + запасной цвет/форма-классификатор ---
    s_det, _, cash_localized = cash_det_cue(tl, zb, cc, cfg["presence"])
    desk_vis = np.ones(n) if desk_box is None else np.array([1.0 - R.occlusion_fraction(desk_box, person_boxes[i]) for i in range(n)])
    z = np.where(s_det > 0, float(cc["w_det"]) * logit(np.clip(0.5 + 0.5 * s_det, 0.05, 0.97)), 0.0)
    c_cash = np.where(s_det > 0, np.minimum(1.0, s_det * 1.25), 0.0)
    if not cash_localized:  # нет зон — «на столе» не проверить: детекция не делает значение известным
        c_cash = np.minimum(c_cash, float(cc.get("unlocalized_c_max", 0.45)))
        trace["warnings"].append("no desk/cash_tray/cashier_area zones: cash location unverifiable → cash_visible unknown")
    # нет детекции: слабое «нет» только если стол виден, детектор работал и кадр годный
    c_absent = np.where(s_det > 0, 0.0, float(cc["det_absent_c"]) * desk_vis * quality * det_ran)
    z = z + c_absent * logit(float(cc["det_absent_p"]))
    prov_cash = ["det"] if (s_det > 0).any() else []
    feats_cash: dict[str, np.ndarray] = {"det_score": s_det, "colour_lo": np.zeros(n), "colour_c": np.zeros(n)}
    if cash_pf:
        pc_, ccol, dv = cash_color_finalize(cash_pf, n, quality, cc)
        z = z + float(cc["w_color"]) * ccol * logit(pc_)
        c_cash = c_cash + float(cc["w_color"]) * ccol
        desk_vis = np.minimum(desk_vis, dv)
        feats_cash.update({"colour_lo": np.where(ccol > 0, logit(pc_), 0.0), "colour_c": ccol})
        if (ccol > 0).any():
            prov_cash.append("colour")
    feats_cash.update({"visibility": desk_vis, "quality": quality})
    p_cash, c_cash_t = _apply_learned("cash_visible", sigmoid(z), np.clip(c_cash + c_absent, 0.0, 1.0), feats_cash, desk_vis, quality, model, cfg, prov_cash)
    trace["features"]["cash_visible"] = feats_cash
    raw["cash_visible"] = Sig(p_cash, c_cash_t, prov_cash or ["none"])
    vis["cash_visible"] = desk_vis
    weighted.add("cash_visible")

    # --- инкассаторская сумка ---
    s_bag = np.array([max((float(d.score) for d in tl.dets_of(i, ("cash_bag",), float(bc["min_det_score"]))), default=0.0) for i in range(n)])
    zb_ = np.where(s_bag > 0, logit(np.clip(0.5 + 0.5 * s_bag, 0.05, 0.97)), 0.0)
    cb = np.where(s_bag > 0, np.minimum(1.0, s_bag * 1.25), float(bc["det_absent_c"]) * quality * det_ran)
    zb_ = zb_ + np.where(s_bag > 0, 0.0, cb * logit(float(bc["det_absent_p"])))
    prov_bag = ["det"] if (s_bag > 0).any() else []
    if green_pf:
        gp, gc = green_bag_finalize(green_pf, n, quality, bc)
        zb_ = zb_ + gc * logit(gp)
        cb = cb + gc
        if (gc > 0).any():
            prov_bag.append("colour")
    raw["cash_bag_visible"] = Sig(sigmoid(zb_), np.clip(cb, 0.0, 1.0), prov_bag or ["none"])
    weighted.add("cash_bag_visible")

    # --- motion_energy ---
    if m_clip is not None:
        me = m_clip
        prov_m = ["clip.quality"]
    else:
        me = 1.0 - np.exp(-_fill_nearest(mot, have, 0.0) / float(cfg["motion"]["scale"]))
        prov_m = ["frame_diff"] if have.any() else ["none"]
    raw["motion_energy"] = Sig(np.clip(me, 0.0, 1.0), np.full(n, 1.0 if (m_clip is not None or have.any()) else 0.0), prov_m)

    # --- pose-сигналы: из pose_states, иначе слабый det-фолбэк ---
    _merge_pose(raw, tl, pose_states, times, quality, cfg, trace)

    # --- сглаживание ---
    out_p: dict[str, np.ndarray] = {}
    out_c: dict[str, np.ndarray] = {}
    tau = float(cfg.get("known", {}).get("hold_tau_s", 1.0))
    for name, sig in raw.items():
        trace["raw"][name] = sig.p
        trace["conf_raw"][name] = sig.c
        trace["provenance"][name] = "+".join(sig.prov) if sig.prov else "none"
        if name == "n_persons":
            out_p[name] = np.round(median_filter(sig.p, int(cfg["presence"]["n_persons_median_win"])))
            out_c[name] = sig.c
            continue
        spec = _smooth_spec(name, cfg)
        if cfg.get("ablation", {}).get("smoothing"):
            spec = dict(cfg["ablation"]["smoothing"])  # только для абляции (ml.scene.ablation)
        out_p[name] = smooth(sig.p, times, spec, conf=None if name in weighted else sig.c)
        out_c[name] = hold_conf(sig.c, times, tau) if spec.get("method") == "hmm" else sig.c
        if cfg.get("ablation", {}).get("naive"):
            # АБЛЯЦИЯ «наивный режим»: неизвестное считается наблюдённым (как будто «не увидели» = «нет»)
            out_p[name] = smooth(np.where(sig.c > 0, sig.p, 0.0), times, spec)
            out_c[name] = np.ones(n)

    # --- составной: оставленные без присмотра наличные (трёхзначная логика) ---
    _unattended(out_p, out_c, pres_extra, raw, times, cfg, trace)

    # --- фаза 2: явные статусы, свет, предикаты, интервалы ---
    bright_f = _fill_nearest(np.nan_to_num(bright, nan=0.0), have, np.nan) if have.any() else None
    if bright_f is not None:
        bright_f = np.where(np.isfinite(bright_f), bright_f, np.nan)
    pctx_lights = _lights(times, bright_f, cfg)
    if cfg["lights"].get("freeze", False):
        _freeze_dark(out_p, pctx_lights, trace)
    sig_tri, why_sig = _statuses(out_p, out_c, vis, quality, pctx_lights, pres_extra, raw, cfg, workplace, bool(dets), have, trace,
                                 cash_localized=cash_localized)
    pred_res = None
    if cfg["predicates"].get("enabled", True):
        from .predicates import PredContext, compute as compute_predicates

        pctx = PredContext(
            times=times, tl=tl, zones=zb, workplace=workplace, sig_tri=sig_tri,
            sig_vis={k: np.asarray(v) for k, v in vis.items()}, quality=np.asarray(quality), brightness=bright_f,
            det_ran=det_ran, pres_extra=pres_extra, safe_roi=roi_of.get("safe_open"), desk_box=desk_box, cfg=cfg,
        )
        pred_res = compute_predicates(pctx)
        trace["events"] = pred_res.events

    # --- сборка ---
    states: list[SceneStateEx] = []
    obs_unknown = pres_extra["unknown_person_in_cashier_area"]
    for i in range(n):
        sigs, conf, known, visd, prov = {}, {}, {}, {}, {}
        for name in SIGNALS:
            p = float(out_p.get(name, np.zeros(n))[i]) if name in out_p else 0.0
            c = float(out_c.get(name, np.zeros(n))[i]) if name in out_c else 0.0
            sigs[name] = float(p) if name == "n_persons" else float(np.clip(p, 0.0, 1.0))
            conf[name] = float(np.clip(c, 0.0, 1.0))
            known[name] = conf[name] >= KNOWN_CONF
            prov[name] = trace["provenance"].get(name, "none")
            visd[name] = float(vis[name][i]) if name in vis else 1.0
        states.append(
            SceneStateEx(
                t=float(times[i]),
                signals=sigs,
                conf=conf,
                known=known,
                visibility=visd,
                provenance=prov,
                observed={
                    "unknown_person_in_cashier_area": float(obs_unknown[i]),
                    # A09 читает observed односторонне (≥ 0.5 = наблюдалось): «пусто» отдаём только подтверждённое
                    "workplace_empty": float(pres_extra["workplace_empty_p"][i]) if pres_extra["workplace_empty_c"][i] >= KNOWN_CONF else 0.0,
                    "workplace_empty_conf": float(pres_extra["workplace_empty_c"][i]),
                    "personal_items_on_desk": float(pred_res is not None and pred_res.tri["personal_item_on_desk"][i] == tri.T),
                    "workplace_track_lost": float(pres_extra["workplace_lost"][i]),
                    "safe_unattended": float(trace["raw"].get("safe_unattended", np.zeros(n))[i]),
                },
                quality=float(quality[i]),
                status={name: tri.NAMES[int(sig_tri[name][i])] for name in SIGNALS},
                why={name: why_sig[name][i] for name in SIGNALS},
                pred={k: float(v[i] == tri.T) for k, v in pred_res.tri.items()} if pred_res else {},
                pred_status={k: tri.NAMES[int(v[i])] for k, v in pred_res.tri.items()} if pred_res else {},
                pred_why={k: pred_res.why[k][i] for k in pred_res.tri} if pred_res else {},
            )
        )
    if pred_res is not None:
        from .intervals import intervals as _intervals

        trace["intervals"] = _intervals(states, kind="pred", policy=str(cfg["predicates"].get("timecode", "onset")))
    trace["info"]["timings_s"] = round(time.perf_counter() - t0, 3)
    trace["info"]["stride"] = stride
    trace["info"]["mock"] = is_mock()
    trace["times"] = times
    return states, trace


def _lights(times: np.ndarray, bright: Optional[np.ndarray], cfg: dict[str, Any]) -> np.ndarray:
    """tri «свет выключен» (используется и статусами, и предикатами)."""
    from .predicates import PredContext, lights_off

    n = len(times)
    ctx = PredContext(times=times, tl=None, zones={}, workplace=None, sig_tri={}, sig_vis={}, quality=np.ones(n), brightness=bright,
                      det_ran=np.ones(n, bool), pres_extra={}, safe_roi=None, desk_box=None, cfg=cfg)
    return lights_off(ctx)[0]


def _freeze_dark(out_p: dict[str, np.ndarray], lights: np.ndarray, trace: dict[str, Any]) -> None:
    """[H3, флаг lights.freeze] В темноте legacy-сигналы держат последнее значение до темноты (не дрейфуют к априору)."""
    dark = lights == tri.T
    if not dark.any():
        return
    for name, arr in out_p.items():
        if name in ("motion_energy", "n_persons"):
            continue
        a = arr.copy()
        for i0, i1 in tri.runs(dark):
            if i0 > 0:
                a[i0 : i1 + 1] = a[i0 - 1]
        out_p[name] = a
    trace["info"]["lights_freeze"] = True


_PRESENCE = ("cashier_present", "cashier_in_workplace", "outsider_in_cashier_area", "collector_present", "n_persons")
_ROI_SIG = ("safe_open", "door_open", "cash_visible")


def _statuses(out_p: dict[str, np.ndarray], out_c: dict[str, np.ndarray], vis: dict[str, np.ndarray], quality: np.ndarray, lights: np.ndarray,
              pres_extra: dict[str, np.ndarray], raw: dict[str, Any], cfg: dict[str, Any], workplace: Any, has_dets: bool, have: np.ndarray,
              trace: dict[str, Any], cash_localized: bool = True) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """true / false / unknown / occluded + причина для каждого сигнала (не увидели ≠ нет)."""
    sc = cfg["status"]
    n = len(quality)
    sig_tri: dict[str, np.ndarray] = {}
    whys: dict[str, list[str]] = {}
    lost = pres_extra.get("workplace_lost", np.zeros(n))
    person_lost = pres_extra.get("person_lost", np.zeros(n))
    unk_person = pres_extra.get("unknown_person_in_cashier_area", np.zeros(n))
    for name in SIGNALS:
        p = out_p.get(name, np.zeros(n))
        c = out_c.get(name, np.zeros(n))
        v = vis.get(name)
        x = tri.from_prob(p, c, vis=v, vis_thr=float(sc["vis_thr"]))
        prov = trace["provenance"].get(name, "none")
        w = []
        for i in range(n):
            if x[i] in (tri.T, tri.F):
                w.append("")
            elif lights[i] == tri.T:
                w.append("dark")
            elif x[i] == tri.O:
                w.append("occluded")
            elif name in _ROI_SIG and not have.any():
                w.append("frames_unavailable")
            elif quality[i] < float(sc["min_quality"]):
                w.append("low_quality")
            elif name in _PRESENCE and not workplace and name != "n_persons":
                w.append("no_zone")
            elif name in _PRESENCE and lost[i] > 0:
                w.append("track_lost")
            elif name in ("n_persons", "collector_present") and person_lost[i] > 0:
                w.append("track_lost")
            elif name == "cash_visible" and not cash_localized:
                w.append("no_zone")
            elif name == "outsider_in_cashier_area" and unk_person[i] > 0:
                w.append("role_unknown")
            elif name in POSE_KEYS_DEFAULT and "pose" not in prov:
                w.append("no_pose")
            elif not has_dets and name not in ("motion_energy",):
                w.append("no_detector")
            else:
                w.append("no_evidence")
        sig_tri[name] = x
        whys[name] = w
    return sig_tri, whys


def _apply_learned(name: str, p: np.ndarray, c: np.ndarray, feats: dict[str, np.ndarray], vis: np.ndarray, quality: np.ndarray, model: Optional[dict[str, Any]], cfg: dict[str, Any], prov: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Смешать ручную фузию с обученной моделью (если есть) в логит-пространстве."""
    m = (model or {}).get(name)
    if not m:
        return p, c
    n = len(p)
    try:
        pl = predict(m, feature_matrix(feats, list(m["features"]), n))
    except (KeyError, ValueError) as e:
        log.warning("learned model for %s ignored: %s", name, e)
        return p, c
    cl = np.clip(vis * quality, 0.0, 1.0) * float(m.get("reliability", 1.0))
    lam = float(np.clip(cfg["learned"]["weight"], 0.0, 1.0))
    z = (1.0 - lam) * logit(p) + lam * cl * logit(pl)
    prov.append("learned")
    return sigmoid(z), np.clip((1.0 - lam) * c + lam * cl, 0.0, 1.0)


def _smooth_spec(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    if name == "safe_open":
        return dict(cfg["safe"]["smoothing"])
    if name == "door_open":
        return dict(cfg["door"]["smoothing"])
    if name == "cash_visible":
        return dict(cfg["cash"]["smoothing"])
    if name == "cash_bag_visible":
        return dict(cfg["cash_bag"]["smoothing"])
    if name in cfg["presence_smoothing"]:
        return dict(cfg["presence_smoothing"][name])
    if name == "motion_energy":
        return dict(cfg["motion"]["smoothing"])
    return dict(cfg["pose"]["smoothing"])


def _merge_pose(raw: dict[str, Sig], tl: Any, pose_states: Sequence[Any], times: np.ndarray, quality: np.ndarray, cfg: dict[str, Any], trace: dict[str, Any]) -> None:
    pc = cfg["pose"]
    keys = list(pc.get("keys") or POSE_KEYS_DEFAULT)
    n = len(times)
    pts = sorted([s for s in (pose_states or []) if getattr(s, "signals", None) is not None], key=lambda s: float(s.t))
    pt = np.array([float(s.t) for s in pts])
    hold = max(float(pc["hold_s"]), sample_dt(times) * 0.51)
    fb_w = float(pc["fallback_weight"])
    for key in keys:
        p = np.zeros(n)
        c = np.zeros(n)
        src = set()
        for i, t in enumerate(times):
            if len(pt):
                j = int(np.argmin(np.abs(pt - t)))
                st = pts[j]
                if abs(pt[j] - t) <= hold and key in st.signals:
                    p[i] = float(np.clip(st.signals[key], 0.0, 1.0))
                    sc = getattr(st, "conf", None)
                    c[i] = float(sc.get(key, 1.0)) if isinstance(sc, dict) else 1.0
                    src.add("pose")
                    continue
            fp, fc = _pose_fallback(key, tl, i, cfg)
            if fc > 0:
                if pc.get("merge", "pose_first") == "max" or c[i] == 0:
                    p[i], c[i] = fp * fb_w, fc * fb_w * float(quality[i])
                    src.add("det_fallback")
        raw[key] = Sig(p, c, sorted(src) or ["none"])
    if not pts:
        trace["warnings"].append("no pose_states: pose signals from weak detection fallback (unknown where unseen)")


def _pose_fallback(key: str, tl: Any, i: int, cfg: dict[str, Any]) -> tuple[float, float]:
    """Слабые det-фолбэки без позы. Возвращает (p, conf); conf=0 — нечего сказать."""
    from .geometry import bbox_gap, bbox_intersection, bbox_area

    persons = sorted(tl.persons[i], key=lambda o: -o.p_cashier)
    if not persons:
        return 0.0, 0.0
    cashier = persons[0]
    if key in ("phone_in_hand", "eating_drinking"):
        labels = ("cell_phone",) if key == "phone_in_hand" else ("cup", "bottle", "food")
        best = 0.0
        for d in tl.dets_of(i, labels, 0.3):
            b = tuple(d.bbox)
            ov = bbox_intersection(b, cashier.bbox) / max(bbox_area(b), 1e-9)
            best = max(best, float(d.score) * ov)
        return (best, 0.4) if best > 0 else (0.0, 0.0)
    if key == "handover_motion":
        gap_max = float(cfg["pose"]["handover_gap"])
        for o in persons[1:]:
            if bbox_gap(cashier.bbox, o.bbox) <= gap_max:
                carry = tl.dets_of(i, ("cash", "cash_bag"), 0.3)
                return (0.6 if carry else 0.35), 0.4
        return 0.0, 0.0
    return 0.0, 0.0


def _unattended(out_p: dict[str, np.ndarray], out_c: dict[str, np.ndarray], pres_extra: dict[str, np.ndarray], raw: dict[str, Sig], times: np.ndarray, cfg: dict[str, Any], trace: dict[str, Any]) -> None:
    """cash_visible(известно «да») ∧ рабочее место пусто(известно «да», уход подтверждён) ≥ min_s.

    Неизвестное никогда не подставляется как 0/1: если любой операнд неизвестен — итог «неизвестно»
    (значение 0, conf низкий). «Пропала детекция кассира» не означает «ушёл».
    """
    uc = cfg["unattended"]
    thr = float(uc["thr"])
    n = len(times)
    cash_p, cash_c = out_p["cash_visible"], out_c["cash_visible"]
    empty_p = pres_extra["workplace_empty_p"]
    empty_c = pres_extra["workplace_empty_c"]  # без «удержания»: уверенность «занято» не должна растекаться на «пусто»
    cash_true = (cash_p >= thr) & (cash_c >= KNOWN_CONF)
    cash_false = (cash_p < thr) & (cash_c >= KNOWN_CONF)
    empty_true = (empty_p >= 0.5) & (empty_c >= KNOWN_CONF)
    occupied = (empty_p < 0.5) & (empty_c >= KNOWN_CONF)
    both = np.where(cash_true & empty_true, np.minimum(cash_p, empty_p), 0.0)
    runs = runs_above(both, times, thr, float(uc["min_s"]), float(uc["max_gap_s"]))
    val = np.zeros(n)
    conf = np.where(cash_false | occupied, np.maximum(np.where(cash_false, cash_c, 0), np.where(occupied, empty_c, 0)), 0.0)
    for i0, i1 in runs:
        val[i0 : i1 + 1] = np.maximum(both[i0 : i1 + 1], thr + 0.01)
        conf[i0 : i1 + 1] = np.minimum(cash_c[i0 : i1 + 1], empty_c[i0 : i1 + 1])
    # открытый сейф без людей на рабочем месте — отдельный след (в SIGNALS не входит)
    safe_p, safe_c = out_p["safe_open"], out_c["safe_open"]
    safe_true = (safe_p >= thr) & (safe_c >= KNOWN_CONF)
    trace["raw"]["safe_unattended"] = np.where(safe_true & empty_true, np.minimum(safe_p, empty_p), 0.0)
    if uc.get("include_open_safe", False):
        val = np.maximum(val, trace["raw"]["safe_unattended"])
    out_p["workplace_unattended_with_cash"] = val
    out_c["workplace_unattended_with_cash"] = conf
    trace["raw"]["workplace_unattended_with_cash"] = both
    trace["provenance"]["workplace_unattended_with_cash"] = "derived"
    trace["info"]["unattended_runs"] = [(float(times[a]), float(times[b])) for a, b in runs]
