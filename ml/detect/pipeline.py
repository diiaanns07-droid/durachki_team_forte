"""detect(clip, every_n) — оркестрация детекции по сэмплированным кадрам клипа.

Стадии на чанк кадров (batch_size):
  1. closed-set (RT-DETRv2/D-FINE и/или дообученный ONNX) на каждом выбранном кадре;
  2. TTA: горизонтальный флип (по умолчанию только на ключевых кадрах) → WBF;
  3. SAHI-стиль тайлы на ключевых кадрах для мелких объектов (телефон, купюры, ...);
  4. open-vocab OWLv2 на ключевых кадрах (~1 fps + всплески движения — activity gating);
  5. финальное слияние: пороги по меткам, подавление дублей тайлов, NMS по меткам.
Бюджет времени: при превышении budget_s отключаются стадии 2–4, при прогнозе > hard_budget_s растёт шаг.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Optional, Sequence

import numpy as np

try:
    from contracts.sentinel_contracts import Detection, VideoClip  # noqa
except ImportError:  # пока ветка A01 не смерджена
    from contracts.sentinel_contracts import Detection, VideoClip  # noqa

from .backends import Backends, DetectorUnavailable, get_backends, is_mock
from .boxes import (RawDets, classwise_nms, hflip_boxes, make_tiles, suppress_contained, tile_to_frame,
                    touches_inner_edge, weighted_boxes_fusion)
from .config import label_threshold, load_config
from .gating import KeyframeSelector, MotionMeter, motion_from_quality

log = logging.getLogger("ml.detect")

_TLS = threading.local()          # last_stats — свои у каждого потока (A03 может гнать несколько видео)
_RUN_LOCK = threading.Lock()      # один detect() за раз на процесс: GPU 8 GB, параллель не ускоряет, а рискует OOM


def _stats() -> dict[str, Any]:
    st = getattr(_TLS, "stats", None)
    if st is None:
        st = _TLS.stats = {}
    return st


def last_stats() -> dict[str, Any]:
    """Тайминги/бэкенды/предупреждения последнего вызова detect() в этом потоке — для meta (A03/A12)."""
    return dict(_stats())


class _Budget:
    def __init__(self, soft_s: float, hard_s: float):
        self.t0 = time.perf_counter()
        self.soft = float(soft_s)
        self.hard = float(hard_s)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def extras_allowed(self) -> bool:
        return self.elapsed < self.soft

    def projected(self, done: int, total: int) -> float:
        if done <= 0:
            return 0.0
        return self.elapsed / done * total


def _safe_predict(backend: Any, frames: Sequence[np.ndarray], stage: str, warnings: list[str]) -> Optional[list[RawDets]]:
    """predict с защитой: при ошибке (напр. CUDA OOM) пробуем по одному кадру; иначе None."""
    if not frames:
        return []
    try:
        return backend.predict(frames)
    except Exception as e:  # noqa: BLE001
        log.warning("detect: %s failed on batch of %d (%s) — retry per-frame", stage, len(frames), e)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        try:
            return [backend.predict([f])[0] for f in frames]
        except Exception as e2:  # noqa: BLE001
            warnings.append(f"detect: stage {stage} disabled: {type(e2).__name__}: {e2}")
            return None


def _apply_thresholds(d: RawDets, table: dict[str, float]) -> RawDets:
    if len(d) == 0:
        return d
    keep = [i for i, lab in enumerate(d.labels) if d.scores[i] >= label_threshold(table, lab)]
    return d.select(keep)


def _closed_views(backends: Sequence[Any], frames: Sequence[np.ndarray], stage: str,
                  warnings: list[str], broken: set[str]) -> list[RawDets]:
    """Прогон всех closed-set бэкендов (дообученный + COCO) и объединение их выходов по кадрам."""
    per_frame: list[list[RawDets]] = [[] for _ in frames]
    for be in backends:
        name = getattr(be, "name", stage)
        if name in broken:
            continue
        res = _safe_predict(be, frames, f"{stage}:{name}", warnings)
        if res is None:
            broken.add(name)
            continue
        for i, r in enumerate(res):
            per_frame[i].append(r)
    return [RawDets.concat(p) for p in per_frame]


def _to_detections(t: float, d: RawDets) -> list[Detection]:
    return [Detection(t=float(t), label=lab,
                      bbox=tuple(round(float(v), 5) for v in d.boxes[i]),  # type: ignore[arg-type]
                      score=round(float(d.scores[i]), 4), track_id=None, source=d.sources[i])
            for i, lab in enumerate(d.labels)]


def detect(clip: VideoClip, every_n: int = 1, *, config: Optional[dict[str, Any]] = None) -> list[Detection]:
    """Детекции на кадрах clip.times[::every_n]. bbox — нормированные (x1,y1,x2,y2) к кадру после поворота.

    Нет локальных весов → DetectorUnavailable (или [] при runtime.on_unavailable=empty); статус — в last_stats().
    """
    with _RUN_LOCK:
        return _detect(clip, every_n, config)


def _detect(clip: VideoClip, every_n: int, config: Optional[dict[str, Any]]) -> list[Detection]:
    cfg = load_config(config)
    rt, cs, tta, tl, ov, gt, mg = (cfg[k] for k in ("runtime", "closed_set", "tta", "tiling", "open_vocab",
                                                     "gating", "merge"))
    soft = float(os.environ.get("SENTINEL_DETECT_BUDGET_S", rt["budget_s"]))
    budget = _Budget(soft, max(soft, float(rt["hard_budget_s"])))
    times = [float(t) for t in getattr(clip, "times", [])]
    n = len(times)
    stats: dict[str, Any] = {"status": "ok", "n_sampled": n, "n_processed": 0, "n_keyframes": 0, "n_flip": 0,
                             "n_tiles": 0, "n_open": 0, "timings": {}, "warnings": [], "backends": [],
                             "mock": is_mock()}
    _stats().clear()
    _stats().update(stats)
    if n == 0:
        stats["coverage"] = coverage([], every_n, [])
        _stats().update(stats)
        return []
    tm = stats["timings"]

    def tick(key: str, t_start: float) -> None:
        tm[key] = round(tm.get(key, 0.0) + time.perf_counter() - t_start, 4)

    t_load = time.perf_counter()
    warnings: list[str] = stats["warnings"]
    try:
        be: Backends = get_backends(cfg)
    except DetectorUnavailable as e:
        stats["status"] = "unavailable"
        warnings.append(f"detect: unavailable: {e}")
        _stats().update(stats)
        if str(rt.get("on_unavailable", "raise")) == "empty":
            return []
        raise
    tick("load", t_load)
    warnings.extend(be.warnings)
    stats["backends"] = be.names

    mock = be.mock or is_mock()
    hflip = bool(tta.get("hflip", True)) and not mock  # для цветовых масок флип бессмысленен
    tiling_on = bool(tl.get("enabled", True)) and tl.get("on", "keyframes") != "off"
    open_on = be.open is not None and bool(ov.get("enabled", True))
    quality_motion = motion_from_quality(clip, n)
    meter = MotionMeter(gt.get("motion_thumb_width", 64))
    selector = KeyframeSelector(interval_s=ov.get("interval_s", 1.0), abs_thr=gt.get("motion_keyframe_thr", 0.035),
                                rel_k=gt.get("motion_rel_k", 2.5), min_gap_s=gt.get("min_gap_s", 0.4),
                                window=gt.get("window", 25), max_frames=ov.get("max_frames", 240))
    broken: set[str] = set()
    closed_names = {getattr(b, "name", "closed") for b in be.closed}
    processed: list[int] = []
    out: list[Detection] = []
    base_stride = max(1, int(every_n))
    stride = base_stride
    batch = max(1, int(rt.get("batch_size", 16)))
    score_thr = cs.get("score_thr", {"default": 0.4})
    max_dets = int(cs.get("max_dets_per_frame", 60))
    i = 0
    planned_total = max(1, (n + base_stride - 1) // base_stride)

    while i < n:
        idx = list(range(i, n, stride))[:batch]
        i = idx[-1] + stride
        t_dec = time.perf_counter()
        frames, fidx = [], []
        for j in idx:
            try:
                f = clip.frame(j)
            except Exception as e:  # noqa: BLE001
                warnings.append(f"detect: frame {j} decode failed: {e}")
                f = None
            if f is None or not hasattr(f, "shape") or f.ndim != 3 or f.size == 0:
                if len(warnings) < 50:
                    warnings.append(f"detect: frame {j} (t={times[j]:.2f}s) empty/invalid — skipped")
                continue
            frames.append(np.ascontiguousarray(f))
            fidx.append(j)
        tick("decode", t_dec)
        if not frames:
            continue
        stats["n_processed"] += len(frames)

        # --- activity gating / ключевые кадры ---
        t_g = time.perf_counter()
        key_flags: list[bool] = []
        energies: list[float] = []
        for f, j in zip(frames, fidx):
            e = quality_motion[j] if quality_motion is not None else meter.update(f)
            energies.append(float(e))
            key_flags.append(selector.offer(times[j], e)[0])
        stats["n_keyframes"] += sum(key_flags)
        tick("gating", t_g)
        extras = budget.extras_allowed()

        # --- 1. closed-set ---
        t_c = time.perf_counter()
        orig = _closed_views(be.closed, frames, "closed", warnings, broken)
        tick("closed", t_c)
        if closed_names and closed_names <= broken:
            # все closed-set бэкенды отказали: эти и дальнейшие кадры НЕ проанализированы (partial, а не «пусто»)
            warnings.append(f"detect: all closed-set backends failed at t={times[fidx[0]]:.2f}s — stopping")
            break
        processed.extend(fidx)

        # --- 2. TTA флип ---
        flip_views: dict[int, RawDets] = {}
        if hflip and extras:
            t_f = time.perf_counter()
            sel = [k for k in range(len(frames)) if tta.get("hflip_on", "keyframes") == "all" or key_flags[k]]
            if sel:
                fl = _closed_views(be.closed, [frames[k][:, ::-1] for k in sel], "flip", warnings, broken)
                for k, d in zip(sel, fl):
                    flip_views[k] = RawDets(hflip_boxes(d.boxes), d.scores, d.labels, [s + ":flip" for s in d.sources])
                stats["n_flip"] += len(sel)
            tick("flip", t_f)

        merged: list[RawDets] = []
        for k in range(len(frames)):
            if k in flip_views:
                if tta.get("merge", "wbf") == "wbf":
                    m = weighted_boxes_fusion([orig[k], flip_views[k]], iou_thr=tta.get("iou_thr", 0.55),
                                              single_view_penalty=tta.get("single_view_penalty", 0.85))
                else:
                    m = classwise_nms(RawDets.concat([orig[k], flip_views[k]]), tta.get("iou_thr", 0.55))
            else:
                m = classwise_nms(orig[k], mg.get("iou_thr", 0.6))
            merged.append(_apply_thresholds(m, score_thr))

        # --- 3. тайлы (SAHI-стиль) ---
        tile_dets: dict[int, RawDets] = {}
        if tiling_on and extras:
            t_t = time.perf_counter()
            keep_labels = set(tl.get("keep_labels", []))
            min_motion = float(tl.get("min_motion", 0.0))
            for k, f in enumerate(frames):
                if tl.get("on", "keyframes") == "keyframes" and not key_flags[k]:
                    continue
                if min_motion > 0 and energies[k] < min_motion:
                    continue
                h, w = f.shape[:2]
                tiles = make_tiles(w, h, tl.get("tile_frac", 1.0), tl.get("overlap", 0.15))
                if len(tiles) <= 1:
                    continue
                crops = [np.ascontiguousarray(f[tt.y0:tt.y1, tt.x0:tt.x1]) for tt in tiles]
                res = _closed_views(be.closed, crops, "tiles", warnings, broken)
                parts = []
                for tt, d in zip(tiles, res):
                    if len(d) == 0:
                        continue
                    edge = touches_inner_edge(d.boxes, tt, w, h, tl.get("edge_margin", 0.01))
                    keep = [q for q in range(len(d)) if not edge[q] and (not keep_labels or d.labels[q] in keep_labels)]
                    d = d.select(keep)
                    if len(d):
                        parts.append(RawDets(tile_to_frame(d.boxes, tt, w, h), d.scores, d.labels,
                                             [s + ":tile" for s in d.sources]))
                td = _apply_thresholds(classwise_nms(RawDets.concat(parts), tl.get("merge_iou", 0.5)), score_thr)
                tile_dets[k] = suppress_contained(td, merged[k], tl.get("contained_ios", 0.8))
                stats["n_tiles"] += len(tiles)
            tick("tiles", t_t)

        # --- 4. open-vocab на ключевых кадрах ---
        open_dets: dict[int, RawDets] = {}
        if open_on and extras and "open" not in broken:
            t_o = time.perf_counter()
            sel = [k for k in range(len(frames)) if key_flags[k]]
            obs = max(1, int(ov.get("batch_size", 4)))
            for s0 in range(0, len(sel), obs):
                part = sel[s0:s0 + obs]
                res = _safe_predict(be.open, [frames[k] for k in part], "open", warnings)
                if res is None:
                    broken.add("open")
                    break
                for k, d in zip(part, res):
                    open_dets[k] = d
                stats["n_open"] += len(part)
            tick("open", t_o)

        # --- 5. финальное слияние ---
        t_m = time.perf_counter()
        for k, j in enumerate(fidx):
            parts = [merged[k], tile_dets.get(k, RawDets()), open_dets.get(k, RawDets())]
            d = classwise_nms(RawDets.concat(parts), mg.get("iou_thr", 0.6))
            if len(d) > max_dets:
                d = d.select(np.arange(max_dets))
            out.extend(_to_detections(times[j], d))
        tick("merge", t_m)

        # --- бюджет: адаптивный шаг ---
        done = stats["n_processed"]
        if budget.projected(done, planned_total) > budget.hard and stride < base_stride * 4 and done >= batch:
            stride *= 2
            planned_total = done + max(0, (n - i + stride - 1) // stride)
            warnings.append(f"detect: over budget → stride {stride} from t={times[min(i, n - 1)]:.1f}s")

    tm["total"] = round(budget.elapsed, 4)
    stats["broken_stages"] = sorted(broken)
    stats["stride_final"] = stride
    stats["n_detections"] = len(out)
    # «срезаны» только реально включённые опции (в baseline-режиме резать нечего)
    stats["extras_cut"] = bool(hflip or tiling_on or open_on) and not budget.extras_allowed()
    stats["coverage"] = coverage(times, base_stride, processed)
    if stats["coverage"]["fraction"] < 1.0:
        stats["status"] = "partial"      # часть кадров не проанализирована: «нет детекций» там = неизвестно
    elif broken or stats["extras_cut"] or be.fallback_used:
        stats["status"] = "degraded"     # кадры покрыты, но часть стадий отключена или работает резервный детектор
    _stats().update(stats)
    out.sort(key=lambda d: (d.t, d.label, -d.score))
    return out


def coverage(times: Sequence[float], base_stride: int, processed: Sequence[int]) -> dict[str, Any]:
    """Полнота анализа (для meta.coverage у A03): какие из запланированных кадров реально прошли closed-set.

    gaps_s — интервалы времени без анализа (битые кадры, увеличенный из-за бюджета шаг, отказ бэкенда).
    В этих интервалах отсутствие детекций означает «неизвестно», а не «объектов нет».
    """
    planned = list(range(0, len(times), max(1, int(base_stride))))
    done = set(processed)
    gaps: list[list[float]] = []
    start: Optional[float] = None
    prev_t: Optional[float] = None
    for j in planned:
        if j in done:
            if start is not None:
                gaps.append([round(start, 3), round(float(prev_t), 3)])
                start = None
        else:
            if start is None:
                start = float(times[j])
            prev_t = float(times[j])
    if start is not None:
        gaps.append([round(start, 3), round(float(prev_t), 3)])
    n_done = sum(1 for j in planned if j in done)
    return {"frames_planned": len(planned), "frames_processed": n_done,
            "fraction": round(n_done / len(planned), 4) if planned else 0.0, "gaps_s": gaps}


def warmup(config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Загружает модели и делает пустой прогон (держим тёплыми). Возвращает {status, backends, warnings, seconds}.
    Не бросает: при отсутствии весов status=unavailable (сеть не трогается)."""
    with _RUN_LOCK:
        return _warmup(config)


def _warmup(config: Optional[dict[str, Any]]) -> dict[str, Any]:
    t0 = time.perf_counter()
    cfg = load_config(config)
    try:
        be = get_backends(cfg)
    except DetectorUnavailable as e:
        return {"status": "unavailable", "backends": [], "warnings": [str(e)],
                "seconds": round(time.perf_counter() - t0, 3)}
    dummy = np.zeros((848, 464, 3), np.uint8)
    w: list[str] = list(be.warnings)
    for b in be.closed:
        _safe_predict(b, [dummy], "warmup", w)
    if be.open is not None:
        _safe_predict(be.open, [dummy], "warmup-open", w)
    return {"status": "ok", "backends": be.names, "warnings": w, "seconds": round(time.perf_counter() - t0, 3)}
