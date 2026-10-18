"""Внешность трека = HSV-гистограмма ОДЕЖДЫ (верх/низ туловища), не лицо.

Кроп: без верхних 25% бокса (голова) и без краёв; лицевые признаки не извлекаются
и не хранятся — только 2×(8·6·4) бинов цвета. Используется лишь для склейки
треков после окклюзий/ухода и для проверки «тот же человек вернулся».
cv2 импортируется лениво; без cv2 или без кадров модуль тихо возвращает {}.
"""
from __future__ import annotations

import math
import os
import time
from collections import defaultdict
from typing import Any, Optional, Sequence

import numpy as np


def bhattacharyya(p: Optional[np.ndarray], q: Optional[np.ndarray]) -> Optional[float]:
    if p is None or q is None:
        return None
    s = float(np.sum(np.sqrt(np.clip(p, 0, None) * np.clip(q, 0, None))))
    return float(math.sqrt(max(0.0, 1.0 - min(1.0, s))))


def clothes_hist(frame: Any, bbox: Sequence[float], cfg: dict) -> Optional[np.ndarray]:
    import cv2

    if frame is None or not hasattr(frame, "shape"):
        return None
    img = frame
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    H, W = img.shape[:2]
    cr = cfg.get("crop", {})
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    bw, bh = x2 - x1, y2 - y1
    px1 = int(round((x1 + cr.get("x_margin", 0.2) * bw) * W))
    px2 = int(round((x2 - cr.get("x_margin", 0.2) * bw) * W))
    py1 = int(round((y1 + cr.get("y_top", 0.25) * bh) * H))
    py2 = int(round((y1 + cr.get("y_bottom", 0.9) * bh) * H))
    px1, px2 = max(0, px1), min(W, px2)
    py1, py2 = max(0, py1), min(H, py2)
    mp = int(cfg.get("min_box_px", 10))
    if px2 - px1 < mp or py2 - py1 < mp:
        return None
    hsv = cv2.cvtColor(np.ascontiguousarray(img[py1:py2, px1:px2]), cv2.COLOR_BGR2HSV)
    bins = [int(b) for b in cfg.get("bins", [8, 6, 4])]
    mid = hsv.shape[0] // 2
    feats = []
    for part in (hsv[:mid], hsv[mid:]):
        if part.size == 0:
            return None
        h = cv2.calcHist([part], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256]).ravel().astype(np.float64)
        feats.append(h / (h.sum() + 1e-9))
    return (np.concatenate(feats) * 0.5).astype(np.float32)


def _nearest_index(times: np.ndarray, t: float) -> int:
    i = int(np.searchsorted(times, t))
    if i <= 0:
        return 0
    if i >= len(times):
        return len(times) - 1
    return i if abs(times[i] - t) < abs(times[i - 1] - t) else i - 1


def _get_frame(clip: Any, key: tuple[str, float], times: Optional[np.ndarray]) -> Any:
    kind, v = key
    try:
        if kind == "i":
            return clip.frame(int(v))
        return clip.frame_at(float(v))
    except Exception:
        return None


def tracklet_appearance(clip: Any, tracklets: Sequence[Any], cfg: dict,
                        times: Optional[np.ndarray]) -> tuple[dict[int, np.ndarray], dict]:
    """Средняя гистограмма одежды по нескольким наблюдениям каждого трека.

    Возвращает (features, info). Ошибки чтения кадров и бюджет времени — не фатальны.
    """
    info: dict[str, Any] = {"enabled": False, "frames": 0, "tracks": 0}
    if not cfg.get("enabled", True) or os.environ.get("SENTINEL_TRACK_APPEARANCE", "1") == "0":
        info["reason"] = "disabled"
        return {}, info
    if clip is None or not (hasattr(clip, "frame") or hasattr(clip, "frame_at")):
        info["reason"] = "no_frames"
        return {}, info
    try:
        import cv2  # noqa: F401
    except ImportError:
        info["reason"] = "no_cv2"
        return {}, info

    use_index = times is not None and len(times) > 0 and hasattr(clip, "frame")
    req: dict[tuple[str, float], list[tuple[int, np.ndarray]]] = defaultdict(list)
    k = max(1, int(cfg.get("samples_per_track", 6)))
    for tr in tracklets:
        n = len(tr.times)
        order = np.argsort(-np.asarray(tr.scores))[: max(k * 3, k)]
        order = np.sort(order)
        pick = order[np.unique(np.linspace(0, len(order) - 1, num=min(k, len(order))).round().astype(int))]
        for i in pick:
            t = float(tr.times[int(i)])
            key = ("i", float(_nearest_index(times, t))) if use_index else ("t", round(t, 4))
            req[key].append((tr.tid, tr.boxes[int(i)]))
        del n
    keys = sorted(req)
    mf = int(cfg.get("max_frames", 150))
    if len(keys) > mf:
        keys = [keys[i] for i in np.unique(np.linspace(0, len(keys) - 1, mf).round().astype(int))]
    t0 = time.monotonic()
    feats: dict[int, list[np.ndarray]] = defaultdict(list)
    budget = float(cfg.get("budget_s", 4.0))
    for key in keys:
        if time.monotonic() - t0 > budget:
            info["budget_exceeded"] = True
            break
        img = _get_frame(clip, key, times)
        if img is None:
            continue
        info["frames"] += 1
        for tid, box in req[key]:
            try:
                h = clothes_hist(img, box, cfg)
            except Exception:
                h = None
            if h is not None:
                feats[tid].append(h)
    out = {}
    for tid, hs in feats.items():
        m = np.mean(np.stack(hs), axis=0)
        out[tid] = (m / (m.sum() + 1e-9)).astype(np.float32)
    info.update(enabled=bool(out), tracks=len(out), time_s=round(time.monotonic() - t0, 3))
    if not out and "reason" not in info:
        info["reason"] = "no_valid_crops"
    return out, info
