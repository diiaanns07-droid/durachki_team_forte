"""Качество кадров и активность: резкость, яркость, засвет/темнота, motion_energy, gating.

Метрики считаются на уменьшенном сером кадре: КОРОТКАЯ сторона -> ``quality.analysis_short_side``
(портрет и ландшафт в одной шкале). Абсолютные значения резкости зависят от этого размера
(пороги в config.yaml — для него).

Записи ``clip.quality[i]`` (i — индекс сэмпла, как в ``clip.times``):
    t            — фактический PTS сэмпла, сек
    sharpness    — var(Laplacian), чем больше, тем резче
    brightness   — средняя яркость 0..1
    contrast     — СКО яркости 0..1
    dark_frac    — доля пикселей <= dark_level
    bright_frac  — доля пикселей >= bright_level (засвет)
    motion_raw   — средняя |разность| с предыдущим сэмплом, 0..1
    motion_frac  — доля «изменившихся» пикселей (|Δ| > diff_threshold), 0..1
    motion_energy— min(1, motion_frac / motion_norm) — то же имя, что сигнал контракта
    is_blurry / is_dark / is_bright / is_frozen — булевы флаги по порогам
"""
from __future__ import annotations

import bisect
from typing import Any, Mapping, Optional, Sequence

import numpy as np


def _gray_small(img: np.ndarray, short_side: int) -> np.ndarray:
    """Серый кадр, уменьшенный так, что короткая сторона = short_side (ориентация не важна)."""
    try:
        import cv2

        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if short_side and min(h, w) > short_side:
            scale = short_side / float(min(h, w))
            size = (max(2, int(round(w * scale))), max(2, int(round(h * scale))))
            gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        return gray
    except ImportError:  # pragma: no cover - numpy-фолбэк
        gray = img if img.ndim == 2 else (img[..., 0] * 0.114 + img[..., 1] * 0.587 + img[..., 2] * 0.299)
        step = max(1, int(round(min(gray.shape[:2]) / max(short_side, 1))))
        return np.ascontiguousarray(gray[::step, ::step]).astype(np.uint8)


def _laplacian_var(gray: np.ndarray) -> float:
    try:
        import cv2

        return float(cv2.Laplacian(gray, cv2.CV_32F).var())
    except ImportError:  # pragma: no cover
        g = gray.astype(np.float32)
        lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
        return float(lap.var())


def _blur(gray: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return gray
    k = ksize if ksize % 2 else ksize + 1
    try:
        import cv2

        return cv2.GaussianBlur(gray, (k, k), 0)
    except ImportError:  # pragma: no cover
        return gray


def frame_metrics(img: np.ndarray, cfg: Mapping[str, Any]) -> tuple[dict[str, float], np.ndarray]:
    """Метрики одного кадра + подготовленный (уменьшенный, размытый) серый кадр для motion."""
    gray = _gray_small(img, int(cfg.get("analysis_short_side", cfg.get("analysis_width", 160))))
    mean = float(gray.mean()) / 255.0
    metrics = {
        "sharpness": _laplacian_var(gray),
        "brightness": mean,
        "contrast": float(gray.std()) / 255.0,
        "dark_frac": float(np.count_nonzero(gray <= int(cfg.get("dark_level", 16)))) / gray.size,
        "bright_frac": float(np.count_nonzero(gray >= int(cfg.get("bright_level", 245)))) / gray.size,
    }
    return metrics, _blur(gray, int(cfg.get("blur_ksize", 5)))


def motion_metrics(prev: Optional[np.ndarray], cur: np.ndarray, cfg: Mapping[str, Any]) -> dict[str, float]:
    if prev is None or prev.shape != cur.shape:
        return {"motion_raw": 0.0, "motion_frac": 0.0, "motion_energy": 0.0}
    diff = np.abs(cur.astype(np.int16) - prev.astype(np.int16))
    frac = float(np.count_nonzero(diff > int(cfg.get("diff_threshold", 18)))) / diff.size
    norm = max(float(cfg.get("motion_norm", 0.06)), 1e-6)
    return {"motion_raw": float(diff.mean()) / 255.0, "motion_frac": frac, "motion_energy": min(1.0, frac / norm)}


def finalize_flags(entry: dict[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    entry["is_blurry"] = bool(entry["sharpness"] < float(cfg.get("blurry_laplacian", 40.0)))
    entry["is_dark"] = bool(entry["brightness"] < float(cfg.get("dark_mean", 0.15)))
    entry["is_bright"] = bool(entry["brightness"] > float(cfg.get("bright_mean", 0.88)))
    entry["is_frozen"] = bool(entry["motion_frac"] < float(cfg.get("frozen_motion", 0.0005)))
    return entry


def fix_first_motion(quality: list[dict[str, Any]], cfg: Mapping[str, Any]) -> None:
    """У первого сэмпла нет предыдущего — берём движение второго, чтобы не было ложного нуля."""
    if len(quality) >= 2:
        for key in ("motion_raw", "motion_frac", "motion_energy"):
            quality[0][key] = quality[1][key]
        finalize_flags(quality[0], cfg)


def summarize(quality: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Сводка по ролику + список предупреждений (по-русски, только наблюдаемое)."""
    if not quality:
        return {"n": 0}, ["нет кадров для оценки качества"]
    arr = {k: np.array([q[k] for q in quality], dtype=np.float64) for k in ("sharpness", "brightness", "motion_energy")}
    frac = {k: float(np.mean([bool(q[k]) for q in quality])) for k in ("is_blurry", "is_dark", "is_bright", "is_frozen")}
    summary = {
        "n": len(quality),
        "sharpness_median": float(np.median(arr["sharpness"])),
        "brightness_mean": float(arr["brightness"].mean()),
        "motion_mean": float(arr["motion_energy"].mean()),
        "motion_p95": float(np.percentile(arr["motion_energy"], 95)),
        "blurry_frac": frac["is_blurry"],
        "dark_frac": frac["is_dark"],
        "bright_frac": frac["is_bright"],
        "frozen_frac": frac["is_frozen"],
    }
    warnings: list[str] = []
    if frac["is_dark"] > 0.5:
        warnings.append(f"видео тёмное: {frac['is_dark']:.0%} кадров с низкой яркостью")
    if frac["is_bright"] > 0.5:
        warnings.append(f"видео пересвечено: {frac['is_bright']:.0%} кадров")
    if frac["is_blurry"] > 0.5:
        warnings.append(f"низкая резкость: {frac['is_blurry']:.0%} кадров размыты")
    return summary, warnings


# ---------------------------------------------------------------- активность / gating


def smooth(times: Sequence[float], values: Sequence[float], window_s: float) -> list[float]:
    """Скользящее среднее по времени (окно ±window_s/2), устойчиво к неравномерным times (VFR)."""
    if window_s <= 0 or len(values) < 2:
        return [float(v) for v in values]
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    csum = np.concatenate([[0.0], np.cumsum(v)])
    half = window_s / 2.0
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right")
    return [float((csum[h] - csum[l]) / max(h - l, 1)) for l, h in zip(lo, hi)]


def activity_segments(
    times: Sequence[float],
    energy: Sequence[float],
    *,
    threshold: float = 0.12,
    smooth_s: float = 0.6,
    pad_s: float = 1.0,
    min_gap_s: float = 1.0,
    duration: Optional[float] = None,
) -> list[tuple[float, float]]:
    """Участки движения [(t0, t1)]: сглаженная энергия > threshold, расширение pad_s, склейка близких."""
    if not times:
        return []
    sm = smooth(times, energy, smooth_s)
    end = float(duration if duration is not None else times[-1])
    segments: list[list[float]] = []
    for t, e in zip(times, sm):
        if e <= threshold:
            continue
        t0, t1 = max(0.0, t - pad_s), min(end, t + pad_s)
        if segments and t0 - segments[-1][1] <= min_gap_s:
            segments[-1][1] = max(segments[-1][1], t1)
        else:
            segments.append([t0, t1])
    return [(float(a), float(b)) for a, b in segments]


def gated_indices(
    times: Sequence[float],
    segments: Sequence[tuple[float, float]],
    *,
    idle_every_n: int = 5,
    active_every_n: int = 1,
) -> list[int]:
    """Индексы сэмплов для тяжёлых моделей: в активных участках чаще, в спокойных — реже.

    Первый и последний сэмплы всегда включены (чтобы видеть состояние сцены на краях).
    """
    n = len(times)
    if n == 0:
        return []
    idle_every_n, active_every_n = max(1, int(idle_every_n)), max(1, int(active_every_n))
    starts = [s for s, _ in segments]
    chosen: list[int] = []
    last_active = last_idle = -10**9
    for i, t in enumerate(times):
        k = bisect.bisect_right(starts, t) - 1
        active = k >= 0 and t <= segments[k][1]
        if active:
            if i - last_active >= active_every_n:
                chosen.append(i)
                last_active = i
        elif i - last_idle >= idle_every_n:
            chosen.append(i)
            last_idle = i
    if chosen[0] != 0:
        chosen.insert(0, 0)
    if chosen[-1] != n - 1:
        chosen.append(n - 1)
    return chosen
