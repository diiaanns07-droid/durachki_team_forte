"""Пост-обработка: вероятности по шагам -> риск-кривая 0.5 с -> события из пиков (общая для CV и инференса)."""
from __future__ import annotations

from typing import Optional

import numpy as np


def risk_curve(probs: np.ndarray, grid: np.ndarray, cfg: dict, duration: Optional[float] = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """Сгладить (скользящее среднее smooth_s) и пересэмплировать на шаг risk_step_s (max в корзине)."""
    inf = cfg["infer"]
    probs = np.asarray(probs, np.float32)
    grid = np.asarray(grid, np.float32)
    if len(probs) == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    dt = float(grid[1] - grid[0]) if len(grid) > 1 else 0.25
    win = max(1, int(round(float(inf.get("smooth_s", 1.0)) / max(dt, 1e-6))))
    if win > 1:
        k = np.ones(win, np.float32) / win
        pad = win // 2
        sm = np.convolve(np.pad(probs, (pad, win - 1 - pad), mode="edge"), k, mode="valid")
    else:
        sm = probs
    step = float(inf.get("risk_step_s", 0.5))
    end = float(duration) if duration else float(grid[-1])
    t_out = np.arange(0.0, end + 1e-6, step, dtype=np.float32)
    if len(t_out) == 0:
        t_out = np.zeros(1, np.float32)
    bins = np.clip(np.floor((grid + step / 2) / step).astype(int), 0, len(t_out) - 1)
    r = np.full(len(t_out), np.nan, np.float32)
    np.fmax.at(r, bins, sm)
    if np.isnan(r).any():  # пустые корзины — интерполяция
        ok = ~np.isnan(r)
        r = np.interp(t_out, t_out[ok], r[ok]).astype(np.float32) if ok.any() else np.zeros_like(t_out)
    return t_out, np.clip(r, 0.0, 1.0)


def find_peaks(t: np.ndarray, r: np.ndarray, threshold: float, min_dist_s: float, max_n: int,
               extent_ratio: float = 0.5) -> list[dict]:
    """Жадные пики выше порога с минимальным расстоянием; границы — где риск >= ratio * пик."""
    out: list[dict] = []
    if len(r) == 0:
        return out
    used = np.zeros(len(r), bool)
    order = np.argsort(-r, kind="stable")
    for i in order:
        if r[i] < threshold or len(out) >= max_n:
            break
        if used[i] or any(abs(float(t[i]) - p["t_peak"]) < min_dist_s for p in out):
            continue
        lvl = extent_ratio * r[i]
        a = i
        while a > 0 and r[a - 1] >= lvl:
            a -= 1
        b = i
        while b < len(r) - 1 and r[b + 1] >= lvl:
            b += 1
        used[a:b + 1] = True
        out.append({"t_peak": float(t[i]), "t_start": float(t[a]), "t_end": float(t[b]), "peak": float(r[i]),
                    "i": int(i)})
    return out


def video_score(r: np.ndarray) -> float:
    return float(np.max(r)) if len(r) else 0.0


# ----------------------------------------------------------------------------- пулинг уровня видео
# max по кадрам даёт длинным (чаще чистым) роликам больше попыток ложного пика -> AUC < 0.5 на реальных данных.
# topk — среднее k лучших окон при ФИКСИРОВАННОМ k; topk_null — то же, нормированное по числу окон:
# s = G(stat)^N_eff, где G — CDF блоковой статистики на чистых роликах (OOF, только train), N_eff = длина/τ.
def topk_mean(r: np.ndarray, k: int) -> float:
    if len(r) == 0:
        return 0.0
    k = max(1, min(int(k), len(r)))
    return float(np.mean(np.sort(np.asarray(r, np.float64))[-k:]))


def block_stats(r: np.ndarray, cfg: dict) -> np.ndarray:
    """Статистика topk по неперекрывающимся блокам τ секунд (для null-распределения чистых роликов)."""
    inf = cfg["infer"]
    step = float(inf.get("risk_step_s", 0.5))
    blk = max(1, int(round(float(inf.get("null_tau_s", 4.0)) / step)))
    k = int(inf.get("pool_k", 4))
    r = np.asarray(r, np.float64)
    if len(r) == 0:
        return np.zeros(0)
    return np.asarray([topk_mean(r[i:i + blk], k) for i in range(0, len(r), blk) if len(r[i:i + blk]) >= max(1, blk // 2)]
                      or [topk_mean(r, k)])


def null_cdf(x: float, null: np.ndarray) -> float:
    """Сглаженная эмпирическая CDF блоковой статистики чистых роликов; выше максимума — линейный хвост до 1."""
    z = np.sort(np.asarray(null, np.float64))
    n = len(z)
    if n == 0:
        return float(np.clip(x, 0.0, 1.0))
    q = (np.arange(n) + 0.5) / (n + 1.0)
    if x <= z[0]:
        return float(q[0] * max(x, 0.0) / max(z[0], 1e-9))
    if x >= z[-1]:
        top = q[-1]
        return float(top + (1.0 - top) * min((x - z[-1]) / max(1.0 - z[-1], 1e-9), 1.0))
    return float(np.interp(x, z, q))


def pool_score(r: np.ndarray, cfg: dict, null: Optional[np.ndarray] = None,
               duration: Optional[float] = None) -> float:
    """Скор видео по риск-кривой (шаг risk_step_s). Режим — cfg.infer.pooling."""
    inf = cfg["infer"]
    mode = str(inf.get("pooling", "topk_null"))
    r = np.asarray(r, np.float64)
    if len(r) == 0:
        return 0.0
    if mode == "max":
        return float(r.max())
    if mode == "topk" or null is None or len(null) == 0:
        return topk_mean(r, int(inf.get("pool_k", 4)))
    # согласовано с null: статистика видео = максимум блоковых topk по той же сетке блоков,
    # P(max ≤ x | n блоков чистого ролика) = G(x)^n  ->  скор = G(stat)^n (нормировка по числу окон)
    blocks = block_stats(r, cfg)
    return float(null_cdf(float(blocks.max()), null) ** max(1, len(blocks)))


# ----------------------------------------------------------------------------- политика таймкода
# Таймкод = момент НАЧАЛА нарушения (переход). Какая политика лучше — решает замер (cv: timecode_by_policy).
def timecode(t: np.ndarray, r: np.ndarray, policy: str, cfg: dict, change: Optional[np.ndarray] = None) -> Optional[float]:
    """peak — argmax риска; onset — первый момент перед пиком, где риск ≥ onset_ratio·пик (в пределах onset_max_s);
    change_onset — максимум кривой изменений в окне [пик − onset_max_s, пик] (если change передан)."""
    t = np.asarray(t)
    r = np.asarray(r)
    if len(r) == 0:
        return None
    i = int(np.argmax(r))
    inf = cfg["infer"]
    lookback = float(inf.get("onset_max_s", 6.0))
    if policy == "onset":
        lvl = float(inf.get("onset_ratio", 0.5)) * r[i]
        a = i
        while a > 0 and r[a - 1] >= lvl and t[i] - t[a - 1] <= lookback:
            a -= 1
        return float(t[a])
    if policy == "change_onset" and change is not None and len(change) == len(r):
        m = (t >= t[i] - lookback) & (t <= t[i])
        idx = np.nonzero(m)[0]
        return float(t[idx[int(np.argmax(np.asarray(change)[idx]))]]) if len(idx) else float(t[i])
    return float(t[i])


def onset_index(t: np.ndarray, r: np.ndarray, i: int, ratio: float, max_s: float) -> int:
    """Индекс начала подъёма перед пиком i: риск ≥ ratio·пик, не дальше max_s секунд назад."""
    lvl = ratio * r[i]
    a = i
    while a > 0 and r[a - 1] >= lvl and t[i] - t[a - 1] <= max_s:
        a -= 1
    return a
