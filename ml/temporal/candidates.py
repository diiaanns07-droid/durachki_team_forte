"""Генератор кандидатов: 3–8 моментов на видео для проверки VLM/правилами (роль модуля в фазе 2).

Источники: (1) пики обученной риск-кривой (если есть чекпоинт); (2) кривая изменений без обучения —
контраст эмбеддингов «до/после» (среднее окна [t−Δ, t) против [t, t+Δ)) по каждому виду + начало движения.
Нормировка кривой изменений — внутри видео (только ранжирование моментов; вердикт по ней не выносится).
Метрика: recall кандидатов в ±1.5 с от размеченной секунды при ≤ 8 кандидатах на видео.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _window_means(E: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Средние окон до/после для каждого шага (края — по доступным кадрам)."""
    T = len(E)
    C = np.concatenate([np.zeros((1, E.shape[1]), np.float64), np.cumsum(E, axis=0, dtype=np.float64)])
    idx = np.arange(T)
    lo = np.clip(idx - d, 0, T)
    hi = np.clip(idx + d, 0, T)
    nb = np.maximum(idx - lo, 1)
    na = np.maximum(hi - idx, 1)
    before = (C[idx] - C[lo]) / nb[:, None]
    after = (C[hi] - C[idx]) / na[:, None]
    before[idx == 0] = E[0]
    return before, after


def _cos_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return 1.0 - np.sum(a * b, axis=1) / np.maximum(na * nb, 1e-8)


def _rank01(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    order = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    return order / max(len(x) - 1, 1)


def change_curve(times: np.ndarray, emb: np.ndarray, dyn: Optional[np.ndarray], cfg: dict) -> np.ndarray:
    """Сырая кривая изменений (T,) на сетке times: max по видам и Δ контраста «до/после» + начало движения."""
    c = cfg.get("propose", {})
    T = len(times)
    if T < 3:
        return np.zeros(T, np.float32)
    dt = float(np.median(np.diff(times))) if T > 1 else 0.25
    out = np.zeros(T, np.float64)
    for delta in c.get("change_deltas_s", [1.0, 3.0]):
        d = max(1, int(round(float(delta) / max(dt, 1e-6))))
        for v in range(emb.shape[1]):
            E = emb[:, v].astype(np.float64)
            if np.isnan(E).any():
                continue
            b, a = _window_means(E, d)
            out = np.maximum(out, _cos_dist(b, a))
    curve = _rank01(out)
    if dyn is not None and len(dyn) == T and float(c.get("motion_weight", 0.5)) > 0:
        en = np.convolve(dyn[:, 0].astype(np.float64), np.ones(3) / 3, mode="same")
        onset = np.maximum(np.diff(en, prepend=en[0]), 0.0)
        curve = np.maximum(curve, float(c.get("motion_weight", 0.5)) * _rank01(onset))
    return curve.astype(np.float32)


def resample_to(t_src: np.ndarray, x: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Максимум по корзинам шага t_dst (как risk_curve), пустые корзины — интерполяция."""
    if len(t_dst) == 0 or len(x) == 0:
        return np.zeros(len(t_dst), np.float32)
    step = float(t_dst[1] - t_dst[0]) if len(t_dst) > 1 else 0.5
    bins = np.clip(np.floor((np.asarray(t_src) + step / 2) / step).astype(int), 0, len(t_dst) - 1)
    r = np.full(len(t_dst), np.nan, np.float64)
    np.fmax.at(r, bins, np.asarray(x, np.float64))
    ok = ~np.isnan(r)
    return (np.interp(t_dst, t_dst[ok], r[ok]) if ok.any() else np.zeros(len(t_dst))).astype(np.float32)


def _nms_peaks(t: np.ndarray, x: np.ndarray, n: int, min_dist: float, taken: Sequence[float] = ()) -> list[int]:
    out: list[int] = []
    used = list(taken)
    for i in np.argsort(-np.asarray(x), kind="stable"):
        if len(out) >= n:
            break
        if all(abs(float(t[i]) - u) >= min_dist for u in used):
            out.append(int(i))
            used.append(float(t[i]))
    return out


def select_candidates(t: np.ndarray, risk: Optional[np.ndarray], change: Optional[np.ndarray], cfg: dict,
                      mode: Optional[str] = None, n_max: Optional[int] = None) -> list[dict]:
    """Список кандидатов [{t, source, salience, risk, change, rank}] — от 0 до n_max (обычно 3–8).

    risk  — пики риск-кривой (порядок по риску); change — пики кривой изменений; hybrid — сначала n_risk
    пиков риска, затем изменения, с минимальным расстоянием nms_s. Короткое видео даёт меньше кандидатов."""
    c = cfg.get("propose", {})
    mode = mode or str(c.get("mode", "hybrid"))
    n_max = int(n_max or c.get("max_candidates", 8))
    n_min = min(int(c.get("min_candidates", 3)), n_max)
    dist = float(c.get("nms_s", 2.0))
    t = np.asarray(t)
    have_risk = risk is not None and len(risk) == len(t) and len(t) > 0
    have_change = change is not None and len(change) == len(t) and len(t) > 0
    if mode == "risk" and not have_risk:
        mode = "change"
    if mode in ("change", "hybrid") and not have_change and have_risk:
        mode = "risk"
    if not (have_risk or have_change):
        return []
    picks: list[tuple[int, str]] = []
    if mode == "risk":
        picks = [(i, "risk") for i in _nms_peaks(t, risk, n_max, dist)]
    elif mode == "change":
        picks = [(i, "change") for i in _nms_peaks(t, change, n_max, dist)]
    else:
        n_r = min(int(c.get("n_risk", 4)), n_max) if have_risk else 0
        picks = [(i, "risk") for i in _nms_peaks(t, risk, n_r, dist)] if n_r else []
        rest = n_max - len(picks)
        picks += [(i, "change") for i in _nms_peaks(t, change, rest, dist, [float(t[i]) for i, _ in picks])]
    if len(picks) < n_min and have_risk and mode != "risk":  # добрать до минимума
        picks += [(i, "risk") for i in _nms_peaks(t, risk, n_min - len(picks), dist, [float(t[i]) for i, _ in picks])]
    out = []
    for rank, (i, src) in enumerate(picks):
        r = float(risk[i]) if have_risk else None
        ch = float(change[i]) if have_change else None
        sal = r if src == "risk" else float(c.get("change_salience_cap", 0.3)) * (ch or 0.0)
        out.append({"t": float(t[i]), "i": i, "source": src, "salience": round(float(sal), 4), "risk": r,
                    "change": ch, "rank": rank})
    return out


def candidate_recall(cands_per_video: Sequence[Sequence[dict]], gt_times: Sequence[Sequence[float]],
                     budgets: Sequence[int] = (3, 5, 8), tol: float = 1.5) -> dict:
    """Доля позитивных видео, где хотя бы один из первых k кандидатов в ±tol от любой размеченной секунды."""
    res: dict = {}
    pos = [(c, g) for c, g in zip(cands_per_video, gt_times) if g]
    for k in budgets:
        hits = [any(abs(x["t"] - gt) <= tol for x in c[:k] for gt in g) for c, g in pos]
        res[f"recall@{k}"] = float(np.mean(hits)) if hits else None
    res["mean_n"] = float(np.mean([len(c) for c in cands_per_video])) if cands_per_video else 0.0
    res["n_pos"] = len(pos)
    res["tol_s"] = tol
    return res
