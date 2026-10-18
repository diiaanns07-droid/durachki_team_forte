"""Сглаживание сигналов: HMM из 2 состояний (forward-backward), гистерезис, EMA, медиана.

Вход HMM — покадровая вероятность p_t (откалибрована при априорной 0.5) и уверенность c_t:
эмиссия задаётся отношением правдоподобий LR_t = (p_t / (1 − p_t))^(gain·c_t). При c_t = 0
кадр неинформативен и HMM держит предыдущее состояние (окклюзия, нет ROI и т.п.).
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

_EPS = 1e-4


def logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def hmm2_smooth(
    p: Sequence[float],
    times: Sequence[float],
    conf: Optional[Sequence[float]] = None,
    switch_per_s: float = 0.1,
    prior: float = 0.5,
    gain: float = 1.0,
) -> np.ndarray:
    """Апостериорная P(state=1 | все наблюдения) для двухсостояний HMM (офлайн, forward-backward)."""
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    if n == 0:
        return np.zeros(0)
    t = np.asarray(times, dtype=np.float64)
    c = np.ones(n) if conf is None else np.clip(np.asarray(conf, dtype=np.float64), 0.0, None)
    z = gain * c * logit(p)                       # лог-отношение правдоподобий эмиссии
    e1 = sigmoid(z)                                # эмиссия ∝ (e0, e1), нормировка не важна
    e0 = 1.0 - e1
    dt = np.diff(t, prepend=t[0])
    dt = np.clip(dt, 0.0, None)
    pr = float(np.clip(prior, 1e-3, 1 - 1e-3))
    # асимметричные переходы: стационарное распределение = prior, поэтому без наблюдений
    # (conf=0) апостериорная плавно возвращается к априорной, а не к 0.5
    lam = max(switch_per_s, 0.0)
    s01 = np.clip(1.0 - np.exp(-2.0 * lam * pr * dt), 1e-9, 0.5)
    s10 = np.clip(1.0 - np.exp(-2.0 * lam * (1.0 - pr) * dt), 1e-9, 0.5)
    alpha = np.zeros((n, 2))
    a = np.array([(1 - pr) * e0[0], pr * e1[0]])
    alpha[0] = a / a.sum()
    for i in range(1, n):
        prev = alpha[i - 1]
        pred = np.array([prev[0] * (1 - s01[i]) + prev[1] * s10[i], prev[1] * (1 - s10[i]) + prev[0] * s01[i]])
        a = pred * np.array([e0[i], e1[i]])
        tot = a.sum()
        alpha[i] = a / tot if tot > 0 else pred
    beta = np.ones(2)
    post = np.zeros(n)
    post[-1] = alpha[-1, 1]
    for i in range(n - 2, -1, -1):
        a01, a10 = s01[i + 1], s10[i + 1]
        eb = np.array([e0[i + 1], e1[i + 1]]) * beta
        beta = np.array([(1 - a01) * eb[0] + a01 * eb[1], a10 * eb[0] + (1 - a10) * eb[1]])
        beta = beta / max(beta.sum(), 1e-300)
        g = alpha[i] * beta
        post[i] = g[1] / max(g.sum(), 1e-300)
    return np.clip(post, 0.0, 1.0)


def hold_conf(c: Sequence[float], times: Sequence[float], tau_s: float = 1.0) -> np.ndarray:
    """Уверенность «недавно наблюдали»: max_j c_j·exp(−|t−t_j|/tau) (двусторонний проход)."""
    c = np.asarray(c, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    n = len(c)
    if n == 0:
        return c
    out = c.copy()
    for rng in (range(1, n), range(n - 2, -1, -1)):
        for i in rng:
            j = i - 1 if rng.step == 1 else i + 1
            decay = np.exp(-abs(t[i] - t[j]) / max(tau_s, 1e-6))
            out[i] = max(out[i], out[j] * decay)
    return np.clip(out, 0.0, 1.0)


def ema(x: Sequence[float], alpha: float = 0.5, bidirectional: bool = True) -> np.ndarray:
    """EMA; bidirectional=True — прямой+обратный проход (без фазового сдвига, офлайн)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x

    def _pass(v: np.ndarray) -> np.ndarray:
        out = np.empty_like(v)
        acc = v[0]
        for i, val in enumerate(v):
            acc = alpha * val + (1 - alpha) * acc
            out[i] = acc
        return out

    fwd = _pass(x)
    if not bidirectional:
        return fwd
    return (fwd + _pass(x[::-1])[::-1]) / 2.0


def hysteresis(
    p: Sequence[float],
    times: Sequence[float],
    hi: float = 0.6,
    lo: float = 0.4,
    min_on_s: float = 0.0,
    min_off_s: float = 0.0,
    alpha: float = 0.5,
) -> np.ndarray:
    """Гистерезис по EMA(p): состояние on/off; выход — EMA, прижатая к стороне 0.5 по состоянию."""
    x = ema(p, alpha=alpha)
    t = np.asarray(times, dtype=np.float64)
    out = np.empty_like(x)
    state = False
    cand_since: Optional[float] = None
    for i, v in enumerate(x):
        want = (v >= hi) if not state else not (v <= lo)
        if want != state:
            if cand_since is None:
                cand_since = t[i]
            need = min_on_s if not state else min_off_s
            if t[i] - cand_since >= need:
                state = want
                cand_since = None
        else:
            cand_since = None
        out[i] = max(v, 0.55) if state else min(v, 0.45)
    return np.clip(out, 0.0, 1.0)


def median_filter(x: Sequence[float], win: int = 5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if win <= 1 or len(x) == 0:
        return x.copy()
    h = win // 2
    pad = np.pad(x, (h, h), mode="edge")
    return np.array([np.median(pad[i : i + win]) for i in range(len(x))])


def smooth(p: Sequence[float], times: Sequence[float], spec: Optional[dict[str, Any]], conf: Optional[Sequence[float]] = None) -> np.ndarray:
    """Применить сглаживание по спецификации из конфига {method: hmm|hysteresis|ema|median|none, ...}."""
    spec = dict(spec or {})
    method = str(spec.pop("method", "none")).lower()
    arr = np.asarray(p, dtype=np.float64)
    if len(arr) == 0:
        return arr
    if method == "hmm":
        return hmm2_smooth(arr, times, conf=conf, **{k: spec[k] for k in ("switch_per_s", "prior", "gain") if k in spec})
    if method == "hysteresis":
        return hysteresis(arr, times, **{k: spec[k] for k in ("hi", "lo", "min_on_s", "min_off_s", "alpha") if k in spec})
    if method == "ema":
        return np.clip(ema(arr, alpha=float(spec.get("alpha", 0.5))), 0.0, 1.0)
    if method == "median":
        return median_filter(arr, int(spec.get("win", 5)))
    return arr.copy()


def runs_above(x: Sequence[float], times: Sequence[float], thr: float, min_s: float, max_gap_s: float = 0.0) -> list[tuple[int, int]]:
    """Интервалы индексов [i0, i1] где x > thr длительностью ≥ min_s (провалы ≤ max_gap_s склеиваются)."""
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    raw: list[list[int]] = []
    for i, v in enumerate(x):
        if v > thr:
            if raw and (i == raw[-1][1] + 1 or t[i] - t[raw[-1][1]] <= max_gap_s + 1e-9):
                raw[-1][1] = i
            else:
                raw.append([i, i])
    out: list[tuple[int, int]] = []
    for i0, i1 in raw:
        if len(t) and t[i1] - t[i0] + _sample_dt(t) >= min_s - 1e-9:
            out.append((i0, i1))
    return out


def _sample_dt(t: np.ndarray) -> float:
    if len(t) < 2:
        return 0.0
    return float(np.median(np.diff(t)))
