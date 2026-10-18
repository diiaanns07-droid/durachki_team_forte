"""Трёхзначная логика на массивах (+ «скрыто»): 1 = true, 0 = false, -1 = unknown, -2 = occluded.

Правила: AND — любой false → false; все true → true; иначе unknown (occluded, если все неизвестные — скрытые).
OR — любой true → true; все false → false; иначе unknown/occluded. Неизвестное никогда не становится 0.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

T, F, U, O = 1, 0, -1, -2
NAMES = {T: "true", F: "false", U: "unknown", O: "occluded"}


def from_prob(p: Sequence[float], conf: Sequence[float], thr: float = 0.5, known: float = 0.5, vis: Optional[Sequence[float]] = None, vis_thr: float = 0.35) -> np.ndarray:
    """Вероятность + уверенность (+ видимость) → tri: известное по порогу, неизвестное при видимости < vis_thr — occluded."""
    p = np.asarray(p, dtype=np.float64)
    c = np.asarray(conf, dtype=np.float64)
    out = np.where(c >= known, np.where(p >= thr, T, F), U).astype(np.int8)
    if vis is not None:
        v = np.asarray(vis, dtype=np.float64)
        out = np.where((out == U) & (v < vis_thr), O, out).astype(np.int8)
    return out


def t_and(*xs: np.ndarray) -> np.ndarray:
    a = np.stack([np.asarray(x, dtype=np.int8) for x in xs])
    out = np.full(a.shape[1], U, dtype=np.int8)
    any_f = (a == F).any(axis=0)
    all_t = (a == T).all(axis=0)
    unk = (a == U).any(axis=0)
    out[~any_f & ~all_t & ~unk] = O
    out[all_t] = T
    out[any_f] = F
    return out


def t_or(*xs: np.ndarray) -> np.ndarray:
    a = np.stack([np.asarray(x, dtype=np.int8) for x in xs])
    out = np.full(a.shape[1], U, dtype=np.int8)
    any_t = (a == T).any(axis=0)
    all_f = (a == F).all(axis=0)
    unk = (a == U).any(axis=0)
    out[~any_t & ~all_f & ~unk] = O
    out[all_f] = F
    out[any_t] = T
    return out


def t_not(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.int8)
    return np.where(x == T, F, np.where(x == F, T, x)).astype(np.int8)


def runs(mask: Sequence[bool]) -> list[tuple[int, int]]:
    """Непрерывные участки True: [(i0, i1)] включительно."""
    m = np.asarray(mask, dtype=bool)
    out: list[tuple[int, int]] = []
    i, n = 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j + 1 < n and m[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def sustain(x: np.ndarray, times: Sequence[float], min_s: float, max_gap_s: float = 0.0) -> np.ndarray:
    """true сохраняется только в участках длительностью ≥ min_s (дребезг ≤ max_gap_s склеивается);
    короткие true становятся unknown (мало наблюдений), а не false."""
    x = np.asarray(x, dtype=np.int8).copy()
    t = np.asarray(times, dtype=np.float64)
    if len(x) == 0:
        return x
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.2
    is_t = x == T
    # склейка коротких разрывов (не через явный false)
    if max_gap_s > 0:
        rs = runs(is_t)
        for (_a0, a1), (b0, _b1) in zip(rs, rs[1:]):
            gap = slice(a1 + 1, b0)
            if t[b0] - t[a1] - dt <= max_gap_s + 1e-9 and not (x[gap] == F).any():
                is_t[gap] = True
    out = np.where(is_t, T, np.where(x == T, U, x)).astype(np.int8)
    for i0, i1 in runs(is_t):
        if t[i1] - t[i0] + dt < min_s - 1e-9:
            out[i0 : i1 + 1] = U
    return out


def names(x: np.ndarray) -> list[str]:
    return [NAMES[int(v)] for v in np.asarray(x, dtype=np.int8)]
