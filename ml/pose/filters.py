"""Временные фильтры: One-Euro для точек (с confidence gating) и офлайн-гистерезис для сигналов."""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class OneEuro:
    """Векторный One-Euro фильтр (Casiez et al., 2012) по неравномерной сетке времени."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 4.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.t_prev: float | None = None
        self.x_prev: np.ndarray | None = None
        self.dx_prev: np.ndarray | None = None

    def reset(self) -> None:
        self.t_prev = self.x_prev = self.dx_prev = None

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.t_prev is None or self.x_prev is None or t <= self.t_prev:
            self.t_prev, self.x_prev, self.dx_prev = t, x.copy(), np.zeros_like(x)
            return x.copy()
        dt = t - self.t_prev
        dx = (x - self.x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        tau = 1.0 / (2.0 * math.pi * np.maximum(cutoff, 1e-6))
        a = 1.0 / (1.0 + tau / dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.t_prev, self.x_prev, self.dx_prev = t, x_hat, dx_hat
        return x_hat


def smooth_keypoint_track(times: Sequence[float], kps: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Сглаживание (T, K, 3) точек одного трека.

    Confidence gating: точка со score < min_score не обновляет фильтр; её (x, y) заменяется последним
    сглаженным значением в течение hold_s (score остаётся исходным, низким). После hold_s — сырое
    значение и сброс фильтра этой точки.
    """
    out = np.array(kps, dtype=np.float32, copy=True)
    if out.size == 0:
        return out
    oe = cfg.get("one_euro", {}) or {}
    min_score = float(cfg.get("min_score", 0.3))
    hold_s = float(cfg.get("hold_s", 0.6))
    n_kp = out.shape[1]
    filters = [OneEuro(oe.get("min_cutoff", 1.0), oe.get("beta", 4.0), oe.get("d_cutoff", 1.0)) for _ in range(n_kp)]
    last_t = [None] * n_kp
    last_xy: list[np.ndarray | None] = [None] * n_kp
    for j, t in enumerate(times):
        t = float(t)
        for k in range(n_kp):
            score = float(out[j, k, 2])
            if score >= min_score:
                if last_t[k] is not None and t - float(last_t[k]) > hold_s:
                    filters[k].reset()
                xy = filters[k](t, out[j, k, :2])
                out[j, k, :2] = xy
                last_t[k], last_xy[k] = t, xy
            elif last_t[k] is not None and t - float(last_t[k]) <= hold_s and last_xy[k] is not None:
                out[j, k, :2] = last_xy[k]
    return out


def ramp(value: float, full: float, zero: float) -> float:
    """Линейная «рампа»: 1 при value == full, 0 при value == zero (в любую сторону), клип 0..1."""
    if full == zero:
        return 1.0 if value == full else 0.0
    return float(min(1.0, max(0.0, (zero - value) / (zero - full))))


def runs_from_mask(mask: Sequence[bool]) -> list[tuple[int, int]]:
    """Непрерывные отрезки True: [(i_start, i_end_inclusive), ...]."""
    runs: list[tuple[int, int]] = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _sample_period(times: np.ndarray) -> float:
    if len(times) < 2:
        return 0.2
    d = np.diff(times)
    d = d[d > 1e-6]
    return float(np.median(d)) if len(d) else 0.2


def hysteresis_mask(times: Sequence[float], evidence: Sequence[float], on: float, off: float,
                    min_duration_s: float, max_gap_s: float) -> np.ndarray:
    """Офлайн-гистерезис: семена ≥ on, рост по ≥ off, склейка разрывов ≤ max_gap_s,
    удаление отрезков короче min_duration_s (длительность = t_last - t_first + период)."""
    t = np.asarray(times, dtype=np.float64)
    e = np.nan_to_num(np.asarray(evidence, dtype=np.float64), nan=0.0)
    n = len(t)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    above_off = e >= off
    for i0, i1 in runs_from_mask(above_off):
        if np.any(e[i0:i1 + 1] >= on):
            mask[i0:i1 + 1] = True
    runs = runs_from_mask(mask)
    if not runs:
        return mask
    merged = [runs[0]]
    for a, b in runs[1:]:
        pa, pb = merged[-1]
        if t[a] - t[pb] <= max_gap_s + 1e-9:
            merged[-1] = (pa, b)
        else:
            merged.append((a, b))
    period = _sample_period(t)
    out = np.zeros(n, dtype=bool)
    for a, b in merged:
        if (t[b] - t[a]) + period >= min_duration_s - 1e-9:
            out[a:b + 1] = True
    return out


def gated_values(times: Sequence[float], evidence: Sequence[float], p: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Evidence → (значения 0..1, маска активности). Активно ⇒ ≥ 0.5, неактивно ⇒ ≤ 0.49."""
    e = np.clip(np.nan_to_num(np.asarray(evidence, dtype=np.float64), nan=0.0), 0.0, 1.0)
    mask = hysteresis_mask(times, e, float(p.get("thr_on", 0.6)), float(p.get("thr_off", 0.4)),
                           float(p.get("min_duration_s", 0.0)), float(p.get("max_gap_s", 0.0)))
    vals = 0.49 * e
    for a, b in runs_from_mask(mask):
        run_mean = float(e[a:b + 1].mean())
        vals[a:b + 1] = 0.5 + 0.5 * np.clip(0.5 * run_mean + 0.5 * e[a:b + 1], 0.0, 1.0)
    return vals, mask
