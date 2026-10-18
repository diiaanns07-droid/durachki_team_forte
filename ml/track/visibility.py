"""Потеря видимости (свет выключили / камера «ослепла») — БЕЗ яркости как признака.

Признак: одновременно пропали ВСЕ статичные предметы-якоря (стол, ноутбук, сейф, дверь…),
которые в остальное время видны почти всегда. Люди уйти могут, мебель — нет.
Во время потери видимости ответы о людях — «неизвестно», а не «ушли».
Если якорей мало (<2 или один, но не почти всегда видимый) — проверка недоступна.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import numpy as np


def _get(o: Any, k: str, default: Any = None) -> Any:
    return o.get(k, default) if isinstance(o, dict) else getattr(o, k, default)


def _times(tr: Any) -> np.ndarray:
    return np.sort(np.array([float(r[0]) for r in (_get(tr, "boxes", []) or [])], float))


def loss_intervals(object_times: Sequence[np.ndarray], grid: np.ndarray, tol: float,
                   min_loss_s: float, min_anchor_frac: float = 0.5) -> tuple[list[tuple[float, float]], str]:
    """object_times — массивы времён наблюдений статичных предметов; grid — обработанные кадры."""
    if len(grid) < 3:
        return [], "unavailable"
    span = float(grid[-1] - grid[0]) + tol
    anchors = []
    for ts in object_times:
        if len(ts) == 0:
            continue
        idx = np.clip(np.searchsorted(ts, grid), 1, len(ts) - 1) if len(ts) > 1 else np.zeros(len(grid), int)
        near = np.minimum(np.abs(ts[idx] - grid), np.abs(ts[np.maximum(idx - 1, 0)] - grid)) <= tol
        frac = float(near.mean())
        if frac >= min_anchor_frac:
            anchors.append((near, frac))
    if len(anchors) < 2 and not (len(anchors) == 1 and anchors[0][1] >= 0.9):
        return [], "unavailable"
    vis = np.zeros(len(grid), bool)
    for near, _ in anchors:
        vis |= near
    out = []
    i = 0
    while i < len(grid):
        if vis[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(grid) and not vis[j + 1]:
            j += 1
        if grid[j] - grid[i] + tol >= min_loss_s:
            out.append((round(float(grid[i]), 4), round(float(grid[j]), 4)))
        i = j + 1
    del span
    return out, "objects"


def loss_from_tracks(tracks: Iterable[Any], frame_dt: float, min_loss_s: float = 0.6,
                     grid: Optional[np.ndarray] = None) -> tuple[list[tuple[float, float]], str]:
    """То же по контрактным Track (для потребителей, у которых нет info)."""
    tl = list(tracks or [])
    objs = [_times(t) for t in tl if _get(t, "label") != "person"]
    if grid is None:
        all_t = np.concatenate([_times(t) for t in tl] or [np.zeros(0)])
        if len(all_t) == 0:
            return [], "unavailable"
        grid = np.arange(float(all_t.min()), float(all_t.max()) + 1e-6, max(frame_dt, 1e-3))
    return loss_intervals(objs, np.asarray(grid, float), 1.5 * frame_dt, min_loss_s)


def in_intervals(t: float, intervals: Sequence[Sequence[float]], pad: float = 0.0) -> bool:
    return any(a - pad <= t <= b + pad for a, b in intervals)
