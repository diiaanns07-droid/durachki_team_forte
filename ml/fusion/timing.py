"""Привязка времени к реально существующим кадрам (PTS): сетка clip.times от ml.ingest или кадровая сетка fps."""
from __future__ import annotations

import bisect
import math
from typing import Optional, Sequence


def nearest_index(times: Sequence[float], t: float) -> Optional[int]:
    if not times:
        return None
    i = bisect.bisect_left(times, t)
    cands = [j for j in (i - 1, i) if 0 <= j < len(times)]
    return min(cands, key=lambda j: abs(times[j] - t))


def snap_time(times: Sequence[float], fps: float, duration: float, t: float) -> float:
    """PTS ближайшего сэмплированного кадра; без сэмплов — ближайший кадр сетки fps; иначе t."""
    i = nearest_index(times, t)
    if i is not None:
        return float(times[i])
    if fps > 0:
        idx = int(round(t * fps))
        if duration > 0:
            idx = min(idx, max(0, int(math.ceil(duration * fps - 1e-6)) - 1))
        return max(0, idx) / fps
    return max(0.0, float(t))


def clip_times(clip: object) -> list[float]:
    try:
        times = [float(t) for t in (getattr(clip, "times", None) or [])]
    except (TypeError, ValueError):
        return []
    return times if all(a <= b for a, b in zip(times, times[1:])) else sorted(times)
