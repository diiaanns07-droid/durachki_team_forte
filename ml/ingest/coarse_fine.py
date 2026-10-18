"""Coarse-to-fine чтение видео — адаптер для A03 (pipeline).

1. coarse: равномерный обзор ВСЕЙ длительности по уже сохранённым сэмплам (``clip.frame(i)``,
   без повторного декода), первый и последний сэмпл включены всегда;
2. fine: плотное чтение выбранных окон точными кадрами исходника (``clip.frames_at`` — полный
   размер, фактические PTS, без дублей одного кадра).

Окна берутся из кандидатов (Event / dict / (t_start, t_end[, score])) или, если кандидатов нет,
из пиков ``motion_energy`` (дешёвый приор, НЕ скор видео: число окон фиксировано и не растёт
с длиной ролика). Шаг, плотность и число кадров — параметры (``coarse_fine.*`` в config.yaml);
их подбирают по бюджету замером L2. Флаг ``coarse_fine.enabled`` по умолчанию выключен.

Пробелы покрытия, попавшие в окно, возвращаются в ``WindowRead.gaps``: «кадров нет» — это
«неизвестно», а не «нарушений нет».
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from .config import load_config


@dataclass
class Window:
    t0: float
    t1: float
    score: float = 0.0
    source: str = "candidate"  # candidate | activity | manual
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> float:
        return max(0.0, self.t1 - self.t0)


@dataclass
class WindowRead:
    window: Window
    requested: list[float]
    frames: list[Any]  # list[FrameRef], по возрастанию фактического PTS, без повторов
    gaps: list[tuple[float, float]] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.gaps and all(getattr(f, "ok", True) for f in self.frames)

    @property
    def timestamps(self) -> list[float]:
        return [f.timestamp_sec for f in self.frames]


@dataclass
class CoarseFinePlan:
    coarse_indices: list[int]
    coarse_times: list[float]
    windows: list[Window]
    dense_times: list[list[float]]
    params: dict[str, Any]

    @property
    def n_frames(self) -> int:
        return len(self.coarse_indices) + sum(len(d) for d in self.dense_times)


def settings(config: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    return dict(load_config(config)["coarse_fine"])


def enabled(config: Optional[Mapping[str, Any]] = None) -> bool:
    return bool(settings(config).get("enabled", False))


# ----------------------------------------------------------------------------- coarse


def coarse_indices(clip: Any, *, step_s: Optional[float] = None, n_frames: Optional[int] = None) -> list[int]:
    """Индексы сэмплов, равномерно покрывающие всю длительность (по фактическим PTS сэмплов)."""
    times = list(clip.times)
    if not times:
        return []
    first, last = float(times[0]), float(times[-1])
    if n_frames is not None and n_frames > 0:
        grid = np.linspace(first, last, int(n_frames)) if n_frames > 1 else np.array([first])
    else:
        step = float(step_s or 1.0)
        grid = np.append(np.arange(first, last, step), last)
    t = np.asarray(times, dtype=np.float64)
    idx = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
    prev = np.clip(idx - 1, 0, len(t) - 1)
    nearest = np.where(np.abs(t[prev] - grid) <= np.abs(t[idx] - grid), prev, idx)
    out = sorted({int(i) for i in nearest} | {0, len(t) - 1})
    return out


# ----------------------------------------------------------------------------- windows


def _event_to_window(ev: Any) -> Optional[Window]:
    if isinstance(ev, Window):
        return ev
    if isinstance(ev, (tuple, list)):
        if len(ev) < 2:
            return None
        return Window(float(ev[0]), float(ev[1]), float(ev[2]) if len(ev) > 2 else 0.0, "candidate")
    get = ev.get if isinstance(ev, Mapping) else (lambda k, d=None: getattr(ev, k, d))
    t0, t1 = get("t_start"), get("t_end")
    peak = get("t_peak")
    if t0 is None and t1 is None and peak is None:
        return None
    t0 = float(t0 if t0 is not None else peak)
    t1 = float(t1 if t1 is not None else peak)
    meta = {k: get(k) for k in ("type_id", "t_peak", "source") if get(k) is not None}
    return Window(min(t0, t1), max(t0, t1), float(get("confidence", 0.0) or 0.0), "candidate", meta)


def merge_windows(
    windows: Iterable[Any],
    *,
    duration: float,
    pad_s: float = 1.5,
    merge_gap_s: float = 1.0,
    min_window_s: float = 1.0,
    max_windows: Optional[int] = None,
) -> list[Window]:
    """Расширить на pad_s, обрезать к [0, duration], склеить пересекающиеся/близкие окна.

    При ``max_windows`` остаются окна с наибольшим score (при равенстве — более ранние).
    Результат отсортирован по времени.
    """
    items = [w for w in (_event_to_window(e) for e in windows) if w is not None]
    if not items:
        return []
    padded = []
    for w in items:
        t0, t1 = max(0.0, w.t0 - pad_s), min(float(duration), w.t1 + pad_s)
        if t1 - t0 < min_window_s:
            c = (t0 + t1) / 2
            t0, t1 = max(0.0, c - min_window_s / 2), min(float(duration), c + min_window_s / 2)
        padded.append(Window(t0, t1, w.score, w.source, {**w.meta, "members": [(w.t0, w.t1)]}))
    padded.sort(key=lambda w: w.t0)
    merged: list[Window] = [padded[0]]
    for w in padded[1:]:
        cur = merged[-1]
        if w.t0 - cur.t1 <= merge_gap_s:
            cur.t1 = max(cur.t1, w.t1)
            cur.score = max(cur.score, w.score)
            cur.meta["members"] = cur.meta.get("members", []) + w.meta.get("members", [])
            if cur.source != w.source:
                cur.source = "mixed"
        else:
            merged.append(w)
    if max_windows is not None and len(merged) > max_windows:
        keep = sorted(range(len(merged)), key=lambda k: (-merged[k].score, merged[k].t0))[: int(max_windows)]
        merged = [merged[k] for k in sorted(keep)]
    return merged


def windows_from_activity(clip: Any, *, top_k: int = 3, window_s: float = 4.0) -> list[Window]:
    """Окна вокруг top_k пиков сглаженной motion_energy (без повторов в одном окне)."""
    if top_k <= 0 or not getattr(clip, "quality", None):
        return []
    from .quality import smooth

    times = list(clip.times)
    energy = smooth(times, [float(q.get("motion_energy", 0.0)) for q in clip.quality], 0.6)
    order = np.argsort(-np.asarray(energy), kind="stable")
    picked: list[Window] = []
    for k in order:
        t = float(times[int(k)])
        if energy[int(k)] <= 0:
            break
        if any(w.t0 <= t <= w.t1 for w in picked):
            continue
        picked.append(Window(t - window_s / 2, t + window_s / 2, float(energy[int(k)]), "activity", {"t_peak": t}))
        if len(picked) >= top_k:
            break
    return picked


# ----------------------------------------------------------------------------- fine


def dense_times(clip: Any, window: Window, *, fps: float = 5.0, max_frames: int = 24) -> list[float]:
    """Моменты внутри окна с шагом 1/fps (не больше max_frames), привязанные к реальным PTS, без дублей."""
    if window.length <= 0:
        grid = np.array([window.t0])
    else:
        step = 1.0 / max(float(fps), 1e-6)
        grid = np.arange(window.t0, window.t1 + 1e-9, step)
        if max_frames and len(grid) > max_frames:
            grid = np.linspace(window.t0, window.t1, int(max_frames))
    snap = getattr(clip, "snap_time", None)
    complete = bool(getattr(clip, "meta", {}).get("source_times_complete", True))
    times = [float(snap(float(t))) if (snap and complete) else float(t) for t in grid]
    return sorted(set(times))


def _overlapping_gaps(clip: Any, window: Window) -> list[tuple[float, float]]:
    cov = getattr(clip, "coverage", None)
    gaps = cov.get("gaps", []) if isinstance(cov, Mapping) else getattr(clip, "meta", {}).get("coverage_gaps", [])
    return [(float(a), float(b)) for a, b in gaps if a < window.t1 and b > window.t0]


def read_window(clip: Any, window: Window, *, fps: float = 5.0, max_frames: int = 24) -> WindowRead:
    times = dense_times(clip, window, fps=fps, max_frames=max_frames)
    if hasattr(clip, "distinct_frames_at"):
        frames = sorted(clip.distinct_frames_at(times), key=lambda r: r.timestamp_sec)
    else:  # чужая реализация VideoClip: только контрактный frame_at
        from .video import FrameRef

        frames = [FrameRef(clip.frame_at(t), float(t), float(t)) for t in times]
    return WindowRead(window, times, frames, _overlapping_gaps(clip, window))


# ----------------------------------------------------------------------------- plan / run


def plan(clip: Any, candidates: Optional[Sequence[Any]] = None, *, config: Optional[Mapping[str, Any]] = None,
         **overrides: Any) -> CoarseFinePlan:
    """Спланировать кадры: равномерный обзор + окна. Бюджет ``budget_frames > 0`` делится так:
    ``coarse_share`` — на обзор, остаток — между окнами пропорционально длине (не меньше 2 на окно)."""
    p = {**settings(config), **overrides}
    duration = float(getattr(clip, "duration", 0.0) or (clip.times[-1] if clip.times else 0.0))
    budget = int(p.get("budget_frames") or 0)
    wins = merge_windows(
        list(candidates or []) or windows_from_activity(clip, top_k=int(p.get("activity_top_k", 3)),
                                                        window_s=float(p.get("activity_window_s", 4.0))),
        duration=duration, pad_s=float(p.get("window_pad_s", 1.5)), merge_gap_s=float(p.get("merge_gap_s", 1.0)),
        min_window_s=float(p.get("min_window_s", 1.0)), max_windows=int(p.get("max_windows", 6)) or None,
    )
    dense_fps = float(p.get("dense_fps", 5.0))
    per_window = int(p.get("max_frames_per_window", 24))
    if budget > 0:
        n_coarse = max(2, int(round(budget * float(p.get("coarse_share", 0.4)))))
        coarse = coarse_indices(clip, n_frames=min(n_coarse, budget))
        left = max(0, budget - len(coarse))
        total_len = sum(w.length for w in wins) or 1.0
        alloc = [max(2, int(math.floor(left * w.length / total_len))) for w in wins] if wins else []
        while alloc and sum(alloc) > left and max(alloc) > 2:
            alloc[int(np.argmax(alloc))] -= 1
        dense = [dense_times(clip, w, fps=dense_fps, max_frames=min(a, per_window)) for w, a in zip(wins, alloc)]
        if sum(len(d) for d in dense) + len(coarse) > budget:  # бюджет меньше минимума — окна по убыванию score
            dense = []
            room = budget - len(coarse)
            for w in sorted(wins, key=lambda w: -w.score):
                if room < 2:
                    break
                d = dense_times(clip, w, fps=dense_fps, max_frames=min(room, per_window))
                dense.append(d)
                room -= len(d)
            wins = sorted(wins, key=lambda w: -w.score)[: len(dense)]
            order = sorted(range(len(wins)), key=lambda k: wins[k].t0)
            wins, dense = [wins[k] for k in order], [dense[k] for k in order]
    else:
        n = int(p.get("coarse_frames") or 0)
        coarse = coarse_indices(clip, n_frames=n) if n > 0 else coarse_indices(clip, step_s=float(p.get("coarse_step_s", 1.0)))
        dense = [dense_times(clip, w, fps=dense_fps, max_frames=per_window) for w in wins]
    return CoarseFinePlan(coarse, [float(clip.times[i]) for i in coarse], wins, dense, p)


def run(clip: Any, candidates: Optional[Sequence[Any]] = None, *, config: Optional[Mapping[str, Any]] = None,
        **overrides: Any) -> tuple[CoarseFinePlan, list[WindowRead]]:
    """План + чтение окон точными кадрами. Обзорные кадры брать через ``clip.frame(i)`` по plan.coarse_indices."""
    pl = plan(clip, candidates, config=config, **overrides)
    reads = []
    for w, times in zip(pl.windows, pl.dense_times):
        frames = sorted(clip.distinct_frames_at(times), key=lambda r: r.timestamp_sec) if hasattr(clip, "distinct_frames_at") \
            else read_window(clip, w).frames
        reads.append(WindowRead(w, times, frames, _overlapping_gaps(clip, w)))
    return pl, reads
