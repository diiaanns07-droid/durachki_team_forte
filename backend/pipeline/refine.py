"""Режим coarse→fine (гипотеза H1, флаг refine.enabled, по умолчанию выключен до замера L2):
обзор всей длительности с низким fps → окна вокруг кандидатов → плотная выборка окна →
detect/track/scene/rules на окне → уточнение начала события (onset) и t_peak.

Правила:
* кандидат, которого окно не подтвердило, НЕ отбрасывается (неизвестно ≠ нет): помечается
  extra.refine="not_confirmed" и идёт в VLM/fusion как был;
* окно не зависит от движения/риска: окна строятся по всем кандидатам (по убыванию уверенности),
  слабый скор временной модели клип не отсеивает;
* событие нового типа, найденное в окне, добавляется кандидатом (extra.refine="window_new").

Политика таймкода (гипотеза H2, timecode.policy, по умолчанию "module" — как отдал модуль):
"onset" — t_peak := t_start (+offset_s), т.е. момент начала нарушения (переход).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional

from .config import cfg_get
from .contract import Event


@dataclass
class Window:
    t0: float
    t1: float
    candidates: list[int] = field(default_factory=list)  # индексы кандидатов в окне
    events: list[Event] = field(default_factory=list)    # события правил на плотной выборке окна
    processed: bool = False
    error: str = ""


def build_windows(candidates: list[Event], cfg: Mapping, duration: Optional[float]) -> list[Window]:
    pre = float(cfg_get(cfg, "refine.pre_s", 4.0))
    post = float(cfg_get(cfg, "refine.post_s", 3.0))
    gap = float(cfg_get(cfg, "refine.merge_gap_s", 2.0))
    max_windows = int(cfg_get(cfg, "refine.max_windows", 4))
    order = sorted(range(len(candidates)), key=lambda i: -float(candidates[i].confidence))
    spans = []
    for i in order:
        c = candidates[i]
        t0 = max(0.0, min(c.t_start, c.t_peak) - pre)
        t1 = max(c.t_end, c.t_peak) + post
        if duration:
            t1 = min(t1, float(duration))
        spans.append((t0, t1, i))
    windows: list[Window] = []
    for t0, t1, i in sorted(spans):
        if windows and t0 - windows[-1].t1 <= gap:
            windows[-1].t1 = max(windows[-1].t1, t1)
            windows[-1].candidates.append(i)
        else:
            windows.append(Window(t0, t1, [i]))
    # окна с самыми уверенными кандидатами — первыми; лишние не обрабатываются (кандидаты остаются)
    rank = {i: r for r, i in enumerate(order)}
    windows.sort(key=lambda w: min(rank[i] for i in w.candidates))
    return windows[:max_windows] if max_windows > 0 else windows


def apply_refinement(candidates: list[Event], windows: list[Window]) -> tuple[list[Event], dict]:
    """Кандидаты → уточнённые кандидаты. Возвращает (события, сводка для meta.refine)."""
    out: list[Event] = []
    stats = {"windows": len(windows), "processed": sum(w.processed for w in windows),
             "confirmed": 0, "not_confirmed": 0, "unrefined": 0, "window_new": 0}
    used: set[int] = set()
    for idx, c in enumerate(candidates):
        win = next((w for w in windows if idx in w.candidates and w.processed), None)
        if win is None:
            stats["unrefined"] += 1
            out.append(c)
            continue
        same = [k for k, e in enumerate(win.events) if e.type_id == c.type_id]
        if not same:
            stats["not_confirmed"] += 1
            out.append(replace(c, extra={**c.extra, "refine": "not_confirmed", "window": [win.t0, win.t1]}))
            continue
        k = min(same, key=lambda j: abs(win.events[j].t_peak - c.t_peak))
        used.add(id(win.events[k]))
        e = win.events[k]
        stats["confirmed"] += 1
        out.append(replace(
            c, t_start=e.t_start, t_end=max(c.t_end, e.t_end), t_peak=e.t_peak,
            confidence=max(float(c.confidence), float(e.confidence)),
            facts=list(dict.fromkeys(list(c.facts) + list(e.facts))),
            bboxes=dict(e.bboxes) or dict(c.bboxes),
            extra={**c.extra, "refine": "confirmed", "window": [win.t0, win.t1],
                   "overview": {"t_start": c.t_start, "t_peak": c.t_peak, "confidence": c.confidence},
                   "window_track_ids": list(e.track_ids)}))
    for win in windows:
        if not win.processed:
            continue
        for e in win.events:
            if id(e) in used or any(candidates[i].type_id == e.type_id for i in win.candidates):
                continue
            stats["window_new"] += 1
            out.append(replace(e, extra={**e.extra, "refine": "window_new", "window": [win.t0, win.t1]}))
    return out, stats


def apply_timecode_policy(events: list[Event], cfg: Mapping) -> list[Event]:
    policy = str(cfg_get(cfg, "timecode.policy", "module") or "module")
    if policy == "module":
        return events
    offset = float(cfg_get(cfg, "timecode.offset_s", 0.0))
    out = []
    for e in events:
        t = min(max(e.t_start + (offset if policy == "onset_offset" else 0.0), e.t_start), max(e.t_end, e.t_start))
        out.append(replace(e, t_peak=t, extra={**e.extra, "t_peak_module": e.t_peak, "timecode_policy": policy}))
    return out
