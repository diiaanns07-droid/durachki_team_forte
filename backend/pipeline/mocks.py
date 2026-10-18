"""Встроенные фейки стадий. Используются ТОЛЬКО в mock-режиме (SENTINEL_MOCK=1 / --mock) и только
для модулей, которых нет в ветке, — чтобы весь пайплайн гонялся без весов и без чужих модулей.
В боевом режиме отсутствующий модуль = сбой стадии (никаких выдуманных данных).

Выход детерминирован и зависит от содержимого кадров (энергия движения), без координат «зон»
из реальных роликов.
"""
from __future__ import annotations

from typing import Any

from .contract import Detection, Event, SceneState, Track, Zone

MOCK_VERSION = "builtin-mock-1"


def _motion(clip: Any) -> list[float]:
    import numpy as np  # noqa: WPS433

    prev = None
    out: list[float] = []
    for i in range(len(clip.times)):
        img = clip.frame(i)
        small = img[::8, ::8].astype("float32").mean(axis=2)
        out.append(0.0 if prev is None else float(np.abs(small - prev).mean()))
        prev = small
    peak = max(out) if out and max(out) > 0 else 1.0
    return [v / peak for v in out]


def detect(clip: Any, every_n: int = 1) -> list[Detection]:
    dets: list[Detection] = []
    for i, t in enumerate(clip.times):
        if i % max(1, every_n):
            continue
        x = 0.3 + 0.2 * ((i % 20) / 20.0)
        dets.append(Detection(t, "person", (x, 0.35, x + 0.2, 0.75), 0.9, None, "mock"))
        dets.append(Detection(t, "safe", (0.7, 0.1, 0.95, 0.3), 0.8, None, "mock"))
    return dets


def track(clip: Any, dets: list[Detection]) -> tuple[list[Track], list[Zone]]:
    by_label: dict[str, Track] = {}
    for d in dets:
        tr = by_label.setdefault(d.label, Track(len(by_label) + 1, d.label,
                                                "cashier" if d.label == "person" else "unknown", 0.5))
        tr.boxes.append((d.t, d.bbox))
    return list(by_label.values()), []


def estimate(clip: Any, tracks: list[Track]) -> list:
    return []


def pose_signals(poses: list, tracks: list[Track], zones: list[Zone]) -> list[SceneState]:
    return []


def scene_states(clip: Any, dets: list, tracks: list, zones: list, pose_states: list) -> list[SceneState]:
    energy = _motion(clip)
    n_persons: dict[float, int] = {}
    for d in dets:
        if d.label == "person":
            n_persons[d.t] = n_persons.get(d.t, 0) + 1
    return [SceneState(t, {"motion_energy": e, "n_persons": float(n_persons.get(t, 0)),
                           "cashier_present": 1.0 if n_persons.get(t) else 0.0})
            for t, e in zip(clip.times, energy)]


def run_rules(states: list[SceneState], tracks: list, zones: list, catalog: dict) -> list[Event]:
    if not states:
        return []
    peak = max(states, key=lambda s: s.signals.get("motion_energy", 0.0))
    if peak.signals.get("motion_energy", 0.0) < 0.5:
        return []
    return [Event("V99", max(0.0, peak.t - 1.0), peak.t + 1.0, peak.t, 0.7, "rules",
                  ["[MOCK] резкое движение в кадре"], {}, [], {"mock": True})]


def score(clip: Any) -> tuple[list[tuple[float, float]], list[Event]]:
    energy = _motion(clip)
    return [(t, round(0.8 * e, 4)) for t, e in zip(clip.times, energy)], []


def verify(clip: Any, events: list[Event], budget_s: float) -> list[Event]:
    return [Event(e.type_id, e.t_start, e.t_end, e.t_peak, e.confidence, "vlm", list(e.facts), dict(e.bboxes),
                  list(e.track_ids), {"mock": True, "vlm_verdict": "confirmed"}) for e in events]


def discover(clip: Any, budget_s: float) -> list[Event]:
    return []


def fuse(clip: Any, events: list[Event], risk: list, catalog: dict, job_id: str) -> Any:
    """Фейк ml.fusion (только mock-режим): встроенное слияние оркестратора."""
    from .fallbacks import fallback_fuse

    res = fallback_fuse(clip, events, risk, catalog, job_id, {})
    res.meta["fusion"] = "builtin-mock"
    return res


BUILTIN: dict[str, Any] = {
    "ml.detect.detect": detect,
    "ml.track.track": track,
    "ml.pose.estimate": estimate,
    "ml.pose.pose_signals": pose_signals,
    "ml.scene.scene_states": scene_states,
    "ml.rules.run_rules": run_rules,
    "ml.temporal.score": score,
    "ml.vlm.verify": verify,
    "ml.vlm.discover": discover,
    "ml.fusion.fuse": fuse,
}
__version__ = MOCK_VERSION
