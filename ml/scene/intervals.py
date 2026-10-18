"""Интервалы для A09: непрерывные участки статуса true у сигналов и предикатов + покрытие (coverage).

    from ml.scene.intervals import intervals, coverage
    iv = intervals(states)              # {"cash_left_after_leave": [{"t_start", "t_end", "t_peak", ...}], ...}
    cov = coverage(states)              # доля сэмплов, где каждый предикат/сигнал известен; для meta.coverage

t_start — момент перехода (начало нарушения: «отошла», «вошёл», «поднесла»), t_peak — по политике
(onset — t_start; mid — середина; max — сэмпл максимального значения сигнала). Политику выбирает замер L2.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .tri import runs


def _series(states: Sequence[Any], name: str, kind: str) -> tuple[list[str], list[float], list[str]]:
    if kind == "pred":
        st = [getattr(s, "pred_status", {}).get(name, "unknown") for s in states]
        val = [float(getattr(s, "pred", {}).get(name, 0.0)) for s in states]
        why = [getattr(s, "pred_why", {}).get(name, "") for s in states]
    else:
        st = [getattr(s, "status", {}).get(name, "unknown") for s in states]
        val = [float(s.signals.get(name, 0.0)) for s in states]
        why = [getattr(s, "why", {}).get(name, "") for s in states]
    return st, val, why


def intervals(
    states: Sequence[Any],
    names: Optional[Iterable[str]] = None,
    kind: str = "pred",
    policy: str = "onset",
    min_s: float = 0.0,
) -> dict[str, list[dict[str, Any]]]:
    """Интервалы статуса true. kind = pred (предикаты) | signal (SIGNALS)."""
    if not states:
        return {}
    t = np.array([s.t for s in states], dtype=np.float64)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.2
    if names is None:
        src = getattr(states[0], "pred_status" if kind == "pred" else "status", {}) or {}
        names = list(src)
    out: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        st, val, _why = _series(states, name, kind)
        is_t = np.array([x == "true" for x in st])
        items = []
        for i0, i1 in runs(is_t):
            dur = float(t[i1] - t[i0] + dt)
            if dur < min_s:
                continue
            seg = np.asarray(val[i0 : i1 + 1])
            if policy == "mid":
                peak = float((t[i0] + t[i1]) / 2.0)
            elif policy == "max":
                peak = float(t[i0 + int(np.argmax(seg))])
            else:
                peak = float(t[i0])
            vis = [getattr(states[i], "visibility", {}).get(name, 1.0) for i in range(i0, i1 + 1)] if kind == "signal" else []
            prov = getattr(states[i0], "provenance", {}).get(name, "") if kind == "signal" else "predicate"
            items.append({
                "t_start": float(t[i0]),
                "t_end": float(t[i1] + dt),
                "t_peak": peak,
                "duration_s": round(dur, 3),
                "value_max": float(seg.max()) if len(seg) else 0.0,
                "visibility_min": float(min(vis)) if vis else None,
                "source": prov,
            })
        out[name] = items
    return out


def coverage(states: Sequence[Any], names: Optional[Iterable[str]] = None, kind: str = "pred") -> dict[str, Any]:
    """Доля сэмплов с известным статусом (true/false) по каждому имени + причины неизвестного.

    Неполный анализ ≠ «нарушений нет»: A03/A12 кладут это в meta.coverage и не превращают
    низкое покрытие в no_violation без проверки."""
    if not states:
        return {"n": 0, "per_name": {}, "frames_quality_ok": 0.0}
    if names is None:
        src = getattr(states[0], "pred_status" if kind == "pred" else "status", {}) or {}
        names = list(src)
    per: dict[str, Any] = {}
    for name in names:
        st, _v, why = _series(states, name, kind)
        known = sum(1 for x in st if x in ("true", "false"))
        reasons: dict[str, int] = {}
        for x, w in zip(st, why):
            if x not in ("true", "false"):
                reasons[w or "no_evidence"] = reasons.get(w or "no_evidence", 0) + 1
        per[name] = {"known_frac": round(known / len(st), 4), "unknown_reasons": reasons}
    q = np.array([float(getattr(s, "quality", 1.0)) for s in states])
    return {"n": len(states), "frames_quality_ok": round(float(np.mean(q >= 0.35)), 4), "per_name": per}
