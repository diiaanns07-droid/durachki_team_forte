"""Полнота анализа: meta.coverage (доля длительности, реально проанализированная стадией) и
meta.completeness в формате contracts/extensions.py (A01).

Принципы:
* coverage считается по фактическим таймстемпам (кадры клипа, состояния сцены, точки риск-кривой),
  а не по флагу «стадия отработала»; провал между кадрами больше ``max_gap_s`` — непокрытое время;
* неизвестно ≠ 0 и ≠ 1: если стадия не сообщает, что именно она смотрела (vlm_discover), coverage=None;
* неполный анализ (сбой обязательной стадии ИЛИ coverage обязательной стадии < min_required)
  никогда не превращается в no_violation.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from .config import cfg_get

# Статусы стадий в терминах contracts.extensions.STAGE_STATUSES
_STATUS_MAP = {"ok": "ok", "cached": "ok", "error": "failed", "crashed": "failed", "missing": "failed",
               "timeout": "timeout", "skipped": "skipped", "disabled": "skipped", "pending": "skipped",
               "cancelled": "failed"}

# Стадии, чьё покрытие = покрытие кадров клипа, если стадия завершилась успешно (контракт: стадия
# обрабатывает клип целиком, а выход содержит только найденное — пустой кадр выхода не оставляет).
_CLIP_WIDE = ("detect", "track", "pose")


def interval_coverage(times: Iterable[float], duration: Optional[float], max_gap_s: float = 2.0) -> Optional[float]:
    """Доля [0, duration], покрытая таймстемпами: кадр в t представляет [t, t + шаг), где шаг —
    медианный собственный шаг выборки, но не больше max_gap_s (редкая выборка ≠ полное покрытие)."""
    ts = sorted({float(t) for t in times if t is not None})
    if not duration or duration <= 0:
        return None
    if not ts:
        return 0.0
    if len(ts) == 1:
        win = max_gap_s
    else:
        diffs = sorted(b - a for a, b in zip(ts, ts[1:]) if b > a)
        step = diffs[len(diffs) // 2] if diffs else max_gap_s
        win = min(step, max_gap_s)
    covered = 0.0
    cur_a, cur_b = None, None
    for t in ts:
        a, b = max(0.0, t), min(duration, t + win)
        if b <= a:
            continue
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                covered += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        covered += cur_b - cur_a
    return round(min(1.0, covered / duration), 4)


def covered_intervals(times: Iterable[float], duration: Optional[float], max_gap_s: float = 2.0) -> list[list[float]]:
    """Интервалы [a, b], покрытые таймстемпами (кадр в t представляет [t, t + шаг)). Хвост/начало короче
    ~1.5 шага считаются покрытыми: это артефакт сетки выборки, а не непросмотренное время."""
    ts = sorted({float(t) for t in times if t is not None})
    if not duration or duration <= 0 or not ts:
        return []
    if len(ts) == 1:
        win = max_gap_s
    else:
        diffs = sorted(b - a for a, b in zip(ts, ts[1:]) if b > a)
        win = min(diffs[len(diffs) // 2] if diffs else max_gap_s, max_gap_s)
    out: list[list[float]] = []
    for k, t in enumerate(ts):
        a, b = max(0.0, t), min(duration, t + win)
        if k == 0 and a <= 1.5 * win:
            a = 0.0
        if k == len(ts) - 1 and duration - t <= 1.5 * win:
            b = duration
        if b <= a:
            continue
        if out and a <= out[-1][1] + 1e-9:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _intersect(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            out.append([lo, hi])
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def canonical_coverage(records: Mapping[str, Any], outputs: Mapping[str, Any], clip_info: Mapping[str, Any],
                       probe: Mapping[str, Any], cfg: Mapping, required: list[str],
                       stages_meta: Mapping[str, Any]) -> dict:
    """meta.coverage в формате contracts.extensions.compute_coverage (A01) + by_stage (доли по стадиям).

    Ядро = пересечение интервалов, реально покрытых обязательными стадиями с таймстемпами
    (кадры ingest; состояния scene; риск temporal — если обязательна). detect/track/rules покрывают то же,
    что их входы. Непокрытое — unknown_spans: «не видели» ≠ «нарушений нет»."""
    max_gap = float(cfg_get(cfg, "coverage.max_gap_s", 2.0))
    tol = float(cfg_get(cfg, "coverage.container_tolerance", 0.15))
    duration = reference_duration(clip_info, probe) or 0.0
    container = float(probe.get("duration") or 0.0)
    eff = container if (duration and container > duration * (1.0 + tol)) else duration
    core = covered_intervals(clip_info.get("times") or [], duration, max_gap) if _ok(records.get("ingest")) else []
    for stage in ("scene", "temporal"):
        if stage not in required:
            continue
        if not _ok(records.get(stage)):
            core = []
            continue
        values = (outputs.get(stage) or (None,))[0] or []
        times = [t for t, _r in values] if stage == "temporal" else [getattr(x, "t", None) for x in values]
        core = _intersect(core, covered_intervals(times, duration, max_gap))
    until = core[-1][1] if core else 0.0
    unknown, cur = [], 0.0
    for a, b in core:
        if a > cur:
            unknown.append([cur, a])
        cur = b
    try:
        from contracts.extensions import compute_coverage  # канон A01
    except ImportError:
        compute_coverage = None
    if compute_coverage is not None:
        cov = dict(compute_coverage(eff, until, unknown_spans=unknown, stages=stages_meta, required=required,
                                    source="backend.pipeline"))
    else:  # та же формула, пока ветка A01 не смерджена
        analyzed = max(0.0, min(until, eff) - sum(b - a for a, b in unknown))
        st = {k: (v.get("status") if isinstance(v, Mapping) else v) for k, v in stages_meta.items()}
        req_ok = all(st.get(r) in ("ok", "done", "cached", "mock") for r in required)
        cov = {"ratio": round(analyzed / eff, 4) if eff else 0.0, "analyzed_sec": round(analyzed, 3),
               "duration_sec": round(eff, 3), "unknown_spans": [[round(a, 3), round(b, 3)] for a, b in unknown],
               "stages": {k: str(v) for k, v in st.items() if v is not None}, "required": list(required),
               "complete": bool(req_ok and until >= eff - 1e-6), "source": "backend.pipeline"}
    gaps = [list(g) for g in unknown] + ([[round(until, 3), round(eff, 3)]] if eff and until < eff - 1e-6 else [])
    cov["gaps"] = [[round(a, 3), round(b, 3)] for a, b in gaps]   # все непросмотренные участки, включая хвост
    cov["by_stage"] = stage_coverage(records, outputs, clip_info, probe, cfg)
    return cov


def reference_duration(clip_info: Mapping[str, Any], probe: Mapping[str, Any]) -> Optional[float]:
    d = float(clip_info.get("duration") or 0.0)
    if d <= 0:
        d = float(probe.get("duration") or 0.0)
    return d or None


def stage_coverage(records: Mapping[str, Any], outputs: Mapping[str, Any], clip_info: Mapping[str, Any],
                   probe: Mapping[str, Any], cfg: Mapping) -> dict[str, Optional[float]]:
    """coverage по стадиям: 0..1 или None (неизвестно)."""
    max_gap = float(cfg_get(cfg, "coverage.max_gap_s", 2.0))
    tol = float(cfg_get(cfg, "coverage.container_tolerance", 0.15))
    duration = reference_duration(clip_info, probe)
    out: dict[str, Optional[float]] = {}

    ingest = interval_coverage(clip_info.get("times") or [], duration, max_gap)
    container = float(probe.get("duration") or 0.0)
    if ingest is not None and duration and container > duration * (1.0 + tol):
        ingest = round(ingest * duration / container, 4)  # декодер дошёл не до конца файла
    out["ingest"] = ingest if _ok(records.get("ingest")) else 0.0

    for stage in _CLIP_WIDE:
        out[stage] = ingest if _ok(records.get(stage)) else (None if stage not in records else 0.0)
    for stage, idx in (("scene", 0), ("temporal", 0)):
        rec = records.get(stage)
        if not _ok(rec):
            out[stage] = None if rec is None else 0.0
            continue
        values = (outputs.get(stage) or (None,))[idx]
        if stage == "temporal":
            times = [t for t, _r in (values or [])]
        else:
            times = [getattr(s, "t", None) for s in (values or [])]
        out[stage] = interval_coverage(times, duration, max_gap) if times else (0.0 if duration else None)
        if out[stage] is not None and ingest is not None:
            out[stage] = min(out[stage], ingest)
    out["rules"] = out.get("scene") if _ok(records.get("rules")) else (None if "rules" not in records else 0.0)
    out["refine"] = None       # окна вокруг кандидатов, а не всё видео: см. meta.refine
    out["vlm_verify"] = None   # проверяет окна кандидатов, а не всё видео
    out["vlm_discover"] = None  # модуль не сообщает просмотренные окна → неизвестно
    out["fusion"] = 1.0 if _ok(records.get("fusion")) else (None if "fusion" not in records else 0.0)
    return out


def _ok(rec: Any) -> bool:
    return rec is not None and getattr(rec, "status", None) in ("ok", "cached")


def coverage_gate(coverage: Mapping[str, Any], cfg: Mapping) -> list[str]:
    """Причины неполноты по покрытию ядра (пусто — покрытие достаточное). coverage — canonical_coverage()."""
    if not cfg_get(cfg, "coverage.gate", True):
        return []
    need = float(cfg_get(cfg, "coverage.min_required", 0.98))
    ratio = coverage.get("ratio")
    if ratio is None:
        return ["покрытие ядра неизвестно"]
    if float(ratio) + 1e-9 < need:
        spans = coverage.get("gaps") or coverage.get("unknown_spans") or []
        return [f"проанализировано {float(ratio):.0%} длительности < {need:.0%}"
                + (f", не просмотрено: {spans[:3]}" if spans else "")]
    return []


def completeness(records: Mapping[str, Any], required: list[str], complete: bool, duration: Optional[float],
                 analyzed_until: Optional[float], reason: Optional[str], mock: bool) -> dict:
    """meta.completeness в формате contracts.extensions.Completeness.to_dict()."""
    stages = {}
    for name, rec in records.items():
        st = _STATUS_MAP.get(getattr(rec, "status", "skipped"), "failed")
        if st == "ok" and (mock or getattr(rec, "origin", "") == "builtin-mock"):
            st = "mock"
        stages[name] = st
    return {"complete": bool(complete), "analyzed_until_sec": analyzed_until, "duration_sec": duration,
            "stages": stages, "required_stages": list(required), "reason": reason}
