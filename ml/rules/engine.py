"""Движок правил: ряды сигналов → события (``Event``, source="rules").

Конвейер для каждого правила из ``rules.yaml``:

1. ``strength`` (нечёткое выражение 0..1) и ``gate`` в **интервальной** логике:
   нижняя граница = доказано наблюдениями, верхняя = возможно (неизвестные сигналы);
   отсутствие сигнала никогда не трактуется как отрицание;
2. сглаживание (медиана/среднее в окне ``smooth.window_s``) — убирает одиночные вспышки;
3. гистерезис ``on``/``off`` по доказанной силе → маска «условия наблюдаются»;
4. дебаунс: склейка сегментов с разрывом ≤ ``merge_gap_s``, отбрасывание короче ``min_duration_s``;
5. для правил «по бездействию» — проверка полноты этапа (предшествующее состояние,
   начало не обрезано, покрытие сигналами); иначе ``insufficient_context``;
6. оценка (``extra.assessment``):
   ``candidate_violation`` — правило вида ``violation`` + у типа есть проверенная норма A13;
   ``review_observation`` — наблюдение для проверяющего без юридического утверждения
   (правило-наблюдение или норма не проверена/отсутствует);
   ``insufficient_context`` — условие могло выполняться, но этап не наблюдался полностью;
7. ``t_peak`` — фактический момент с доказательством (политика ``evidence`` по умолчанию),
   ``confidence`` от силы и длительности (монотонно по длительности), факты по шаблонам,
   ``bboxes`` участников на ``t_peak`` из треков, ``extra.trace`` — почему сработало.

Затем по всем правилам: слияние пересекающихся событий одного типа, «уступка» типов
(``yield_to``), NMS одного типа в окне ``nms_window_s``, порог уверенности, лимит числа событий.
"""
from __future__ import annotations

import math
import string
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .catalog import legal_basis_of, norm_status, resolve_catalog, type_info
from .config import Clause, RuleSpec, RulesConfig, load_rules_config
from .expr import EvalContext
from .signals import SignalSeries, build_series, derive_from_tracks, polygon_bbox, track_box_at

try:
    from contracts.sentinel_contracts import Event  # noqa
except ImportError:  # пока ветка A01 не смерджена
    from contracts.sentinel_contracts import Event  # noqa

__all__ = [
    "run_rules", "evaluate", "RulesReport", "SIGNAL_RU", "ASSESSMENTS", "CANDIDATE", "REVIEW", "INSUFFICIENT",
    "assessment_of", "split_by_assessment", "hysteresis_mask", "mask_to_segments", "merge_segments",
    "confidence_for", "format_fact",
]

SOURCE = "rules"
CANDIDATE = "candidate_violation"
REVIEW = "review_observation"
INSUFFICIENT = "insufficient_context"
ASSESSMENTS = (CANDIDATE, REVIEW, INSUFFICIENT)
_PRIORITY = {CANDIDATE: 3, REVIEW: 2, INSUFFICIENT: 1}

# Человекочитаемые названия сигналов — для trace/UI («почему сработало»).
SIGNAL_RU: dict[str, str] = {
    "cashier_present": "кассир в кадре",
    "cashier_in_workplace": "кассир на рабочем месте",
    "n_persons": "число людей в кадре",
    "outsider_in_cashier_area": "посторонний в зоне кассира",
    "collector_present": "инкассатор в кадре",
    "safe_open": "сейф открыт",
    "door_open": "дверь открыта",
    "cash_visible": "наличные видны",
    "cash_bag_visible": "инкассаторская сумка видна",
    "phone_in_hand": "телефон в руке",
    "phone_to_ear": "телефон у уха",
    "eating_drinking": "приём пищи/напитков",
    "hands_hidden": "руки скрыты от камеры",
    "cashier_facing_away": "кассир спиной к камере",
    "workplace_unattended_with_cash": "рабочее место с наличными без присмотра",
    "handover_motion": "движение передачи",
    "motion_energy": "интенсивность движения",
}


DETECTIONS = ("violation_candidate", "observation", "insufficient_context")


def detection_of(rule: RuleSpec, assessment: str) -> str:
    """Ось обнаружения (поведение по видео), независимая от правовой опоры."""
    if assessment == INSUFFICIENT:
        return "insufficient_context"
    return "violation_candidate" if rule.kind == "violation" else "observation"


def assessment_of(ev: Any) -> str:
    """Оценка события правил (для A12/fusion): candidate_violation | review_observation | insufficient_context."""
    extra = getattr(ev, "extra", None) or {}
    a = extra.get("assessment")
    return a if a in ASSESSMENTS else REVIEW


def split_by_assessment(events: Sequence[Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {a: [] for a in ASSESSMENTS}
    for ev in events:
        out[assessment_of(ev)].append(ev)
    return out


# --------------------------------------------------------------------------- helpers

def _r(x: Any, nd: int = 3) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if math.isfinite(x) else None


def _rf(x: Any, nd: int = 3) -> float:
    v = _r(x, nd)
    return 0.0 if v is None else v


def smooth(x: np.ndarray, dt: float, window_s: float, method: str) -> np.ndarray:
    if method == "none" or window_s <= 0 or x.shape[0] < 3:
        return x
    k = int(round(window_s / max(dt, 1e-6)))
    if k < 2:
        return x
    if k % 2 == 0:
        k += 1
    k = min(k, x.shape[0] if x.shape[0] % 2 == 1 else x.shape[0] - 1)
    if k < 3:
        return x
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(xp, k)
    if method == "median":
        return np.median(win, axis=1)
    return win.mean(axis=1)


def hysteresis_mask(x: np.ndarray, on: float, off: float) -> np.ndarray:
    """Включаемся при x ≥ on, держимся пока x ≥ off."""
    mask = np.zeros(x.shape[0], dtype=bool)
    active = False
    for i, v in enumerate(x):
        active = bool(v >= off) if active else bool(v >= on)
        mask[i] = active
    return mask


def mask_to_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Сегменты подряд идущих True как пары индексов (включительно)."""
    segs: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, int(mask.shape[0]) - 1))
    return segs


def merge_segments(times: np.ndarray, segs: list[tuple[int, int]], gap_s: float) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for s in segs:
        if out and float(times[s[0]] - times[out[-1][1]]) <= gap_s + 1e-9:
            out[-1] = (out[-1][0], s[1])
        else:
            out.append(s)
    return out


def confidence_for(rule: RuleSpec, mean_strength: float, max_strength: float, duration: float,
                   n_frames: int = 0, severity_factor: float = 1.0) -> tuple[float, dict[str, float]]:
    """Уверенность события; неубывающая по ``duration`` при фиксированной силе."""
    dur_term = 1.0 - math.exp(-max(0.0, duration) / rule.conf_tau_s)
    if rule.conf_expr is not None:
        from .expr import evaluate_scalar

        raw = evaluate_scalar(rule.conf_expr, {
            "mean_strength": mean_strength, "max_strength": max_strength, "duration": duration,
            "dur_term": dur_term, "base": rule.conf_base, "w_strength": rule.conf_w_strength,
            "w_duration": rule.conf_w_duration, "tau_s": rule.conf_tau_s, "n_frames": float(n_frames),
        })
        terms = {"expr": _rf(raw)}
    else:
        s_term = rule.conf_w_strength * float(np.clip(mean_strength, 0.0, 1.0))
        d_term = rule.conf_w_duration * dur_term
        raw = rule.conf_base + s_term + d_term
        terms = {"base": _rf(rule.conf_base), "strength": _rf(s_term), "duration": _rf(d_term)}
    factor = rule.weight * severity_factor
    conf = float(np.clip(raw * factor, rule.conf_min, rule.conf_max))
    terms["factor"] = _rf(factor)
    return conf, terms


class _UnknownValue(Exception):
    pass


class _SafeFormatter(string.Formatter):
    """Только простые имена из контекста: без атрибутов/индексов; неизвестное значение → факт не печатается."""

    def get_field(self, field_name: str, args: Sequence[Any], kwargs: Mapping[str, Any]) -> Any:
        if not field_name.isidentifier():
            raise _UnknownValue(field_name)
        v = kwargs.get(field_name)
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            raise _UnknownValue(field_name)
        return v, field_name

    def format_field(self, value: Any, format_spec: str) -> str:
        try:
            return format(value, format_spec)
        except (TypeError, ValueError):
            return str(value)


_FMT = _SafeFormatter()


def format_fact(template: str, ctx: Mapping[str, Any]) -> Optional[str]:
    """Факт по шаблону; ``None``, если какое-то значение неизвестно (не утверждаем невидимое)."""
    try:
        return _FMT.vformat(template, (), dict(ctx))
    except (_UnknownValue, ValueError, IndexError, KeyError):
        return None


# --------------------------------------------------------------------------- report

@dataclass
class RulesReport:
    events: list[Any]
    series: SignalSeries
    strengths: dict[str, np.ndarray] = field(default_factory=dict)   # доказанная сила (нижняя граница)
    possible: dict[str, np.ndarray] = field(default_factory=dict)    # возможная сила (верхняя граница)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    catalog_source: str = "fallback"
    config: Optional[RulesConfig] = None

    def risk_curve(self, step_s: float = 0.0) -> list[tuple[float, float]]:
        """Кривая риска по правилам: максимум доказанной силы (правила-наблюдения ×0.5)."""
        t = self.series.times
        if t.shape[0] == 0:
            return []
        risk = np.zeros(t.shape[0])
        kinds = {r.id: r.kind for r in (self.config.rules if self.config else [])}
        for rid, arr in self.strengths.items():
            w = 1.0 if kinds.get(rid) == "violation" else 0.5
            np.maximum(risk, arr * w, out=risk)
        pts = [(_rf(ti, 2), _rf(ri)) for ti, ri in zip(t, np.clip(risk, 0, 1))]
        if step_s > 0 and len(pts) > 1:
            out, last = [], -1e9
            for p in pts:
                if p[0] - last >= step_s - 1e-9:
                    out.append(p)
                    last = p[0]
            return out
        return pts

    def coverage(self) -> dict[str, Any]:
        """Полнота наблюдений для meta.coverage: неполный анализ ≠ «нарушений нет».

        ``signals`` — доля кадров с известным значением сигнала; ``rules`` — доля кадров, где известны
        все сигналы обязательных предикатов правила (0 — правило не могло ничего доказать).
        """
        s = self.series
        out: dict[str, Any] = {"n_frames": s.n,
                               "t_first": _r(s.times[0], 2) if s.n else None,
                               "t_last": _r(s.times[-1], 2) if s.n else None,
                               "signals": {}, "rules": {}}
        if s.n == 0:
            return out
        for nm in sorted(s.signals):
            out["signals"][nm] = _rf(s.known(nm).mean(), 3)
        for rule in (self.config.enabled_rules if self.config else []):
            names = sorted(rule.mandatory_names)
            if not names:
                continue
            known = np.ones(s.n, dtype=bool)
            for nm in names:
                known &= s.known(nm)
            out["rules"][rule.id] = _rf(known.mean(), 3)
        out["not_evaluable_rules"] = sorted(r for r, v in out["rules"].items() if v == 0.0)
        return out

    def summary(self) -> dict[str, Any]:
        groups = split_by_assessment(self.events)
        det: dict[str, int] = {d: 0 for d in DETECTIONS}
        for e in self.events:
            det[e.extra.get("detection", "observation")] = det.get(e.extra.get("detection", "observation"), 0) + 1
        return {"n_events": len(self.events), **{a: len(v) for a, v in groups.items()},
                "detection": det,
                "absent_signals": sorted(self.series.absent), "derived_signals": dict(self.series.derived),
                "catalog_source": self.catalog_source, "warnings": list(self.warnings)}


# --------------------------------------------------------------------------- core

def _get(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)


def _bboxes_at(rule: RuleSpec, t: float, tracks: Sequence[Any], zones: Sequence[Any],
               cfg: RulesConfig) -> tuple[dict[str, tuple[float, ...]], list[int]]:
    boxes: dict[str, tuple[float, ...]] = {}
    ids: list[int] = []
    for spec in rule.highlight:
        if len(boxes) >= cfg.max_bboxes:
            break
        kind, _, value = spec.partition(":")
        if kind == "zone":
            for z in zones or ():
                if _get(z, "name") == value and _get(z, "polygon"):
                    bb = polygon_bbox(_get(z, "polygon"))
                    if bb:
                        boxes[f"zone:{value}"] = tuple(_rf(v, 4) for v in bb)
            continue
        if kind not in ("role", "label"):
            continue
        for tr in tracks or ():
            label, role, tid = _get(tr, "label"), _get(tr, "role"), _get(tr, "track_id")
            if kind == "role" and not (label == "person" and role == value):
                continue
            if kind == "label" and label != value:
                continue
            bb = track_box_at(tr, t, cfg.bbox_tol_s)
            if bb is None:
                continue
            boxes[f"{role if kind == 'role' else label}#{tid}"] = tuple(_rf(min(1.0, max(0.0, v)), 4) for v in bb)
            try:
                ids.append(int(tid))
            except (TypeError, ValueError):
                pass
            if len(boxes) >= cfg.max_bboxes:
                break
    return boxes, sorted(set(ids))


def _signal_stats(series: SignalSeries, names: Sequence[str], i0: int, i1: int, ip: int) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for nm in sorted(names):
        arr = series.signals.get(nm)
        if arr is None or arr.shape[0] == 0:
            continue
        seg = arr[i0:i1 + 1]
        known = np.isfinite(seg)
        st: dict[str, Any] = {"at_peak": _r(arr[ip]), "known_frac": _rf(known.mean(), 2)}
        if known.any():
            st.update(mean=_r(seg[known].mean()), min=_r(seg[known].min()), max=_r(seg[known].max()))
        else:
            st.update(mean=None, min=None, max=None)
        if nm in series.derived:
            st["source"] = series.derived[nm]
        out[nm] = st
    return out


def _peak_index(rule: RuleSpec, times: np.ndarray, strength: np.ndarray, i0: int, i1: int,
                policy: Optional[str] = None) -> int:
    policy = policy or rule.t_peak_policy
    seg = strength[i0:i1 + 1]
    if policy == "onset":
        return i0
    if policy == "max":
        return i0 + int(np.argmax(seg))
    if policy == "center":
        return int(np.argmin(np.abs(times - (times[i0] + times[i1]) / 2.0)))
    if policy == "onset_plus":
        t = float(np.clip(times[i0] + rule.t_peak_offset_s, times[0], times[i1]))
        return int(np.argmin(np.abs(times - t)))
    # evidence: самый ранний кадр сегмента с доказательством, близким к максимальному
    thr = max(rule.on, float(seg.max()) - rule.evidence_tol)
    hits = np.nonzero(seg >= thr - 1e-9)[0]
    return i0 + int(hits[0]) if hits.size else i0


def _transition_index(times: np.ndarray, tlo: np.ndarray, thi: np.ndarray, i0: int, i1: int,
                      lookback_s: float) -> tuple[Optional[int], str]:
    """Наблюдаемое начало действия: где trigger стал доказанно истинным для ЭТОГО события.

    Берётся первый кадр сегмента, где trigger истинен, и от него назад — начало непрерывного
    отрезка «trigger истинен» (не раньше ``t_start − lookback_s``). Перед ним доказанно ложно →
    ``false→true``; «неизвестно» → ``unknown→true`` (начало могло быть раньше). Если отрезок тянется
    за границу окна или с начала ролика — перехода не видно (``None``), используется fallback.
    Более ранние короткие эпизоды trigger в окне не выбираются.
    """
    true_idx = np.nonzero(tlo[i0:i1 + 1] >= 0.5)[0]
    if true_idx.size == 0:
        return None, "trigger_not_true"
    j = i0 + int(true_idx[0])
    j_start = int(np.searchsorted(times, float(times[i0]) - lookback_s - 1e-9, side="left"))
    while j - 1 >= j_start and tlo[j - 1] >= 0.5:
        j -= 1
    if j == 0 or j - 1 < j_start:
        return None, "not_observed"
    return j, ("false→true" if thi[j - 1] < 0.5 else "unknown→true")


def _clause_status(lo: float, hi: float) -> str:
    if lo >= 0.5:
        return "true"
    if hi < 0.5:
        return "false"
    return "unknown"


def _clause_trace(kind: str, clause: Any, bounds: tuple[np.ndarray, np.ndarray], i0: int, i1: int,
                  ip: int) -> dict[str, Any]:
    lo, hi = bounds
    return {"kind": kind, "id": clause.id, "label": clause.label, "expr": clause.expr.source,
            "at_peak": _clause_status(float(lo[ip]), float(hi[ip])),
            "frac_true": _rf((lo[i0:i1 + 1] >= 0.5).mean(), 3),
            "frac_unknown": _rf(((lo[i0:i1 + 1] < 0.5) & (hi[i0:i1 + 1] >= 0.5)).mean(), 3)}


def _make_event(rule: RuleSpec, series: SignalSeries, strength: np.ndarray, i0: int, i1: int,
                assessment: str, reasons: list[str], tracks: Sequence[Any], zones: Sequence[Any],
                catalog: Mapping[str, Any], cfg: RulesConfig, from_anchor: bool = False,
                aux: Optional[dict[str, Any]] = None) -> Any:
    aux = aux or {}
    times = series.times
    info = type_info(catalog, rule.type_id)
    norm = norm_status(catalog, rule.type_id)
    sev_factor = float(cfg.severity_factor.get(info["severity"], 1.0))
    seg_start = float(times[i0])
    seg = strength[i0:i1 + 1]
    mean_s, max_s = float(seg.mean()), float(seg.max())
    policy = cfg.t_peak_policy_override or rule.t_peak_policy
    transition: dict[str, Any] = {"policy": policy}
    if policy == "transition" and aux.get("trigger") is not None:
        tlo, thi = aux["trigger"]
        jt, kind = _transition_index(times, tlo, thi, i0, i1, rule.t_peak_lookback_s)
        transition.update(kind=kind, trigger_expr=aux.get("trigger_src"))
        if jt is not None:
            ip = jt
        else:
            ip = _peak_index(rule, times, strength, i0, i1, rule.t_peak_fallback)
            transition["fallback"] = rule.t_peak_fallback
    elif policy == "transition":
        ip = _peak_index(rule, times, strength, i0, i1, rule.t_peak_fallback)
        transition.update(kind="no_trigger", fallback=rule.t_peak_fallback)
    else:
        ip = _peak_index(rule, times, strength, i0, i1, policy)
    t_peak = float(times[ip])
    if ip < i0:  # переход (начало действия) раньше, чем доказан весь набор предикатов
        i0_eff = ip
    else:
        i0_eff = i0
    t_start, t_end = float(times[i0_eff]), float(times[i1])
    duration = t_end - t_start + series.dt
    t_max = float(times[i0 + int(np.argmax(seg))])
    conf, terms = confidence_for(rule, mean_s, max_s, duration, i1 - i0 + 1, sev_factor)
    if any(n in series.derived for n in rule.names):
        conf *= cfg.derived_conf_factor  # грубые сигналы из треков — ниже доверие
        terms["derived_factor"] = cfg.derived_conf_factor
    conf_raw = conf  # до потолков review/insufficient — A12 восстанавливает ранжирование norm_review
    if assessment == REVIEW:
        conf = min(conf, cfg.observation_conf_cap)
    elif assessment == INSUFFICIENT:
        conf = min(conf, cfg.insufficient_conf_cap)
    terms["cap"] = rule.conf_max
    names = sorted(set(rule.names) | {"n_persons"})
    stats = _signal_stats(series, names, i0, i1, ip)
    ctx: dict[str, Any] = {
        "t_start": t_start, "t_end": t_end, "t_peak": t_peak, "duration": duration,
        "strength": mean_s, "strength_max": max_s, "confidence": conf, "type_name": info["name"],
    }
    for nm, st in stats.items():
        ctx[nm] = st["at_peak"]
        ctx[f"{nm}_mean"] = st["mean"]
        ctx[f"{nm}_min"] = st["min"]
        ctx[f"{nm}_max"] = st["max"]
    if from_anchor:
        # условие правила не доказано — печатаем только наблюдаемый якорь
        seen = [SIGNAL_RU.get(c.name, c.name) for c in (rule.anchor.comparisons if rule.anchor else ())]
        templates = [f"С {{t_start:.1f}} по {{t_end:.1f}} с наблюдается: {', '.join(seen)}."] if seen else []
    else:
        templates = list(rule.facts) + (list(rule.facts_stage) if assessment != INSUFFICIENT else [])
    facts = [f for f in (format_fact(tpl, ctx) for tpl in templates) if f]
    if assessment == INSUFFICIENT and reasons:
        facts.append("Недостаточно контекста: " + "; ".join(reasons) + ".")
    conditions = []
    for expr_obj in (rule.strength, rule.gate, rule.anchor):
        if expr_obj is None:
            continue
        for c in expr_obj.comparisons:
            v = series.signals.get(c.name)
            val = float(v[ip]) if v is not None and v.shape[0] else float("nan")
            conditions.append({"signal": c.name, "label": SIGNAL_RU.get(c.name, c.name), "op": c.op,
                               "threshold": c.threshold, "value": _r(val), "ok": c.check(val)})
    bboxes, track_ids = _bboxes_at(rule, t_peak, tracks, zones, cfg)
    evidence_t = sorted({_rf(t_peak, 2), _rf(t_max, 2), _rf(t_start, 2)})[:3]
    trace = {
        "rule_id": rule.id,
        "type_id": rule.type_id,
        "kind": rule.kind,
        "comment": rule.comment,
        "strength_expr": rule.strength.source,
        "gate_expr": rule.gate.source if rule.gate is not None else None,
        "gate_mode": rule.gate_mode,
        "anchor_expr": rule.anchor.source if rule.anchor is not None else None,
        "inaction_before": (rule.inaction.before.source if rule.inaction and rule.inaction.before else None),
        "thresholds": {"on": rule.on, "off": rule.off, "min_duration_s": rule.min_duration_s,
                       "merge_gap_s": rule.merge_gap_s, "smooth_s": rule.smooth_s},
        "t_peak_policy": rule.t_peak_policy,
        "strength": {"mean": _r(mean_s), "max": _r(max_s), "at_peak": _r(strength[ip])},
        "duration_s": _r(duration, 2),
        "signals": {nm: dict(st, label=SIGNAL_RU.get(nm, nm)) for nm, st in stats.items()},
        "conditions": conditions,
        "confidence_terms": terms,
        "norm_status": norm,
        "absent_signals": sorted(n for n in rule.names if n in series.absent),
        "derived_signals": {n: series.derived[n] for n in rule.names if n in series.derived},
    }
    clauses = []
    if ("legacy", "strength") in aux:
        clauses.append(_clause_trace("predicate", Clause("strength", rule.comment[:80] or "условие правила",
                                                         rule.strength), aux[("legacy", "strength")], i0, i1, ip))
    if ("legacy", "gate") in aux and rule.gate is not None:
        kind = "applicability" if rule.gate_mode == "strict" else "gate_lenient"
        clauses.append(_clause_trace(kind, Clause("gate", "условие применимости" if kind == "applicability"
                                                  else "мягкое условие (блокирует, только если доказанно не выполнено)",
                                                  rule.gate), aux[("legacy", "gate")], i0, i1, ip))
    if rule.applicability is not None and "app" in aux:
        clauses.append(_clause_trace("applicability", rule.applicability, aux["app"], i0, i1, ip))
    for c in rule.predicates:
        if ("pred", c.id) in aux:
            clauses.append(_clause_trace("predicate", c, aux[("pred", c.id)], i0, i1, ip))
    for c in rule.exclusions:
        if ("excl", c.id) in aux:
            clauses.append(_clause_trace("exclusion", c, aux[("excl", c.id)], i0, i1, ip))
    trace["clauses"] = clauses
    trace["interval"] = {"t_start": _r(t_start, 2), "t_end": _r(t_end, 2), "segment_t_start": _r(seg_start, 2),
                         "duration_s": _r(duration, 2)}
    trace["evidence"] = dict(transition, t_peak=_r(t_peak, 2))
    if from_anchor:
        trace["from_anchor"] = True
    boundary: dict[str, Any] = {}
    if rule.state_after_window_s > 0:
        t_dep = t_start
        sa_trig = aux.get("sa_trigger")
        if sa_trig is not None:
            jd, _ = _transition_index(times, sa_trig[0], sa_trig[1], i0, i1, rule.t_peak_lookback_s)
            if jd is not None:
                t_dep = float(times[jd])
        boundary["state_after"] = _state_after(series, t_dep, rule, tracks, cfg)
    if rule.person_side:
        boundary.update(_person_fields(rule, series, ip))
    negative = [{"fact": c.label, "strength": _rf((aux[("excl", c.id)][0][i0:i1 + 1] >= 0.5).mean(), 3)}
                for c in rule.exclusions if ("excl", c.id) in aux
                and (aux[("excl", c.id)][0][i0:i1 + 1] >= 0.5).any()]
    if negative:
        boundary["negative"] = negative
    return Event(
        type_id=rule.type_id,
        t_start=_rf(t_start, 2),
        t_end=_rf(t_end, 2),
        t_peak=_rf(t_peak, 2),
        confidence=_rf(conf),
        source=SOURCE,
        facts=facts,
        bboxes=bboxes,
        track_ids=track_ids,
        extra={
            "rule_id": rule.id,
            "rule_ids": [rule.id],
            "assessment": assessment,
            "detection": detection_of(rule, assessment),
            "legal_basis": legal_basis_of(norm["status"]),
            "assessment_reasons": list(reasons),
            "legal_claim": False,  # правила не делают юридических выводов — только признаки
            "type_name": info["name"],
            "severity": info["severity"],
            "in_catalog": info["in_catalog"],
            "norm_status": norm["status"],
            "confidence_raw": _rf(conf_raw),
            "evidence_t": evidence_t,
            **boundary,
            "trace": trace,
        },
    )


_VALUABLES_AFTER = ("cash_visible", "cash_bag_visible", "safe_open")


def _state_after(series: SignalSeries, t_dep: float, rule: RuleSpec, tracks: Sequence[Any],
                 cfg: RulesConfig) -> dict[str, Any]:
    """Стол и сейф в окне ПОСЛЕ ухода (A12: v02_after_window, кадр «после»). None = неизвестно.

    Значение — среднее по кадрам окна с известным сигналом; если известна меньше чем
    ``state_after_min_known`` доля кадров (свет выключен, окклюзия) — None: «стол пуст» не выводится.
    """
    times = series.times
    t_end = min(float(times[-1]), t_dep + rule.state_after_window_s)
    j0 = int(np.searchsorted(times, t_dep - 1e-9, side="left"))
    j1 = int(np.searchsorted(times, t_end + 1e-9, side="right"))
    out: dict[str, Any] = {"t_departure": _r(t_dep, 2), "window": [_r(t_dep, 2), _r(t_end, 2)],
                           "known_frac": {}, "t_after": None, "valuables_bboxes": {}}
    if j1 <= j0:
        for nm in _VALUABLES_AFTER:
            out[nm] = None
        return out
    present_any = np.zeros(j1 - j0, dtype=bool)
    known_any = np.zeros(j1 - j0, dtype=bool)
    for nm in _VALUABLES_AFTER:
        arr = series.signals.get(nm)
        if arr is None:
            out[nm], out["known_frac"][nm] = None, 0.0
            continue
        win = arr[j0:j1]
        known = np.isfinite(win)
        frac = float(known.mean())
        out["known_frac"][nm] = _rf(frac, 3)
        out[nm] = _r(win[known].mean()) if known.any() and frac + 1e-9 >= cfg.state_after_min_known else None
        known_any |= known
        present_any |= known & (np.nan_to_num(win, nan=0.0) >= 0.5)
    late = times[j0:j1] >= t_dep + rule.state_after_min_s - 1e-9
    for mask in (present_any & late, known_any & late):
        hits = np.nonzero(mask)[0]
        if hits.size:
            out["t_after"] = _r(times[j0 + int(hits[0])], 2)
            break
    if out["t_after"] is not None:
        for tr in tracks or ():
            label, tid = _get(tr, "label"), _get(tr, "track_id")
            if label in ("cash", "cash_bag", "safe_open"):
                bb = track_box_at(tr, float(out["t_after"]), cfg.bbox_tol_s)
                if bb is not None:
                    out["valuables_bboxes"][f"{label}#{tid}"] = tuple(_rf(min(1.0, max(0.0, v)), 4) for v in bb)
    return out


def _person_fields(rule: RuleSpec, series: SignalSeries, ip: int) -> dict[str, Any]:
    """Сторона человека и контекст для A12 (v01_client_side): только наблюдаемые значения, иначе None."""
    def at(name: str) -> Optional[float]:
        arr = series.signals.get(name)
        if arr is None or not np.isfinite(arr[ip]):
            return None
        return float(arr[ip])

    n_persons = at("n_persons")
    n_out = None
    if n_persons is not None:
        staff = sum(1 for nm in ("cashier_present", "collector_present") if (at(nm) or 0.0) >= 0.5)
        n_out = max(0, int(round(n_persons)) - staff)
    opened = [v for v in (at("safe_open"), at("cash_bag_visible")) if v is not None]
    return {"person_side": rule.person_side, "n_outsiders": n_out,
            "valuables_open": _r(max(opened)) if opened else None}


def _assess(rule: RuleSpec, catalog: Mapping[str, Any], cfg: RulesConfig) -> tuple[str, list[str]]:
    if rule.kind == "observation":
        return REVIEW, ["правило-наблюдение: само по себе не доказывает нарушение"]
    norm = norm_status(catalog, rule.type_id)
    if cfg.require_verified_norm and norm["status"] != "verified":
        return REVIEW, [f"нет проверенной применимой нормы для {rule.type_id} в каталоге A13 "
                        f"(статус: {norm['status']})"]
    return CANDIDATE, []


REJECT = "__reject__"  # маркер: предшествующее состояние доказанно ложно и правило задано before_false=reject


def _stage_reasons(rule: RuleSpec, series: SignalSeries, s_lo: np.ndarray, s_hi: np.ndarray, i0: int, i1: int,
                   before_lo: Optional[np.ndarray], before_hi: Optional[np.ndarray] = None) -> list[str]:
    """Проверка полноты этапа для правил «по бездействию»."""
    ina = rule.inaction
    if ina is None:
        return []
    times = series.times
    reasons: list[str] = []
    t_start = float(times[i0])
    if t_start - float(times[0]) <= ina.edge_s + 1e-9:
        reasons.append("начало ситуации не попало в ролик")
    elif before_lo is not None:
        lo_idx = int(np.searchsorted(times, t_start - ina.before_window_s - 1e-9, side="left"))
        window = before_lo[lo_idx:i0]
        if window.size == 0 or not bool((window >= 0.5).any()):
            hi_win = before_hi[lo_idx:i0] if before_hi is not None else None
            if (ina.before_false == "reject" and hi_win is not None and hi_win.size
                    and not bool((hi_win >= 0.5).any())):
                return [REJECT]  # доказанно не было (не «не видно») — это не та ситуация
            reasons.append(f"предшествующее состояние ({ina.before.source}) "
                           f"не наблюдалось за {ina.before_window_s:g} с")
    # кадр «не наблюдён», если условие не доказано, но возможно (неизвестные сигналы);
    # отсутствующая OR-альтернатива при доказанной другой ветви покрытие не портит
    unknown = (s_lo[i0:i1 + 1] < rule.on) & (s_hi[i0:i1 + 1] >= rule.on)
    cov = 1.0 - float(unknown.mean())
    if cov + 1e-9 < ina.min_coverage:
        reasons.append(f"сигналы правила известны лишь в {cov:.0%} кадров (окклюзия/пропуски)")
    return reasons


def _rule_events(rule: RuleSpec, series: SignalSeries, ctx: EvalContext, s_lo: np.ndarray, s_hi: np.ndarray,
                 tracks: Sequence[Any], zones: Sequence[Any], catalog: Mapping[str, Any], cfg: RulesConfig,
                 candidates: list[dict[str, Any]]) -> list[Any]:
    times = series.times
    events: list[Any] = []
    base_assessment, base_reasons = _assess(rule, catalog, cfg)
    aux: dict[Any, Any] = {}
    if not rule.predicates:  # старый формат: strength — единственный обязательный предикат, gate — условие
        aux[("legacy", "strength")] = rule.strength.bounds(ctx)
        if rule.gate is not None:
            aux[("legacy", "gate")] = rule.gate.bounds(ctx)
    if rule.applicability is not None:
        aux["app"] = rule.applicability.expr.bounds(ctx)
    for c in rule.predicates:
        aux[("pred", c.id)] = c.expr.bounds(ctx)
    for c in rule.exclusions:
        aux[("excl", c.id)] = c.expr.bounds(ctx)
    if rule.state_after_window_s > 0:  # момент ухода для окна «после» — независимо от политики t_peak
        trig = rule.trigger or rule.strength
        slo, shi = trig.bounds(ctx)
        slo = smooth(np.clip(slo, 0.0, 1.0), series.dt, rule.smooth_s, rule.smooth_method)
        shi = np.maximum(smooth(np.clip(shi, 0.0, 1.0), series.dt, rule.smooth_s, rule.smooth_method), slo)
        aux["sa_trigger"] = (slo, shi)
    if (cfg.t_peak_policy_override or rule.t_peak_policy) == "transition":
        trig = rule.trigger or rule.strength
        tlo, thi = trig.bounds(ctx)
        tlo = smooth(np.clip(tlo, 0.0, 1.0), series.dt, rule.smooth_s, rule.smooth_method)
        thi = np.maximum(smooth(np.clip(thi, 0.0, 1.0), series.dt, rule.smooth_s, rule.smooth_method), tlo)
        aux["trigger"] = (tlo, thi)
        aux["trigger_src"] = trig.source
    before_lo = before_hi = None
    if rule.inaction is not None and rule.inaction.before is not None:
        before_lo, before_hi = rule.inaction.before.bounds(ctx)
    edge_end = rule.inaction.edge_s if rule.inaction else 0.0

    # 1) доказанные сегменты
    segs = merge_segments(times, mask_to_segments(hysteresis_mask(s_lo, rule.on, rule.off)), rule.merge_gap_s)
    definite = np.zeros(series.n, dtype=bool)
    for i0, i1 in segs:
        definite[i0:i1 + 1] = True
        t_start, t_end = float(times[i0]), float(times[i1])
        duration = t_end - t_start + series.dt
        cand = {"rule_id": rule.id, "type_id": rule.type_id, "t_start": _rf(t_start, 2), "t_end": _rf(t_end, 2),
                "duration_s": _rf(duration, 2)}
        reasons: list[str] = []
        if duration + 1e-9 < rule.min_duration_s:
            at_end = float(times[-1]) - t_end <= max(edge_end, series.dt) + 1e-9
            if rule.inaction is not None and at_end:
                reasons.append(f"ролик заканчивается раньше, чем истекают {rule.min_duration_s:g} с")
            else:
                cand["rejected"] = f"короче min_duration_s={rule.min_duration_s}"
                candidates.append(cand)
                continue
        stage = _stage_reasons(rule, series, s_lo, s_hi, i0, i1, before_lo, before_hi)
        if REJECT in stage:
            cand["rejected"] = "предшествующее состояние доказанно отсутствовало (before_false=reject)"
            candidates.append(cand)
            continue
        reasons += [r for r in stage if r not in reasons]
        if reasons:
            assessment = INSUFFICIENT
        else:
            assessment, reasons = base_assessment, list(base_reasons)
        if (assessment == REVIEW and not cfg.emit_review_observations) or \
                (assessment == INSUFFICIENT and not cfg.emit_insufficient_context):
            cand["rejected"] = f"{assessment} отключён в global"
            candidates.append(cand)
            continue
        ev = _make_event(rule, series, s_lo, i0, i1, assessment, reasons, tracks, zones, catalog, cfg, aux=aux)
        cand.update(accepted=True, assessment=assessment, confidence=ev.confidence, t_peak=ev.t_peak)
        candidates.append(cand)
        events.append(ev)

    # 2) «могло быть, но не наблюдалось»: якорь доказан, остальное неизвестно
    if rule.anchor is not None and cfg.emit_insufficient_context:
        anchor_lo = rule.anchor(ctx)
        # кадры, где пропал хотя бы один сигнал правила, который в ролике вообще есть
        partial_unknown = np.zeros(series.n, dtype=bool)
        for nm in rule.names:
            if nm not in series.absent:
                partial_unknown |= ~series.known(nm)
        possible = (s_hi >= rule.on) & (anchor_lo >= 0.5) & ~definite & partial_unknown
        for i0, i1 in merge_segments(times, mask_to_segments(possible), rule.merge_gap_s):
            duration = float(times[i1] - times[i0]) + series.dt
            if duration + 1e-9 < rule.min_duration_s:
                continue
            # только сигналы, которые в ролике есть, но здесь пропали (окклюзия/потеря);
            # сигнал, отсутствующий во всём ролике (модуль не работал), — предупреждение, не событие
            unknown = [nm for nm in sorted(rule.names)
                       if nm not in series.absent and series.known(nm)[i0:i1 + 1].mean() < 0.5]
            if not unknown:
                continue
            reasons = ["не наблюдались сигналы: " + ", ".join(SIGNAL_RU.get(n, n) for n in unknown)]
            ev = _make_event(rule, series, s_hi, i0, i1, INSUFFICIENT, reasons, tracks, zones, catalog, cfg,
                             from_anchor=True, aux=aux)
            candidates.append({"rule_id": rule.id, "type_id": rule.type_id, "t_start": ev.t_start,
                               "t_end": ev.t_end, "accepted": True, "assessment": INSUFFICIENT,
                               "confidence": ev.confidence, "t_peak": ev.t_peak})
            events.append(ev)
    events.sort(key=lambda e: (-_PRIORITY[e.extra["assessment"]], -e.confidence))
    return events[: rule.max_events]


def _overlap(a: Any, b: Any, gap: float) -> bool:
    return a.t_start <= b.t_end + gap and b.t_start <= a.t_end + gap


def _cap_for(assessment: str, cfg: RulesConfig, default_cap: float) -> float:
    if assessment == REVIEW:
        return min(default_cap, cfg.observation_conf_cap)
    if assessment == INSUFFICIENT:
        return min(default_cap, cfg.insufficient_conf_cap)
    return default_cap


def _merge_same_type(events: list[Any], cfg: RulesConfig) -> list[Any]:
    by_type: dict[str, list[Any]] = {}
    for ev in events:
        by_type.setdefault(ev.type_id, []).append(ev)
    out: list[Any] = []
    for type_id, evs in by_type.items():
        evs.sort(key=lambda e: e.t_start)
        groups: list[list[Any]] = []
        for ev in evs:
            if groups and any(_overlap(ev, g, cfg.same_type_merge_gap_s) for g in groups[-1]):
                groups[-1].append(ev)
            else:
                groups.append([ev])
        for g in groups:
            if len(g) == 1:
                out.append(g[0])
                continue
            best_level = max(_PRIORITY[e.extra["assessment"]] for e in g)
            top = [e for e in g if _PRIORITY[e.extra["assessment"]] == best_level]
            primary = max(top, key=lambda e: e.confidence)
            first = min(top, key=lambda e: (e.t_start, e.t_peak))
            rule_ids: list[str] = []
            for e in g:
                for rid in e.extra.get("rule_ids", []):
                    if rid not in rule_ids:
                        rule_ids.append(rid)
            facts: list[str] = []
            for e in sorted(top, key=lambda e: e.t_start):
                for f in e.facts:
                    if f not in facts:
                        facts.append(f)
            bboxes = dict(first.bboxes)
            for k, v in primary.bboxes.items():
                bboxes.setdefault(k, v)
            n_corrob = len({e.extra.get("rule_id") for e in top}) - 1
            cap = max(e.extra.get("trace", {}).get("confidence_terms", {}).get("cap", 0.95) for e in top)
            cap = _cap_for(primary.extra["assessment"], cfg, cap)
            conf = min(cap, primary.confidence + cfg.corroboration_bonus * max(0, n_corrob))
            extra = dict(primary.extra)
            extra["rule_ids"] = rule_ids
            det_rank = {"violation_candidate": 3, "observation": 2, "insufficient_context": 1}
            extra["detection"] = max((e.extra.get("detection", "observation") for e in g),
                                     key=lambda d: det_rank.get(d, 0))
            extra["evidence_t"] = sorted({t for e in top for t in e.extra.get("evidence_t", [])} | {first.t_peak})[:3]
            extra["confidence_raw"] = max(float(e.extra.get("confidence_raw", e.confidence)) for e in top)
            if "state_after" not in extra:
                sa = next((e.extra["state_after"] for e in sorted(top, key=lambda e: e.t_start)
                           if "state_after" in e.extra), None)
                if sa is not None:
                    extra["state_after"] = sa
            negs = [n for e in g for n in e.extra.get("negative", [])]
            if negs:
                extra["negative"] = negs
            extra["merged"] = [{"rule_id": e.extra.get("rule_id"), "assessment": e.extra.get("assessment"),
                                "t_start": e.t_start, "t_end": e.t_end, "t_peak": e.t_peak,
                                "confidence": e.confidence} for e in g]
            extra["traces"] = [e.extra.get("trace") for e in g if e is not primary]
            out.append(Event(
                type_id=type_id,
                t_start=min(e.t_start for e in g),
                t_end=max(e.t_end for e in g),
                t_peak=first.t_peak,
                confidence=_rf(conf),
                source=SOURCE,
                facts=facts,
                bboxes=bboxes,
                track_ids=sorted({t for e in g for t in e.track_ids}),
                extra=extra,
            ))
    return out


def _apply_yield(events: list[Any], cfg: RulesConfig) -> None:
    rules = {r.id: r for r in cfg.rules}
    for ev in events:
        rule = rules.get(ev.extra.get("rule_id", ""))
        if rule is None or not rule.yield_to:
            continue
        winners = [o for o in events if o is not ev and o.type_id in rule.yield_to and _overlap(ev, o, 0.0)]
        if winners:
            ev.confidence = _rf(ev.confidence * rule.yield_factor)
            ev.extra["yielded_to"] = sorted({o.type_id for o in winners})


def _nms(events: list[Any], window_s: float) -> list[Any]:
    kept: list[Any] = []
    order = sorted(events, key=lambda e: (-_PRIORITY[e.extra["assessment"]], -e.confidence))
    for ev in order:
        clash = next((k for k in kept if k.type_id == ev.type_id and abs(k.t_peak - ev.t_peak) < window_s), None)
        if clash is None:
            kept.append(ev)
        else:
            clash.extra.setdefault("suppressed", []).append(
                {"rule_id": ev.extra.get("rule_id"), "assessment": ev.extra.get("assessment"),
                 "t_peak": ev.t_peak, "confidence": ev.confidence})
    return kept


def evaluate(states: Sequence[Any], tracks: Sequence[Any] = (), zones: Sequence[Any] = (),
             catalog: Optional[Mapping[str, Any]] = None, config: Optional[RulesConfig] = None) -> RulesReport:
    """Полный прогон правил с диагностикой (для replay, UI и интегратора)."""
    cfg = config or load_rules_config()
    cat, cat_src = resolve_catalog(catalog)
    warnings = list(cfg.warnings)
    needed: set[str] = set()
    for r in cfg.enabled_rules:
        needed |= r.names
    series = build_series(states, needed, defaults=cfg.signal_defaults, ffill_max_s=cfg.ffill_max_s,
                          merge_policy=cfg.merge_policy, known_min_conf=cfg.known_min_conf,
                          observed_one_sided=cfg.observed_one_sided)
    report = RulesReport(events=[], series=series, warnings=warnings, catalog_source=cat_src, config=cfg)
    if series.n == 0:
        warnings.append("нет состояний сцены — правила не применялись")
        return report
    if cfg.derive_from_tracks and tracks:
        absent = {n for n in needed if n in series.absent}
        if absent:
            try:
                derived = derive_from_tracks(series.times, tracks, zones, presence_tol_s=cfg.presence_tol_s)
            except Exception as exc:  # pragma: no cover - страховка
                derived = {}
                warnings.append(f"производные сигналы из треков: {exc}")
            for name, arr in derived.items():
                if name in absent:
                    series.signals[name] = arr
                    series.derived[name] = "tracks"
                    series.absent.discard(name)
    ctx = EvalContext(times=series.times, signals=series.signals, ranges=cfg.signal_ranges)
    events: list[Any] = []
    for rule in cfg.enabled_rules:
        try:
            lo, hi = rule.strength.bounds(ctx)
            lo, hi = np.clip(lo, 0.0, 1.0), np.clip(hi, 0.0, 1.0)
            if rule.gate is not None:
                glo, ghi = rule.gate.bounds(ctx)
                definite_gate = (glo > 0.5) if rule.gate_mode == "strict" else (ghi > 0.5)
                lo = np.where(definite_gate, lo, 0.0)
                hi = np.where(ghi > 0.5, hi, 0.0)
            if rule.applicability is not None:          # применимость: должна быть доказана
                alo, ahi = rule.applicability.expr.bounds(ctx)
                lo = np.where(alo > 0.5, lo, 0.0)
                hi = np.where(ahi > 0.5, hi, 0.0)
            for exc in rule.exclusions:                 # исключение блокирует, только если доказано
                elo, _ = exc.expr.bounds(ctx)
                lo = np.where(elo > 0.5, 0.0, lo)
                hi = np.where(elo > 0.5, 0.0, hi)
            s_lo = np.clip(smooth(lo, series.dt, rule.smooth_s, rule.smooth_method), 0.0, 1.0)
            s_hi = np.maximum(np.clip(smooth(hi, series.dt, rule.smooth_s, rule.smooth_method), 0.0, 1.0), s_lo)
            report.strengths[rule.id] = s_lo
            report.possible[rule.id] = s_hi
            absent_inputs = sorted(n for n in rule.names if n in series.absent)
            if absent_inputs:
                warnings.append(f"{rule.id}: нет сигналов {', '.join(absent_inputs)} — они считаются неизвестными")
            events.extend(_rule_events(rule, series, ctx, s_lo, s_hi, tracks, zones, cat, cfg, report.candidates))
        except Exception as exc:  # одно сломанное правило не валит остальные
            warnings.append(f"{rule.id}: ошибка вычисления: {exc}")
    events = _merge_same_type(events, cfg)
    _apply_yield(events, cfg)
    events = _nms(events, cfg.nms_window_s)
    events = [e for e in events if e.confidence >= cfg.min_confidence]
    events.sort(key=lambda e: (-_PRIORITY[e.extra["assessment"]], -e.confidence))
    events = events[: cfg.max_events]
    events.sort(key=lambda e: (e.t_peak, e.type_id))
    report.events = events
    return report


def run_rules(states: Sequence[Any], tracks: Sequence[Any], zones: Sequence[Any],
              catalog: Optional[Mapping[str, Any]], config: Optional[RulesConfig] = None) -> list[Any]:
    """Публичная функция модуля (сигнатура контракта v1 + необязательный ``config``).

    Возвращает события всех трёх оценок; к вердикту «нарушение» допускаются только
    ``extra["assessment"] == "candidate_violation"`` (см. :func:`split_by_assessment`).
    """
    return evaluate(states, tracks, zones, catalog, config).events
