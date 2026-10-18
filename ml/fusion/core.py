"""ml.fusion.core — слияние событий rules / temporal / vlm в итоговый AnalysisResult.

Чистая часть (``fuse_scores``) не трогает кадры и используется тюнером порогов;
``fuse`` (сигнатура заморожена контрактом) добавляет описания, нормы и кадры-доказательства.
"""
from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from ._types import AnalysisResult, BBox, Violation
from .catalog import known_type, resolve_catalog, rule_refs, type_info
from .completeness import coverage_info
from .policies import apply_policies
from .config import load_config, load_thresholds, threshold_for
from .describe import build_description, clean_facts, format_time
from .serialize import event_to_dict, to_jsonable
from .timing import clip_times, snap_time

log = logging.getLogger(__name__)

FUSION_VERSION = "a12-fusion-1.0"
SOURCE_PRIORITY = ("rules", "vlm", "temporal", "fusion", "other")
_SOURCE_ALIASES = {
    "rules": "rules", "rule": "rules", "rules_engine": "rules",
    "temporal": "temporal", "temporal_head": "temporal", "siglip": "temporal",
    "vlm": "vlm", "vlm_verify": "vlm", "vlm_discover": "vlm", "verify": "vlm", "discover": "vlm",
    "fusion": "fusion",
}
_CONFIRM = {"confirm", "confirmed", "yes", "true", "1", "violation", "verified", "да", "подтверждено",
            "подтверждаю", "accept", "accepted", "positive"}
_REJECT = {"reject", "rejected", "no", "false", "0", "no_violation", "not_verified", "нет", "отклонено",
           "refuted", "negative", "deny", "denied"}


# --------------------------------------------------------------------------- модели данных


@dataclass
class NormEvent:
    """Нормализованное входное событие."""

    type_id: str
    t_start: float
    t_end: float
    t_peak: float
    confidence: float
    source: str                  # rules | temporal | vlm | fusion | other
    raw_source: str
    facts: list[str]
    bboxes: dict[str, BBox]
    track_ids: list[int]
    extra: dict[str, Any]
    vlm: Optional[tuple[str, float]] = None   # ("confirm"|"reject"|"unsure", score)
    positive: bool = True        # False — только мнение VLM (отказ), без положительного вклада
    orig_type_id: Optional[str] = None
    assessment: str = "candidate"   # candidate | norm_review | observation | insufficient (оценка A09)
    conf_in: float = 0.0            # уверенность как пришла (до восстановления срезанной A09)


@dataclass
class Candidate:
    """Кандидат-нарушение после слияния."""

    type_id: str
    t_start: float
    t_end: float
    t_peak: float
    confidence: float
    origin: str = "events"       # events | risk
    conf_by_source: dict[str, float] = field(default_factory=dict)   # сырые max по источнику
    weighted: dict[str, float] = field(default_factory=dict)         # w_s * c_s
    base: float = 0.0
    bonus: float = 0.0
    vlm_state: Optional[str] = None
    vlm_factor: float = 1.0
    risk_support: float = 0.0
    n_events: int = 0
    facts: list[str] = field(default_factory=list)
    obs_facts: list[str] = field(default_factory=list)   # наблюдаемые факты от rules/VLM (для гейта)
    bboxes: dict[str, BBox] = field(default_factory=dict)
    track_ids: list[int] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    peak_source: str = ""
    threshold: float = 0.5
    reported: bool = False
    suppressed_by: Optional[str] = None
    gate: list[str] = field(default_factory=list)        # почему НЕ нарушение (пусто — гейт пройден)
    assessment: str = "candidate"                        # сводная оценка upstream (см. NormEvent)

    @property
    def key(self) -> str:
        return f"{self.type_id}@{self.t_peak:.1f}"

    def to_raw(self) -> dict[str, Any]:
        return to_jsonable(
            {
                "type_id": self.type_id,
                "t_start": self.t_start,
                "t_end": self.t_end,
                "t_peak": self.t_peak,
                "confidence": self.confidence,
                "origin": self.origin,
                "conf_by_source": self.conf_by_source,
                "weighted": self.weighted,
                "base": self.base,
                "bonus": self.bonus,
                "vlm_state": self.vlm_state,
                "vlm_factor": self.vlm_factor,
                "risk_support": self.risk_support,
                "n_events": self.n_events,
                "peak_source": self.peak_source,
                "threshold": self.threshold,
                "reported": self.reported,
                "suppressed_by": self.suppressed_by,
                "gate": self.gate,
                "assessment": self.assessment,
                "n_obs_facts": len(self.obs_facts),
            },
            4,
        )


@dataclass
class FusionOutcome:
    candidates: list[Candidate]
    reported: list[Candidate]
    verdict: str
    timeline: list[tuple[float, float]]
    duration: float
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- утилиты


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(out) or math.isinf(out) else out


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _get(ev: Any, name: str, default: Any = None) -> Any:
    if isinstance(ev, dict):
        return ev.get(name, default)
    return getattr(ev, name, default)


def canonical_source(source: str) -> str:
    s = str(source or "").strip().lower()
    if s in _SOURCE_ALIASES:
        return _SOURCE_ALIASES[s]
    for prefix, canon in (("rule", "rules"), ("temporal", "temporal"), ("vlm", "vlm")):
        if s.startswith(prefix):
            return canon
    return "other"


def _verdict_word(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return "confirm" if value else "reject"
    s = str(value).strip().lower()
    if s in _CONFIRM:
        return "confirm"
    if s in _REJECT:
        return "reject"
    if s in {"unsure", "uncertain", "unknown", "maybe", "не уверен", "неясно", "abstain"}:
        return "unsure"
    return None


def vlm_opinion(source: str, confidence: float, extra: dict[str, Any]) -> Optional[tuple[str, float]]:
    """Мнение VLM о событии в разных конвенциях A11. Возвращает (verdict, score) или None.

    Поддерживаются: extra["vlm"] = {"verdict"|"label"|"answer", "score"|"confidence"|"prob"} или строка;
    extra["vlm_verdict"], extra["vlm_score"], extra["vlm_confirmed"] (bool), extra["verified"] (bool),
    а для событий с source=vlm — extra["verdict"].
    """
    verdict: Optional[str] = None
    score: Optional[float] = None
    vlm_block = extra.get("vlm")
    if isinstance(vlm_block, dict):
        for k in ("verdict", "label", "answer", "decision", "status"):
            if k in vlm_block:
                verdict = _verdict_word(vlm_block[k])
                if verdict:
                    break
        for k in ("score", "confidence", "prob", "p"):
            if k in vlm_block:
                score = _f(vlm_block[k], -1.0)
                if score >= 0:
                    break
                score = None
    elif isinstance(vlm_block, (str, bool)):
        verdict = _verdict_word(vlm_block)
    if verdict is None:
        for k in ("vlm_verdict", "vlm_confirmed", "vlm_label"):
            if k in extra:
                verdict = _verdict_word(extra[k])
                if verdict:
                    break
    if verdict is None and "verified" in extra and isinstance(extra["verified"], bool):
        verdict = "confirm" if extra["verified"] else "reject"
    if verdict is None and source == "vlm" and "verdict" in extra:
        verdict = _verdict_word(extra["verdict"])
    if score is None:
        for k in ("vlm_score", "vlm_conf", "vlm_confidence"):
            if k in extra:
                score = _f(extra[k], -1.0)
                if score >= 0:
                    break
                score = None
    if verdict is None:
        return None
    if score is None:
        score = confidence if (verdict == "confirm" and source == "vlm") else 1.0
    return verdict, _clamp(score, 0.0, 1.0)


def _norm_bbox(box: Any) -> Optional[BBox]:
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if any(math.isnan(v) or math.isinf(v) for v in (x1, y1, x2, y2)):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    # bbox нормирован 0..1; если пришли пиксели — оставляем как есть, отрисовщик разберётся
    return (x1, y1, x2, y2)


_ASSESSMENT_RANK = ("candidate", "norm_review", "observation", "insufficient")


def _a09_uncapped(conf: float, extra: dict[str, Any]) -> float:
    """Уверенность правила до потолка review_observation: extra.confidence_raw или trace.confidence_terms A09."""
    for key in ("confidence_raw", "confidence_uncapped", "raw_confidence"):
        if key in extra:
            v = _f(extra[key], -1.0)
            if v >= 0:
                return _clamp(max(conf, v), 0.0, 1.0)
    trace = extra.get("trace")
    terms = trace.get("confidence_terms") if isinstance(trace, dict) else None
    if not isinstance(terms, dict):
        return conf
    if "expr" in terms:
        raw = _f(terms.get("expr"), float("nan"))
    elif all(k in terms for k in ("base", "strength", "duration")):
        raw = _f(terms["base"], 0.0) + _f(terms["strength"], 0.0) + _f(terms["duration"], 0.0)
    else:
        return conf
    if math.isnan(raw):
        return conf
    value = min(_f(terms.get("cap", 1.0), 1.0), raw * _f(terms.get("factor", 1.0), 1.0))
    value *= _f(terms.get("derived_factor", 1.0), 1.0)
    return _clamp(max(conf, value), 0.0, 1.0)


def upstream_assessment(conf: float, extra: dict[str, Any], cfg: dict[str, Any]) -> tuple[str, float]:
    """Оценка события правил A09 → (candidate|norm_review|observation|insufficient, уверенность для слияния)."""
    gcfg = cfg.get("report_gate") or {}
    if not gcfg.get("upstream_assessment", True):
        return "candidate", conf
    a = str(extra.get("assessment") or "").strip().lower()
    if not a or a in {"candidate_violation", "candidate", "violation"}:
        return "candidate", conf
    if a in {"insufficient_context", "insufficient"}:
        return "insufficient", conf
    if a in {"review_observation", "review", "observation"}:
        reasons = " ".join(str(r) for r in extra.get("assessment_reasons") or []).lower()
        if "наблюдени" in reasons or "observation" in reasons:
            return "observation", conf
        if "норм" in reasons or "norm" in reasons or str(extra.get("norm_status", "")).lower() in {"unverified",
                                                                                                 "absent"}:
            return "norm_review", (_a09_uncapped(conf, extra) if gcfg.get("uncap_norm_review", True) else conf)
        return "observation", conf
    return "candidate", conf


def normalize_events(
    events: Optional[Iterable[Any]],
    duration: float,
    types: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
) -> tuple[list[NormEvent], list[str]]:
    """Проверка и приведение входных событий: времена в границах видео, conf 0..1, известный type_id."""
    out: list[NormEvent] = []
    warnings: list[str] = []
    fallback_type = str((cfg.get("risk") or {}).get("fallback_type", "V99"))
    for i, ev in enumerate(events or []):
        try:
            type_id = str(_get(ev, "type_id", "") or "").strip().upper()
            orig_type: Optional[str] = None
            if not type_id or not known_type(types, type_id):
                orig_type = type_id or None
                type_id = fallback_type
            t_peak = _f(_get(ev, "t_peak"), float("nan"))
            t_start = _f(_get(ev, "t_start"), float("nan"))
            t_end = _f(_get(ev, "t_end"), float("nan"))
            if math.isnan(t_peak):
                if math.isnan(t_start) and math.isnan(t_end):
                    warnings.append(f"event #{i} skipped: no timestamps")
                    continue
                vals = [v for v in (t_start, t_end) if not math.isnan(v)]
                t_peak = sum(vals) / len(vals)
            t_start = t_peak if math.isnan(t_start) else t_start
            t_end = t_peak if math.isnan(t_end) else t_end
            if t_end < t_start:
                t_start, t_end = t_end, t_start
            hi = duration if duration > 0 else max(t_end, t_peak, 0.0)
            t_start = _clamp(t_start, 0.0, hi)
            t_end = _clamp(t_end, 0.0, hi)
            t_peak = _clamp(t_peak, t_start, t_end) if t_end >= t_start else _clamp(t_peak, 0.0, hi)
            conf = _clamp(_f(_get(ev, "confidence"), 0.0), 0.0, 1.0)
            raw_source = str(_get(ev, "source", "") or "")
            source = canonical_source(raw_source)
            extra = dict(_get(ev, "extra", {}) or {})
            bboxes: dict[str, BBox] = {}
            for key, box in dict(_get(ev, "bboxes", {}) or {}).items():
                nb = _norm_bbox(box)
                if nb is not None:
                    bboxes[str(key)] = nb
            track_ids = []
            for tid in _get(ev, "track_ids", []) or []:
                try:
                    track_ids.append(int(tid))
                except (TypeError, ValueError):
                    continue
            conf_in = conf
            assessment, conf = upstream_assessment(conf, extra, cfg)
            opinion = vlm_opinion(source, conf, extra)
            positive = not (source == "vlm" and opinion is not None and opinion[0] != "confirm")
            if positive and source == "vlm" and conf <= 0.0:
                positive = False
            out.append(
                NormEvent(
                    type_id=type_id,
                    t_start=t_start,
                    t_end=t_end,
                    t_peak=t_peak,
                    confidence=conf,
                    source=source,
                    raw_source=raw_source,
                    facts=[str(f) for f in (_get(ev, "facts", []) or []) if str(f).strip()],
                    bboxes=bboxes,
                    track_ids=track_ids,
                    extra=extra,
                    vlm=opinion,
                    positive=positive,
                    orig_type_id=orig_type,
                    assessment=assessment,
                    conf_in=conf_in,
                )
            )
        except Exception as exc:  # noqa: BLE001 — одно битое событие не должно ронять слияние
            warnings.append(f"event #{i} skipped: {exc!r}")
    return out, warnings


def group_events(events: Sequence[NormEvent], gap: float) -> list[list[NormEvent]]:
    """Группировка по типу и пересечению во времени (зазор ≤ gap)."""
    by_type: dict[str, list[NormEvent]] = {}
    for ev in events:
        by_type.setdefault(ev.type_id, []).append(ev)
    groups: list[list[NormEvent]] = []
    for type_id in sorted(by_type):
        evs = sorted(by_type[type_id], key=lambda e: (e.t_start, e.t_end))
        current: list[NormEvent] = []
        end = -math.inf
        for ev in evs:
            if current and ev.t_start <= end + gap:
                current.append(ev)
                end = max(end, ev.t_end)
            else:
                if current:
                    groups.append(current)
                current = [ev]
                end = ev.t_end
        if current:
            groups.append(current)
    return groups


class RiskCurve:
    """Риск-кривая temporal: (t, risk) → max в окне / значение в точке."""

    def __init__(self, risk: Optional[Iterable[Any]]):
        pts: list[tuple[float, float]] = []
        for item in risk or []:
            try:
                t, r = item[0], item[1]
            except (TypeError, IndexError, KeyError):
                continue
            t, r = _f(t, float("nan")), _f(r, float("nan"))
            if math.isnan(t) or math.isnan(r):
                continue
            pts.append((t, _clamp(r, 0.0, 1.0)))
        pts.sort()
        self.points = pts
        diffs = [b[0] - a[0] for a, b in zip(pts, pts[1:]) if b[0] > a[0]]
        diffs.sort()
        self.dt = diffs[len(diffs) // 2] if diffs else 0.2

    def __bool__(self) -> bool:
        return bool(self.points)

    @property
    def max(self) -> float:
        return max((r for _, r in self.points), default=0.0)

    def window_max(self, a: float, b: float) -> float:
        return max((r for t, r in self.points if a <= t <= b), default=0.0)

    def at(self, t: float, half_window: float) -> float:
        """max в [t-hw, t+hw]; если точек нет — линейная интерполяция соседей (не дальше 2 с)."""
        if not self.points:
            return 0.0
        vals = [r for tt, r in self.points if t - half_window <= tt <= t + half_window]
        if vals:
            return max(vals)
        left = next((p for p in reversed(self.points) if p[0] < t), None)
        right = next((p for p in self.points if p[0] > t), None)
        if left and right and right[0] - left[0] <= 2.0:
            w = (t - left[0]) / (right[0] - left[0])
            return left[1] * (1 - w) + right[1] * w
        return 0.0

    def segments(self, high: float, min_duration: float, gap: float) -> list[tuple[float, float, float, float]]:
        """Участки risk ≥ high: (t_start, t_end, t_peak, peak)."""
        segs: list[list[float]] = []
        for t, r in self.points:
            if r < high:
                continue
            if segs and t - segs[-1][1] <= max(gap, 1.5 * self.dt):
                seg = segs[-1]
                seg[1] = t
                if r > seg[3]:
                    seg[2], seg[3] = t, r
            else:
                segs.append([t, t, t, r])
        out = []
        for a, b, tp, pk in segs:
            if (b - a) + self.dt >= min_duration:
                out.append((a, b, tp, pk))
        return out


def _source_weight(cfg: dict[str, Any], source: str) -> float:
    weights = cfg.get("sources") or {}
    return _f(weights.get(source, weights.get("other", 0.6)), 0.6)


def temporal_mode(cfg: dict[str, Any]) -> str:
    """off (по умолчанию) | bonus | score — влияние temporal/риск-кривой на скор кластера."""
    raw = cfg.get("temporal_mode", "off")
    mode = "off" if raw is False or raw is None else str(raw).strip().lower()   # YAML: голое off → False
    return mode if mode in {"off", "bonus", "score"} else "off"


def score_group(group: Sequence[NormEvent], risk: RiskCurve, cfg: dict[str, Any]) -> Candidate:
    """Объединение уверенностей источников в одном кластере."""
    positives = [e for e in group if e.positive]
    pool = positives or list(group)
    type_id = group[0].type_id
    tmode = temporal_mode(cfg)
    # temporal_mode=off|bonus: при наличии фактических источников границы и пик берём по ним
    factual = [e for e in pool if e.source != "temporal"] if tmode != "score" else []
    span = factual or pool
    t_start = min(e.t_start for e in span)
    t_end = max(e.t_end for e in span)

    conf_by_source: dict[str, float] = {}
    for e in positives:
        conf_by_source[e.source] = max(conf_by_source.get(e.source, 0.0), e.confidence)

    rcfg = cfg.get("risk") or {}
    pad = _f(rcfg.get("support_pad_s", 1.0), 1.0)
    risk_support = risk.window_max(t_start - pad, t_end + pad) if risk else 0.0

    weighted: dict[str, float] = {}
    only_vlm = bool(conf_by_source) and set(conf_by_source) == {"vlm"}
    for src, c in conf_by_source.items():
        w = _source_weight(cfg, "vlm_discover" if (src == "vlm" and only_vlm) else src)
        weighted[src] = _clamp(w * c, 0.0, 1.0)
    if positives and "temporal" not in weighted and risk_support >= _f(rcfg.get("support_min", 0.5), 0.5):
        weighted["risk_curve"] = _clamp(_f(rcfg.get("support_weight", 0.6), 0.6) * risk_support, 0.0, 1.0)

    # Какие вклады задают базовый скор. temporal_mode=score — как в фазе 1; bonus — temporal/кривая только
    # в бонусе согласия; off — temporal/кривая не влияют на скор кластера с фактическими источниками.
    temporal_keys = {"temporal", "risk_curve"}
    has_factual = any(k not in temporal_keys for k in weighted)
    base_parts = {k: v for k, v in weighted.items()
                  if tmode == "score" or k not in temporal_keys or not has_factual}
    agree_parts = {k: v for k, v in weighted.items()
                   if tmode != "off" or k not in temporal_keys or not has_factual}

    if not base_parts:
        base = 0.0
    elif str(cfg.get("combine", "noisy_or")).lower() == "max":
        base = max(base_parts.values())
    else:
        prod = 1.0
        for v in base_parts.values():
            prod *= 1.0 - v
        base = 1.0 - prod

    acfg = cfg.get("agreement") or {}
    n_agree = sum(1 for v in agree_parts.values() if v >= _f(acfg.get("min_conf", 0.3), 0.3))
    bonus = 0.0
    if n_agree >= 2:
        bonus = min(_f(acfg.get("max_bonus", 0.15), 0.15),
                    _f(acfg.get("bonus_per_source", 0.07), 0.07) * (n_agree - 1))

    # VLM-понижение / подтверждение: берём самое уверенное мнение VLM в кластере.
    vcfg = cfg.get("vlm") or {}
    opinions = [e.vlm for e in group if e.vlm is not None]
    vlm_state: Optional[str] = None
    vlm_factor = 1.0
    if opinions:
        verdict, score = max(opinions, key=lambda o: (o[1], o[0] == "confirm"))
        vlm_state = verdict
        if verdict == "reject" and score >= _f(vcfg.get("reject_min_score", 0.5), 0.5):
            vlm_factor = _f(vcfg.get("reject_factor", 0.6), 0.6)
        elif verdict == "unsure":
            vlm_factor = _f(vcfg.get("unsure_factor", 1.0), 1.0)
        elif verdict == "reject":
            vlm_state = "reject_weak"

    confidence = _clamp((base + bonus) * vlm_factor, 0.0, 0.99) if positives else 0.0

    # Пиковое событие: max(w*c), при равенстве — приоритет источника.
    def peak_key(e: NormEvent) -> tuple[float, int]:
        w = weighted.get(e.source, 0.0) / max(conf_by_source.get(e.source, 1e-9), 1e-9)
        prio = SOURCE_PRIORITY.index(e.source) if e.source in SOURCE_PRIORITY else len(SOURCE_PRIORITY)
        return (round(w * e.confidence, 6), -prio)

    peak_ev = max(span, key=peak_key)

    # Факты: rules → vlm → temporal → прочие; боксы — с пикового события, затем близкие по времени.
    ordered = sorted(pool, key=lambda e: (SOURCE_PRIORITY.index(e.source) if e.source in SOURCE_PRIORITY
                                          else len(SOURCE_PRIORITY), -e.confidence))
    facts: list[str] = []
    obs_facts: list[str] = []
    fact_sources = set((cfg.get("report_gate") or {}).get("fact_sources") or ["rules", "vlm"])
    for e in ordered:
        facts.extend(e.facts)
        if e.positive and e.source in fact_sources:
            obs_facts.extend(e.facts)
    bboxes: dict[str, BBox] = dict(peak_ev.bboxes)
    box_dt = _f((cfg.get("evidence") or {}).get("box_max_dt_s", 1.5), 1.5)
    for e in ordered:
        if e is peak_ev or abs(e.t_peak - peak_ev.t_peak) > box_dt:
            continue
        for k, b in e.bboxes.items():
            bboxes.setdefault(k, b)
    track_ids: list[int] = []
    for e in [peak_ev] + ordered:
        for tid in e.track_ids:
            if tid not in track_ids:
                track_ids.append(tid)
    extra: dict[str, Any] = {}
    for e in [peak_ev] + ordered:
        for k in ("keypoints", "poses", "head_boxes", "person_bboxes", "persons", "roles", "type_probs",
                  "type_scores", "state_after"):
            if k in e.extra and k not in extra:
                extra[k] = e.extra[k]
    ev_t = sorted({round(_f(t), 3) for e in span for t in (e.extra.get("evidence_t") or [])
                   if not math.isnan(_f(t, float("nan")))})
    if ev_t:
        extra["evidence_t"] = ev_t[:6]
    orig_types = sorted({e.orig_type_id for e in group if e.orig_type_id})
    if orig_types:
        extra["orig_type_ids"] = orig_types

    return Candidate(
        type_id=type_id,
        t_start=t_start,
        t_end=t_end,
        t_peak=_clamp(peak_ev.t_peak, t_start, t_end),
        confidence=confidence,
        origin="events",
        conf_by_source=conf_by_source,
        weighted=weighted,
        base=base,
        bonus=bonus,
        vlm_state=vlm_state,
        vlm_factor=vlm_factor,
        risk_support=risk_support,
        n_events=len(group),
        facts=facts,
        obs_facts=obs_facts,
        bboxes=bboxes,
        track_ids=track_ids,
        extra=extra,
        peak_source=peak_ev.source,
        assessment=_cluster_assessment(positives),
    )


def _cluster_assessment(positives: Sequence[NormEvent]) -> str:
    """Лучшая оценка среди фактических источников (rules/VLM); подтверждение VLM считается candidate.

    temporal/прочие события без оценки upstream не «повышают» кластер: иначе правило-наблюдение A09
    рядом с temporal-событием обходило бы гейт observation_only.
    """
    voters = [e for e in positives if e.source in ("rules", "vlm") or "assessment" in e.extra]
    if not voters:
        return "candidate" if positives else "insufficient"
    best = len(_ASSESSMENT_RANK) - 1
    for e in voters:
        a = "candidate" if (e.vlm is not None and e.vlm[0] == "confirm") else e.assessment
        if a in _ASSESSMENT_RANK:
            best = min(best, _ASSESSMENT_RANK.index(a))
    return _ASSESSMENT_RANK[best]


def _best_type_near(events: Sequence[NormEvent], a: float, b: float, fallback: str,
                    types: dict[str, dict[str, Any]]) -> str:
    """«Лучший тип» для участка высокого риска: type_probs/type_scores событий рядом → иначе fallback."""
    scores: dict[str, float] = {}
    for e in events:
        if e.t_end < a - 5.0 or e.t_start > b + 5.0:
            continue
        for key in ("type_probs", "type_scores"):
            probs = e.extra.get(key)
            if isinstance(probs, dict):
                for tid, p in probs.items():
                    tid = str(tid).upper()
                    if known_type(types, tid):
                        scores[tid] = max(scores.get(tid, 0.0), _f(p))
    if scores:
        best = max(scores.items(), key=lambda kv: kv[1])
        if best[1] > 0:
            return best[0]
    return fallback


def risk_only_candidates(
    risk: RiskCurve,
    existing: Sequence[Candidate],
    events: Sequence[NormEvent],
    types: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
) -> list[Candidate]:
    """Высокий риск temporal без событий других источников → ПОДСКАЗКА (V99/лучший тип) для просмотра человеком.

    Такие кандидаты никогда не становятся нарушением: «модель заметила аномалию» — не наблюдаемый факт.
    """
    if not risk:
        return []
    rcfg = cfg.get("risk") or {}
    gap = _f((cfg.get("grouping") or {}).get("merge_gap_s", 3.0), 3.0)
    segs = risk.segments(_f(rcfg.get("high", 0.65), 0.65), _f(rcfg.get("min_duration_s", 0.4), 0.4), gap)
    out: list[Candidate] = []
    for a, b, tp, peak in segs:
        covered = any(c.t_start - gap <= b and a <= c.t_end + gap for c in existing if c.confidence > 0)
        if covered:
            continue
        type_id = _best_type_near(events, a, b, str(rcfg.get("fallback_type", "V99")), types)
        conf = _clamp(peak * _f(rcfg.get("only_factor", 0.7), 0.7), 0.0, 0.99)
        fact = (f"на интервале {format_time(a)}–{format_time(b)} временная модель отмечает "
                f"нетипичную активность в кадре (оценка риска {peak:.2f})")
        out.append(
            Candidate(
                type_id=type_id,
                t_start=a,
                t_end=b,
                t_peak=tp,
                confidence=conf,
                origin="risk",
                conf_by_source={"risk_curve": peak},
                weighted={"risk_curve": conf},
                base=conf,
                risk_support=peak,
                n_events=0,
                facts=[fact],
                peak_source="risk_curve",
                gate=["risk_only"],
            )
        )
    out.sort(key=lambda c: -c.confidence)
    return out[: int(_f(rcfg.get("max_events", 3), 3))]


def cross_type_nms(cands: list[Candidate], cfg: dict[str, Any]) -> None:
    """Слабое нарушение другого типа, совпадающее по времени с сильным, помечается suppressed_by."""
    gcfg = cfg.get("grouping") or {}
    if not gcfg.get("cross_type_nms", True):
        return
    ratio = _f(gcfg.get("cross_type_ratio", 0.5), 0.5)
    min_ov = _f(gcfg.get("cross_type_min_overlap", 0.5), 0.5)
    kept: list[Candidate] = []
    for c in sorted(cands, key=lambda x: -x.confidence):
        if c.confidence <= 0:
            continue
        for k in kept:
            if k.type_id == c.type_id:
                continue
            a1, b1 = c.t_start - 0.25, c.t_end + 0.25
            a2, b2 = k.t_start - 0.25, k.t_end + 0.25
            inter = min(b1, b2) - max(a1, a2)
            shorter = min(b1 - a1, b2 - a2)
            if inter > 0 and shorter > 0 and inter / shorter >= min_ov and c.confidence < ratio * k.confidence:
                c.suppressed_by = k.key
                break
        if c.suppressed_by is None:
            kept.append(c)


def catalog_has_norms(types: dict[str, dict[str, Any]]) -> bool:
    return any(entry.get("rules") for entry in types.values() if isinstance(entry, dict))


def gate_reasons(c: Candidate, types: dict[str, dict[str, Any]], cfg: dict[str, Any], catalog_ok: bool
                 ) -> list[str]:
    """Нарушение = конкретный наблюдаемый факт + применимая норма (+ не только temporal)."""
    gcfg = cfg.get("report_gate") or {}
    reasons: list[str] = []
    if c.origin == "risk":
        return ["risk_only"]
    if gcfg.get("require_fact", True) and not clean_facts(c.obs_facts, cfg, max_facts=1):
        reasons.append("no_observable_fact")
    if not gcfg.get("temporal_only_types", False) and set(c.conf_by_source) <= {"temporal"}:
        reasons.append("temporal_only")
    req = gcfg.get("require_norm_types")
    if req is None:  # совместимость со старым ключом require_norm: true|false
        req = ["*"] if gcfg.get("require_norm", False) else []
    if "*" in req or c.type_id in req:
        if catalog_ok:
            if not rule_refs(types, c.type_id):
                reasons.append("no_norm")
        elif str(gcfg.get("norm_missing_if_no_catalog", "report")).lower() == "hint":
            reasons.append("norms_catalog_unavailable")
    reasons.extend(str(r) for r in c.extra.get("policy_gate") or [] if str(r) not in reasons)
    if gcfg.get("upstream_assessment", True):
        if c.assessment == "observation" and str(gcfg.get("observation_policy", "hint")) == "hint":
            reasons.append("observation_only")
        elif c.assessment == "insufficient" and str(gcfg.get("insufficient_policy", "hint")) == "hint":
            reasons.append("insufficient_context")
    return reasons


def timecode(t: float, duration: float, ndigits: int = 1) -> float:
    """Таймкод в границах видео, округление до 0.1 (не выходит за длительность)."""
    t = _f(t)
    if duration > 0:
        t = _clamp(t, 0.0, duration)
        q = round(t, ndigits)
        if q > duration:
            q = math.floor(duration * 10**ndigits) / 10**ndigits
        return float(q)
    return float(round(max(0.0, t), ndigits))


def build_timeline(duration: float, risk: RiskCurve, cands: Sequence[Candidate], cfg: dict[str, Any]
                   ) -> list[tuple[float, float]]:
    """Риск с шагом 0.5 с = max(риск-кривая temporal, уверенность слитых кандидатов на интервале)."""
    ocfg = cfg.get("output") or {}
    step = max(0.1, _f(ocfg.get("timeline_step_s", 0.5), 0.5))
    floor = _f(ocfg.get("candidate_floor", 0.15), 0.15)
    end = duration if duration > 0 else max([c.t_end for c in cands] + [p[0] for p in risk.points] + [0.0])
    n = int(math.floor(end / step + 1e-6)) + 1
    active = [c for c in cands if c.suppressed_by is None and c.confidence >= floor]
    out: list[tuple[float, float]] = []
    for k in range(n):
        t = round(k * step, 3)
        v = risk.at(t, step / 2)
        for c in active:
            if c.t_start - step / 2 <= t <= c.t_end + step / 2:
                v = max(v, c.confidence)
        out.append((t, round(_clamp(v, 0.0, 1.0), 3)))
    return out


def fuse_scores(
    events: Optional[Iterable[Any]],
    risk: Optional[Iterable[Any]],
    duration: float,
    types: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
    thresholds: dict[str, Any],
    catalog_ok: Optional[bool] = None,
) -> FusionOutcome:
    """Чистое слияние без кадров: кандидаты, вердикт, timeline. Используется fuse() и тюнером.

    ``confidence`` кандидата — эвристический скор 0..1 (источники коррелированы), не вероятность.
    """
    if catalog_ok is None:
        catalog_ok = catalog_has_norms(types)
    norm, warnings = normalize_events(events, duration, types, cfg)
    curve = RiskCurve(risk)
    gap = _f((cfg.get("grouping") or {}).get("merge_gap_s", 3.0), 3.0)
    cands = []
    for g in group_events(norm, gap):
        cand = score_group(g, curve, cfg)
        apply_policies(cand, g, cfg)
        cands.append(cand)
    cands = [c for c in cands if c.confidence > 0 or c.n_events > 0]
    cands += risk_only_candidates(curve, cands, norm, types, cfg)
    cross_type_nms(cands, cfg)

    ocfg = cfg.get("output") or {}
    nd = int(_f(ocfg.get("timecode_round", 1), 1))
    tc_policy = str((cfg.get("timecode") or {}).get("policy", "peak")).lower()
    for c in cands:
        c.extra.setdefault("t_peak_source", round(c.t_peak, 3))
        if tc_policy == "onset" and c.origin == "events":
            c.t_peak = c.t_start   # момент начала (переход: отошла, вошёл, поднесла)
        c.t_peak = timecode(c.t_peak, duration, nd)
        c.t_start = timecode(min(c.t_start, c.t_peak), duration, nd)
        c.t_end = timecode(max(c.t_end, c.t_peak), duration, nd)
        c.confidence = round(c.confidence, 4)
        c.threshold = threshold_for(thresholds, c.type_id)
        if c.origin != "risk":
            c.gate = gate_reasons(c, types, cfg, catalog_ok)
        c.reported = (c.suppressed_by is None and not c.gate and c.confidence > 0
                      and c.confidence >= c.threshold)

    max_v = int(_f(ocfg.get("max_violations", 10), 10))
    reported = sorted((c for c in cands if c.reported), key=lambda c: -c.confidence)
    for c in reported[max_v:]:
        c.reported = False
    reported = sorted(reported[:max_v], key=lambda c: (c.t_peak, -c.confidence))
    verdict = "violation" if reported else "no_violation"
    timeline = build_timeline(duration, curve, cands, cfg)
    cands.sort(key=lambda c: (c.t_peak, c.type_id))
    return FusionOutcome(cands, reported, verdict, timeline, duration, warnings)


# --------------------------------------------------------------------------- публичный API


def _clip_duration(clip: Any, events: Sequence[Any], risk: Sequence[Any]) -> float:
    d = _f(getattr(clip, "duration", 0.0), 0.0)
    if d > 0:
        return d
    cands = [0.0]
    times = getattr(clip, "times", None) or []
    if len(times):
        cands.append(_f(times[-1]))
    for ev in events or []:
        cands.append(_f(_get(ev, "t_end"), 0.0))
    for item in risk or []:
        try:
            cands.append(_f(item[0]))
        except (TypeError, IndexError, KeyError):
            continue
    return max(cands)


def _raw_scores(events: Sequence[Any], risk: Sequence[Any], outcome: FusionOutcome) -> dict[str, Any]:
    risk_pts = []
    for item in risk or []:
        try:
            risk_pts.append([round(_f(item[0]), 3), round(_f(item[1]), 4)])
        except (TypeError, IndexError, KeyError):
            continue
    return {
        "version": 1,
        "fusion_version": FUSION_VERSION,
        "duration": round(outcome.duration, 3),
        "verdict": outcome.verdict,
        "max_conf": max((c.confidence for c in outcome.candidates if c.suppressed_by is None), default=0.0),
        "risk_max": max((p[1] for p in risk_pts), default=0.0),
        "candidates": [c.to_raw() for c in outcome.candidates],
        "inputs": {"events": [event_to_dict(e) for e in events or []], "risk": risk_pts},
    }


def _evidence_enabled(cfg: dict[str, Any]) -> bool:
    if os.environ.get("SENTINEL_FUSION_NO_FRAMES", "").strip() in {"1", "true", "yes"}:
        return False
    return bool((cfg.get("evidence") or {}).get("enabled", True))


_GATE_RU = {
    "risk_only": "только высокий риск временной модели — нет наблюдаемого факта",
    "no_observable_fact": "нет конкретного наблюдаемого факта от правил/VLM",
    "temporal_only": "сигнал только временной модели",
    "no_norm": "для типа нет применимой нормы в каталоге",
    "observation_only": "наблюдение (телефон, еда, дверь, второй человек) само по себе не нарушение",
    "valuables_removed": "после ухода кассира наличные и сумка убраны, сейф закрыт — норма",
    "client_side": "человек на клиентской стороне напротив кассира — норма",
    "insufficient_context": "этап не наблюдался полностью — неизвестно, а не «нет» и не «да»",
    "norms_catalog_unavailable": "каталог норм недоступен",
    "below_threshold": "скор ниже порога",
    "suppressed": "перекрыт более сильным нарушением другого типа",
    "over_limit": "превышен лимит нарушений в ответе",
}


def _hint(c: Candidate, types: dict[str, dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    reasons = list(c.gate)
    if c.suppressed_by:
        reasons.append("suppressed")
    if not reasons and c.confidence < c.threshold:
        reasons.append("below_threshold")
    if not reasons:
        reasons.append("over_limit")
    return {
        "type_id": c.type_id,
        "type": str(type_info(types, c.type_id).get("name") or c.type_id),
        "timecode_sec": c.t_peak,
        "t_start": c.t_start,
        "t_end": c.t_end,
        "score": c.confidence,
        "threshold": c.threshold,
        "facts": clean_facts(c.facts, cfg),
        "reasons": reasons,
        "reasons_ru": [_GATE_RU.get(r, r) for r in reasons],
        "suppressed_by": c.suppressed_by,
        "origin": c.origin,
        "assessment": c.assessment,
        "policies": c.extra.get("policies", []),
        "counter_facts": c.extra.get("counter_facts", []),
    }


def _norms_ui(types: dict[str, dict[str, Any]], type_id: str) -> list[dict[str, Any]]:
    """Нормы для UI-отчёта: статус проверки явно, неподтверждённые не выдаются за подтверждённые."""
    out = []
    for r in rule_refs(types, type_id):
        out.append({
            "document": r.document, "clause": r.clause, "url": r.url, "quote": r.quote,
            "verified": bool(r.verified),
            "status_ru": "норма сверена с источником" if r.verified else "НЕ подтверждено: требует сверки юристом",
        })
    return out


def _verified_norm_text(types: dict[str, dict[str, Any]], type_id: str, cfg: dict[str, Any]) -> Optional[str]:
    """Короткая ссылка на первую СВЕРЕННУЮ норму для текста описания; неподтверждённые не цитируются."""
    if str((cfg.get("description") or {}).get("cite_norm", "verified_only")).lower() != "verified_only":
        return None
    for r in rule_refs(types, type_id):
        if r.verified:
            return f"{r.document}, {r.clause}" if r.clause else r.document
    return None


def fuse(clip: Any, events: list[Any], risk: list[tuple[float, float]], catalog: dict, job_id: str
         ) -> AnalysisResult:
    """Слияние событий → AnalysisResult(status="done"). Сигнатура заморожена контрактом v1.

    Полноту анализа (упавшие стадии) fuse не видит — оркестратор вызывает
    ``ml.fusion.apply_completeness(result, stages)`` после fuse (см. README).
    """
    t0 = time.perf_counter()
    events = list(events or [])
    risk = list(risk or [])
    cfg = load_config()
    thresholds = load_thresholds()
    try:
        types, warnings = resolve_catalog(catalog)
        catalog_ok = catalog_has_norms(types)
        duration = _clip_duration(clip, events, risk)
        outcome = fuse_scores(events, risk, duration, types, cfg, thresholds, catalog_ok=catalog_ok)
    except Exception as exc:  # noqa: BLE001 — аварийный режим: не теряем нарушение и не выдаём ложно-чистый
        log.exception("fusion failed, emergency mode")
        return _emergency_result(clip, events, catalog, job_id, cfg, thresholds, exc)
    warnings = warnings + outcome.warnings
    # Таймкод = PTS реального кадра (сетка clip.times / fps) → совпадает с основным кадром-доказательством.
    times, fps = clip_times(clip), _f(getattr(clip, "fps", 0.0), 0.0)
    nd = int(_f((cfg.get("output") or {}).get("timecode_round", 1), 1))
    for cand in outcome.reported:
        pts = snap_time(times, fps, duration, cand.t_peak)
        if abs(pts - cand.t_peak) <= 0.5:
            cand.extra["t_peak_pts"] = round(pts, 4)
            cand.t_peak = pts
            cand.t_start = min(cand.t_start, pts)
            cand.t_end = max(cand.t_end, pts)
        cand.t_peak = timecode(cand.t_peak, duration, nd)
    if not catalog_ok:
        warnings.append("norms: в каталоге нет ни одной нормы — нарушения выдаются без ссылок на нормы")
    coverage = coverage_info(clip, duration, cfg)
    t_scores = time.perf_counter() - t0

    renderer = None
    if _evidence_enabled(cfg) and outcome.reported:
        try:
            from .evidence import EvidenceRenderer  # lazy: numpy/cv2

            renderer = EvidenceRenderer(clip, cfg)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"evidence disabled: {exc!r}")

    violations: list[Violation] = []
    overlays: list[dict[str, Any]] = []
    for i, cand in enumerate(outcome.reported):
        info = type_info(types, cand.type_id)
        try:
            description = build_description(cand.type_id, info, cand.facts, cand.t_peak, cfg, cand.confidence,
                                            norm_ref=_verified_norm_text(types, cand.type_id, cfg))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"description fallback for {cand.key}: {exc!r}")
            description = (f"На {format_time(cand.t_peak)} зафиксированы признаки возможного нарушения "
                           f"«{info.get('name', cand.type_id)}». Требуется проверка человеком.")
        frames: list[Any] = []
        ui: dict[str, Any] = {}
        if renderer is not None:
            try:
                frames, ui = renderer.render(cand, info)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"evidence failed for {cand.key}: {exc!r}")
        violations.append(
            Violation(
                type=str(info.get("name") or cand.type_id),
                timecode_sec=float(cand.t_peak),
                description=description,
                evidence_frames=frames,
                type_id=cand.type_id,
                confidence=round(float(cand.confidence), 3),
                t_start=float(cand.t_start),
                t_end=float(cand.t_end),
                rules=rule_refs(types, cand.type_id),
            )
        )
        overlays.append({
            "violation_index": i,
            "type_id": cand.type_id,
            "timecode_sec": cand.t_peak,
            "frames": ui.get("frames", []),
            "shapes": ui.get("shapes", []),
            "facts": clean_facts(cand.facts, cfg),
            "norms": _norms_ui(types, cand.type_id),
            "sources": {k: round(v, 3) for k, v in cand.conf_by_source.items()},
            "vlm": cand.vlm_state,
        })
    if renderer is not None:
        warnings.extend(renderer.warnings)

    floor = _f((cfg.get("output") or {}).get("candidate_floor", 0.15), 0.15)
    hints = [_hint(c, types, cfg) for c in outcome.candidates if not c.reported and c.confidence >= floor]
    meta = {
        "scores_raw": _raw_scores(events, risk, outcome),
        "fusion": {
            "version": FUSION_VERSION,
            "combine": str(cfg.get("combine", "max")),
            "score_note": "confidence — эвристический скор 0..1 по коррелированным источникам, не вероятность",
            "thresholds": {"default": thresholds.get("default"), "per_type": thresholds.get("per_type"),
                           "tuned": thresholds.get("tuned", False)},
            "n_events_in": len(events),
            "n_candidates": len(outcome.candidates),
            "timings": {
                "scores_s": round(t_scores, 4),
                "evidence_s": round(renderer.elapsed if renderer is not None else 0.0, 4),
                "total_s": round(time.perf_counter() - t0, 4),
            },
        },
        "ui": {
            "review_hints": hints,
            "evidence_overlays": overlays,
            "norms_catalog": "ok" if catalog_ok else "unavailable",
        },
        "warnings": warnings,
        "disclaimer": "Результат — подсказка для проверяющего, а не юридическое заключение.",
    }
    # Наблюдение fusion о покрытии. Канонический meta["coverage"] (contracts.extensions.Coverage) считает A03 по
    # статусам стадий, проверяет сериализатор A01 перед no_violation — этот ключ fusion НЕ пишет.
    meta["fusion"]["coverage"] = coverage
    ccfg = cfg.get("completeness") or {}
    unknown_is_gap = str(ccfg.get("unknown_coverage", "no_verdict")).lower() not in {"done", "ignore"}
    incomplete = coverage["complete"] is False or (coverage["complete"] is None and unknown_is_gap)
    if coverage["complete"] is None:
        warnings.append("coverage: нет данных о покрытии видео (clip.meta / clip.times)")
    verdict: Optional[str] = outcome.verdict
    if ccfg.get("coverage_gate", True) and incomplete:
        reason_list = coverage["reasons"] or ["покрытие видео анализом неизвестно"]
        meta["analysis_complete"] = False
        if outcome.verdict == "violation":
            warnings.append("анализ видео неполный (" + "; ".join(reason_list) + "): найденные нарушения "
                            "показаны, но другие могли быть пропущены")
        else:
            # «Нарушений нет» не утверждаем: вердикт не выносится (verdict=None). status остаётся "done":
            # status="failed" от fusion оркестратор A03 трактует как падение модуля и подменяет результат
            # встроенным слиянием; итоговый failed/no_violation решают A03 (покрытие) и сериализатор A01.
            msg = ("Анализ видео неполный: " + "; ".join(reason_list) +
                   ". Отсутствие нарушений не подтверждено — вердикт не выносится.")
            meta["ui"]["incomplete"] = {"failed_stages": [], "details": {"coverage": reason_list}, "message_ru": msg}
            warnings.append(msg)
            verdict = None
    return AnalysisResult(
        job_id=str(job_id),
        status="done",
        verdict=verdict,  # type: ignore[arg-type]
        violations=violations,
        timeline=outcome.timeline,
        meta=meta,
    )


def _emergency_result(clip: Any, events: list[Any], catalog: Any, job_id: str, cfg: dict[str, Any],
                      thresholds: dict[str, Any], exc: Exception) -> AnalysisResult:
    """Слияние упало. Найденное событие с фактом ≥ τ → violation (неполный анализ помечен);
    иначе — технический status=failed без вердикта, а НЕ скрытый «чистый» no_violation."""
    try:
        types, _ = resolve_catalog(catalog)
    except Exception:  # noqa: BLE001
        types = {}
    duration = _f(getattr(clip, "duration", 0.0), 0.0)
    violations: list[Violation] = []
    for ev in events:
        try:
            type_id = str(_get(ev, "type_id", "V99") or "V99")
            conf = _clamp(_f(_get(ev, "confidence"), 0.0), 0.0, 1.0)
            facts = clean_facts(_get(ev, "facts", []) or [], cfg)
            if conf < threshold_for(thresholds, type_id) or not facts:
                continue
            info = type_info(types, type_id) if types else {"name": type_id}
            t = timecode(_f(_get(ev, "t_peak"), 0.0), duration)
            violations.append(Violation(
                type=str(info.get("name") or type_id), timecode_sec=t,
                description=(f"На {format_time(t)} видно: {'; '.join(facts)}. Признаки возможного нарушения "
                             f"«{info.get('name', type_id)}». Требуется проверка человеком."),
                evidence_frames=[], type_id=type_id, confidence=round(conf, 3), t_start=t, t_end=t,
                rules=rule_refs(types, type_id) if types else []))
        except Exception:  # noqa: BLE001
            continue
    violations = sorted(violations, key=lambda v: -v.confidence)[:10]
    violations.sort(key=lambda v: v.timecode_sec)
    error = {"stage": "fusion", "error": repr(exc)}
    meta: dict[str, Any] = {"warnings": [f"fusion emergency mode: {exc!r}"], "fusion": {"version": FUSION_VERSION},
                            "analysis_complete": False, "errors": [error]}
    if violations:
        return AnalysisResult(job_id=str(job_id), status="done", verdict="violation", violations=violations,
                              timeline=[], meta=meta)
    meta["failure"] = {"reason": "fusion_error",
                       "message_ru": "Анализ не завершён из-за технической ошибки слияния; вердикт не выносится."}
    return AnalysisResult(job_id=str(job_id), status="failed", verdict=None, violations=[], timeline=[], meta=meta)
