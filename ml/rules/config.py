"""Загрузка и валидация ``rules.yaml`` (декларативные правила)."""
from __future__ import annotations

import copy
import os
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .expr import CompiledExpr, ExprError, compile_expr

try:
    from contracts.sentinel_contracts import SIGNALS  # noqa
except ImportError:  # пока ветка A01 не смерджена
    from contracts.sentinel_contracts import SIGNALS  # noqa

__all__ = [
    "Clause",
    "RuleSpec",
    "InactionSpec",
    "RulesConfig",
    "ConfigError",
    "DEFAULT_RULES_PATH",
    "load_rules_config",
    "build_config",
    "hypothesis_overrides",
    "profile_hypotheses",
    "parse_rules_config",
    "apply_overrides",
    "read_raw_config",
    "RULE_KINDS",
    "T_PEAK_POLICIES",
]

DEFAULT_RULES_PATH = Path(__file__).with_name("rules.yaml")
ENV_RULES_PATH = "SENTINEL_RULES_YAML"
ENV_RULES_PROFILE = "SENTINEL_RULES_PROFILE"
T_PEAK_POLICIES = ("evidence", "transition", "onset", "onset_plus", "max", "center")
SMOOTH_METHODS = ("none", "median", "mean")
RULE_KINDS = ("violation", "observation")
GATE_MODES = ("strict", "lenient")
TYPE_ID_RE = re.compile(r"^V\d{2}$")
CONF_VARS = ("mean_strength", "max_strength", "duration", "dur_term", "base", "w_strength",
             "w_duration", "tau_s", "n_frames")


class ConfigError(ValueError):
    """Ошибка в rules.yaml."""


@dataclass
class InactionSpec:
    """Нарушение «по бездействию»: нужен наблюдаемый полный этап.

    * ``before`` должно быть *доказано* (нижняя граница) хотя бы в одном кадре за
      ``before_window_s`` до начала сегмента (например, кассир был у открытого сейфа);
    * сегмент не должен начинаться в первые ``edge_s`` ролика (начало этапа не снято);
    * доля кадров сегмента, где все сигналы правила известны, ≥ ``min_coverage``;
    * сегмент короче ``min_duration_s``, упирающийся в конец ролика, — ``insufficient_context``.
    """

    before: Optional[CompiledExpr]
    before_window_s: float = 30.0
    edge_s: float = 1.0
    min_coverage: float = 0.8
    # предшествующее состояние ДОКАЗАННО ложно во всём окне: insufficient (по умолчанию) или reject
    # (это не та ситуация — например, V08: клиента не было вовсе → внутренний пересчёт, исключение A13)
    before_false: str = "insufficient"


@dataclass
class Clause:
    """Именованная часть правила: обязательный предикат, исключение или применимость (для trace)."""

    id: str
    label: str
    expr: CompiledExpr


@dataclass
class RuleSpec:
    id: str
    type_id: str
    comment: str
    strength: CompiledExpr
    gate: Optional[CompiledExpr]
    kind: str = "observation"
    enabled: bool = True
    status_reason: str = ""
    gate_mode: str = "strict"
    anchor: Optional[CompiledExpr] = None
    inaction: Optional[InactionSpec] = None
    applicability: Optional[Clause] = None                       # строго: должна быть доказана
    predicates: list[Clause] = field(default_factory=list)       # обязательные видимые предикаты (И)
    exclusions: list[Clause] = field(default_factory=list)       # блокируют, только если доказаны
    trigger: Optional[CompiledExpr] = None                       # переход «начало действия» для t_peak
    t_peak_lookback_s: float = 5.0
    t_peak_fallback: str = "evidence"
    on: float = 0.5
    off: float = 0.35
    smooth_method: str = "median"
    smooth_s: float = 0.6
    min_duration_s: float = 1.5
    merge_gap_s: float = 1.5
    t_peak_policy: str = "evidence"
    t_peak_offset_s: float = 0.0
    evidence_tol: float = 0.05
    conf_base: float = 0.35
    conf_w_strength: float = 0.35
    conf_w_duration: float = 0.25
    conf_tau_s: float = 6.0
    conf_min: float = 0.05
    conf_max: float = 0.95
    conf_expr: Optional[CompiledExpr] = None
    weight: float = 1.0
    facts: list[str] = field(default_factory=list)
    facts_stage: list[str] = field(default_factory=list)   # только если этап «по бездействию» наблюдался полностью
    highlight: list[str] = field(default_factory=list)
    yield_to: list[str] = field(default_factory=list)
    yield_factor: float = 0.7
    max_events: int = 3
    state_after_window_s: float = 0.0       # >0: событие несёт extra.state_after (стол/сейф после ухода)
    state_after_min_s: float = 1.0          # кадр «после» не раньше, чем через столько секунд после ухода
    person_side: Optional[str] = None       # extra.person_side для A12 (cashier | client); None — не задаётся

    @property
    def mandatory_names(self) -> frozenset[str]:
        """Сигналы обязательных предикатов (strength), без исключений/якорей."""
        return frozenset(self.strength.names)

    @property
    def names(self) -> frozenset[str]:
        names = set(self.strength.names)
        extra = [self.gate, self.anchor, self.inaction.before if self.inaction else None, self.trigger,
                 self.applicability.expr if self.applicability else None]
        extra += [c.expr for c in self.exclusions]
        for e in extra:
            if e is not None:
                names |= e.names
        return frozenset(names)


@dataclass
class RulesConfig:
    version: int
    rules: list[RuleSpec]
    signal_defaults: dict[str, float] = field(default_factory=dict)
    signal_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    ffill_max_s: float = 1.0
    known_min_conf: float = 0.5
    observed_one_sided: bool = True
    merge_policy: str = "max"
    derive_from_tracks: bool = True
    presence_tol_s: float = 0.6
    bbox_tol_s: float = 1.0
    max_bboxes: int = 6
    nms_window_s: float = 5.0
    same_type_merge_gap_s: float = 2.0
    corroboration_bonus: float = 0.05
    max_events: int = 12
    min_confidence: float = 0.15
    require_verified_norm: bool = True
    emit_review_observations: bool = True
    emit_insufficient_context: bool = True
    t_peak_policy_override: Optional[str] = None   # A/B: одна политика t_peak для всех правил
    observation_conf_cap: float = 0.45
    insufficient_conf_cap: float = 0.3
    derived_conf_factor: float = 0.85
    state_after_min_known: float = 0.6     # доля известных кадров окна «после», иначе значение None
    severity_factor: dict[str, float] = field(default_factory=lambda: {"high": 1.0, "medium": 0.97, "low": 0.93})
    known_signals: frozenset[str] = frozenset(SIGNALS)
    source_path: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    profile: str = "baseline"
    active_hypotheses: list[str] = field(default_factory=list)
    hypotheses: dict[str, dict[str, Any]] = field(default_factory=dict)
    profiles: dict[str, list[str]] = field(default_factory=dict)

    @property
    def enabled_rules(self) -> list[RuleSpec]:
        return [r for r in self.rules if r.enabled]

    def rule(self, rule_id: str) -> RuleSpec:
        for r in self.rules:
            if r.id == rule_id:
                return r
        raise KeyError(rule_id)

    def rules_table(self) -> list[dict[str, Any]]:
        """Таблица правил: id, тип, вид, включено ли и почему (для README/UI)."""
        return [{"id": r.id, "type_id": r.type_id, "kind": r.kind, "enabled": r.enabled,
                 "inaction": r.inaction is not None, "reason": r.status_reason} for r in self.rules]


def _num(d: Mapping[str, Any], key: str, default: float, lo: float = -1e9, hi: float = 1e9, where: str = "") -> float:
    v = d.get(key, default)
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ConfigError(f"{where}{key}: ожидалось число, получено {v!r}") from None
    if not (lo <= f <= hi):
        raise ConfigError(f"{where}{key}={f} вне диапазона [{lo}, {hi}]")
    return f


def _deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for k, v in (over or {}).items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _template_fields(template: str) -> list[str]:
    fields = []
    for _, fname, _, _ in string.Formatter().parse(template):
        if fname is None:
            continue
        if not fname or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", fname):
            raise ConfigError(f"Недопустимый плейсхолдер {{{fname}}} в шаблоне {template!r}")
        fields.append(fname)
    return fields


def read_raw_config(path: Optional[str | os.PathLike] = None) -> tuple[dict[str, Any], str]:
    import yaml  # ленивый импорт

    p = Path(path or os.environ.get(ENV_RULES_PATH) or DEFAULT_RULES_PATH)
    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{p}: корень YAML должен быть словарём")
    return dict(raw), str(p)


def _parse_value(text: str) -> Any:
    import yaml

    return yaml.safe_load(text)


def apply_overrides(raw: Mapping[str, Any], sets: Sequence[str]) -> dict[str, Any]:
    """Применить переопределения ``a.b.c=value`` (значение — YAML-литерал).

    ``rules.<rule_id>.<param>`` адресует правило по id (rules в YAML — список).
    """
    out = copy.deepcopy(dict(raw))
    for item in sets or ():
        if "=" not in item:
            raise ConfigError(f"Переопределение без '=': {item!r}")
        key, value = item.split("=", 1)
        parts = [p for p in key.strip().split(".") if p]
        if not parts:
            raise ConfigError(f"Пустой ключ в {item!r}")
        val = _parse_value(value)
        node: Any = out
        if parts[0] == "rules" and len(parts) >= 3:
            rules = node.get("rules") or []
            target = next((r for r in rules if isinstance(r, dict) and r.get("id") == parts[1]), None)
            if target is None:
                raise ConfigError(f"Нет правила с id={parts[1]!r}")
            node, parts = target, parts[2:]
        for p in parts[:-1]:
            nxt = node.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                node[p] = nxt
            node = nxt
        node[parts[-1]] = val
    return out


def hypothesis_overrides(raw: Mapping[str, Any], names: Sequence[str]) -> list[str]:
    """Переопределения для списка гипотез (``hypotheses.<H>.set``)."""
    hyps = raw.get("hypotheses") or {}
    out: list[str] = []
    for name in names:
        h = hyps.get(name)
        if not isinstance(h, Mapping):
            raise ConfigError(f"Нет гипотезы {name!r} (есть: {', '.join(sorted(hyps)) or '—'})")
        out += [str(x) for x in (h.get("set") or [])]
    return out


def profile_hypotheses(raw: Mapping[str, Any], profile: str) -> list[str]:
    profiles = raw.get("profiles") or {"baseline": []}
    if profile not in profiles:
        raise ConfigError(f"Нет профиля {profile!r} (есть: {', '.join(sorted(profiles))})")
    return [str(x) for x in (profiles.get(profile) or [])]


def _compile(r: Mapping[str, Any], key: str, where: str) -> Optional[CompiledExpr]:
    v = r.get(key)
    if v in (None, ""):
        return None
    try:
        return compile_expr(v)
    except ExprError as exc:
        raise ConfigError(f"{where}{key}: {exc}") from None


def _clauses(items: Any, where: str) -> list[Clause]:
    out: list[Clause] = []
    for i, item in enumerate(items or []):
        if isinstance(item, str):
            item = {"id": f"p{i + 1}", "expr": item}
        if not isinstance(item, Mapping) or item.get("expr") in (None, ""):
            raise ConfigError(f"{where}[{i}]: ожидался {{id, label, expr}}")
        try:
            expr = compile_expr(item["expr"])
        except ExprError as exc:
            raise ConfigError(f"{where}[{i}]: {exc}") from None
        cid = str(item.get("id") or f"p{i + 1}")
        out.append(Clause(id=cid, label=str(item.get("label") or cid), expr=expr))
    return out


def parse_rules_config(raw: Mapping[str, Any], source_path: Optional[str] = None) -> RulesConfig:
    warnings: list[str] = []
    glob = dict(raw.get("global") or {})
    defaults = dict(raw.get("defaults") or {})
    extra_signals = [str(s) for s in (raw.get("extra_signals") or [])]
    known = frozenset(SIGNALS) | frozenset(extra_signals)
    rules_raw = raw.get("rules")
    if not isinstance(rules_raw, list) or not rules_raw:
        raise ConfigError("rules: ожидался непустой список правил")

    ranges: dict[str, tuple[float, float]] = {}
    for k, v in (raw.get("signal_ranges") or {}).items():
        try:
            lo, hi = float(v[0]), float(v[1])
        except (TypeError, ValueError, IndexError):
            raise ConfigError(f"signal_ranges.{k}: ожидалась пара [lo, hi]") from None
        if hi < lo:
            raise ConfigError(f"signal_ranges.{k}: hi < lo")
        ranges[str(k)] = (lo, hi)

    sev = glob.get("severity_factor") or {"high": 1.0, "medium": 0.97, "low": 0.93}
    cfg = RulesConfig(
        version=int(raw.get("version", 1)),
        rules=[],
        signal_defaults={str(k): float(v) for k, v in (raw.get("signal_defaults") or {}).items()},
        signal_ranges=ranges,
        ffill_max_s=_num(glob, "ffill_max_s", 1.0, 0, 60, "global."),
        known_min_conf=_num(glob, "known_min_conf", 0.5, 0, 1, "global."),
        observed_one_sided=bool(glob.get("observed_one_sided", True)),
        merge_policy=str(glob.get("merge_policy", "max")),
        derive_from_tracks=bool(glob.get("derive_from_tracks", True)),
        presence_tol_s=_num(glob, "presence_tol_s", 0.6, 0, 10, "global."),
        bbox_tol_s=_num(glob, "bbox_tol_s", 1.0, 0, 10, "global."),
        max_bboxes=int(_num(glob, "max_bboxes", 6, 0, 50, "global.")),
        nms_window_s=_num(glob, "nms_window_s", 5.0, 0, 600, "global."),
        same_type_merge_gap_s=_num(glob, "same_type_merge_gap_s", 2.0, 0, 600, "global."),
        corroboration_bonus=_num(glob, "corroboration_bonus", 0.05, 0, 0.5, "global."),
        max_events=int(_num(glob, "max_events", 12, 1, 1000, "global.")),
        min_confidence=_num(glob, "min_confidence", 0.15, 0, 1, "global."),
        require_verified_norm=bool(glob.get("require_verified_norm", True)),
        emit_review_observations=bool(glob.get("emit_review_observations", True)),
        emit_insufficient_context=bool(glob.get("emit_insufficient_context", True)),
        t_peak_policy_override=(str(glob["t_peak_policy_override"])
                                if glob.get("t_peak_policy_override") not in (None, "", "none") else None),
        observation_conf_cap=_num(glob, "observation_conf_cap", 0.45, 0, 1, "global."),
        insufficient_conf_cap=_num(glob, "insufficient_conf_cap", 0.3, 0, 1, "global."),
        derived_conf_factor=_num(glob, "derived_conf_factor", 0.85, 0, 1, "global."),
        state_after_min_known=_num(glob, "state_after_min_known", 0.6, 0, 1, "global."),
        severity_factor={str(k): float(v) for k, v in sev.items()},
        known_signals=known,
        source_path=source_path,
        warnings=warnings,
    )
    if cfg.t_peak_policy_override is not None and cfg.t_peak_policy_override not in T_PEAK_POLICIES:
        raise ConfigError(f"global.t_peak_policy_override={cfg.t_peak_policy_override!r}")
    if cfg.merge_policy not in ("max", "min", "last"):
        raise ConfigError(f"global.merge_policy: {cfg.merge_policy!r} (ожидалось max|min|last)")

    seen: set[str] = set()
    for i, item in enumerate(rules_raw):
        if not isinstance(item, Mapping):
            raise ConfigError(f"rules[{i}]: ожидался словарь")
        r = _deep_merge(defaults, item)
        rid = str(r.get("id") or "").strip()
        where = f"rules[{rid or i}]."
        if not rid:
            raise ConfigError(f"rules[{i}]: нет id")
        if rid in seen:
            raise ConfigError(f"Дублирующийся id правила: {rid}")
        seen.add(rid)
        type_id = str(r.get("type_id") or "")
        if not TYPE_ID_RE.match(type_id):
            raise ConfigError(f"{where}type_id={type_id!r}: ожидалось V01..V99")
        kind = str(r.get("kind", "observation"))
        if kind not in RULE_KINDS:
            raise ConfigError(f"{where}kind={kind!r} (ожидалось {'|'.join(RULE_KINDS)})")
        gate_mode = str(r.get("gate_mode", "strict"))
        if gate_mode not in GATE_MODES:
            raise ConfigError(f"{where}gate_mode={gate_mode!r} (ожидалось {'|'.join(GATE_MODES)})")
        predicates = _clauses(r.get("predicates"), where + "predicates")
        exclusions = _clauses(r.get("exclusions"), where + "exclusions")
        applicability = None
        if r.get("applicability") not in (None, ""):
            app = r["applicability"]
            app = app if isinstance(app, Mapping) else {"id": "applicability", "expr": app}
            applicability = _clauses([dict(app, id=app.get("id", "applicability"))], where + "applicability")[0]
        if predicates and r.get("strength") not in (None, ""):
            raise ConfigError(f"{where}: задайте либо strength, либо predicates")
        if predicates:
            joined = ", ".join(f"({c.expr.source})" for c in predicates)
            strength = compile_expr(f"min({joined})" if len(predicates) > 1 else predicates[0].expr.source)
        else:
            strength = _compile(r, "strength", where)
        if strength is None:
            raise ConfigError(f"{where}strength/predicates: обязательное выражение")
        gate = _compile(r, "gate", where)
        anchor = _compile(r, "anchor", where)
        inaction = None
        if r.get("inaction"):
            ina = r["inaction"]
            if not isinstance(ina, Mapping):
                raise ConfigError(f"{where}inaction: ожидался словарь")
            inaction = InactionSpec(
                before=_compile(ina, "before", where + "inaction."),
                before_window_s=_num(ina, "before_window_s", 30.0, 0, 3600, where + "inaction."),
                edge_s=_num(ina, "edge_s", 1.0, 0, 60, where + "inaction."),
                min_coverage=_num(ina, "min_coverage", 0.8, 0, 1, where + "inaction."),
                before_false=str(ina.get("before_false", "insufficient")),
            )
            if inaction.before_false not in ("insufficient", "reject"):
                raise ConfigError(f"{where}inaction.before_false={inaction.before_false!r} (insufficient | reject)")
        tp = r.get("t_peak") or {}
        if isinstance(tp, str):
            tp = {"policy": tp}
        policy = str(tp.get("policy", "evidence"))
        if policy not in T_PEAK_POLICIES:
            raise ConfigError(f"{where}t_peak.policy={policy!r} (ожидалось {'|'.join(T_PEAK_POLICIES)})")
        fallback = str(tp.get("fallback", "evidence"))
        if fallback not in T_PEAK_POLICIES or fallback == "transition":
            raise ConfigError(f"{where}t_peak.fallback={fallback!r}")
        trigger = _compile(tp, "trigger", where + "t_peak.")
        smooth = r.get("smooth") or {}
        if isinstance(smooth, str):
            smooth = {"method": smooth}
        method = str(smooth.get("method", "median"))
        if method not in SMOOTH_METHODS:
            raise ConfigError(f"{where}smooth.method={method!r}")
        conf = r.get("confidence") or {}
        conf_expr = None
        if conf.get("expr"):
            try:
                conf_expr = compile_expr(conf["expr"], allowed_names=set(CONF_VARS))
            except ExprError as exc:
                raise ConfigError(f"{where}confidence.expr: {exc}") from None
        on = _num(r, "on", 0.5, 0, 1, where)
        off = _num(r, "off", min(on, 0.35), 0, 1, where)
        if off > on:
            raise ConfigError(f"{where}off={off} > on={on}")
        facts = [str(f) for f in (r.get("facts") or [])]
        facts_stage = [str(f) for f in (r.get("facts_stage") or [])]
        for tmpl in facts + facts_stage:
            _template_fields(tmpl)
        sa_cfg = r.get("state_after") or {}
        if not isinstance(sa_cfg, Mapping):
            raise ConfigError(f"{where}state_after: ожидался словарь {{window_s, after_min_s}}")
        if r.get("person_side") not in (None, "", "cashier", "client"):
            raise ConfigError(f"{where}person_side={r.get('person_side')!r} (ожидалось cashier | client)")
        enabled = bool(r.get("enabled", True))
        reason = str(r.get("status_reason") or "").strip()
        if not reason:
            warnings.append(f"{rid}: не указана причина enabled/disabled (status_reason)")
        spec = RuleSpec(
            id=rid,
            type_id=type_id,
            comment=str(r.get("comment") or "").strip(),
            strength=strength,
            gate=gate,
            kind=kind,
            enabled=enabled,
            status_reason=reason,
            gate_mode=gate_mode,
            anchor=anchor,
            inaction=inaction,
            applicability=applicability,
            predicates=predicates,
            exclusions=exclusions,
            trigger=trigger,
            t_peak_lookback_s=_num(tp, "lookback_s", 5.0, 0, 120, where + "t_peak."),
            t_peak_fallback=fallback,
            on=on,
            off=off,
            smooth_method=method,
            smooth_s=_num(smooth, "window_s", 0.6, 0, 30, where + "smooth."),
            min_duration_s=_num(r, "min_duration_s", 1.5, 0, 3600, where),
            merge_gap_s=_num(r, "merge_gap_s", 1.5, 0, 600, where),
            t_peak_policy=policy,
            t_peak_offset_s=_num(tp, "offset_s", 0.0, -30, 30, where + "t_peak."),
            evidence_tol=_num(tp, "evidence_tol", 0.05, 0, 1, where + "t_peak."),
            conf_base=_num(conf, "base", 0.35, 0, 1, where + "confidence."),
            conf_w_strength=_num(conf, "w_strength", 0.35, 0, 1, where + "confidence."),
            conf_w_duration=_num(conf, "w_duration", 0.25, 0, 1, where + "confidence."),
            conf_tau_s=_num(conf, "tau_s", 6.0, 0.01, 3600, where + "confidence."),
            conf_min=_num(conf, "min", 0.05, 0, 1, where + "confidence."),
            conf_max=_num(conf, "max", 0.95, 0, 1, where + "confidence."),
            conf_expr=conf_expr,
            weight=_num(r, "weight", 1.0, 0, 2, where),
            facts=facts,
            facts_stage=facts_stage,
            highlight=[str(h) for h in (r.get("highlight") or [])],
            yield_to=[str(y) for y in (r.get("yield_to") or [])],
            yield_factor=_num(r, "yield_factor", 0.7, 0, 1, where),
            max_events=int(_num(r, "max_events", 3, 1, 100, where)),
            state_after_window_s=_num(sa_cfg, "window_s", 0.0, 0, 600, where + "state_after."),
            state_after_min_s=_num(sa_cfg, "after_min_s", 1.0, 0, 60, where + "state_after."),
            person_side=(str(r["person_side"]) if r.get("person_side") not in (None, "") else None),
        )
        for nm in sorted(spec.names):
            if nm not in known:
                warnings.append(f"{rid}: сигнал {nm!r} не входит в SIGNALS/extra_signals")
        cfg.rules.append(spec)
    return cfg


_CACHE: dict[tuple[str, float, str, tuple[str, ...]], RulesConfig] = {}


def build_config(raw: Mapping[str, Any], src: Optional[str] = None, profile: Optional[str] = None,
                 hypotheses: Sequence[str] = (), overrides: Sequence[str] = ()) -> RulesConfig:
    """Профиль → его гипотезы → дополнительные гипотезы → ручные переопределения (в этом порядке)."""
    prof = profile or str(raw.get("default_profile") or "baseline")
    hyps = profile_hypotheses(raw, prof) + [h for h in hypotheses if h not in profile_hypotheses(raw, prof)]
    sets = hypothesis_overrides(raw, hyps) + list(overrides)
    cfg = parse_rules_config(apply_overrides(raw, sets) if sets else raw, source_path=src)
    cfg.profile = prof
    cfg.active_hypotheses = hyps
    cfg.hypotheses = {str(k): dict(v) for k, v in (raw.get("hypotheses") or {}).items() if isinstance(v, Mapping)}
    cfg.profiles = {str(k): [str(x) for x in (v or [])] for k, v in (raw.get("profiles") or {}).items()}
    return cfg


def load_rules_config(path: Optional[str | os.PathLike] = None, overrides: Sequence[str] = (),
                      profile: Optional[str] = None, hypotheses: Sequence[str] = ()) -> RulesConfig:
    """Загрузить rules.yaml с профилем (аргумент → env SENTINEL_RULES_PROFILE → default_profile).

    Production никогда не включает гипотезы сам: только явный профиль или переменная окружения.
    """
    p = Path(path or os.environ.get(ENV_RULES_PATH) or DEFAULT_RULES_PATH)
    prof = profile or os.environ.get(ENV_RULES_PROFILE) or None
    key = (str(p.resolve()), p.stat().st_mtime, prof or "", tuple(hypotheses))
    if not overrides and key in _CACHE:
        return _CACHE[key]
    raw, src = read_raw_config(p)
    cfg = build_config(raw, src, prof, hypotheses, overrides)
    if not overrides:
        _CACHE[key] = cfg
    return cfg
