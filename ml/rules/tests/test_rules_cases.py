"""Позитив/негатив для каждого включённого правила, чистые сценарии, валидность rules.yaml."""
from __future__ import annotations

import json

import numpy as np
import pytest

from ml.rules import CANDIDATE, load_rules_config, run_rules
from ml.rules.config import parse_rules_config, read_raw_config
from ml.rules.engine import confidence_for, evaluate
from ml.rules.synth import CLEAN_SCENARIOS, RULE_CASES, make_states, verified_catalog

CFG = load_rules_config()
CATALOG = verified_catalog()


def _events(spec: dict, catalog: dict = CATALOG) -> list:
    states = make_states(duration_s=spec["duration_s"], segments=spec["segments"], noise=spec.get("noise", 0.0))
    return run_rules(states, [], [], catalog)


def test_rules_yaml_valid_and_documented() -> None:
    assert CFG.warnings == [], CFG.warnings
    for r in CFG.rules:
        assert r.status_reason, f"{r.id}: нет status_reason"
        assert r.comment, f"{r.id}: нет comment"
        assert r.t_peak_policy == "evidence", f"{r.id}: t_peak по умолчанию — фактический момент доказательства"
        assert r.t_peak_offset_s == 0.0
    enabled = {r.id for r in CFG.enabled_rules}
    assert enabled == set(RULE_CASES), "у каждого включённого правила должен быть позитив/негатив в synth.RULE_CASES"


def test_observation_kinds_never_violation() -> None:
    by_id = {r.id: r for r in CFG.rules}
    for rid in ("V04_door_open_with_valuables", "V05_phone_in_hand_at_workplace", "V05_phone_to_ear",
                "V06_eating_drinking"):
        assert by_id[rid].kind == "observation", rid


def test_v08_checks_client_presence_not_posture() -> None:
    """V08 (A13: пересчёт не в присутствии клиента) — не «спиной к камере» и не «руки перекрыты»."""
    v08 = [r for r in CFG.rules if r.type_id == "V08"]
    assert v08, "нет правила V08"
    for r in v08:
        assert "client_present" in r.names, r.id
        assert not ({"hands_hidden", "cashier_facing_away"} & set(r.names)), r.id
        assert not r.enabled and r.status_reason


@pytest.mark.parametrize("rule_id", sorted(RULE_CASES))
def test_rule_positive(rule_id: str) -> None:
    case = RULE_CASES[rule_id]
    evs = [e for e in _events(case["pos"]) if e.type_id == case["type_id"]]
    assert evs, f"{rule_id}: нет события на позитиве"
    ev = max(evs, key=lambda e: e.confidence)
    assert rule_id in ev.extra["rule_ids"]
    assert ev.extra["assessment"] == case["assessment"]
    assert ev.source == "rules"
    # фактический момент доказательства: первый кадр ситуации, а не onset + 0.5
    assert abs(ev.t_peak - case["onset"]) <= 0.25, (ev.t_peak, case["onset"])
    assert ev.t_start <= ev.t_peak <= ev.t_end
    assert 0.0 < ev.confidence <= 1.0
    assert ev.facts and all(isinstance(f, str) and "?" not in f for f in ev.facts)
    tr = ev.extra["trace"]
    assert tr["rule_id"] in ev.extra["rule_ids"] and tr["strength"]["max"] >= 0.5
    json.dumps(ev.extra, ensure_ascii=False)  # trace сериализуем для API/UI


@pytest.mark.parametrize("rule_id", sorted(RULE_CASES))
def test_rule_negatives(rule_id: str) -> None:
    case = RULE_CASES[rule_id]
    for i, neg in enumerate(case["neg"]):
        evs = [e for e in _events(neg) if rule_id in e.extra["rule_ids"]]
        assert not evs, f"{rule_id}: негатив #{i} дал событие {[(e.t_start, e.extra['assessment']) for e in evs]}"


@pytest.mark.parametrize("name", sorted(CLEAN_SCENARIOS))
def test_clean_scenarios_produce_no_events(name: str) -> None:
    evs = _events(CLEAN_SCENARIOS[name])
    assert evs == [], [(e.type_id, e.t_start, e.extra["assessment"]) for e in evs]


@pytest.mark.parametrize("rule_id", sorted(r.id for r in CFG.rules))
def test_confidence_monotone_in_duration_formula(rule_id: str) -> None:
    rule = CFG.rule(rule_id)
    for strength in (0.5, 0.8, 1.0):
        confs = [confidence_for(rule, strength, strength, d)[0] for d in np.linspace(0.2, 120, 60)]
        assert all(b >= a - 1e-12 for a, b in zip(confs, confs[1:])), rule_id


@pytest.mark.parametrize("rule_id", [r for r, c in RULE_CASES.items() if c["assessment"] == CANDIDATE])
def test_confidence_monotone_in_duration_end_to_end(rule_id: str) -> None:
    case = RULE_CASES[rule_id]
    confs = []
    for extra in (0.0, 3.0, 8.0, 15.0):
        segs = []
        for t0, t1, vals in case["pos"]["segments"]:
            segs.append((t0, t1 + extra, vals))
        states = make_states(duration_s=case["pos"]["duration_s"] + 20.0, segments=segs)
        evs = [e for e in run_rules(states, [], [], CATALOG) if rule_id in e.extra["rule_ids"]]
        assert evs, (rule_id, extra)
        confs.append(max(e.confidence for e in evs))
    assert all(b >= a - 1e-9 for a, b in zip(confs, confs[1:])), (rule_id, confs)


def test_overrides_change_thresholds() -> None:
    raw, src = read_raw_config()
    from ml.rules.config import apply_overrides

    raw2 = apply_overrides(raw, ["rules.V01_outsider_in_cashier_area.min_duration_s=10", "global.nms_window_s=3"])
    cfg2 = parse_rules_config(raw2, src)
    assert cfg2.rule("V01_outsider_in_cashier_area").min_duration_s == 10.0
    assert cfg2.nms_window_s == 3.0
    case = RULE_CASES["V01_outsider_in_cashier_area"]
    states = make_states(duration_s=60.0, segments=case["pos"]["segments"])
    assert not [e for e in evaluate(states, [], [], CATALOG, cfg2).events if e.type_id == "V01"]


def test_invalid_configs_rejected() -> None:
    from ml.rules.config import ConfigError

    raw, _ = read_raw_config()
    bad_variants = [
        {**raw, "rules": []},
        {**raw, "rules": [{"id": "x", "type_id": "BAD", "strength": "a"}]},
        {**raw, "rules": [{"id": "x", "type_id": "V01", "strength": "__import__('os')"}]},
        {**raw, "rules": [{"id": "x", "type_id": "V01", "strength": "a", "kind": "legal"}]},
        {**raw, "rules": [{"id": "x", "type_id": "V01", "strength": "a", "on": 0.3, "off": 0.6}]},
        {**raw, "rules": [{"id": "x", "type_id": "V01", "strength": "a", "facts": ["{a.__class__}"]}]},
        {**raw, "rules": [{"id": "x", "type_id": "V01", "strength": "a", "t_peak": {"policy": "guess"}}]},
        {**raw, "rules": [{"id": "x", "type_id": "V01", "strength": "a"},
                          {"id": "x", "type_id": "V01", "strength": "a"}]},
    ]
    for v in bad_variants:
        with pytest.raises(ConfigError):
            parse_rules_config(v)


def test_facts_have_no_legal_claims() -> None:
    forbidden = ("нарушил", "виновн", "умысел", "без допуска", "похитил", "украл", "сумма", "подлинн")
    for rule in CFG.rules:
        for tpl in rule.facts:
            low = tpl.lower()
            assert not any(w in low for w in forbidden), (rule.id, tpl)
