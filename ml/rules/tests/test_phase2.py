"""Фаза 2: на каждую гипотезу — нарушение + чистый двойник + неизвестное; профиль phase2 на чистых."""
from __future__ import annotations

import pytest

from ml.rules import evaluate
from ml.rules.config import load_rules_config
from ml.rules.synth import CLEAN_SCENARIOS_PHASE2, PHASE2_CASES, make_states, make_tracks, make_zones, \
    verified_catalog

CAT = verified_catalog(["V01", "V02", "V03", "V04", "V05", "V06", "V07", "V08", "V09"])


def _events(case: dict, key: str, *hyps: str, profile: str = "baseline") -> list:
    cfg = load_rules_config(profile=profile, hypotheses=list(hyps))
    return evaluate(make_states(60.0, segments=case[key]), [], [], CAT, cfg).events


def _from_rule(evs: list, rule_id: str) -> list:
    return [e for e in evs if rule_id in e.extra.get("rule_ids", [])]


def test_baseline_profile_keeps_phase2_rules_off() -> None:
    cfg = load_rules_config()
    assert cfg.profile == "baseline" and cfg.active_hypotheses == []
    assert cfg.t_peak_policy_override is None
    for case in PHASE2_CASES.values():
        rule = cfg.rule(case["rule_id"])
        if case["hypothesis"] == "H6_phone_candidate":
            assert rule.enabled and rule.kind == "observation"
        else:
            assert not rule.enabled, rule.id


@pytest.mark.parametrize("name", sorted(PHASE2_CASES))
@pytest.mark.parametrize("tpeak", ["evidence", "transition"])
def test_positive(name: str, tpeak: str) -> None:
    case = PHASE2_CASES[name]
    hyps = [case["hypothesis"]] + (["H0_tpeak_transition"] if tpeak == "transition" else [])
    evs = _from_rule(_events(case, "pos", *hyps), case["rule_id"])
    assert evs, f"{name}: нет события"
    ev = evs[0]
    assert ev.type_id == case["type_id"]
    assert ev.extra["detection"] == case["detection"]
    assert abs(ev.t_peak - case["onset"]) <= 0.25, (name, tpeak, ev.t_peak)
    tr = ev.extra["trace"]
    assert tr["evidence"]["policy"] == tpeak and tr["interval"]["t_start"] <= ev.t_peak
    if tpeak == "transition" and case["detection"] == "violation_candidate":
        assert tr["evidence"]["kind"] == "false→true", tr["evidence"]
    assert all(c["at_peak"] == "true" for c in tr["clauses"] if c["kind"] == "predicate")
    assert all(c["at_peak"] != "true" for c in tr["clauses"] if c["kind"] == "exclusion")


@pytest.mark.parametrize("name", sorted(PHASE2_CASES))
def test_clean_twin(name: str) -> None:
    case = PHASE2_CASES[name]
    for key in ("twin", "twin2"):
        if key not in case:
            continue
        evs = _events(case, key, case["hypothesis"])
        assert not _from_rule(evs, case["rule_id"]), (name, key)
        assert not [e for e in evs if e.extra["detection"] == "violation_candidate"], (name, key)


@pytest.mark.parametrize("name", sorted(PHASE2_CASES))
def test_unknown_is_not_violation(name: str) -> None:
    case = PHASE2_CASES[name]
    evs = _events(case, "unknown", case["hypothesis"])
    assert not [e for e in evs if e.extra["detection"] == "violation_candidate"], name
    own = _from_rule(evs, case["rule_id"])
    if case["expect_insufficient"]:
        assert own and all(e.extra["detection"] == "insufficient_context" for e in own), name
    else:
        assert not own, name


@pytest.mark.parametrize("name", sorted(CLEAN_SCENARIOS_PHASE2))
def test_phase2_profile_clean_scenarios(name: str) -> None:
    spec = CLEAN_SCENARIOS_PHASE2[name]
    cfg = load_rules_config(profile="phase2")
    states = make_states(spec["duration_s"], segments=spec["segments"], noise=spec.get("noise", 0.0))
    evs = evaluate(states, [], [], CAT, cfg).events
    assert evs == [], [(e.type_id, e.extra["rule_ids"], e.t_start, e.extra["detection"]) for e in evs]


def test_multiple_real_events_kept_and_bounce_merged() -> None:
    cfg = load_rules_config(hypotheses=["H1_left_after_departure", "H0_tpeak_transition"])
    segs = [(20.0, 28.0, {"cashier_in_workplace": 0.0}), (40.0, 48.0, {"cashier_in_workplace": 0.0})]
    evs = _from_rule(evaluate(make_states(60.0, segments=segs), [], [], CAT, cfg).events,
                     "V02_left_cash_after_departure")
    assert [round(e.t_peak) for e in evs] == [20, 40]
    # дребезг: кассир «мигает» обратно на 0.4 с — одно событие
    segs = [(20.0, 35.0, {"cashier_in_workplace": 0.0}), (25.0, 25.4, {"cashier_in_workplace": 1.0})]
    evs = _from_rule(evaluate(make_states(60.0, segments=segs), [], [], CAT, cfg).events,
                     "V02_left_cash_after_departure")
    assert len(evs) == 1 and evs[0].t_peak == pytest.approx(20.0)


def test_v09_from_tracks_when_signal_absent() -> None:
    cfg = load_rules_config(hypotheses=["H8_personal_items"])
    tracks = make_tracks(60.0, spans=[(1, "person", "cashier", 0.0, 60.0, (0.3, 0.5, 0.45, 0.7)),
                                      (9, "bag", "unknown", 20.0, 40.0, (0.5, 0.6, 0.6, 0.7))])
    evs = evaluate(make_states(60.0), tracks, make_zones(), CAT, cfg).events
    v09 = [e for e in evs if e.type_id == "V09"]
    assert v09 and v09[0].extra["trace"]["derived_signals"].get("personal_items_on_desk") == "tracks"
    assert abs(v09[0].t_peak - 20.0) <= 0.25 and "bag#9" in v09[0].bboxes


def test_departure_at_clip_start_is_not_a_departure() -> None:
    """Кассир входит/отсутствует с начала ролика: ухода не было — кандидата нет."""
    cfg = load_rules_config(hypotheses=["H1_left_after_departure"])
    evs = evaluate(make_states(60.0, segments=[(0.0, 20.0, {"cashier_in_workplace": 0.0})]), [], [], CAT, cfg).events
    assert not [e for e in _from_rule(evs, "V02_left_cash_after_departure")
                if e.extra["detection"] == "violation_candidate"]
