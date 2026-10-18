"""Негативные сценарии: неизвестные сигналы, неполный этап, непроверенные нормы.

Принципы: отсутствие сигнала ≠ отрицание; «по бездействию» — только при наблюдаемом
полном этапе; candidate_violation — только при проверенной норме A13 и правиле вида violation.
"""
from __future__ import annotations

import pytest

from ml.rules import CANDIDATE, INSUFFICIENT, REVIEW, evaluate, run_rules, split_by_assessment
from ml.rules.catalog import FALLBACK_CATALOG, norm_status
from ml.rules.synth import RULE_CASES, make_states, unverified_catalog, verified_catalog

CAT = verified_catalog()
LEFT_OPEN = [(10.0, 30.0, {"safe_open": 1.0}), (15.0, 30.0, {"cashier_present": 0.0, "cashier_in_workplace": 0.0})]


def _v03(evs: list) -> list:
    return [e for e in evs if e.type_id == "V03"]


# ----------------------------------------------------------------------------- unknown

def test_unknown_signal_entirely_absent_gives_no_event() -> None:
    """Кассирский сигнал не пришёл вовсе: «кассира нет» НЕ выводится."""
    states = make_states(60.0, segments=[(10.0, 30.0, {"safe_open": 1.0})],
                         omit=("cashier_present", "cashier_in_workplace", "workplace_unattended_with_cash"))
    rep = evaluate(states, [], [], CAT)
    assert not _v03(rep.events)
    assert not [e for e in rep.events if e.type_id == "V02"]
    assert "cashier_present" in rep.series.absent
    assert any("cashier_present" in w for w in rep.warnings)


def test_all_signals_absent_gives_no_events() -> None:
    states = make_states(60.0, base={})
    assert run_rules(states, [], [], CAT) == []


def test_empty_states() -> None:
    rep = evaluate([], [], [], CAT)
    assert rep.events == [] and rep.warnings


def test_occlusion_gives_insufficient_context_not_violation() -> None:
    """Сейф доказанно открыт, кассир неизвестен (окклюзия 15–30 с) → insufficient_context."""
    states = make_states(60.0, segments=[(10.0, 30.0, {"safe_open": 1.0}),
                                         (15.0, 30.0, {"cashier_present": None, "cashier_in_workplace": None})])
    evs = _v03(run_rules(states, [], [], CAT))
    assert evs, "ожидалось событие insufficient_context"
    assert all(e.extra["assessment"] == INSUFFICIENT for e in evs)
    assert all(e.confidence <= 0.3 for e in evs)
    assert any("Недостаточно контекста" in f for e in evs for f in e.facts)


def test_unknown_outsider_signal_does_not_trigger_v01() -> None:
    states = make_states(60.0, segments=[(20.0, 30.0, {"n_persons": 3.0})], omit=("outsider_in_cashier_area",))
    assert not [e for e in run_rules(states, [], [], CAT) if e.type_id == "V01"]


def test_unknown_gate_context_blocks_observation() -> None:
    """Телефон в руке, но контекст работы (рабочее место/наличные) неизвестен → без события."""
    states = make_states(60.0, segments=[(20.0, 30.0, {"phone_in_hand": 1.0})],
                         omit=("cashier_in_workplace", "cash_visible"))
    assert not [e for e in run_rules(states, [], [], CAT) if e.type_id == "V05"]


def test_short_gaps_are_forward_filled_long_gaps_are_unknown() -> None:
    # пропуски по 0.6 с (< ffill_max_s=1.0) не ломают доказанный сегмент
    segs = list(LEFT_OPEN) + [(t, t + 0.6, {"cashier_present": None}) for t in (17.0, 20.0, 23.0, 26.0)]
    evs = _v03(run_rules(make_states(60.0, segments=segs), [], [], CAT))
    assert evs and evs[0].extra["assessment"] == CANDIDATE


# ----------------------------------------------------------------------------- incomplete

def test_inaction_starting_at_clip_start_is_insufficient() -> None:
    """Ролик начинается с открытого сейфа без кассира — момент ухода не наблюдался."""
    states = make_states(30.0, segments=[(0.0, 20.0, {"safe_open": 1.0, "cashier_present": 0.0,
                                                      "cashier_in_workplace": 0.0})])
    evs = _v03(run_rules(states, [], [], CAT))
    assert evs and all(e.extra["assessment"] == INSUFFICIENT for e in evs)
    assert any("начало ситуации не попало в ролик" in r for e in evs for r in e.extra["assessment_reasons"])


def test_inaction_truncated_at_clip_end_is_insufficient() -> None:
    """Кассир ушёл за 3 с до конца ролика: этап «≥ 5 с без присмотра» не завершён."""
    states = make_states(20.0, segments=[(10.0, 20.0, {"safe_open": 1.0}),
                                         (17.0, 20.0, {"cashier_present": 0.0, "cashier_in_workplace": 0.0})])
    evs = _v03(run_rules(states, [], [], CAT))
    assert evs and all(e.extra["assessment"] == INSUFFICIENT for e in evs)
    assert any("ролик заканчивается" in r for e in evs for r in e.extra["assessment_reasons"])


def test_inaction_without_observed_before_state_is_insufficient() -> None:
    """Сейф открылся, когда кассира уже не было: «кассир у открытого сейфа» не наблюдалось."""
    states = make_states(60.0, segments=[(5.0, 30.0, {"cashier_present": 0.0, "cashier_in_workplace": 0.0}),
                                         (10.0, 30.0, {"safe_open": 1.0})])
    evs = [e for e in _v03(run_rules(states, [], [], CAT)) if "V03_safe_left_open" in e.extra["rule_ids"]]
    assert evs and all(e.extra["assessment"] == INSUFFICIENT for e in evs)
    assert any("предшествующее состояние" in r for e in evs for r in e.extra["assessment_reasons"])


def test_inaction_low_coverage_is_insufficient() -> None:
    """Кассир неизвестен в половине кадров сегмента (пропуски > ffill) → insufficient_context."""
    # пропуски по 2.6 с каждые 3 с: forward-fill закрывает 1 с, остальное — неизвестно (~50% кадров)
    segs = list(LEFT_OPEN) + [(t, t + 2.6, {"cashier_present": None}) for t in (16.0, 19.0, 22.0, 25.0)]
    evs = [e for e in _v03(run_rules(make_states(60.0, segments=segs), [], [], CAT))
           if "V03_safe_left_open" in e.extra["rule_ids"]]
    assert evs and all(e.extra["assessment"] == INSUFFICIENT for e in evs)
    assert any("известны лишь" in r for e in evs for r in e.extra["assessment_reasons"])


def test_bag_left_without_observed_collector_is_insufficient() -> None:
    states = make_states(60.0, segments=[(10.0, 40.0, {"cash_bag_visible": 1.0}),
                                         (22.0, 40.0, {"cashier_present": 0.0, "cashier_in_workplace": 0.0})])
    evs = [e for e in run_rules(states, [], [], CAT) if e.type_id == "V07"]
    assert evs and all(e.extra["assessment"] == INSUFFICIENT for e in evs)


# ----------------------------------------------------------------------------- unverified

@pytest.mark.parametrize("rule_id", [r for r, c in RULE_CASES.items() if c["assessment"] == CANDIDATE])
def test_unverified_norm_gives_review_observation(rule_id: str) -> None:
    case = RULE_CASES[rule_id]
    states = make_states(case["pos"]["duration_s"], segments=case["pos"]["segments"])
    evs = [e for e in run_rules(states, [], [], unverified_catalog()) if e.type_id == case["type_id"]]
    assert evs and all(e.extra["assessment"] == REVIEW for e in evs)
    assert all(e.confidence <= 0.45 for e in evs)
    assert all(e.extra["legal_claim"] is False for e in evs)


def test_fallback_catalog_never_yields_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    import ml.rules.catalog as catalog_mod

    monkeypatch.setattr(catalog_mod, "_try_norms_catalog", lambda: None)
    states = make_states(60.0, segments=LEFT_OPEN)
    rep = evaluate(states, [], [], {})
    assert rep.catalog_source == "fallback"
    groups = split_by_assessment(rep.events)
    assert rep.events and not groups[CANDIDATE]
    for tdef in FALLBACK_CATALOG["types"].values():
        assert "rules" not in tdef  # в fallback нет норм — и не должно быть


def test_observation_rule_stays_review_even_with_verified_norm() -> None:
    case = RULE_CASES["V05_phone_in_hand_at_workplace"]
    states = make_states(60.0, segments=case["pos"]["segments"])
    evs = [e for e in run_rules(states, [], [], CAT) if e.type_id == "V05"]
    assert evs and all(e.extra["assessment"] == REVIEW for e in evs)


def test_norm_status_requires_verified_flag_and_url() -> None:
    cat = verified_catalog(["V03"])
    assert norm_status(cat, "V03")["status"] == "verified"
    cat["types"]["V03"]["rules"][0]["url"] = ""
    assert norm_status(cat, "V03")["status"] == "unverified"
    cat["types"]["V03"]["rules"][0]["verified"] = "yes"  # не bool True
    assert norm_status(cat, "V03")["status"] == "unverified"
    assert norm_status(cat, "V01")["status"] == "absent"
    assert norm_status({"types": {"V01": {"name": "x"}}}, "V01")["status"] == "absent"


def test_disable_review_and_insufficient_emission() -> None:
    from ml.rules.config import apply_overrides, parse_rules_config, read_raw_config

    raw, src = read_raw_config()
    cfg = parse_rules_config(apply_overrides(raw, ["global.emit_review_observations=false",
                                                   "global.emit_insufficient_context=false"]), src)
    states = make_states(30.0, segments=[(0.0, 20.0, {"safe_open": 1.0, "cashier_present": 0.0}),
                                         (5.0, 15.0, {"phone_in_hand": 1.0})])
    assert evaluate(states, [], [], CAT, cfg).events == []


# ----------------------------------------------------------------------------- facts honesty

def test_absent_or_branch_does_not_spoil_coverage() -> None:
    """workplace_unattended_with_cash не приходит, но доказана другая ветвь OR → кандидат."""
    case = RULE_CASES["V02_cash_left_unattended"]
    states = make_states(60.0, segments=case["pos"]["segments"], omit=("workplace_unattended_with_cash",))
    evs = [e for e in run_rules(states, [], [], CAT) if e.type_id == "V02"]
    assert evs and evs[0].extra["assessment"] == CANDIDATE


def test_insufficient_event_does_not_claim_unobserved_stage() -> None:
    states = make_states(60.0, segments=[(5.0, 30.0, {"cashier_present": 0.0, "cashier_in_workplace": 0.0}),
                                         (10.0, 30.0, {"safe_open": 1.0})])
    evs = [e for e in _v03(run_rules(states, [], [], CAT)) if e.extra["assessment"] == INSUFFICIENT]
    assert evs
    assert not any("До этого" in f for e in evs for f in e.facts)


def test_occlusion_facts_state_only_the_anchor() -> None:
    states = make_states(60.0, segments=[(10.0, 30.0, {"safe_open": 1.0}),
                                         (15.0, 30.0, {"cashier_present": None, "cashier_in_workplace": None})])
    evs = [e for e in _v03(run_rules(states, [], [], CAT)) if e.extra["trace"].get("from_anchor")]
    assert evs
    facts = " ".join(evs[0].facts)
    assert "наблюдается: сейф открыт" in facts
    assert "кассир в кадре не наблюдается" not in facts


def test_fact_with_unknown_value_is_dropped() -> None:
    case = RULE_CASES["V01_outsider_in_cashier_area"]
    states = make_states(60.0, segments=[(20.0, 26.0, {"outsider_in_cashier_area": 1.0})], omit=("n_persons",))
    evs = [e for e in run_rules(states, [], [], CAT) if e.type_id == "V01"]
    assert evs and len(evs[0].facts) == 1 and "?" not in evs[0].facts[0]
    assert case["type_id"] == "V01"


def test_occlusion_with_absent_or_branch_still_insufficient() -> None:
    """A08 не шлёт workplace_unattended_with_cash; кассир пропал из сигналов (окклюзия) при наличных."""
    states = make_states(60.0, segments=[(30.0, 42.0, {"cashier_present": None, "cashier_in_workplace": None})],
                         omit=("workplace_unattended_with_cash",))
    evs = [e for e in run_rules(states, [], [], CAT) if e.type_id == "V02"]
    assert evs and all(e.extra["assessment"] == INSUFFICIENT for e in evs)
    assert all(e.t_start >= 29.0 for e in evs)  # не по всему ролику — только где кассир неизвестен
