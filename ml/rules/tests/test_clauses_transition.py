"""Структура правила (применимость/предикаты/исключения) и t_peak по наблюдаемому переходу."""
from __future__ import annotations

import pytest

from ml.rules import evaluate
from ml.rules.config import ConfigError, apply_overrides, parse_rules_config, read_raw_config
from ml.rules.synth import make_states, verified_catalog

CAT = verified_catalog()

RULE = {
    "id": "T_left_cash", "type_id": "V02", "kind": "violation", "enabled": True, "status_reason": "тест",
    "comment": "тестовое правило",
    "applicability": {"id": "cash_known", "label": "состояние стола наблюдается", "expr": "cash_visible >= 0"},
    "predicates": [
        {"id": "away", "label": "кассир вне рабочего места", "expr": "cashier_in_workplace < 0.3"},
        {"id": "cash", "label": "на столе наличные", "expr": "cash_visible > 0.5"},
    ],
    "exclusions": [{"id": "collection", "label": "идёт инкассация", "expr": "collector_present > 0.5"}],
    "min_duration_s": 3.0,
    "t_peak": {"policy": "transition", "trigger": "cashier_in_workplace < 0.3", "lookback_s": 5.0},
    "facts": ["С {t_start:.1f} по {t_end:.1f} с кассир вне рабочего места, на столе наличные."],
}


def _cfg(rule: dict, *sets: str):
    raw, src = read_raw_config()
    raw = dict(raw)
    raw["rules"] = [rule]
    return parse_rules_config(apply_overrides(raw, list(sets)), src)


def _ev(cfg, segments, **kw):
    evs = evaluate(make_states(60.0, segments=segments, **kw), [], [], CAT, cfg).events
    return [e for e in evs if e.extra["rule_id"] == "T_left_cash"]


def test_predicates_compile_to_and_and_trace_each_clause() -> None:
    cfg = _cfg(RULE)
    evs = _ev(cfg, [(20.0, 30.0, {"cashier_in_workplace": 0.0})])
    assert len(evs) == 1
    clauses = {c["id"]: c for c in evs[0].extra["trace"]["clauses"]}
    assert clauses["away"]["at_peak"] == "true" and clauses["cash"]["at_peak"] == "true"
    assert clauses["collection"]["kind"] == "exclusion" and clauses["collection"]["at_peak"] == "false"
    assert clauses["cash_known"]["kind"] == "applicability"
    ev = evs[0].extra["trace"]["evidence"]
    assert ev["policy"] == "transition" and ev["kind"] == "false→true" and ev["t_peak"] == pytest.approx(20.0)


def test_clean_twin_empty_desk_gives_nothing() -> None:
    cfg = _cfg(RULE)
    assert _ev(cfg, [(20.0, 30.0, {"cashier_in_workplace": 0.0, "cash_visible": 0.0})]) == []


def test_exclusion_blocks_only_when_proven() -> None:
    cfg = _cfg(RULE)
    assert _ev(cfg, [(20.0, 30.0, {"cashier_in_workplace": 0.0, "collector_present": 1.0})]) == []
    # инкассатор неизвестен → исключение не доказано → правило работает
    assert _ev(cfg, [(20.0, 30.0, {"cashier_in_workplace": 0.0, "collector_present": None})])


def test_applicability_must_be_proven() -> None:
    cfg = _cfg(RULE)
    assert _ev(cfg, [(20.0, 30.0, {"cashier_in_workplace": 0.0, "cash_visible": None})]) == []


def test_transition_precedes_segment_extends_interval() -> None:
    # кассир ушёл в 20 с, наличные стали видны только в 22 с (были закрыты корпусом → неизвестно)
    cfg = _cfg(RULE)
    evs = _ev(cfg, [(18.0, 22.0, {"cash_visible": None}), (20.0, 30.0, {"cashier_in_workplace": 0.0})])
    assert len(evs) == 1
    assert evs[0].t_peak == pytest.approx(20.0)
    assert evs[0].t_start == pytest.approx(20.0)
    assert evs[0].extra["trace"]["interval"]["segment_t_start"] >= 21.9


def test_transition_from_unknown_is_flagged_and_not_observed_falls_back() -> None:
    cfg = _cfg(RULE)
    evs = _ev(cfg, [(17.0, 20.0, {"cashier_in_workplace": None}), (20.0, 30.0, {"cashier_in_workplace": 0.0})])
    assert evs and evs[0].extra["trace"]["evidence"]["kind"] == "unknown→true"
    # кассира нет с начала ролика → перехода не видно → fallback evidence (первый кадр сегмента)
    evs = _ev(cfg, [(0.0, 30.0, {"cashier_in_workplace": 0.0})])
    assert evs and evs[0].extra["trace"]["evidence"]["kind"] == "not_observed"
    assert evs[0].extra["trace"]["evidence"]["fallback"] == "evidence" and evs[0].t_peak == pytest.approx(0.0)


def test_global_policy_override_for_ab() -> None:
    rule = dict(RULE, t_peak={"policy": "evidence"})
    cfg = _cfg(rule, "global.t_peak_policy_override=transition")
    evs = _ev(cfg, [(20.0, 30.0, {"cashier_in_workplace": 0.0})])
    assert evs[0].extra["trace"]["evidence"]["policy"] == "transition"
    assert evs[0].extra["trace"]["evidence"]["trigger_expr"]  # без trigger — переход по strength
    with pytest.raises(ConfigError):
        _cfg(RULE, "global.t_peak_policy_override=guess")


def test_strength_and_predicates_are_exclusive() -> None:
    with pytest.raises(ConfigError):
        _cfg(dict(RULE, strength="cash_visible"))


def test_transition_ignores_earlier_brief_episode_in_lookback() -> None:
    """Кассир на 1 с отходил в 16 с и вернулся; настоящий уход — в 20 с. t_peak = 20, а не 16."""
    cfg = _cfg(RULE)
    evs = _ev(cfg, [(16.0, 17.0, {"cashier_in_workplace": 0.0}), (20.0, 30.0, {"cashier_in_workplace": 0.0})])
    assert len(evs) == 1 and evs[0].t_peak == pytest.approx(20.0)
    assert evs[0].extra["trace"]["evidence"]["kind"] == "false→true"
