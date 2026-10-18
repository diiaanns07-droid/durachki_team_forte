"""Совместимость входа (SceneStateEx A08: known/conf/observed/quality), оси detection/legal_basis, покрытие."""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from ml.rules import evaluate, run_rules
from ml.rules.catalog import legal_basis_of, norm_status
from ml.rules.signals import build_series
from ml.rules.synth import CLEAN_BASE, RULE_CASES, make_states, verified_catalog

try:
    from contracts.sentinel_contracts import SceneState  # noqa
except ImportError:  # pragma: no cover
    from contracts.sentinel_contracts import SceneState  # noqa

CAT = verified_catalog()


@dataclass
class _StateEx(SceneState):
    """Минимальная копия формы SceneStateEx (A08) для тестов — без импорта чужого модуля."""

    conf: dict = field(default_factory=dict)
    known: dict = field(default_factory=dict)
    observed: dict = field(default_factory=dict)
    quality: float = 1.0


def test_known_false_means_unknown_even_with_prior_value() -> None:
    # A08: неизвестное в signals = posterior → prior (0.5 у cashier_in_workplace), known=False
    st = [_StateEx(t=0.0, signals={"cashier_in_workplace": 0.5, "safe_open": 1.0},
                   known={"cashier_in_workplace": False, "safe_open": True})]
    s = build_series(st, ["cashier_in_workplace", "safe_open"])
    assert "cashier_in_workplace" in s.absent and s.signals["safe_open"][0] == 1.0


def test_conf_threshold_and_missing_conf_is_unknown() -> None:
    st = [{"t": 0.0, "signals": {"a": 1.0, "b": 1.0, "c": 1.0}, "conf": {"a": 0.9, "b": 0.2}}]
    s = build_series(st, ["a", "b", "c"], known_min_conf=0.5)
    assert s.signals["a"][0] == 1.0 and "b" in s.absent and "c" in s.absent


def test_observed_one_sided_and_quality() -> None:
    st = [{"t": 0.0, "signals": {}, "known": {}, "observed": {"workplace_empty": 1.0, "safe_unattended": 0.0,
                                                               "workplace_empty_conf": 0.3}, "quality": 0.8}]
    s = build_series(st, ["workplace_empty", "safe_unattended"])
    assert s.signals["workplace_empty"][0] == 1.0
    assert "safe_unattended" in s.absent           # 0 в observed = «не наблюдали», не «нет»
    assert s.signals["workplace_empty_conf"][0] == pytest.approx(0.3)
    assert s.signals["frame_quality"][0] == pytest.approx(0.8)


def test_extended_states_end_to_end_unknown_cashier_does_not_fire() -> None:
    """Сейф открыт, кассир «неизвестен» (prior 0.5, known=False) → никакого кандидата V03."""
    states = []
    for i in range(300):
        t = i / 5
        sig = dict(CLEAN_BASE)
        known = {k: True for k in sig}
        if 10 <= t < 40:
            sig["safe_open"] = 1.0
        if 15 <= t < 40:
            sig["cashier_present"], sig["cashier_in_workplace"] = 0.6, 0.5   # priors
            known["cashier_present"] = known["cashier_in_workplace"] = False
        states.append(_StateEx(t=t, signals=sig, known=known, conf={k: (1.0 if v else 0.0) for k, v in known.items()}))
    evs = [e for e in run_rules(states, [], [], CAT) if e.type_id == "V03"]
    assert all(e.extra["detection"] != "violation_candidate" for e in evs)
    assert all(e.extra["assessment"] != "candidate_violation" for e in evs)


def test_legal_status_from_a13_catalog() -> None:
    cat = verified_catalog(["V01", "V02", "V03"])
    cat["types"]["V01"]["legal_status"] = "disabled"
    cat["types"]["V02"]["legal_status"] = "candidate"
    cat["types"]["V03"]["legal_status"] = "confirmed"
    assert norm_status(cat, "V01")["status"] == "disabled"
    assert norm_status(cat, "V02")["status"] == "candidate"
    assert norm_status(cat, "V03")["status"] == "verified"
    assert legal_basis_of("verified") == "confirmed" and legal_basis_of("candidate") == "candidate"
    assert legal_basis_of("disabled") == "none" and legal_basis_of("absent") == "none"


def test_detection_axis_is_independent_of_norms() -> None:
    """Без подтверждённой нормы: detection=violation_candidate (поведение), assessment=review (право)."""
    case = RULE_CASES["V03_safe_left_open"]
    states = make_states(60.0, segments=case["pos"]["segments"])
    cat = verified_catalog(["V03"])
    cat["types"]["V03"]["legal_status"] = "disabled"
    cat["types"]["V03"]["rules"] = []
    ev = next(e for e in run_rules(states, [], [], cat) if e.type_id == "V03")
    assert ev.extra["detection"] == "violation_candidate"
    assert ev.extra["assessment"] == "review_observation" and ev.extra["legal_basis"] == "none"
    phone = RULE_CASES["V05_phone_in_hand_at_workplace"]
    ev2 = next(e for e in run_rules(make_states(60.0, segments=phone["pos"]["segments"]), [], [], CAT)
               if e.type_id == "V05")
    assert ev2.extra["detection"] == "observation"


def test_coverage_report() -> None:
    states = make_states(60.0, segments=[(20.0, 40.0, {"cashier_present": None})],
                         omit=("outsider_in_cashier_area",))
    cov = evaluate(states, [], [], CAT).coverage()
    assert cov["n_frames"] == 300
    assert cov["signals"]["safe_open"] == 1.0
    assert 0.6 < cov["signals"]["cashier_present"] < 0.75
    assert cov["rules"]["V01_outsider_in_cashier_area"] == 0.0
    assert "V01_outsider_in_cashier_area" in cov["not_evaluable_rules"]
