"""Границы A08 → A09 → A12 на синтетике: что правила обязаны различать и что получает fusion.

Пары: клиент ↔ посторонний, мобильный ↔ стационарный телефон, уход с убранными ↔ с оставленными
ценностями, нет обзора ↔ нет объекта, V08 с клиентом ↔ без клиента. Поля для A12: confidence_raw,
state_after, person_side/n_outsiders/valuables_open, negative, trace. Тесты с настоящими модулями A08/A12
(``importorskip``) выполняются только там, где эти модули есть (интеграция, локальный L2).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from ml.rules import evaluate
from ml.rules.config import load_rules_config
from ml.rules.synth import CLEAN_BASE, make_states, make_tracks, make_zones, verified_catalog

try:
    from contracts.sentinel_contracts import SceneState  # noqa
except ImportError:  # pragma: no cover
    from contracts.sentinel_contracts import SceneState  # noqa

TYPES = ["V01", "V02", "V03", "V04", "V05", "V06", "V07", "V08", "V09"]
CAT = verified_catalog(TYPES)
AWAY = {"cashier_present": 0.0, "cashier_in_workplace": 0.0}
AWAY_H1 = {"cashier_in_workplace": 0.0}


def _a13_like_catalog() -> dict:
    """Как в каталоге A13 сейчас: нормы не подтверждены (legal_status disabled, rules: [])."""
    return {"version": 1, "types": {t: {"name": f"Тип {t}", "severity": "high", "legal_status": "disabled",
                                        "rules": []} for t in TYPES}}


def _events(states, tracks=(), zones=(), catalog=CAT, hyps=()) -> list:
    cfg = load_rules_config(hypotheses=list(hyps))
    return evaluate(states, tracks, zones, catalog, cfg).events


def _of(evs: list, type_id: str) -> list:
    return [e for e in evs if e.type_id == type_id]


# ----------------------------------------------------------------------------- клиент ↔ посторонний

def test_client_opposite_vs_outsider_on_cashier_side() -> None:
    client = make_states(60.0, segments=[(10.0, 40.0, {"n_persons": 2.0, "outsider_in_cashier_area": 0.0})])
    outsider = make_states(60.0, segments=[(20.0, 30.0, {"n_persons": 2.0, "outsider_in_cashier_area": 1.0})])
    assert not _of(_events(client), "V01")
    v01 = _of(_events(outsider), "V01")
    assert v01 and v01[0].extra["detection"] == "violation_candidate"
    assert v01[0].extra["person_side"] == "cashier" and v01[0].extra["n_outsiders"] == 1


def test_client_vs_outsider_from_tracks_and_zones() -> None:
    """Сигналов людей от A08 нет — роли и сторону дают треки и зоны A06 (без пиксельных констант в правилах)."""
    omit = ("outsider_in_cashier_area", "n_persons", "collector_present", "cashier_present", "cashier_in_workplace")
    cashier = (1, "person", "cashier", 0.0, 60.0, (0.3, 0.55, 0.45, 0.7))
    at_window = (7, "person", "visitor", 20.0, 30.0, (0.5, 0.1, 0.7, 0.3))       # client_area
    behind_desk = (7, "person", "visitor", 20.0, 30.0, (0.4, 0.65, 0.5, 0.85))   # cashier_area
    states = make_states(60.0, omit=omit)
    rep_client = evaluate(states, make_tracks(60.0, spans=[cashier, at_window]), make_zones(), CAT,
                          load_rules_config())
    assert not _of(rep_client.events, "V01")
    from ml.rules.signals import derive_from_tracks

    derived = derive_from_tracks(rep_client.series.times, make_tracks(60.0, spans=[cashier, at_window]), make_zones())
    assert derived["client_present"][int(25.0 * 5)] > 0.5            # клиент у окна виден
    assert derived["client_present"][int(40.0 * 5)] != derived["client_present"][int(40.0 * 5)]  # NaN: не «нет»
    assert derived["outsider_in_cashier_area"][int(25.0 * 5)] == 0.0
    rep_out = evaluate(states, make_tracks(60.0, spans=[cashier, behind_desk]), make_zones(), CAT,
                       load_rules_config())
    v01 = _of(rep_out.events, "V01")
    assert v01 and v01[0].extra["trace"]["derived_signals"].get("outsider_in_cashier_area") == "tracks"


def test_unknown_role_on_cashier_side_is_not_outsider() -> None:
    """A08: роль неясна → outsider неизвестен + observed.unknown_person_in_cashier_area. Кандидата нет."""
    segs = [(20.0, 30.0, {"outsider_in_cashier_area": None, "unknown_person_in_cashier_area": 1.0,
                          "n_persons": 2.0})]
    assert not _of(_events(make_states(60.0, segments=segs)), "V01")
    obs = _of(_events(make_states(60.0, segments=segs), hyps=["H3_unknown_role_observation"]), "V01")
    assert obs and all(e.extra["detection"] == "observation" for e in obs)


# ----------------------------------------------------------------------------- мобильный ↔ стационарный

@pytest.mark.parametrize("hyps,detection", [((), "observation"), (("H6_phone_candidate",), "violation_candidate")])
def test_mobile_vs_landline(hyps: tuple, detection: str) -> None:
    mobile = make_states(60.0, segments=[(20.0, 26.0, {"phone_in_hand": 1.0})])
    landline = make_states(60.0, segments=[(20.0, 26.0, {"phone_to_ear": 0.9, "phone_in_hand": 0.0})])
    v05 = _of(_events(mobile, hyps=hyps), "V05")
    assert v05 and v05[0].extra["detection"] == detection
    assert not _of(_events(landline, hyps=hyps), "V05")   # трубка у уха без мобильного — не сигнал


# ----------------------------------------------------------------------------- уход: убрано ↔ оставлено

@pytest.mark.parametrize("hyps,away", [((), AWAY), (("H1_left_after_departure",), AWAY_H1)])
def test_departure_valuables_removed_vs_left(hyps: tuple, away: dict) -> None:
    left_cash = make_states(60.0, segments=[(20.0, 35.0, away)])
    left_safe = make_states(60.0, segments=[(10.0, 35.0, {"safe_open": 1.0}),
                                            (20.0, 35.0, {**away, "cash_visible": 0.0})])
    removed = make_states(60.0, segments=[(20.0, 35.0, {**away, "cash_visible": 0.0, "cash_bag_visible": 0.0,
                                                        "safe_open": 0.0})])
    v02 = _of(_events(left_cash, hyps=hyps), "V02")
    assert v02 and v02[0].extra["detection"] == "violation_candidate"
    sa = v02[0].extra["state_after"]
    assert sa["cash_visible"] >= 0.5 and sa["t_after"] >= sa["t_departure"] + 1.0
    assert abs(sa["t_departure"] - 20.0) <= 0.25
    v03 = _of(_events(left_safe, hyps=hyps), "V03")
    assert v03 and v03[0].extra["state_after"]["safe_open"] >= 0.5
    assert _events(removed, hyps=hyps) == []


# ----------------------------------------------------------------------------- нет обзора ↔ нет объекта

@pytest.mark.parametrize("hyps,away", [((), AWAY), (("H1_left_after_departure",), AWAY_H1)])
def test_no_view_vs_object_absent(hyps: tuple, away: dict) -> None:
    absent = make_states(60.0, segments=[(20.0, 35.0, {**away, "cash_visible": 0.0, "cash_bag_visible": 0.0})])
    no_view = make_states(60.0, segments=[(20.0, 35.0, {**away, "cash_visible": None, "cash_bag_visible": None})])
    assert not _of(_events(absent, hyps=hyps), "V02")
    evs = _of(_events(no_view, hyps=hyps), "V02")
    assert evs, "стол не виден после ухода — нужно insufficient_context, а не молчание"
    assert all(e.extra["detection"] == "insufficient_context" for e in evs)
    assert all(e.extra["state_after"]["cash_visible"] is None for e in evs if "state_after" in e.extra)


@dataclass
class _StateEx(SceneState):
    """Форма SceneStateEx A08 (known/conf) без импорта чужого модуля."""

    conf: dict = field(default_factory=dict)
    known: dict = field(default_factory=dict)


def test_no_view_with_scene_state_ex_priors() -> None:
    """A08 кладёт prior в signals и known=False — это «не видно», а не «стол пуст» и не «наличные есть»."""
    states = []
    for i in range(300):
        t = i / 5
        sig, known = dict(CLEAN_BASE), {k: True for k in CLEAN_BASE}
        if 20 <= t < 35:
            sig.update(AWAY)
            sig["cash_visible"], sig["cash_bag_visible"] = 0.1, 0.15      # priors
            known["cash_visible"] = known["cash_bag_visible"] = False
        states.append(_StateEx(t=t, signals=sig, known=known))
    evs = _of(_events(states), "V02")
    assert evs and all(e.extra["detection"] == "insufficient_context" for e in evs)


# ----------------------------------------------------------------------------- V08: присутствие клиента

V08 = ("H10_counting_without_client",)


def _v08(segments: list, **kw) -> list:
    return _of(_events(make_states(60.0, segments=segments, **kw), hyps=V08), "V08")


def test_v08_counting_after_client_left_vs_with_client() -> None:
    left = [(0.0, 20.0, {"client_present": 1.0}), (20.0, 60.0, {"client_present": 0.0}),
            (20.0, 30.0, {"cash_counting": 1.0})]
    with_client = [(0.0, 60.0, {"client_present": 1.0}), (20.0, 30.0, {"cash_counting": 1.0})]
    evs = _v08(left)
    assert evs and evs[0].extra["detection"] == "violation_candidate"
    assert abs(evs[0].t_peak - 20.0) <= 0.25
    assert not _v08(with_client)


def test_v08_internal_counting_without_any_client_is_not_v08() -> None:
    """Клиента не было вовсе (доказано) — внутренний пересчёт, исключение A13."""
    assert not _v08([(0.0, 60.0, {"client_present": 0.0}), (20.0, 30.0, {"cash_counting": 1.0})])


def test_v08_client_presence_unknown_is_insufficient() -> None:
    evs = _v08([(0.0, 20.0, {"client_present": 1.0}), (20.0, 60.0, {"client_present": None}),
                (20.0, 30.0, {"cash_counting": 1.0})])
    assert evs and all(e.extra["detection"] == "insufficient_context" for e in evs)


def test_v08_ignores_posture_and_needs_counting_signal() -> None:
    """Спиной к камере / руки перекрыты без клиента — не V08; без сигнала пересчёта правило молчит."""
    posture = [(0.0, 20.0, {"client_present": 1.0}), (20.0, 60.0, {"client_present": 0.0}),
               (20.0, 30.0, {"hands_hidden": 1.0, "cashier_facing_away": 1.0})]
    assert not _v08(posture)
    rep = evaluate(make_states(60.0, segments=posture), [], [], CAT, load_rules_config(hypotheses=list(V08)))
    assert "cash_counting" in rep.series.absent


def test_v08_client_from_tracks_cannot_prove_absence() -> None:
    """Производный client_present односторонний: ушёл из кадра ≠ доказанно нет → только insufficient."""
    cashier = (1, "person", "cashier", 0.0, 60.0, (0.3, 0.55, 0.45, 0.7))
    client = (7, "person", "visitor", 0.0, 20.0, (0.5, 0.1, 0.7, 0.3))
    states = make_states(60.0, segments=[(20.0, 30.0, {"cash_counting": 1.0})])
    evs = _of(evaluate(states, make_tracks(60.0, spans=[cashier, client]), make_zones(), CAT,
                       load_rules_config(hypotheses=list(V08))).events, "V08")
    assert all(e.extra["detection"] != "violation_candidate" for e in evs)


# ----------------------------------------------------------------------------- поля для A12

def _a12_reconstruct(conf: float, extra: dict) -> float:
    """Формула восстановления A12 (ml/fusion/core.py::_a09_uncapped) по trace.confidence_terms."""
    terms = extra["trace"]["confidence_terms"]
    raw = terms["base"] + terms["strength"] + terms["duration"]
    value = min(terms.get("cap", 1.0), raw * terms.get("factor", 1.0)) * terms.get("derived_factor", 1.0)
    return max(conf, value)


def test_confidence_raw_is_uncapped_and_matches_a12_formula() -> None:
    states = make_states(60.0, segments=[(10.0, 30.0, {"safe_open": 1.0}), (15.0, 30.0, AWAY)])
    evs = _of(_events(states, catalog=_a13_like_catalog()), "V03")
    assert evs
    ev = evs[0]
    assert ev.extra["assessment"] == "review_observation" and ev.confidence <= 0.45
    assert any("норм" in r for r in ev.extra["assessment_reasons"])     # A12 → norm_review
    assert ev.extra["confidence_raw"] >= ev.confidence
    assert ev.extra["confidence_raw"] == pytest.approx(_a12_reconstruct(ev.confidence, ev.extra), abs=1e-3)
    assert ev.extra["detection"] == "violation_candidate" and ev.extra["legal_basis"] == "none"


def test_observation_and_insufficient_keep_their_meaning() -> None:
    phone = _of(_events(make_states(60.0, segments=[(20.0, 26.0, {"phone_in_hand": 1.0})]),
                        catalog=_a13_like_catalog()), "V05")
    assert phone and any("наблюдени" in r for r in phone[0].extra["assessment_reasons"])
    no_view = _of(_events(make_states(60.0, segments=[(20.0, 35.0, {**AWAY, "cash_visible": None})]),
                          catalog=_a13_like_catalog()), "V02")
    assert no_view and no_view[0].extra["assessment"] == "insufficient_context"
    assert no_view[0].confidence <= 0.3


def test_negative_only_from_proven_exclusions() -> None:
    segs = [(20.0, 45.0, {"safe_open": 1.0}), (30.0, 31.0, {"collector_present": 1.0, "n_persons": 2.0})]
    evs = _of(_events(make_states(60.0, segments=segs), hyps=["H2_work_with_safe_open"]), "V03")
    assert evs and evs[0].extra.get("negative")
    neg = evs[0].extra["negative"][0]
    assert "инкассац" in neg["fact"] and 0.0 < neg["strength"] < 0.2
    # инкассатор неизвестен (а не «нет») — отрицательного свидетельства нет
    segs_unknown = [(20.0, 45.0, {"safe_open": 1.0, "collector_present": None})]
    evs = _of(_events(make_states(60.0, segments=segs_unknown), hyps=["H2_work_with_safe_open"]), "V03")
    assert evs and "negative" not in evs[0].extra


def test_valuables_open_unknown_vs_closed() -> None:
    seg = {"n_persons": 2.0, "outsider_in_cashier_area": 1.0}
    unknown = _of(_events(make_states(60.0, segments=[(20.0, 30.0, seg),
                                                      (15.0, 30.0, {"safe_open": None, "cash_bag_visible": None})])),
                  "V01")
    closed = _of(_events(make_states(60.0, segments=[(20.0, 30.0, seg)])), "V01")
    assert unknown[0].extra["valuables_open"] is None
    assert closed[0].extra["valuables_open"] == 0.0


# ----------------------------------------------------------------------------- настоящие модули A12 / A08

def test_a12_upstream_assessment_preserves_meaning() -> None:
    core = pytest.importorskip("ml.fusion.core")
    a13 = _a13_like_catalog()
    norm_ev = _of(_events(make_states(60.0, segments=[(10.0, 30.0, {"safe_open": 1.0}), (15.0, 30.0, AWAY)]),
                          catalog=a13), "V03")[0]
    a, conf = core.upstream_assessment(norm_ev.confidence, norm_ev.extra, {})
    assert a == "norm_review" and conf == pytest.approx(norm_ev.extra["confidence_raw"], abs=1e-6)
    obs_ev = _of(_events(make_states(60.0, segments=[(20.0, 26.0, {"phone_in_hand": 1.0})]), catalog=a13), "V05")[0]
    assert core.upstream_assessment(obs_ev.confidence, obs_ev.extra, {}) == ("observation", obs_ev.confidence)
    ins_ev = _of(_events(make_states(60.0, segments=[(20.0, 35.0, {**AWAY, "cash_visible": None})]),
                         catalog=a13), "V02")[0]
    assert core.upstream_assessment(ins_ev.confidence, ins_ev.extra, {}) == ("insufficient", ins_ev.confidence)


def test_a12_policies_read_state_after_side_and_negative() -> None:
    pol = pytest.importorskip("ml.fusion.policies")
    left = _of(_events(make_states(60.0, segments=[(20.0, 35.0, AWAY)])), "V02")[0]
    assert pol._state_after([left])["cash"] >= 0.5                      # «ценности остались»
    no_view = _of(_events(make_states(60.0, segments=[(20.0, 35.0, {**AWAY, "cash_visible": None,
                                                                    "cash_bag_visible": None})])), "V02")[0]
    st = pol._state_after([no_view])
    assert st["cash"] is None and st["bag"] is None                     # неизвестно, а не «пусто»
    v01 = _of(_events(make_states(60.0, segments=[(20.0, 30.0, {"n_persons": 2.0,
                                                                "outsider_in_cashier_area": 1.0})])), "V01")[0]
    assert pol._side([v01]) == "cashier"                                # клиентская подсказка не гасит V01
    segs = [(20.0, 45.0, {"safe_open": 1.0}), (30.0, 31.0, {"collector_present": 1.0, "n_persons": 2.0})]
    neg_ev = _of(_events(make_states(60.0, segments=segs), hyps=["H2_work_with_safe_open"]), "V03")[0]
    strength, facts = pol._negative_strength([neg_ev])
    assert 0.0 < strength < 0.2


def test_a08_scene_state_ex_roundtrip_keeps_unknown() -> None:
    types = pytest.importorskip("ml.scene.types")
    from ml.rules.signals import build_series

    st = types.SceneStateEx(t=0.0, signals={"cashier_in_workplace": 0.5, "safe_open": 1.0},
                            conf={"cashier_in_workplace": 0.0, "safe_open": 0.9},
                            known={"cashier_in_workplace": False, "safe_open": True})
    for state in (st, types.state_to_dict(st), types.state_from_dict(types.state_to_dict(st))):
        s = build_series([state], ["cashier_in_workplace", "safe_open"])
        assert "cashier_in_workplace" in s.absent and s.signals["safe_open"][0] == 1.0
