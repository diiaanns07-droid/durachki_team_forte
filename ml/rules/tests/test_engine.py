"""Механика движка: гистерезис, дебаунс, NMS, слияние, t_peak, bboxes, производные сигналы."""
from __future__ import annotations

import numpy as np
import pytest

from ml.rules import CANDIDATE, evaluate, run_rules
from ml.rules.config import apply_overrides, parse_rules_config, read_raw_config
from ml.rules.engine import hysteresis_mask, mask_to_segments, merge_segments
from ml.rules.signals import build_series, point_in_polygon, track_box_at
from ml.rules.synth import make_states, make_tracks, make_zones, verified_catalog

CAT = verified_catalog()


def _cfg(*sets: str):
    raw, src = read_raw_config()
    return parse_rules_config(apply_overrides(raw, list(sets)), src)


def test_hysteresis_on_off() -> None:
    x = np.array([0.0, 0.6, 0.4, 0.4, 0.3, 0.6, 0.2])
    assert hysteresis_mask(x, 0.5, 0.35).tolist() == [False, True, True, True, False, True, False]
    assert hysteresis_mask(x, 0.5, 0.5).tolist() == [False, True, False, False, False, True, False]


def test_segments_and_merge() -> None:
    mask = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1], dtype=bool)
    segs = mask_to_segments(mask)
    assert segs == [(1, 2), (5, 5), (10, 11)]
    t = np.arange(12) * 0.2
    assert merge_segments(t, segs, 0.6) == [(1, 5), (10, 11)]
    assert merge_segments(t, segs, 2.0) == [(1, 11)]


def test_flicker_is_debounced_into_one_event() -> None:
    # посторонний мигает (детектор теряет его на 0.4 с), в сумме 8 с → одно событие
    segs = [(20.0, 30.0, {"outsider_in_cashier_area": 1.0, "n_persons": 2.0})]
    segs += [(t, t + 0.4, {"outsider_in_cashier_area": 0.0}) for t in (22.0, 25.0, 27.0)]
    evs = [e for e in run_rules(make_states(60.0, segments=segs), [], [], CAT) if e.type_id == "V01"]
    assert len(evs) == 1 and evs[0].t_start == pytest.approx(20.0) and evs[0].t_end >= 29.0


def test_nms_same_type_within_window() -> None:
    cfg = _cfg("rules.V01_outsider_in_cashier_area.merge_gap_s=0.5", "global.same_type_merge_gap_s=0")
    segs = [(20.0, 23.0, {"outsider_in_cashier_area": 1.0, "n_persons": 2.0}),
            (24.0, 27.0, {"outsider_in_cashier_area": 1.0, "n_persons": 2.0}),
            (40.0, 43.0, {"outsider_in_cashier_area": 1.0, "n_persons": 2.0})]
    evs = [e for e in evaluate(make_states(60.0, segments=segs), [], [], CAT, cfg).events if e.type_id == "V01"]
    assert len(evs) == 2  # 20 и 24 в окне 5 с → одно; 40 — отдельно
    kept = min(evs, key=lambda e: e.t_peak)
    assert kept.extra.get("suppressed")


def test_max_events_limit() -> None:
    cfg = _cfg("global.max_events=2", "global.nms_window_s=0", "global.same_type_merge_gap_s=0",
               "rules.V01_outsider_in_cashier_area.max_events=10")
    segs = [(float(t), float(t) + 3.0, {"outsider_in_cashier_area": 1.0, "n_persons": 2.0}) for t in range(5, 55, 8)]
    evs = evaluate(make_states(60.0, segments=segs), [], [], CAT, cfg).events
    assert len(evs) == 2


def test_same_type_rules_merge_with_corroboration() -> None:
    # сейф открыт; посторонний в зоне 14–18 с, кассир уходит в 15 с → два правила V03 перекрываются
    segs = [(10.0, 30.0, {"safe_open": 1.0}),
            (14.0, 18.0, {"outsider_in_cashier_area": 1.0, "n_persons": 2.0}),
            (15.0, 30.0, {"cashier_present": 0.0, "cashier_in_workplace": 0.0})]
    evs = [e for e in run_rules(make_states(60.0, segments=segs), [], [], CAT) if e.type_id == "V03"]
    assert len(evs) == 1
    ev = evs[0]
    assert set(ev.extra["rule_ids"]) == {"V03_safe_open_with_outsider", "V03_safe_left_open"}
    assert ev.t_peak == pytest.approx(14.0)  # самое раннее доказанное начало
    assert ev.extra["assessment"] == CANDIDATE and len(ev.extra["merged"]) == 2


def test_t_peak_evidence_is_actual_frame_not_shift() -> None:
    # сила растёт ступенями: 0.6 → 1.0; evidence = первый кадр с почти максимальным доказательством
    segs = [(20.0, 22.0, {"outsider_in_cashier_area": 0.6, "n_persons": 2.0}),
            (22.0, 30.0, {"outsider_in_cashier_area": 1.0, "n_persons": 2.0})]
    states = make_states(60.0, segments=segs)
    ev = next(e for e in run_rules(states, [], [], CAT) if e.type_id == "V01")
    assert ev.t_start == pytest.approx(20.0)
    assert ev.t_peak == pytest.approx(22.0)
    assert ev.t_peak in [round(s.t, 2) for s in states]  # фактический кадр ряда
    assert ev.extra["evidence_t"][0] == pytest.approx(20.0) and 22.0 in ev.extra["evidence_t"]
    cfg = _cfg("rules.V01_outsider_in_cashier_area.t_peak.policy=onset")
    ev2 = next(e for e in evaluate(states, [], [], CAT, cfg).events if e.type_id == "V01")
    assert ev2.t_peak == pytest.approx(20.0)


def test_bboxes_and_track_ids_at_t_peak() -> None:
    visitor_box = (0.4, 0.6, 0.5, 0.9)
    tracks = make_tracks(60.0, spans=[(1, "person", "cashier", 0.0, 60.0, (0.3, 0.5, 0.45, 0.8)),
                                      (7, "person", "visitor", 20.0, 27.0, visitor_box)])
    segs = [(20.0, 26.0, {"outsider_in_cashier_area": 1.0, "n_persons": 2.0})]
    ev = next(e for e in run_rules(make_states(60.0, segments=segs), tracks, make_zones(), CAT) if e.type_id == "V01")
    assert ev.bboxes["visitor#7"] == pytest.approx(visitor_box)
    assert "cashier#1" in ev.bboxes and "zone:cashier_area" in ev.bboxes
    assert ev.track_ids == [1, 7]


def test_derived_signals_from_tracks_when_scene_signal_absent() -> None:
    tracks = make_tracks(60.0, spans=[(1, "person", "cashier", 0.0, 60.0, (0.3, 0.55, 0.45, 0.7)),
                                      (7, "person", "visitor", 20.0, 28.0, (0.4, 0.6, 0.5, 0.9))])
    states = make_states(60.0, omit=("outsider_in_cashier_area", "collector_present", "n_persons"))
    rep = evaluate(states, tracks, make_zones(), CAT)
    evs = [e for e in rep.events if e.type_id == "V01"]
    assert evs and evs[0].extra["trace"]["derived_signals"].get("outsider_in_cashier_area") == "tracks"
    assert rep.series.derived.get("outsider_in_cashier_area") == "tracks"
    assert evs[0].extra["trace"]["confidence_terms"]["derived_factor"] == pytest.approx(0.85)
    direct = make_states(60.0, segments=[(20.0, 28.0, {"outsider_in_cashier_area": 1.0, "n_persons": 2.0})])
    ev_direct = next(e for e in run_rules(direct, [], [], CAT) if e.type_id == "V01")
    assert evs[0].confidence < ev_direct.confidence


def test_states_from_two_sources_are_merged() -> None:
    pose = [{"t": i / 5, "signals": {"phone_in_hand": 1.0 if 20 <= i / 5 < 26 else 0.0}} for i in range(300)]
    scene = [{"t": float(i), "signals": {"cashier_in_workplace": 1.0, "cash_visible": 1.0}} for i in range(60)]
    evs = [e for e in run_rules(pose + scene, [], [], CAT) if e.type_id == "V05"]
    assert evs and evs[0].t_start == pytest.approx(20.0)


def test_build_series_unknown_and_ffill() -> None:
    states = [{"t": 0.0, "a": 1.0}, {"t": 0.2, "b": 0.5}, {"t": 0.4}, {"t": 2.0, "a": 0.0}]
    s = build_series(states, ["c"], ffill_max_s=0.5)
    assert s.signals["a"][1] == 1.0 and s.signals["a"][2] == 1.0  # ffill ≤ 0.5 c
    assert np.isnan(s.signals["b"][0]) and s.signals["b"][2] == 0.5
    assert np.isnan(s.signals["c"]).all() and "c" in s.absent
    assert s.coverage["a"] == 0.5


def test_bad_states_are_ignored() -> None:
    states = [{"t": "x"}, {"t": float("nan")}, None, {"t": 1.0, "safe_open": "bad"}]
    states = [s for s in states if s is not None]
    rep = evaluate(states, [], [], CAT)
    assert rep.events == []


def test_geometry_helpers() -> None:
    square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert point_in_polygon(0.5, 0.5, square) and not point_in_polygon(1.5, 0.5, square)
    tr = {"track_id": 1, "label": "person", "boxes": [(0.0, (0, 0, 0.2, 0.2)), (1.0, (0.2, 0.2, 0.4, 0.4))]}
    assert track_box_at(tr, 0.5) == pytest.approx((0.1, 0.1, 0.3, 0.3))
    assert track_box_at(tr, 5.0, tol_s=1.0) is None


def test_risk_curve_and_summary() -> None:
    segs = [(10.0, 30.0, {"safe_open": 1.0}), (15.0, 30.0, {"cashier_present": 0.0, "cashier_in_workplace": 0.0})]
    rep = evaluate(make_states(60.0, segments=segs), [], [], CAT)
    curve = rep.risk_curve(step_s=1.0)
    assert curve and all(0.0 <= r <= 1.0 for _, r in curve)
    assert max(r for t, r in curve if 16 <= t <= 29) >= 0.9
    summ = rep.summary()
    assert summ["candidate_violation"] >= 1 and summ["catalog_source"] == "arg"


def test_broken_rule_does_not_break_others() -> None:
    cfg = _cfg()
    rule = cfg.rule("V01_outsider_in_cashier_area")

    class Boom:
        names = frozenset()
        source = "boom"
        comparisons = ()

        def bounds(self, ctx):  # noqa: ANN001
            raise RuntimeError("boom")

    rule.strength = Boom()  # type: ignore[assignment]
    segs = [(10.0, 30.0, {"safe_open": 1.0}), (15.0, 30.0, {"cashier_present": 0.0, "cashier_in_workplace": 0.0})]
    rep = evaluate(make_states(60.0, segments=segs), [], [], CAT, cfg)
    assert any("boom" in w for w in rep.warnings)
    assert [e for e in rep.events if e.type_id == "V03"]
