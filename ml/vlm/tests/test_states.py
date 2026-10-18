"""Временная логика режима states: пары «нарушение / чистый двойник» на синтетических состояниях."""
from ml.vlm.states import QUESTIONS, infer_events, sample_times


def _seq(rows):
    base = {k: 0 for k in QUESTIONS}
    return [(float(t), {**base, **r}) for t, r in rows]


def test_leave_with_bag_on_desk_is_violation_at_leave_time():
    seq = _seq([(0, {"cs": 1}), (1, {"cs": 1, "gb": 1}), (2, {"cs": 1, "gb": 1}), (3, {"cs": 0, "gb": 1}),
                (4, {"cs": 0, "gb": 1}), (5, {"cs": 0, "gb": 1})])
    ev = infer_events(seq)
    assert [e.type_id for e in ev] == ["V07"]
    assert abs(ev[0].t_peak - 2.5) < 1e-6


def test_clean_leave_gives_nothing():
    seq = _seq([(0, {"cs": 1}), (1, {"cs": 1}), (2, {"cs": 0}), (3, {"cs": 0}), (4, {"cs": 0})])
    assert infer_events(seq) == []


def test_leave_with_safe_open_is_v03():
    seq = _seq([(0, {"cs": 1, "so": 1, "as": 1}), (1, {"cs": 0, "so": 1}), (2, {"cs": 0, "so": 1})])
    assert [e.type_id for e in infer_events(seq)] == ["V03"]


def test_client_opposite_is_normal_but_two_at_cashier_side_is_v01():
    normal = _seq([(t, {"cs": 1, "cl": 1}) for t in range(6)])
    assert infer_events(normal) == []
    two = _seq([(0, {"cs": 1}), (1, {"cs": 2}), (2, {"cs": 2}), (3, {"cs": 2})])
    assert [e.type_id for e in infer_events(two)] == ["V01"]


def test_phone_needs_two_frames_and_single_drink_counts():
    one = _seq([(0, {"cs": 1}), (1, {"cs": 1, "ph": 1}), (2, {"cs": 1})])
    assert infer_events(one) == []
    drink = _seq([(0, {"cs": 1}), (1, {"cs": 1, "ed": 1}), (2, {"cs": 1})])
    assert [e.type_id for e in infer_events(drink)] == ["V06"]


def test_sampling_density_depends_on_duration():
    assert len(sample_times(20, {})) == 20
    assert len(sample_times(140, {})) == 70
