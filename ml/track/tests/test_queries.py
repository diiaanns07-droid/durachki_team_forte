"""Явные функции-события: значение + уверенность + источник; None = «неизвестно»."""
import pytest

from ml.track import TrackQueries, track_full
from ml.track.config import load_config
from ml.track.tests import synth as S
from ml.track.tests._util import run


def _q(res, cfg=None):
    return TrackQueries(res.tracks, res.zones, res.info, cfg)


def test_clean_no_false_events():
    res, *_ = run(S.scene_clean())
    q = _q(res)
    kinds = [e.kind for e in q.events()]
    assert "cashier_left_workplace" not in kinds and "outsider_at_cashier_side" not in kinds
    assert kinds.count("person_entered") == 1  # только клиент через дверь
    a = q.n_outsiders(20.0)
    assert a.value == 1 and a.detail["unknown"] == 0  # клиент напротив — посетитель, это норма
    assert q.outsider_at_cashier_side(20.0).value is False


def test_v02_left_workplace_and_room_with_onset():
    res, *_ = run(S.scene_v02())
    q = _q(res)
    a = q.cashier_left_workplace(30.0)
    assert a.value is True and a.conf > 0.5 and 15.0 <= a.t_event <= 19.0 and a.source
    r = q.cashier_left_room(30.0)
    assert r.value is True and r.source == "last_seen_in_door_zone" and abs(r.t_event - 19.0) < 0.5
    assert q.cashier_left_workplace(10.0).value is False
    back = q.cashier_left_workplace(55.0)
    assert back.value is False


def test_occlusion_is_unknown_not_left():
    sc = S.scene_clean()
    sc.persons[0].miss = [(20.0, 24.0)]  # кассир сидит, детектор его не видит 4 с
    res, ids, by_id, _, _ = run(sc)
    q = _q(res)
    a = q.cashier_left_workplace(22.0)
    assert a.value is None and "lost_in_view" in a.source
    assert q.cashier_left_room(22.0).value is None
    assert not [e for e in q.events() if e.kind in ("cashier_left_workplace", "cashier_left_room")]


def test_visibility_loss_answers_unknown_and_no_false_entries():
    res, ids, by_id, _, _ = run(S.scene_clean(), blackout=[(20.0, 25.0)])
    cov = res.info["coverage"]
    assert cov["visibility_check"] == "objects" and len(cov["visibility_loss"]) == 1
    a, b = cov["visibility_loss"][0]
    assert 19.5 <= a <= 20.5 and 24.5 <= b <= 25.5
    q = _q(res)
    for fn in (q.cashier_left_workplace, q.cashier_left_room, q.n_outsiders, q.outsider_at_cashier_side):
        assert fn(22.0).value is None and fn(22.0).source == "visibility_loss"
    entries = [e for e in q.events() if e.kind == "person_entered"]
    assert len(entries) == 1 and entries[0].t < 10  # после включения света никто «не вошёл»
    assert not [e for e in q.events() if e.kind in ("cashier_left_workplace", "cashier_left_room")]


def test_fragment_birth_in_view_is_not_entry_eliminated_fp_source():
    """Устранённый источник ложных срабатываний: обрыв трека в глубине кадра ≠ вход человека."""
    sc = S.scene_clean()
    sc.persons[1].miss = [(16.0, 20.0)]  # клиент стоит у стола, детектор теряет его на 4 с
    res, *_ = run(sc)
    strict = [e for e in _q(res).events() if e.kind == "person_entered"]
    naive_cfg = load_config(overrides={"queries": {"entry_requires_boundary": False}})
    naive = [e for e in _q(res, naive_cfg).events() if e.kind == "person_entered"]
    assert len(strict) == 1                      # только реальный вход через дверь
    assert len(naive) > len(strict)              # наивный режим: лишний «вход» в 20 с


def test_v01_unknown_passed_as_unknown_not_outsider():
    res, ids, *_ = run(S.scene_v01())
    q = _q(res)
    a = q.outsider_at_cashier_side(18.0)
    assert a.value is None and a.source == "unknown_role_at_cashier_side"
    n = q.n_outsiders(18.0)
    assert n.value == 0 and n.detail["upper_bound"] == 1 and n.conf < 0.9
    assert any(e.kind == "unknown_at_cashier_side" for e in q.events())


def test_person_entered_role_filter_and_collector():
    res, *_ = run(S.scene_v07())
    q = _q(res)
    assert q.person_entered(10.4, role="collector").value is True
    assert q.person_entered(10.4, role="visitor").value is False
    assert q.outsider_at_cashier_side(16.0).value in (False, None)


def test_no_cashier_or_no_zone_is_unknown():
    clip, dets, _ = S.make(S.Scene(20.0, [S.walker("v", [(2, *S.DOOR_PT), (6, *S.CLIENT), (12, *S.DOOR_PT)], (0, 0, 200))], []))
    res = track_full(clip, dets)
    q = _q(res)
    assert q.cashier_left_workplace(5.0).value is None
    assert q.cashier_left_room(5.0).value is None
    assert q.outsider_at_cashier_side(5.0).value is None


def test_spurious_low_score_tracks_do_not_inflate_counts():
    res, *_ = run(S.scene_clean(), noise_dets=0.08, seed=4)
    q = _q(res)
    for t in (5.0, 15.0, 35.0):
        assert q.n_outsiders(t).detail["upper_bound"] <= 1
    assert sum(1 for e in q.events() if e.kind == "person_entered") == 1


@pytest.mark.parametrize("mode", ["off", "hysteresis"])
def test_smoothing_modes_keep_true_event(mode):
    cfg = load_config(overrides={"queries": {"smoothing": mode}})
    res, *_ = run(S.scene_v02(), cfg=cfg)
    q = TrackQueries(res.tracks, res.zones, res.info, cfg)
    lw = [e for e in q.events() if e.kind == "cashier_left_workplace"]
    assert len(lw) == 1 and 15.0 <= lw[0].t <= 19.0
