"""Присутствие и составные сигналы: уход кассира, исчезновение трека, два сотрудника, посторонний."""
from __future__ import annotations

import numpy as np

from ml.scene.tests.synth import BANKNOTE, make_clip, person_dets, person_track, zones_default
from ml.scene._contract import Detection
from ml.scene.core import scene_states_with_trace

SEAT = (0.30, 0.78, 0.46, 0.92)  # кассир за столом (в cashier_area)


def run(clip, dets=(), tracks=(), zones=None):
    return scene_states_with_trace(clip, list(dets), list(tracks), zones if zones is not None else zones_default(), [])


def arr(states, name, field="signals"):
    return np.array([getattr(s, field)[name] for s in states])


def cash_dets(times, until=1e9):
    return [Detection(t=t, label="cash", bbox=BANKNOTE, score=0.8) for t in times if t < until]


def test_cashier_leaves_with_cash_on_desk_gives_unattended():
    times = [round(k / 5.0, 4) for k in range(60)]
    boxes = {}
    for t in times:
        if t < 3.0:
            boxes[t] = SEAT
        elif t < 4.0:  # уходит за правый край кадра
            a = min(1.0, (t - 3.0) / 0.8)
            boxes[t] = (0.30 + 0.60 * a, 0.78, 0.46 + 0.54 * a, 0.92)
    tr = person_track(1, boxes, role="cashier", conf=0.9)
    clip = make_clip(12.0, cash=lambda t: True, persons=lambda t: [boxes[t]] if t in boxes else [])
    st, trace = run(clip, person_dets([tr]) + cash_dets(times), [tr])
    t = arr(st, "workplace_unattended_with_cash")
    k = arr(st, "workplace_unattended_with_cash", "known")
    ts = np.array([s.t for s in st])
    assert (t[ts < 2.8] < 0.5).all()
    on = (ts >= 4.4) & (ts <= 11.0)
    assert (t[on] >= 0.5).all() and k[on].all()
    assert trace["info"]["unattended_runs"] and trace["info"]["unattended_runs"][0][0] <= 4.6
    ciw = arr(st, "cashier_in_workplace")
    ciw_k = arr(st, "cashier_in_workplace", "known")
    assert (ciw[ts > 5] < 0.5).all() and ciw_k[ts > 5].all(), "уход подтверждён: трек ушёл к краю кадра"


def test_track_lost_mid_workplace_is_unknown_not_unattended():
    times = [round(k / 5.0, 4) for k in range(60)]
    boxes = {t: SEAT for t in times if t < 4.0 or t >= 9.0}  # детекции пропали на 5 с посреди рабочего места
    tr1 = person_track(1, {t: b for t, b in boxes.items() if t < 4.0}, role="cashier", conf=0.9)
    tr2 = person_track(2, {t: b for t, b in boxes.items() if t >= 9.0}, role="cashier", conf=0.9)
    clip = make_clip(12.0, cash=lambda t: True, persons=lambda t: [SEAT])
    st, _ = run(clip, person_dets([tr1, tr2]) + cash_dets(times), [tr1, tr2])
    ts = np.array([s.t for s in st])
    gap = (ts >= 4.2) & (ts < 8.8)
    u = arr(st, "workplace_unattended_with_cash")
    assert (u < 0.5).all(), "пропавшая детекция кассира не доказывает оставленные без присмотра ценности"
    ciw_known = arr(st, "cashier_in_workplace", "known")
    ciw = arr(st, "cashier_in_workplace")
    assert not (ciw_known[gap] & (ciw[gap] < 0.5)).any(), "в разрыве трека отсутствие кассира не утверждается"
    lost = np.array([s.observed["workplace_track_lost"] for s in st])
    assert lost[gap].min() == 1.0


def test_two_employees_unknown_role_is_not_outsider():
    times = [round(k / 5.0, 4) for k in range(40)]
    b1, b2 = SEAT, (0.60, 0.70, 0.76, 0.86)
    tr1 = person_track(1, {t: b1 for t in times}, role="cashier", conf=0.9)
    tr2 = person_track(2, {t: b2 for t in times}, role="unknown", conf=0.0)
    clip = make_clip(8.0, persons=lambda t: [b1, b2])
    st, _ = run(clip, person_dets([tr1, tr2]), [tr1, tr2])
    out = arr(st, "outsider_in_cashier_area")
    known = arr(st, "outsider_in_cashier_area", "known")
    assert (out < 0.5).all(), "сотрудник с неизвестной ролью ≠ посторонний"
    assert not (known & (out >= 0.5)).any()
    unk = np.array([s.observed["unknown_person_in_cashier_area"] for s in st])
    assert unk.min() >= 0.99, "неизвестный человек в зоне кассира передан как наблюдение для проверки"
    assert arr(st, "n_persons").max() == 2


def test_two_explicit_cashiers_outsider_known_false():
    times = [round(k / 5.0, 4) for k in range(30)]
    b1, b2 = SEAT, (0.60, 0.70, 0.76, 0.86)
    tr1 = person_track(1, {t: b1 for t in times}, role="cashier", conf=0.9)
    tr2 = person_track(2, {t: b2 for t in times}, role="cashier", conf=0.85)
    st, _ = run(make_clip(6.0, persons=lambda t: [b1, b2]), person_dets([tr1, tr2]), [tr1, tr2])
    out = arr(st, "outsider_in_cashier_area")
    known = arr(st, "outsider_in_cashier_area", "known")
    assert (out < 0.5).all() and known[3:].all()


def test_explicit_visitor_in_cashier_area_is_outsider():
    times = [round(k / 5.0, 4) for k in range(40)]
    vb = (0.55, 0.60, 0.70, 0.75)
    tr1 = person_track(1, {t: SEAT for t in times}, role="cashier", conf=0.9)
    tr2 = person_track(2, {t: vb for t in times if t >= 3.0}, role="visitor", conf=0.85)
    st, _ = run(make_clip(8.0, persons=lambda t: [SEAT] + ([vb] if t >= 3.0 else [])), person_dets([tr1, tr2]), [tr1, tr2])
    ts = np.array([s.t for s in st])
    out = arr(st, "outsider_in_cashier_area")
    known = arr(st, "outsider_in_cashier_area", "known")
    assert (out[ts >= 3.6] >= 0.5).all() and known[ts >= 3.6].all()
    assert (out[ts < 2.4] < 0.5).all()


def test_unknown_person_coming_from_client_area_is_outsider():
    times = [round(k / 5.0, 4) for k in range(50)]
    path = {}
    for t in times:
        y = 0.10 if t < 4.0 else 0.62  # стоял у клиентской зоны, затем — в зоне кассира
        path[t] = (0.55, y, 0.70, y + 0.14)
    tr1 = person_track(1, {t: SEAT for t in times}, role="cashier", conf=0.9)
    tr2 = person_track(2, path, role="unknown", conf=0.0)
    st, _ = run(make_clip(10.0, persons=lambda t: [SEAT, path[t]]), person_dets([tr1, tr2]), [tr1, tr2])
    ts = np.array([s.t for s in st])
    out = arr(st, "outsider_in_cashier_area")
    assert (out[ts >= 4.6] >= 0.5).all()
    assert (out[ts < 3.4] < 0.5).all()


def test_no_detections_everything_unknown():
    st, tr = run(make_clip(5.0))
    for name in ("cash_visible", "cashier_in_workplace", "cashier_present", "workplace_unattended_with_cash", "collector_present"):
        assert not arr(st, name, "known").any(), name
    assert (arr(st, "workplace_unattended_with_cash") < 0.5).all()
    assert any("no detections" in w for w in tr["warnings"])


def test_detections_only_no_tracks_cannot_confirm_departure():
    times = [round(k / 5.0, 4) for k in range(50)]
    dets = [Detection(t=t, label="person", bbox=SEAT, score=0.9) for t in times if t < 3.0] + cash_dets(times)
    st, _ = run(make_clip(10.0, cash=lambda t: True), dets, [])
    assert (arr(st, "workplace_unattended_with_cash") < 0.5).all()


def test_collector_with_bag_is_not_outsider():
    times = [round(k / 5.0, 4) for k in range(40)]
    cb = (0.56, 0.62, 0.72, 0.78)
    bag = (0.60, 0.70, 0.66, 0.76)
    tr1 = person_track(1, {t: SEAT for t in times}, role="cashier", conf=0.9)
    tr2 = person_track(2, {t: cb for t in times}, role="collector", conf=0.8)
    dets = person_dets([tr1, tr2]) + [Detection(t=t, label="cash_bag", bbox=bag, score=0.7) for t in times]
    st, _ = run(make_clip(8.0, persons=lambda t: [SEAT, cb]), dets, [tr1, tr2])
    assert (arr(st, "outsider_in_cashier_area") < 0.5).all()
    assert (arr(st, "collector_present")[3:] >= 0.5).all()
    assert (arr(st, "cash_bag_visible")[3:] >= 0.5).all()
