"""Граница A08 → A09 (после аудита): неизвестное/скрытое не становится «нет» и не создаёт нарушений.

Синтетика (без реальных данных). ``a09_view`` повторяет документированное чтение A09 (ml/rules/signals.py
build_series): значение берётся только при known; observed — односторонне (≥ 0.5 = наблюдалось).
"""
from __future__ import annotations

import cv2
import numpy as np

from ml.scene._contract import SIGNALS, Detection, SceneState, Zone
from ml.scene.core import scene_states_with_trace
from ml.scene.tests import test_phase2 as S
from ml.scene.tests.synth import H, W, make_clip, person_dets, person_track
from ml.scene.types import state_from_dict, state_to_dict

T = S.TIMES


def poly(b):
    return [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]


def a06_like_zones():
    """Как строит A06: клиенты по другую сторону стола, зона кассира = место кассира + полоса стола."""
    return [Zone("client_area", poly((0.02, 0.0, 0.98, 0.40))), Zone("desk", poly((0.10, 0.40, 0.70, 0.55))),
            Zone("cashier_area", poly((0.02, 0.47, 0.98, 0.98))), Zone("safe", poly(S.SAFE)), Zone("door", poly(S.DOOR))]


def a09_view(states, name):
    """Ряд, который увидит A09: NaN — неизвестно."""
    out = []
    for s in states:
        if name in s.signals:
            out.append(float(s.signals[name]) if s.known.get(name, False) else np.nan)
        else:
            v = float(s.observed.get(name, 0.0))
            out.append(v if v >= 0.5 else np.nan)
    return np.array(out)


def run(dets=(), tracks=(), zones=None, pose=(), clip=None, config=None):
    return scene_states_with_trace(clip or make_clip(S.DUR), list(dets), list(tracks), zones if zones is not None else a06_like_zones(),
                                   list(pose), config=config)


SEAT = (0.30, 0.62, 0.46, 0.80)


# ------------------------------------------------------------------ клиент напротив / сторона кассира
def test_client_opposite_sitting_or_leaning_never_outsider():
    cashier = person_track(1, {t: SEAT for t in T})
    for box in ((0.30, 0.20, 0.46, 0.38), (0.30, 0.24, 0.46, 0.62)):   # сидит / наклонился над столом
        client = person_track(2, {t: box for t in T})
        st, _ = run(person_dets([cashier, client]), [cashier, client])
        assert "true" not in [s.status["outsider_in_cashier_area"] for s in st]
        assert np.nanmax(np.r_[a09_view(st, "outsider_in_cashier_area"), 0.0]) < 0.5
        assert np.isnan(a09_view(st, "unknown_person_in_cashier_area")).all(), "клиент у стойки — не «человек в зоне кассира»"
        assert "true" not in [s.pred_status["outsider_cashier_side"] for s in st]


def test_unknown_role_on_cashier_side_is_unknown_and_visitor_from_client_side_is_true():
    cashier = person_track(1, {t: SEAT for t in T})
    other = person_track(2, {t: (0.60, 0.64, 0.76, 0.82) for t in T})
    st, _ = run(person_dets([cashier, other]), [cashier, other])
    assert {s.status["outsider_in_cashier_area"] for s in st[5:]} == {"unknown"}
    assert {s.why["outsider_in_cashier_area"] for s in st[5:]} == {"role_unknown"}
    assert np.isnan(a09_view(st, "outsider_in_cashier_area")).all(), "неизвестная роль ≠ посторонний"
    assert (a09_view(st, "unknown_person_in_cashier_area")[5:] >= 0.5).all(), "факт для проверки человеком/VLM"
    # положительный контроль: пришёл со стороны клиента и стоит за столом — посторонний
    vb = {t: ((0.60, 0.20, 0.74, 0.36) if t < 4.0 else (0.60, 0.64, 0.74, 0.82)) for t in T}
    visitor = person_track(3, vb)
    st2, _ = run(person_dets([cashier, visitor]), [cashier, visitor])
    tt = np.array([s.t for s in st2])
    assert (a09_view(st2, "outsider_in_cashier_area")[tt >= 5.0] >= 0.5).all()


# ------------------------------------------------------------------ пропавший трек
def test_lost_track_makes_presence_unknown_not_absent():
    t1 = person_track(1, {t: SEAT for t in T if t < 4.0}, role="cashier", conf=0.9)
    t2 = person_track(2, {t: SEAT for t in T if t >= 9.0}, role="cashier", conf=0.9)
    st, _ = run(person_dets([t1, t2]) + S.cash(box=S.NOTE), [t1, t2])
    tt = np.array([s.t for s in st])
    gap = (tt >= 4.8) & (tt < 8.2)  # ±~0.7 с у краёв — «недавно наблюдали» (удержание conf, tau 1 с)
    for name in ("cashier_present", "cashier_in_workplace", "collector_present", "n_persons"):
        assert np.isnan(a09_view(st, name)[gap]).all(), name
        assert {s.why[name] for s, g in zip(st, gap) if g} == {"track_lost"}, name
    assert np.isnan(a09_view(st, "workplace_empty")[gap]).all(), "observed «пусто» без подтверждения не отдаётся"
    assert (np.array([s.observed["workplace_empty_conf"] for s in st])[gap] == 0).all()
    for s in st:
        assert s.status["workplace_unattended_with_cash"] != "true"
        assert s.pred_status["cash_left_after_leave"] != "true"


def test_confirmed_exit_gives_known_absence():
    tr = S.leaving_cashier()
    st, _ = run(person_dets([tr]) + S.cash(box=S.NOTE), [tr])
    tt = np.array([s.t for s in st])
    after = tt >= 4.4
    assert (a09_view(st, "cashier_present")[after] < 0.5).all()
    assert (a09_view(st, "n_persons")[after] == 0).all()
    assert (a09_view(st, "collector_present")[after] < 0.5).all()
    assert (a09_view(st, "workplace_empty")[after] >= 0.5).all()


# ------------------------------------------------------------------ перекрытые ценности
def test_occluded_valuables_are_occluded_not_false():
    cover_desk = (0.08, 0.38, 0.72, 0.57)
    cover_safe = (S.SAFE[0] - 0.04, S.SAFE[1] - 0.03, S.SAFE[2] + 0.03, S.SAFE[3] + 0.04)
    a = person_track(1, {t: cover_desk for t in T}, role="cashier", conf=0.9)
    b = person_track(2, {t: cover_safe for t in T}, role="cashier", conf=0.9)
    st, _ = run(person_dets([a, b]), [a, b])
    for s in st[5:]:
        assert s.status["cash_visible"] == "occluded"
        assert s.status["safe_open"] in ("occluded", "unknown")
    assert np.isnan(a09_view(st, "cash_visible")[5:]).all()


def test_cash_in_client_hands_is_not_cash_on_desk():
    tr = S.leaving_cashier()
    client = person_track(2, {t: (0.30, 0.20, 0.46, 0.38) for t in T})
    held = S.cash(box=(0.33, 0.30, 0.41, 0.36))          # купюры в руках клиента, над клиентской зоной
    st, _ = run(person_dets([tr, client]) + held, [tr, client])
    v = a09_view(st, "cash_visible")
    assert not (v >= 0.5).any(), "наличные у клиента не дают cash_visible (иначе ложный V02 после ухода)"
    on_desk = S.cash(box=(0.30, 0.44, 0.40, 0.50))
    st2, _ = run(person_dets([tr]) + on_desk, [tr])
    assert (a09_view(st2, "cash_visible")[10:] >= 0.5).all(), "на столе — видны"


# ------------------------------------------------------------------ поза отсутствует
POSE_KEYS = ("phone_in_hand", "phone_to_ear", "eating_drinking", "hands_hidden", "cashier_facing_away", "handover_motion")


def test_missing_or_partial_pose_states_are_unknown():
    tr = person_track(1, {t: SEAT for t in T}, role="cashier", conf=0.9)
    st, _ = run(person_dets([tr]), [tr], pose=[])
    for s in st:
        for k in POSE_KEYS:
            assert s.status[k] == "unknown" and s.why[k] == "no_pose", (k, s.status[k])
    # A07: «нет ключа = неизвестно» — состояния есть, но без phone_in_hand
    pose = [SceneState(t=t, signals={"eating_drinking": 0.0}) for t in T]
    st2, _ = run(person_dets([tr]), [tr], pose=pose)
    assert {s.status["phone_in_hand"] for s in st2} == {"unknown"}
    assert {s.status["eating_drinking"] for s in st2} == {"false"}


# ------------------------------------------------------------------ смена зон
def test_zone_change_absent_shifted_and_foreign_names():
    tr = S.leaving_cashier()
    dets = person_dets([tr]) + S.cash(box=S.NOTE)
    st_none, trace = run(dets, [tr], zones=[])
    for s in st_none:
        for name in ("cashier_in_workplace", "outsider_in_cashier_area", "cash_visible"):
            assert s.status[name] == "unknown", (name, s.status[name])
        assert "true" not in s.pred_status.values()
    assert st_none[20].why["cash_visible"] == "no_zone" and st_none[20].why["cashier_in_workplace"] == "no_zone"
    # те же решения при сдвинутом ракурсе (зоны и боксы сдвинуты одинаково) и с посторонними именами зон
    m = np.float32([[1, 0, 0.03 * W], [0, 1, -0.02 * H]])

    def sh(b):
        return (b[0] + 0.03, b[1] - 0.02, b[2] + 0.03, b[3] - 0.02)

    zones2 = [Zone(z.name, [(x + 0.03, y - 0.02) for x, y in z.polygon]) for z in a06_like_zones()] + [Zone("lobby", poly((0, 0, 1, 0.1)))]
    tr2 = person_track(1, {t: sh(b) for t, b in tr.boxes}, role="cashier", conf=0.9)
    clip2 = make_clip(S.DUR, post=lambda f, t, i: cv2.warpAffine(f, m, (W, H), borderMode=cv2.BORDER_REFLECT))
    base, _ = run(dets, [tr])
    moved, _ = run(person_dets([tr2]) + S.cash(box=sh(S.NOTE)), [tr2], zones=zones2, clip=clip2)
    for name in ("cashier_in_workplace", "cashier_present", "cash_visible", "workplace_unattended_with_cash"):
        assert [s.status[name] for s in base] == [s.status[name] for s in moved], name
    assert [s.pred_status["cash_left_after_leave"] for s in base] == [s.pred_status["cash_left_after_leave"] for s in moved]


# ------------------------------------------------------------------ имена и сериализация для A09/A03
def test_personal_items_exposed_under_a09_name():
    tr = person_track(1, {t: SEAT for t in T}, role="cashier", conf=0.9)
    item = [Detection(t=t, label="bag", bbox=(0.50, 0.42, 0.62, 0.52), score=0.7) for t in T if t >= 3.0]
    st, _ = run(person_dets([tr]) + item, [tr])
    v = a09_view(st, "personal_items_on_desk")
    assert (v[np.array([s.t for s in st]) >= 5.2] >= 0.5).all()
    assert np.isnan(v[:10]).all(), "до появления вещей — не наблюдалось (не «нет»)"


def test_json_roundtrip_keeps_unknown_gating_fields():
    t1 = person_track(1, {t: SEAT for t in T if t < 4.0}, role="cashier", conf=0.9)
    st, _ = run(person_dets([t1]), [t1])
    back = [state_from_dict(state_to_dict(s)) for s in st]
    for name in SIGNALS:
        assert np.array_equal(np.isnan(a09_view(st, name)), np.isnan(a09_view(back, name))), name
