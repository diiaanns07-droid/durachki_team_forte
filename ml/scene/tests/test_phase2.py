"""Фаза 2: статусы false/unknown/occluded, предикаты для A09, свет, интервалы, replay.

Синтетика без реальных данных. Планировка «стойка»: сверху клиентская зона, посередине стол, снизу —
сторона кассира (координаты только в тестах; модуль берёт зоны из A06)."""
from __future__ import annotations

import numpy as np

from ml.scene._contract import Detection, SceneState, Zone
from ml.scene.core import scene_states_with_trace
from ml.scene.intervals import coverage, intervals
from ml.scene.io import dump_states, load_states
from ml.scene.tests.synth import make_clip, person_dets, person_track
from ml.scene.types import predicate

CLIENT = (0.02, 0.0, 0.98, 0.40)
DESK = (0.10, 0.40, 0.70, 0.55)
CASHIER = (0.02, 0.55, 0.98, 0.98)
SAFE = (0.72, 0.60, 0.92, 0.75)
DOOR = (0.0, 0.42, 0.08, 0.62)
SEAT = (0.30, 0.60, 0.46, 0.76)            # кассир за столом (опорная точка — сторона кассы)
CLIENT_SEAT = (0.30, 0.20, 0.46, 0.36)     # клиент напротив
CLIENT_LEAN = (0.30, 0.30, 0.46, 0.50)     # клиент наклонился над столом (опорная точка — на столе)
NOTE = (0.30, 0.45, 0.40, 0.50)            # купюры на столе
DUR = 12.0


def poly(b):
    return [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]


def zones():
    return [Zone("client_area", poly(CLIENT)), Zone("desk", poly(DESK)), Zone("cashier_area", poly(CASHIER)),
            Zone("safe", poly(SAFE)), Zone("door", poly(DOOR))]


TIMES = [round(k / 5.0, 4) for k in range(int(DUR * 5))]


def run(dets=(), tracks=(), pose=(), clip=None, config=None):
    clip = clip or make_clip(DUR)
    return scene_states_with_trace(clip, list(dets), list(tracks), zones(), list(pose), config=config)


def pst(states, name):
    return np.array([s.pred_status[name] for s in states])


def ts(states):
    return np.array([s.t for s in states])


def leaving_cashier(t_walk=3.0, role="cashier", conf=0.9):
    """Кассир сидит до t_walk, затем за 0.6 с уходит за правый край кадра."""
    boxes = {}
    for t in TIMES:
        if t < t_walk:
            boxes[t] = SEAT
        elif t < t_walk + 0.8:
            a = min(1.0, (t - t_walk) / 0.6)
            boxes[t] = (0.30 + 0.62 * a, 0.60, 0.46 + 0.54 * a, 0.76)
    return person_track(1, boxes, role=role, conf=conf)


def cash(until=1e9, since=-1.0, box=NOTE, label="cash"):
    return [Detection(t=t, label=label, bbox=box, score=0.8) for t in TIMES if since <= t < until]


# ------------------------------------------------------------------ статусы
def test_status_distinguishes_unknown_occluded_false():
    st, _ = run()  # нет детекций вообще
    s = st[10]
    assert s.status["cash_visible"] == "unknown" and s.why["cash_visible"] == "no_detector"
    assert s.status["outsider_in_cashier_area"] in ("unknown", "false")
    # человек закрыл весь стол → наличные «скрыто», а не «нет»
    cover = (0.08, 0.38, 0.72, 0.57)
    tr = person_track(1, {t: cover for t in TIMES}, role="cashier", conf=0.9)
    st2, _ = run(person_dets([tr]), [tr])
    assert st2[20].status["cash_visible"] == "occluded"
    # стол виден, детектор работает, наличных нет → «нет»
    tr2 = person_track(1, {t: SEAT for t in TIMES}, role="cashier", conf=0.9)
    st3, _ = run(person_dets([tr2]), [tr2])
    assert st3[20].status["cash_visible"] == "false"


# ------------------------------------------------------------------ уход и инвентарь стола
def test_cash_left_after_leave_true_with_onset_at_leave():
    tr = leaving_cashier()
    st, trace = run(person_dets([tr]) + cash(), [tr])
    t = ts(st)
    s = pst(st, "cash_left_after_leave")
    assert (s[t < 2.8] == "false").all()
    assert (s[t >= 4.2] == "true").all()
    iv = trace["intervals"]["cash_left_after_leave"]
    assert len(iv) == 1 and 3.0 <= iv[0]["t_start"] <= 4.0
    leave = [e for e in trace["events"] if e["type"] == "leave"]
    assert len(leave) == 1 and leave[0]["decisions"]["cash_left_after_leave"] == "true"
    assert (pst(st, "cashier_left_workplace")[t >= 4.2] == "true").all()


def test_leave_after_removing_cash_is_normal():
    tr = leaving_cashier()
    st, trace = run(person_dets([tr]) + cash(until=2.8), [tr])
    s = pst(st, "cash_left_after_leave")
    assert "true" not in s
    assert [e for e in trace["events"] if e["type"] == "leave"][0]["decisions"]["cash_left_after_leave"] == "false"


def test_lights_off_after_leave_freezes_last_known_state():
    t_off = 4.0  # свет выключили сразу после ухода: в окне только 1 «светлый» сэмпл с наличными

    def dark(f, t, i):
        return (f.astype(np.float32) * (0.08 if t >= t_off else 1.0)).astype(np.uint8)

    tr = leaving_cashier(t_walk=3.0)
    clip = make_clip(DUR, post=dark)
    st, trace = run(person_dets([tr]) + cash(until=t_off), [tr], clip=clip)
    t = ts(st)
    lo = pst(st, "lights_off")
    assert (lo[t >= t_off + 1.2] == "true").all() and (lo[t < 3.0] == "false").all()
    assert (pst(st, "cash_left_after_leave")[t >= 4.4] == "true").all(), "последнее известное состояние заморожено"
    # в темноте сигналы — unknown(dark), а не «нет»
    assert st[-1].status["cash_visible"] != "false" and st[-1].why["cash_visible"] == "dark"


def test_lights_off_without_known_state_is_unknown_dark():
    def dark(f, t, i):
        return (f.astype(np.float32) * (0.08 if t >= 3.6 else 1.0)).astype(np.uint8)

    tr = leaving_cashier(t_walk=3.0)
    st, _ = run(person_dets([tr]), [tr], clip=make_clip(DUR, post=dark))
    s = pst(st, "cash_left_after_leave")
    assert "true" not in s
    assert st[-1].pred_why["cash_left_after_leave"] in ("dark", "no_evidence")


def test_safe_open_after_leave_and_while_working():
    tr = leaving_cashier(t_walk=6.0)
    safe = [Detection(t=t, label="safe_open", bbox=SAFE, score=0.8) for t in TIMES]
    st, trace = run(person_dets([tr]) + safe, [tr])
    t = ts(st)
    assert (pst(st, "safe_open_after_leave")[t >= 7.2] == "true").all()
    sow = pst(st, "safe_open_while_working")
    assert (sow[(t >= 3.6) & (t < 5.8)] == "true").all(), "работа за столом при открытом сейфе — кандидат"


def test_bag_left_on_desk_after_leave():
    tr = leaving_cashier()
    st, _ = run(person_dets([tr]) + cash(label="cash_bag", box=(0.20, 0.42, 0.30, 0.52)), [tr])
    assert (pst(st, "bag_left_after_leave")[ts(st) >= 4.2] == "true").all()


# ------------------------------------------------------------------ люди
def test_client_opposite_does_not_raise_outsider():
    cashier = person_track(1, {t: SEAT for t in TIMES})
    client_boxes = {t: (CLIENT_LEAN if 4.0 <= t < 7.0 else CLIENT_SEAT) for t in TIMES}
    client = person_track(2, client_boxes)
    st, _ = run(person_dets([cashier, client]), [cashier, client])
    s = pst(st, "outsider_cashier_side")
    assert "true" not in s
    assert (s[5:] == "false").mean() > 0.9
    assert "true" not in pst(st, "second_person_client_side")


def test_outsider_from_client_side_is_true_with_onset():
    cashier = person_track(1, {t: SEAT for t in TIMES})
    vb = {t: ((0.60, 0.20, 0.74, 0.36) if t < 4.0 else (0.60, 0.62, 0.74, 0.78)) for t in TIMES}
    visitor = person_track(2, vb)
    st, trace = run(person_dets([cashier, visitor]), [cashier, visitor])
    t = ts(st)
    s = pst(st, "outsider_cashier_side")
    assert (s[t < 3.8] != "true").all() and (s[t >= 4.2] == "true").all()
    assert abs(trace["intervals"]["outsider_cashier_side"][0]["t_start"] - 4.0) <= 0.21
    assert any(e["type"] == "entry" and e["non_staff"] for e in trace["events"])


def test_two_employees_second_is_unknown_not_outsider():
    a = person_track(1, {t: SEAT for t in TIMES})
    b = person_track(2, {t: (0.60, 0.62, 0.76, 0.78) for t in TIMES if t >= 2.0})
    st, _ = run(person_dets([a, b]), [a, b])
    s = pst(st, "outsider_cashier_side")
    assert "true" not in s
    assert st[30].pred_why["outsider_cashier_side"] == "role_unknown"
    a2 = person_track(1, {t: SEAT for t in TIMES}, role="cashier", conf=0.9)
    b2 = person_track(2, {t: (0.60, 0.62, 0.76, 0.78) for t in TIMES}, role="cashier", conf=0.9)
    st2, _ = run(person_dets([a2, b2]), [a2, b2])
    assert (pst(st2, "outsider_cashier_side")[5:] == "false").all()


def test_cashier_entering_through_door_first_is_not_outsider():
    boxes = {}
    for t in TIMES:
        if t < 1.0:
            boxes[t] = (0.0, 0.44, 0.10, 0.60)          # в двери
        elif t < 2.0:
            a = t - 1.0
            boxes[t] = (0.05 + 0.25 * a, 0.58, 0.18 + 0.28 * a, 0.74)
        else:
            boxes[t] = SEAT
    cashier = person_track(1, boxes)
    st, trace = run(person_dets([cashier]), [cashier])
    assert "true" not in pst(st, "outsider_cashier_side")
    assert not [e for e in trace["events"] if e["type"] == "entry"]


def test_entry_while_safe_open():
    cashier = person_track(1, {t: SEAT for t in TIMES})
    vb = {t: ((0.60, 0.20, 0.74, 0.36) if t < 4.0 else (0.60, 0.62, 0.74, 0.78)) for t in TIMES}
    visitor = person_track(2, vb)
    safe = [Detection(t=t, label="safe_open", bbox=SAFE, score=0.8) for t in TIMES]
    st, trace = run(person_dets([cashier, visitor]) + safe, [cashier, visitor])
    t = ts(st)
    e = pst(st, "entry_while_valuables_open")
    assert (e[(t >= 4.0) & (t <= 5.8)] == "true").all() and (e[t < 3.8] != "true").all()


# ------------------------------------------------------------------ предметы
def _cashier_with(pose_key, obj_label=None, obj_box=(0.36, 0.64, 0.40, 0.68), static=False):
    tr = person_track(1, {t: SEAT for t in TIMES}, role="cashier", conf=0.9)
    pose = [SceneState(t=t, signals={pose_key: 1.0 if 2.0 <= t < 6.0 else 0.0}) for t in TIMES]
    dets = person_dets([tr])
    if obj_label:
        rng = TIMES if static else [t for t in TIMES if 2.0 <= t < 6.0]
        dets += [Detection(t=t, label=obj_label, bbox=obj_box, score=0.7) for t in rng]
    return dets, [tr], pose


def test_mobile_phone_needs_visible_phone_not_landline():
    st, _ = run(*_cashier_with("phone_in_hand", "cell_phone"))
    t = ts(st)
    assert (pst(st, "mobile_phone_in_hand")[(t >= 2.4) & (t < 5.6)] == "true").all()
    st2, _ = run(*_cashier_with("phone_in_hand"))  # рука у лица без телефона
    assert "true" not in pst(st2, "mobile_phone_in_hand")
    assert st2[20].pred_why["mobile_phone_in_hand"] == "no_object_evidence"
    st3, _ = run(*_cashier_with("phone_in_hand", "cell_phone", static=True))  # стационарный телефон
    assert "true" not in pst(st3, "mobile_phone_in_hand")


def test_eating_needs_cup_and_valuables():
    dets, tracks, pose = _cashier_with("eating_drinking", "cup")
    st, _ = run(dets + cash(), tracks, pose)
    t = ts(st)
    assert (pst(st, "eating_drinking_with_object")[(t >= 2.4) & (t < 5.6)] == "true").all()
    assert (pst(st, "eating_at_workplace_with_valuables")[(t >= 2.4) & (t < 5.6)] == "true").all()
    st2, _ = run(*_cashier_with("eating_drinking"))
    assert "true" not in pst(st2, "eating_drinking_with_object")


def test_personal_item_on_desk_and_bag_unattended():
    tr = person_track(1, {t: SEAT for t in TIMES}, role="cashier", conf=0.9)
    item = [Detection(t=t, label="bag", bbox=(0.50, 0.42, 0.62, 0.52), score=0.7) for t in TIMES if t >= 3.0]
    st, _ = run(person_dets([tr]) + item, [tr])
    t = ts(st)
    assert (pst(st, "personal_item_on_desk")[t >= 5.2] == "true").all()
    assert (pst(st, "personal_item_on_desk")[(t > 0.8) & (t < 2.8)] == "false").all()
    bag = [Detection(t=t, label="cash_bag", bbox=(0.12, 0.42, 0.20, 0.50), score=0.7) for t in TIMES]
    st2, _ = run(person_dets([tr]) + bag, [tr])
    assert (pst(st2, "bag_left_unattended")[ts(st2) >= 3.2] == "true").all()
    held = {t: (0.12, 0.30, 0.24, 0.44) for t in TIMES}
    col = person_track(2, held, role="collector", conf=0.9)
    st3, _ = run(person_dets([tr, col]) + bag, [tr, col])
    assert "true" not in pst(st3, "bag_left_unattended")


# ------------------------------------------------------------------ replay / отсутствие данных
def test_replay_never_creates_violations_from_missing_data(tmp_path):
    class NoFrames:
        times = TIMES
        width, height, fps, duration, path = 464, 848, 30.0, DUR, "missing"

        def frame(self, i):
            raise IOError("no decoder")

    st, trace = scene_states_with_trace(NoFrames(), [], [], zones(), [])
    for s in st:
        assert "true" not in s.pred_status.values(), s.pred_status
    cov = coverage(st)
    assert cov["per_name"]["cash_left_after_leave"]["known_frac"] == 0.0
    tr = leaving_cashier()
    st2, trace2 = run(person_dets([tr]) + cash(), [tr])
    p = dump_states(st2, tmp_path / "scene_states.json", trace=trace2)
    back, _ = load_states(p)
    assert intervals(back) == intervals(st2)
    assert predicate(back[40], "cash_left_after_leave") is True
    assert predicate(back[40], "outsider_cashier_side") is not True


def test_cash_in_client_hands_after_leave_is_not_cash_left_on_desk():
    tr = leaving_cashier()
    client = person_track(2, {t: CLIENT_SEAT for t in TIMES})
    held = cash(box=(0.33, 0.24, 0.41, 0.30))  # купюры в руках клиента со стороны клиента
    st, trace = run(person_dets([tr, client]) + held, [tr, client])
    assert "true" not in pst(st, "cash_left_after_leave")


def test_h6_legacy_outsider_anchor_only_flag():
    from ml.scene.tests.synth import zones_default

    cashier = person_track(1, {t: (0.30, 0.78, 0.46, 0.92) for t in TIMES})
    tall_client = person_track(2, {t: (0.55, 0.0, 0.70, 0.50) for t in TIMES})  # опорная точка — в client_area
    dets = person_dets([cashier, tall_client])
    clip = make_clip(DUR)
    _, tr_off = scene_states_with_trace(clip, dets, [cashier, tall_client], zones_default(), [])
    _, tr_on = scene_states_with_trace(clip, dets, [cashier, tall_client], zones_default(), [], config={"presence": {"outsider_anchor_only": True}})
    # клиент со своей стороны (членство в client_area ≥ членства в зоне кассира) не вносит вклада уже по умолчанию
    # (исправление после аудита); с H6 — тоже. Остаётся только априор 0.15 кассира с неизвестной ролью.
    assert float(np.max(tr_off["raw"]["outsider_in_cashier_area"])) <= 0.15 + 1e-9
    assert float(np.max(tr_on["raw"]["outsider_in_cashier_area"])) <= 0.15 + 1e-9
