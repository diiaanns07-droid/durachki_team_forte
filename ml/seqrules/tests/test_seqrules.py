"""Синтетические регрессии ml.seqrules: только сгенерированные боксы/суставы, без реальных кадров.

Покрыто: пересечение людей, краткая потеря сустава/предмета, телефон в руке / рядом / отдельно, сейф
open/closed/unknown/не проверен, вход в зону без обвинения, посторонний предмет у кармана, VFR/PTS и
необработанные кадры, ошибка правила, нет поз, тип вне каталога, потолок кадров для UI, инвариантность
к порядку и имени файла, совместимость с гейтом fusion.
"""
from __future__ import annotations

import json
import random

import numpy as np
import pytest

from ml.seqrules import api as sq_api
from ml.seqrules import run

W, H = 480, 848                                    # портретный кадр, как у потолочной камеры
ZONE = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.62], [0.0, 0.62]]
ZONES = [{"name": "cashier_area", "polygon": ZONE, "conf": 0.9}]
CASHIER = (0.10, 0.10, 0.40, 0.55)                 # нижняя половина целиком в зоне


def rest_wrist(box):
    """Правое запястье в покое — относительно bbox своего человека."""
    return (round((box[0] + box[2]) / 2 + 0.27 * (box[2] - box[0]), 4), round(box[1] + 0.711 * (box[3] - box[1]), 4))


REST_R = rest_wrist(CASHIER)


class Clip:
    def __init__(self, times, path="synthetic.mov"):
        self.times, self.width, self.height, self.path = list(times), W, H, path


class FakeChecker:
    def __init__(self, safe="open", phone="unsure", site="open", app=None):
        self.safe, self.phone, self.site, self.app = safe, phone, site, app
        self.calls = {"safe": 0, "phone": 0, "site": 0, "app": 0}

    def site_state(self, t, box):
        self.calls["site"] += 1
        st = self.site(t) if callable(self.site) else self.site
        return {"t": round(t, 4), "state": st, "bbox": list(box)}

    def site_appearance(self, t, box):
        self.calls["app"] += 1
        return None if self.app is None else self.app(t)

    def phone_contrast(self, t, box):
        self.calls["phone"] += 1
        return {"t": t, "state": self.phone}

    def safe_state(self, t, zones):
        self.calls["safe"] += 1
        st = self.safe(t) if callable(self.safe) else self.safe
        return {"t": round(t, 4), "state": st, "bbox": [0.6, 0.1, 0.8, 0.3] if st == "open" else None}

    def summary(self):
        return {"fake": True, **self.calls}


def kp_for(box, rw=None, lw=None, wrist_vis=0.9, head_vis=0.9):
    x1, y1, x2, y2 = box
    cx, w, h = (x1 + x2) / 2, x2 - x1, y2 - y1
    pts = np.zeros((17, 3))
    for i in range(5):
        pts[i] = (cx + (i - 2) * 0.01, y1 + 0.1 * h, head_vis)
    pts[5], pts[6] = (cx - 0.25 * w, y1 + 0.25 * h, 0.9), (cx + 0.25 * w, y1 + 0.25 * h, 0.9)
    pts[7], pts[8] = (cx - 0.3 * w, y1 + 0.4 * h, 0.9), (cx + 0.3 * w, y1 + 0.4 * h, 0.9)
    pts[9] = (*(lw or (cx - 0.3 * w, y1 + 0.71 * h)), wrist_vis)
    pts[10] = (*(rw or rest_wrist(box)), wrist_vis)
    for i, fy in zip(range(11, 17), (0.6, 0.6, 0.8, 0.8, 0.98, 0.98)):
        pts[i] = (cx + (-0.15 if i % 2 else 0.15) * w, y1 + fy * h, 0.8)
    return pts.tolist()


def phone_box(c, label="cell_phone", score=0.6):
    return {"label": label, "bbox": [c[0] - 0.015, c[1] - 0.01, c[0] + 0.015, c[1] + 0.01], "score": score,
            "source": "rtdetr_v2_r18vd"}


class Scene:
    """Синтетическая сцена: треки людей с позами + предметы; кадр «обработан», если на нём есть детекции."""

    def __init__(self, times, unprocessed=()):
        self.times = [round(float(t), 4) for t in times]
        self.unprocessed = {round(float(t), 4) for t in unprocessed}
        self.persons: dict[int, dict] = {}
        self.objects: list[dict] = []

    def person(self, tid, role, box_at, kp_at=None):
        boxes, poses = [], []
        for t in self.times:
            if t in self.unprocessed:
                continue
            b = box_at(t)
            if b is None:
                continue
            boxes.append((t, tuple(b)))
            k = kp_at(t, b) if kp_at else kp_for(b)
            if k is not None:
                poses.append({"t": t, "track_id": tid, "keypoints": k})
        self.persons[tid] = {"track_id": tid, "label": "person", "role": role, "role_conf": 0.8,
                             "boxes": boxes, "_poses": poses}

    def obj(self, obj_at):
        for t in self.times:
            if t in self.unprocessed:
                continue
            o = obj_at(t)
            for d in ([o] if isinstance(o, dict) else (o or [])):
                self.objects.append({"t": t, **d})

    def run(self, checker=None, config=None, catalog=None, shuffle=False, path="synthetic.mov"):
        dets = [{"t": t, "label": "laptop", "bbox": [0.2, 0.55, 0.4, 0.6], "score": 0.9, "source": "rtdetr_v2_r18vd"}
                for t in self.times if t not in self.unprocessed]
        for p in self.persons.values():
            dets += [{"t": t, "label": "person", "bbox": list(b), "score": 0.8, "source": "rtdetr_v2_r18vd"}
                     for t, b in p["boxes"]]
        dets += self.objects
        tracks = [{k: v for k, v in p.items() if k != "_poses"} for p in self.persons.values()]
        poses = [x for p in self.persons.values() for x in p["_poses"]]
        if shuffle:
            rng = random.Random(7)
            rng.shuffle(dets)
            rng.shuffle(tracks)
            rng.shuffle(poses)
        return run(Clip(self.times, path), dets, tracks, ZONES, poses, catalog, config=config,
                   checker=checker or FakeChecker())


def grid(t0, t1, dt=0.2):
    return [round(t0 + i * dt, 4) for i in range(int(round((t1 - t0) / dt)) + 1)]


def moving_wrist(t, t0=1.0, t1=4.0):
    a = min(max((t - t0) / (t1 - t0), 0.0), 1.0)
    return (REST_R[0] - 0.03 * a, REST_R[1] - 0.12 * a)


def cashier_with_phone(sc, t0=1.0, t1=4.0, drop=(), tid=1):
    sc.person(tid, "cashier", lambda t: CASHIER, lambda t, b: kp_for(b, rw=moving_wrist(t, t0, t1)))
    sc.obj(lambda t: phone_box((moving_wrist(t, t0, t1)[0] + 0.012, moving_wrist(t, t0, t1)[1]))
           if t0 <= t <= t1 and round(t, 4) not in {round(x, 4) for x in drop} else None)


def phone_eps(obs):
    return [e for e in obs[1:] if e["rule"] == "phone"]


# ------------------------------------------------------------------ правило А

def test_phone_in_moving_hand_gives_one_event_for_holder():
    sc = Scene(grid(0, 6))
    cashier_with_phone(sc)
    events, obs = sc.run()
    assert len(events) == 1
    e = events[0]
    assert e.type_id == "V05" and e.source == "rules_seq" and e.track_ids == [1]
    assert e.t_start == pytest.approx(1.0) and e.t_end == pytest.approx(4.0) and e.t_peak == pytest.approx(1.0)
    assert e.extra["assessment"] == "candidate_violation" and e.extra["anchor_policy"] == "onset"
    assert 0.55 <= e.confidence <= 0.8 and "phone" in e.bboxes and "person" in e.bboxes
    assert "не требования норм" in e.extra["thresholds_note"]
    assert obs[0]["kind"] == "meta" and obs[0]["rules"]["phone"]["status"] == "ok"
    json.dumps(obs, ensure_ascii=False)          # наблюдения для UI сериализуемы как есть


def test_phone_lying_away_from_hands_is_not_linked():
    sc = Scene(grid(0, 6))
    sc.person(1, "cashier", lambda t: CASHIER)
    sc.obj(lambda t: phone_box((0.45, 0.58)) if 1.0 <= t <= 4.0 else None)
    events, obs = sc.run()
    assert events == [] and phone_eps(obs) == []


def test_static_phone_next_to_resting_hand_is_not_event():
    sc = Scene(grid(0, 6))
    sc.person(1, "cashier", lambda t: CASHIER)
    sc.obj(lambda t: phone_box((REST_R[0] + 0.015, REST_R[1])) if 1.0 <= t <= 4.0 else None)
    events, obs = sc.run()
    assert events == []
    assert [e["status"] for e in phone_eps(obs)] == ["static_object"]


def test_brief_loss_keeps_one_episode_long_loss_splits():
    sc = Scene(grid(0, 6))
    cashier_with_phone(sc, drop=(2.2, 2.4, 2.6))
    events, obs = sc.run()
    assert len(events) == 1 and events[0].t_start == pytest.approx(1.0) and events[0].t_end == pytest.approx(4.0)
    assert phone_eps(obs)[0]["counts"].get("absent") == 3

    sc = Scene(grid(0, 6))
    cashier_with_phone(sc, drop=(2.2, 2.4, 2.6, 2.8, 3.0, 3.2))
    events, obs = sc.run()
    assert events == [] and [e["status"] for e in phone_eps(obs)] == ["too_short", "too_short"]


def test_crossing_person_does_not_take_phone_and_order_is_not_identity():
    def visitor_box(t):                            # проходит мимо руки кассира около t≈2.3 с
        x = 0.95 - 0.3 * t
        if x + 0.15 < 0.02:
            return None
        return (max(0.0, x - 0.15), 0.12, min(1.0, x + 0.15), 0.58)

    def build():
        sc = Scene(grid(0, 6))
        cashier_with_phone(sc)
        sc.person(2, "visitor", visitor_box)
        return sc

    events, obs = build().run()
    v05 = [e for e in events if e.type_id == "V05"]
    assert len(v05) == 1 and v05[0].track_ids == [1] and v05[0].t_start == pytest.approx(1.0)
    assert not [e for e in phone_eps(obs) if e["track_id"] == 2]
    # проход посетителя через рабочую зону — отдельное наблюдение В, не нарушение
    assert all(e.extra["assessment"] == "review_observation" for e in events if e.type_id == "V01")
    ev2, _ = build().run(shuffle=True, path="другое_имя_X-99.mov")
    key = lambda es: [(e.type_id, e.t_start, e.t_end, tuple(e.track_ids), e.confidence) for e in es]
    assert key(ev2) == key(events)


def test_vfr_pts_and_unprocessed_frames_are_unknown_not_absent():
    rng = np.random.default_rng(3)
    times = np.cumsum(np.r_[0.0, rng.uniform(0.08, 0.3, 40)])
    times = [round(float(t), 4) for t in times if t <= 6.0]
    gap = [t for t in times if 1.8 <= t <= 2.4]
    assert gap, "в синтетике должен быть пропуск обработки"
    sc = Scene(times, unprocessed=gap)
    on = [t for t in times if t >= 1.0][0]
    off = [t for t in times if t <= 3.0][-1]
    cashier_with_phone(sc, t0=on, t1=off)
    events, obs = sc.run()
    assert len(events) == 1
    ep = phone_eps(obs)[0]
    assert ep["counts"].get("unknown", 0) == len(gap) and ep["counts"].get("absent", 0) == 0
    assert events[0].extra["duration_s"] == pytest.approx(off - on, abs=1e-3)   # по PTS, а не по числу кадров
    assert events[0].t_start == pytest.approx(on)


def test_unknown_object_near_pocket_and_hidden_wrists_give_no_event():
    sc = Scene(grid(0, 6))
    sc.person(1, "cashier", lambda t: CASHIER, lambda t, b: kp_for(b, rw=moving_wrist(t)))
    sc.obj(lambda t: phone_box((moving_wrist(t)[0] + 0.012, moving_wrist(t)[1]), label="keys")
           if 1.0 <= t <= 4.0 else None)
    events, obs = sc.run()
    assert events == [] and phone_eps(obs) == []

    sc = Scene(grid(0, 6))
    sc.person(1, "cashier", lambda t: CASHIER, lambda t, b: kp_for(b, wrist_vis=0.1, head_vis=0.1))
    sc.obj(lambda t: phone_box((0.30, 0.52)) if 1.0 <= t <= 4.0 else None)    # у кармана, суставов не видно
    events, obs = sc.run()
    assert events == [] and phone_eps(obs) == []


def test_contrast_check_rejects_desk_object_and_confirms_mobile():
    sc = Scene(grid(0, 6))
    cashier_with_phone(sc)
    events, obs = sc.run(checker=FakeChecker(phone="not_mobile"))
    assert events == [] and phone_eps(obs)[0]["status"] == "rejected_not_mobile"
    base, _ = sc.run(checker=FakeChecker(phone="unsure"))
    plus, _ = sc.run(checker=FakeChecker(phone="mobile"))
    assert plus[0].confidence > base[0].confidence


def test_repeats_of_same_person_are_merged():
    sc = Scene(grid(0, 8))
    sc.person(1, "cashier", lambda t: CASHIER,
              lambda t, b: kp_for(b, rw=moving_wrist(t, 1.0, 3.0) if t <= 3.0 else moving_wrist(t, 4.4, 6.4)))

    def ph(t):
        if 1.0 <= t <= 3.0:
            w = moving_wrist(t, 1.0, 3.0)
        elif 4.4 <= t <= 6.4:
            w = moving_wrist(t, 4.4, 6.4)
        else:
            return None
        return phone_box((w[0] + 0.012, w[1]))

    sc.obj(ph)
    events, obs = sc.run()
    assert len(phone_eps(obs)) == 2 and len(events) == 1
    assert events[0].t_start == pytest.approx(1.0) and events[0].t_end == pytest.approx(6.4)
    assert len(events[0].extra["merged_episodes"]) == 2


# ------------------------------------------------------------------ правило Б

def _safe_scene(after="leave", tail=12.0):
    sc = Scene(grid(0, tail))

    def cashier(t):
        if t <= 3.0:
            return CASHIER
        if after == "lost":
            return None
        if after == "nearby":
            return (0.55, 0.15, 0.85, 0.6)
        if t <= 3.8:
            x = 0.55 + (t - 3.2) * 0.3
            return (x, 0.15, x + 0.3, 0.6)
        return None

    sc.person(1, "cashier", cashier)
    return sc


def safe_eps(obs):
    return [e for e in obs[1:] if e["rule"] == "safe"]


def test_safe_open_after_observed_exit_gives_event_with_onset():
    ck = FakeChecker(safe="open")
    events, obs = _safe_scene().run(checker=ck)
    ev = [e for e in events if e.type_id == "V03"]
    assert len(ev) == 1 and ev[0].t_start == pytest.approx(3.2) and ev[0].extra["absence"] == "exit_observed"
    assert ev[0].extra["assessment"] == "candidate_violation" and ck.calls["safe"] == 4
    assert ev[0].confidence == pytest.approx(0.75)


def test_safe_closed_and_unknown_and_unchecked_give_no_event():
    _ev, obs = _safe_scene().run(checker=FakeChecker(safe="closed"))
    assert [e["status"] for e in safe_eps(obs)] == ["safe_closed"]
    events, obs = _safe_scene().run(checker=FakeChecker(safe="unknown"))
    assert not [e for e in events if e.type_id == "V03"] and safe_eps(obs)[0]["status"] == "safe_unknown"
    assert obs[0]["rules"]["safe"]["checked_intervals"]
    events, obs = _safe_scene().run(checker=FakeChecker(safe="skipped"))
    assert safe_eps(obs)[0]["status"] == "not_checked"
    assert obs[0]["rules"]["safe"]["unchecked_intervals"] and not obs[0]["rules"]["safe"]["checked_intervals"]


def test_safe_onset_is_start_of_last_open_run():
    seq = iter(["open", "closed", "open", "open"])
    events, _ = _safe_scene().run(checker=FakeChecker(safe=lambda t: next(seq)))
    ev = [e for e in events if e.type_id == "V03"]
    assert len(ev) == 1 and ev[0].t_start == pytest.approx(8.2)


def test_lost_detection_inside_zone_is_not_leaving():
    ck = FakeChecker(safe="open")
    events, obs = _safe_scene(after="lost").run(checker=ck)
    assert not [e for e in events if e.type_id == "V03"] and ck.calls["safe"] == 0
    assert safe_eps(obs)[0]["status"] == "uncertain" and safe_eps(obs)[0]["absence"] == "lost_in_zone"


def test_staff_nearby_outside_zone_is_not_unattended():
    ck = FakeChecker(safe="open")
    events, obs = _safe_scene(after="nearby").run(checker=ck)
    assert not [e for e in events if e.type_id == "V03"] and ck.calls["safe"] == 0
    assert safe_eps(obs)[0]["absence"] == "nearby"


def test_zone_empty_from_start_is_uncertain():
    sc = Scene(grid(0, 8))
    sc.person(1, "visitor", lambda t: (0.6, 0.1, 0.9, 0.6))
    ck = FakeChecker(safe="open")
    events, obs = sc.run(checker=ck)
    assert not [e for e in events if e.type_id == "V03"] and ck.calls["safe"] == 0
    assert safe_eps(obs)[0]["absence"] == "empty_from_start"


# ------------------------------------------------------------------ правило В

def test_zone_entry_is_observation_not_violation():
    from ml.fusion.core import upstream_assessment

    sc = Scene(grid(0, 6))
    sc.person(2, "visitor", lambda t: (0.6, 0.15, 0.9, 0.6) if t < 1.2 else (0.15, 0.15, 0.45, 0.6))
    events, obs = sc.run()
    ev = [e for e in events if e.type_id == "V01"]
    assert len(ev) == 1 and ev[0].t_start == pytest.approx(1.2) and ev[0].confidence == pytest.approx(0.35)
    assert ev[0].extra["assessment"] == "review_observation"
    assert upstream_assessment(ev[0].confidence, ev[0].extra, {"report_gate": {}})[0] == "observation"
    assert any("не вывод о нарушении" in f for f in ev[0].facts)


def test_entry_not_observed_for_tracks_starting_inside_staff_or_short_stay():
    sc = Scene(grid(0, 6))
    sc.person(2, "visitor", lambda t: (0.15, 0.15, 0.45, 0.6))                       # с начала внутри
    sc.person(3, "visitor", lambda t: (0.6, 0.15, 0.9, 0.6) if not 2.0 <= t <= 2.8 else (0.15, 0.15, 0.45, 0.6))
    sc.person(4, "cashier", lambda t: (0.6, 0.15, 0.9, 0.6) if t < 1.0 else CASHIER)  # сотрудник
    events, obs = sc.run()
    assert not [e for e in events if e.type_id == "V01"]
    st = {e["track_id"]: e["status"] for e in obs[1:] if e["rule"] == "entry"}
    assert st == {2: "no_entry_observed", 3: "too_short"}


# ------------------------------------------------------------------ устойчивость и статусы

def test_rule_error_is_isolated_and_reported(monkeypatch):
    def boom(ctx):
        raise RuntimeError("синтетический сбой")

    rules = tuple((n, boom if n == "phone" else f) for n, f in sq_api.RULES)
    monkeypatch.setattr(sq_api, "RULES", rules)
    sc = Scene(grid(0, 6))
    cashier_with_phone(sc)
    events, obs = sc.run()
    assert obs[0]["rules"]["phone"]["status"] == "error" and obs[0]["rules"]["entry"]["status"] == "ok"
    assert any("seqrules/phone: ошибка" in w for w in obs[0]["warnings"])


def test_no_poses_makes_rule_a_unavailable_not_negative():
    sc = Scene(grid(0, 6))
    sc.person(1, "cashier", lambda t: CASHIER, lambda t, b: None)
    sc.obj(lambda t: phone_box((REST_R[0] + 0.01, REST_R[1])) if 1.0 <= t <= 4.0 else None)
    events, obs = sc.run()
    assert events == [] and obs[0]["rules"]["phone"]["status"] == "unavailable"
    assert any("seqrules/phone: unavailable" in w for w in obs[0]["warnings"])


def test_type_missing_from_catalog_disables_rule():
    sc = Scene(grid(0, 6))
    cashier_with_phone(sc)
    events, obs = sc.run(catalog={"types": {"V01": {}, "V03": {}}})
    assert events == [] and obs[0]["rules"]["phone"]["status"] == "unavailable"


def test_observation_frames_are_capped_for_ui():
    sc = Scene(grid(0, 6))
    cashier_with_phone(sc)
    _ev, obs = sc.run(config={"max_observation_frames": 10})
    assert sum(len(e.get("frames", [])) for e in obs[1:]) <= 10


def test_no_work_zone_rules_b_c_unavailable():
    sc = Scene(grid(0, 6))
    cashier_with_phone(sc)
    dets_zones = []
    events, obs = run(Clip(sc.times), [], [], dets_zones, [], None, checker=FakeChecker())
    assert obs[0]["rules"]["safe"]["status"] == "unavailable" and obs[0]["rules"]["entry"]["status"] == "unavailable"
    assert any("cashier_area не построена" in w for w in obs[0]["warnings"])


# ------------------------------------------------------------------ правило Б2: место хранения по рукам (два прохода)

DESK = [[0.1, 0.56], [0.45, 0.56], [0.45, 0.7], [0.1, 0.7]]
CLIENT = [[0.0, 0.72], [0.55, 0.72], [0.55, 1.0], [0.0, 1.0]]
STORAGE_ZONES = ZONES + [{"name": "desk", "polygon": DESK, "conf": 0.9}, {"name": "client_area", "polygon": CLIENT, "conf": 0.9}]
SIDE = (0.583, 0.2125)          # ≈ 2,2 ширины плеч вправо от центра плеч кассира, вне стола и зоны клиента
ON = {"storage": {"enabled": True}}
_rng = np.random.default_rng(5)
A0, A1 = _rng.standard_normal(1024), _rng.standard_normal(1024)


def _norm(v):
    return (v - v.mean()) / v.std()


def app_changed(t):             # вид места после касания другой (не вернулся к исходному)
    return _norm(A0) if t < 2.0 else _norm(A1)


def app_same(t):
    return _norm(A0)


REST_DESK_R, REST_DESK_L = (0.28, 0.3475), (0.22, 0.3475)     # руки в покое — на столе перед собой (≈1,6 ширины плеч)


def storage_scene(reach=(2.0, 3.0), target=SIDE, role="cashier", wrist_vis=0.9, tail=12.0):
    sc = Scene(grid(0, tail))
    sc.person(1, role, lambda t: CASHIER,
              lambda t, b: kp_for(b, rw=target if reach[0] <= t <= reach[1] else REST_DESK_R, lw=REST_DESK_L,
                                  wrist_vis=wrist_vis))
    return sc


def run_storage(sc, ck):
    dets_zones = STORAGE_ZONES
    times = sc.times
    dets = [{"t": t, "label": "laptop", "bbox": [0.2, 0.57, 0.4, 0.62], "score": 0.9, "source": "rtdetr_v2_r18vd"} for t in times]
    tracks = [{k: v for k, v in p.items() if k != "_poses"} for p in sc.persons.values()]
    for p in sc.persons.values():
        dets += [{"t": t, "label": "person", "bbox": list(b), "score": 0.8, "source": "rtdetr_v2_r18vd"} for t, b in p["boxes"]]
    poses = [x for p in sc.persons.values() for x in p["_poses"]]
    return run(Clip(times), dets, tracks, dets_zones, poses, None, config=ON, checker=ck)


def storage_eps(obs):
    return [e for e in obs[1:] if e["rule"] == "storage"]


def test_storage_left_open_after_reach_gives_event_at_hand_leave():
    ck = FakeChecker(site="open", app=app_changed)
    events, obs = run_storage(storage_scene(), ck)
    ev = [e for e in events if e.extra.get("rule") == "B2_storage"]
    assert len(ev) == 1 and ev[0].type_id == "V03" and ev[0].t_start == pytest.approx(3.0)
    assert ev[0].extra["assessment"] == "candidate_violation" and ev[0].track_ids == [1]
    assert any("не утверждает" in f for f in ev[0].facts)
    ep = storage_eps(obs)[0]
    assert ep["status"] == "event" and ep["appearance_baseline"] is True and ck.calls["site"] >= 2
    assert all(c["t"] > 3.0 for c in ep["safe_checks"])              # проверки — только после ухода руки


def test_storage_closed_or_reverted_gives_no_event():
    _ev, obs = run_storage(storage_scene(), FakeChecker(site="closed", app=app_changed))
    assert storage_eps(obs)[0]["status"] == "closed"
    events, obs = run_storage(storage_scene(), FakeChecker(site="open", app=app_same))
    assert not [e for e in events if e.extra.get("rule") == "B2_storage"]
    assert storage_eps(obs)[0]["status"] == "reverted"


def test_reach_into_desk_or_by_visitor_or_hidden_is_not_a_site():
    for sc in (storage_scene(target=(0.3, 0.62)),                    # тянется в стол — обычная работа
               storage_scene(role="visitor"),                          # не сотрудник
               storage_scene(wrist_vis=0.1)):                          # запястье не видно — вывода нет
        events, obs = run_storage(sc, FakeChecker(site="open", app=app_changed))
        assert not [e for e in events if e.extra.get("rule") == "B2_storage"] and storage_eps(obs) == []


def test_without_baseline_view_one_more_open_check_is_needed():
    seq = ["open", "open", "closed", "closed"]
    it = iter(seq)
    events, obs = run_storage(storage_scene(reach=(0.2, 1.2)), FakeChecker(site=lambda t: next(it), app=app_changed))
    assert storage_eps(obs)[0]["appearance_baseline"] is False
    assert not [e for e in events if e.extra.get("rule") == "B2_storage"]
    events, obs = run_storage(storage_scene(reach=(0.2, 1.2)), FakeChecker(site="open", app=app_changed))
    assert [e for e in events if e.extra.get("rule") == "B2_storage"]


def test_storage_disabled_by_default():
    sc = storage_scene()
    _ev, obs = sc.run()
    assert obs[0]["rules"]["storage"]["status"] == "disabled"


def test_hanging_arms_posture_is_not_a_reach():
    # стоящий человек с опущенными руками: запястья далеко от плеч весь ролик — это поза, а не движение к месту
    sc = Scene(grid(0, 12))
    sc.person(1, "cashier", lambda t: CASHIER, lambda t, b: kp_for(b))
    events, obs = run_storage(sc, FakeChecker(site="open", app=app_changed))
    assert storage_eps(obs) == [] and obs[0]["rules"]["storage"]["reaches"] == 0
