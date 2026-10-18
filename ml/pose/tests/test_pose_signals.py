"""pose_signals на синтетических позах: позитивы/негативы + семантика unknown."""
from __future__ import annotations

import math

import numpy as np
import pytest

from ml.pose import pose_signals
from ml.pose._contract import SIGNALS, SceneState, Zone
from ml.pose.config import load_config
from ml.pose.signals import compute_pose_signals

from .synth_pose import (L_WR, R_EL, R_WR, desk_zone, object_track, person_track, poses_for, times, values)

BOX = (0.30, 0.30, 0.60, 0.70)          # кассир
_DEFAULT_ENABLED = load_config()["signals"]["enabled"]


def _cfg_with(*names, **extras):
    """Конфиг с включёнными opt-in сигналами (по умолчанию выключены как хрупкие)."""
    return load_config({"signals": {"enabled": list(_DEFAULT_ENABLED) + list(names), "extras": extras}})


def _sig(poses, tracks, zones, *names, **extras):
    return compute_pose_signals(poses, tracks, zones, _cfg_with(*names, **extras))
HEAD_BOX = (0.42, 0.32, 0.48, 0.37)     # bbox «телефона» у головы (норм.)
EAR_R = {R_WR: (0.43, 0.14), R_EL: (0.28, 0.24)}


def _phone_fn(t0: float, t1: float):
    return lambda t: (EAR_R if t0 <= t <= t1 else None, None)


def test_output_types_and_signal_names():
    ts = times(4)
    st = pose_signals(poses_for(1, BOX, ts), [person_track(1, "cashier", BOX, ts)], [desk_zone()])
    assert len(st) == len(ts)
    for s in st:
        assert isinstance(s, SceneState)
        assert set(s.signals) <= set(SIGNALS)
        assert all(0.0 <= v <= 1.0 for v in s.signals.values())


def test_phone_to_ear_with_phone_object_positive():
    ts = times(10)
    tracks = [person_track(1, "cashier", BOX, ts), object_track(50, "cell_phone", HEAD_BOX, [t for t in ts if 2 <= t <= 6])]
    st = _sig(poses_for(1, BOX, ts, _phone_fn(2.0, 6.0)), tracks, [desk_zone()], "phone_to_ear")
    inside = values(st, "phone_to_ear", 2.4, 5.6)
    assert inside and all(v is not None and v >= 0.5 for v in inside)
    before = values(st, "phone_to_ear", 0.0, 1.4)
    assert all(v is not None and v < 0.5 for v in before)  # обе кисти видны и далеко — наблюдаемое «нет»
    s = min(st, key=lambda x: abs(x.t - 4.0))
    assert s.extra["track_ids"]["phone_to_ear"] == [1]
    assert s.extra["boxes"]["phone_to_ear"] == BOX


def test_hand_at_ear_without_phone_is_unknown_not_violation():
    ts = times(10)
    st = _sig(poses_for(1, BOX, ts, _phone_fn(2.0, 6.0)), [person_track(1, "cashier", BOX, ts)], [desk_zone()],
              "phone_to_ear")
    inside = [s for s in st if 2.4 <= s.t <= 5.6]
    assert all("phone_to_ear" not in s.signals for s in inside)
    assert all(s.extra["gestures"].get("hand_at_head", 0) >= 0.5 for s in inside)
    assert all("phone_to_ear" in s.extra["unknown"] for s in inside)


def test_phone_to_ear_too_short_not_active():
    ts = times(8)
    tracks = [person_track(1, "cashier", BOX, ts), object_track(50, "cell_phone", HEAD_BOX, ts)]
    st = _sig(poses_for(1, BOX, ts, _phone_fn(3.0, 3.8)), tracks, [desk_zone()], "phone_to_ear")
    assert all((v or 0.0) < 0.5 for v in values(st, "phone_to_ear", 0, 8))


def test_phone_to_ear_only_for_cashier_role():
    ts = times(10)
    tracks = [person_track(2, "visitor", BOX, ts), object_track(50, "cell_phone", HEAD_BOX, ts)]
    st = _sig(poses_for(2, BOX, ts, _phone_fn(2.0, 6.0)), tracks, [desk_zone()], "phone_to_ear")
    assert all("phone_to_ear" not in s.signals for s in st)  # посетитель с телефоном — не сигнал кассира
    s = min(st, key=lambda x: abs(x.t - 4.0))
    assert s.extra["by_track"][2]["signals"]["phone_to_ear"] >= 0.5


def test_phone_in_hand_requires_phone_track():
    ts = times(6)
    hand_box = (0.40, 0.50, 0.46, 0.55)  # у правого запястья (u=0.38, v=0.55)
    poses = poses_for(1, BOX, ts)
    st_no = pose_signals(poses, [person_track(1, "cashier", BOX, ts)], [])
    assert all("phone_in_hand" not in s.signals for s in st_no)
    st = pose_signals(poses, [person_track(1, "cashier", BOX, ts), object_track(9, "cell_phone", hand_box, ts)], [])
    assert all(v is not None and v >= 0.5 for v in values(st, "phone_in_hand", 0.4, 5.4))


def _eat_fn(t0: float, t1: float, period: float = 4.0, dwell: float = 0.6):
    def fn(t):
        if t0 <= t <= t1 and ((t - t0) % period) < dwell:
            return {R_WR: (0.49, 0.15), R_EL: (0.30, 0.33)}, None
        return None, None
    return fn


def _cup_follows_hand(t0: float, t1: float, ts, period: float = 4.0, dwell: float = 0.6):
    """Стакан у рта, пока кисть у рта; иначе — на столе."""
    from ml.pose._contract import Track
    at_mouth, on_desk = (0.43, 0.33, 0.49, 0.38), (0.40, 0.50, 0.46, 0.56)
    return Track(track_id=7, label="cup", boxes=[
        (t, at_mouth if (t0 <= t <= t1 and ((t - t0) % period) < dwell) else on_desk) for t in ts])


def test_eating_with_cup_positive():
    ts = times(20)
    tracks = [person_track(1, "cashier", BOX, ts), _cup_follows_hand(4.0, 16.0, ts)]
    st = pose_signals(poses_for(1, BOX, ts, _eat_fn(4.0, 16.0)), tracks, [desk_zone()])
    vals = [v for v in values(st, "eating_drinking", 4.0, 16.0) if v is not None]
    assert vals and max(vals) >= 0.5
    assert all((v or 0) < 0.5 for v in values(st, "eating_drinking", 0.0, 2.0))
    s = next(x for x in st if (x.signals.get("eating_drinking") or 0) >= 0.5)
    run = s.extra["runs"]["eating_drinking"]
    assert 3.4 <= run["t_start"] <= 4.0           # начало перехода: кисть пошла ко рту
    assert 4.0 <= run["t_peak"] <= 4.6            # первый контакт кисти со ртом


def test_cup_on_desk_near_hand_is_not_drinking():
    """Стакан стоит на столе у руки, кисть ходит ко рту без стакана — не питьё (предмет не у рта)."""
    ts = times(20)
    cup_on_desk = (0.40, 0.50, 0.46, 0.56)
    tracks = [person_track(1, "cashier", BOX, ts), object_track(7, "cup", cup_on_desk, ts)]
    st = pose_signals(poses_for(1, BOX, ts, _eat_fn(4.0, 16.0)), tracks, [desk_zone()])
    assert all((v or 0.0) < 0.5 for v in values(st, "eating_drinking", 0, 20))


def test_hand_to_mouth_without_food_is_unknown():
    ts = times(20)
    st = pose_signals(poses_for(1, BOX, ts, _eat_fn(4.0, 16.0)), [person_track(1, "cashier", BOX, ts)], [desk_zone()])
    assert all((v or 0.0) < 0.5 for v in values(st, "eating_drinking", 0, 20))
    mid = [s for s in st if 4.0 <= s.t <= 12.6]  # отрезок повторов «кисть ко рту» (3 цикла)
    assert all("eating_drinking" not in s.signals for s in mid)
    assert any(s.extra["gestures"].get("hand_to_mouth", 0) > 0 for s in mid)


def test_money_counting_is_not_eating():
    """Быстрые мелкие движения кистей над столом + наличные и стакан рядом → не eating_drinking."""
    ts = times(20)
    cash = (0.38, 0.52, 0.52, 0.60)
    cup = (0.55, 0.52, 0.60, 0.58)

    def counting(t):
        dx = 0.03 * math.sin(2 * math.pi * 2.0 * t)
        return {L_WR: (0.56 + dx, 0.56), R_WR: (0.44 - dx, 0.56 + 0.5 * dx)}, None

    tracks = [person_track(1, "cashier", BOX, ts), object_track(3, "cash", cash, ts), object_track(4, "cup", cup, ts)]
    st = pose_signals(poses_for(1, BOX, ts, counting, jitter=0.003, seed=1), tracks, [desk_zone()])
    vals = values(st, "eating_drinking", 0, 20)
    assert all(v is not None and v < 0.5 for v in vals)


def test_head_bending_to_hands_is_not_eating():
    """Голова наклоняется к рукам (кисти неподвижны) — не «кисть ко рту», даже со стаканом рядом."""
    ts = times(20)
    cup = (0.44, 0.50, 0.50, 0.56)

    def bend(t):
        if (t % 4.0) < 1.0:  # голова опускается к кистям
            return {0: (0.48, 0.50), 1: (0.50, 0.48), 2: (0.46, 0.48), 3: (0.53, 0.49), 4: (0.43, 0.49)}, None
        return None, None

    tracks = [person_track(1, "cashier", BOX, ts), object_track(4, "cup", cup, ts)]
    st = pose_signals(poses_for(1, BOX, ts, bend), tracks, [desk_zone()])
    assert all((v or 0.0) < 0.5 for v in values(st, "eating_drinking", 0, 20))


def test_hidden_wrists_are_unknown_not_hands_hidden():
    ts = times(10)

    def hidden(t):
        return None, ({L_WR: 0.05, R_WR: 0.05} if 2 <= t <= 8 else None)

    st = pose_signals(poses_for(1, BOX, ts, hidden), [person_track(1, "cashier", BOX, ts)], [desk_zone()])
    mid = [s for s in st if 2.0 <= s.t <= 8.0]
    assert all("hands_hidden" not in s.signals for s in mid)              # unknown, не нарушение
    assert all(s.extra["gestures"].get("hands_not_visible") == 1.0 for s in mid)
    assert all(v is not None and v < 0.5 for v in values(st, "hands_hidden", 0, 1.6))  # кисти видны


def test_hands_hidden_positive_only_when_explicitly_enabled():
    ts = times(10)

    def hidden(t):
        return None, ({L_WR: 0.05, R_WR: 0.05} if 2 <= t <= 8 else None)

    cfg = load_config({"signals": {"hands_hidden": {"allow_positive_from_pose": True}}})
    st = compute_pose_signals(poses_for(1, BOX, ts, hidden), [person_track(1, "cashier", BOX, ts)], [desk_zone()], cfg)
    assert all(v is not None and v >= 0.5 for v in values(st, "hands_hidden", 2.4, 7.6))


def _handover_poses(t0, t1, ts, reach: bool = True):
    cashier_box = (0.10, 0.30, 0.40, 0.70)
    visitor_box = (0.55, 0.30, 0.85, 0.70)

    def cashier(t):
        if reach and t0 <= t <= t1:
            return {L_WR: (1.12, 0.85)}, None   # тянется вправо над столом, к посетителю
        return None, None

    def visitor(t):
        if reach and t0 <= t <= t1:
            return {R_WR: (-0.22, 0.85)}, None  # тянется влево над столом
        return None, None

    poses = poses_for(1, cashier_box, ts, cashier) + poses_for(2, visitor_box, ts, visitor)
    tracks = [person_track(1, "cashier", cashier_box, ts), person_track(2, "visitor", visitor_box, ts)]
    return poses, tracks


def test_handover_motion_positive():
    ts = times(8)
    poses, tracks = _handover_poses(3.0, 4.2, ts)
    st = pose_signals(poses, tracks, [desk_zone()])
    # первые 1–2 кадра — лаг One-Euro после резкого движения
    assert all(v is not None and v >= 0.5 for v in values(st, "handover_motion", 3.4, 4.2))
    assert all(v is not None and v < 0.5 for v in values(st, "handover_motion", 0.0, 2.0))
    s = min(st, key=lambda x: abs(x.t - 3.6))
    assert s.extra["track_ids"]["handover_motion"] == [1, 2]
    assert s.extra["handover_pairs"][0][2:4] == ["cashier", "visitor"]


def test_handover_negative_when_not_reaching():
    ts = times(8)
    poses, tracks = _handover_poses(3.0, 4.2, ts, reach=False)
    st = pose_signals(poses, tracks, [desk_zone()])
    assert all((v or 0.0) < 0.5 for v in values(st, "handover_motion", 0, 8))


def test_facing_away_with_desk_zone():
    ts = times(10)
    back = None

    def turned(t):
        if 3.0 <= t <= 7.0:  # повернулся спиной к столу: плечи «зеркально»
            return {5: (0.32, 0.25), 6: (0.68, 0.25), 7: (0.26, 0.42), 8: (0.74, 0.42)}, None
        return None, None

    tracks = [person_track(1, "cashier", BOX, ts)]
    poses = poses_for(1, BOX, ts, turned)
    st = _sig(poses, tracks, [desk_zone()], "cashier_facing_away")  # стол ниже человека, лицом к нему = «вниз»
    assert all(v is not None and v >= 0.5 for v in values(st, "cashier_facing_away", 3.4, 6.6))
    assert all(v is not None and v < 0.5 for v in values(st, "cashier_facing_away", 0, 2.4))
    st_nozone = _sig(poses, tracks, [], "cashier_facing_away")
    assert all("cashier_facing_away" not in s.signals for s in st_nozone)  # нет зоны — unknown
    st_default = pose_signals(poses, tracks, [desk_zone()])
    assert all("cashier_facing_away" not in s.signals for s in st_default)  # по умолчанию выключен
    assert back is None


def test_low_confidence_everything_unknown():
    ts = times(6)
    low = {k: 0.1 for k in range(17)}
    st = pose_signals(poses_for(1, BOX, ts, lambda t: (None, low)), [person_track(1, "cashier", BOX, ts)], [desk_zone()])
    for s in st:
        assert not s.signals, s.signals


def test_no_poses_gives_unknown_states_for_track_times():
    ts = times(3)
    st = pose_signals([], [person_track(1, "cashier", BOX, ts)], [])
    assert len(st) == len(ts)
    assert all(not s.signals and s.extra["n_posed"] == 0 for s in st)


def test_bending_to_safe_extra():
    ts = times(6)
    safe = Zone(name="safe", polygon=[(0.30, 0.48), (0.42, 0.48), (0.42, 0.58), (0.30, 0.58)])
    st = _sig(poses_for(1, BOX, ts), [person_track(1, "cashier", BOX, ts)], [safe], bending_to_safe=True)
    vals = [s.extra["bending_to_safe"] for s in st if 0.4 <= s.t <= 5.4]
    assert all(v is not None and v >= 0.5 for v in vals)
    st2 = _sig(poses_for(1, BOX, ts), [person_track(1, "cashier", BOX, ts)], [], bending_to_safe=True)
    assert all(s.extra["bending_to_safe"] is None for s in st2)
    st3 = pose_signals(poses_for(1, BOX, ts), [person_track(1, "cashier", BOX, ts)], [safe])
    assert all(s.extra["bending_to_safe"] is None for s in st3)  # по умолчанию выключен


def test_pseudo_ids_for_untracked_poses():
    ts = times(4)
    poses = poses_for(1, BOX, ts)
    for p in poses:
        p.track_id = None
    st = pose_signals(poses, [], [])
    assert len(st) == len(ts)
    assert all(len(s.extra["by_track"]) == 1 for s in st)
    assert all(next(iter(s.extra["by_track"])) < 0 for s in st)


@pytest.mark.parametrize("aspect_wh", [(464, 848), (1280, 720)])
def test_frame_aspect_from_extra(aspect_wh):
    from ml.pose._contract import PoseFrameX
    ts = times(4)
    poses = [PoseFrameX(t=p.t, track_id=p.track_id, keypoints=p.keypoints, extra={"frame_wh": aspect_wh})
             for p in poses_for(1, BOX, ts)]
    st = pose_signals(poses, [person_track(1, "cashier", BOX, ts)], [])
    assert len(st) == len(ts)
    assert np.isfinite([v for s in st for v in s.signals.values()]).all()


def test_default_signal_set_is_small():
    assert set(_DEFAULT_ENABLED) == {"phone_in_hand", "eating_drinking", "hands_hidden", "handover_motion"}
    ts = times(10)
    tracks = [person_track(1, "cashier", BOX, ts), object_track(50, "cell_phone", HEAD_BOX, ts)]
    st = pose_signals(poses_for(1, BOX, ts, _phone_fn(2.0, 6.0)), tracks, [desk_zone()])
    assert all("phone_to_ear" not in s.signals and "phone_to_ear" not in s.extra["unknown"] for s in st)


def test_adjusting_hair_with_cup_and_phone_on_desk_is_nothing():
    """Поправила волосы (кисть у макушки/виска ~1.2 с, дважды); стакан и телефон лежат на столе."""
    ts = times(16)

    def hair(t):
        if 3.0 <= t <= 4.2 or 9.0 <= t <= 10.2:
            return {R_WR: (0.40, 0.04), R_EL: (0.26, 0.20)}, None
        return None, None

    tracks = [person_track(1, "cashier", BOX, ts), object_track(7, "cup", (0.40, 0.50, 0.46, 0.56), ts),
              object_track(8, "cell_phone", (0.52, 0.60, 0.58, 0.64), ts)]
    st = _sig(poses_for(1, BOX, ts, hair), tracks, [desk_zone()], "phone_to_ear")
    for name in ("eating_drinking", "phone_to_ear", "phone_in_hand"):
        assert all((v or 0.0) < 0.5 for v in values(st, name, 0, 16)), name


def test_scratching_ear_with_phone_on_desk_is_not_phone():
    """Почесала ухо 2 с (дольше порога 1.5 с), телефон лежит на столе далеко от головы и рук."""
    ts = times(10)

    def scratch(t):
        if 3.0 <= t <= 5.0:
            jig = 0.01 * ((int(t * 5) % 2) * 2 - 1)
            return {R_WR: (0.42 + jig, 0.12), R_EL: (0.28, 0.24)}, None
        return None, None

    tracks = [person_track(1, "cashier", BOX, ts), object_track(8, "cell_phone", (0.52, 0.66, 0.58, 0.70), ts)]
    st = _sig(poses_for(1, BOX, ts, scratch), tracks, [desk_zone()], "phone_to_ear")
    assert all((v or 0.0) < 0.5 for v in values(st, "phone_to_ear", 0, 10))
    assert all((v or 0.0) < 0.5 for v in values(st, "phone_in_hand", 0, 10))
    mid = [s for s in st if 3.4 <= s.t <= 4.6]
    assert all("phone_to_ear" not in s.signals for s in mid)   # жест без телефона у головы — unknown
    assert any(s.extra["gestures"].get("hand_at_head", 0) >= 0.5 for s in mid)


def test_handover_run_onset():
    ts = times(8)
    poses, tracks = _handover_poses(3.0, 4.2, ts)
    st = pose_signals(poses, tracks, [desk_zone()])
    s = min(st, key=lambda x: abs(x.t - 3.8))
    run = s.extra["runs"]["handover_motion"]
    assert 3.0 <= run["t_start"] <= 3.4 and run["t_end"] >= 4.0
