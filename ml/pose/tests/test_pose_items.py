"""item_placed_on_desk (extra, opt-in): кто и когда положил предмет на стол — позитив и негативы."""
from __future__ import annotations

from ml.pose._contract import Track, Zone
from ml.pose.config import load_config
from ml.pose.signals import compute_pose_signals

from .synth_pose import R_WR, desk_zone, person_track, poses_for, times

BOX = (0.30, 0.30, 0.60, 0.70)
SPOT = (0.43, 0.64, 0.47, 0.68)                 # на столе (зона desk: y 0.62..0.80)


def _cfg(on: bool = True):
    return load_config({"signals": {"extras": {"item_placed_on_desk": on}}})


def _reach(t0: float, t1: float):
    return lambda t: ({R_WR: (0.50, 0.875)} if t0 <= t <= t1 else None, None)  # кисть у SPOT


def _obj(tid: int, label: str, box, ts) -> Track:
    return Track(track_id=tid, label=label, boxes=[(t, box) for t in ts])


def _placed(states):
    return [s for s in states if (s.extra.get("item_placed_on_desk") or 0) >= 0.5]


def test_item_placed_positive_with_onset():
    ts = times(12)
    tracks = [person_track(1, "cashier", BOX, ts), _obj(20, "bag", SPOT, [t for t in ts if t >= 4.4])]
    st = compute_pose_signals(poses_for(1, BOX, ts, _reach(4.0, 5.4)), tracks, [desk_zone()], _cfg())
    hit = _placed(st)
    assert hit, "событие не найдено"
    ev = hit[0].extra["placements"][0]
    assert ev["person_track_id"] == 1 and ev["role"] == "cashier" and ev["label"] == "bag"
    assert 3.8 <= ev["t_start"] <= 4.4            # кисть подошла к месту
    assert abs(ev["t_settle"] - 4.4) < 1e-6        # предмет «лёг»
    assert all((s.extra["item_placed_on_desk"] or 0) < 0.5 for s in st if s.t < 3.6)


def test_preexisting_item_is_not_placement():
    ts = times(12)
    tracks = [person_track(1, "cashier", BOX, ts), _obj(20, "bag", SPOT, ts)]
    st = compute_pose_signals(poses_for(1, BOX, ts, _reach(4.0, 5.4)), tracks, [desk_zone()], _cfg())
    assert not _placed(st)


def test_item_appears_without_hand_is_not_placement():
    ts = times(12)
    tracks = [person_track(1, "cashier", BOX, ts), _obj(20, "bag", SPOT, [t for t in ts if t >= 4.4])]
    st = compute_pose_signals(poses_for(1, BOX, ts), tracks, [desk_zone()], _cfg())
    assert not _placed(st)


def test_item_outside_desk_is_not_placement():
    ts = times(12)
    off_desk = (0.43, 0.40, 0.47, 0.44)
    tracks = [person_track(1, "cashier", BOX, ts), _obj(20, "bag", off_desk, [t for t in ts if t >= 4.4])]
    st = compute_pose_signals(poses_for(1, BOX, ts, lambda t: ({R_WR: (0.50, 0.30)} if 4.0 <= t <= 5.4 else None, None)),
                              tracks, [desk_zone()], _cfg())
    assert not _placed(st)


def test_track_id_switch_is_not_placement():
    ts = times(12)
    tracks = [person_track(1, "cashier", BOX, ts), _obj(20, "bag", SPOT, [t for t in ts if t < 6.0]),
              _obj(21, "bag", SPOT, [t for t in ts if t >= 6.4])]
    st = compute_pose_signals(poses_for(1, BOX, ts, _reach(5.8, 7.0)), tracks, [desk_zone()], _cfg())
    assert not [e for s in _placed(st) for e in s.extra["placements"] if e["object_track_id"] == 21]


def test_unknown_without_desk_zone_and_off_by_default():
    ts = times(8)
    tracks = [person_track(1, "cashier", BOX, ts), _obj(20, "bag", SPOT, [t for t in ts if t >= 4.4])]
    st = compute_pose_signals(poses_for(1, BOX, ts, _reach(4.0, 5.4)), tracks,
                              [Zone("door", [(0.0, 0.0), (0.1, 0.0), (0.1, 0.2), (0.0, 0.2)])], _cfg())
    assert all(s.extra["item_placed_on_desk"] is None for s in st)
    st_default = compute_pose_signals(poses_for(1, BOX, ts, _reach(4.0, 5.4)), tracks, [desk_zone()], load_config())
    assert all("item_placed_on_desk" not in s.extra for s in st_default)
