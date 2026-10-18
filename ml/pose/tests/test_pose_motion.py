"""extra['motion'] (opt-in): sitting / standing / walking + направление относительно зон."""
from __future__ import annotations

import numpy as np

from ml.pose._contract import PoseFrame, Track, Zone
from ml.pose.config import load_config
from ml.pose.signals import compute_pose_signals

from .synth_pose import desk_zone, kps_in_box, person_track, poses_for, times

DOOR = Zone("door", [(0.88, 0.25), (1.0, 0.25), (1.0, 0.60), (0.88, 0.60)])


def _cfg(on: bool = True):
    return load_config({"signals": {"extras": {"motion": on}}})


def _walker(ts, t0, t1, vx=0.16, x_start=0.10, w=0.30, jitter=0.0, seed=0):
    """Кассир идёт вправо (к двери) с t0 до t1 со скоростью vx (норм. x в секунду)."""
    rng = np.random.default_rng(seed)
    poses, boxes = [], []
    for t in ts:
        dx = vx * (min(max(t, t0), t1) - t0)
        box = (x_start + dx, 0.30, x_start + dx + w, 0.70)
        boxes.append((t, box))
        poses.append(PoseFrame(t=t, track_id=1, keypoints=kps_in_box(box, jitter=jitter, rng=rng)))
    return poses, Track(track_id=1, label="person", role="cashier", role_conf=0.9, boxes=boxes)


def test_walking_toward_door_with_onset():
    ts = times(8)
    poses, tr = _walker(ts, 2.0, 5.0)
    st = compute_pose_signals(poses, [tr], [desk_zone(), DOOR], _cfg())
    mid = [s for s in st if 2.8 <= s.t <= 4.4]
    assert all(s.extra["motion"]["state"] == "walking" for s in mid)
    run = mid[0].extra["motion"]["run"]
    assert 1.6 <= run["t_start"] <= 2.6 and run["toward"] == "door"
    assert run["direction"][0] > 0.9                      # вправо
    assert all(s.extra["motion"]["state"] != "walking" for s in st if s.t <= 1.0 or s.t >= 6.4)


def test_standing_with_jitter_is_not_walking():
    ts = times(8)
    poses, tr = _walker(ts, 100.0, 101.0, jitter=0.003)
    st = compute_pose_signals(poses, [tr], [desk_zone(), DOOR], _cfg())
    states = {s.extra["motion"]["state"] for s in st if s.extra["motion"]}
    assert "walking" not in states and "standing" in states


def test_sitting_detected_from_thigh_angle():
    ts = times(4)
    box = (0.30, 0.30, 0.60, 0.70)
    # сидит боком к камере: оба бедра горизонтально в одну сторону от таза (колени на высоте таза)
    seated = {13: (0.88, 0.62), 14: (0.84, 0.64)}
    poses = poses_for(1, box, ts, lambda t: (seated, None))
    st = compute_pose_signals(poses, [person_track(1, "cashier", box, ts)], [], _cfg())
    assert all(s.extra["motion"]["state"] == "sitting" for s in st)


def test_low_confidence_motion_unknown():
    ts = times(4)
    box = (0.30, 0.30, 0.60, 0.70)
    low = {k: 0.05 for k in range(17)}
    poses = poses_for(1, box, ts, lambda t: (None, low))
    tr = Track(track_id=1, label="person", role="cashier", boxes=[])  # нет и bbox трека
    st = compute_pose_signals(poses, [tr], [], _cfg())
    assert all(s.extra["motion"] is None for s in st)


def test_motion_off_by_default():
    ts = times(4)
    poses, tr = _walker(ts, 1.0, 3.0)
    st = compute_pose_signals(poses, [tr], [DOOR], load_config())
    assert all("motion" not in s.extra for s in st)
