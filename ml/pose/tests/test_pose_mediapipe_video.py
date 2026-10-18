"""MediaPipe VIDEO-режим (L2): поток на track_id, время из PTS, выбор позы по bbox трека, лимит потоков.

Фейковые landmarker-ы — без mediapipe и модели; реальные кадры не используются.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ml.pose.backends.mediapipe_backend import MediaPipePose

pytest.importorskip("cv2")


def _lm(x, y, vis=0.9, pres=0.95):
    return SimpleNamespace(x=x, y=y, z=0.0, visibility=vis, presence=pres)


class FakeVideoLandmarker:
    """detect_for_video: запоминает отметки времени; поза — облако точек в заданной точке кропа."""

    def __init__(self, poses):
        self.poses = poses
        self.ts: list[int] = []
        self.closed = False

    def detect(self, rgb):
        return SimpleNamespace(pose_landmarks=[[_lm(x, y) for x, y in p] for p in self.poses])

    def detect_for_video(self, rgb, ts_ms):
        self.ts.append(ts_ms)
        return self.detect(rgb)

    def close(self):
        self.closed = True


def _cloud(cx, cy, spread=0.05):
    rng = np.random.default_rng(1)
    return [(cx + float(rng.uniform(-spread, spread)), cy + float(rng.uniform(-spread, spread))) for _ in range(33)]


def _cfg(**mp):
    base = {"running_mode": "video", "max_streams": 2, "crop_padding": 1.0, "upscale_min_side": 64, "upscale_max": 1.0}
    base.update(mp)
    return {"backend": {"mediapipe": base}, "signals": {"kp_min_score": 0.3}}


def test_each_track_gets_own_stream_with_monotonic_pts():
    made = []

    def factory():
        lm = FakeVideoLandmarker([_cloud(0.5, 0.5)])
        made.append(lm)
        return lm

    be = MediaPipePose(_cfg(), landmarker_factory=factory)
    frame = np.zeros((200, 200, 3), np.uint8)
    box_a, box_b = np.array([10, 10, 90, 190], np.float32), np.array([110, 10, 190, 190], np.float32)
    # VFR: неравномерные PTS; повтор отметки трека 1 — должен быть сдвинут вперёд, а не отброшен
    be.infer([(frame, box_a), (frame, box_b)], [(0.000, 1), (0.000, 2)])
    be.infer([(frame, box_a)], [(0.034, 1)])
    be.infer([(frame, box_a), (frame, box_b)], [(0.034, 1), (0.101, 2)])
    streams = {tid: lm for tid, lm in be._streams.items()}
    assert set(streams) == {1, 2}
    assert streams[1].ts == [0, 34, 35] and streams[2].ts == [0, 101]
    assert be.stats["ts_bumped"] == 1 and be.stats["streams_opened"] == 2


def test_pose_of_neighbour_in_crop_is_not_assigned_to_track():
    # единственная поза в кропе лежит вне bbox трека (это сосед) → нулевые score, а не чужой скелет
    be = MediaPipePose(_cfg(crop_padding=2.0), landmarker_factory=lambda: FakeVideoLandmarker([_cloud(0.05, 0.05, 0.02)]))
    frame = np.zeros((200, 200, 3), np.uint8)
    (kp, sc), = be.infer([(frame, np.array([80, 80, 120, 120], np.float32))], [(0.5, 7)])
    assert float(sc.max()) == 0.0


def test_order_of_poses_is_not_identity():
    # две позы в кропе в разном порядке на разных кадрах: выбирается та, что внутри bbox трека
    inside, outside = _cloud(0.5, 0.5, 0.05), _cloud(0.02, 0.02, 0.01)
    lms = [FakeVideoLandmarker([inside, outside]), FakeVideoLandmarker([outside, inside])]
    it = iter(lms)
    be = MediaPipePose(_cfg(crop_padding=2.0, max_streams=4), landmarker=FakeVideoLandmarker([]),
                       landmarker_factory=lambda: next(it))
    frame = np.zeros((200, 200, 3), np.uint8)
    box = np.array([80, 80, 120, 120], np.float32)
    (kp1, sc1), = be.infer([(frame, box)], [(0.0, 1)])
    (kp2, sc2), = be.infer([(frame, box)], [(0.0, 2)])
    for kp, sc in ((kp1, sc1), (kp2, sc2)):
        assert float(sc.max()) > 0.3
        assert 70 <= float(np.median(kp[:, 0])) <= 130


def test_stream_limit_evicts_oldest_and_closes_it():
    made = []

    def factory():
        lm = FakeVideoLandmarker([_cloud(0.5, 0.5)])
        made.append(lm)
        return lm

    be = MediaPipePose(_cfg(max_streams=2), landmarker=FakeVideoLandmarker([]), landmarker_factory=factory)
    frame = np.zeros((100, 100, 3), np.uint8)
    box = np.array([10, 10, 90, 90], np.float32)
    for tid in (1, 2, 3):
        be.infer([(frame, box)], [(0.1 * tid, tid)])
    assert set(be._streams) == {2, 3} and made[0].closed and be.stats["streams_evicted"] == 1


def test_missing_track_id_uses_image_mode_and_is_counted():
    img_lm = FakeVideoLandmarker([_cloud(0.5, 0.5)])
    be = MediaPipePose(_cfg(), landmarker=img_lm, landmarker_factory=lambda: FakeVideoLandmarker([]))
    frame = np.zeros((100, 100, 3), np.uint8)
    be.infer([(frame, np.array([10, 10, 90, 90], np.float32))], [(0.2, None)])
    assert be.stats["no_track_id"] == 1 and img_lm.ts == [] and not be._streams
