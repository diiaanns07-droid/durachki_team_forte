"""MediaPipe Pose Landmarker (Tasks API): кроп + апскейл, обратный маппинг, выбор позы, выбор кассира.

Основные тесты — с внедрённым фейковым landmarker-ом (без mediapipe и модели). Реальный smoke — slow,
пропускается без mediapipe и $SENTINEL_POSE_MP_TASK.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

import ml.pose.backends as backends
from ml.pose._contract import Track
from ml.pose.backends.mediapipe_backend import (MP33_NAMES, MP33_TO_COCO17, MediaPipePose, crop_upscale,
                                                pick_pose, to_frame)
from ml.pose.config import load_config
from ml.pose.estimator import allowed_tracks, backend_selection, estimate_impl
from ml.pose.keypoints import COCO17_NAMES

cv2 = pytest.importorskip("cv2")


def _lm(x, y, vis=0.9, pres=0.95):
    return SimpleNamespace(x=x, y=y, z=0.0, visibility=vis, presence=pres)


class FakeLandmarker:
    """Возвращает заданные позы (33 точки в норм. координатах входного изображения)."""

    def __init__(self, poses):
        self.poses = poses
        self.shapes = []

    def detect(self, rgb):
        self.shapes.append(rgb.shape)
        return SimpleNamespace(pose_landmarks=[[_lm(x, y) for x, y in p] for p in self.poses])


def _pose_at(cx, cy, spread=0.1):
    rng = np.random.default_rng(0)
    return [(cx + float(rng.uniform(-spread, spread)), cy + float(rng.uniform(-spread, spread))) for _ in range(33)]


def test_mapping_names_explicit():
    assert len(MP33_NAMES) == 33
    assert [MP33_NAMES[i] for i in MP33_TO_COCO17] == COCO17_NAMES


def test_crop_upscale_geometry_and_backmapping():
    frame = np.zeros((848, 464, 3), np.uint8)
    box = (100.0, 50.0, 180.0, 210.0)            # маленький человек: 80×160 px
    img, geom = crop_upscale(frame, box, padding=1.4, min_side=384, max_scale=3.0)
    assert min(img.shape[:2]) >= 336              # апскейл (ограничен max_scale=3 → 112*3=336)
    x0, y0, cw, ch = geom
    assert cw == pytest.approx(80 * 1.4, rel=1e-3) and ch == pytest.approx(160 * 1.4, rel=1e-3)
    centre = to_frame(np.array([[0.5, 0.5]]), geom)[0]
    assert np.allclose(centre, [140.0, 130.0], atol=0.6)  # центр кропа = центр bbox
    big, _ = crop_upscale(frame, (0, 0, 400, 800), padding=1.0, min_side=384, max_scale=3.0)
    assert big.shape[:2] == (800, 400)            # крупный кроп не уменьшаем и не увеличиваем


def test_infer_maps_to_frame_and_scores():
    cfg = load_config()
    be = MediaPipePose(cfg, landmarker=FakeLandmarker([[(0.5, 0.5)] * 33]))
    frame = np.zeros((848, 464, 3), np.uint8)
    (kp, sc), = be.infer([(frame, np.array([100, 50, 180, 210], np.float32))])
    assert kp.shape == (17, 2) and sc.shape == (17,)
    assert np.allclose(kp, [140.0, 130.0], atol=0.6)
    assert np.allclose(sc, 0.9)                   # min(visibility, presence)
    assert min(be.landmarker.shapes[0][:2]) >= 336


def test_no_pose_gives_zero_scores_not_fake_points():
    be = MediaPipePose(load_config(), landmarker=FakeLandmarker([]))
    (kp, sc), = be.infer([(np.zeros((100, 100, 3), np.uint8), np.array([10, 10, 60, 90], np.float32))])
    assert (sc == 0).all()


def test_pick_pose_prefers_target_box():
    box = (100, 100, 200, 300)
    inside = (np.full((17, 2), 150.0, np.float32), np.full(17, 0.8, np.float32))
    outside = (np.full((17, 2), 400.0, np.float32), np.full(17, 0.9, np.float32))
    assert pick_pose([outside, inside], box) == 1


def test_mediapipe_selection_only_cashier(monkeypatch):
    cfg = load_config()
    sel = backend_selection(cfg, "mediapipe")
    assert sel["max_persons_per_frame"] == 1 and sel["roles"] == ["cashier"]
    tr = {1: Track(1, "person", "visitor"), 2: Track(2, "person", "cashier")}
    assert allowed_tracks(tr, sel) == {2}
    assert allowed_tracks({1: Track(1, "person", "unknown")}, sel) is None  # fallback top1
    assert "roles" not in backend_selection(cfg, "rtmpose-onnx")


def test_estimate_with_injected_mediapipe(monkeypatch):
    monkeypatch.delenv("SENTINEL_MOCK", raising=False)
    monkeypatch.setenv("SENTINEL_POSE_BACKEND", "mediapipe")
    fake = FakeLandmarker([[(0.5, 0.45)] * 33])
    monkeypatch.setattr(backends, "_make", lambda name, cfg: MediaPipePose(cfg, landmarker=fake))
    backends.clear_cache()

    class Clip:
        times = [0.0, 0.2, 0.4]

        def frame(self, i):
            return np.zeros((848, 464, 3), np.uint8)

    tracks = [Track(1, "person", "visitor", 0.9, [(t, (0.6, 0.3, 0.9, 0.7)) for t in Clip.times]),
              Track(2, "person", "cashier", 0.9, [(t, (0.2, 0.3, 0.5, 0.7)) for t in Clip.times])]
    poses = estimate_impl(Clip(), tracks, load_config())
    assert {p.track_id for p in poses} == {2}
    assert all(p.extra["source"] == "mediapipe" and len(p.keypoints) == 17 for p in poses)
    backends.clear_cache()


@pytest.mark.slow
def test_real_mediapipe_smoke():
    pytest.importorskip("mediapipe")
    task = os.environ.get("SENTINEL_POSE_MP_TASK")
    if not task or not os.path.isfile(task):
        pytest.skip("нет $SENTINEL_POSE_MP_TASK")
    try:
        be = MediaPipePose(load_config())
    except Exception as exc:  # noqa: BLE001 — например, нет libGLESv2 в контейнере
        pytest.skip(f"PoseLandmarker недоступен: {exc}")
    img = np.full((424, 232, 3), 128, np.uint8)
    (kp, sc), = be.infer([(img, np.array([60, 80, 170, 400], np.float32))])
    assert kp.shape == (17, 2) and sc.shape == (17,) and np.isfinite(kp).all()
    be.close()
