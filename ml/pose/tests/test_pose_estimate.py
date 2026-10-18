"""estimate(): mock-бэкенд на синтетическом видео (cv2.VideoWriter), лимиты кропов, сбой бэкенда, RTMPose-декод."""
from __future__ import annotations

import inspect

import numpy as np
import pytest

import ml.pose as pose
from ml.pose._contract import PoseFrame, Track
from ml.pose.backends import clear_cache
from ml.pose.backends.mediapipe_backend import MP33_TO_COCO17, mp33_to_coco17
from ml.pose.config import load_config
from ml.pose.estimator import BoxLookup, estimate_impl
from ml.pose.keypoints import COCO17_NAMES

cv2 = pytest.importorskip("cv2")


class SynthClip:
    """Минимальный VideoClip поверх файла, записанного cv2.VideoWriter (ml.ingest не нужен)."""

    def __init__(self, path: str, sample_fps: float = 5.0) -> None:
        cap = cv2.VideoCapture(path)
        self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        self.path = path
        self.height, self.width = frames[0].shape[:2]
        self.duration = len(frames) / self.fps
        step = max(1, int(round(self.fps / sample_fps)))
        self._idx = list(range(0, len(frames), step))
        self._frames = frames
        self.times = [i / self.fps for i in self._idx]

    def frame(self, i: int):
        return self._frames[self._idx[i]]

    def frame_at(self, t: float):
        return self._frames[min(len(self._frames) - 1, int(round(t * self.fps)))]


@pytest.fixture()
def synth_clip(tmp_path):
    path = str(tmp_path / "synth.avi")
    w, h, fps = 232, 424, 10
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for k in range(fps * 4):
        img = np.full((h, w, 3), 40, np.uint8)
        cv2.rectangle(img, (70 + k % 5, 120), (140 + k % 5, 300), (200, 200, 200), -1)
        vw.write(img)
    vw.release()
    return SynthClip(path)


@pytest.fixture()
def mock_env(monkeypatch):
    monkeypatch.setenv("SENTINEL_MOCK", "1")
    monkeypatch.delenv("SENTINEL_POSE_MOCK_SCRIPT", raising=False)
    clear_cache()
    yield
    clear_cache()


def _tracks(clip, n=1):
    boxes = [(0.30, 0.28, 0.62, 0.72), (0.05, 0.30, 0.28, 0.70), (0.70, 0.30, 0.95, 0.70)]
    roles = ["cashier", "visitor", "unknown"]
    return [Track(track_id=i + 1, label="person", role=roles[i], role_conf=0.8,
                  boxes=[(t, boxes[i]) for t in clip.times]) for i in range(n)]


def test_public_signatures_match_contract():
    assert list(inspect.signature(pose.estimate).parameters) == ["clip", "tracks"]
    assert list(inspect.signature(pose.pose_signals).parameters) == ["poses", "tracks", "zones"]


def test_estimate_mock_on_synthetic_video(synth_clip, mock_env):
    poses = pose.estimate(synth_clip, _tracks(synth_clip, 1))
    assert len(poses) == len(synth_clip.times)
    for p in poses:
        assert isinstance(p, PoseFrame) and p.track_id == 1
        assert len(p.keypoints) == 17
        assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= s <= 1.0 for x, y, s in p.keypoints)
        assert p.extra["source"] == "mock" and p.extra["schema"] == "coco17"
        assert p.extra["frame_wh"] == (synth_clip.width, synth_clip.height)
    assert [p.t for p in poses] == sorted(p.t for p in poses)
    st = pose.status()
    assert st["state"] == "mock" and st["n_crops"] == len(poses)
    states = pose.pose_signals(poses, _tracks(synth_clip, 1), [])
    assert len(states) == len(synth_clip.times)


def test_crop_limit_per_frame_prefers_cashier(synth_clip, mock_env):
    cfg = load_config({"selection": {"max_persons_per_frame": 1}})
    poses = estimate_impl(synth_clip, _tracks(synth_clip, 3), cfg)
    assert {p.track_id for p in poses} == {1}


def test_crop_limit_total(synth_clip, mock_env):
    cfg = load_config({"selection": {"max_crops_total": 9}})
    poses = estimate_impl(synth_clip, _tracks(synth_clip, 3), cfg)
    assert len(poses) <= 9
    assert any("max_crops_total" in w for w in pose.status()["warnings"])


def test_backend_failure_does_not_raise_and_is_reported(synth_clip, monkeypatch, tmp_path):
    monkeypatch.delenv("SENTINEL_MOCK", raising=False)
    monkeypatch.delenv("SENTINEL_POSE_BACKEND", raising=False)
    monkeypatch.setenv("SENTINEL_POSE_ONNX", str(tmp_path / "missing.onnx"))
    clear_cache()
    poses = pose.estimate(synth_clip, _tracks(synth_clip, 1))
    assert poses == []                       # никаких поддельных поз
    st = pose.status()
    assert st["state"] == "unavailable" and st["stage_status"] == "skipped" and st["available"] is False
    assert st["error"]
    states = pose.pose_signals(poses, _tracks(synth_clip, 1), [])
    assert all(not s.signals and s.extra["pose_status"] == "unavailable" for s in states)
    clear_cache()


def test_no_person_tracks():
    class Empty:
        times = [0.0, 0.2]

    assert pose.estimate(Empty(), []) == []
    assert pose.status()["state"] == "no_persons"


def test_warmup_mock(mock_env):
    assert pose.warmup() is True


def test_box_lookup_interpolates_and_respects_gap():
    tr = Track(track_id=1, label="person", boxes=[(0.0, (0.0, 0.0, 0.2, 0.2)), (1.0, (0.2, 0.2, 0.4, 0.4)),
                                                   (5.0, (0.5, 0.5, 0.6, 0.6))])
    lk = BoxLookup(tr, tol=0.1, max_gap=1.5)
    assert np.allclose(lk.at(0.5), [0.1, 0.1, 0.3, 0.3])
    assert lk.at(3.0) is None
    assert np.allclose(lk.at(5.05), [0.5, 0.5, 0.6, 0.6])


def test_mediapipe_index_map_is_explicit_coco17():
    mp_names = ["nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
                "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder",
                "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky",
                "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", "right_heel",
                "left_foot_index", "right_foot_index"]
    assert [mp_names[i] for i in MP33_TO_COCO17] == COCO17_NAMES
    lm = np.stack([np.arange(33), np.arange(33) * 2, np.full(33, 0.7)], axis=1).astype(np.float32)
    xy, sc = mp33_to_coco17(lm)
    assert xy.shape == (17, 2) and np.allclose(xy[:, 0], MP33_TO_COCO17) and np.allclose(sc, 0.7)
    with pytest.raises(ValueError):
        mp33_to_coco17(np.zeros((17, 3)))


def test_rtmpose_decode_roundtrip():
    """Пик SimCC в известной точке входа модели → правильные пиксели кадра (как в rtmlib)."""
    from ml.pose.backends.rtmpose_onnx import bbox_xyxy2cs, fix_aspect, simcc_decode, warp_matrix

    box = np.array([100.0, 50.0, 220.0, 350.0])
    center, scale = bbox_xyxy2cs(box, 1.25)
    scale = fix_aspect(scale, (192, 256))
    mat = warp_matrix(center, scale, (192, 256))
    target = np.array([150.0, 120.0])                     # точка в кадре
    inp = mat @ np.array([target[0], target[1], 1.0])     # её координаты во входе модели
    sx = np.zeros((1, 17, 192 * 2), np.float32)
    sy = np.zeros((1, 17, 256 * 2), np.float32)
    sx[0, :, int(round(inp[0] * 2))] = 0.9
    sy[0, :, int(round(inp[1] * 2))] = 0.8
    locs, scores = simcc_decode(sx, sy, 2.0)
    kp = locs[0] / np.array([192, 256]) * scale + center - scale / 2
    assert np.allclose(kp[0], target, atol=1.5)
    assert np.allclose(scores, 0.85)
