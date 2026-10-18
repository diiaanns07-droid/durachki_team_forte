"""Честная доступность позы (аудит): bbox-only, пустые tracks, закрытые голова/руки, неверная геометрия,
переходы available/unavailable, честные статусы и model_info() для A03. Только синтетика, без весов.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import ml.pose as pose
import ml.pose.backends as backends
from ml.pose._contract import PoseFrame, Track, Zone
from ml.pose.backends.mock import MockPose
from ml.pose.config import load_config
from ml.pose.estimator import estimate_impl, valid_box
from ml.pose.signals import compute_pose_signals

from .synth_pose import L_WR, R_WR, desk_zone, object_track, person_track, poses_for, times

BOX = (0.30, 0.30, 0.60, 0.70)
ENV = ("SENTINEL_MOCK", "SENTINEL_POSE_BACKEND", "SENTINEL_POSE_ENABLED", "SENTINEL_POSE_ONNX",
       "SENTINEL_POSE_MP_TASK", "SENTINEL_WEIGHTS_DIR", "SENTINEL_POSE_CONFIG")


class Clip:
    path = "synthetic"
    duration = 2.0
    fps = 30.0
    width, height = 232, 424
    times = [i / 5 for i in range(10)]

    def frame(self, i):
        return np.zeros((self.height, self.width, 3), np.uint8)

    def frame_at(self, t):
        return self.frame(0)


def _tracks(box=BOX, role="cashier", tid=1):
    return [Track(track_id=tid, label="person", role=role, role_conf=0.9, boxes=[(t, box) for t in Clip.times])]


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)  # ./weights/pose не должен случайно найтись
    backends.clear_cache()
    yield
    backends.clear_cache()


class _Fake:
    """Подменный «реальный» бэкенд для тестов (поведение задаётся функцией)."""

    def __init__(self, name, fn, batch_size=4):
        self.name, self.fn, self.batch_size, self.calls = name, fn, batch_size, 0

    def infer(self, items, meta=None):
        self.calls += 1
        return self.fn(self.calls, items)


def _inject(monkeypatch, fake):
    monkeypatch.setenv("SENTINEL_POSE_BACKEND", "rtmpose_onnx")
    monkeypatch.setattr(backends, "_make", lambda name, cfg: fake)
    backends.clear_cache()


def _template(items):
    return MockPose(load_config()).infer(items, [(0.0, 1)] * len(items))


# ---------------------------------------------------------------- доступность и статусы


def test_import_is_not_availability_without_weights():
    av = pose.availability()
    assert av["available"] is False and av["stage_status"] == "skipped"
    assert av["backends"]["rtmpose_onnx"]["ready"] is False and "веса" in av["backends"]["rtmpose_onnx"]["reason"]
    info = pose.model_info()
    assert info["available"] is False and info["checkpoint"] == [] and info["models_loaded"] == []
    json.dumps(info)  # A03 кладёт в identity/ключ кэша — должно сериализоваться


def test_bbox_only_mode_is_unknown_everywhere():
    """Нет весов → поз нет; сигналы — только из bbox нельзя: всё unknown, без координат рук/головы."""
    tracks = _tracks()
    poses = pose.estimate(Clip(), tracks)
    st = pose.status()
    assert poses == [] and st["state"] == "unavailable" and st["stage_status"] == "skipped"
    assert st["coverage"]["frames_posed"] == 0 and st["available"] is False
    cfg = load_config({"signals": {"extras": {"motion": True, "item_placed_on_desk": True}}})
    tracks_obj = tracks + [object_track(9, "bag", (0.43, 0.64, 0.47, 0.68), [t for t in Clip.times if t >= 1.0])]
    states = compute_pose_signals(poses, tracks_obj, [desk_zone()], cfg)
    assert len(states) == len(Clip.times)
    for s in states:
        assert s.signals == {}
        assert set(s.extra["unknown"]) == set(s.extra["enabled"])
        assert s.extra["gestures"] == {} and s.extra["boxes"] == {} and s.extra["track_ids"] == {}
        assert s.extra["motion"] is None and s.extra["item_placed_on_desk"] is None  # не «0 = не клали»
        assert s.extra["pose_status"] == "unavailable"


def test_empty_tracks_and_none():
    assert pose.estimate(Clip(), []) == []
    assert pose.status()["state"] == "no_persons"
    assert pose.estimate(Clip(), None) == []
    assert pose.pose_signals([], [], []) == []
    assert pose.pose_signals([], None, None) == []


def test_transition_unavailable_available_unavailable(monkeypatch, tmp_path):
    """Веса появились → available и новый checkpoint (A03 сменит ключ кэша); пропали → снова skipped."""
    fake_w = tmp_path / "rtmpose-m_test.onnx"
    monkeypatch.setenv("SENTINEL_POSE_ONNX", str(fake_w))
    a0 = pose.model_info()
    assert a0["available"] is False and a0["checkpoint"] == []
    fake_w.write_bytes(b"not a real onnx")
    a1 = pose.model_info()
    assert a1["available"] is True and a1["checkpoint"] == [str(fake_w)] and a1["models_loaded"] == ["rtmpose_onnx"]
    # «готов» ≠ проверен: битый файл не загрузится → честный failed (веса есть, но бэкенд не поднялся)
    assert pose.estimate(Clip(), _tracks()) == []
    st = pose.status()
    assert st["state"] == "unavailable" and st["stage_status"] == "failed"
    fake_w.unlink()
    a2 = pose.model_info()
    assert a2["available"] is False and a2["stage_status"] == "skipped"


def test_model_info_changes_with_flags(monkeypatch):
    base = pose.model_info()
    monkeypatch.setenv("SENTINEL_POSE_ENABLED", "0")
    off = pose.model_info()
    assert off != base and off["stage_status"] == "skipped" and off["env"]["SENTINEL_POSE_ENABLED"] == "0"
    monkeypatch.setenv("SENTINEL_POSE_ENABLED", "1")
    on = pose.model_info()
    assert on["stage_status"] == "failed" and on != base  # принудительно включена, а весов нет
    monkeypatch.delenv("SENTINEL_POSE_ENABLED")
    monkeypatch.setenv("SENTINEL_MOCK", "1")
    mk = pose.model_info()
    assert mk["mock"] is True and mk["stage_status"] == "mock" and mk["checkpoint"] == []


def test_switching_backend_does_not_reuse_warm_mock(monkeypatch):
    monkeypatch.setenv("SENTINEL_POSE_BACKEND", "mock")
    assert len(pose.estimate(Clip(), _tracks())) == len(Clip.times)
    assert pose.status()["stage_status"] == "mock"
    monkeypatch.setenv("SENTINEL_POSE_BACKEND", "rtmpose_onnx")
    assert pose.estimate(Clip(), _tracks()) == []          # тёплый mock не подставился вместо реального
    assert pose.status()["stage_status"] == "skipped"


def test_crash_mid_run_is_failed_not_ok(monkeypatch):
    def fn(call, items):
        if call >= 2:
            raise RuntimeError("CUDA error (синтетика)")
        return _template(items)

    _inject(monkeypatch, _Fake("rtmpose-onnx", fn, batch_size=2))
    poses = pose.estimate(Clip(), _tracks())
    st = pose.status()
    assert 0 < len(poses) < len(Clip.times)                # реальные позы до сбоя сохранены
    assert st["state"] == "failed" and st["stage_status"] == "failed" and "CUDA" in st["error"]
    assert st["coverage"]["stopped_early"] is True


def test_backend_finds_no_pose_gives_no_skeleton(monkeypatch):
    _inject(monkeypatch, _Fake("rtmpose-onnx", lambda c, items: [(np.zeros((17, 2), np.float32),
                                                                     np.zeros(17, np.float32)) for _ in items]))
    poses = pose.estimate(Clip(), _tracks())
    st = pose.status()
    assert poses == []                                      # нули не выдаются как «скелет в (0,0)»
    assert st["state"] == "no_poses" and st["stage_status"] == "ok" and st["empty_poses"] == len(Clip.times)


def test_crop_cap_is_timeout(monkeypatch):
    orig_make = backends._make
    _inject(monkeypatch, _Fake("rtmpose-onnx", lambda c, items: _template(items)))
    cfg = load_config({"selection": {"max_crops_total": 3}})
    poses = estimate_impl(Clip(), _tracks(), cfg)
    st = pose.status()
    assert len(poses) == 3 and st["stage_status"] == "timeout"   # конец видео без позы — не «ok»
    assert st["coverage"]["analyzed_until_sec"] < Clip.times[-1]
    monkeypatch.setattr(backends, "_make", orig_make)
    monkeypatch.setenv("SENTINEL_POSE_BACKEND", "mock")          # в mock-режиме главное — «не настоящие»
    backends.clear_cache()
    estimate_impl(Clip(), _tracks(), cfg)
    assert pose.status()["stage_status"] == "mock"


# ---------------------------------------------------------------- неверная геометрия


@pytest.mark.parametrize("box", [(np.nan, 0.3, 0.6, 0.7), (0.6, 0.7, 0.3, 0.3), (60, 120, 180, 300),
                                 (0.3, 0.3, 0.3, 0.7), (0.3, 0.3, 0.31, 0.31)])
def test_invalid_track_boxes_are_skipped(monkeypatch, box):
    monkeypatch.setenv("SENTINEL_MOCK", "1")
    poses = pose.estimate(Clip(), _tracks(box=box))
    st = pose.status()
    assert poses == [] and st["invalid_boxes"] == len(Clip.times)
    assert any("невалидные bbox" in w for w in st["warnings"])


def test_valid_box_contract():
    assert valid_box((0.1, 0.2, 0.5, 0.9), 0.002) is not None
    assert valid_box((-0.01, 0.2, 0.5, 1.01), 0.002) is not None    # небольшой выход за кадр — обрезаем
    assert valid_box((0.5, 0.2, 0.1, 0.9), 0.002) is None
    assert valid_box((0.1, 0.2, 0.5), 0.002) is None
    assert valid_box(("a", 0, 1, 1), 0.002) is None


def test_keypoints_far_outside_bbox_are_not_attributed(monkeypatch):
    """Бэкенд вернул точки другого человека (далеко за bbox трека) → поза не приписывается треку."""
    def fn(call, items):
        out = []
        for frame, box in items:
            kp, sc = _template([(frame, box)])[0]
            out.append((kp + np.array([400.0, 0.0], np.float32), sc))  # сдвиг далеко вправо за bbox
        return out

    _inject(monkeypatch, _Fake("rtmpose-onnx", fn))
    assert pose.estimate(Clip(), _tracks()) == []
    assert pose.status()["empty_poses"] == len(Clip.times)


@pytest.mark.parametrize("bad", [
    [(0.5, 0.5)] * 17,                                    # чужая схема: 2 числа на точку
    [(0.5, 0.5, 0.9)] * 33,                               # 33 точки MediaPipe вместо COCO-17
    [(float("nan"), 0.5, 0.9)] * 17,                      # NaN-координаты при высоком score
    [(0.5, 0.5, 5.0), "x", None] + [(0.5, 0.5)] * 14,     # мусор
    [],
])
def test_malformed_keypoints_do_not_crash_or_claim(bad):
    ts = times(3)
    poses = [PoseFrame(t=t, track_id=1, keypoints=list(bad)) for t in ts]
    st = pose.pose_signals(poses, [person_track(1, "cashier", BOX, ts)], [desk_zone()])
    assert len(st) == len(ts)
    for s in st:
        assert all(v < 0.5 for v in s.signals.values())   # ничего не «активно» из мусора


def test_nan_keypoints_are_unknown_not_confident():
    from ml.pose.keypoints import kp_array

    arr = kp_array([(float("nan"), 0.5, 0.9), (0.2, 0.3, 1.7), (0.1, 0.1)] + [(0.5, 0.5, 0.8)] * 14)
    assert arr[0, 2] == 0.0 and arr[1, 2] == 1.0 and arr[2, 2] == 0.0 and arr[3, 2] == pytest.approx(0.8)


# ---------------------------------------------------------------- закрытые голова / руки


def test_occluded_head_makes_head_signals_unknown():
    """Голова не видна → «кисть ко рту»/еда неопределимы (unknown), а не «не ест»; телефон в руке — определим."""
    ts = times(6)
    head_hidden = {k: 0.0 for k in range(5)}
    poses = poses_for(1, BOX, ts, lambda t: (None, head_hidden))
    tracks = [person_track(1, "cashier", BOX, ts), object_track(9, "cup", (0.40, 0.50, 0.46, 0.56), ts),
              object_track(8, "cell_phone", (0.40, 0.50, 0.46, 0.55), ts)]
    st = pose.pose_signals(poses, tracks, [desk_zone()])
    for s in st:
        assert "eating_drinking" not in s.signals            # рот не виден — не утверждаем «не ест»
        assert "hand_to_mouth" not in s.extra["gestures"]
    assert any((s.signals.get("phone_in_hand") or 0) >= 0.5 for s in st)  # кисть + телефон видны


def test_occluded_hands_are_unknown_not_hidden_or_negative():
    ts = times(6)
    hands_hidden = {L_WR: 0.0, R_WR: 0.0, 7: 0.0, 8: 0.0}
    poses = poses_for(1, BOX, ts, lambda t: (None, hands_hidden))
    tracks = [person_track(1, "cashier", BOX, ts), object_track(8, "cell_phone", (0.40, 0.50, 0.46, 0.55), ts)]
    st = pose.pose_signals(poses, tracks, [desk_zone()])
    for s in st:
        for name in ("hands_hidden", "phone_in_hand", "eating_drinking", "handover_motion"):
            assert name not in s.signals, (name, s.signals)  # unknown, ни «скрыл руки», ни «нет телефона»


def test_one_person_frame_has_no_handover_claim():
    ts = times(4)
    st = pose.pose_signals(poses_for(1, BOX, ts), [person_track(1, "cashier", BOX, ts)], [desk_zone()])
    assert all("handover_motion" not in s.signals for s in st)  # второго человека нет — передача неопределима


def test_zone_with_bad_geometry_is_ignored():
    ts = times(4)
    bad = [Zone("desk", [(0.1, 0.1), (0.2, 0.2)]), Zone("desk", [(np.nan, 0.1), (0.2, np.nan), (0.3, 0.3)])]
    st = pose.pose_signals(poses_for(1, BOX, ts), [person_track(1, "cashier", BOX, ts)], bad)
    assert len(st) == len(ts)
