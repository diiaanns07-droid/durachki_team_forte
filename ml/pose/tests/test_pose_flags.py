"""Главный флаг позы (enabled), выбор бэкенда через env, stage_status/coverage для meta.completeness."""
from __future__ import annotations

import numpy as np
import pytest

import ml.pose as pose
from ml.pose._contract import Track
from ml.pose.backends import clear_cache
from ml.pose.config import backend_override, enabled_mode, load_config


class _Clip:
    path = "synthetic"
    duration = 2.0
    fps = 30.0
    width, height = 232, 424
    times = [i / 5 for i in range(10)]

    def frame(self, i):
        return np.zeros((self.height, self.width, 3), np.uint8)

    def frame_at(self, t):
        return self.frame(0)


def _tracks():
    return [Track(track_id=1, label="person", role="cashier", role_conf=0.9,
                  boxes=[(t, (0.3, 0.3, 0.6, 0.7)) for t in _Clip.times])]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in ("SENTINEL_MOCK", "SENTINEL_POSE_BACKEND", "SENTINEL_POSE_ENABLED", "SENTINEL_POSE_ONNX"):
        monkeypatch.delenv(k, raising=False)
    clear_cache()
    yield
    clear_cache()


@pytest.mark.parametrize("raw,mode", [("0", "false"), ("false", "false"), ("1", "true"), ("auto", "auto"), ("", "auto")])
def test_enabled_mode_env(monkeypatch, raw, mode):
    monkeypatch.setenv("SENTINEL_POSE_ENABLED", raw)
    assert enabled_mode(load_config()) == mode


def test_disabled_returns_empty_even_in_mock(monkeypatch):
    monkeypatch.setenv("SENTINEL_MOCK", "1")
    monkeypatch.setenv("SENTINEL_POSE_ENABLED", "0")
    assert pose.estimate(_Clip(), _tracks()) == []
    st = pose.status()
    assert st["state"] == "disabled" and st["stage_status"] == "skipped"
    assert pose.warmup() is False


def test_auto_missing_weights_is_skipped_not_failed(monkeypatch, tmp_path):
    monkeypatch.setenv("SENTINEL_POSE_ONNX", str(tmp_path / "none.onnx"))
    assert pose.estimate(_Clip(), _tracks()) == []
    st = pose.status()
    assert st["stage_status"] == "skipped" and "не найдены" in st["error"]


def test_forced_on_missing_weights_is_failed(monkeypatch, tmp_path):
    monkeypatch.setenv("SENTINEL_POSE_ONNX", str(tmp_path / "none.onnx"))
    monkeypatch.setenv("SENTINEL_POSE_ENABLED", "1")
    assert pose.estimate(_Clip(), _tracks()) == []
    assert pose.status()["stage_status"] == "failed"


def test_backend_override_env(monkeypatch):
    monkeypatch.setenv("SENTINEL_POSE_BACKEND", "mediapipe")
    assert backend_override() == "mediapipe"
    monkeypatch.setenv("SENTINEL_POSE_BACKEND", "bogus")
    assert backend_override() is None
    monkeypatch.setenv("SENTINEL_POSE_BACKEND", "mock")
    poses = pose.estimate(_Clip(), _tracks())
    assert len(poses) == len(_Clip.times) and pose.status()["stage_status"] == "mock"


def test_coverage_reported(monkeypatch):
    monkeypatch.setenv("SENTINEL_MOCK", "1")
    pose.estimate(_Clip(), _tracks())
    cov = pose.status()["coverage"]
    assert cov["frames_total"] == len(_Clip.times)
    assert cov["frames_posed"] == len(_Clip.times) and cov["cashier_frames_posed"] == len(_Clip.times)
    assert cov["stride"] == 1 and cov["crops_capped"] is False and cov["stopped_early"] is False
    assert cov["analyzed_until_sec"] == pytest.approx(_Clip.times[-1])
