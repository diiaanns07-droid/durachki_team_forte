"""Фикстуры тестов ml.vlm: всё на синтетике и mock-бэкенде — без GPU, весов и сети."""
from __future__ import annotations

import numpy as np
import pytest

from ml.vlm import runtime
from ml.vlm.budget import FakeClock
from ml.vlm.catalog import set_catalog
from ml.vlm.config import load_config


def pytest_configure(config):  # маркеры, если корневой pyproject (A01) их ещё не объявил
    config.addinivalue_line("markers", "gpu: требует CUDA и скачанных весов")
    config.addinivalue_line("markers", "slow: долгий тест")


class SyntheticClip:
    """Минимальная реализация протокола VideoClip на numpy (портрет 464x848, 5 fps)."""

    def __init__(self, duration: float = 12.0, sample_fps: float = 5.0, w: int = 464, h: int = 848,
                 moving_from: float = 0.0, moving_to: float = 0.0):
        self.path = "synthetic"
        self.duration = float(duration)
        self.fps = 30.0
        self.width, self.height = w, h
        self.times = [round(i / sample_fps, 4) for i in range(int(duration * sample_fps))]
        self.moving = (moving_from, moving_to)
        self.frame_at_calls = 0

    def _render(self, t: float):
        f = np.full((self.height, self.width, 3), 60, np.uint8)
        a, b = self.moving
        x = 40 + int(30 * t) % 300 if a <= t <= b else 40
        f[300:420, x:x + 80] = (0, 180, 0)
        return f

    def frame(self, i: int):
        return self._render(self.times[i])

    def frame_at(self, t: float):
        self.frame_at_calls += 1
        return self._render(float(t))


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Каждый тест — с чистым состоянием бэкенда и без влияния env разработчика."""
    for k in ("SENTINEL_VLM_BACKEND", "SENTINEL_VLM_PRESET", "SENTINEL_VLM_MODEL", "SENTINEL_VLM_URL",
              "SENTINEL_VLM_CONFIG", "SENTINEL_VLM_LANG", "SENTINEL_VLM_MOCK_MODE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SENTINEL_MOCK", "1")
    runtime.set_backend(None)
    runtime.unload_backend()
    runtime.set_gpu_lock(None)
    set_catalog(None)
    yield
    runtime.set_backend(None)
    runtime.unload_backend()
    set_catalog(None)


@pytest.fixture
def clip():
    return SyntheticClip()


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def cfg():
    """Конфиг для тестов: mock, быстрый est_call_s, discover включён явно там, где нужно."""
    c = load_config(use_env=False, overrides={"backend": "mock"})
    c["generation"]["est_call_s"] = 0.5
    return c
