"""Общие фикстуры тестов ml.fusion: синтетические клипы (без видео, весов и GPU)."""
from __future__ import annotations

import os
from typing import Any, Callable, Optional

import numpy as np
import pytest

from ml.fusion._types import Event

V01_RULE = {"document": "Тестовый документ", "clause": "п. 1", "url": "https://adilet.zan.kz/rus/docs/TEST",
            "quote": "тестовая цитата", "verified": True}


class FakeClip:
    """Минимальная реализация VideoClip: кадр i кодирует номер исходного кадра (fps-сетка) в пикселях."""

    def __init__(self, duration: float = 20.0, fps: float = 30.0, sample_fps: float = 5.0, w: int = 96,
                 h: int = 160, maker: Optional[Callable[[int], np.ndarray]] = None):
        self.path = "synthetic.mov"
        self.duration = duration
        self.fps = fps
        self.width, self.height = w, h
        n = int(duration * sample_fps)
        self.times = [round(i / sample_fps, 4) for i in range(n)]
        self._maker = maker or self.coded_frame
        self.frame_at_calls: list[float] = []

    def coded_frame(self, idx: int) -> np.ndarray:
        img = np.zeros((self.height, self.width, 3), np.uint8)
        img[:, :] = (idx * 7 % 256, idx // 256 * 40 % 256, 128)
        img[: self.height // 4, : self.width // 2] = (idx % 256, idx % 256, idx % 256)  # «номер кадра»
        return img

    def frame(self, i: int) -> np.ndarray:
        return self._maker(int(round(self.times[i] * self.fps)))

    def frame_at(self, t: float) -> np.ndarray:
        self.frame_at_calls.append(t)
        return self._maker(int(round(t * self.fps)))


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("SENTINEL_FUSION_CONFIG", "SENTINEL_FUSION_THRESHOLDS", "SENTINEL_FUSION_NO_FRAMES"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def catalog() -> dict[str, Any]:
    return {
        "version": 1,
        "types": {
            "V01": {"name": "Посторонний в кассовом помещении", "severity": "high", "rules": [V01_RULE]},
            "V03": {"name": "Сейф открыт без присмотра", "severity": "high", "rules": [dict(V01_RULE, clause="п. 3")]},
            "V05": {"name": "Телефон при работе с ценностями", "severity": "medium",
                    "rules": [dict(V01_RULE, clause="п. 5", verified=False)]},
            "V06": {"name": "Еда на рабочем месте", "severity": "low", "rules": []},
            "V99": {"name": "Иное подозрительное действие", "severity": "medium",
                    "rules": [dict(V01_RULE, clause="п. 99", verified="false")]},
        },
    }


@pytest.fixture
def no_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTINEL_FUSION_NO_FRAMES", "1")


def ev(type_id: str = "V01", t: float = 5.0, conf: float = 0.8, source: str = "rules", dur: float = 1.0,
       facts: Optional[list[str]] = None, **kw: Any) -> Event:
    if facts is None:
        facts = ["посетитель находится за кассовым столом"] if source in ("rules", "vlm") else []
    return Event(type_id=type_id, t_start=t - dur / 2, t_end=t + dur / 2, t_peak=t, confidence=conf,
                 source=source, facts=facts, **kw)


def env_flag() -> bool:
    return os.environ.get("SENTINEL_FUSION_NO_FRAMES") == "1"
