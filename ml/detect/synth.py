"""Синтетика для тестов и mock-интеграции + минимальный VideoClip (без зависимости от ml.ingest).

Конвенция (совпадает с mock.palette в config.yaml): тёмный фон (20,20,20), объекты — залитые
эллипсы/прямоугольники «чистых» цветов палитры. Mock-бэкенд (SENTINEL_MOCK=1) находит их по цвету.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .config import load_config

BG = (20, 20, 20)


@dataclass
class SynthObject:
    label: str
    shape: str            # rect | ellipse
    box0: tuple[float, float, float, float]   # норм. xyxy в начале
    box1: Optional[tuple[float, float, float, float]] = None  # в конце (линейное движение), None — статичен
    t_on: float = 0.0
    t_off: float = 1e9

    def box_at(self, t: float, duration: float) -> Optional[tuple[float, float, float, float]]:
        if not (self.t_on <= t < self.t_off):
            return None
        if self.box1 is None or duration <= 0:
            return self.box0
        a = min(1.0, max(0.0, t / duration))
        return tuple(float(p + (q - p) * a) for p, q in zip(self.box0, self.box1))  # type: ignore[return-value]


def default_scene() -> list[SynthObject]:
    """Касса сверху (портрет): стол, сейф, дверь; кассир двигается, телефон появляется на 1–3 с."""
    return [
        SynthObject("safe", "rect", (0.70, 0.62, 0.95, 0.80)),
        SynthObject("door", "rect", (0.02, 0.02, 0.30, 0.20)),
        SynthObject("laptop", "rect", (0.40, 0.40, 0.62, 0.52)),
        SynthObject("money_counter", "rect", (0.20, 0.42, 0.32, 0.50)),
        SynthObject("person", "ellipse", (0.35, 0.60, 0.60, 0.85), (0.45, 0.58, 0.70, 0.83)),
        SynthObject("cash", "rect", (0.25, 0.30, 0.33, 0.35), t_on=0.5),
        SynthObject("cell_phone", "rect", (0.55, 0.66, 0.60, 0.70), t_on=1.0, t_off=3.0),
    ]


def render(objects: Sequence[SynthObject], t: float, duration: float, width: int, height: int,
           palette: Optional[dict[str, Sequence[int]]] = None) -> tuple[np.ndarray, list[tuple[str, tuple]]]:
    import cv2

    if palette is None:
        m = load_config()["mock"]
        palette = {**m["palette"], **m.get("extra_palette", {})}
    pal = palette
    img = np.full((height, width, 3), BG, np.uint8)
    gt = []
    for ob in objects:
        b = ob.box_at(t, duration)
        if b is None:
            continue
        x1, y1, x2, y2 = (int(round(b[0] * width)), int(round(b[1] * height)),
                          int(round(b[2] * width)), int(round(b[3] * height)))
        color = tuple(int(c) for c in pal[ob.label])
        if ob.shape == "ellipse":
            cv2.ellipse(img, (((x1 + x2) / 2, (y1 + y2) / 2), (x2 - x1, y2 - y1), 0), color, -1)
        else:
            cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), color, -1)
        gt.append((ob.label, (x1 / width, y1 / height, x2 / width, y2 / height)))
    return img, gt


def make_synth_video(path: str | Path, duration_s: float = 4.0, fps: float = 10.0, width: int = 464,
                     height: int = 848, objects: Optional[Sequence[SynthObject]] = None) -> list[list[tuple[str, tuple]]]:
    """Пишет синтетическое видео (mp4v) через cv2.VideoWriter; возвращает GT по каждому кадру."""
    import cv2

    objs = list(objects) if objects is not None else default_scene()
    n = int(round(duration_s * fps))
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not wr.isOpened():
        raise RuntimeError(f"cv2.VideoWriter не открылся для {path}")
    gts = []
    try:
        for k in range(n):
            img, gt = render(objs, k / fps, duration_s, width, height)
            wr.write(img)
            gts.append(gt)
    finally:
        wr.release()
    return gts


@dataclass
class ArrayClip:
    """Минимальная реализация протокола VideoClip поверх списка кадров (для тестов/CLI)."""

    frames: list[np.ndarray]
    times: list[float]
    path: str = ""
    fps: float = 5.0
    duration: float = 0.0
    width: int = 0
    height: int = 0
    quality: Any = None
    _all_frames: list[np.ndarray] = field(default_factory=list, repr=False)
    _all_times: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.frames and not self.width:
            self.height, self.width = self.frames[0].shape[:2]
        if not self.duration and self.times:
            self.duration = float(self.times[-1]) + 1.0 / max(self.fps, 1e-6)

    def frame(self, i: int) -> np.ndarray:
        return self.frames[i]

    def frame_at(self, t: float) -> np.ndarray:
        frames, times = (self._all_frames, self._all_times) if self._all_frames else (self.frames, self.times)
        j = int(np.argmin([abs(x - t) for x in times]))
        return frames[j]

    @classmethod
    def from_file(cls, path: str | Path, sample_fps: float = 5.0, max_side: Optional[int] = None) -> "ArrayClip":
        """Простое чтение через cv2 (rotation из метаданных применяет OpenCV; время — из PTS). Для продакшна — ml.ingest."""
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"не удалось открыть {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        period = 1.0 / float(sample_fps)
        frames, times, k = [], [], 0
        next_t, last_pts, pts_ok = 0.0, -1.0, True
        while True:
            ok, f = cap.read()
            if not ok:
                break
            # время — из PTS (переменный fps); если бэкенд не отдаёт PTS — k/fps
            pts = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if pts_ok and k > 0 and pts <= last_pts:
                pts_ok = False
            t = pts if pts_ok else k / fps
            last_pts = pts
            if t + 1e-6 >= next_t:
                if max_side and max(f.shape[:2]) > max_side:
                    s = max_side / max(f.shape[:2])
                    f = cv2.resize(f, (int(f.shape[1] * s), int(f.shape[0] * s)), interpolation=cv2.INTER_AREA)
                frames.append(f)
                times.append(round(t, 4))
                next_t += period * max(1, int((t + 1e-6 - next_t) // period) + 1)
            k += 1
        cap.release()
        dur = (times[-1] + period) if times else 0.0
        return cls(frames=frames, times=times, path=str(path), fps=float(fps), duration=max(dur, k / fps if fps else 0.0))


def synth_clip(duration_s: float = 4.0, sample_fps: float = 5.0, width: int = 464, height: int = 848,
               objects: Optional[Sequence[SynthObject]] = None) -> tuple[ArrayClip, list[list[tuple[str, tuple]]]]:
    """Синтетический клип в памяти (без кодека) + GT по сэмплам."""
    objs = list(objects) if objects is not None else default_scene()
    n = int(math.floor(duration_s * sample_fps + 1e-9))
    frames, gts, times = [], [], []
    for k in range(n):
        t = k / sample_fps
        img, gt = render(objs, t, duration_s, width, height)
        frames.append(img)
        gts.append(gt)
        times.append(t)
    return ArrayClip(frames=frames, times=times, path="synth://default", fps=30.0, duration=duration_s), gts
