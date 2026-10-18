"""Синтетика для тестов (своих и чужих модулей) — без реальных данных.

* :func:`write_test_video` — .mov/.mp4 через cv2.VideoWriter: «стол», «сейф», движущийся
  «человек» и штрих-код номера кадра в верхней полосе (переживает сжатие);
* :func:`read_index_code` — прочитать номер кадра обратно (проверка таймкодов/seek);
* :func:`make_synthetic_clip` / ``load_video("synthetic://10")`` — Clip в памяти без файла.
"""
from __future__ import annotations

import math
import re
from typing import Any, Optional, Sequence

import numpy as np

CODE_BITS = 14  # до 16383 кадров (9 мин при 30 fps)


def _code_geometry(width: int, height: int, bits: int = CODE_BITS) -> tuple[int, int, int]:
    cell = max(4, width // (bits + 2))
    band = max(8, min(height // 10, cell))
    return cell, band, (width - cell * bits) // 2


def draw_index_code(img: np.ndarray, index: int, bits: int = CODE_BITS) -> np.ndarray:
    """Нарисовать номер кадра бинарным кодом (белый=1) в верхней полосе кадра (in place)."""
    h, w = img.shape[:2]
    cell, band, x0 = _code_geometry(w, h, bits)
    img[:band, :] = 0
    for b in range(bits):
        if (int(index) >> (bits - 1 - b)) & 1:
            img[2 : band - 2, x0 + b * cell + 2 : x0 + (b + 1) * cell - 2] = 255
    return img


def read_index_code(img: np.ndarray, bits: int = CODE_BITS) -> int:
    """Обратное к :func:`draw_index_code` (кадр в исходной ориентации)."""
    h, w = img.shape[:2]
    cell, band, x0 = _code_geometry(w, h, bits)
    gray = img.mean(axis=2) if img.ndim == 3 else img
    value = 0
    for b in range(bits):
        patch = gray[band // 4 : band - band // 4, x0 + b * cell + cell // 4 : x0 + (b + 1) * cell - cell // 4]
        value = (value << 1) | int(float(patch.mean()) > 127)
    return value


def render_scene(index: int, t: float, size: tuple[int, int] = (464, 848), code: bool = True,
                 marker: bool = True) -> np.ndarray:
    """Кадр синтетической «кассы»: размер (w, h); простые фигуры, без реальных данных."""
    import cv2

    w, h = size
    img = np.full((h, w, 3), 70, dtype=np.uint8)
    cv2.rectangle(img, (int(0.1 * w), int(0.45 * h)), (int(0.9 * w), int(0.65 * h)), (120, 90, 60), -1)  # стол
    cv2.rectangle(img, (int(0.65 * w), int(0.7 * h)), (int(0.95 * w), int(0.95 * h)), (150, 150, 150), -1)  # сейф
    cx = int((0.2 + 0.6 * (0.5 + 0.5 * math.sin(t * 1.3))) * w)
    cy = int(0.3 * h)
    cv2.circle(img, (cx, cy), max(6, w // 14), (60, 160, 220), -1)  # «человек»
    if marker:  # асимметричная метка в правом нижнем углу — для проверки поворотов/зеркала
        cv2.rectangle(img, (w - w // 6, h - h // 10), (w - 4, h - 4), (0, 0, 255), -1)
    if code:
        draw_index_code(img, index)
    return img


def write_test_video(path: str, duration_s: float = 3.0, fps: float = 30.0, size: tuple[int, int] = (464, 848),
                     fourcc: str = "mp4v", code: bool = True) -> str:
    """Записать синтетическое видео (CFR) через cv2.VideoWriter. size = (width, height)."""
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), float(fps), (int(size[0]), int(size[1])))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter не открылся для {path} ({fourcc})")
    n = max(1, int(round(duration_s * fps)))
    for i in range(n):
        writer.write(render_scene(i, i / fps, size, code))
    writer.release()
    return str(path)


# ----------------------------------------------------------------------------- in-memory clip


class SyntheticClip:
    """Минимальная реализация VideoClip в памяти (для тестов модулей без видеофайлов)."""

    def __init__(self, duration_s: float = 10.0, sample_fps: float = 5.0, fps: float = 30.0,
                 size: tuple[int, int] = (464, 848), frames: Optional[Sequence[np.ndarray]] = None) -> None:
        self.path = f"synthetic://{duration_s:g}"
        self.fps = float(fps)
        self.sample_fps = float(sample_fps)
        if frames is not None:
            self._frames = [np.ascontiguousarray(f) for f in frames]
            self.height, self.width = self._frames[0].shape[:2]
            self.times = [i / sample_fps for i in range(len(self._frames))]
            self.duration = len(self._frames) / sample_fps
        else:
            self.width, self.height = int(size[0]), int(size[1])
            self.duration = float(duration_s)
            n = max(1, int(math.floor(duration_s * sample_fps + 1e-9)))
            src = [int(round(i * fps / sample_fps)) for i in range(n)]
            self.times = [k / fps for k in src]
            self._frames = [render_scene(k, k / fps, (self.width, self.height)) for k in src]
        self.source_times = list(self.times)
        self.quality = [{"t": t, "motion_energy": 0.0, "sharpness": 100.0, "brightness": 0.3} for t in self.times]
        # МОК: явная пометка, чтобы его нельзя было принять за анализ реального видео
        self.meta: dict[str, Any] = {"backend": "synthetic", "mock": True, "warnings": ["MOCK: синтетический клип"],
                                     "complete": True, "coverage": 1.0, "coverage_gaps": []}

    def __len__(self) -> int:
        return len(self.times)

    def frame(self, i: int) -> np.ndarray:
        return self._frames[i].copy()

    def frame_at(self, t: float) -> np.ndarray:
        return self.frame_at_ts(t)[0]

    def frame_at_ts(self, t: float) -> tuple[np.ndarray, float]:
        i = min(range(len(self.times)), key=lambda k: abs(self.times[k] - t))
        return self._frames[i].copy(), self.times[i]

    def snap_time(self, t: float) -> float:
        return self.frame_at_ts(t)[1]

    def close(self) -> None:
        pass


def make_synthetic_clip(duration_s: float = 10.0, sample_fps: float = 5.0, **kw: Any) -> SyntheticClip:
    return SyntheticClip(duration_s=duration_s, sample_fps=sample_fps, **kw)


def synthetic_clip_from_uri(uri: str, sample_fps: float = 5.0) -> SyntheticClip:
    """``synthetic://<секунды>`` -> SyntheticClip (по умолчанию 10 с)."""
    m = re.match(r"synthetic://([0-9.]+)?", uri)
    duration = float(m.group(1)) if m and m.group(1) else 10.0
    return SyntheticClip(duration_s=duration, sample_fps=sample_fps if sample_fps and sample_fps > 0 else 5.0)


# ----------------------------------------------------------------------------- «съёмка с миганием»


def write_blink_recording(path: str, scene_lengths: Sequence[float], *, blink_s: float = 1.0, preamble_s: float = 2.0,
                          size: tuple[int, int] = (240, 432), fps: float = 30.0,
                          scenario_dark: Optional[dict[int, tuple[float, float]]] = None,
                          flicker_scene: Optional[int] = None) -> list[dict[str, float]]:
    """Синтетическая запись для split_blinks: подготовка, затем для каждой сцены — «мигание»
    (свет выключен blink_s) и сама сцена. ``scenario_dark={k: (offset, length)}`` — длинное
    выключение света внутри сцены k (НЕ метка); ``flicker_scene`` — метка этой сцены «двойная»
    (короткое мерцание внутри провала). Возвращает ожидаемые off_t/on_t/end_t по сценам."""
    import cv2

    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (int(w), int(h)))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter не открылся: {path}")
    rng = np.random.default_rng(0)
    plan: list[tuple[str, float, int]] = [("bright", preamble_s, -1)]
    for k, length in enumerate(scene_lengths):
        plan += [("blink", blink_s, k), ("scene", float(length), k)]
    expected: list[dict[str, float]] = []
    n = 0
    for kind, length, k in plan:
        frames = int(round(length * fps))
        if kind == "blink":
            expected.append({"off_t": n / fps, "on_t": (n + frames) / fps})
        for j in range(frames):
            img = render_scene(n, n / fps, (w, h), code=False)
            dark = kind == "blink"
            if kind == "blink" and flicker_scene == k and frames // 2 - 2 <= j < frames // 2 + 1:
                dark = False  # мерцание: на 0.1 с свет «моргнул» внутри метки
            if kind == "scene" and scenario_dark and k in scenario_dark:
                off, ln = scenario_dark[k]
                dark = off <= j / fps < off + ln
            if dark:
                gain = 0.12 + 0.15 * min(1.0, j / max(frames, 1))  # автоэкспозиция «вытягивает» темноту
                img = np.clip(img.astype(np.float32) * gain + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
            writer.write(img)
            n += 1
        if kind == "scene":
            expected[-1]["end_t"] = n / fps
    writer.release()
    return expected
