"""Mock-бэкенд (SENTINEL_MOCK=1): детерминированные детекции по цветовым маскам синтетики.

Палитра BGR — в config.yaml (mock.palette), генератор синтетики — ml/detect/synth.py.
На реальном видео (палитры нет) опционально выдаёт person по крупным движущимся пятнам,
чтобы e2e-прогон без весов давал правдоподобные треки.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from ..boxes import RawDets


class MockColorDetector:
    def __init__(self, palette: dict[str, Sequence[int]], tolerance: int = 28, min_area_px: int = 30,
                 score: float = 0.9, labels: Optional[Iterable[str]] = None, motion_person: bool = False,
                 motion_person_min_area: float = 0.02, name: str = "mock"):
        allowed = set(labels) if labels is not None else None
        self.palette = {k: np.asarray(v, dtype=np.int16) for k, v in palette.items()
                        if allowed is None or k in allowed}
        self.all_colors = [np.asarray(v, dtype=np.int16) for v in palette.values()]
        self.tol = int(tolerance)
        self.min_area_px = int(min_area_px)
        self.score = float(score)
        self.motion_person = bool(motion_person)
        self.motion_min_area = float(motion_person_min_area)
        self.name = name
        self._bg: dict[tuple[int, int], np.ndarray] = {}  # фон по размеру входа (кадр/тайл)

    def reset(self) -> None:
        self._bg = {}

    def _components(self, mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
        import cv2

        n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        out = []
        for j in range(1, n):
            x, y, w, h, a = (int(v) for v in stats[j])
            if a >= self.min_area_px:
                out.append((x, y, w, h, a))
        return out

    def _has_palette(self, f16: np.ndarray) -> bool:
        for c in self.all_colors:
            if np.any(np.all(np.abs(f16 - c) <= self.tol, axis=-1)):
                return True
        return False

    def _motion_persons(self, frame: np.ndarray) -> RawDets:
        import cv2

        h, w = frame.shape[:2]
        tw = 96
        th = max(1, int(round(h * tw / w)))
        g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)
        bg = self._bg.get((h, w))
        if bg is None:
            self._bg[(h, w)] = g
            return RawDets()
        diff = np.abs(g - bg) > 25
        self._bg[(h, w)] = 0.9 * bg + 0.1 * g
        m = cv2.morphologyEx(diff.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        boxes, scores = [], []
        for j in range(1, n):
            x, y, bw, bh, a = (int(v) for v in stats[j])
            if a / float(tw * th) >= self.motion_min_area:
                boxes.append([x / tw, y / th, (x + bw) / tw, (y + bh) / th])
                scores.append(0.5)
        if not boxes:
            return RawDets()
        return RawDets(np.asarray(boxes, np.float32), np.asarray(scores, np.float32),
                       ["person"] * len(boxes), [f"{self.name}:motion"] * len(boxes))

    def predict_one(self, frame: np.ndarray) -> RawDets:
        h, w = frame.shape[:2]
        f16 = frame.astype(np.int16)
        boxes, scores, labels = [], [], []
        for label, color in self.palette.items():
            mask = np.all(np.abs(f16 - color) <= self.tol, axis=-1)
            if not mask.any():
                continue
            for x, y, bw, bh, _a in self._components(mask):
                boxes.append([x / w, y / h, (x + bw) / w, (y + bh) / h])
                scores.append(self.score)
                labels.append(label)
        if not boxes:
            if self.motion_person and not self._has_palette(f16):
                return self._motion_persons(frame)
            return RawDets()
        return RawDets(np.asarray(boxes, np.float32), np.asarray(scores, np.float32), labels,
                       [self.name] * len(labels))

    def predict(self, frames: Sequence[np.ndarray]) -> list[RawDets]:
        return [self.predict_one(f) for f in frames]


class MockOpenVocabScorer:
    """Mock сырого open-vocab скоринга (только тесты/демо, SENTINEL_MOCK=1): компоненты цветов палитры →
    боксы; скор запроса = high, если цвет компоненты входит в группу запроса, иначе low."""

    def __init__(self, texts: Sequence[str], text_colors: dict[str, list[Sequence[int]]], tolerance: int = 28,
                 min_area_px: int = 30, high: float = 0.9, low: float = 0.05, name: str = "mock:owl"):
        self.texts = list(texts)
        self.text_colors = {t: [np.asarray(c, np.int16) for c in cs] for t, cs in text_colors.items()}
        self.colors: list[np.ndarray] = []
        for cs in self.text_colors.values():
            for c in cs:
                if not any(np.array_equal(c, x) for x in self.colors):
                    self.colors.append(c)
        self.tol, self.min_area_px, self.high, self.low, self.name = int(tolerance), int(min_area_px), high, low, name

    def score(self, frames: Sequence[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        import cv2

        out = []
        for f in frames:
            h, w = f.shape[:2]
            f16 = f.astype(np.int16)
            boxes, probs = [], []
            for c in self.colors:
                mask = np.all(np.abs(f16 - c) <= self.tol, axis=-1).astype(np.uint8)
                n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                for j in range(1, n):
                    x, y, bw, bh, a = (int(v) for v in stats[j])
                    if a < self.min_area_px:
                        continue
                    boxes.append([x / w, y / h, (x + bw) / w, (y + bh) / h])
                    probs.append([self.high if any(np.array_equal(c, tc) for tc in self.text_colors.get(t, []))
                                  else self.low for t in self.texts])
            out.append((np.asarray(boxes, np.float32).reshape(-1, 4),
                        np.asarray(probs, np.float32).reshape(-1, len(self.texts))))
        return out
