"""Activity gating: motion_energy по сэмплированным кадрам и выбор ключевых кадров для дорогих стадий."""
from __future__ import annotations

from collections import deque
from typing import Any, Optional, Sequence

import numpy as np


def motion_from_quality(clip: Any, n: int) -> Optional[list[float]]:
    """Пытается взять motion_energy, посчитанный A04 (clip.quality). Форматы — терпимо к вариантам:
    clip.quality = {"motion_energy": [...]}, или [{"motion_energy": x}, ...], или объект с атрибутом."""
    q = getattr(clip, "quality", None)
    if q is None:
        return None
    try:
        if callable(q):
            q = q()
        seq = None
        if isinstance(q, dict):
            seq = q.get("motion_energy") or q.get("motion")
        elif hasattr(q, "motion_energy"):
            seq = q.motion_energy
        elif isinstance(q, (list, tuple)) and q and isinstance(q[0], dict):
            seq = [float(x.get("motion_energy", 0.0)) for x in q]
        if seq is None:
            return None
        seq = [float(x) for x in seq]
        if len(seq) != n:
            return None
        return seq
    except Exception:  # noqa: BLE001 — чужой формат не должен ронять детекцию
        return None


class MotionMeter:
    """Онлайн-оценка motion_energy (средняя |разница| миниатюр в оттенках серого, 0..1)."""

    def __init__(self, thumb_width: int = 64):
        self.thumb_width = int(thumb_width)
        self._prev: Optional[np.ndarray] = None

    def thumb(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        h, w = frame.shape[:2]
        tw = self.thumb_width
        th = max(1, int(round(h * tw / max(w, 1))))
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        small = cv2.resize(g, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)
        return cv2.GaussianBlur(small, (3, 3), 0)

    def update(self, frame: np.ndarray) -> float:
        cur = self.thumb(frame)
        if self._prev is None or self._prev.shape != cur.shape:
            self._prev = cur
            return 0.0
        e = float(np.mean(np.abs(cur - self._prev)) / 255.0)
        self._prev = cur
        return e


class KeyframeSelector:
    """Онлайн-выбор ключевых кадров: регулярная сетка (interval_s) + всплески движения."""

    def __init__(self, interval_s: float = 1.0, abs_thr: float = 0.035, rel_k: float = 2.5,
                 min_gap_s: float = 0.4, window: int = 25, max_frames: int = 10**9):
        self.interval_s = float(interval_s)
        self.abs_thr = float(abs_thr)
        self.rel_k = float(rel_k)
        self.min_gap_s = float(min_gap_s)
        self.max_frames = int(max_frames)
        self._hist: deque[float] = deque(maxlen=max(3, int(window)))
        self._next_grid = 0.0
        self._last_t = -1e9
        self.count = 0

    def is_motion_spike(self, energy: float) -> bool:
        med = float(np.median(self._hist)) if len(self._hist) >= 3 else 0.0
        return energy >= self.abs_thr or (med > 1e-6 and energy >= self.rel_k * med and energy >= 0.3 * self.abs_thr)

    def offer(self, t: float, energy: float) -> tuple[bool, str]:
        """Возвращает (выбран ли кадр, причина: grid|motion|'')."""
        spike = self.is_motion_spike(energy)
        self._hist.append(float(energy))
        if self.count >= self.max_frames:
            return False, ""
        if t + 1e-6 >= self._next_grid:
            self._next_grid = t + self.interval_s
            self._last_t = t
            self.count += 1
            return True, "grid"
        if spike and t - self._last_t >= self.min_gap_s:
            self._last_t = t
            self.count += 1
            return True, "motion"
        return False, ""


def select_keyframes(times: Sequence[float], energies: Sequence[float], **kw: Any) -> list[int]:
    """Офлайн-вариант (для тестов/аналитики)."""
    sel = KeyframeSelector(**kw)
    return [i for i, (t, e) in enumerate(zip(times, energies)) if sel.offer(float(t), float(e))[0]]
