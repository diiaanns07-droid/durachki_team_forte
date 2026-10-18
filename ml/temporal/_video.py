"""Минимальный загрузчик видео (fallback, если ml.ingest ещё не смерджен). Реализует протокол VideoClip."""
from __future__ import annotations

from typing import Any, Optional


class SimpleClip:
    """Кадры сэмплируются один раз при создании (держим в памяти уменьшенными до ``max_side``)."""

    def __init__(self, path: str, sample_fps: float = 5.0, max_side: int = 640) -> None:
        import cv2  # лениво
        import numpy as np

        self.path = str(path)
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise IOError(f"не удалось открыть видео: {path}")
        try:
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # учесть rotation из метаданных
        except Exception:  # pragma: no cover - старые сборки cv2
            pass
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = fps
        step = 1.0 / max(sample_fps, 1e-6)
        frames: list[Any] = []
        times: list[float] = []
        next_t = 0.0
        idx = 0
        last: Optional[Any] = None
        pts_ok = False
        last_t = -1.0
        while True:
            ok = cap.grab()
            if not ok:
                break
            pts = cap.get(cv2.CAP_PROP_POS_MSEC)  # PTS кадра: при переменном fps idx/fps врёт
            if pts and pts > 0:
                pts_ok = True
            t = pts / 1000.0 if pts_ok else idx / fps
            if t <= last_t:  # защита от немонотонных PTS
                t = last_t + 1e-3
            last_t = t
            if t + 0.5 / fps >= next_t:
                ok, fr = cap.retrieve()
                if ok and fr is not None:
                    h, w = fr.shape[:2]
                    s = max_side / max(h, w)
                    if s < 1.0:
                        fr = cv2.resize(fr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
                    frames.append(fr)
                    times.append(round(t, 4))
                    last = fr
                    next_t += step
            idx += 1
        cap.release()
        if last is None:
            raise IOError(f"в видео нет кадров: {path}")
        self._frames = frames
        self.times = times
        self.height, self.width = int(last.shape[0]), int(last.shape[1])
        self.duration = max(last_t + 1.0 / fps, times[-1]) if pts_ok else max(n / fps if n > 0 else 0.0, idx / fps)
        self.time_source = "pts" if pts_ok else "index/fps"
        self._np = np

    def frame(self, i: int) -> Any:
        return self._frames[i]

    def frame_at(self, t: float) -> Any:
        np = self._np
        i = int(np.argmin(np.abs(np.asarray(self.times) - t)))
        return self._frames[i]


def load_clip(path: str, sample_fps: float = 4.0) -> Any:
    """ml.ingest.load_video, если доступен; иначе SimpleClip."""
    try:
        from ml.ingest import load_video  # type: ignore

        return load_video(path, sample_fps=sample_fps)
    except ImportError:
        return SimpleClip(path, sample_fps=sample_fps)
