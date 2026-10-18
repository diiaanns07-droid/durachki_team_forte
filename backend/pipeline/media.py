"""Работа с видео на стороне оркестратора: probe, запасной декодер, потокобезопасный клип,
JPEG base64 для доказательств, ASCII-безопасные пути (Windows + кириллица), нормализация имён."""
from __future__ import annotations

import base64
import os
import shutil
import threading
import unicodedata
from pathlib import Path
from typing import Any, Optional

# Кириллические гомоглифы → латиница (Х-3 → X-3 и т.п.).
_HOMOGLYPHS = str.maketrans({
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C",
    "Т": "T", "У": "Y", "Х": "X", "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y",
    "х": "x",
})

VIDEO_EXTS = (".mov", ".mp4", ".avi", ".mkv", ".m4v", ".webm")


def normalize_video_name(name: str) -> str:
    """'Х-3.MOV' (кириллица) → 'X-3'. Регистр цифр/латиницы сохраняется, расширение отбрасывается."""
    stem = Path(unicodedata.normalize("NFC", name)).stem
    stem = stem.translate(_HOMOGLYPHS).strip()
    return "".join(ch if (ch.isascii() and (ch.isalnum() or ch in "-_.")) else "_" for ch in stem) or "video"


def probe_video(path: str) -> dict:
    """Быстрые метаданные через cv2 (без декодирования): duration, fps, frames, width, height."""
    try:
        import cv2  # noqa: WPS433
    except Exception:  # noqa: BLE001
        return {}
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return {}
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frames / fps if fps > 0 and frames > 0 else 0.0
        return {"duration": duration, "fps": fps, "frames": frames, "width": w, "height": h}
    finally:
        cap.release()


def ascii_safe_path(path: str, workdir: Path, mode: str = "auto", tag: str = "input") -> str:
    """На Windows OpenCV/FFmpeg не всегда открывают пути с кириллицей → хардлинк/копия с ASCII-именем."""
    if mode == "never" or (mode == "auto" and os.name != "nt"):
        return path
    try:
        str(Path(path).resolve()).encode("ascii")
        return path
    except UnicodeEncodeError:
        pass
    workdir.mkdir(parents=True, exist_ok=True)
    target = workdir / f"{tag}{Path(path).suffix.lower() or '.mov'}"
    if target.exists():
        try:
            if target.stat().st_size == Path(path).stat().st_size:
                return str(target)
            target.unlink()
        except OSError:
            pass
    try:
        os.link(path, target)
    except OSError:
        shutil.copyfile(path, target)
    return str(target)


def _pts_seconds(cap: Any, cv2: Any) -> Optional[float]:
    try:
        ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
    except Exception:  # noqa: BLE001
        return None
    return ms / 1000.0 if ms == ms and ms >= 0 else None


def read_frame_at_pts(path: str, t: float, preroll_s: float = 1.0) -> tuple[Any, Optional[float]]:
    """Кадр с PTS, ближайшим к t (переменный fps: индекс×fps врёт). Seek чуть раньше t, затем
    декодирование вперёд по PTS; если seek «перелетел» t — декодирование с начала файла."""
    try:
        import cv2  # noqa: WPS433
    except Exception:  # noqa: BLE001
        return None, None
    t = max(0.0, float(t))
    for start in (max(0.0, t - preroll_s), 0.0):
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                return None, None
            if start > 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
            best, best_pts, first = None, None, True
            while True:
                if not cap.grab():
                    break
                pts = _pts_seconds(cap, cv2)
                if pts is None:
                    break
                if first and start > 0 and pts > t + 1e-3:
                    best = "overshoot"
                    break
                first = False
                if best_pts is not None and abs(pts - t) > abs(best_pts - t):
                    break  # прошли ближайший кадр
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    best, best_pts = frame, pts
                if pts >= t:
                    break
            if isinstance(best, str):
                continue
            return best, best_pts
        finally:
            cap.release()
    return None, None


class SimpleClip:
    """Запасной декодер (если ml.ingest недоступен/упал). Сэмплированные кадры хранятся как JPEG
    в памяти (≈60 КБ/кадр), frame_at() делает точный seek. Поворот учитывает сам OpenCV ≥ 4.5."""

    def __init__(self, path: str, sample_fps: float = 5.0, jpeg_quality: int = 92) -> None:
        import cv2  # noqa: WPS433
        import numpy as np  # noqa: WPS433

        self.path = str(path)
        self._lock = threading.Lock()
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError(f"не удалось открыть видео: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        self.fps = fps
        step_s = 1.0 / max(0.1, float(sample_fps))
        half_frame = 0.5 / fps
        self.times: list[float] = []
        self._jpegs: list[bytes] = []
        next_pick = 0.0
        idx = 0
        last_pts = -1.0
        width = height = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            # время — только из PTS (переменный fps); индекс/fps — лишь запасной вариант
            pts = _pts_seconds(cap, cv2)
            if pts is None or pts <= last_pts:
                pts = (last_pts + 1.0 / fps) if last_pts >= 0 else idx / fps
            if pts + half_frame >= next_pick:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    height, width = frame.shape[:2]
                    okj, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                    if okj:
                        self.times.append(round(pts, 4))
                        self._jpegs.append(buf.tobytes())
                while next_pick <= pts + half_frame:
                    next_pick += step_s
            last_pts = pts
            idx += 1
        cap.release()
        if not self._jpegs:
            raise RuntimeError(f"в видео нет декодируемых кадров: {path}")
        self.n_source_frames = idx
        self.duration = round(last_pts + 1.0 / fps, 4)
        self.timebase = "pts"
        self.width = int(width)
        self.height = int(height)
        self._np = np

    def __len__(self) -> int:
        return len(self.times)

    def frame(self, i: int) -> Any:
        import cv2  # noqa: WPS433

        buf = self._np.frombuffer(self._jpegs[i], dtype=self._np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def frame_at(self, t: float) -> Any:
        with self._lock:
            frame, _pts = read_frame_at_pts(self.path, t)
        if frame is not None:
            return frame
        nearest = min(range(len(self.times)), key=lambda k: abs(self.times[k] - float(t)))
        return self.frame(nearest)


class LockedClip:
    """Потокобезопасная обёртка над VideoClip модуля ingest + общий кэш декодированных кадров.

    temporal читает кадры в своём потоке параллельно с detect/pose: обёртка сериализует декодирование
    (cv2.VideoCapture не потокобезопасен) и раздаёт уже декодированные кадры из памяти.
    Атрибуты/методы, кроме frame/frame_at, прозрачно проксируются к исходному клипу.
    """

    def __init__(self, clip: Any, cache_mb: float = 512.0) -> None:
        object.__setattr__(self, "_clip", clip)
        object.__setattr__(self, "_lock", threading.RLock())
        object.__setattr__(self, "_frames", {})
        object.__setattr__(self, "_bytes", 0)
        object.__setattr__(self, "_limit", int(max(0.0, cache_mb) * 1024 * 1024))
        object.__setattr__(self, "hits", 0)
        object.__setattr__(self, "misses", 0)

    @property
    def inner(self) -> Any:
        return self._clip

    def __getattr__(self, name: str) -> Any:
        return getattr(self._clip, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._clip, name, value)

    def __len__(self) -> int:
        return len(self._clip.times)

    def frame(self, i: int) -> Any:
        frames = self._frames
        cached = frames.get(i)
        if cached is not None:
            object.__setattr__(self, "hits", self.hits + 1)
            return cached
        with self._lock:
            cached = frames.get(i)
            if cached is not None:
                return cached
            img = self._clip.frame(i)
            object.__setattr__(self, "misses", self.misses + 1)
            nbytes = int(getattr(img, "nbytes", 0) or 0)
            if img is not None and self._bytes + nbytes <= self._limit:
                frames[i] = img
                object.__setattr__(self, "_bytes", self._bytes + nbytes)
            return img

    def frame_at(self, t: float) -> Any:
        with self._lock:
            return self._clip.frame_at(t)

    def cache_stats(self) -> dict:
        return {"cached_frames": len(self._frames), "cached_mb": round(self._bytes / 1048576, 1),
                "hits": self.hits, "misses": self.misses}


def encode_jpeg_b64(img: Any, max_side: int = 640, quality: int = 82) -> Optional[str]:
    if img is None:
        return None
    try:
        import cv2  # noqa: WPS433
    except Exception:  # noqa: BLE001
        return None
    h, w = img.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w))) if max_side else 1.0
    if scale < 1.0:
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def blur_regions(img: Any, boxes: list[tuple[float, float, float, float]], head_frac: float = 0.28) -> Any:
    """Размывает верхнюю часть (≈голову) каждого бокса человека. Боксы нормированы 0..1."""
    if img is None or not boxes:
        return img
    try:
        import cv2  # noqa: WPS433
    except Exception:  # noqa: BLE001
        return img
    out = img.copy()
    h, w = out.shape[:2]
    for x1, y1, x2, y2 in boxes:
        bw, bh = (x2 - x1) * w, (y2 - y1) * h
        if bw < 4 or bh < 4:
            continue
        # с потолочной камеры голова — верх бокса; берём квадрат по ширине, но не больше head_frac высоты
        side = min(bw, bh * max(head_frac, 0.5))
        cx = (x1 + x2) / 2 * w
        top = y1 * h
        xa, xb = int(max(0, cx - side / 2)), int(min(w, cx + side / 2))
        ya, yb = int(max(0, top)), int(min(h, top + side))
        if xb - xa < 2 or yb - ya < 2:
            continue
        roi = out[ya:yb, xa:xb]
        k = max(3, (min(roi.shape[:2]) // 3) | 1)
        out[ya:yb, xa:xb] = cv2.GaussianBlur(roi, (k, k), 0)
    return out


def make_synthetic_video(path: str | Path, seconds: float = 3.0, fps: float = 30.0,
                         size: tuple[int, int] = (232, 424), event_at: Optional[float] = None) -> str:
    """Синтетическое «кассовое» видео (портрет): стол, «сейф», движущийся «человек».
    Используется для прогрева и тестов; если event_at задан — около него резкое движение."""
    import cv2  # noqa: WPS433
    import numpy as np  # noqa: WPS433

    w, h = size
    path = str(path)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    n = max(1, int(round(seconds * fps)))
    rng = np.random.default_rng(0)
    for i in range(n):
        t = i / fps
        img = np.full((h, w, 3), 90, np.uint8)
        cv2.rectangle(img, (int(w * .1), int(h * .55)), (int(w * .9), int(h * .75)), (60, 110, 150), -1)  # стол
        cv2.rectangle(img, (int(w * .7), int(h * .1)), (int(w * .95), int(h * .3)), (130, 130, 130), -1)  # сейф
        x = int((0.2 + 0.5 * (0.5 + 0.5 * np.sin(t))) * w)
        if event_at is not None and abs(t - event_at) < 0.6:
            x = int(w * (0.15 + 0.7 * rng.random()))
        cv2.circle(img, (x, int(h * .45)), max(6, w // 12), (40, 60, 200), -1)  # «человек»
        img = cv2.add(img, rng.integers(0, 6, img.shape, dtype=np.uint8))
        writer.write(img)
    writer.release()
    return path
