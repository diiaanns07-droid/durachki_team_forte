"""Загрузка видео -> :class:`Clip` (реализация контрактного ``VideoClip``).

Ключевые решения:
  * время кадра — только из presentation timestamps (PTS) потока, ``t = pts*time_base - t0``,
    где t0 — PTS первого выданного декодером кадра (первый кадр t=0). ``frame_index / fps``
    временем НЕ считается (VFR, пропуски, edit lists);
  * сэмплинг: для сетки k / sample_fps берётся ближайший по PTS кадр (без дублей);
    ``clip.times`` — фактические PTS выбранных кадров, а не узлы сетки;
  * поворот: display matrix / тег rotate (как autorotate ffmpeg), размер берётся из
    декодированного кадра — никаких зашитых 464x848;
  * сэмплы хранятся JPEG-байтами в RAM (лимит ``storage.max_ram_mb``, сверх — temp-файл на диске),
    декод по запросу через LRU (лимит по числу и по МБ);
  * ``frame_at(t)`` — точный кадр из исходника в полном разрешении (контракт: только картинка);
    ``frame_at_ts(t)`` / ``read_exact(t)`` дополнительно возвращают фактический PTS кадра —
    именно его нужно отдавать в ``EvidenceFrame.timestamp_sec``;
  * ошибки: нет кадров / слишком много битых пакетов / декодировано < ``decode.min_coverage``
    заявленной длительности -> :class:`DecodeError`, а не «пустой успешный» Clip.
"""
from __future__ import annotations

try:
    from contracts.sentinel_contracts import *  # noqa: F401,F403
except ImportError:  # пока ветка A01 не смерджена
    from contracts.sentinel_contracts import *  # noqa: F401,F403

import bisect
import math
import os
import tempfile
import threading
import time
import weakref
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

import numpy as np

from . import quality as Q
from .config import load_config
from .errors import DecodeError, FrameGapError, IngestError, NoVideoStreamError, VideoNotFoundError, VideoProbeError
from .orientation import IDENTITY, Orientation
from .probe import probe_and_validate, pyav_frame_orientation

ProgressFn = Callable[[float], None]
_EPS = 1e-9


@dataclass(frozen=True)
class FrameRef:
    """Точный кадр + его фактический таймкод (для доказательств)."""

    image: np.ndarray
    timestamp_sec: float  # фактический PTS выбранного кадра, сек от первого кадра (без округления)
    requested_sec: float  # что просили
    in_gap: bool = False        # |timestamp − requested| > frame_at.max_offset_s: в этот момент кадра нет
    out_of_range: bool = False  # requested вне [0, duration]: отдан крайний кадр

    @property
    def offset_sec(self) -> float:
        return self.timestamp_sec - self.requested_sec

    @property
    def ok(self) -> bool:
        return not (self.in_gap or self.out_of_range)


# ----------------------------------------------------------------------------- caches / storage


class _LRU:
    """LRU с лимитом по числу элементов и по байтам. Потокобезопасный."""

    def __init__(self, max_items: int, max_bytes: int) -> None:
        self.max_items = max(0, int(max_items))
        self.max_bytes = max(0, int(max_bytes))
        self._data: OrderedDict[Any, tuple[Any, int]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> Any:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return item[0]

    def put(self, key: Any, value: Any, nbytes: int) -> None:
        if self.max_items == 0 or nbytes > self.max_bytes > 0:
            return
        with self._lock:
            old = self._data.pop(key, None)
            if old is not None:
                self._bytes -= old[1]
            self._data[key] = (value, nbytes)
            self._bytes += nbytes
            while self._data and (len(self._data) > self.max_items or (self.max_bytes and self._bytes > self.max_bytes)):
                _, (_, nb) = self._data.popitem(last=False)
                self._bytes -= nb

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._bytes = 0

    def info(self) -> dict[str, int]:
        with self._lock:
            return {"items": len(self._data), "bytes": self._bytes, "hits": self.hits, "misses": self.misses,
                    "max_items": self.max_items, "max_bytes": self.max_bytes}


def _have_cv2() -> bool:
    try:
        import cv2  # noqa: F401

        return True
    except ImportError:  # pragma: no cover
        return False


class _SampleStore:
    """Сэмплы по порядку: ('raw', ndarray) | ('mem', jpeg bytes) | ('disk', (offset, length))."""

    def __init__(self, mode: str, jpeg_quality: int, max_ram_bytes: int) -> None:
        if mode not in ("jpeg", "raw", "disk"):
            mode = "jpeg"
        if mode != "raw" and not _have_cv2():  # pragma: no cover
            mode = "raw"
        self.mode = mode
        self.jpeg_quality = int(jpeg_quality)
        self.max_ram_bytes = int(max_ram_bytes)
        self._items: list[tuple[str, Any]] = []
        self.ram_bytes = 0
        self.disk_bytes = 0
        self._spill = None
        self._spill_path: Optional[str] = None
        self._lock = threading.Lock()
        self._finalizer: Optional[weakref.finalize] = None

    # вызывается в рабочих потоках
    def encode(self, img: np.ndarray) -> Any:
        if self.mode == "raw":
            return img
        return self._jpeg(img)

    def _jpeg(self, img: np.ndarray) -> bytes:
        import cv2

        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:  # pragma: no cover
            raise DecodeError("JPEG-кодирование кадра не удалось")
        return buf.tobytes()

    def append(self, payload: Any) -> None:
        if isinstance(payload, np.ndarray):
            if self.ram_bytes + payload.nbytes <= self.max_ram_bytes or not _have_cv2():
                self._items.append(("raw", payload))
                self.ram_bytes += payload.nbytes
                return
            payload = self._jpeg(payload)  # лимит RAM: дальше храним сжатым
        if self.mode != "disk" and self.ram_bytes + len(payload) <= self.max_ram_bytes:
            self._items.append(("mem", payload))
            self.ram_bytes += len(payload)
            return
        self._items.append(("disk", self._write_spill(payload)))

    def _write_spill(self, data: bytes) -> tuple[int, int]:
        with self._lock:
            if self._spill is None:
                fd, self._spill_path = tempfile.mkstemp(prefix="sentinel_ingest_", suffix=".bin")
                self._spill = os.fdopen(fd, "w+b")
                self._finalizer = weakref.finalize(self, _close_spill, self._spill, self._spill_path)
            self._spill.seek(0, os.SEEK_END)
            offset = self._spill.tell()
            self._spill.write(data)
            self.disk_bytes += len(data)
            return offset, len(data)

    def _read_bytes(self, i: int) -> Any:
        kind, value = self._items[i]
        if kind == "disk":
            with self._lock:
                self._spill.seek(value[0])
                return self._spill.read(value[1])
        return value

    def get(self, i: int) -> np.ndarray:
        kind = self._items[i][0]
        data = self._read_bytes(i)
        if kind == "raw":
            return data
        import cv2

        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:  # pragma: no cover
            raise DecodeError(f"повреждён сохранённый сэмпл #{i}")
        return img

    def __len__(self) -> int:
        return len(self._items)

    def materialize(self) -> list[tuple[str, Any]]:
        """Все сэмплы в RAM (для pickle)."""
        return [(k if k != "disk" else "mem", self._read_bytes(i)) for i, (k, _) in enumerate(self._items)]

    def close(self) -> None:
        if self._finalizer is not None:
            self._finalizer()
            self._finalizer = None
        self._spill = None


def _close_spill(fh: Any, path: Optional[str]) -> None:
    try:
        fh.close()
    except Exception:
        pass
    if path:
        try:
            os.unlink(path)
        except OSError:
            pass


# ----------------------------------------------------------------------------- sampling


class _NearestSampler:
    """Для каждой цели сетки k / sample_fps выбирает ближайший по времени кадр (без повторов).

    Держит один предыдущий кадр, поэтому работает потоково. sample_fps <= 0 — каждый кадр.
    """

    def __init__(self, sample_fps: float) -> None:
        self.rate = float(sample_fps or 0.0)
        self.k = 0
        self.prev: Optional[tuple[float, Any]] = None
        self.last_t = -math.inf

    def push(self, t: float, obj: Any) -> list[tuple[float, Any]]:
        out: list[tuple[float, Any]] = []
        if self.rate <= 0:
            out.append((t, obj))
            self.last_t = t
            return out
        while True:
            target = self.k / self.rate
            if t < target - _EPS:
                break
            cand = (t, obj)
            if self.prev is not None and (target - self.prev[0]) <= (t - target) + _EPS:
                cand = self.prev
            if cand[0] > self.last_t + _EPS:
                out.append(cand)
                self.last_t = cand[0]
            self.k += 1
        self.prev = (t, obj)
        return out

    def finish(self, half_dt: float) -> list[tuple[float, Any]]:
        """Хвост: последний кадр берём, только если следующий узел сетки ещё внутри видео
        (не дальше полукадра после последнего PTS) — так же, как у OpenCV-бэкенда."""
        if self.rate > 0 and self.prev is not None and self.prev[0] > self.last_t + _EPS:
            target = self.k / self.rate
            if target - self.prev[0] <= half_dt + _EPS:
                self.last_t = self.prev[0]
                return [self.prev]
        return []


def _target_size(width: int, height: int, orient: Orientation, max_side: int) -> Optional[tuple[int, int]]:
    """Размер (в координатах потока, до поворота) для даунскейла сэмплов; None — без изменений."""
    if not max_side or max(width, height) <= max_side:
        return None
    scale = max_side / float(max(width, height))
    return max(2, int(round(width * scale))), max(2, int(round(height * scale)))


def _resize(img: np.ndarray, size: Optional[tuple[int, int]]) -> np.ndarray:
    if size is None or (img.shape[1], img.shape[0]) == size:
        return img
    import cv2

    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _ffmpeg_error_types(av: Any) -> tuple[type, ...]:
    err = getattr(getattr(av, "error", None), "FFmpegError", None) or getattr(av, "FFmpegError", None)
    return (err,) if isinstance(err, type) else (Exception,)


def _pick_stream(container: Any) -> Any:
    for stream in container.streams.video:
        disp = getattr(stream, "disposition", 0)
        try:
            if int(disp) & 0x400:  # AV_DISPOSITION_ATTACHED_PIC — обложка, не видео
                continue
        except (TypeError, ValueError):
            pass
        return stream
    return None


def _info_orientation(info: Mapping[str, Any]) -> Orientation:
    rot = int(info.get("rotation") or 0)
    flip = bool(info.get("hflip"))
    if not rot and not flip:
        return IDENTITY
    return Orientation(rot, flip, str(info.get("orientation_source") or "probe"))


# ----------------------------------------------------------------------------- decoders


@dataclass
class _DecodeStats:
    backend: str
    stream_index: int = 0
    t0_abs: float = 0.0
    orientation: Orientation = IDENTITY
    width: int = 0
    height: int = 0
    n_decoded: int = 0
    decode_errors: int = 0
    corrupt_frames: int = 0
    nonmonotonic: int = 0
    skip_nonref: bool = False
    skip_switched_at: Optional[float] = None
    error_samples: tuple[str, ...] = ()


def _decode_pyav(
    path: str,
    info: Mapping[str, Any],
    sample_fps: float,
    cfg: Mapping[str, Any],
    sink: Callable[[float, Callable[[], np.ndarray]], None],
    source_times: list[float],
    progress: Optional[ProgressFn],
) -> _DecodeStats:
    import av

    dcfg, scfg = cfg["decode"], cfg["storage"]
    ff_errors = _ffmpeg_error_types(av)
    try:
        container = av.open(path)
    except ff_errors as exc:
        raise VideoProbeError("Не удалось открыть видео (PyAV)", path=path, details={"error": str(exc)}) from exc
    stats = _DecodeStats(backend="pyav")
    errors: list[str] = []
    try:
        stream = _pick_stream(container)
        if stream is None:
            raise NoVideoStreamError("В файле нет видеопотока", path=path)
        stats.stream_index = int(stream.index)
        stream.thread_type = "AUTO"
        cc = stream.codec_context
        if int(dcfg.get("threads") or 0) > 0:
            cc.thread_count = int(dcfg["threads"])
        if dcfg.get("skip_nonref"):
            cc.skip_frame = "NONREF"
            stats.skip_nonref = True
        tb = stream.time_base
        declared = float(info.get("duration") or 0.0) or (float(stream.duration * tb) if stream.duration else 0.0)
        fps_hint = float(info.get("fps") or 0.0) or float(stream.average_rate or 0.0) or 30.0
        max_side = int(scfg.get("max_side") or 0)
        max_err = int(dcfg.get("max_decode_errors", 200))
        budget = float(dcfg.get("budget_s") or 0.0)
        adaptive = bool(dcfg.get("adaptive_skip")) and budget > 0

        sampler = _NearestSampler(sample_fps)
        target: Optional[tuple[int, int]] = None
        t0: Optional[float] = None
        last_abs: Optional[float] = None
        last_t = -math.inf
        wall0 = time.perf_counter()
        next_check = wall0 + 1.0

        def convert(fr: Any, size: Optional[tuple[int, int]], orient: Orientation) -> np.ndarray:
            # выполняется в пуле потоков: у каждого VideoFrame свой reformatter, sws_scale без GIL
            if size is not None:
                arr = fr.to_ndarray(format="bgr24", width=size[0], height=size[1], interpolation="AREA")
            else:
                arr = fr.to_ndarray(format="bgr24")
            return orient.apply(arr)

        def emit(selected: list[tuple[float, Any]]) -> None:
            for ts, fr in selected:
                sink(ts, lambda fr=fr, size=target, orient=stats.orientation: convert(fr, size, orient))

        for packet in container.demux(stream):
            try:
                frames = packet.decode()
            except ff_errors as exc:
                if isinstance(exc, getattr(av.error, "EOFError", ())):
                    break
                stats.decode_errors += 1
                if len(errors) < 5:
                    errors.append(str(exc)[:200])
                if stats.decode_errors > max_err:
                    raise DecodeError(
                        f"Слишком много ошибок декодирования (> {max_err}): файл повреждён",
                        path=path, details={"errors": errors},
                    ) from exc
                continue
            for fr in frames:
                stats.n_decoded += 1
                if fr.pts is not None:
                    t_abs = float(fr.pts * tb)
                elif last_abs is not None:
                    t_abs = last_abs + 1.0 / fps_hint
                else:
                    t_abs = 0.0
                last_abs = t_abs
                if t0 is None:
                    t0 = t_abs
                    stats.t0_abs = t0
                    orient = pyav_frame_orientation(fr)
                    stats.orientation = orient if not orient.is_identity else _info_orientation(info)
                    stats.width, stats.height = stats.orientation.display_size(fr.width, fr.height)
                    target = _target_size(fr.width, fr.height, stats.orientation, max_side)
                if getattr(fr, "is_corrupt", False):
                    stats.corrupt_frames += 1
                    continue
                t = t_abs - t0
                if t <= last_t + _EPS:  # повтор/непорядок PTS — не используем для сэмплинга
                    stats.nonmonotonic += 1
                    continue
                last_t = t
                source_times.append(t)
                emit(sampler.push(t, fr))
            now = time.perf_counter()
            if now >= next_check and last_t > 0:
                next_check = now + 1.0
                if progress is not None and declared > 0:
                    progress(min(0.99, last_t / declared))
                if adaptive and not stats.skip_nonref and declared > 0:
                    projected = (now - wall0) / last_t * declared
                    if projected > budget:
                        cc.skip_frame = "NONREF"
                        stats.skip_nonref = True
                        stats.skip_switched_at = round(last_t, 3)
        emit(sampler.finish(0.5 / fps_hint))
    finally:
        stats.error_samples = tuple(errors)
        container.close()
    return stats


def _decode_opencv(
    path: str,
    info: Mapping[str, Any],
    sample_fps: float,
    cfg: Mapping[str, Any],
    sink: Callable[[float, Callable[[], np.ndarray]], None],
    source_times: list[float],
    progress: Optional[ProgressFn],
) -> _DecodeStats:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoProbeError("Не удалось открыть видео (OpenCV)", path=path)
    stats = _DecodeStats(backend="opencv")
    try:
        auto_off = hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO") and bool(cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0))
        fps_hint = float(info.get("fps") or 0.0) or float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        half_dt = 0.5 / fps_hint
        declared = float(info.get("duration") or 0.0)
        max_side = int(cfg["storage"].get("max_side") or 0)
        rate = float(sample_fps or 0.0)
        next_target = 0.0
        t0: Optional[float] = None
        last_t = -math.inf
        target: Optional[tuple[int, int]] = None
        last_progress = time.perf_counter()
        while True:
            if not cap.grab():
                break
            stats.n_decoded += 1
            t_abs = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
            if t0 is None:
                t0 = t_abs
                stats.t0_abs = t0
                ok, img = cap.retrieve()
                if not ok or img is None:
                    raise DecodeError("OpenCV: первый кадр не декодируется", path=path)
                h, w = img.shape[:2]
                orient = _info_orientation(info) if auto_off else IDENTITY
                if not auto_off and orient.is_identity and info.get("rotation"):
                    # старый OpenCV мог повернуть сам: сверяем размер с «сырым» размером потока
                    raw = (int(info.get("width") or 0), int(info.get("height") or 0))
                    if (w, h) == raw:
                        orient = _info_orientation(info)
                stats.orientation = orient
                stats.width, stats.height = orient.display_size(w, h)
                target = _target_size(w, h, orient, max_side)
                first_img = img
            else:
                first_img = None
            t = t_abs - t0
            if t <= last_t + _EPS:
                stats.nonmonotonic += 1
                continue
            last_t = t
            source_times.append(t)
            if rate <= 0 or t + half_dt >= next_target - _EPS:
                if first_img is not None:
                    img = first_img
                else:
                    ok, img = cap.retrieve()
                    if not ok or img is None:
                        stats.decode_errors += 1
                        continue
                sink(t, lambda img=img, size=target, orient=stats.orientation: orient.apply(_resize(img, size)))
                if rate > 0:
                    next_target = (math.floor((t + half_dt) * rate + _EPS) + 1) / rate
            now = time.perf_counter()
            if progress is not None and declared > 0 and now - last_progress > 1.0:
                last_progress = now
                progress(min(0.99, t / declared))
    finally:
        cap.release()
    return stats


_DECODERS = {"pyav": _decode_pyav, "opencv": _decode_opencv}


# ----------------------------------------------------------------------------- random access


class _PyAVReader:
    """Точные кадры по времени: seek к ключевому кадру <= t, декод вперёд до ближайшего PTS."""

    def __init__(self, path: str, stream_index: int, t0_abs: float, orient: Orientation, window_s: float) -> None:
        self.path, self.stream_index, self.t0, self.orient = path, stream_index, t0_abs, orient
        self.window = float(window_s)
        self._container: Any = None
        self._stream: Any = None
        self._it: Optional[Iterator[tuple[float, Any]]] = None
        self._prev: Optional[tuple[float, Any]] = None
        self._cur: Optional[tuple[float, Any]] = None

    def _open(self) -> None:
        if self._container is None:
            import av

            self._av = av
            self._errors = _ffmpeg_error_types(av)
            self._container = av.open(self.path)
            self._stream = self._container.streams[self.stream_index]
            self._stream.thread_type = "AUTO"

    def _frames(self) -> Iterator[tuple[float, Any]]:
        tb = self._stream.time_base
        for packet in self._container.demux(self._stream):
            try:
                frames = packet.decode()
            except self._errors:
                continue
            for fr in frames:
                if fr.pts is None or getattr(fr, "is_corrupt", False):
                    continue
                yield float(fr.pts * tb) - self.t0, fr

    def _seek(self, t: float) -> None:
        tb = self._stream.time_base
        for margin in (0.0, 1.0, 4.0, 15.0, None):
            start = None if margin is None else t - margin
            if start is None or start <= 0:
                offset = int(self._stream.start_time or 0)
            else:
                offset = int(math.floor((start + self.t0) / tb))
            self._container.seek(offset, stream=self._stream, backward=True, any_frame=False)
            self._it = self._frames()
            self._prev, self._cur = None, next(self._it, None)
            if self._cur is not None and (self._cur[0] <= t + _EPS or start is None or start <= 0):
                return

    def _can_continue(self, t: float) -> bool:
        if self._it is None or (self._cur is None and self._prev is None):
            return False
        lo = self._prev[0] if self._prev is not None else self._cur[0]
        hi = self._cur[0] if self._cur is not None else self._prev[0]
        return lo - _EPS <= t <= hi + self.window

    def read_one(self, t: float) -> tuple[np.ndarray, float]:
        self._open()
        if not self._can_continue(t):
            self._seek(t)
        while self._cur is not None and self._cur[0] < t:
            self._prev, self._cur = self._cur, next(self._it, None)
        cands = [c for c in (self._prev, self._cur) if c is not None]
        if not cands:
            raise DecodeError(f"Не удалось декодировать кадр около t={t:.3f} с", path=self.path)
        best = min(cands, key=lambda c: (abs(c[0] - t), c[0]))
        return self.orient.apply(best[1].to_ndarray(format="bgr24")), best[0]

    def close(self) -> None:
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
        self._container = self._stream = self._it = self._prev = self._cur = None


class _CV2Reader:
    """Фолбэк точного доступа через OpenCV (seek по POS_MSEC + декод вперёд)."""

    def __init__(self, path: str, t0_abs: float, orient: Orientation, fps: float) -> None:
        self.path, self.t0, self.orient = path, t0_abs, orient
        self.fps = fps or 30.0
        self._cap: Any = None

    def read_one(self, t: float) -> tuple[np.ndarray, float]:
        import cv2

        if self._cap is None:
            self._cap = cv2.VideoCapture(self.path)
            if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
                self._cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        cap = self._cap
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t + self.t0 - 1.0) * 1000.0)
        prev: Optional[tuple[float, np.ndarray]] = None
        best: Optional[tuple[float, np.ndarray]] = None
        for _ in range(int(self.fps * 4) + 8):
            if not cap.grab():
                break
            ft = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0 - self.t0
            ok, img = cap.retrieve()
            if not ok or img is None:
                continue
            cur = (ft, img)
            if ft >= t:
                best = min([c for c in (prev, cur) if c is not None], key=lambda c: abs(c[0] - t))
                break
            prev = cur
        if best is None:
            best = prev
        if best is None:
            raise DecodeError(f"Не удалось декодировать кадр около t={t:.3f} с", path=self.path)
        return self.orient.apply(best[1]), best[0]

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# ----------------------------------------------------------------------------- Clip


class Clip:
    """Видео с сэмплами (по умолчанию 5 fps). Реализует контрактный ``VideoClip``.

    Атрибуты контракта: path, duration, fps, width, height (после поворота), times.
    Дополнительно: sample_fps, quality (list[dict]), quality_summary, source_times,
    meta (probe, тайминги, предупреждения, бэкенд), warnings.
    """

    def __init__(
        self,
        *,
        path: str,
        duration: float,
        fps: float,
        width: int,
        height: int,
        times: list[float],
        sample_fps: float,
        quality: list[dict[str, Any]],
        source_times: list[float],
        store: _SampleStore,
        reader: Any,
        meta: dict[str, Any],
        config: Mapping[str, Any],
    ) -> None:
        self.path = path
        self.duration = float(duration)
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self.times = times
        self.sample_fps = float(sample_fps)
        self.quality = quality
        self.source_times = source_times
        self.meta = meta
        self._store = store
        self._reader = reader
        self._cfg = dict(config)
        self._init_runtime()

    def _init_runtime(self) -> None:
        s, f = self._cfg["storage"], self._cfg["frame_at"]
        self._copy = bool(s.get("copy_on_access", True))
        self._lru = _LRU(int(s.get("lru_size", 96)), int(float(s.get("lru_max_mb", 256)) * 2**20))
        self._exact = _LRU(int(f.get("cache_size", 32)), int(float(f.get("cache_mb", 128)) * 2**20))
        self._reader_lock = threading.Lock()

    # --- контракт ---------------------------------------------------------------------------
    def frame(self, i: int) -> np.ndarray:
        """BGR uint8 сэмпла i (индекс в ``times``). Из LRU — копия за доли мс."""
        n = len(self.times)
        i = int(i)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(f"индекс сэмпла {i} вне [0, {n})")
        img = self._lru.get(i)
        if img is None:
            img = self._store.get(i)
            self._lru.put(i, img, img.nbytes)
        return img.copy() if self._copy else img

    def frame_at(self, t: float) -> np.ndarray:
        """Точный кадр исходника, ближайший к t, в полном разрешении (контракт: только картинка).

        Фактический таймкод этого кадра: :meth:`frame_at_ts` / :meth:`read_exact`.
        """
        return self.read_exact(t).image

    # --- расширения (совместимы с контрактом) --------------------------------------------------
    def frame_at_ts(self, t: float) -> tuple[np.ndarray, float]:
        """(картинка, фактический PTS кадра в сек) — для EvidenceFrame.timestamp_sec."""
        ref = self.read_exact(t)
        return ref.image, ref.timestamp_sec

    def read_exact(self, t: float, strict: bool = False) -> FrameRef:
        """Точный кадр + фактический PTS. strict=True: нет кадра в момент t -> FrameGapError."""
        return self.frames_at([t], strict=strict)[0]

    def frames_at(self, times: Sequence[float], strict: bool = False) -> list[FrameRef]:
        """Точные кадры исходника (полный размер) для моментов ``times``; порядок ответа = порядок times.

        Цепочка: t -> ближайший PTS таймлайна загрузки (без округления) -> декод именно этого кадра.
        Запросы, попавшие в один и тот же кадр, декодируются один раз (одинаковый timestamp_sec).
        Вне [0, duration] — крайний кадр с ``out_of_range``; в пробеле покрытия — ``in_gap``;
        strict=True в обоих случаях бросает :class:`FrameGapError`.
        """
        req = [float(t) for t in times]
        max_off = float(self._cfg["frame_at"].get("max_offset_s", 0.25))
        st = self.source_times
        exact_timeline = bool(st) and bool(self.meta.get("source_times_complete", True))
        t_hi = st[-1] if st else math.inf

        def target_of(t: float) -> float:
            tc = min(max(t, 0.0), t_hi)
            return self.snap_time(tc) if exact_timeline else tc

        targets = [target_of(t) for t in req]
        found: dict[float, tuple[np.ndarray, float]] = {}
        missing: list[float] = []
        for tg in targets:
            if tg in found or tg in missing:
                continue
            hit = self._exact.get(tg)
            if hit is not None:
                found[tg] = hit
            else:
                missing.append(tg)
        if missing:
            with self._reader_lock:
                if self._reader is None:
                    raise DecodeError("Clip закрыт: точный доступ к кадрам недоступен", path=self.path)
                for tg in sorted(missing):
                    img, ts = self._reader.read_one(tg)
                    found[tg] = (img, ts)
                    self._exact.put(tg, (img, ts), img.nbytes)
        out = []
        for t, tg in zip(req, targets):
            img, ts = found[tg]
            oor = t < -_EPS or t > self.duration + _EPS
            gap = (not oor) and abs(ts - t) > max_off
            if strict and (oor or gap):
                where = "вне видео" if oor else "в пробеле покрытия"
                raise FrameGapError(
                    f"Нет кадра в t={t:.3f} с ({where}); ближайший кадр t={ts:.3f} с", path=self.path,
                    details={"requested_sec": t, "nearest_sec": ts, "duration": self.duration,
                             "gaps": self.meta.get("coverage_gaps", [])},
                )
            out.append(FrameRef(img.copy() if self._copy else img, float(ts), t, in_gap=gap, out_of_range=oor))
        return out

    def distinct_frames_at(self, times: Sequence[float], min_separation_s: float = 0.0,
                           strict: bool = False) -> list[FrameRef]:
        """Как :meth:`frames_at`, но без повторов одного и того же кадра (по фактическому PTS)
        и, при ``min_separation_s > 0``, без кадров ближе этого интервала. Порядок — как в times."""
        out: list[FrameRef] = []
        for ref in self.frames_at(times, strict=strict):
            if all(abs(ref.timestamp_sec - o.timestamp_sec) > max(min_separation_s, _EPS) for o in out):
                out.append(ref)
        return out

    def window_indices(self, t0: float, t1: float) -> list[int]:
        """Индексы сэмплов с t0 <= times[i] <= t1."""
        lo = bisect.bisect_left(self.times, t0 - _EPS)
        hi = bisect.bisect_right(self.times, t1 + _EPS)
        return list(range(lo, hi))

    @property
    def coverage(self) -> dict[str, Any]:
        """Полнота декодирования — проверять до вывода «нарушений нет»."""
        m = self.meta
        gaps = [(float(a), float(b)) for a, b in m.get("coverage_gaps", [])]
        return {"complete": bool(m.get("complete", True)), "coverage": float(m.get("coverage", 1.0)),
                "duration": float(self.duration),
                "decoded_span": float(m.get("decoded_span", self.duration)),
                "declared_duration": float(m.get("declared_duration", 0.0)),
                "gaps": gaps, "unknown_spans": [[a, b] for a, b in gaps],
                "source_times_complete": bool(m.get("source_times_complete", True))}

    def snap_time(self, t: float) -> float:
        """PTS ближайшего кадра исходника к t (по таймлайну, собранному при загрузке)."""
        st = self.source_times
        if not st:
            return float(t)
        j = bisect.bisect_left(st, t)
        cands = [st[k] for k in (j - 1, j) if 0 <= k < len(st)]
        return min(cands, key=lambda x: (abs(x - t), x))

    def index_at(self, t: float) -> int:
        """Индекс ближайшего сэмпла к t."""
        if not self.times:
            raise IndexError("нет сэмплов")
        j = bisect.bisect_left(self.times, t)
        cands = [k for k in (j - 1, j) if 0 <= k < len(self.times)]
        return min(cands, key=lambda k: (abs(self.times[k] - t), k))

    def __len__(self) -> int:
        return len(self.times)

    def iter_frames(self, indices: Optional[Sequence[int]] = None) -> Iterator[tuple[int, float, np.ndarray]]:
        for i in indices if indices is not None else range(len(self.times)):
            yield i, self.times[i], self.frame(i)

    # --- качество / активность ---------------------------------------------------------------
    @property
    def warnings(self) -> list[str]:
        return self.meta.setdefault("warnings", [])

    @property
    def quality_summary(self) -> dict[str, Any]:
        return self.meta.get("quality_summary", {})

    def motion_curve(self) -> list[tuple[float, float]]:
        """[(t, motion_energy 0..1)] — формат как AnalysisResult.timeline."""
        return [(q["t"], q["motion_energy"]) for q in self.quality]

    def activity_segments(self, **overrides: Any) -> list[tuple[float, float]]:
        a = {**self._cfg["activity"], **overrides}
        return Q.activity_segments(
            self.times, [q["motion_energy"] for q in self.quality], threshold=float(a["threshold"]),
            smooth_s=float(a["smooth_s"]), pad_s=float(a["pad_s"]), min_gap_s=float(a["min_gap_s"]),
            duration=self.duration,
        )

    def gated_indices(self, **overrides: Any) -> list[int]:
        """Индексы сэмплов для тяжёлых детекторов (чаще там, где есть движение)."""
        a = {**self._cfg["activity"], **overrides}
        segs = self.activity_segments(**{k: v for k, v in overrides.items() if k in ("threshold", "smooth_s", "pad_s", "min_gap_s")})
        return Q.gated_indices(self.times, segs, idle_every_n=int(a["idle_every_n"]), active_every_n=int(a["active_every_n"]))

    # --- служебное ---------------------------------------------------------------------------
    def cache_info(self) -> dict[str, Any]:
        return {"samples": self._lru.info(), "exact": self._exact.info(),
                "store": {"mode": self._store.mode, "ram_bytes": self._store.ram_bytes, "disk_bytes": self._store.disk_bytes}}

    def close(self) -> None:
        with self._reader_lock:
            if self._reader is not None:
                self._reader.close()
                self._reader = None
        self._lru.clear()
        self._exact.clear()
        self._store.close()

    def __enter__(self) -> "Clip":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = {k: v for k, v in self.__dict__.items() if k not in ("_lru", "_exact", "_reader_lock", "_reader", "_store")}
        state["_items"] = self._store.materialize()
        state["_store_cfg"] = (self._store.mode, self._store.jpeg_quality, self._store.max_ram_bytes)
        state["_reader_args"] = self.meta.get("_reader_args")
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        items = state.pop("_items")
        mode, q, cap = state.pop("_store_cfg")
        reader_args = state.pop("_reader_args", None)
        self.__dict__.update(state)
        store = _SampleStore(mode, q, cap)
        store._items = items
        store.ram_bytes = sum(v.nbytes if k == "raw" else len(v) for k, v in items)
        self._store = store
        self._reader = _make_reader(*reader_args) if reader_args else None
        self._init_runtime()

    def __repr__(self) -> str:
        return (f"Clip({os.path.basename(self.path)!r}, {self.duration:.2f}s, {self.width}x{self.height}, "
                f"fps={self.fps:.2f}, samples={len(self.times)}@{self.sample_fps:g}fps, backend={self.meta.get('backend')})")


def _make_reader(backend: str, path: str, stream_index: int, t0: float, rot: int, hflip: bool, fps: float, window: float) -> Any:
    orient = Orientation(rot, hflip, "clip")
    if backend == "pyav":
        return _PyAVReader(path, stream_index, t0, orient, window)
    return _CV2Reader(path, t0, orient, fps)


# ----------------------------------------------------------------------------- load_video


def _median_dt(times: Sequence[float], fallback: float) -> float:
    if len(times) < 2:
        return fallback
    return float(np.median(np.diff(np.asarray(times, dtype=np.float64))))


def find_gaps(source_times: Sequence[float], declared: float, dt: float, gap_factor: float = 3.0,
              min_gap_s: float = 0.25, edge_factor: float = 1.5) -> list[tuple[float, float]]:
    """Пробелы покрытия [(a, b)], сек: между соседними PTS (a — последний кадр до дыры, b — первый после),
    в начале (0, первый PTS) и в хвосте (конец декодированного, заявленная длительность).

    Внутренние — от ``max(min_gap_s, gap_factor*dt)`` (не путать с дрожанием VFR); голова и хвост —
    от ``edge_factor*dt``: там любой отсутствующий отрезок длиннее ~1.5 кадра — недекодированное содержимое."""
    if not source_times:
        return [(0.0, float(declared))] if declared > 0 else []
    thr = max(float(min_gap_s), float(gap_factor) * float(dt))
    edge = max(float(edge_factor) * float(dt), _EPS)
    st = np.asarray(source_times, dtype=np.float64)
    gaps: list[tuple[float, float]] = []
    if st[0] > edge:
        gaps.append((0.0, float(st[0])))
    d = np.diff(st)
    for k in np.nonzero(d > thr)[0]:
        gaps.append((float(st[k]), float(st[k + 1])))
    end = float(st[-1] + dt)
    if declared > 0 and declared - end > edge:
        gaps.append((end, float(declared)))
    return gaps


def _build_clip(
    path: str,
    info: dict[str, Any],
    sample_fps: float,
    cfg: Mapping[str, Any],
    backend: str,
    progress: Optional[ProgressFn],
) -> Clip:
    scfg, qcfg, dcfg = cfg["storage"], cfg["quality"], cfg["decode"]
    store = _SampleStore(str(scfg.get("mode", "jpeg")), int(scfg.get("jpeg_quality", 92)),
                         int(float(scfg.get("max_ram_mb", 512)) * 2**20))
    times: list[float] = []
    quality: list[dict[str, Any]] = []
    source_times: list[float] = []
    workers = max(0, int(scfg.get("encode_workers", 4)))
    pending: deque = deque()
    prev_gray: list[Optional[np.ndarray]] = [None]

    def process(make_img: Callable[[], np.ndarray]) -> tuple[Any, dict[str, float], np.ndarray]:
        img = make_img()
        metrics, gray = Q.frame_metrics(img, qcfg)
        return store.encode(img), metrics, gray

    def collect(t: float, result: tuple[Any, dict[str, float], np.ndarray]) -> None:
        payload, metrics, gray = result
        store.append(payload)
        entry: dict[str, Any] = {"t": t, **metrics, **Q.motion_metrics(prev_gray[0], gray, qcfg)}
        prev_gray[0] = gray
        quality.append(Q.finalize_flags(entry, qcfg))
        times.append(t)  # без округления: округляем только при показе

    pool = None
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ingest")

    def sink(t: float, make_img: Callable[[], np.ndarray]) -> None:
        if pool is None:
            collect(t, process(make_img))
            return
        pending.append((t, pool.submit(process, make_img)))
        while len(pending) > workers * 3:
            ts, fut = pending.popleft()
            collect(ts, fut.result())

    t_start = time.perf_counter()
    try:
        stats = _DECODERS[backend](path, info, sample_fps, cfg, sink, source_times, progress)
        while pending:
            ts, fut = pending.popleft()
            collect(ts, fut.result())
    except BaseException:
        store.close()
        raise
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    decode_s = time.perf_counter() - t_start

    if not times or not source_times:
        store.close()
        raise DecodeError("Не удалось декодировать ни одного кадра: файл повреждён или неполный", path=path,
                          details={"backend": backend, "decode_errors": stats.decode_errors,
                                   "errors": list(stats.error_samples)})

    Q.fix_first_motion(quality, qcfg)
    fps_src = float(info.get("fps") or 0.0)
    measured_fps = (len(source_times) - 1) / source_times[-1] if len(source_times) > 1 and source_times[-1] > 0 else 0.0
    if fps_src <= 0:
        fps_src = measured_fps or 30.0
    dt = _median_dt(source_times, 1.0 / fps_src)
    decoded_span = source_times[-1] + dt
    declared = float(info.get("duration") or 0.0)
    coverage = min(1.0, decoded_span / declared) if declared > 0 else 1.0
    warnings: list[str] = list(info.get("warnings") or [])
    if declared > 0 and coverage < float(dcfg.get("min_coverage", 0.5)):
        store.close()
        raise DecodeError(
            f"Видео неполное: декодировано {decoded_span:.1f} с из заявленных {declared:.1f} с",
            path=path, code="incomplete_video",
            details={"decoded_s": decoded_span, "declared_s": declared, "coverage": coverage, "backend": backend},
        )
    complete = coverage >= float(dcfg.get("warn_coverage", 0.98))
    if not complete:
        warnings.append(f"видео неполное: декодировано {decoded_span:.1f} с из {declared:.1f} с ({coverage:.0%})")
    if stats.decode_errors:
        warnings.append(f"ошибок декодирования: {stats.decode_errors} (битые пакеты пропущены)")
    if stats.corrupt_frames:
        warnings.append(f"повреждённых кадров пропущено: {stats.corrupt_frames}")
    if stats.skip_switched_at is not None:
        warnings.append(f"медленный декод: с t={stats.skip_switched_at} с декодируются только опорные кадры")
    diffs = np.diff(np.asarray(source_times)) if len(source_times) > 2 else np.array([])
    vfr = bool(info.get("vfr_suspected")) or bool(diffs.size and float(diffs.std()) > 0.05 * max(float(diffs.mean()), 1e-6))
    summary, qwarn = Q.summarize(quality, qcfg)
    warnings.extend(qwarn)
    fa = cfg["frame_at"]
    gaps = find_gaps(source_times, declared, dt, float(fa.get("gap_factor", 3.0)), float(fa.get("min_gap_s", 0.25)))
    # при skip_nonref промежутки между опорными кадрами — норма; голова/хвост по-прежнему пробелы
    spans = gaps if not stats.skip_nonref else [g for g in gaps if g[0] <= _EPS or (declared > 0 and g[1] >= declared - _EPS)]
    if spans:
        total_gap = sum(b - a for a, b in spans)
        warnings.append(f"пробелы покрытия: {len(spans)} шт., всего {total_gap:.2f} с (кадров нет)")

    sample_h, sample_w = store.get(0).shape[:2] if len(store) else (0, 0)
    window = float(cfg["frame_at"].get("forward_window_s", 1.5))
    reader_args = (backend, path, stats.stream_index, stats.t0_abs, stats.orientation.rotation_cw,
                   stats.orientation.hflip, fps_src, window)
    meta: dict[str, Any] = {
        "backend": backend,
        "probe": info,
        "orientation": stats.orientation.to_dict(),
        "requested_sample_fps": float(sample_fps or 0.0),
        "n_samples": len(times),
        "n_source_frames": len(source_times),
        "n_decoded": stats.n_decoded,
        "decode_errors": stats.decode_errors,
        "corrupt_frames": stats.corrupt_frames,
        "nonmonotonic_pts": stats.nonmonotonic,
        "skip_nonref": stats.skip_nonref,
        "source_times_complete": not stats.skip_nonref,
        "measured_fps": measured_fps,
        "vfr": vfr,
        "t0_pts_sec": stats.t0_abs,
        "declared_duration": declared,
        "decoded_span": decoded_span,
        "coverage": coverage,
        "coverage_gaps": spans,
        "unknown_spans": [[a, b] for a, b in spans],  # = coverage_gaps в формате A01 ([[t0, t1], ...], сек)
        "median_frame_dt": dt,
        "complete": complete,
        "sample_size": [int(sample_w), int(sample_h)],
        "storage": {"mode": store.mode, "ram_bytes": store.ram_bytes, "disk_bytes": store.disk_bytes},
        "timings": {"decode_s": round(decode_s, 3)},
        "quality_summary": summary,
        "warnings": warnings,
        "_reader_args": reader_args,
    }
    # duration — НОМИНАЛЬНАЯ длительность видеопотока (контейнер), а не декодированный отрезок: недекодированный
    # хвост остаётся внутри [0, duration] и виден потребителю (A03 сравнивает duration с times) как
    # unknown_spans, а не исчезает. Контейнер занижает (декодировано больше) — берём декодированное.
    duration = max(declared, decoded_span) if declared > 0 else decoded_span
    return Clip(
        path=path, duration=duration, fps=fps_src, width=stats.width, height=stats.height, times=times,
        sample_fps=float(sample_fps or 0.0) or measured_fps, quality=quality, source_times=source_times,
        store=store, reader=_make_reader(*reader_args), meta=meta, config=cfg,
    )


def _backend_order(requested: str) -> list[str]:
    requested = (requested or "auto").lower()
    if requested in _DECODERS:
        return [requested]
    return ["pyav", "opencv"]


def load_video(
    path: str,
    sample_fps: float = 5.0,
    *,
    config: Optional[Mapping[str, Any]] = None,
    backend: Optional[str] = None,
    progress: Optional[ProgressFn] = None,
) -> Clip:
    """Открыть видео и подготовить сэмплы (контракт: ``load_video(path, sample_fps=5.0)``).

    Keyword-only расширения: ``config`` — частичный override config.yaml, ``backend`` —
    pyav|opencv|auto, ``progress(fraction)`` — прогресс декодирования.
    ``sample_fps <= 0`` — брать каждый кадр. Ошибки — подклассы :class:`IngestError`.
    """
    path = str(path)
    t_all = time.perf_counter()
    cfg = load_config(config)
    if path.startswith("synthetic://"):
        # мок: только явно (тесты/демо). Production сам на синтетику не переключается.
        if not (os.environ.get("SENTINEL_MOCK") == "1" or cfg.get("mock", {}).get("allow_synthetic_uri")):
            raise VideoNotFoundError("synthetic:// разрешён только в тестах/демо (SENTINEL_MOCK=1)", path=path)
        from .synthetic import synthetic_clip_from_uri

        return synthetic_clip_from_uri(path, sample_fps)
    t_probe = time.perf_counter()
    info = probe_and_validate(path, cfg)
    probe_s = time.perf_counter() - t_probe

    rate = float(sample_fps or 0.0)
    max_samples = int(cfg["sampling"].get("max_samples") or 0)
    duration = float(info.get("duration") or 0.0)
    extra_warn: list[str] = []
    if rate > 0 and max_samples and duration > 0 and duration * rate > max_samples:
        new_rate = max_samples / duration
        extra_warn.append(f"sample_fps снижен с {rate:g} до {new_rate:.3f} (лимит sampling.max_samples={max_samples})")
        rate = new_rate

    errors: list[dict[str, Any]] = []
    last: Optional[IngestError] = None
    for name in _backend_order(backend or str(cfg["decode"].get("backend", "auto"))):
        try:
            clip = _build_clip(path, info, rate, cfg, name, progress)
        except ImportError as exc:
            errors.append({"backend": name, "error": f"не установлен: {exc}"})
            continue
        except NoVideoStreamError:
            raise
        except IngestError as exc:
            errors.append({"backend": name, **exc.to_dict()})
            if last is None or (last.code != "incomplete_video" and exc.code == "incomplete_video"):
                last = exc
            continue
        clip.meta["timings"]["probe_s"] = round(probe_s, 3)
        clip.meta["timings"]["total_s"] = round(time.perf_counter() - t_all, 3)
        clip.meta["warnings"][:0] = extra_warn
        if errors:
            clip.meta["fallback_errors"] = errors
            clip.meta["warnings"].append(f"основной декодер не сработал, использован {name}")
        if progress is not None:
            progress(1.0)
        return clip
    if last is not None:
        last.details.setdefault("backends", errors)
        raise last
    raise DecodeError("Нет доступного декодера видео: установите PyAV (pip install av) или OpenCV",
                      path=path, details={"backends": errors})
