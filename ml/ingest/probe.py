"""Проба видеофайла: ffprobe (основной) -> PyAV -> OpenCV (фолбэки) + валидация.

    from ml.ingest.probe import probe, validate
    info = validate(probe("X-1.mov"))   # dict, либо IngestError с понятным сообщением

CLI: ``python -m ml.ingest.probe video.mov [--no-validate]`` печатает JSON.

Ключи результата: path, size_bytes, format_name, duration, fps, avg_fps, r_fps, nb_frames,
codec, pix_fmt, profile, bit_rate, width/height (как в потоке), rotation (по часовой, 0/90/180/270),
hflip, display_width/display_height (после поворота), has_audio, start_time, time_base,
vfr_suspected, tags, source (ffprobe|pyav|opencv), warnings.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional

from .errors import IngestError, InvalidDurationError, NoVideoStreamError, VideoNotFoundError, VideoProbeError
from .orientation import IDENTITY, Orientation, from_display_matrix, from_rotation_cw, matrix_from_ffprobe

_TAG_KEYS = (
    "creation_time",
    "com.apple.quicktime.make",
    "com.apple.quicktime.model",
    "com.apple.quicktime.software",
    "encoder",
    "handler_name",
    "major_brand",
)


def _env_bin(env_names: tuple[str, ...], exe: str) -> Optional[str]:
    for name in env_names:
        value = os.environ.get(name)
        if value and (Path(value).is_file() or shutil.which(value)):
            return value
    found = shutil.which(exe)
    if found:
        return found
    return None


def ffprobe_bin() -> Optional[str]:
    """Путь к ffprobe: $SENTINEL_FFPROBE | $FFPROBE_BIN | PATH."""
    return _env_bin(("SENTINEL_FFPROBE", "FFPROBE_BIN"), "ffprobe")


def ffmpeg_bin() -> Optional[str]:
    """Путь к ffmpeg: $SENTINEL_FFMPEG | $FFMPEG_BIN | PATH | imageio-ffmpeg (если установлен)."""
    found = _env_bin(("SENTINEL_FFMPEG", "FFMPEG_BIN"), "ffmpeg")
    if found:
        return found
    try:  # опционально: pip install imageio-ffmpeg (BSD-2, бинарь ffmpeg внутри)
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _ratio(value: Any) -> float:
    """'30000/1001' | '30/1' | '0/0' | Fraction | float -> float (0.0 если не определено)."""
    if value is None:
        return 0.0
    try:
        if isinstance(value, str):
            if "/" in value:
                num, den = value.split("/", 1)
                den_f = float(den)
                return float(num) / den_f if den_f else 0.0
            return float(value)
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _num(value: Any, cast=float) -> Optional[Any]:
    try:
        if value is None or value == "N/A":
            return None
        return cast(value)
    except (TypeError, ValueError):
        return None


def _check_file(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise VideoNotFoundError(f"Файл не найден: {path}", path=str(path))
    if not p.is_file():
        raise VideoNotFoundError(f"Это не файл: {path}", path=str(path))
    if p.stat().st_size == 0:
        raise VideoProbeError(f"Файл пустой (0 байт): {path}", path=str(path))
    return p


def _finalize(info: dict[str, Any], orient: Orientation) -> dict[str, Any]:
    w, h = int(info.get("width") or 0), int(info.get("height") or 0)
    dw, dh = orient.display_size(w, h)
    info["rotation"] = orient.rotation_cw
    info["hflip"] = orient.hflip
    info["orientation_source"] = orient.source
    info["display_width"], info["display_height"] = dw, dh
    avg, r = float(info.get("avg_fps") or 0.0), float(info.get("r_fps") or 0.0)
    info["fps"] = avg if avg > 0 else r
    # r_frame_rate — «базовый» темп, avg — средний: заметное расхождение = вероятный VFR
    info["vfr_suspected"] = bool(avg > 0 and r > 0 and abs(avg - r) / r > 0.02)
    info.setdefault("warnings", [])
    return info


def _probe_ffprobe(path: Path, timeout: float) -> Optional[dict[str, Any]]:
    exe = ffprobe_bin()
    if not exe:
        return None
    cmd = [exe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace").strip()
        raise VideoProbeError(
            "Не удалось прочитать файл как видео (повреждён или неподдерживаемый формат)",
            path=str(path),
            details={"ffprobe_stderr": stderr[-800:]},
        )
    try:
        data = json.loads(proc.stdout.decode("utf-8", "replace") or "{}")
    except ValueError:
        return None
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    videos = [
        s
        for s in streams
        if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")
    ]
    info: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "format_name": fmt.get("format_name"),
        "has_video": bool(videos),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "source": "ffprobe",
    }
    if not videos:
        info["duration"] = _num(fmt.get("duration")) or 0.0
        return info
    v = videos[0]
    orient: Orientation = IDENTITY
    for sd in v.get("side_data_list") or []:
        if "displaymatrix" in sd or "rotation" in sd:
            matrix = matrix_from_ffprobe(sd.get("displaymatrix", ""))
            if matrix:
                orient = from_display_matrix(matrix, "displaymatrix")
            elif sd.get("rotation") is not None:
                orient = from_rotation_cw(-float(sd["rotation"]), "displaymatrix")
            break
    tags = v.get("tags") or {}
    if orient is IDENTITY and tags.get("rotate"):
        orient = from_rotation_cw(_num(tags.get("rotate")) or 0, "rotate_tag")
    duration = _num(v.get("duration")) or _num(fmt.get("duration")) or 0.0
    all_tags = {**(fmt.get("tags") or {}), **tags}
    info.update(
        {
            "duration": float(duration),
            "format_duration": _num(fmt.get("duration")),
            "avg_fps": _ratio(v.get("avg_frame_rate")),
            "r_fps": _ratio(v.get("r_frame_rate")),
            "nb_frames": _num(v.get("nb_frames"), int),
            "codec": v.get("codec_name"),
            "profile": v.get("profile"),
            "pix_fmt": v.get("pix_fmt"),
            "bit_rate": _num(v.get("bit_rate"), int) or _num(fmt.get("bit_rate"), int),
            "width": int(v.get("width") or 0),
            "height": int(v.get("height") or 0),
            "sample_aspect_ratio": v.get("sample_aspect_ratio"),
            "start_time": _num(v.get("start_time")) or 0.0,
            "time_base": v.get("time_base"),
            "stream_index": int(v.get("index") or 0),
            "tags": {k: all_tags[k] for k in _TAG_KEYS if k in all_tags},
        }
    )
    return _finalize(info, orient)


def _probe_pyav(path: Path) -> Optional[dict[str, Any]]:
    try:
        import av  # lazy
    except ImportError:
        return None
    try:
        container = av.open(str(path))
    except Exception as exc:  # av.error.InvalidDataError и пр.
        raise VideoProbeError(
            "Не удалось прочитать файл как видео (повреждён или неподдерживаемый формат)",
            path=str(path),
            details={"pyav_error": str(exc)},
        ) from exc
    try:
        info: dict[str, Any] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "format_name": getattr(container.format, "name", None),
            "has_video": bool(container.streams.video),
            "has_audio": bool(container.streams.audio),
            "source": "pyav",
        }
        if not container.streams.video:
            info["duration"] = (container.duration or 0) / 1_000_000
            return info
        stream = container.streams.video[0]
        cc = stream.codec_context
        tb = stream.time_base or Fraction(1, 1)
        duration = float(stream.duration * tb) if stream.duration else (container.duration or 0) / 1_000_000
        orient = _pyav_first_frame_orientation(container, stream)
        if orient is IDENTITY and stream.metadata.get("rotate"):
            orient = from_rotation_cw(_num(stream.metadata.get("rotate")) or 0, "rotate_tag")
        info.update(
            {
                "duration": float(duration),
                "avg_fps": float(stream.average_rate or 0),
                "r_fps": float(getattr(stream, "base_rate", None) or getattr(stream, "guessed_rate", None) or 0),
                "nb_frames": int(stream.frames) if stream.frames else None,
                "codec": cc.name,
                "pix_fmt": getattr(cc, "pix_fmt", None),
                "bit_rate": cc.bit_rate or container.bit_rate,
                "width": int(cc.width or 0),
                "height": int(cc.height or 0),
                "start_time": float(stream.start_time * tb) if stream.start_time else 0.0,
                "time_base": f"{tb.numerator}/{tb.denominator}",
                "stream_index": int(stream.index),
                "tags": {k: v for k, v in {**container.metadata, **stream.metadata}.items() if k in _TAG_KEYS},
            }
        )
        return _finalize(info, orient)
    finally:
        container.close()


def _pyav_first_frame_orientation(container: Any, stream: Any) -> Orientation:
    """Display matrix с первого кадра (ffmpeg >= 6 кладёт её в side data кадра)."""
    try:
        for frame in container.decode(stream):
            return pyav_frame_orientation(frame)
    except Exception:
        return IDENTITY
    return IDENTITY


def pyav_frame_orientation(frame: Any) -> Orientation:
    from .orientation import matrix_from_bytes

    try:
        for sd in frame.side_data:
            if "DISPLAYMATRIX" in str(getattr(sd, "type", "")).upper():
                matrix = matrix_from_bytes(bytes(sd))
                if matrix:
                    return from_display_matrix(matrix, "displaymatrix")
    except Exception:
        pass
    rot = getattr(frame, "rotation", None)  # PyAV >= 14: против часовой
    if rot:
        return from_rotation_cw(-float(rot), "displaymatrix")
    return IDENTITY


def _probe_opencv(path: Path) -> Optional[dict[str, Any]]:
    try:
        import cv2  # lazy
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoProbeError(
                "Не удалось прочитать файл как видео (повреждён или неподдерживаемый формат)", path=str(path)
            )
        auto_off = hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO") and cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        rot = cap.get(cv2.CAP_PROP_ORIENTATION_META) if hasattr(cv2, "CAP_PROP_ORIENTATION_META") else 0
        orient = from_rotation_cw(rot, "opencv") if auto_off else IDENTITY
        info: dict[str, Any] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "format_name": None,
            "has_video": w > 0 and h > 0,
            "has_audio": None,
            "duration": (n / fps) if fps > 0 else 0.0,
            "avg_fps": fps,
            "r_fps": fps,
            "nb_frames": n or None,
            "codec": None,
            "width": w,
            "height": h,
            "start_time": 0.0,
            "time_base": None,
            "stream_index": 0,
            "tags": {},
            "source": "opencv",
        }
        return _finalize(info, orient)
    finally:
        cap.release()


def probe(path: str, *, timeout: float = 20.0, backends: tuple[str, ...] = ("ffprobe", "pyav", "opencv")) -> dict[str, Any]:
    """Метаданные видео. Бросает VideoNotFoundError / VideoProbeError, если файл не читается ничем."""
    p = _check_file(path)
    errors: list[IngestError] = []
    for name in backends:
        fn = {"ffprobe": lambda: _probe_ffprobe(p, timeout), "pyav": lambda: _probe_pyav(p), "opencv": lambda: _probe_opencv(p)}.get(name)
        if fn is None:
            continue
        try:
            info = fn()
        except IngestError as exc:
            errors.append(exc)
            continue
        if info is not None:
            if errors:
                info.setdefault("warnings", []).append(f"probe fallback: {errors[0].message}")
            return info
    if errors:
        raise errors[0]
    raise VideoProbeError(
        "Нет инструментов для чтения видео: установите PyAV (pip install av) или ffmpeg/ffprobe", path=str(path)
    )


def validate(info: dict[str, Any], min_duration: float = 1.0, max_duration: float = 900.0) -> dict[str, Any]:
    """Проверить пробу: есть видеопоток, длительность в [min, max] сек. Возвращает info."""
    path = info.get("path")
    if not info.get("has_video") or not (info.get("width") and info.get("height")):
        raise NoVideoStreamError("В файле нет видеопотока", path=path)
    duration = float(info.get("duration") or 0.0)
    if duration <= 0 and info.get("nb_frames") and info.get("fps"):
        duration = float(info["nb_frames"]) / float(info["fps"])
        info["duration"] = duration
    if duration <= 0:
        # длительность не записана в контейнере — не повод отказывать, декодер посчитает сам
        info.setdefault("warnings", []).append("длительность не указана в контейнере")
        return info
    if duration < min_duration or duration > max_duration:
        raise InvalidDurationError(
            f"Длительность видео {duration:.1f} с вне допустимого диапазона {min_duration:g}–{max_duration:g} с",
            path=path,
            details={"duration": duration, "min": min_duration, "max": max_duration},
        )
    return info


def probe_and_validate(path: str, config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    from .config import load_config

    cfg = (config or load_config())["probe"]
    info = probe(path, timeout=float(cfg["ffprobe_timeout_s"]))
    return validate(info, float(cfg["min_duration_s"]), float(cfg["max_duration_s"]))


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m ml.ingest.probe", description="ffprobe-обёртка Kassa Sentinel")
    ap.add_argument("video")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args(argv)
    try:
        info = probe(args.video) if args.no_validate else probe_and_validate(args.video)
    except IngestError as exc:
        print(json.dumps({"error": exc.to_dict()}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(info, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
