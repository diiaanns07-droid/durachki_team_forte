"""Подготовка кадров для VLM: выбор моментов, подсветка боксов, ресайз (max side 448), коллаж, motion energy."""
from __future__ import annotations

import base64
import io
import logging
import math
from typing import Any, Iterable, Optional, Sequence

log = logging.getLogger(__name__)

BBoxT = tuple[float, float, float, float]


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def clip_duration(clip: Any) -> float:
    d = float(getattr(clip, "duration", 0.0) or 0.0)
    times = list(getattr(clip, "times", []) or [])
    if d <= 0 and times:
        d = float(times[-1])
    return max(d, 0.0)


def verify_times(t_start: float, t_end: float, t_peak: float, duration: float, n: int = 5,
                 pad: float = 1.0, max_span: float = 12.0) -> list[float]:
    """n моментов из [t_start-pad, t_end+pad] (с обрезкой по видео); t_peak всегда входит."""
    n = max(1, int(n))
    end_lim = duration if duration > 0 else max(t_end + pad, t_peak)
    a = clamp(min(t_start, t_peak) - pad, 0.0, end_lim)
    b = clamp(max(t_end, t_peak) + pad, 0.0, end_lim)
    if b - a > max_span:  # длинное событие — окно вокруг пика
        a = clamp(t_peak - max_span / 2, 0.0, end_lim)
        b = clamp(a + max_span, 0.0, end_lim)
        a = clamp(b - max_span, 0.0, end_lim)
    if n == 1 or b - a < 1e-6:
        return [round(clamp(t_peak, a, b), 3)]
    ts = [a + (b - a) * i / (n - 1) for i in range(n)]
    peak = clamp(t_peak, a, b)
    j = min(range(n), key=lambda i: abs(ts[i] - peak))
    ts[j] = peak
    out: list[float] = []
    for t in sorted(ts):
        t = round(t, 3)
        if not out or t - out[-1] > 1e-3:
            out.append(t)
    return out


def get_frame(clip: Any, t: float) -> Any:
    """Точный кадр через clip.frame_at(t); иначе ближайший сэмплированный clip.frame(i)."""
    return get_frame_ts(clip, t)[0]


def get_frame_ts(clip: Any, t: float) -> tuple[Any, Optional[float]]:
    """(кадр, ФАКТИЧЕСКОЕ время кадра, с). Время в отчётах VLM (frame_times, t_best, t_transition) — это время
    реально показанного модели кадра, а не запрошенное.

    1) clip.frame_at_ts(t) (A04: точный кадр + его PTS; VFR, пропуски и края ролика учтены);
    2) clip.frame_at(t) — PTS неизвестен, берём запрошенное t;
    3) ближайший сэмплированный clip.frame(i) — его время clip.times[i].
    """
    fn = getattr(clip, "frame_at_ts", None)
    if callable(fn):
        try:
            f, ts = fn(float(t))
            if f is not None:
                return f, (float(ts) if ts is not None else float(t))
        except Exception as e:  # noqa: BLE001
            log.debug("frame_at_ts(%.2f) failed: %s", t, e)
    try:
        f = clip.frame_at(float(t))
        if f is not None:
            return f, float(t)
    except Exception as e:  # noqa: BLE001
        log.debug("frame_at(%.2f) failed: %s", t, e)
    times = list(getattr(clip, "times", []) or [])
    if not times:
        return None, None
    i = min(range(len(times)), key=lambda k: abs(times[k] - t))
    try:
        return clip.frame(i), float(times[i])
    except Exception as e:  # noqa: BLE001
        log.debug("frame(%d) failed: %s", i, e)
        return None, None


def frames_at(clip: Any, times: Iterable[float]) -> tuple[list[Any], list[float]]:
    """Кадры для моментов times с фактическим временем; один и тот же кадр (VFR, пропуск, край) — один раз."""
    frames: list[Any] = []
    ftimes: list[float] = []
    for t in times:
        f, ts = get_frame_ts(clip, t)
        if f is None or ts is None:
            continue
        if any(abs(ts - x) < 1e-4 for x in ftimes):
            continue
        frames.append(f)
        ftimes.append(round(ts, 4))
    return frames, ftimes


def to_pil_rgb(frame: Any) -> Any:
    """np.ndarray BGR uint8 (или gray) -> PIL RGB. PIL.Image проходит как есть."""
    from PIL import Image  # лениво
    import numpy as np

    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, "L").convert("RGB")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return Image.fromarray(np.ascontiguousarray(arr[:, :, ::-1]), "RGB")


def resize_max_side(img: Any, max_side: int) -> Any:
    from PIL import Image

    w, h = img.size
    s = max(w, h)
    if max_side <= 0 or s <= max_side:
        return img
    k = max_side / float(s)
    return img.resize((max(1, round(w * k)), max(1, round(h * k))), Image.BILINEAR)


def _font(size: int) -> Any:
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _label(draw: Any, xy: tuple[float, float], text: str, color: tuple[int, int, int], size: int) -> None:
    font = _font(size)
    x, y = xy
    try:
        l, t, r, b = draw.textbbox((x, y), text, font=font)
    except AttributeError:
        l, t, r, b = x, y, x + 6 * len(text), y + size
    draw.rectangle((l - 1, t - 1, r + 1, b + 1), fill=(0, 0, 0))
    draw.text((x, y), text, fill=color, font=font)


def annotate(img: Any, boxes: Optional[dict[str, BBoxT]] = None, ts: Optional[float] = None,
             index: Optional[int] = None, color: Sequence[int] = (255, 40, 40), emphasize: bool = True) -> Any:
    """Рисует нормированные боксы (x1,y1,x2,y2 в 0..1) и подпись «#k t=..s»."""
    from PIL import ImageDraw

    img = img.copy()
    d = ImageDraw.Draw(img)
    w, h = img.size
    col = tuple(int(c) for c in color)[:3]
    lw = max(2, round(min(w, h) / 120)) if emphasize else 1
    fs = max(10, round(min(w, h) / 22))
    for name, bb in (boxes or {}).items():
        try:
            x1, y1, x2, y2 = (float(v) for v in bb)
        except (TypeError, ValueError):
            continue
        x1, x2 = sorted((clamp(x1, 0, 1) * (w - 1), clamp(x2, 0, 1) * (w - 1)))
        y1, y2 = sorted((clamp(y1, 0, 1) * (h - 1), clamp(y2, 0, 1) * (h - 1)))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        d.rectangle((x1, y1, x2, y2), outline=col, width=lw)
        _label(d, (x1 + 2, max(0, y1 - fs - 2)), str(name)[:24], col, fs)
    if ts is not None or index is not None:
        tag = " ".join(p for p in (f"#{index}" if index is not None else "", f"t={ts:.1f}s" if ts is not None else "") if p)
        _label(d, (4, 4), tag, (255, 255, 0), fs)
    return img


def make_grid(images: list[Any], cols: int = 3, pad: int = 4) -> Any:
    from PIL import Image

    if not images:
        raise ValueError("no images")
    cols = max(1, min(cols, len(images)))
    rows = math.ceil(len(images) / cols)
    cw = max(i.size[0] for i in images)
    ch = max(i.size[1] for i in images)
    grid = Image.new("RGB", (cols * cw + (cols - 1) * pad, rows * ch + (rows - 1) * pad), (0, 0, 0))
    for k, im in enumerate(images):
        r, c = divmod(k, cols)
        grid.paste(im, (c * (cw + pad), r * (ch + pad)))
    return grid


def prepare_images(frames: list[Any], times: list[float], boxes: Optional[dict[str, BBoxT]] = None,
                   peak_t: Optional[float] = None, cfg: Optional[dict[str, Any]] = None) -> list[Any]:
    """BGR-кадры -> список PIL (или 1 коллаж при image_mode=grid), max side соблюдается."""
    cfg = cfg or {}
    max_side = int(cfg.get("max_side", 448))
    out = []
    # кадр-«пик» — ближайший по фактическому времени (PTS может не совпасть с запрошенным t_peak)
    k_peak = min(range(len(times)), key=lambda j: abs(times[j] - peak_t)) if peak_t is not None and times else None
    for k, (f, t) in enumerate(zip(frames, times)):
        img = resize_max_side(to_pil_rgb(f), max_side)
        is_peak = k_peak is not None and k == k_peak
        img = annotate(
            img,
            boxes if cfg.get("draw_boxes", True) else None,
            ts=t if cfg.get("draw_timestamps", True) else None,
            index=k + 1,
            color=cfg.get("box_color", (255, 40, 40)),
            emphasize=is_peak or peak_t is None,
        )
        out.append(img)
    if cfg.get("image_mode", "multi") == "grid" and len(out) > 1:
        grid = make_grid(out, int(cfg.get("grid_cols", 3)))
        return [resize_max_side(grid, max(max_side * 2, max_side))]
    return out


def pil_to_jpeg_b64(img: Any, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def motion_energy(clip: Any, downscale: int = 64, target_fps: float = 2.0,
                  times: Optional[Iterable[float]] = None) -> list[tuple[float, float]]:
    """Средняя |разность| соседних кадров (серые, уменьшенные) — дешёвый сигнал активности, 0..1."""
    import numpy as np

    ts = list(times if times is not None else (getattr(clip, "times", []) or []))
    if len(ts) < 2:
        return [(t, 0.0) for t in ts]
    dt = (ts[-1] - ts[0]) / max(1, len(ts) - 1)
    step = max(1, int(round((1.0 / target_fps) / dt))) if dt > 0 else 1
    idx = list(range(0, len(ts), step))
    out: list[tuple[float, float]] = []
    prev = None
    for i in idx:
        try:
            f = clip.frame(i)
        except Exception:  # noqa: BLE001
            f = None
        if f is None:
            continue
        arr = np.asarray(f)
        sy = max(1, arr.shape[0] // downscale)
        sx = max(1, arr.shape[1] // downscale)
        sub = arr[::sy, ::sx].astype(np.float32)
        # яркость по BT.601 для BGR (простое среднее каналов «не видит» смену цвета при равной сумме)
        g = sub[:, :, :3] @ np.array([0.114, 0.587, 0.299], np.float32) if sub.ndim == 3 else sub
        e = 0.0 if prev is None or prev.shape != g.shape else float(np.abs(g - prev).mean() / 255.0)
        out.append((float(ts[i]), e))
        prev = g
    if len(out) >= 2:
        out[0] = (out[0][0], out[1][1])
    return out
