"""Контакт-лист: сетка кадров вокруг момента t с подписями фактических таймкодов.

    python -m ml.ingest.contact_sheet video.mov --t 22 --window 4 --out sheet.jpg
    python -m ml.ingest.contact_sheet video.mov --every 5 --out overview.jpg      # обзор всего ролика

ПРИВАТНОСТЬ: лист пишется ТОЛЬКО в локальный файл — для человека-разметчика или ЛОКАЛЬНОЙ VLM.
CLI печатает лишь путь и таймкоды; картинки/base64/описания сцен в stdout не выводятся.
Не загружайте листы с реальных роликов в облачные ассистенты (включая расширения IDE).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def sheet_times(t: Optional[float], window: float, step: float, duration: float, every: Optional[float] = None) -> list[float]:
    if t is None:
        every = every or max(1.0, duration / 24.0)
        return [float(x) for x in np.arange(0.0, max(duration, 1e-3), every)]
    lo, hi = max(0.0, t - window), min(duration, t + window)
    times = list(np.arange(lo, hi + 1e-6, step))
    if not any(abs(x - t) < 1e-6 for x in times) and lo <= t <= hi:
        times.append(t)
    return sorted(float(x) for x in times)


def build_sheet(
    images: Sequence[np.ndarray],
    labels: Sequence[str],
    *,
    cols: int = 4,
    tile_width: int = 240,
    highlight: Optional[int] = None,
    title: str = "",
) -> np.ndarray:
    import cv2

    if not images:
        raise ValueError("нет кадров для листа")
    h0, w0 = images[0].shape[:2]
    tw = int(tile_width)
    th = max(2, int(round(h0 * tw / w0)))
    cols = max(1, min(cols, len(images)))
    rows = math.ceil(len(images) / cols)
    pad, cap, head = 4, 22, 26 if title else 0
    sheet = np.full((head + rows * (th + cap + pad) + pad, cols * (tw + pad) + pad, 3), 32, np.uint8)
    if title:
        cv2.putText(sheet, title[:90], (pad, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    for k, (img, label) in enumerate(zip(images, labels)):
        r, c = divmod(k, cols)
        x, y = pad + c * (tw + pad), head + pad + r * (th + cap + pad)
        tile = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        sheet[y : y + th, x : x + tw] = tile
        color = (0, 0, 255) if k == highlight else (220, 220, 220)
        if k == highlight:
            cv2.rectangle(sheet, (x - 2, y - 2), (x + tw + 1, y + th + 1), color, 2)
        cv2.putText(sheet, label, (x + 2, y + th + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return sheet


def make_contact_sheet(
    video: str,
    out: str,
    *,
    t: Optional[float] = None,
    window: float = 4.0,
    step: float = 0.5,
    every: Optional[float] = None,
    cols: int = 4,
    tile_width: int = 240,
    quality: int = 90,
) -> dict:
    """Собрать лист и записать JPEG в ``out``. Возвращает только метаданные (путь, таймкоды)."""
    import cv2

    from .names import normalize_video_id
    from .video import load_video

    clip = load_video(video, sample_fps=1.0)
    try:
        times = sheet_times(t, window, step, clip.duration, every)
        refs = clip.frames_at(times)
        actual = [r.timestamp_sec for r in refs]
        hi = None
        if t is not None:
            hi = int(np.argmin([abs(a - t) for a in actual]))
        labels = [f"t={a:.2f}s" for a in actual]
        title = f"{normalize_video_id(Path(video).name)}  {clip.width}x{clip.height}  {clip.duration:.1f}s"
        if t is not None:
            title += f"  t*={t:g}s +/-{window:g}s"
        sheet = build_sheet([r.image for r in refs], labels, cols=cols, tile_width=tile_width, highlight=hi, title=title)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]):
            raise OSError(f"не удалось записать {out}")
        return {"output": str(out), "times": actual, "highlight_index": hi, "size": [sheet.shape[1], sheet.shape[0]]}
    finally:
        clip.close()


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ml.ingest.contact_sheet",
                                 description="Контакт-лист кадров (только локальный файл; не для облачных сервисов)")
    ap.add_argument("video")
    ap.add_argument("--t", type=float, default=None, help="центр, сек (без --t — обзор всего ролика)")
    ap.add_argument("--window", type=float, default=4.0, help="± секунд вокруг t")
    ap.add_argument("--step", type=float, default=0.5, help="шаг между кадрами, сек")
    ap.add_argument("--every", type=float, default=None, help="шаг обзора без --t, сек")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--tile-width", type=int, default=240)
    ap.add_argument("--out", required=True, help="путь к .jpg")
    args = ap.parse_args(argv)
    meta = make_contact_sheet(args.video, args.out, t=args.t, window=args.window, step=args.step, every=args.every,
                              cols=args.cols, tile_width=args.tile_width)
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
