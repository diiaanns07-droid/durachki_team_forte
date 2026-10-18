"""Контакт-листы 4×N с подписями времени — для ПРОСМОТРА ЧЕЛОВЕКОМ (или локальной VLM), локально.

Результат класть в cache/ (в git не попадает). Шрифт — Windows arialbd.ttf (иначе встроенный).
Кадр для t — ближайший по PTS (последовательное декодирование: видео VFR); в памяти держится только
ближайший кадр на каждый запрошенный момент, а не весь ролик.

usage: python -I tools/dataprep/contact_sheets.py <video> <out.jpg> <t1,t2,...|overview[:N]> [scale]
       [crop=x1,y1,x2,y2 (норм. 0..1)]
"""
from __future__ import annotations

import math
import sys

import cv2
from PIL import Image, ImageDraw, ImageFont


def parse_times(spec: str, dur: float) -> list[float]:
    if spec.startswith("overview"):
        n = int(spec.split(":")[1]) if ":" in spec else 12
        return [dur * (i + 0.5) / n for i in range(n)]
    return [float(v) for v in spec.split(",") if v.strip()]


def video_duration(path: str) -> float:
    cap = cv2.VideoCapture(path)
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return n / fps


def nearest_frames(path: str, times: list[float]) -> dict[float, tuple[float, object]]:
    best = {t: (math.inf, 0.0, None) for t in times}
    cap = cv2.VideoCapture(path)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        for t in times:
            d = abs(pts - t)
            if d < best[t][0]:
                best[t] = (d, pts, fr)
    cap.release()
    return {t: (b[1], b[2]) for t, b in best.items() if b[2] is not None}


def load_font(size: int = 22):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", size)
    except OSError:
        return ImageFont.load_default()


def make_sheet(path: str, out: str, spec: str, scale: float = 0.75, crop=None, cols: int = 4) -> tuple[int, int]:
    dur = video_duration(path)
    times = parse_times(spec, dur)
    valid = [t for t in times if 0 <= t <= dur + 0.05]
    frames = nearest_frames(path, valid)
    font = load_font()
    tiles = []
    for t in times:
        if t not in frames:
            tiles.append(None)
            continue
        pts, fr = frames[t]
        img = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        if crop:
            w, h = img.size
            img = img.crop((int(crop[0] * w), int(crop[1] * h), int(crop[2] * w), int(crop[3] * h)))
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 118, 30], fill=(0, 0, 0))
        d.text((5, 3), f"t={pts:.1f}s", fill=(255, 255, 0), font=font)
        tiles.append(img)
    real = [t for t in tiles if t]
    if not real:
        raise SystemExit("нет ни одного кадра в запрошенном диапазоне")
    tw, th = max(t.width for t in real), max(t.height for t in real)
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw + (cols - 1) * 4, rows * th + (rows - 1) * 4), (40, 40, 40))
    for i, t in enumerate(tiles):
        if t:
            sheet.paste(t, ((i % cols) * (tw + 4), (i // cols) * (th + 4)))
    sheet.save(out, quality=88)
    return sheet.size


def main(argv: list[str]):
    video, out, spec = argv[1], argv[2], argv[3]
    scale = float(argv[4]) if len(argv) > 4 and not argv[4].startswith("crop=") else 0.75
    crop = next(([float(v) for v in x[5:].split(",")] for x in argv[4:] if x.startswith("crop=")), None)
    size = make_sheet(video, out, spec, scale, crop)
    print(out, size)


if __name__ == "__main__":
    main(sys.argv)
