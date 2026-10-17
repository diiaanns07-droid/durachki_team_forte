#!/usr/bin/env python3
"""Синтетическое видео для smoke/healthcheck/SLA (НЕ данные датасета, ничего реального).

Портрет 464×848, 30 fps, как у потолочной камеры. Движущиеся прямоугольники — «люди»/«предметы»,
чтобы декодер, сэмплер и детектор получили не пустые кадры. --target-mb (через ffmpeg с шумом)
даёт файл заданного размера — для проверки лимитов загрузки туннеля.

  python deploy/synth_video.py --out cache/synth.mov --seconds 20
  python deploy/synth_video.py --out cache/synth_30mb.mov --seconds 60 --target-mb 30
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _ffmpeg(out: Path, seconds: float, fps: float, width: int, height: int, target_mb: float | None,
            seed: int = 0) -> None:
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("нет ни cv2, ни ffmpeg")
    # seed → другой шум → другой sha256 (A02 дедуплицирует одинаковые файлы, A03 кэширует стадии по sha)
    vf = f"noise=alls=60:allf=t+u:all_seed={seed}" if target_mb else (f"noise=alls=8:allf=t:all_seed={seed}" if seed
                                                                       else "null")
    cmd = [ff, "-y", "-loglevel", "error", "-f", "lavfi",
           "-i", f"testsrc2=size={width}x{height}:rate={fps:g}", "-t", f"{seconds:g}", "-vf", vf,
           "-pix_fmt", "yuv420p", "-c:v", "mpeg4"]
    if target_mb:
        kbps = int(target_mb * 8 * 1024 / max(seconds, 0.1))
        cmd += ["-b:v", f"{kbps}k", "-minrate", f"{kbps}k", "-maxrate", f"{kbps}k", "-bufsize", f"{kbps}k"]
    else:
        cmd += ["-q:v", "6"]
    subprocess.run(cmd + [str(out)], check=True)


def _opencv(out: Path, seconds: float, fps: float, width: int, height: int, seed: int = 0) -> None:
    import cv2  # noqa: WPS433 — ленивый импорт
    import numpy as np

    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter не открылся")
    n = int(round(seconds * fps))
    for i in range(n):
        t = i / fps
        img = np.full((height, width, 3), 90, np.uint8)
        cv2.rectangle(img, (40, 380), (width - 40, 560), (120, 110, 100), -1)          # стол
        cv2.rectangle(img, (width - 150, 600), (width - 40, 760), (140, 140, 150), -1)  # «сейф»
        x = int(60 + (width - 200) * (0.5 + 0.5 * np.sin(t / 2 + seed)))
        cv2.rectangle(img, (x, 150), (x + 90, 360), (60, 80, 160), -1)                   # «человек»
        cv2.putText(img, f"SYNTH#{seed} {t:5.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(img)
    writer.release()


def make_video(out: Path, seconds: float = 10.0, fps: float = 30.0, width: int = 464, height: int = 848,
               target_mb: float | None = None, seed: int = 0) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if target_mb:
        _ffmpeg(out, seconds, fps, width, height, target_mb, seed)
    else:
        try:
            _opencv(out, seconds, fps, width, height, seed)
        except (ImportError, RuntimeError):
            _ffmpeg(out, seconds, fps, width, height, None, seed)
    if not out.is_file() or out.stat().st_size == 0:
        raise RuntimeError(f"видео не создано: {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=464)
    ap.add_argument("--height", type=int, default=848)
    ap.add_argument("--target-mb", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0, help="разный seed → разный файл (без дедупа/кэша backend)")
    a = ap.parse_args(argv)
    p = make_video(a.out, a.seconds, a.fps, a.width, a.height, a.target_mb, a.seed)
    print(f"{p} {p.stat().st_size / 2**20:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
