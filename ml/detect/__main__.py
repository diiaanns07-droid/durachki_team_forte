"""CLI:  python -m ml.detect VIDEO [--every-n 1] [--fps 5] [--out dets.json] [--extras tta,tiles,open_vocab]

Читает видео через ml.ingest.load_video (если модуль A04 доступен), иначе — простой cv2-ридер (synth.ArrayClip).
Печатает сводку и тайминги (last_stats) — для замеров скорости на целевом железе.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import asdict
from typing import Any, Optional


def load_clip(path: str, fps: float) -> Any:
    try:
        from ml.ingest import load_video  # type: ignore[import-not-found]

        return load_video(path, sample_fps=fps)
    except ImportError:
        from .synth import ArrayClip

        return ArrayClip.from_file(path, sample_fps=fps)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ml.detect")
    ap.add_argument("video")
    ap.add_argument("--every-n", type=int, default=1)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--extras", default="", help="tta,tiles,open_vocab,finetuned (опционально)")
    a = ap.parse_args(argv)
    if a.extras:
        os.environ["SENTINEL_DETECT_EXTRAS"] = a.extras
    from . import detect, last_stats

    clip = load_clip(a.video, a.fps)
    dets = detect(clip, every_n=a.every_n)
    st = last_stats()
    print(json.dumps({"n_detections": len(dets), "labels": Counter(d.label for d in dets),
                      "stats": st}, ensure_ascii=False, indent=2, default=str))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"detections": [asdict(d) for d in dets], "stats": st}, f, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
