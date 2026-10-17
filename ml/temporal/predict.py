"""Ручная проверка score() на одном видео (модуль назван predict, чтобы не затенять функцию ml.temporal.score).

    python -m ml.temporal.predict --video data/raw/X-1.mov [--json out.json]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ml.temporal: score(clip) для одного видео")
    ap.add_argument("--video", required=True)
    ap.add_argument("--sample-fps", type=float, default=5.0)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--json", default=None, help="сохранить полный результат (кривая + события)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from . import TemporalUnavailable, score, status
    from ._video import load_clip

    clip = load_clip(a.video, sample_fps=a.sample_fps)
    try:
        risk, events = score(clip, budget_s=a.budget)
    except TemporalUnavailable:  # не проанализировано: печатаем честный статус, код возврата 2
        risk, events = [], []
    st = status()
    out = {"video": Path(a.video).name, "status": st, "risk": risk,
           "events": [dataclasses.asdict(e) for e in events]}
    print(json.dumps({"status": st, "max_risk": max((r for _, r in risk), default=None),
                      "events": [{"t_peak": e.t_peak, "t_start": e.t_start, "t_end": e.t_end,
                                  "confidence": e.confidence, "type_id": e.type_id,
                                  "type_source": e.extra.get("type_source")} for e in events]},
                     ensure_ascii=False, indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if st.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
