"""Контрольные листы: готовая разметка (reviewed.jsonl) с подписанными классами — для второго, независимого прохода.

    python datasets/det/qa_sheets.py                 # data/frames/_review/qa/sheets/<video>/<кадр>.jpg + батчи

Первый проход выбирал рамки из кандидатов; этот проход проверяет итог: неверный класс, лишняя рамка, рамка
не по объекту, пропущенный объект. Кадры, где контроль нашёл проблему, уходят человеку первыми
(ls_project.py tabs → вкладка «0 · Спорные»). Человека контроль не заменяет: Submit в Label Studio — его решение.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, FRAMES_DIR  # noqa: E402
from review_sheets import REVIEW_DIR  # noqa: E402

QA_DIR = REVIEW_DIR / "qa"
COLORS = {c["name"]: c["color"] for c in CLASSES}


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def render(img, boxes: list[dict], out: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    W, H = img.size
    sheet = Image.new("RGB", (2 * W + 8, H), (255, 255, 255))
    sheet.paste(img, (0, 0))
    right = img.copy()
    d = ImageDraw.Draw(right)
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:
        font = ImageFont.load_default()
    for i, b in enumerate(boxes, 1):
        col = _rgb(COLORS.get(b["label"], "#ffffff"))
        x1, y1, x2, y2 = b["xyxy"]
        d.rectangle([x1, y1, x2, y2], outline=col, width=2)
        tag = f"{i} {b['label']}"
        tw = d.textlength(tag, font=font)
        tx, ty = min(max(0, x1), W - tw - 6), max(0, y1 - 16) if y1 >= 16 else min(H - 16, y1 + 1)
        d.rectangle([tx, ty, tx + tw + 6, ty + 15], fill=col)
        d.text((tx + 3, ty), tag, fill=(0, 0, 0) if sum(col) > 380 else (255, 255, 255), font=font)
    sheet.paste(right, (W + 8, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=90)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reviewed", type=Path, default=REVIEW_DIR / "reviewed.jsonl")
    ap.add_argument("--frames", type=Path, default=FRAMES_DIR)
    ap.add_argument("--out-dir", type=Path, default=QA_DIR)
    ap.add_argument("--batch-size", type=int, default=50)
    args = ap.parse_args(argv)

    from PIL import Image

    rows = [json.loads(line) for line in args.reviewed.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if not r.get("skip")]
    (args.out_dir / "batches").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results").mkdir(parents=True, exist_ok=True)
    for r in rows:
        render(Image.open(args.frames / r["file"]).convert("RGB"), r["boxes"],
               args.out_dir / "sheets" / Path(r["file"]).with_suffix(".jpg"))
    files = [r["file"] for r in rows]
    n = 0
    for k in range(0, len(files), args.batch_size):
        n += 1
        (args.out_dir / "batches" / f"qa_{n:02d}.txt").write_text("\n".join(files[k:k + args.batch_size]) + "\n",
                                                                  encoding="utf-8")
    print(f"листов: {len(rows)}, батчей: {n} → {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
