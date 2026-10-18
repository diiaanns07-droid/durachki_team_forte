"""Листы для проверки пред-разметки: кадр + пронумерованные рамки-кандидаты.

    python datasets/det/review_sheets.py            # для отобранных кадров (data/frames/_ls/selected.txt)

Кандидаты: сначала боксы OWLv2 из predictions.jsonl (те же, что в Label Studio), затем дополнительные
места, которые OWLv2 считает объектом хоть какого-то класса (скор ≥ --min-score, без учёта класса), чтобы
проверяющий мог выбрать рамку для пропущенного объекта, а не рисовать её на глаз. Классы на листе НЕ
подписаны — проверяющий назначает их сам. Рамки остаются пиксельными рамками OWLv2.

Выход (не в git): data/frames/_review/candidates.jsonl и data/frames/_review/sheets/<video>/<кадр>.jpg —
слева чистый кадр, справа рамки с номерами и сетка координат через 100 px (для дорисовки).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, FRAMES_DIR, LS_DIR, PRED_JSONL  # noqa: E402
from prelabel_owlv2 import Owl, iou, nms  # noqa: E402

REVIEW_DIR = FRAMES_DIR / "_review"
PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
           (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40),
           (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128)]


def extra_candidates(scores, boxes, width: int, height: int, min_score: float) -> list[dict]:
    """Без учёта класса: лучший класс патча, скор ≥ min_score, NMS 0.5. Боксы в пикселях кадра."""
    side = float(max(width, height))
    best = scores.max(axis=1)
    cand = []
    for p in (best >= min_score).nonzero()[0]:
        cx, cy, w, h = (float(v) * side for v in boxes[p])
        x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
        x2, y2 = min(float(width), cx + w / 2), min(float(height), cy + h / 2)
        if x2 - x1 < 6 or y2 - y1 < 6 or (x2 - x1) * (y2 - y1) > 0.5 * width * height:
            continue
        cand.append({"label": CLASSES[int(scores[p].argmax())]["name"], "score": round(float(best[p]), 4),
                     "xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]})
    return nms(cand, 0.5)


def merge(owl_boxes: list[dict], extra: list[dict], total: int) -> list[dict]:
    out = [{**b, "n": i + 1, "src": "owl"} for i, b in enumerate(owl_boxes)]
    for b in extra:
        if len(out) >= total:
            break
        if all(iou(b["xyxy"], o["xyxy"]) < 0.5 for o in out):
            out.append({**b, "n": len(out) + 1, "src": "extra"})
    return out


def render(img, cands: list[dict], out: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    W, H = img.size
    sheet = Image.new("RGB", (2 * W + 8, H), (255, 255, 255))
    sheet.paste(img, (0, 0))
    right = img.copy()
    d = ImageDraw.Draw(right)
    try:
        font = ImageFont.load_default(size=15)
        small = ImageFont.load_default(size=11)
    except TypeError:                                  # Pillow < 10.1
        font = small = ImageFont.load_default()
    for x in range(100, W, 100):
        d.line([(x, 0), (x, H)], fill=(255, 255, 255), width=1)
        d.text((x + 2, 2), str(x), fill=(255, 255, 0), font=small)
    for y in range(100, H, 100):
        d.line([(0, y), (W, y)], fill=(255, 255, 255), width=1)
        d.text((2, y + 2), str(y), fill=(255, 255, 0), font=small)
    for c in cands:
        col = PALETTE[(c["n"] - 1) % len(PALETTE)]
        x1, y1, x2, y2 = c["xyxy"]
        d.rectangle([x1, y1, x2, y2], outline=col, width=2 if c["src"] == "owl" else 1)
        tag = str(c["n"])
        tx, ty = min(max(0, x1 + 1), W - 22), min(max(0, y1 + 1), H - 18)
        tw = d.textlength(tag, font=font)
        d.rectangle([tx, ty, tx + tw + 4, ty + 17], fill=col)
        d.text((tx + 2, ty), tag, fill=(0, 0, 0) if sum(col) > 380 else (255, 255, 255), font=font)
    sheet.paste(right, (W + 8, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=90)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selected", type=Path, default=LS_DIR / "selected.txt")
    ap.add_argument("--pred", type=Path, default=PRED_JSONL)
    ap.add_argument("--frames", type=Path, default=FRAMES_DIR)
    ap.add_argument("--out-dir", type=Path, default=REVIEW_DIR)
    ap.add_argument("--min-score", type=float, default=0.06)
    ap.add_argument("--total", type=int, default=16, help="максимум рамок на кадр (OWLv2 + доп. кандидаты)")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args(argv)

    from PIL import Image

    files = args.selected.read_text(encoding="utf-8").split()
    preds = {}
    for line in args.pred.read_text(encoding="utf-8").splitlines():
        if line.strip():
            j = json.loads(line)
            preds[j["file"]] = j["boxes"]
    owl = Owl()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_extra = 0
    with open(args.out_dir / "candidates.jsonl", "w", encoding="utf-8") as f:
        for i in range(0, len(files), args.batch):
            chunk = files[i:i + args.batch]
            imgs = [Image.open(args.frames / fn).convert("RGB") for fn in chunk]
            for fn, im, (scores, boxes) in zip(chunk, imgs, owl.scores(imgs)):
                cands = merge(preds.get(fn, []), extra_candidates(scores, boxes, im.width, im.height, args.min_score),
                              args.total)
                n_extra += sum(c["src"] == "extra" for c in cands)
                render(im, cands, args.out_dir / "sheets" / Path(fn).with_suffix(".jpg"))
                f.write(json.dumps({"file": fn, "width": im.width, "height": im.height, "cands": cands},
                                   ensure_ascii=False) + "\n")
            if (i // args.batch) % 20 == 0:
                print(f"  {i + len(chunk)}/{len(files)}", flush=True)
    print(f"готово: {len(files)} листов, доп. кандидатов {n_extra} → {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
