"""Оценка детектора: COCO-style AP (101 точка), AP50 и AP@[.50:.95] по классам. Своя реализация (numpy).

    python -m ml.detect.eval_det --gt data/autolabel/verified.json --pred preds.json
    python -m ml.detect.eval_det --gt data/autolabel/verified.json --run --frames data/frames

ВАЖНО: метрика качества имеет смысл только на ПРОВЕРЕННОЙ человеком разметке (info.verified=true).
На псевдоразметке отчёт помечается quality_claim=false — это лишь согласие модели с предложениями,
а не качество детектора, и в отчётах/питче как качество не приводится.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .autolabel import category_label, default_images_root, label_status
from .boxes import iou_matrix

IOU_THRS = np.round(np.arange(0.5, 0.951, 0.05), 2)
UNVERIFIED_WARNING = ("GT не подтверждена как человеческая разметка (нет mark-verified и не указан --annotator): "
                      "результат = согласие с псевдоразметкой, НЕ метрика качества.")


def _xywh_to_xyxy(b: Sequence[float]) -> list[float]:
    x, y, w, h = (float(v) for v in b)
    return [x, y, x + w, y + h]


def average_precision(tp: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
    """COCO: 101-точечная интерполяция precision по recall."""
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = tp[order].astype(np.float64)
    fp = 1.0 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    for i in range(len(prec) - 2, -1, -1):  # огибающая
        prec[i] = max(prec[i], prec[i + 1])
    rs = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, rs, side="left")
    q = np.array([prec[i] if i < len(prec) else 0.0 for i in idx])
    return float(q.mean())


def evaluate(gt: dict[str, Any], preds: Sequence[dict[str, Any]], iou_thrs: Sequence[float] = IOU_THRS,
             annotator: Optional[str] = None) -> dict[str, Any]:
    cats = {c["id"]: c["name"] for c in gt.get("categories", [])}
    gt_by = defaultdict(list)  # (img, cat) -> boxes
    for a in gt.get("annotations", []):
        if not a.get("iscrowd", 0):
            gt_by[(a["image_id"], a["category_id"])].append(_xywh_to_xyxy(a["bbox"]))
    pr_by = defaultdict(list)
    for p in preds:
        pr_by[(p["image_id"], p["category_id"])].append((float(p["score"]), _xywh_to_xyxy(p["bbox"])))
    per_class: dict[str, Any] = {}
    for cid, name in cats.items():
        keys = {k for k in list(gt_by) + list(pr_by) if k[1] == cid}
        n_gt = sum(len(gt_by[k]) for k in keys)
        n_pred = sum(len(pr_by[k]) for k in keys)
        aps = []
        for thr in iou_thrs:
            tps, scs = [], []
            for k in keys:
                g = np.asarray(gt_by.get(k, []), np.float32).reshape(-1, 4)
                p = sorted(pr_by.get(k, []), key=lambda x: -x[0])
                if not p:
                    continue
                pb = np.asarray([b for _, b in p], np.float32)
                ious = iou_matrix(pb, g)
                used = np.zeros(len(g), bool)
                for i, (s, _) in enumerate(p):
                    hit = False
                    if len(g):
                        cand = np.where(~used & (ious[i] >= thr))[0]
                        if cand.size:
                            j = cand[np.argmax(ious[i][cand])]
                            used[j] = True
                            hit = True
                    tps.append(1.0 if hit else 0.0)
                    scs.append(s)
            aps.append(average_precision(np.asarray(tps), np.asarray(scs), n_gt))
        per_class[name] = {"n_gt": n_gt, "n_pred": n_pred, "AP50": aps[0] if aps else float("nan"),
                           "AP": float(np.nanmean(aps)) if n_gt else float("nan")}
    valid = [v for v in per_class.values() if v["n_gt"] > 0]
    status = label_status(gt, annotator)
    verified = status in ("verified", "manual")
    rep = {
        "quality_claim": verified,
        "gt_status": status,
        "gt_kind": (gt.get("info") or {}).get("kind", "unknown"),
        "n_images": len(gt.get("images", [])),
        "mAP50": float(np.mean([v["AP50"] for v in valid])) if valid else float("nan"),
        "mAP": float(np.mean([v["AP"] for v in valid])) if valid else float("nan"),
        "per_class": per_class,
    }
    if not verified:
        rep["warning"] = UNVERIFIED_WARNING
    return rep


def run_predictions(gt: dict[str, Any], frames_root: str | Path) -> list[dict[str, Any]]:
    """Прогон текущей конфигурации детектора (как в рантайме: closed-set + опц. open-vocab) по кадрам GT."""
    import cv2

    from .backends import get_backends
    from .boxes import RawDets, classwise_nms
    from .config import load_config

    cfg = load_config()
    be = get_backends(cfg)
    label_to_id: dict[str, int] = {}
    for c in gt["categories"]:      # метка контракта → категория GT (det-labels: safe ← safe_closed)
        lab = category_label(c)
        if lab is not None:
            label_to_id.setdefault(lab, c["id"])
    out = []
    for im in gt["images"]:
        img = cv2.imread(str(Path(frames_root) / im["file_name"]))
        if img is None:
            continue
        h, w = img.shape[:2]
        parts = [b.predict([img])[0] for b in be.closed]
        if be.open is not None:
            parts.append(be.open.predict([img])[0])
        d = classwise_nms(RawDets.concat(parts), cfg["merge"]["iou_thr"])
        for i, lab in enumerate(d.labels):
            if lab not in label_to_id:
                continue
            x1, y1, x2, y2 = (float(v) for v in d.boxes[i])
            out.append({"image_id": im["id"], "category_id": label_to_id[lab], "score": float(d.scores[i]),
                        "bbox": [x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h]})
    return out


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m ml.detect.eval_det")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", default="")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--frames", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--annotator", default=None, help="автор ручной разметки (иначе GT считается непроверенной)")
    a = ap.parse_args(argv)
    gt = json.loads(Path(a.gt).read_text(encoding="utf-8"))
    if a.run:
        preds = run_predictions(gt, a.frames or default_images_root(a.gt, gt))
    else:
        preds = json.loads(Path(a.pred).read_text(encoding="utf-8"))
    rep = evaluate(gt, preds, annotator=a.annotator)
    txt = json.dumps(rep, ensure_ascii=False, indent=2)
    print(txt)
    if not rep["quality_claim"]:
        print("\n!!! " + UNVERIFIED_WARNING)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
