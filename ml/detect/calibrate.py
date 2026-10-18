"""Калибровка порогов по классам — ТОЛЬКО на человеческой bbox-разметке (Дамир: datasets/det/annotations.json).

    python -m ml.detect.calibrate --gt datasets/det/annotations.json --annotator "<имя>" \
        --target-precision 0.9 --out calib/detect_thresholds.yaml --report calib/report.json [--with-open-vocab]
    SENTINEL_DETECT_CONFIG=calib/detect_thresholds.yaml python -m ml.detect.probe ...   # применить

Процедура:
1. Разметка проверяется (verified или manual с --annotator; псевдоразметка — отказ).
2. Кадры делятся по ВИДЕО на calib / holdout (holdout_frac). Порог подбирается на calib, проверяется на holdout.
3. Для каждого класса с n_gt ≥ min_support берётся минимальный порог, при котором precision ≥ target на calib
   (максимум полноты при заданной точности). Если цель недостижима — порог из config.yaml остаётся, класс
   помечается unreachable. Классы с малым n_gt не трогаются (insufficient_support).
4. Выход — YAML-override (closed_set.score_thr и, с --with-open-vocab, inventory.rare_open_vocab.thr) + отчёт
   с P/R на calib и holdout. Это калибровка порогов, а не mAP: mAP — eval_det.py.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .autolabel import DEFAULT_DATASET, category_label, default_images_root, group_key
from .boxes import RawDets, classwise_nms, iou_matrix
from .config import label_threshold, load_config
from .finetune import load_verified, split_by_video

# группы запросов inventory (OWLv2) → метка GT
GROUP_TO_LABEL = {"cash": "cash", "cash_bag": "cash_bag", "generic_bag": "bag", "phone": "cell_phone",
                  "cup": "cup", "food": "food", "safe_open": "safe_open", "safe_closed": "safe"}


def gt_boxes(coco: dict[str, Any]) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, str]]:
    """{image_id: {label: (N,4) норм. xyxy}}; категории → LABELS через contract_label (det-labels) или имя."""
    cat = {}
    unmapped: dict[str, str] = {}
    for c in coco.get("categories", []):
        lab = category_label(c)
        if lab is None:
            unmapped[str(c["id"])] = c["name"]
        else:
            cat[c["id"]] = lab
    size = {im["id"]: (float(im.get("width") or 0), float(im.get("height") or 0)) for im in coco["images"]}
    out: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for a in coco.get("annotations", []):
        lab = cat.get(a["category_id"])
        w, h = size.get(a["image_id"], (0, 0))
        if lab is None or a.get("iscrowd", 0) or w <= 0 or h <= 0:
            continue
        x, y, bw, bh = (float(v) for v in a["bbox"])
        out[a["image_id"]][lab].append([x / w, y / h, (x + bw) / w, (y + bh) / h])
    return ({i: {k: np.asarray(v, np.float32) for k, v in d.items()} for i, d in out.items()}, unmapped)


def match(dets: RawDets, gts: dict[str, np.ndarray], iou_thr: float = 0.5) -> list[tuple[str, float, bool]]:
    """Жадное сопоставление по убыванию score внутри класса: [(label, score, is_tp)]."""
    res: list[tuple[str, float, bool]] = []
    for lab in sorted(set(dets.labels)):
        idx = [i for i, x in enumerate(dets.labels) if x == lab]
        idx.sort(key=lambda i: -float(dets.scores[i]))
        g = gts.get(lab, np.zeros((0, 4), np.float32))
        used = np.zeros(len(g), bool)
        ious = iou_matrix(dets.boxes[idx], g) if len(g) else np.zeros((len(idx), 0))
        for r, i in enumerate(idx):
            hit = False
            if len(g):
                cand = np.where(~used & (ious[r] >= iou_thr))[0]
                if cand.size:
                    used[cand[np.argmax(ious[r][cand])]] = True
                    hit = True
            res.append((lab, float(dets.scores[i]), hit))
    return res


def pr_at(scores: np.ndarray, tps: np.ndarray, n_gt: int, thr: float) -> tuple[float, float, int]:
    m = scores >= thr
    k = int(m.sum())
    tp = int(tps[m].sum())
    return (tp / k if k else float("nan"), tp / n_gt if n_gt else float("nan"), k)


def choose_threshold(scores: Sequence[float], tps: Sequence[bool], n_gt: int, target_precision: float) -> Optional[float]:
    """Минимальный порог (макс. полнота), при котором precision ≥ target; None — недостижимо."""
    s = np.asarray(scores, np.float64)
    t = np.asarray(tps, bool)
    if s.size == 0 or n_gt == 0:
        return None
    for thr in sorted(set(s.tolist())):          # кандидаты — сами score (без округления: включительно)
        p, _r, k = pr_at(s, t, n_gt, thr)
        if k and p >= target_precision:
            return float(np.floor(thr * 1e4) / 1e4)   # округление ВНИЗ сохраняет этот score над порогом
    return None


def collect(images: Sequence[dict[str, Any]], root: Path, gt: dict[int, dict[str, np.ndarray]],
            producer: Callable[[np.ndarray], RawDets], iou_thr: float) -> tuple[dict[str, list], dict[str, int]]:
    import cv2

    per: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    n_gt: dict[str, int] = defaultdict(int)
    for im in images:
        g = gt.get(im["id"], {})
        for lab, b in g.items():
            n_gt[lab] += len(b)
        img = cv2.imread(str(root / im["file_name"]))
        if img is None:
            continue
        for lab, sc, tp in match(producer(img), g, iou_thr):
            per[lab].append((sc, tp))
    return per, n_gt


def calibrate(coco: dict[str, Any], root: Path, producer: Callable[[np.ndarray], RawDets], current: dict[str, float],
              target_precision: float = 0.9, min_support: int = 10, holdout_frac: float = 0.3, iou_thr: float = 0.5,
              seed: int = 0, labels: Optional[Sequence[str]] = None) -> dict[str, Any]:
    gt, unmapped = gt_boxes(coco)
    cal_ids, hold_ids = split_by_video(coco, holdout_frac, seed)
    by_id = {im["id"]: im for im in coco["images"]}
    cal, n_cal = collect([by_id[i] for i in cal_ids], root, gt, producer, iou_thr)
    hold, n_hold = collect([by_id[i] for i in hold_ids], root, gt, producer, iou_thr)
    classes: dict[str, Any] = {}
    chosen: dict[str, float] = {}
    for lab in sorted(set(n_cal) | set(cal)):
        if labels is not None and lab not in labels:
            continue
        sc = np.asarray([x[0] for x in cal.get(lab, [])])
        tp = np.asarray([x[1] for x in cal.get(lab, [])], bool)
        cur = float(current.get(lab, current.get("default", 0.4)))
        row: dict[str, Any] = {"n_gt_calib": n_cal.get(lab, 0), "n_det_calib": int(sc.size), "current_thr": cur}
        if n_cal.get(lab, 0) < min_support:
            row["status"] = "insufficient_support"
        else:
            thr = choose_threshold(sc, tp, n_cal[lab], target_precision)
            if thr is None:
                row["status"] = "unreachable"
            else:
                row["status"] = "calibrated"
                row["thr"] = thr
                chosen[lab] = thr
        use = row.get("thr", cur)
        p, r, _ = pr_at(sc, tp, n_cal.get(lab, 0), use) if sc.size else (float("nan"), float("nan"), 0)
        row["calib_P"], row["calib_R"] = _r4(p), _r4(r)
        hs = np.asarray([x[0] for x in hold.get(lab, [])])
        ht = np.asarray([x[1] for x in hold.get(lab, [])], bool)
        hp, hr, _ = pr_at(hs, ht, n_hold.get(lab, 0), use) if hs.size else (float("nan"), float("nan"), 0)
        row.update(n_gt_holdout=n_hold.get(lab, 0), holdout_P=_r4(hp), holdout_R=_r4(hr))
        classes[lab] = row
    return {"classes": classes, "thresholds": chosen, "unmapped_categories": unmapped,
            "split": {"calib_images": len(cal_ids), "holdout_images": len(hold_ids),
                      "holdout_videos": sorted({group_key(by_id[i]) for i in hold_ids})}}


def _r4(x: float) -> Optional[float]:
    return None if x != x else round(float(x), 4)   # NaN → None (JSON)


def closed_producer(cfg: dict[str, Any]) -> Callable[[np.ndarray], RawDets]:
    from .backends import get_backends

    be = get_backends(cfg)

    def run(img: np.ndarray) -> RawDets:
        return classwise_nms(RawDets.concat([b.predict([img])[0] for b in be.closed]), cfg["merge"]["iou_thr"])
    return run


def owl_producer(cfg: dict[str, Any], min_score: float = 0.05) -> Callable[[np.ndarray], RawDets]:
    from .backends import get_inventory_scorer
    from .inventory import _group_scores

    warnings: list[str] = []
    sc = get_inventory_scorer(cfg, warnings)
    if sc is None:
        raise RuntimeError("; ".join(warnings) or "open-vocab scorer unavailable")
    queries = cfg["inventory"]["rare_open_vocab"]["queries"]

    def run(img: np.ndarray) -> RawDets:
        boxes, probs = sc.score([img])[0]
        G = _group_scores(probs, sc.texts, queries)
        parts = []
        for g in GROUP_TO_LABEL:
            m = G[g] >= min_score
            if m.any():
                parts.append(RawDets(boxes[m], G[g][m].astype(np.float32), [g] * int(m.sum()), ["owl"] * int(m.sum())))
        d = classwise_nms(RawDets.concat(parts), 0.4)
        return RawDets(d.boxes, d.scores, [GROUP_TO_LABEL[g] for g in d.labels], [f"group:{g}" for g in d.labels])
    return run


def to_yaml_override(closed_thr: dict[str, float], owl_thr: dict[str, float], meta: dict[str, Any]) -> str:
    import yaml

    doc: dict[str, Any] = {}
    if closed_thr:
        doc["closed_set"] = {"score_thr": {k: float(v) for k, v in sorted(closed_thr.items())}}
    if owl_thr:
        inv_groups = {g: float(owl_thr[lab]) for g, lab in GROUP_TO_LABEL.items() if lab in owl_thr}
        doc["inventory"] = {"rare_open_vocab": {"thr": inv_groups}}
    head = "".join(f"# {k}: {v}\n" for k, v in meta.items())
    return head + yaml.safe_dump(doc, allow_unicode=True, sort_keys=True)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ml.detect.calibrate")
    ap.add_argument("--gt", default=str(DEFAULT_DATASET))
    ap.add_argument("--images", default="", help="корень изображений (по умолчанию info.images_root или <dir>/images)")
    ap.add_argument("--annotator", default=None)
    ap.add_argument("--target-precision", type=float, default=0.9)
    ap.add_argument("--min-support", type=int, default=10)
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--with-open-vocab", action="store_true", help="также пороги групп OWLv2 для inventory")
    ap.add_argument("--out", default="calib/detect_thresholds.yaml")
    ap.add_argument("--report", default="calib/detect_calibration.json")
    a = ap.parse_args(argv)
    coco = load_verified(a.gt, a.annotator)   # псевдоразметка / неизвестное происхождение → отказ
    root = Path(a.images) if a.images else default_images_root(a.gt, coco)
    cfg = load_config()
    kw = dict(target_precision=a.target_precision, min_support=a.min_support, holdout_frac=a.holdout_frac, iou_thr=a.iou)
    rep: dict[str, Any] = {"gt": a.gt, "label_status": coco["info"].get("label_status"),
                           "annotator": coco["info"].get("annotator") or coco["info"].get("reviewer"),
                           "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), **kw}
    cur = {k: label_threshold(cfg["closed_set"]["score_thr"], k) for k in ["default", *cfg["closed_set"]["score_thr"]]}
    rep["closed_set"] = calibrate(coco, root, closed_producer(cfg), cur, **kw)
    owl_thr: dict[str, float] = {}
    if a.with_open_vocab:
        cur_owl = {GROUP_TO_LABEL[g]: v for g, v in cfg["inventory"]["rare_open_vocab"]["thr"].items() if g in GROUP_TO_LABEL}
        rep["open_vocab"] = calibrate(coco, root, owl_producer(cfg), cur_owl, labels=sorted(set(GROUP_TO_LABEL.values())), **kw)
        owl_thr = rep["open_vocab"]["thresholds"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    meta = {"generated_by": "python -m ml.detect.calibrate", "gt": a.gt, "label_status": rep["label_status"],
            "annotator": rep["annotator"], "target_precision": a.target_precision, "created": rep["created"]}
    Path(a.out).write_text(to_yaml_override(rep["closed_set"]["thresholds"], owl_thr, meta), encoding="utf-8")
    Path(a.report).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    for part in ("closed_set", "open_vocab"):
        for lab, row in rep.get(part, {}).get("classes", {}).items():
            print(f"[{part}] {lab:14s} {row['status']:21s} thr={row.get('thr', row['current_thr'])} "
                  f"calib P/R={row['calib_P']}/{row['calib_R']} holdout P/R={row['holdout_P']}/{row['holdout_R']} "
                  f"n_gt={row['n_gt_calib']}+{row['n_gt_holdout']}")
    print(f"→ {a.out} (SENTINEL_DETECT_CONFIG={a.out}), отчёт {a.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
