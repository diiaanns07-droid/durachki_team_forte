"""Псевдоразметка = ПРЕДЛОЖЕНИЯ для локальной ручной проверки (не истинные bbox).

    python -m ml.detect.autolabel propose --frames data/frames --out data/autolabel/proposals.json
    # → импорт в CVAT / Label Studio (COCO), человек правит/удаляет/добавляет боксы, экспорт COCO →
    python -m ml.detect.autolabel mark-verified reviewed.json data/autolabel/verified.json --reviewer "<ФИО>"

Только verified.json (info.verified=true, все аннотации verified=true) принимает finetune.py.
Выход: COCO json; каждая аннотация несёт score/source/verified=false. Данные и кадры в git не коммитятся.
Запуск локально; кадры и описания сцен никуда не отправляются (OWLv2 — локальные веса).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .boxes import RawDets, area, classwise_nms, hflip_boxes, iou_matrix

log = logging.getLogger("ml.detect.autolabel")

# Нестандартные классы домена (метки контракта), ради которых дообучаем; совпадают с разметкой det-labels
# (safe_closed → contract_label safe). person/cup/food покрывает COCO; их можно добавить через --classes.
FINETUNE_CLASSES = ["cash", "cash_bag", "safe", "safe_open", "money_counter", "cell_phone", "door_open"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# Пороги для предложений строже рантайма: лучше пропустить, чем засорить проверку мусором
DEFAULT_MIN_SCORE = {"cash": 0.30, "cash_bag": 0.30, "safe": 0.30, "safe_open": 0.30, "money_counter": 0.30,
                     "cell_phone": 0.30, "door_open": 0.30}

PROPOSAL_WARNING = ("Псевдоразметка: предложения модели для ручной проверки, НЕ истинные bbox. "
                    "Не использовать для обучения/оценки без проверки человеком (mark-verified).")


def list_images(root: str | Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXT)


def video_key(file_name: str) -> str:
    """Группа (видео) кадра для честного train/val split: 'X-1/000123.jpg' → 'X-1'; 'X-1_t12.4.jpg' → 'X-1'.
    Кириллическая Х нормализуется в латинскую X."""
    p = Path(file_name)
    key = p.parent.name if p.parent.name not in ("", ".") else p.stem.split("_t")[0].split("_f")[0]
    return key.replace("Х", "X").replace("х", "x")


def flip_consistent(d: RawDets, d_flip: RawDets, iou_thr: float = 0.5) -> RawDets:
    """Оставляет боксы, подтверждённые на отражённом кадре (та же метка, IoU ≥ thr)."""
    if len(d) == 0 or len(d_flip) == 0:
        return RawDets()
    fb = hflip_boxes(d_flip.boxes)
    ious = iou_matrix(d.boxes, fb)
    fl = np.asarray(d_flip.labels)
    keep = [i for i, lab in enumerate(d.labels) if np.any((ious[i] >= iou_thr) & (fl == lab))]
    return d.select(keep)


def filter_proposals(d: RawDets, classes: Sequence[str], min_score: dict[str, float], min_area: float = 0.0004,
                     max_area: float = 0.5, max_aspect: float = 8.0, max_per_image: int = 20) -> RawDets:
    if len(d) == 0:
        return d
    a = area(d.boxes)
    w = np.clip(d.boxes[:, 2] - d.boxes[:, 0], 1e-6, None)
    h = np.clip(d.boxes[:, 3] - d.boxes[:, 1], 1e-6, None)
    asp = np.maximum(w / h, h / w)
    keep = [i for i, lab in enumerate(d.labels)
            if lab in classes and d.scores[i] >= min_score.get(lab, 0.3)
            and min_area <= a[i] <= max_area and asp[i] <= max_aspect]
    d = classwise_nms(d.select(keep), 0.5)
    return d.select(np.arange(min(len(d), max_per_image)))


def propose(images: Iterable[Path], backend: Any, classes: Sequence[str] = FINETUNE_CLASSES,
            min_score: Optional[dict[str, float]] = None, flip_check: bool = True, root: Optional[Path] = None,
            generator: str = "") -> dict[str, Any]:
    """Прогон бэкенда (OWLv2 / любой FrameDetector) по кадрам → COCO-словарь ПРЕДЛОЖЕНИЙ."""
    import cv2

    ms = {**DEFAULT_MIN_SCORE, **(min_score or {})}
    cats = [{"id": i + 1, "name": c} for i, c in enumerate(classes)]
    cat_id = {c["name"]: c["id"] for c in cats}
    coco: dict[str, Any] = {
        "info": {"kind": "proposals", "verified": False, "warning": PROPOSAL_WARNING, "generator": generator,
                 "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")},
        "images": [], "annotations": [], "categories": cats,
    }
    ann_id = 1
    for img_id, path in enumerate(images, start=1):
        img = cv2.imread(str(path))
        if img is None:
            log.warning("skip unreadable %s", path)
            continue
        h, w = img.shape[:2]
        rel = str(path.relative_to(root)) if root else path.name
        coco["images"].append({"id": img_id, "file_name": rel, "width": w, "height": h, "video": video_key(rel)})
        d = backend.predict([img])[0]
        if flip_check:
            d = flip_consistent(d, backend.predict([np.ascontiguousarray(img[:, ::-1])])[0])
        d = filter_proposals(d, classes, ms)
        for i, lab in enumerate(d.labels):
            x1, y1, x2, y2 = (float(v) for v in d.boxes[i])
            bw, bh = (x2 - x1) * w, (y2 - y1) * h
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": cat_id[lab],
                "bbox": [round(x1 * w, 2), round(y1 * h, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2), "iscrowd": 0,
                "score": round(float(d.scores[i]), 4), "source": d.sources[i], "verified": False,
            })
            ann_id += 1
    return coco


DEFAULT_DATASET = Path("datasets/det/annotations.json")   # ручная разметка (Дамир), COCO


def is_verified(coco: dict[str, Any]) -> bool:
    info = coco.get("info") or {}
    return bool(info.get("verified")) and all(bool(a.get("verified")) for a in coco.get("annotations", []))


def has_proposal_markers(coco: dict[str, Any]) -> bool:
    """Следы псевдоразметки: info.kind=proposals, verified=false или score без verified у аннотаций."""
    info = coco.get("info") or {}
    if str(info.get("kind", "")).lower() == "proposals":
        return True
    return any(a.get("verified") is False or ("score" in a and "verified" not in a) for a in coco.get("annotations", []))


def label_status(coco: dict[str, Any], annotator: Optional[str] = None) -> str:
    """verified (autolabel mark-verified) | manual (ручная разметка, указан аннотатор) | proposals | unknown."""
    if is_verified(coco):
        return "verified"
    if has_proposal_markers(coco):
        return "proposals"
    if annotator and annotator.strip():
        return "manual"
    return "unknown"


def default_images_root(ann_path: str | Path, coco: Optional[dict[str, Any]] = None) -> Path:
    """Корень изображений для COCO file_name.
    1) info.images_root (так пишет det-labels/export_coco.py: «data/frames», от корня репо);
    2) иначе <dir>/images, если есть; 3) иначе папка аннотаций."""
    from .weights import REPO_ROOT

    root = ((coco or {}).get("info") or {}).get("images_root")
    if root:
        p = Path(root).expanduser()
        return p if p.is_absolute() else REPO_ROOT / p
    d = Path(ann_path).parent
    return d / "images" if (d / "images").is_dir() else d


def category_label(cat: dict[str, Any]) -> Optional[str]:
    """Категория COCO → метка контракта: contract_label (det-labels: safe_closed → safe), иначе по имени."""
    from .labels import map_label

    cl = cat.get("contract_label")
    if cl:
        lab = map_label(str(cl))
        if lab is not None:
            return lab
    return map_label(str(cat.get("name", "")))


def group_key(im: dict[str, Any]) -> str:
    """Группа для split без утечки: group (det-labels: одинаковые по md5 ролики вместе) → video_id → video → папка."""
    for k in ("group", "video_id", "video"):
        if im.get(k):
            return str(im[k]).replace("Х", "X").replace("х", "x")
    return video_key(im["file_name"])


def mark_verified(coco: dict[str, Any], reviewer: str) -> dict[str, Any]:
    """Фиксирует, что человек просмотрел ВСЕ кадры файла (явное действие проверяющего, не автоматика)."""
    if not reviewer.strip():
        raise ValueError("нужно указать --reviewer")
    out = json.loads(json.dumps(coco))
    info = out.setdefault("info", {})
    info.update({"kind": "verified", "verified": True, "reviewer": reviewer.strip(),
                 "verified_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")})
    info.pop("warning", None)
    for a in out.get("annotations", []):
        a["verified"] = True
    return out


def _open_vocab_backend() -> Any:
    from .backends.owlv2 import OwlV2Backend
    from .config import load_config

    cfg = load_config()
    ov, rt = cfg["open_vocab"], cfg["runtime"]
    queries = [q for q in ov["queries"] if q["label"] in FINETUNE_CLASSES]
    return OwlV2Backend(ov["hf_model_id"], queries, device=rt["device"], fp16=rt["fp16"], input_size=ov["input_size"],
                        nms_iou=ov["nms_iou"], max_dets=ov["max_dets_per_frame"], min_area=ov.get("min_area", 0.0),
                        max_area=ov.get("max_area"))


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m ml.detect.autolabel")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose", help="OWLv2 → COCO-предложения (verified=false)")
    p.add_argument("--frames", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--no-flip-check", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    v = sub.add_parser("mark-verified", help="после ручной проверки в CVAT/Label Studio")
    v.add_argument("src")
    v.add_argument("dst")
    v.add_argument("--reviewer", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if a.cmd == "propose":
        imgs = list_images(a.frames)
        if a.limit:
            imgs = imgs[: a.limit]
        be = _open_vocab_backend()
        coco = propose(imgs, be, flip_check=not a.no_flip_check, root=Path(a.frames), generator=be.name)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(coco, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{len(coco['images'])} images, {len(coco['annotations'])} PROPOSALS → {a.out}\n{PROPOSAL_WARNING}")
    else:
        coco = json.loads(Path(a.src).read_text(encoding="utf-8"))
        Path(a.dst).write_text(json.dumps(mark_verified(coco, a.reviewer), ensure_ascii=False, indent=1),
                               encoding="utf-8")
        print(f"verified → {a.dst}")


if __name__ == "__main__":
    main()
