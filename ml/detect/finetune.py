"""Дообучение RT-DETRv2 под домен (cash, cash_bag, safe_open, money_counter, cell_phone) — ОПЦИОНАЛЬНО.

Принимает ТОЛЬКО человеческую разметку: ручную (datasets/det/annotations.json, COCO, с --annotator) или
проверенные предложения (autolabel.py mark-verified). На псевдометках обучение запрещено (UnverifiedLabelsError).

    python -m ml.detect.finetune train --coco datasets/det/annotations.json --annotator "<имя>" \
        --out runs/detect_ft --epochs 30 --batch 8
    python -m ml.detect.finetune export --model runs/detect_ft/hf --out "$SENTINEL_WEIGHTS/detect/finetuned"
    SENTINEL_DETECT_EXTRAS=finetuned python -m ml.detect video.mov    # подключение в рантайме

Split train/val — по видео (кадры одного ролика не попадают в обе части). Аугментации (albumentations, MIT):
перспектива/аффинные — имитация другого угла камеры, яркость/цвет/смаз — другое освещение.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .autolabel import (DEFAULT_DATASET, FINETUNE_CLASSES, category_label, default_images_root, group_key,
                        label_status)
from .config import load_config
from .weights import local_model_dir, pretrained_kwargs

log = logging.getLogger("ml.detect.finetune")


class UnverifiedLabelsError(ValueError):
    pass


def load_verified(path: str | Path, annotator: Optional[str] = None) -> dict[str, Any]:
    """Только человеческая разметка: verified (mark-verified) или ручная (manual) с указанным аннотатором.
    Псевдоразметка (proposals) и файл неизвестного происхождения без --annotator — отказ."""
    coco = json.loads(Path(path).read_text(encoding="utf-8"))
    status = label_status(coco, annotator)
    if status in ("verified", "manual"):
        coco.setdefault("info", {})
        coco["info"]["label_status"] = status
        if status == "manual":
            coco["info"]["annotator"] = annotator
        return coco
    if status == "proposals":
        raise UnverifiedLabelsError(
            f"{path}: это псевдоразметка (предложения модели). Сначала ручная проверка и autolabel mark-verified.")
    raise UnverifiedLabelsError(
        f"{path}: происхождение разметки неизвестно. Для ручной разметки укажите аннотатора (--annotator <имя>).")


def split_by_video(coco: dict[str, Any], val_frac: float = 0.2, seed: int = 0) -> tuple[list[int], list[int]]:
    """(train_ids, val_ids). Если у всех кадров есть split (det-labels фиксирует его в split.json) — берём его;
    иначе делим по группам видео (group_key: дубликаты по md5 не разъезжаются по train/val)."""
    ims = coco["images"]
    if ims and all(im.get("split") in ("train", "val") for im in ims):
        return ([im["id"] for im in ims if im["split"] == "train"], [im["id"] for im in ims if im["split"] == "val"])
    groups: dict[str, list[int]] = {}
    for im in ims:
        groups.setdefault(group_key(im), []).append(im["id"])
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_val = int(round(len(keys) * val_frac)) if len(keys) > 1 else 0
    val_keys = set(keys[:n_val])
    tr = [i for k in keys if k not in val_keys for i in groups[k]]
    va = [i for k in keys if k in val_keys for i in groups[k]]
    return tr, va


def build_augment() -> Any:
    try:
        import albumentations as A
    except ImportError:
        log.warning("albumentations не установлен — обучение без аугментаций")
        return None
    return A.Compose([
        A.Perspective(scale=(0.02, 0.08), p=0.5),
        A.Affine(rotate=(-15, 15), scale=(0.85, 1.15), translate_percent=(-0.05, 0.05), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=20, p=0.3),
        A.MotionBlur(blur_limit=5, p=0.2),
    ], bbox_params=A.BboxParams(format="coco", label_fields=["category"], min_visibility=0.3))


class CocoDetDataset:
    def __init__(self, coco: dict[str, Any], image_ids: Sequence[int], root: str | Path, processor: Any,
                 classes: Sequence[str], augment: Any = None):
        self.root = Path(root)
        self.proc = processor
        self.aug = augment
        self.images = {im["id"]: im for im in coco["images"]}
        self.ids = [i for i in image_ids if i in self.images]
        cat_label = {c["id"]: category_label(c) for c in coco["categories"]}
        self.cls_idx = {c: i for i, c in enumerate(classes)}
        self.anns: dict[int, list[tuple[list[float], int]]] = {i: [] for i in self.ids}
        for a in coco["annotations"]:
            name = cat_label.get(a["category_id"])
            if a["image_id"] in self.anns and name in self.cls_idx and not a.get("iscrowd", 0):
                self.anns[a["image_id"]].append(([float(v) for v in a["bbox"]], self.cls_idx[name]))

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, k: int) -> dict[str, Any]:
        import cv2

        iid = self.ids[k]
        img = cv2.imread(str(self.root / self.images[iid]["file_name"]))
        if img is None:
            raise FileNotFoundError(self.images[iid]["file_name"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        boxes, cats = [], []
        for (x, y, bw, bh), c in self.anns[iid]:
            x, y = max(0.0, x), max(0.0, y)
            bw, bh = min(bw, w - x), min(bh, h - y)
            if bw > 1 and bh > 1:
                boxes.append([x, y, bw, bh])
                cats.append(c)
        if self.aug is not None:
            r = self.aug(image=img, bboxes=boxes, category=cats)
            img, boxes, cats = r["image"], [list(b) for b in r["bboxes"]], list(r["category"])
        anns = [{"bbox": b, "category_id": int(c), "area": float(b[2] * b[3]), "iscrowd": 0} for b, c in zip(boxes, cats)]
        enc = self.proc(images=img, annotations={"image_id": iid, "annotations": anns}, return_tensors="pt")
        return {"pixel_values": enc["pixel_values"][0], "labels": enc["labels"][0]}


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    return {"pixel_values": torch.stack([b["pixel_values"] for b in batch]), "labels": [b["labels"] for b in batch]}


def train(coco_path: str | Path, frames_root: Optional[str | Path], out_dir: str | Path, base_model: Optional[str] = None,
          classes: Sequence[str] = FINETUNE_CLASSES, epochs: int = 30, batch: int = 8, lr: float = 1e-4,
          max_steps: int = -1, val_frac: float = 0.2, augment: bool = True, model: Any = None,
          processor: Any = None, seed: int = 0, annotator: Optional[str] = None) -> Path:
    import torch
    from transformers import Trainer, TrainingArguments

    coco = load_verified(coco_path, annotator)
    frames_root = Path(frames_root) if frames_root else default_images_root(coco_path, coco)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    classes = list(classes)
    id2label = dict(enumerate(classes))
    label2id = {c: i for i, c in id2label.items()}
    base = base_model or load_config()["closed_set"]["hf_model_id"]
    if model is None or processor is None:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        src = str(local_model_dir(base) or base)
        kw = {} if local_model_dir(base) else pretrained_kwargs(base)
        processor = processor or AutoImageProcessor.from_pretrained(src, **kw)
        model = model or AutoModelForObjectDetection.from_pretrained(
            src, id2label=id2label, label2id=label2id, num_labels=len(classes), ignore_mismatched_sizes=True, **kw)
    tr_ids, va_ids = split_by_video(coco, val_frac, seed)
    ds_tr = CocoDetDataset(coco, tr_ids, frames_root, processor, classes, build_augment() if augment else None)
    ds_va = CocoDetDataset(coco, va_ids, frames_root, processor, classes, None) if va_ids else None
    cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=str(out / "trainer"), num_train_epochs=epochs, max_steps=max_steps,
        per_device_train_batch_size=batch, per_device_eval_batch_size=batch, learning_rate=lr, weight_decay=1e-4,
        warmup_steps=0 if max_steps > 0 else 100, fp16=cuda, eval_strategy="epoch" if ds_va else "no",
        save_strategy="epoch" if max_steps < 0 else "no", save_total_limit=2, remove_unused_columns=False,
        report_to=[], logging_steps=10, dataloader_num_workers=2 if cuda else 0, seed=seed,
        use_cpu=not cuda,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va, data_collator=collate)
    trainer.train()
    hf_dir = out / "hf"
    model.save_pretrained(hf_dir)
    processor.save_pretrained(hf_dir)
    info = coco.get("info", {})
    (out / "train_meta.json").write_text(json.dumps({
        "base_model": base, "classes": classes, "n_train_images": len(ds_tr), "n_val_images": len(va_ids),
        "val_videos": sorted({group_key(im) for im in coco["images"] if im["id"] in set(va_ids)}),
        "frames_root": str(frames_root),
        "labels_status": info.get("label_status"), "labels_reviewer": info.get("reviewer") or info.get("annotator"),
        "labels_verified_at": info.get("verified_at"),
        "epochs": epochs, "max_steps": max_steps, "lr": lr, "batch": batch,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return hf_dir


def export_onnx(model_dir: str | Path, out_dir: str | Path, opset: int = 17, model: Any = None,
                processor: Any = None) -> Path:
    """HF-модель → model.onnx (вход pixel_values B×3×S×S; выходы logits, pred_boxes) + labels.json."""
    import torch

    if model is None:
        from transformers import AutoModelForObjectDetection

        model = AutoModelForObjectDetection.from_pretrained(str(model_dir))
    if processor is None:
        try:
            from transformers import AutoImageProcessor

            processor = AutoImageProcessor.from_pretrained(str(model_dir))
        except Exception:  # noqa: BLE001
            processor = None
    model = model.eval().float()
    size = 640
    if processor is not None:
        from .backends.hf_detr import _size_from_processor

        size = _size_from_processor(processor, 640)[0]

    class _Wrap(torch.nn.Module):
        def __init__(self, m: Any):
            super().__init__()
            self.m = m

        def forward(self, pixel_values: Any) -> Any:
            o = self.m(pixel_values=pixel_values)
            return o.logits, o.pred_boxes

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dummy = torch.rand(1, 3, size, size)
    kw: dict[str, Any] = dict(input_names=["pixel_values"], output_names=["logits", "pred_boxes"],
                              dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"},
                                            "pred_boxes": {0: "batch"}}, opset_version=opset)
    try:
        torch.onnx.export(_Wrap(model), (dummy,), str(out / "model.onnx"), dynamo=False, **kw)
    except TypeError:  # старые torch без аргумента dynamo
        torch.onnx.export(_Wrap(model), (dummy,), str(out / "model.onnx"), **kw)
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    do_norm = bool(getattr(processor, "do_normalize", False)) if processor is not None else False
    meta = {"labels": [id2label[i] for i in range(len(id2label))], "input_size": size, "do_normalize": do_norm,
            "image_mean": list(getattr(processor, "image_mean", []) or []) if do_norm else None,
            "image_std": list(getattr(processor, "image_std", []) or []) if do_norm else None}
    (out / "labels.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out / "model.onnx"


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m ml.detect.finetune")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--coco", default=str(DEFAULT_DATASET), help="COCO (по умолчанию datasets/det/annotations.json)")
    t.add_argument("--frames", default="", help="корень изображений (по умолчанию info.images_root или <dir>/images)")
    t.add_argument("--classes", default=",".join(FINETUNE_CLASSES), help="метки контракта через запятую")
    t.add_argument("--annotator", default=None, help="автор ручной разметки (обязателен для файла без mark-verified)")
    t.add_argument("--out", required=True)
    t.add_argument("--base", default=None)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--no-augment", action="store_true")
    e = sub.add_parser("export")
    e.add_argument("--model", required=True)
    e.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if a.cmd == "train":
        classes = [c.strip() for c in a.classes.split(",") if c.strip()]
        print(train(a.coco, a.frames or None, a.out, a.base, classes=classes, epochs=a.epochs, batch=a.batch, lr=a.lr,
                    augment=not a.no_augment, annotator=a.annotator))
    else:
        print(export_onnx(a.model, a.out))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
