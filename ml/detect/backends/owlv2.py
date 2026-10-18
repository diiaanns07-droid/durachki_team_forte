"""Open-vocab бэкенд: OWLv2 (google/owlv2-base-patch16-ensemble, Apache-2.0) через transformers.

Препроцессинг на GPU повторяет Owlv2ImageProcessor: паддинг справа/снизу до квадрата серым (0.5),
resize до 960, CLIP-нормировка. Боксы модели нормированы к квадрату → пересчитываем к кадру.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

from ..boxes import RawDets, classwise_nms, clip01, cxcywh_to_xyxy, filter_by_area
from ..labels import validate_label
from ..weights import local_model_dir, pretrained_kwargs
from .base import group_by_shape, resolve_device, torch_preprocess

log = logging.getLogger("ml.detect.owlv2")

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class OwlV2Backend:
    def __init__(self, model_id: str, queries: Sequence[dict], device: str = "auto", fp16: bool = True,
                 input_size: int = 960, nms_iou: float = 0.4, max_dets: int = 40, min_area: float = 0.0,
                 max_area: Optional[dict[str, float]] = None, name: str = "owlv2",
                 model: Any = None, processor: Any = None):
        import torch

        self.device = resolve_device(device)
        self.fp16 = bool(fp16) and self.device.startswith("cuda")
        self.dtype = torch.float16 if self.fp16 else torch.float32
        self.name = name
        self.queries = [dict(q) for q in queries]
        self.texts = [str(q["text"]) for q in self.queries]
        # label нужен только для detect(); для сырого скоринга (inventory) запросы могут быть без метки
        self.labels = [validate_label(str(q["label"])) if q.get("label") else "" for q in self.queries]
        self.thr = np.asarray([float(q.get("thr", 0.2)) for q in self.queries], np.float32)
        self.nms_iou = float(nms_iou)
        self.max_dets = int(max_dets)
        self.min_area = float(min_area)
        self.max_area = dict(max_area or {})
        if model is None or processor is None:
            from transformers import Owlv2ForObjectDetection, Owlv2Processor

            src = str(local_model_dir(model_id) or model_id)
            kw = {} if local_model_dir(model_id) else pretrained_kwargs(model_id)
            model = model or Owlv2ForObjectDetection.from_pretrained(src, **kw)
            processor = processor or Owlv2Processor.from_pretrained(src, **kw)
        self.model = model.eval().to(self.device)
        if self.fp16:
            self.model = self.model.half()
        ip = getattr(processor, "image_processor", None)
        vc = getattr(getattr(self.model, "config", None), "vision_config", None)
        self.native = int(getattr(vc, "image_size", 0) or input_size)
        self.size = (int(input_size), int(input_size))
        # не родной размер → интерполяция позиционных эмбеддингов (быстрее на 768, чуть хуже качество)
        self.interp = self.size[0] != self.native
        self.mean = tuple(getattr(ip, "image_mean", None) or CLIP_MEAN)
        self.std = tuple(getattr(ip, "image_std", None) or CLIP_STD)
        tok = getattr(processor, "tokenizer", processor)
        enc = tok(self.texts, padding="max_length", max_length=16, truncation=True, return_tensors="pt")
        self.input_ids = enc["input_ids"].to(self.device)
        self.attention_mask = enc["attention_mask"].to(self.device)

    def _raw(self, frames: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Кадры одного размера → (boxes (B,P,4) xyxy в долях КАДРА, probs (B,P,Q) sigmoid по всем запросам)."""
        import torch

        h, w = frames[0].shape[:2]
        s = float(max(h, w))
        x = torch_preprocess(frames, self.size, self.device, self.dtype, rescale=1 / 255.0,
                             mean=self.mean, std=self.std, pad_square=True, pad_value=0.5)
        b = x.shape[0]
        with torch.inference_mode():
            kw = {"interpolate_pos_encoding": True} if self.interp else {}
            out = self.model(input_ids=self.input_ids.repeat(b, 1), pixel_values=x,
                             attention_mask=self.attention_mask.repeat(b, 1), **kw)
        probs = out.logits.float().sigmoid().cpu().numpy()      # (B, P, Q)
        boxes = out.pred_boxes.float().cpu().numpy()            # (B, P, 4) cxcywh, доли квадрата
        xyxy = cxcywh_to_xyxy(boxes.reshape(-1, 4)).reshape(boxes.shape)
        xyxy[..., [0, 2]] *= s / w                              # снимаем паддинг до квадрата
        xyxy[..., [1, 3]] *= s / h
        return np.clip(xyxy, 0.0, 1.0), probs

    def score(self, frames: Sequence[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Сырые скоры всех запросов по всем патчам: [(boxes (P,4), probs (P,Q))] на кадр — для контрастных проверок."""
        out: list[Optional[tuple[np.ndarray, np.ndarray]]] = [None] * len(frames)
        for _shape, idx in group_by_shape(frames).items():
            boxes, probs = self._raw([frames[j] for j in idx])
            for k, i in enumerate(idx):
                out[i] = (clip01(boxes[k]), probs[k])
        return [o for o in out if o is not None]

    def _forward(self, frames: Sequence[np.ndarray]) -> list[RawDets]:
        boxes, probs = self._raw(frames)
        res = []
        for bi in range(len(frames)):
            p = probs[bi]
            qbest = p.argmax(axis=1)
            sbest = p[np.arange(len(p)), qbest]
            keep = (sbest >= self.thr[qbest]) & np.asarray([bool(self.labels[q]) for q in qbest])
            if not keep.any():
                res.append(RawDets())
                continue
            d = RawDets(clip01(boxes[bi][keep]), sbest[keep].astype(np.float32),
                        [self.labels[q] for q in qbest[keep]],
                        [f"{self.name}:{self.texts[q]}" for q in qbest[keep]])
            d = filter_by_area(d, self.min_area, self.max_area)
            d = classwise_nms(d, self.nms_iou)
            if len(d) > self.max_dets:
                d = d.select(np.arange(self.max_dets))
            res.append(d)
        return res

    def predict(self, frames: Sequence[np.ndarray]) -> list[RawDets]:
        out: list[Optional[RawDets]] = [None] * len(frames)
        for _shape, idx in group_by_shape(frames).items():
            for r, i in zip(self._forward([frames[j] for j in idx]), idx):
                out[i] = r
        return [r if r is not None else RawDets() for r in out]
