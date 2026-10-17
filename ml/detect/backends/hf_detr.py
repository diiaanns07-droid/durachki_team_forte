"""Closed-set бэкенд: RT-DETRv2 / D-FINE (Apache-2.0) через transformers, препроцессинг на GPU, fp16."""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

from ..boxes import RawDets, clip01, cxcywh_to_xyxy
from ..labels import STANDARD_LABELS, build_index_map, source_tag
from ..weights import local_model_dir, pretrained_kwargs
from .base import group_by_shape, resolve_device, torch_preprocess

log = logging.getLogger("ml.detect.hf_detr")


def _size_from_processor(proc: Any, default: int) -> tuple[int, int]:
    """size бывает dict (transformers 4.x) или SizeDict (5.x) — читаем терпимо к обоим."""
    size = getattr(proc, "size", None)
    if isinstance(size, int):
        return size, size

    def get(key: str) -> Any:
        if size is None:
            return None
        try:
            v = size[key] if isinstance(size, dict) else getattr(size, key, None)
        except (KeyError, TypeError):
            v = None
        return v

    h = get("height") or get("shortest_edge") or default
    w = get("width") or get("shortest_edge") or default
    return int(h), int(w)


class HFDetrBackend:
    """RT-DETR / RT-DETRv2 / D-FINE: logits (B,Q,C) с sigmoid, pred_boxes (B,Q,4) cxcywh в долях входа.

    Вход ресайзится в квадрат без сохранения пропорций (как в RTDetrImageProcessor), поэтому
    нормированные боксы модели == нормированные боксы исходного кадра.
    """

    def __init__(self, model_id: str, device: str = "auto", fp16: bool = True, input_size: int = 640,
                 min_score: float = 0.15, max_dets: int = 100, name: Optional[str] = None,
                 model: Any = None, processor: Any = None, allow_proxy: bool = True):
        import torch

        self.device = resolve_device(device)
        self.fp16 = bool(fp16) and self.device.startswith("cuda")
        self.dtype = torch.float16 if self.fp16 else torch.float32
        self.min_score = float(min_score)
        self.max_dets = int(max_dets)
        self.model_id = model_id
        self.name = name or model_id.rstrip("/").split("/")[-1]
        if model is None:
            from transformers import AutoModelForObjectDetection

            src = str(local_model_dir(model_id) or model_id)
            kw = {} if local_model_dir(model_id) else pretrained_kwargs(model_id)
            model = AutoModelForObjectDetection.from_pretrained(src, **kw)
            if processor is None:
                try:
                    from transformers import AutoImageProcessor

                    processor = AutoImageProcessor.from_pretrained(src, **kw)
                except Exception as e:  # noqa: BLE001
                    log.warning("image processor for %s not loaded (%s) — defaults used", model_id, e)
        self.model = model.eval().to(self.device)
        if self.fp16:
            self.model = self.model.half()
        if self.device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True  # вход фиксированного размера
        self.size = _size_from_processor(processor, input_size) if processor is not None else (input_size, input_size)
        self.rescale = float(getattr(processor, "rescale_factor", 1 / 255.0)) if processor is not None else 1 / 255.0
        do_norm = bool(getattr(processor, "do_normalize", False)) if processor is not None else False
        self.mean = getattr(processor, "image_mean", None) if do_norm else None
        self.std = getattr(processor, "image_std", None) if do_norm else None
        id2label = {int(k): v for k, v in (getattr(self.model.config, "id2label", {}) or {}).items()}
        # общий (COCO) детектор не выдаёт cash/safe/door/... даже если в id2label есть похожие имена
        self.index_map = build_index_map(id2label, allow_proxy=allow_proxy, allowed=STANDARD_LABELS)
        if not self.index_map:
            raise RuntimeError(f"{model_id}: ни один класс id2label не отображается в LABELS")
        self._cls_idx = sorted(self.index_map)
        self._cls_labels = [self.index_map[i] for i in self._cls_idx]
        self._cls_src = [source_tag(self.name, id2label[i]) for i in self._cls_idx]

    def _forward(self, frames: Sequence[np.ndarray]) -> list[RawDets]:
        import torch

        x = torch_preprocess(frames, self.size, self.device, self.dtype, rescale=self.rescale,
                             mean=self.mean, std=self.std)
        with torch.inference_mode():
            out = self.model(pixel_values=x)
        logits = out.logits.float()
        boxes = out.pred_boxes.float()
        scores = logits[..., self._cls_idx].sigmoid()  # (B, Q, K)
        b, q, k = scores.shape
        flat = scores.reshape(b, q * k)
        topk = min(self.max_dets, q * k)
        top_s, top_i = flat.topk(topk, dim=1)
        res: list[RawDets] = []
        top_s, top_i = top_s.cpu().numpy(), top_i.cpu().numpy()
        boxes_np = boxes.cpu().numpy()
        for bi in range(b):
            keep = top_s[bi] >= self.min_score
            qi = top_i[bi][keep] // k
            ki = top_i[bi][keep] % k
            if not keep.any():
                res.append(RawDets())
                continue
            xyxy = clip01(cxcywh_to_xyxy(boxes_np[bi][qi]))
            res.append(RawDets(xyxy, top_s[bi][keep].astype(np.float32),
                               [self._cls_labels[j] for j in ki], [self._cls_src[j] for j in ki]))
        return res

    def _forward_safe(self, frames: Sequence[np.ndarray]) -> list[RawDets]:
        try:
            return self._forward(frames)
        except RuntimeError as e:
            if not self.fp16 or "out of memory" in str(e).lower():
                raise
            import torch

            log.warning("%s: fp16 forward failed (%s) — switching to fp32", self.name, e)
            self.fp16, self.dtype = False, torch.float32
            self.model = self.model.float()
            return self._forward(frames)

    def predict(self, frames: Sequence[np.ndarray]) -> list[RawDets]:
        out: list[Optional[RawDets]] = [None] * len(frames)
        for _shape, idx in group_by_shape(frames).items():
            for r, i in zip(self._forward_safe([frames[j] for j in idx]), idx):
                out[i] = r
        return [r if r is not None else RawDets() for r in out]
