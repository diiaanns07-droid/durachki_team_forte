"""ONNX-бэкенд для дообученной (finetune.py → export) RT-DETRv2/D-FINE модели: onnxruntime(-gpu)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..boxes import RawDets, clip01, cxcywh_to_xyxy
from ..labels import build_index_map, source_tag

log = logging.getLogger("ml.detect.onnx")


class OnnxDetrBackend:
    """Ожидает вход pixel_values (B,3,S,S) и выходы logits (B,Q,C), pred_boxes (B,Q,4) cxcywh."""

    def __init__(self, onnx_path: str | Path, labels_path: str | Path, device: str = "auto",
                 input_size: int = 640, min_score: float = 0.15, max_dets: int = 100,
                 name: str = "finetuned", rescale: float = 1 / 255.0,
                 mean: Optional[Sequence[float]] = None, std: Optional[Sequence[float]] = None):
        import onnxruntime as ort

        meta = json.loads(Path(labels_path).read_text(encoding="utf-8"))
        names = meta["labels"] if isinstance(meta, dict) else meta
        if isinstance(meta, dict):
            input_size = int(meta.get("input_size", input_size))
            mean = meta.get("image_mean", mean) if meta.get("do_normalize") else None
            std = meta.get("image_std", std) if meta.get("do_normalize") else None
        self.index_map = build_index_map(list(names))
        self._cls_idx = sorted(self.index_map)
        self._cls_labels = [self.index_map[i] for i in self._cls_idx]
        self._cls_src = [source_tag(name, list(names)[i]) for i in self._cls_idx]
        providers = ["CPUExecutionProvider"]
        avail = ort.get_available_providers()
        if device != "cpu" and "CUDAExecutionProvider" in avail:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(onnx_path), providers=providers)
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.input_fp16 = "float16" in str(inp.type)
        outs = [o.name for o in self.sess.get_outputs()]
        self.out_logits = "logits" if "logits" in outs else outs[0]
        self.out_boxes = "pred_boxes" if "pred_boxes" in outs else outs[1]
        self.size = int(input_size)
        self.rescale = float(rescale)
        self.mean = np.asarray(mean, np.float32).reshape(1, 3, 1, 1) if mean is not None else None
        self.std = np.asarray(std, np.float32).reshape(1, 3, 1, 1) if std is not None else None
        self.min_score = float(min_score)
        self.max_dets = int(max_dets)
        self.name = name

    def _prep(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        import cv2

        arr = np.stack([cv2.resize(f, (self.size, self.size), interpolation=cv2.INTER_LINEAR)[..., ::-1]
                        for f in frames]).astype(np.float32) * self.rescale
        x = arr.transpose(0, 3, 1, 2)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        return np.ascontiguousarray(x.astype(np.float16 if self.input_fp16 else np.float32))

    def predict(self, frames: Sequence[np.ndarray]) -> list[RawDets]:
        if not frames:
            return []
        logits, boxes = self.sess.run([self.out_logits, self.out_boxes], {self.input_name: self._prep(frames)})
        logits = np.asarray(logits, np.float32)[..., self._cls_idx]
        scores = 1.0 / (1.0 + np.exp(-logits))  # (B,Q,K)
        boxes = np.asarray(boxes, np.float32)
        res = []
        b, q, k = scores.shape
        for bi in range(b):
            flat = scores[bi].reshape(-1)
            order = np.argsort(-flat)[: self.max_dets]
            order = order[flat[order] >= self.min_score]
            if order.size == 0:
                res.append(RawDets())
                continue
            qi, ki = order // k, order % k
            res.append(RawDets(clip01(cxcywh_to_xyxy(boxes[bi][qi])), flat[order].astype(np.float32),
                               [self._cls_labels[j] for j in ki], [self._cls_src[j] for j in ki]))
        return res
