"""RTMPose (COCO-17, SimCC) на onnxruntime с батчингом кропов. Пре/пост-обработка как в rtmlib (Apache-2.0).

Веса: ONNX из rtmlib/openmmlab (rtmpose-m_simcc-body7_pt-body7_420e-256x192). Поиск файла:
  1) backend.rtmpose.onnx_path  2) $SENTINEL_POSE_ONNX  3) $SENTINEL_WEIGHTS_DIR/{pose,}/**/*<model>*.onnx
  4) ./weights/pose/  5) кэш rtmlib (~/.cache/rtmlib/hub/checkpoints). Сеть не используется.
"""
from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .base import PoseBackendError, WeightsNotFound

log = logging.getLogger("ml.pose")


def find_weights(cfg: dict[str, Any]) -> Optional[str]:
    rc = cfg.get("backend", {}).get("rtmpose", {})
    explicit = rc.get("onnx_path") or os.environ.get("SENTINEL_POSE_ONNX")
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    key = str(rc.get("model", "rtmpose-m"))
    dirs: list[str] = []
    wd = os.environ.get("SENTINEL_WEIGHTS_DIR")
    if wd:
        dirs += [os.path.join(wd, "pose"), wd]
    dirs += [os.path.join(os.getcwd(), "weights", "pose"),
             os.path.join(os.environ.get("TORCH_HOME", os.path.join(os.environ.get("XDG_CACHE_HOME", "~/.cache"), "rtmlib")), "hub", "checkpoints")]
    for d in dirs:
        d = os.path.expanduser(d)
        if not os.path.isdir(d):
            continue
        hits = sorted(glob.glob(os.path.join(d, "**", "*.onnx"), recursive=True))
        for h in hits:
            if key in Path(h).as_posix():
                return h
    return None


def bbox_xyxy2cs(box: np.ndarray, padding: float) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = (float(v) for v in box)
    center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
    scale = np.array([(x2 - x1) * padding, (y2 - y1) * padding], dtype=np.float32)
    return center, scale


def fix_aspect(scale: np.ndarray, input_wh: tuple[int, int]) -> np.ndarray:
    w, h = input_wh
    ar = w / h
    bw, bh = float(scale[0]), float(scale[1])
    return np.array([bw, bw / ar] if bw > bh * ar else [bh * ar, bh], dtype=np.float32)


def warp_matrix(center: np.ndarray, scale: np.ndarray, input_wh: tuple[int, int]) -> np.ndarray:
    """Аффинная матрица bbox → вход модели (rot=0), эквивалент rtmlib.get_warp_matrix."""
    import cv2

    w, h = input_wh
    src_w = float(scale[0])
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0] = center
    src[1] = center + np.array([0.0, -0.5 * src_w], dtype=np.float32)
    d = src[0] - src[1]
    src[2] = src[1] + np.array([-d[1], d[0]], dtype=np.float32)
    dst[0] = [w * 0.5, h * 0.5]
    dst[1] = dst[0] + np.array([0.0, -0.5 * w], dtype=np.float32)
    d = dst[0] - dst[1]
    dst[2] = dst[1] + np.array([-d[1], d[0]], dtype=np.float32)
    return cv2.getAffineTransform(src, dst)


def simcc_decode(simcc_x: np.ndarray, simcc_y: np.ndarray, split_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """(N, K, Wx), (N, K, Wy) → locs (N, K, 2) во входных координатах модели, scores (N, K)."""
    n, k, _ = simcc_x.shape
    sx = simcc_x.reshape(n * k, -1)
    sy = simcc_y.reshape(n * k, -1)
    locs = np.stack([np.argmax(sx, axis=1), np.argmax(sy, axis=1)], axis=-1).astype(np.float32)
    vals = 0.5 * (np.amax(sx, axis=1) + np.amax(sy, axis=1))
    locs[vals <= 0.0] = -1
    return locs.reshape(n, k, 2) / split_ratio, vals.reshape(n, k)


class RTMPoseOnnx:
    name = "rtmpose-onnx"

    def __init__(self, cfg: dict[str, Any]) -> None:
        rc = cfg.get("backend", {}).get("rtmpose", {})
        path = find_weights(cfg)
        if not path:
            raise WeightsNotFound("ONNX-веса RTMPose не найдены (см. ml/pose/README.md → «Веса»)")
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise PoseBackendError("onnxruntime не установлен") from exc
        device = str(cfg.get("backend", {}).get("device", "auto")).lower()
        avail = ort.get_available_providers()
        providers: list[Any] = ["CPUExecutionProvider"]
        if device in ("auto", "cuda") and "CUDAExecutionProvider" in avail:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        threads = int(rc.get("intra_op_threads", 0) or 0)
        if threads > 0:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, sess_options=so, providers=providers)
        self.path = path
        self.providers = self.sess.get_providers()
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.dynamic_batch = not isinstance(inp.shape[0], int) or int(inp.shape[0]) <= 0
        self.input_wh = (int(rc.get("input_size", [192, 256])[0]), int(rc.get("input_size", [192, 256])[1]))
        if isinstance(inp.shape[2], int) and isinstance(inp.shape[3], int):
            self.input_wh = (int(inp.shape[3]), int(inp.shape[2]))
        self.mean = np.array(rc.get("mean", [123.675, 116.28, 103.53]), dtype=np.float32)
        self.std = np.array(rc.get("std", [58.395, 57.12, 57.375]), dtype=np.float32)
        self.bgr_to_rgb = bool(rc.get("bgr_to_rgb", False))
        self.padding = float(rc.get("padding", 1.25))
        self.split = float(rc.get("simcc_split_ratio", 2.0))
        self.batch_size = int(rc.get("batch_size", 16)) if self.dynamic_batch else 1
        log.info("pose: RTMPose ONNX %s providers=%s dynamic_batch=%s", path, self.providers, self.dynamic_batch)

    def _prep(self, frame: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import cv2

        center, scale = bbox_xyxy2cs(box, self.padding)
        scale = fix_aspect(scale, self.input_wh)
        mat = warp_matrix(center, scale, self.input_wh)
        img = cv2.warpAffine(frame, mat, self.input_wh, flags=cv2.INTER_LINEAR)
        if self.bgr_to_rgb:
            img = img[..., ::-1]
        img = (img.astype(np.float32) - self.mean) / self.std
        return np.ascontiguousarray(img.transpose(2, 0, 1)), center, scale

    def infer(self, items: Sequence[tuple[np.ndarray, np.ndarray]], meta: Any = None) -> list[tuple[np.ndarray, np.ndarray]]:
        if not items:
            return []
        preps = [self._prep(f, b) for f, b in items]
        batch = np.stack([p[0] for p in preps]).astype(np.float32)
        xs, ys = [], []
        for i in range(0, len(batch), self.batch_size):
            out = self.sess.run(self.out_names, {self.input_name: batch[i:i + self.batch_size]})
            xs.append(out[0])
            ys.append(out[1])
        locs, scores = simcc_decode(np.concatenate(xs), np.concatenate(ys), self.split)
        wh = np.array(self.input_wh, dtype=np.float32)
        res = []
        for (_, center, scale), loc, sc in zip(preps, locs, scores):
            kp = loc / wh * scale + center - scale / 2.0
            res.append((kp[:17].astype(np.float32), np.clip(sc[:17], 0.0, 1.0).astype(np.float32)))
        return res
