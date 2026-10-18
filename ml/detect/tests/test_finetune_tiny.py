"""Опциональный путь дообучения: verified COCO → 2 шага Trainer на крошечной RT-DETRv2 → ONNX → OnnxDetrBackend."""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("accelerate")

pytestmark = pytest.mark.slow


def _tiny(classes):
    from transformers import RTDetrImageProcessor, RTDetrResNetConfig, RTDetrV2Config, RTDetrV2ForObjectDetection

    id2label = dict(enumerate(classes))
    bb = RTDetrResNetConfig(embedding_size=16, hidden_sizes=[16, 32, 64, 128], depths=[1, 1, 1, 1],
                            layer_type="basic", out_features=["stage2", "stage3", "stage4"])
    cfg = RTDetrV2Config(backbone_config=bb, encoder_hidden_dim=32, encoder_in_channels=[32, 64, 128], d_model=32,
                         encoder_ffn_dim=64, decoder_ffn_dim=64, num_queries=20, decoder_layers=2, encoder_layers=1,
                         num_labels=len(classes), decoder_n_points=2, num_denoising=0, decoder_in_channels=[32, 32, 32],
                         id2label=id2label, label2id={v: k for k, v in id2label.items()})
    return RTDetrV2ForObjectDetection(cfg), RTDetrImageProcessor(size={"height": 128, "width": 128})


def test_train_export_and_onnx_backend(tmp_path):
    import cv2

    from ml.detect.autolabel import FINETUNE_CLASSES, mark_verified
    from ml.detect.backends.onnx_det import OnnxDetrBackend
    from ml.detect.finetune import export_onnx, train

    rng = np.random.default_rng(0)
    images, anns = [], []
    for i in range(6):
        img = np.full((160, 96, 3), 30, np.uint8)
        x, y = int(rng.integers(5, 50)), int(rng.integers(5, 100))
        img[y:y + 20, x:x + 30] = (140, 255, 140)
        p = tmp_path / "frames" / f"X-{i % 3}" / f"{i:03d}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), img)
        images.append({"id": i + 1, "file_name": f"X-{i % 3}/{i:03d}.png", "width": 96, "height": 160})
        anns.append({"id": i + 1, "image_id": i + 1, "category_id": 1, "bbox": [x, y, 30, 20], "area": 600,
                     "iscrowd": 0, "verified": False})
    coco = {"info": {"verified": False}, "images": images, "annotations": anns,
            "categories": [{"id": k + 1, "name": c} for k, c in enumerate(FINETUNE_CLASSES)]}
    path = tmp_path / "verified.json"
    path.write_text(json.dumps(mark_verified(coco, "test-reviewer")))
    model, proc = _tiny(FINETUNE_CLASSES)
    hf_dir = train(path, tmp_path / "frames", tmp_path / "run", classes=FINETUNE_CLASSES, max_steps=2, batch=2,
                   model=model, processor=proc, augment=True)
    assert (hf_dir / "config.json").exists()
    meta = json.loads((tmp_path / "run" / "train_meta.json").read_text())
    assert meta["labels_reviewer"] == "test-reviewer" and meta["n_val_images"] > 0
    onnx_path = export_onnx(hf_dir, tmp_path / "onnx")
    be = OnnxDetrBackend(onnx_path, tmp_path / "onnx" / "labels.json", device="cpu", min_score=0.0, max_dets=5)
    res = be.predict([np.zeros((160, 96, 3), np.uint8), np.zeros((160, 96, 3), np.uint8)])
    assert len(res) == 2 and len(res[0]) == 5
    assert set(res[0].labels) <= set(FINETUNE_CLASSES)
    assert all(s.startswith("finetuned:") for s in res[0].sources)
    assert ((res[0].boxes >= 0) & (res[0].boxes <= 1)).all()
