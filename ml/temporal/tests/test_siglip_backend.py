"""Путь SigLIP-бэкенда на крошечной случайной модели (без скачивания весов): формы и нормировка."""
from __future__ import annotations

import numpy as np
import pytest


def test_siglip_backend_image_path_tiny_random(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import SiglipConfig, SiglipImageProcessor, SiglipModel

    cfg = SiglipConfig(
        text_config={"hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2, "intermediate_size": 37,
                     "vocab_size": 100, "max_position_embeddings": 64},
        vision_config={"hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
                       "intermediate_size": 37, "image_size": 224, "patch_size": 32})
    SiglipModel(cfg).save_pretrained(tmp_path)
    try:  # transformers 5.x: процессору нужны Pillow+torchvision; без них проверяем fallback бэкенда
        SiglipImageProcessor(size={"height": 224, "width": 224}).save_pretrained(tmp_path)
    except ImportError:
        pass

    from ml.temporal.features import SiglipBackend

    be = SiglipBackend(str(tmp_path), device="cpu", fp16=False)
    assert be.dim == 32 and be.processor is None
    assert np.allclose(be.mean, 0.5) and np.allclose(be.std, 0.5)        # нормировка SigLIP (из конфига или по умолчанию)
    e = be.encode_images(np.random.default_rng(0).integers(0, 255, (3, 224, 224, 3), dtype=np.uint8))
    assert e.shape == (3, 32) and np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-4)
    assert np.isfinite(be.logit_scale) and np.isfinite(be.logit_bias)
