"""[H2, по умолчанию выкл] Open-vocab «именование» кластеров вида ROI: какой из видов — «открыто».

SigLIP2 zero-shot по нескольким цветным кропам-представителям каждого кластера (≤ 2×k кропов на ролик,
одна партия). Модель грузится один раз из офлайн-кэша (``weights/hf/hub``, local_files_only) и держится
тёплой. Любая ошибка загрузки → именователь выключен с предупреждением (НИКОГДА не подменяется моком).
Моки — только в тестах (monkeypatch ``get_namer``).

Лицензия модели: google/siglip2-base-patch16-224 — Apache-2.0 (models.json).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .config import is_mock

log = logging.getLogger("ml.scene.namer")

PROMPTS: dict[str, dict[str, list[str]]] = {
    "safe": {
        "open": ["a photo of an open safe", "an open metal safe with its door open", "a safe with an open door and shelves inside"],
        "closed": ["a photo of a closed safe", "a closed metal safe with its door shut", "a grey metal safe with the door closed"],
    },
    "door": {
        "open": ["a photo of an open door", "a doorway with the door wide open"],
        "closed": ["a photo of a closed door", "a closed glass door"],
    },
}
Namer = Callable[[list[np.ndarray], str], list[float]]   # RGB-кропы, объект → P(open) на кроп
_STATE: dict[str, Any] = {}
_REPO_ROOT = Path(__file__).resolve().parents[2]


def get_namer(cfg_namer: dict[str, Any], backend: str) -> Optional[Namer]:
    """backend: none | siglip2. В mock-режиме CI и при ошибке загрузки — None (именования нет)."""
    if backend in ("", "none", None) or is_mock():
        return None
    if backend == "siglip2":
        return _siglip2(cfg_namer)
    log.warning("unknown namer backend %r — disabled", backend)
    return None


def _siglip2(cfg: dict[str, Any]) -> Optional[Namer]:
    key = f"siglip2:{cfg.get('model')}"
    if key in _STATE:
        return _STATE[key]
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
    except ImportError as e:
        log.warning("namer disabled: %s", e)
        _STATE[key] = None
        return None
    repo = str(cfg.get("model", "google/siglip2-base-patch16-224"))
    cache = Path(os.environ.get("SENTINEL_HF_CACHE") or cfg.get("cache_dir", "weights/hf/hub"))
    cache = cache if cache.is_absolute() else _REPO_ROOT / cache
    dev = str(cfg.get("device", "auto"))
    device = "cuda" if dev == "auto" and torch.cuda.is_available() else ("cpu" if dev == "auto" else dev)
    dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        hf = dict(cache_dir=str(cache), local_files_only=True)  # без сети
        proc = AutoProcessor.from_pretrained(repo, **hf)
        model = AutoModel.from_pretrained(repo, dtype=dtype, **hf).to(device).eval()
    except Exception as e:  # noqa: BLE001 — офлайн-веса недоступны → именователь выключен
        log.warning("namer disabled (cannot load %s from %s): %s", repo, cache, e)
        _STATE[key] = None
        return None
    text_cache: dict[str, Any] = {}

    def _feat(out: Any) -> Any:
        return out if isinstance(out, torch.Tensor) else out.pooler_output   # transformers 5.x: объект с pooler_output

    @torch.inference_mode()
    def text_emb(obj: str) -> tuple[Any, int]:
        if obj not in text_cache:
            pr = PROMPTS.get(obj, PROMPTS["safe"])
            texts = pr["open"] + pr["closed"]
            inp = proc(text=texts, padding="max_length", max_length=64, return_tensors="pt").to(device)
            f = torch.nn.functional.normalize(_feat(model.get_text_features(**inp)).float(), dim=-1)
            text_cache[obj] = (f, len(pr["open"]))
        return text_cache[obj]

    @torch.inference_mode()
    def score(crops_rgb: list[np.ndarray], obj: str) -> list[float]:
        if not crops_rgb:
            return []
        tf, n_open = text_emb(obj)
        inp = proc(images=list(crops_rgb), return_tensors="pt").to(device)
        inp["pixel_values"] = inp["pixel_values"].to(model.dtype)
        img = torch.nn.functional.normalize(_feat(model.get_image_features(**inp)).float(), dim=-1)
        sims = img @ tf.T                                    # косинус (n_crops × n_prompts)
        scale = float(model.logit_scale.exp()) if hasattr(model, "logit_scale") else 10.0
        margin = sims[:, :n_open].mean(dim=1) - sims[:, n_open:].mean(dim=1)
        return [float(v) for v in torch.sigmoid(scale * margin).cpu().numpy()]

    _STATE[key] = score
    _STATE["device"] = device
    return score


def vram_peak_mb() -> Optional[float]:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except ImportError:
        return None
    return None
