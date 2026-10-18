"""Эмбеддеры для RAG.

* ``E5Embedder`` — intfloat/multilingual-e5-small (MIT), локальные веса, CPU по умолчанию.
* ``HashingEmbedder`` — без весов: хэширование символьных n-грамм (устойчиво к русской морфологии).
  Используется в мок-режиме и как graceful fallback, если весов e5 нет.
"""
from __future__ import annotations

import logging
import math
import re
import zlib
from typing import Any, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[0-9a-zа-яё]+", re.I)


def tokenize(text: str) -> list[str]:
    return [w.lower().replace("ё", "е") for w in _WORD_RE.findall(text or "")]


def stem(word: str) -> str:
    """Грубый стеммер для русского: обрезка до 6 символов (достаточно для лексического скора)."""
    return word[:6] if len(word) > 6 else word


class HashingEmbedder:
    name = "hashing"

    def __init__(self, dim: int = 4096, ngram: tuple[int, int] = (3, 5)) -> None:
        self.dim = int(dim)
        self.ngram = ngram

    def _features(self, text: str) -> dict[int, float]:
        feats: dict[int, float] = {}
        for w in tokenize(text):
            grams = [f"w:{stem(w)}"]
            padded = f"<{w}>"
            lo, hi = self.ngram
            for n in range(lo, hi + 1):
                grams.extend(padded[i:i + n] for i in range(max(0, len(padded) - n + 1)))
            for g in grams:
                h = zlib.crc32(g.encode("utf-8"))
                idx = h % self.dim
                sign = 1.0 if (h >> 31) & 1 == 0 else -1.0
                feats[idx] = feats.get(idx, 0.0) + sign * (2.0 if g.startswith("w:") else 1.0)
        return feats

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for idx, val in self._features(t).items():
                out[i, idx] = math.copysign(math.log1p(abs(val)), val)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)


class E5Embedder:
    """multilingual-e5: префиксы «query: » / «passage: », mean pooling, L2-нормировка."""

    name = "e5"

    def __init__(self, model: str = "intfloat/multilingual-e5-small", device: str = "cpu",
                 batch_size: int = 32, hf_cache_dir: Optional[str] = None) -> None:
        self.model_name = model
        self.device = device
        self.batch_size = batch_size
        self.hf_cache_dir = hf_cache_dir
        self._st: Any = None
        self._tok: Any = None
        self._model: Any = None
        self._load()

    def _local_path(self) -> str:
        """Только локальные веса: папка на диске или уже скачанный снапшот из кэша HF.

        Никогда не скачивает: snapshot_download(local_files_only=True) бросает исключение,
        если модели нет в кэше (тогда make_embedder в режиме auto переходит на hashing).
        """
        from pathlib import Path

        from .config import resolve_path

        for cand in (Path(self.model_name), resolve_path(self.model_name)):
            if cand is not None and cand.is_dir():
                return str(cand)
        from huggingface_hub import snapshot_download  # type: ignore

        kwargs: dict[str, Any] = {"repo_id": self.model_name, "local_files_only": True}
        cache = resolve_path(self.hf_cache_dir) if self.hf_cache_dir else None
        if cache is not None and cache.is_dir():  # офлайн-веса L1: weights/hf/hub
            kwargs["cache_dir"] = str(cache)
        return str(snapshot_download(**kwargs))

    def _load(self) -> None:
        path = self._local_path()
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError:
            SentenceTransformer = None  # noqa: N806
        if SentenceTransformer is not None:
            self._st = SentenceTransformer(path, device=self.device)
            return
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        self._tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self._model = AutoModel.from_pretrained(path, local_files_only=True).to(self.device).eval()

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._st is not None:
            vecs = self._st.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True,
                                   show_progress_bar=False, convert_to_numpy=True)
            return np.asarray(vecs, dtype=np.float32)
        import torch  # type: ignore

        out: list[np.ndarray] = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                batch = list(texts[i:i + self.batch_size])
                enc = self._tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                hidden = self._model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                out.append(pooled.float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 384), dtype=np.float32)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([f"query: {t}" for t in texts])

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([f"passage: {t}" for t in texts])


def make_embedder(cfg: dict[str, Any], warnings: Optional[list[str]] = None) -> Any:
    """auto: e5 при наличии локальных весов, иначе hashing (с предупреждением)."""
    from .config import is_mock

    ecfg = cfg.get("embed", {})
    backend = str(ecfg.get("backend", "auto")).lower()
    if backend == "hashing" or (backend == "auto" and is_mock()):
        return HashingEmbedder(dim=int(ecfg.get("hashing_dim", 4096)))
    try:
        return E5Embedder(model=str(ecfg.get("model", "intfloat/multilingual-e5-small")),
                          device=str(ecfg.get("device", "cpu")),
                          batch_size=int(ecfg.get("batch_size", 32)),
                          hf_cache_dir=ecfg.get("hf_cache_dir"))
    except Exception as exc:
        if backend == "e5":
            raise
        msg = f"e5 недоступен ({type(exc).__name__}) — RAG на hashing-эмбеддингах"
        log.warning(msg)
        if warnings is not None:
            warnings.append(msg)
        return HashingEmbedder(dim=int(ecfg.get("hashing_dim", 4096)))
