"""Гибридный ретривер: косинус по эмбеддингам (numpy) + BM25-подобный лексический скор.

Корпус маленький (сотни чанков), поэтому numpy-индекса достаточно; эмбеддинги корпуса кэшируются
на диск (cache/copilot, в .gitignore) по отпечатку корпуса и имени эмбеддера.
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from .config import default_config, resolve_path
from .corpus import Chunk, Corpus
from .embed import make_embedder, stem, tokenize

log = logging.getLogger(__name__)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense: float = 0.0
    lexical: float = 0.0
    overlap: int = 0          # сколько НЕобщих слов вопроса есть во фрагменте (для порога «возможно относится»)


# слова, совпадение по которым ничего не говорит о релевантности нормы
_GENERIC = {"норма", "нормы", "пункт", "наруше", "примен", "исполь", "работ", "работе", "какая", "какой", "какие",
            "почему", "кассир", "правил", "статья", "докуме", "требов", "можно", "нужно", "должен", "случае",
            "являет", "основа", "отмече", "систем", "видео"}


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in docs]
        self.lens = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg = float(self.lens.mean()) if len(docs) else 1.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = max(len(docs), 1)
        self.idf = {t: math.log(1.0 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs), dtype=np.float32)
        q = set(query)
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in q:
                tf = d.get(t)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self.lens[i] / max(self.avg, 1e-6))
                s += self.idf.get(t, 0.0) * tf * (self.k1 + 1) / denom
            out[i] = s
        return out


def _terms(text: str) -> list[str]:
    return [stem(w) for w in tokenize(text) if len(w) > 2]


def content_overlap(query: str, text: str) -> int:
    q = {t for t in _terms(query) if len(t) > 3 and t not in _GENERIC}
    return len(q & set(_terms(text)))


class Retriever:
    def __init__(self, corpus: Corpus, cfg: Optional[dict[str, Any]] = None, embedder: Any = None) -> None:
        self.cfg = cfg or default_config()
        self.corpus = corpus
        self.warnings: list[str] = []
        self.embedder = embedder if embedder is not None else make_embedder(self.cfg, self.warnings)
        self.chunks = list(corpus.chunks)
        self.bm25 = BM25([_terms(c.embed_text) for c in self.chunks])
        self.matrix = self._passage_matrix()

    @property
    def backend(self) -> str:
        return str(getattr(self.embedder, "name", type(self.embedder).__name__))

    def _cache_path(self) -> Optional[Path]:
        cache_dir = resolve_path(self.cfg.get("embed", {}).get("cache_dir"))
        if not cache_dir:
            return None
        name = str(getattr(self.embedder, "model_name", self.backend)).replace("/", "_")
        return cache_dir / f"emb_{name}_{self.corpus.fingerprint()}.npy"

    def _passage_matrix(self) -> np.ndarray:
        if not self.chunks:
            return np.zeros((0, 1), dtype=np.float32)
        cache = self._cache_path() if self.backend != "hashing" else None
        if cache and cache.is_file():
            try:
                mat = np.load(cache)
                if mat.shape[0] == len(self.chunks):
                    return mat
            except Exception:
                pass
        mat = self.embedder.encode_passages([c.embed_text for c in self.chunks]).astype(np.float32)
        if cache:
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, mat)
            except OSError:
                pass
        return mat

    def relevant(self, hit: Hit) -> bool:
        """Достаточно ли фрагмент похож на вопрос, чтобы показать его как «найдено поиском»."""
        rcfg = self.cfg.get("rag", {})
        min_dense = float(rcfg.get("retrieved_min_dense_e5" if self.backend == "e5" else "retrieved_min_dense",
                                   0.82 if self.backend == "e5" else 0.25))
        return hit.overlap >= int(rcfg.get("retrieved_min_overlap", 2)) or hit.dense >= min_dense

    def min_score(self) -> float:
        rcfg = self.cfg.get("rag", {})
        if self.backend == "e5":
            return float(rcfg.get("min_score_e5", 0.78))
        return float(rcfg.get("min_score", 0.12))

    def search(self, query: str, top_k: Optional[int] = None,
               type_ids: Iterable[str] = (), dedupe: bool = True) -> list[Hit]:
        rcfg = self.cfg.get("rag", {})
        k = int(top_k or rcfg.get("top_k", 4))
        if not self.chunks or not (query or "").strip():
            return []
        qv = self.embedder.encode_queries([query])[0]
        dense = self.matrix @ qv
        lex = self.bm25.scores(_terms(query))
        lex_n = lex / lex.max() if lex.max() > 0 else lex
        w = float(rcfg.get("hybrid_lexical_weight", 0.35))
        boost = float(rcfg.get("boost_violation_types", 0.15))
        wanted = set(type_ids or ())
        scores = (1 - w) * dense + w * lex_n
        if wanted:
            scores = scores + np.array([boost if wanted & set(c.type_ids) else 0.0 for c in self.chunks],
                                       dtype=np.float32)
        order = np.argsort(-scores)
        hits: list[Hit] = []
        seen: set[tuple[str, str]] = set()
        threshold = self.min_score()
        for idx in order:
            c = self.chunks[int(idx)]
            if float(dense[idx]) < threshold and float(lex_n[idx]) < 0.5 and not (wanted & set(c.type_ids)):
                continue
            key = c.key if c.key[1] else (c.key[0], c.id)
            if dedupe and key in seen:
                continue
            seen.add(key)
            hits.append(Hit(chunk=c, score=float(scores[idx]), dense=float(dense[idx]), lexical=float(lex_n[idx]),
                            overlap=content_overlap(query, c.embed_text)))
            if len(hits) >= k:
                break
        return hits
