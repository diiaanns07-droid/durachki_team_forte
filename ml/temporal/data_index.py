"""Индекс извлечённых признаков (cache/features/index.json) -> обучающие Sample (дубликаты свёрнуты в группы)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from .config import cache_dir
from .dataset import Sample
from .features import FeatureSet, TextBank, get_backend, get_text_bank, text_bank_from_lists
from .labels import VideoLabel, merge_duplicates

log = logging.getLogger("sentinel.temporal")


def index_path(cfg: dict, path: Optional[str] = None) -> Path:
    return Path(path) if path else cache_dir(cfg) / "index.json"


def load_index(cfg: dict, path: Optional[str] = None) -> dict:
    p = index_path(cfg, path)
    if not p.exists():
        raise FileNotFoundError(f"нет индекса признаков {p}: сначала python -m ml.temporal.extract")
    return json.loads(p.read_text(encoding="utf-8"))


def load_samples(cfg: dict, path: Optional[str] = None, max_aug: Optional[int] = None) -> tuple[list[Sample], dict]:
    """Sample на группу дубликатов; признаки — первого файла группы (контент идентичен), метки — все исходные."""
    idx = load_index(cfg, path)
    base = index_path(cfg, path).parent
    entries = {v["name"]: v for v in idx["videos"]}
    labs = {n: VideoLabel(name=n, label=int(v["label"]), times=list(v.get("times", [])),
                          type_id=v.get("type_id", ""), files=[v.get("file", n)], group=v.get("group", n),
                          md5=v.get("md5", ""), pseudo_type_id=v.get("pseudo_type_id", ""))
            for n, v in entries.items()}
    merged = merge_duplicates(labs, cfg["labels"].get("conflict_policy", "positive"),
                              cfg["labels"].get("time_conflict_policy", "both"))
    samples: list[Sample] = []
    for g, lab in merged.items():
        src = next((entries[s["name"]] for s in lab.sources if entries[s["name"]].get("features")), None)
        if src is None:
            log.warning("temporal: нет признаков для группы %s", g)
            continue
        feats = src["features"][: (1 + max_aug) if max_aug is not None else None]
        versions = []
        for fp in feats:
            p = Path(fp)
            versions.append(FeatureSet.load(p if p.is_absolute() else base / p))
        if lab.conflict:
            log.warning("temporal: конфликт меток в группе %s: %s", g,
                        [(s["name"], s["label"], s["times"]) for s in lab.sources])
        samples.append(Sample(name=g, group=g, label=lab.label, times=lab.times, type_id=lab.type_id,
                              md5=lab.md5, sources=lab.sources, conflict=lab.conflict,
                              time_conflict=lab.time_conflict, pseudo_type_id=lab.pseudo_type_id,
                              versions=versions))
    return samples, idx


def bank_for_index(cfg: dict, idx: dict) -> TextBank:
    backend = get_backend(cfg, force=idx.get("backend"))
    return get_text_bank(cfg, backend)


def bank_from_meta(cfg: dict, meta: dict, backend: Any) -> TextBank:
    p = meta.get("prompts") or {}
    if p.get("en"):
        return text_bank_from_lists(cfg, backend, p["en"], p.get("ru", p["en"]), p["owner"])
    return get_text_bank(cfg, backend)
