"""Кэш промежуточных результатов по sha256 видео: cache/<sha>/{dets.jsonl, tracks.json, ...}.

Запись стадии валидна только при совпадении ключа (считает оркестратор, см. stages.module_identity):
sha видео + sample_fps + идентичность ingest + версия/код/конфиги/промпты модуля + checkpoint/revision
и fold/split из model_info() + конфиг стадии + ключи входных стадий + хэш каталога норм + mock.
«Поменял код или веса — пересчитал» без ручного бампа версий; тюнинг rules/fusion на 40 видео идёт
без повторного прогона CV. manifest.json хранит ключи и идентичности модулей по стадиям.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from . import serde

log = logging.getLogger("sentinel.pipeline")

# stage -> [(файл, вид содержимого)]
STAGE_FILES: dict[str, list[tuple[str, str]]] = {
    "detect": [("dets.jsonl", "detection")],
    "track": [("tracks.json", "track"), ("zones.json", "zone")],
    "pose": [("poses.jsonl", "pose"), ("pose_states.jsonl", "state")],
    "scene": [("states.jsonl", "state")],
    "rules": [("events_rules.json", "event")],
    "seqrules": [("events_seqrules.json", "event"), ("observations.json", "obs")],
    "temporal": [("risk.json", "risk"), ("events_temporal.json", "event")],
    "vlm_verify": [("events_vlm_verify.json", "event")],
    "vlm_discover": [("events_vlm_discover.json", "event")],
}

_meta_locks: dict[str, threading.Lock] = {}
_meta_locks_guard = threading.Lock()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def stable_hash(*parts: Any) -> str:
    raw = json.dumps(serde.to_jsonable(list(parts)), sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _write_kind(path: Path, kind: str, value: Any) -> None:
    if path.suffix == ".jsonl":
        serde.write_jsonl(path, value)
    else:
        serde.write_json(path, value)


def _read_kind(path: Path, kind: str) -> Any:
    raw = serde.read_jsonl(path) if path.suffix == ".jsonl" else serde.read_json(path)
    if kind == "risk":
        return serde.load_risk(raw)
    loader = serde.LOADERS[kind]
    return [loader(x) for x in raw]


class StageCache:
    """Кэш одного видео. ``mode``: off | read_write | from_cache."""

    def __init__(self, root: Path, sha: str, mode: str = "read_write") -> None:
        self.sha = sha
        self.mode = mode if mode in ("off", "read_write", "from_cache") else "read_write"
        self.dir = Path(root) / sha
        self.hits: list[str] = []
        self.writes: list[str] = []

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def _lock(self) -> threading.Lock:
        with _meta_locks_guard:
            return _meta_locks.setdefault(str(self.dir), threading.Lock())

    def manifest(self) -> dict:
        try:
            return serde.read_json(self.dir / "manifest.json")
        except (OSError, ValueError):
            return {}

    def load(self, stage: str, key: str, *, allow_stale: bool = False) -> Optional[tuple]:
        """Возвращает кортеж выходов стадии (в порядке STAGE_FILES) или None."""
        if not self.enabled or stage not in STAGE_FILES:
            return None
        entry = (self.manifest().get("stages") or {}).get(stage)
        if not entry:
            return None
        if entry.get("key") != key and not allow_stale:
            return None
        try:
            values = tuple(_read_kind(self.dir / fname, kind) for fname, kind in STAGE_FILES[stage])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.warning("кэш %s/%s повреждён: %s", self.sha[:12], stage, exc)
            return None
        self.hits.append(stage if entry.get("key") == key else f"{stage}(stale)")
        return values

    def save(self, stage: str, key: str, values: tuple, info: Optional[dict] = None) -> None:
        if not self.enabled or stage not in STAGE_FILES:
            return
        try:
            for (fname, kind), value in zip(STAGE_FILES[stage], values):
                _write_kind(self.dir / fname, kind, value)
            with self._lock():
                man = self.manifest()
                man.setdefault("sha256", self.sha)
                man.setdefault("stages", {})[stage] = {
                    "key": key, "files": [f for f, _ in STAGE_FILES[stage]],
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S"), **(info or {})}
                serde.write_json(self.dir / "manifest.json", man, indent=1)
            self.writes.append(stage)
        except OSError as exc:
            log.warning("не удалось записать кэш %s: %s", stage, exc)

    def save_video_info(self, info: dict) -> None:
        if not self.enabled:
            return
        try:
            with self._lock():
                man = self.manifest()
                man["sha256"] = self.sha
                man["video"] = info
                serde.write_json(self.dir / "manifest.json", man, indent=1)
        except OSError as exc:
            log.warning("не удалось записать manifest: %s", exc)


def load_or_compute(cache: Optional[StageCache], stage: str, key: str, compute: Callable[[], tuple],
                    *, allow_stale: bool = False) -> tuple[tuple, bool]:
    """Утилита для внешних скриптов: (значения, из_кэша)."""
    if cache is not None:
        hit = cache.load(stage, key, allow_stale=allow_stale)
        if hit is not None:
            return hit, True
    values = compute()
    if cache is not None:
        cache.save(stage, key, values)
    return values, False
