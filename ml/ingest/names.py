"""Имена файлов и дубликаты: ``Х-3`` (кириллица) == ``X-3`` (латиница), md5 для поиска копий.

    normalize_video_id("Х-3.MOV") -> "X-3"
    find_duplicates(paths) -> {md5: [path, ...]}   # только группы из 2+ файлов
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Iterable

# кириллические буквы, визуально совпадающие с латинскими
_HOMOGLYPHS = str.maketrans({
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "У": "Y", "Х": "X", "а": "a", "е": "e", "к": "k", "м": "m", "о": "o", "р": "p", "с": "c", "т": "t",
    "у": "y", "х": "x", "Ё": "E", "ё": "e",
})
_VIDEO_EXT = re.compile(r"\.(mov|mp4|m4v|avi|mkv|qt)$", re.IGNORECASE)


def normalize_video_id(name: str) -> str:
    """Каноничный ID ролика: без пути и расширения, гомоглифы -> латиница, верхний регистр,
    разделители (пробел, _, –, —) -> '-'."""
    stem = _VIDEO_EXT.sub("", Path(str(name)).name)
    stem = unicodedata.normalize("NFKC", stem).translate(_HOMOGLYPHS)
    stem = re.sub(r"[\s_‐-―]+", "-", stem.strip())
    stem = re.sub(r"-{2,}", "-", stem)
    return stem.upper()


def file_md5(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def find_duplicates(paths: Iterable[str]) -> dict[str, list[str]]:
    """Группы файлов с одинаковым содержимым (сначала по размеру, затем md5)."""
    by_size: dict[int, list[str]] = {}
    for p in paths:
        by_size.setdefault(Path(p).stat().st_size, []).append(str(p))
    groups: dict[str, list[str]] = {}
    for same in by_size.values():
        if len(same) < 2:
            continue
        for p in same:
            groups.setdefault(file_md5(p), []).append(p)
    return {k: sorted(v) for k, v in groups.items() if len(v) > 1}
