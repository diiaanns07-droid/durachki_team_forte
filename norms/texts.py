"""Загрузка скачанных полных текстов НПА и разбиение на пункты (для norms.pick и norms.fetch).

Источники текстов (в порядке поиска по умолчанию):
  1. каталоги ``--dir`` (например, ``norms_fetched/`` — его кладёт L1 в ветку ``data/norms-sources``);
  2. git-ссылка ``--git-ref origin/data/norms-sources`` — файлы читаются через ``git show`` без checkout;
  3. ``norms/sources/*.md`` с ``retrieval: fulltext``.
Поддерживаются .html/.htm (разбор stdlib), .md/.txt. Документ определяется по коду adilet
(например ``V1900019680``) в имени файла или тексте и сопоставляется с ``documents`` каталога.
Только стандартная библиотека; в рантайме пайплайна не используется.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from . import NORMS_DIR

TEXT_SUFFIXES = (".html", ".htm", ".md", ".txt")
_ADILET_CODE_RE = re.compile(r"(?<![A-Za-z0-9])([VZPKU]\d{9,10}_?)(?!\d)", re.I)
_HEADING_RE = re.compile(r"^\s*((?:Глава|Параграф|Раздел)\s+\d+[\-\d]*\.?.*)$", re.I)
_CLAUSE_RE = re.compile(r"^\s*(\d{1,3}(?:-\d{1,2})?)\.\s+(\S.*)$")


@dataclass
class DocText:
    doc_id: str
    url: str
    label: str            # путь или ref:path — для отчёта
    text: str
    meta: dict[str, str] = field(default_factory=dict)


@dataclass
class Clause:
    doc_id: str
    number: Optional[str]  # "12" | "12-1" | None (текст до первого пункта)
    heading: str
    line: int
    text: str

    @property
    def ref(self) -> str:
        return f"п. {self.number}" if self.number else "—"


def adilet_code(s: str) -> Optional[str]:
    m = _ADILET_CODE_RE.search(s)
    return m.group(1).upper() if m else None


def _doc_index(catalog: Optional[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    """adilet-код → (doc_id, url) из раздела documents каталога."""
    out: dict[str, tuple[str, str]] = {}
    for doc_id, doc in ((catalog or {}).get("documents") or {}).items():
        code = adilet_code(str(doc.get("url", "")))
        if code:
            out[code.rstrip("_")] = (str(doc_id), str(doc["url"]))
    return out


def _to_text(name: str, raw: str) -> tuple[str, dict[str, str]]:
    from .validate import parse_front_matter

    if name.lower().endswith((".html", ".htm")):
        from .fetch import html_to_text

        return html_to_text(raw), {}
    meta, body = parse_front_matter(raw)
    return body, meta


def _identify(name: str, text: str, index: dict[str, tuple[str, str]]) -> tuple[str, str]:
    for probe in (name, text[:5000]):
        code = adilet_code(probe)
        if code and code.rstrip("_") in index:
            return index[code.rstrip("_")]
    code = adilet_code(name) or adilet_code(text[:5000])
    if code:
        return code, f"https://adilet.zan.kz/rus/docs/{code}"
    return Path(name).stem, ""


def load_dir(path: Path, catalog: Optional[dict[str, Any]] = None) -> list[DocText]:
    index = _doc_index(catalog)
    docs: list[DocText] = []
    for f in sorted(p for p in Path(path).rglob("*") if p.suffix.lower() in TEXT_SUFFIXES and p.is_file()):
        text, meta = _to_text(f.name, f.read_text(encoding="utf-8", errors="replace"))
        doc_id, url = _identify(f.name, text, index)
        docs.append(DocText(doc_id, meta.get("url") or url, str(f), text, meta))
    return docs


def load_git_ref(ref: str, subdir: str = "norms_fetched", catalog: Optional[dict[str, Any]] = None,
                 repo: Optional[Path] = None) -> list[DocText]:
    """Читает файлы ``<ref>:<subdir>/**`` через git (ветку не нужно checkout'ить)."""
    cwd = str(repo or NORMS_DIR.parent)
    ls = subprocess.run(["git", "ls-tree", "-r", "--name-only", ref, "--", subdir],
                        cwd=cwd, capture_output=True, text=True)
    if ls.returncode != 0:
        raise FileNotFoundError(f"git ref {ref!r} недоступен: {ls.stderr.strip()}")
    index = _doc_index(catalog)
    docs: list[DocText] = []
    for name in ls.stdout.splitlines():
        if not name.lower().endswith(TEXT_SUFFIXES):
            continue
        raw = subprocess.run(["git", "show", f"{ref}:{name}"], cwd=cwd, capture_output=True,
                             text=True, check=True).stdout
        text, meta = _to_text(name, raw)
        doc_id, url = _identify(name, text, index)
        docs.append(DocText(doc_id, meta.get("url") or url, f"{ref}:{name}", text, meta))
    return docs


def load_fulltext_sources(sources_dir: Path = NORMS_DIR / "sources") -> list[DocText]:
    from .validate import parse_front_matter

    docs: list[DocText] = []
    for f in sorted(Path(sources_dir).glob("*.md")):
        meta, body = parse_front_matter(f.read_text(encoding="utf-8"))
        if meta.get("retrieval") == "fulltext":
            docs.append(DocText(meta.get("id") or f.stem, meta.get("url", ""), str(f), body, meta))
    return docs


def segment(doc: DocText) -> list[Clause]:
    """Разбивает текст на пункты «N. …» с учётом заголовков «Глава/Параграф». Подпункты «N)» остаются внутри."""
    clauses: list[Clause] = []
    heading, number, start, buf = "", None, 1, []

    def flush() -> None:
        body = " ".join(x.strip() for x in buf if x.strip())
        if body:
            clauses.append(Clause(doc.doc_id, number, heading, start, body))

    for i, line in enumerate(doc.text.splitlines(), start=1):
        h = _HEADING_RE.match(line)
        if h:
            flush()
            heading, number, start, buf = h.group(1).strip(), None, i + 1, []
            continue
        m = _CLAUSE_RE.match(line)
        if m:
            flush()
            number, start, buf = m.group(1), i, [m.group(2)]
            continue
        buf.append(line)
    flush()
    return clauses


def iter_clauses(docs: Iterable[DocText]) -> Iterable[Clause]:
    for d in docs:
        yield from segment(d)
