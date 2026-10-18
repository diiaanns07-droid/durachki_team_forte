"""Скачать ПОЛНЫЕ тексты источников с adilet.zan.kz в norms/sources/*.md (для сверки цитат).

Запускать на машине с интернетом (в песочнице агента A13 adilet был недоступен):

    python -m norms.fetch                 # все документы из catalog.yaml → sources/*.md
    python -m norms.fetch nbrk_14_2020    # один документ по id из раздела documents
    python -m norms.fetch --dry-run       # только показать, что будет скачано
    python -m norms.fetch --from-dir norms_fetched            # импорт файлов, скачанных L1
    python -m norms.fetch --git-ref origin/data/norms-sources # импорт прямо из ветки L1 (без checkout)

После скачивания файл получает ``retrieval: fulltext``, а прежние поисковые фрагменты
остаются в git-истории. Затем ``python -m norms.validate`` покажет цитаты, которых нет
в полном тексте (их надо исправить), и только после этого можно ставить ``verified: true``.
Только стандартная библиотека; рантайм-пайплайн этот модуль не использует.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import re
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

from . import CATALOG_PATH, NORMS_DIR

_BLOCK_TAGS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table", "article", "section"}
_SKIP_TAGS = {"script", "style", "noscript", "head", "nav", "footer", "header", "form", "button", "svg"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:  # type: ignore[override]
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def html_to_text(raw_html: str) -> str:
    p = _TextExtractor()
    p.feed(raw_html)
    text = html.unescape("".join(p.parts))
    lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in text.splitlines()]
    out: list[str] = []
    for ln in lines:
        if ln or (out and out[-1]):
            out.append(ln)
    return "\n".join(out).strip() + "\n"


def fetch_url(url: str, timeout: float = 60.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (KassaSentinel norms fetch)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — только https adilet
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace")


def render_source_md(doc_id: str, doc: dict, text: str, accessed: Optional[str] = None) -> str:
    accessed = accessed or dt.date.today().isoformat()
    return (
        "---\n"
        f"id: {doc_id}\n"
        f"title: \"{doc.get('title', '')}\"\n"
        f"act: \"{doc.get('act', '')}\"\n"
        f"url: {doc['url']}\n"
        f"accessed: {accessed}\n"
        "retrieval: fulltext\n"
        "---\n\n"
        f"# {doc.get('title', doc_id)}\n\n"
        f"Источник: {doc['url']} (полный текст страницы, скачан `python -m norms.fetch` {accessed}).\n\n"
        f"{text}"
    )


def import_local(catalog: dict, ids: list[str], args: argparse.Namespace) -> int:
    """Полные тексты из папки/ветки L1 → sources/<doc>.md (retrieval: fulltext) для известных документов."""
    from . import texts

    try:
        loaded = (texts.load_dir(args.from_dir, catalog) if args.from_dir
                  else texts.load_git_ref(args.git_ref, args.git_subdir, catalog))
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    docs: dict = catalog.get("documents", {})
    written = 0
    for d in loaded:
        if d.doc_id not in docs or d.doc_id not in ids:
            print(f"SKIP {d.label}: документ {d.doc_id!r} не из раздела documents (для pick годится и так)")
            continue
        target = args.out_dir / docs[d.doc_id]["source_file"]
        print(f"{d.doc_id}: {d.label} -> {target}")
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(render_source_md(d.doc_id, docs[d.doc_id], d.text), encoding="utf-8")
            written += 1
    if not args.dry_run:
        print(f"Импортировано: {written}. Дальше: python -m norms.pick --check-catalog и python -m norms.validate")
    return 0 if written or args.dry_run else 1


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m norms.fetch", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*", help="id документов из catalog.yaml: documents (по умолчанию все)")
    ap.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--from-dir", type=Path, default=None, help="импорт локальных .html/.md/.txt вместо скачивания")
    ap.add_argument("--git-ref", default=None, help="импорт из git-ветки, например origin/data/norms-sources")
    ap.add_argument("--git-subdir", default="norms_fetched")
    ap.add_argument("--out-dir", type=Path, default=NORMS_DIR, help="куда писать sources/ (по умолчанию norms/)")
    args = ap.parse_args(argv)

    import yaml

    catalog = yaml.safe_load(args.catalog.read_text(encoding="utf-8"))
    docs: dict = catalog.get("documents", {})
    ids = args.ids or list(docs)
    if args.from_dir or args.git_ref:
        return import_local(catalog, ids, args)
    rc = 0
    for doc_id in ids:
        doc = docs.get(doc_id)
        if not doc:
            print(f"SKIP {doc_id}: нет в documents", file=sys.stderr)
            rc = 1
            continue
        target = NORMS_DIR / doc["source_file"]
        print(f"{doc_id}: {doc['url']} -> {target.relative_to(NORMS_DIR)}")
        if args.dry_run:
            continue
        try:
            text = html_to_text(fetch_url(doc["url"]))
        except Exception as exc:  # сеть/HTTP — продолжаем остальные документы
            print(f"FAIL {doc_id}: {exc}", file=sys.stderr)
            rc = 1
            continue
        target.write_text(render_source_md(doc_id, doc, text), encoding="utf-8")
        print(f"  OK, {len(text)} символов")
    if not args.dry_run:
        print("Дальше: python -m norms.validate — исправьте NOT_FOUND и только потом ставьте verified: true")
    return rc


if __name__ == "__main__":
    sys.exit(main())
