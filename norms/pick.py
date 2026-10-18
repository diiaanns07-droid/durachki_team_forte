"""Поиск пунктов-кандидатов в ПОЛНЫХ текстах НПА по ключевым словам и сверка цитат каталога.

    python -m norms.pick --git-ref origin/data/norms-sources        # тексты L1 прямо из ветки (без checkout)
    python -m norms.pick --dir norms_fetched --type V09 --top 5      # из папки
    python -m norms.pick --check-catalog                             # где в полном тексте стоят цитаты каталога
    python -m norms.pick --out norms/PICK_CANDIDATES.md --emit-yaml  # отчёт для человека + YAML-заготовки

Это поиск КАНДИДАТОВ: человек читает пункт, решает, применим ли он к поведению из спецификации,
переносит норму в catalog.yaml (verified: false), сверяет цитату и только потом ставит
verified: true и флаги *_checked_by_human в rule_checks. Ключевые слова — norms/pick_keywords.yaml.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import CATALOG_PATH, NORMS_DIR, load_catalog
from .texts import Clause, DocText, load_dir, load_fulltext_sources, load_git_ref, segment
from .validate import normalize

KEYWORDS_PATH = NORMS_DIR / "pick_keywords.yaml"
_SENTENCE_SPLIT = re.compile(r"(?<=[.;])\s+")


@dataclass
class Candidate:
    type_id: str
    doc_id: str
    url: str
    clause: str
    line: int
    heading: str
    score: int
    matched: list[str]
    quote: str
    text: str = field(repr=False)


@dataclass
class QuoteLocation:
    type_id: str
    index: int
    catalog_clause: Optional[str]
    found_in: list[str]          # ["nbrk_231_2019 п. 12 (стр. 140)", ...]
    status: str                  # FOUND | FOUND_OTHER_CLAUSE | NOT_FOUND | NO_FULLTEXT


def load_keywords(path: Path = KEYWORDS_PATH) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def score_clause(text_norm: str, kw: dict[str, Any], modal: list[str]) -> tuple[int, list[str]]:
    any_hits = [w for w in kw.get("any", []) if normalize(w) in text_norm]
    if not any_hits:
        return 0, []
    boost_hits = [w for w in kw.get("boost", []) if normalize(w) in text_norm]
    modal_hits = [w for w in modal if normalize(w) in text_norm]
    return 2 * len(any_hits) + len(boost_hits) + len(modal_hits), any_hits + boost_hits + modal_hits


def best_sentence(text: str, matched: list[str]) -> str:
    """Дословное предложение пункта с первым совпадением (для поля quote)."""
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]
    for stem in matched:
        for p in parts:
            if normalize(stem) in normalize(p):
                return p
    return parts[0] if parts else text.strip()


def pick(docs: list[DocText], keywords: dict[str, Any], types: Optional[list[str]] = None,
         top: int = 5, min_score: int = 3) -> dict[str, list[Candidate]]:
    clauses = [(d, c) for d in docs for c in segment(d)]
    normed = [normalize(c.text) for _, c in clauses]
    out: dict[str, list[Candidate]] = {}
    for tid, kw in (keywords.get("types") or {}).items():
        if types and tid not in types:
            continue
        found: list[Candidate] = []
        for (doc, c), tn in zip(clauses, normed):
            score, matched = score_clause(tn, kw, keywords.get("modal") or [])
            if score >= min_score:
                found.append(Candidate(tid, c.doc_id, doc.url, c.ref, c.line, c.heading, score, matched,
                                       best_sentence(c.text, matched), c.text))
        found.sort(key=lambda x: (-x.score, x.doc_id, x.line))
        out[tid] = found[:top]
    return out


def check_catalog(catalog: dict[str, Any], docs: list[DocText]) -> list[QuoteLocation]:
    """Для каждой цитаты каталога: в каком пункте полного текста она стоит (и совпадает ли номер)."""
    by_doc: dict[str, list[Clause]] = {}
    for d in docs:
        by_doc.setdefault(d.url.rstrip("/"), []).extend(segment(d))
    res: list[QuoteLocation] = []
    for tid, t in (catalog.get("types") or {}).items():
        for i, r in enumerate(t.get("rules") or []):
            quote = normalize(str(r.get("quote") or ""))
            clauses = by_doc.get(str(r.get("url", "")).rstrip("/"))
            if not quote:
                continue
            if clauses is None:
                res.append(QuoteLocation(tid, i, r.get("clause"), [], "NO_FULLTEXT"))
                continue
            hits = [c for c in clauses if quote in normalize(c.text)]
            where = [f"{c.doc_id} {c.ref} (стр. {c.line})" for c in hits]
            if not hits:
                status = "NOT_FOUND"
            elif r.get("clause") and any(c.ref == r.get("clause") for c in hits):
                status = "FOUND"
            else:
                status = "FOUND_OTHER_CLAUSE"
            res.append(QuoteLocation(tid, i, r.get("clause"), where, status))
    return res


def emit_yaml(c: Candidate, document: str) -> str:
    q = c.quote.replace('"', '\\"')
    clause = f'"{c.clause}"' if c.clause != "—" else "null"
    return (f"# {c.type_id}: кандидат из {c.doc_id}, {c.clause} (стр. {c.line}), score={c.score}; "
            f"совпадения: {', '.join(c.matched)}\n"
            f"- document: \"{document}\"\n  clause: {clause}\n  url: \"{c.url}\"\n"
            f"  quote: \"{q}\"\n  verified: false\n")


def render_markdown(results: dict[str, list[Candidate]], checks: list[QuoteLocation],
                    sources: list[str], emit: bool, titles: dict[str, str]) -> str:
    lines = ["# Кандидаты норм (python -m norms.pick)", "",
             "Поиск по ключевым словам — это подсказка человеку, не вывод о применимости.", "",
             "Тексты: " + (", ".join(sources) if sources else "—"), ""]
    if checks:
        lines += ["## Цитаты каталога в полном тексте", "", "| тип | # | пункт в каталоге | статус | где найдено |",
                  "|---|---|---|---|---|"]
        for q in checks:
            lines.append(f"| {q.type_id} | {q.index} | {q.catalog_clause or '—'} | {q.status} | "
                         f"{'; '.join(q.found_in) or '—'} |")
        lines.append("")
    for tid, cands in results.items():
        lines += [f"## {tid}", ""]
        if not cands:
            lines += ["Кандидатов не найдено (тип остаётся без пункта и без правового вывода).", ""]
            continue
        for c in cands:
            excerpt = c.text if len(c.text) <= 500 else c.text[:500] + "…"
            lines += [f"- **{c.doc_id} {c.clause}** (стр. {c.line}; {c.heading or 'без заголовка'}), score={c.score}, "
                      f"совпадения: {', '.join(c.matched)}", f"  > {excerpt}", ""]
            if emit:
                lines += ["```yaml", emit_yaml(c, titles.get(c.doc_id, c.doc_id)).rstrip(), "```", ""]
    return "\n".join(lines).rstrip() + "\n"


def collect_docs(dirs: list[Path], git_ref: Optional[str], git_subdir: str, use_sources: bool,
                 catalog: dict[str, Any]) -> tuple[list[DocText], list[str]]:
    docs: list[DocText] = []
    labels: list[str] = []
    for d in dirs:
        if d.is_dir():
            got = load_dir(d, catalog)
            docs += got
            labels.append(f"{d} ({len(got)})")
        else:
            print(f"WARN: нет папки {d}", file=sys.stderr)
    if git_ref:
        got = load_git_ref(git_ref, git_subdir, catalog)
        docs += got
        labels.append(f"{git_ref}:{git_subdir} ({len(got)})")
    if use_sources:
        got = load_fulltext_sources()
        docs += got
        if got:
            labels.append(f"norms/sources fulltext ({len(got)})")
    return docs, labels


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m norms.pick", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, action="append", default=[], help="папка с текстами (можно несколько)")
    ap.add_argument("--git-ref", default=None, help="например origin/data/norms-sources")
    ap.add_argument("--git-subdir", default="norms_fetched")
    ap.add_argument("--no-sources", action="store_true", help="не брать norms/sources/*.md (fulltext)")
    ap.add_argument("--type", action="append", default=[], help="V01..V99 (можно несколько)")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--min-score", type=int, default=3)
    ap.add_argument("--check-catalog", action="store_true", help="найти цитаты каталога в полном тексте")
    ap.add_argument("--emit-yaml", action="store_true", help="YAML-заготовки норм (verified: false)")
    ap.add_argument("--out", type=Path, default=None, help="записать отчёт в markdown")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    ap.add_argument("--keywords", type=Path, default=KEYWORDS_PATH)
    args = ap.parse_args(argv)

    catalog = load_catalog(args.catalog, use_cache=False)
    dirs = list(args.dir)
    default_dir = NORMS_DIR.parent / "norms_fetched"
    if not dirs and not args.git_ref and default_dir.is_dir():
        dirs.append(default_dir)
    try:
        docs, labels = collect_docs(dirs, args.git_ref, args.git_subdir, not args.no_sources, catalog)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not docs:
        print("Нет полных текстов. Варианты: python -m norms.fetch (интернет), "
              "--git-ref origin/data/norms-sources (тексты L1), --dir norms_fetched.", file=sys.stderr)
        return 2
    results = pick(docs, load_keywords(args.keywords), args.type or None, args.top, args.min_score)
    checks = check_catalog(catalog, docs) if args.check_catalog else []
    titles = {k: str(v.get("title", k)) for k, v in (catalog.get("documents") or {}).items()}
    if args.json:
        print(json.dumps({"sources": labels,
                          "candidates": {k: [{**asdict(c), "text": c.text[:500]} for c in v] for k, v in results.items()},
                          "catalog_quotes": [asdict(q) for q in checks]}, ensure_ascii=False, indent=2))
    else:
        md = render_markdown(results, checks, labels, args.emit_yaml, titles)
        if args.out:
            args.out.write_text(md, encoding="utf-8")
            print(f"Отчёт: {args.out}")
        else:
            print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
