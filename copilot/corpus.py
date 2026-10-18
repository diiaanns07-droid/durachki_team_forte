"""Корпус норм для RAG: norms/catalog.yaml (A13) + norms/sources/*.md -> чанки с источником.

Каждый чанк несёт документ, пункт и URL — это и есть единица цитирования. Ссылки на нормы
берутся ТОЛЬКО из каталога/корпуса (правило 5); фикстура copilot/fixtures — лишь для тестов.
"""
from __future__ import annotations

import glob
import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .config import default_config, fixture_allowed, resolve_path

log = logging.getLogger(__name__)

_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_URL_RE = re.compile(r"https?://[^\s)\]>»\"']+")
_ADILET_RE = re.compile(r"https?://adilet\.zan\.kz[^\s)\]>»\"']*")
# «п. 12», «пункт 12.3», «пп. 2)», «подпункт 2», «статья 5», «ст. 5», «глава 3»
_CLAUSE_RE = re.compile(
    r"(?<![\wА-Яа-яЁё])(?P<kind>подпункт(?:а|е|ом|у)?|пп\.|пункт(?:а|е|ом|у|ы|ов)?|п\.|стать(?:я|и|е|ю|ей)|ст\.)"
    r"\s*(?P<num>\d+(?:[.\-]\d+)*\)?)",
    re.I,
)
_BARE_NUM_RE = re.compile(r"^\s*(\d{1,3}(?:\.\d{1,3})*)\.\s+\S")


def norm_text(s: Optional[str]) -> str:
    s = (s or "").lower().replace("ё", "е")
    s = re.sub(r"[«»\"'“”„`]", "", s)
    return re.sub(r"\s+", " ", s).strip(" .,;:")


def norm_clause(clause: Optional[str]) -> str:
    """Канонический ключ пункта: «пункт 3», «п.3», «п. 3» -> «п.3»; «статья 5» -> «ст.5»."""
    if not clause:
        return ""
    s = norm_text(clause)
    m = _CLAUSE_RE.search(s)
    if m:
        kind = m.group("kind").lower()
        num = m.group("num").rstrip(")").replace("-", ".")
        if kind.startswith(("подпункт", "пп")):
            prefix = "пп."
        elif kind.startswith(("стат", "ст.")):
            prefix = "ст."
        else:
            prefix = "п."
        return prefix + num
    m2 = re.match(r"^\s*(\d+(?:\.\d+)*)", s)
    if m2:
        return "п." + m2.group(1)
    return s.replace(" ", "")


def norm_url(url: Optional[str]) -> str:
    u = (url or "").strip().lower()
    u = re.sub(r"^https?://", "", u)
    u = u.split("#", 1)[0].rstrip("/")
    return u


@dataclass
class Chunk:
    id: str
    text: str                  # текст для показа и цитаты
    document: str
    clause: Optional[str]
    url: str
    kind: str                  # catalog_rule | source
    type_ids: list[str] = field(default_factory=list)
    source_file: str = ""
    heading: str = ""          # для источника — метка фрагмента («Ф3. Пункт 20 (приём наличных)»)
    verified: bool = False     # цитата сверена с полным текстом (A13: verified / retrieval=full_text)
    status: str = ""           # legal_status типа из каталога: confirmed | candidate | disabled | ""
    note: str = ""             # оговорки: «номер пункта не сверен», «фрагмент поисковой выдачи» …

    @property
    def embed_text(self) -> str:
        head = " · ".join(x for x in (self.document, self.heading, self.clause or "") if x)
        return f"{head}. {self.text}" if head else self.text

    @property
    def key(self) -> tuple[str, str]:
        return norm_text(self.document), norm_clause(self.clause)

    def citation(self) -> dict[str, Any]:
        out: dict[str, Any] = {"document": self.document, "clause": self.clause, "url": self.url}
        if self.kind == "source" and self.heading:
            out["fragment"] = self.heading
        return out


@dataclass
class Corpus:
    chunks: list[Chunk]
    catalog: dict[str, Any]
    is_fixture: bool = False
    warnings: list[str] = field(default_factory=list)

    # --- каталог ---
    @property
    def types(self) -> dict[str, Any]:
        types = self.catalog.get("types") if isinstance(self.catalog, dict) else None
        return types if isinstance(types, dict) else {}

    def type_name(self, type_id: str) -> str:
        info = self.types.get(type_id) or {}
        return str(info.get("name") or type_id)

    def type_info(self, type_id: str) -> dict[str, Any]:
        info = self.types.get(type_id)
        return info if isinstance(info, dict) else {}

    def rules_for_type(self, type_id: str) -> list[Chunk]:
        return [c for c in self.chunks if c.kind == "catalog_rule" and type_id in c.type_ids]

    # --- проверка цитат ---
    def fingerprint(self) -> str:
        h = hashlib.sha1()
        for c in self.chunks:
            h.update(c.embed_text.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()[:16]

    def documents(self) -> set[str]:
        return {norm_text(c.document) for c in self.chunks if c.document}

    def find(self, document: Optional[str] = None, clause: Optional[str] = None,
             url: Optional[str] = None) -> Optional[Chunk]:
        """Найти чанк, подтверждающий цитату (документ+пункт, либо URL+пункт)."""
        doc_k = norm_text(document)
        cl_k = norm_clause(clause)
        url_k = norm_url(url)
        for c in self.chunks:
            c_doc, c_cl = c.key
            # подстрока засчитывается только для длинных названий: «Правила» не должно совпадать с любым документом
            doc_ok = bool(doc_k) and (doc_k == c_doc or (len(doc_k) >= 12 and doc_k in c_doc)
                                      or (len(c_doc) >= 12 and c_doc in doc_k))
            url_ok = bool(url_k) and url_k == norm_url(c.url)
            if not (doc_ok or url_ok):
                continue
            if cl_k and cl_k != c_cl:
                continue
            return c
        return None

    def known_urls(self) -> set[str]:
        return {norm_url(c.url) for c in self.chunks if c.url}

    def short_name(self, document: str, url: str = "") -> str:
        """Короткое имя документа из каталога A13 (``documents[*].short``), если есть."""
        docs = self.catalog.get("documents") if isinstance(self.catalog, dict) else None
        if isinstance(docs, dict):
            u = norm_url(url)
            d = norm_text(document)
            for info in docs.values():
                if not isinstance(info, dict) or not info.get("short"):
                    continue
                if (u and norm_url(info.get("url")) == u) or (d and norm_text(info.get("title")) == d):
                    return str(info["short"])
        return document


# ----------------------------------------------------------------------------- loading

def _read_yaml(path: Path) -> Any:
    import yaml

    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_catalog(cfg: dict[str, Any], warnings: list[str]) -> tuple[dict[str, Any], bool, bool]:
    """Возвращает (catalog, is_fixture, used_fallback_fixture)."""
    ccfg = cfg.get("corpus", {})
    path = resolve_path(ccfg.get("catalog_path"))
    if path and path.is_file():
        try:
            data = _read_yaml(path) or {}
            if isinstance(data, dict):
                return data, bool(data.get("fixture")), False
        except Exception as exc:  # битый YAML не должен ронять ассистента
            warnings.append(f"catalog read error: {exc}")
    # A13 может отдавать каталог функцией (если файл лежит в другом месте/собирается)
    if path and path == resolve_path("norms/catalog.yaml"):
        try:
            from norms import load_catalog  # type: ignore

            data = load_catalog()
            if isinstance(data, dict) and data.get("types"):
                return data, False, False
        except Exception:
            pass
    if fixture_allowed(cfg):
        fx = resolve_path(ccfg.get("fixture_dir", "copilot/fixtures/norms"))
        fx_path = fx / "catalog.yaml" if fx else None
        if fx_path and fx_path.is_file():
            warnings.append("norms/catalog.yaml не найден — используется ТЕСТОВАЯ фикстура (не нормы РК)")
            return _read_yaml(fx_path) or {}, True, True
    warnings.append("каталог норм недоступен — ссылки на нормы не будут приводиться")
    return {"version": 1, "types": {}}, False, False


def catalog_chunks(catalog: dict[str, Any], cite_disabled: bool = False) -> list[Chunk]:
    """Нормы каталога. Типы с legal_status=disabled (норма не подтверждена/неприменима) по умолчанию
    не цитируются: для них честный ответ — «без пункта»."""
    out: list[Chunk] = []
    types = catalog.get("types") if isinstance(catalog, dict) else None
    if not isinstance(types, dict):
        return out
    for type_id, info in types.items():
        if not isinstance(info, dict):
            continue
        status = str(info.get("legal_status") or "").strip().lower()
        if status == "disabled" and not cite_disabled:
            continue
        for j, rule in enumerate(info.get("rules") or []):
            if not isinstance(rule, dict) or not rule.get("document"):
                continue
            quote = str(rule.get("quote") or "").strip()
            name = str(info.get("name") or type_id)
            out.append(Chunk(
                id=f"cat:{type_id}:{j}",
                text=quote or name,
                document=str(rule.get("document")),
                clause=(str(rule["clause"]) if rule.get("clause") else None),
                url=str(rule.get("url") or ""),
                kind="catalog_rule",
                type_ids=[str(type_id)],
                heading=name,
                verified=bool(rule.get("verified", False)),
                status=status,
                note="" if rule.get("verified") else "цитата не сверена с полным текстом",
            ))
    # одна и та же норма может относиться к нескольким типам — объединяем type_ids
    merged: dict[tuple[str, str, str], Chunk] = {}
    for c in out:
        k = (c.key[0], c.key[1], norm_text(c.text))
        if k in merged:
            for t in c.type_ids:
                if t not in merged[k].type_ids:
                    merged[k].type_ids.append(t)
        else:
            merged[k] = c
    return list(merged.values())


def _split_window(text: str, max_chars: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    sentences = re.split(r"(?<=[.!?;])\s+", text)
    parts: list[str] = []
    cur = ""
    for s in sentences:
        while len(s) > max_chars:  # очень длинное «предложение» режем жёстко
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(s[:max_chars])
            s = s[max_chars - overlap:] if overlap < max_chars else s[max_chars:]
        if len(cur) + len(s) + 1 <= max_chars:
            cur = f"{cur} {s}".strip()
        else:
            if cur:
                parts.append(cur)
            tail = cur[-overlap:] if overlap and cur else ""
            if tail and " " in tail:
                tail = tail.split(" ", 1)[1]
            cur = f"{tail} {s}".strip() if tail else s
    if cur:
        parts.append(cur)
    return parts


_UNVERIFIED_NUM_RE = re.compile(r"не\s+сверен|не\s+установлен|по\s+сводке", re.I)
_NO_CLAUSE_RE = re.compile(r"номер\s+пункта\s+не\s+установлен", re.I)


def _clause_from_heading(title: str) -> tuple[Optional[str], bool]:
    """(пункт, номер_сверен). «### п. 12», «12.», «Ф3. Пункт 20 (…)», «Г1. … (пункт 8; номер не сверен)»."""
    if _NO_CLAUSE_RE.search(title):
        return None, False
    num_ok = not _UNVERIFIED_NUM_RE.search(title)
    m = _CLAUSE_RE.search(title)
    if m:
        kind = m.group("kind").lower()
        num = m.group("num")
        if kind.startswith(("подпункт", "пп")):
            # подпункт без номера пункта — не самостоятельная ссылка
            return (f"пп. {num}", num_ok) if m.start() < 4 else (None, False)
        if kind.startswith(("стат", "ст.")):
            return f"ст. {num.rstrip(')')}", num_ok
        return f"п. {num.rstrip(')')}", num_ok
    m2 = re.match(r"^\s*(\d{1,3}(?:\.\d{1,3})*)\.?\s*$", title)
    if m2:
        return f"п. {m2.group(1)}", num_ok
    return None, num_ok


_SKIP_SECTIONS_DEFAULT = r"не\s+цитат|пересказ|todo|реквизит|статус|провенанс"
_SKIP_LINES_DEFAULT = r"^\s*(?:провенанс|источник|\*\*внимание|внимание:)"


def chunk_markdown(text: str, source_file: str = "", max_chars: int = 900, overlap: int = 120,
                   min_chars: int = 40, skip_sections: Optional[str] = _SKIP_SECTIONS_DEFAULT,
                   skip_lines: Optional[str] = _SKIP_LINES_DEFAULT, skip_preamble: bool = True) -> list[Chunk]:
    """Чанкинг markdown-источника нормы: front matter / H1 -> документ; заголовки и «N.» -> пункты.

    Формат A13 (norms/sources/*.md): front matter ``title``/``url``/``retrieval``; разделы «Реквизиты»,
    «Не цитаты: пересказы», «TODO» не являются текстом нормы и не цитируются; строки «Провенанс:»
    отбрасываются; ``retrieval`` не ``full_text`` -> все фрагменты помечаются как не сверенные.
    Преамбула под заголовком документа (до первого раздела или пункта «N.») — не норма и не цитируется.
    """
    meta: dict[str, Any] = {}
    m = _FRONT_MATTER_RE.match(text)
    if m:
        try:
            import yaml

            meta = yaml.safe_load(m.group(1)) or {}
            if not isinstance(meta, dict):
                meta = {}
        except Exception:
            meta = {}
        text = text[m.end():]
    lines = text.splitlines()
    document = str(meta.get("title") or meta.get("document") or meta.get("name") or "").strip()
    if not document:
        for ln in lines:
            hm = _HEADING_RE.match(ln)
            if hm and len(hm.group(1)) == 1:
                document = hm.group(2).strip()
                break
    if not document:
        document = Path(source_file).stem if source_file else "Документ"
    url = str(meta.get("url") or "").strip()
    if not url:
        um = _ADILET_RE.search(text) or _URL_RE.search(text)
        url = um.group(0) if um else ""
    doc_id = str(meta.get("doc_id") or meta.get("id") or Path(source_file).stem or "doc")
    retrieval = str(meta.get("retrieval") or "full_text").strip().lower()
    fragmentary = retrieval not in {"full_text", "full", "adilet"}
    doc_verified = bool(meta.get("verified", False)) and not fragmentary
    skip_sec_re = re.compile(skip_sections, re.I) if skip_sections else None
    skip_line_re = re.compile(skip_lines, re.I) if skip_lines else None

    # (clause, clause_verified, heading_path, lines)
    sections: list[tuple[Optional[str], bool, str, list[str]]] = []
    headings: dict[int, str] = {}
    cur_clause: Optional[str] = None
    cur_num_ok = True
    cur_lines: list[str] = []
    cur_heading = ""
    skip_level: Optional[int] = None  # внутри пропускаемого раздела этого уровня
    in_preamble = skip_preamble       # текст до первого раздела/пункта

    def flush() -> None:
        nonlocal cur_lines
        body = "\n".join(cur_lines).strip()
        if body and skip_level is None:
            sections.append((cur_clause, cur_num_ok, cur_heading, [body]))
        cur_lines = []

    for ln in lines:
        hm = _HEADING_RE.match(ln)
        if hm:
            level = len(hm.group(1))
            title = hm.group(2).strip()
            flush()
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if level == 1:  # заголовок документа (дублирует title)
                headings = {}
                cur_clause, cur_num_ok, cur_heading = None, True, ""
                in_preamble = skip_preamble
                continue
            in_preamble = False
            if skip_sec_re is not None and skip_sec_re.search(title):
                skip_level = level if skip_level is None else min(skip_level, level)
                continue
            headings = {k: v for k, v in headings.items() if k < level}
            clause, num_ok = _clause_from_heading(title)
            headings[level] = title
            cur_clause, cur_num_ok = clause, num_ok
            cur_heading = " / ".join(headings[k] for k in sorted(headings))
            continue
        if skip_level is not None:
            continue
        if skip_line_re is not None and skip_line_re.search(ln):
            continue
        bm = _BARE_NUM_RE.match(ln)
        if bm and not ln.lstrip().startswith(("#", ">")):
            flush()
            in_preamble = False
            cur_clause, cur_num_ok = f"п. {bm.group(1)}", True
        if in_preamble:
            continue
        cur_lines.append(ln.strip().lstrip("> ").strip())
    flush()

    chunks: list[Chunk] = []
    for i, (clause, num_ok, heading, body_lines) in enumerate(sections):
        body = re.sub(r"\s+", " ", " ".join(body_lines)).strip()
        if len(body) < min_chars and not clause:
            continue
        notes = []
        if fragmentary:
            notes.append("фрагмент поисковой выдачи, дословность не сверена")
        if clause and not num_ok:
            notes.append("номер пункта не сверен")
        for j, part in enumerate(_split_window(body, max_chars, overlap)):
            chunks.append(Chunk(
                id=f"src:{doc_id}:{i}:{j}",
                text=part,
                document=document,
                clause=clause,
                url=url,
                kind="source",
                source_file=source_file,
                heading=heading.split(" / ")[-1] if heading else "",
                verified=doc_verified and num_ok,
                note="; ".join(notes),
            ))
    return chunks


def _source_files(cfg: dict[str, Any], use_fixture: bool) -> list[Path]:
    ccfg = cfg.get("corpus", {})
    if use_fixture:
        fx = resolve_path(ccfg.get("fixture_dir", "copilot/fixtures/norms"))
        pattern = str(fx / "sources" / "*.md") if fx else ""
    else:
        p = resolve_path(ccfg.get("sources_glob"))
        pattern = str(p) if p else ""
    return [Path(p) for p in sorted(glob.glob(pattern))] if pattern else []


def build_corpus(catalog: dict[str, Any], source_texts: Iterable[tuple[str, str]] = (),
                 cfg: Optional[dict[str, Any]] = None, is_fixture: bool = False,
                 warnings: Optional[list[str]] = None) -> Corpus:
    """Собрать корпус из каталога и пар (имя_файла, markdown). Удобно для тестов."""
    ccfg = (cfg or default_config()).get("corpus", {})
    chunks = catalog_chunks(catalog, cite_disabled=bool(ccfg.get("cite_disabled", False)))
    for name, text in source_texts:
        chunks.extend(chunk_markdown(
            text, source_file=name,
            max_chars=int(ccfg.get("chunk_max_chars", 900)),
            overlap=int(ccfg.get("chunk_overlap_chars", 120)),
            min_chars=int(ccfg.get("chunk_min_chars", 40)),
            skip_sections=ccfg.get("skip_sections_regex", _SKIP_SECTIONS_DEFAULT),
            skip_lines=ccfg.get("skip_lines_regex", _SKIP_LINES_DEFAULT),
            skip_preamble=bool(ccfg.get("skip_preamble", True)),
        ))
    # привязка источников к типам: пункт совпадает с пунктом каталога (по документу или по URL —
    # в каталоге A13 название документа в rules и title в источнике сформулированы по-разному)
    by_key: dict[tuple[str, str], list[str]] = {}
    for c in chunks:
        if c.kind == "catalog_rule" and c.key[1]:
            by_key.setdefault(c.key, []).extend(c.type_ids)
            if c.url:
                by_key.setdefault((norm_url(c.url), c.key[1]), []).extend(c.type_ids)
    for c in chunks:
        if c.kind != "source" or not c.key[1]:
            continue
        linked = by_key.get(c.key, []) + by_key.get((norm_url(c.url), c.key[1]), [])
        if linked:
            c.type_ids = sorted(set(linked))
    return Corpus(chunks=chunks, catalog=catalog, is_fixture=is_fixture, warnings=list(warnings or []))


def load_corpus(cfg: Optional[dict[str, Any]] = None) -> Corpus:
    cfg = cfg or default_config()
    warnings: list[str] = []
    catalog, is_fixture, used_fallback = _load_catalog(cfg, warnings)
    texts: list[tuple[str, str]] = []
    for path in _source_files(cfg, use_fixture=used_fallback):
        try:
            texts.append((path.name, path.read_text(encoding="utf-8")))
        except OSError as exc:
            warnings.append(f"source read error {path.name}: {exc}")
    corpus = build_corpus(catalog, texts, cfg=cfg, is_fixture=is_fixture, warnings=warnings)
    log.info("copilot corpus: %d chunks (fixture=%s)", len(corpus.chunks), is_fixture)
    return corpus
