from __future__ import annotations

from pathlib import Path

import pytest

from copilot.config import load_config

from copilot.corpus import build_corpus, chunk_markdown, norm_clause, norm_url
from copilot.rag import Retriever

TOY_MD = """---
title: "Тестовые правила кассы"
url: "https://example.invalid/toy"
---
# Тестовые правила кассы

## Глава 1. Общие положения

1. Кассир ведёт операции только на рабочем месте.

2. Посторонние лица в кассовое помещение не допускаются.
1) клиенты обслуживаются через окно;
2) инкассаторы допускаются по процедуре.

### Пункт 3
Мобильный телефон при работе с ценностями не используется.
"""


def test_norm_clause_variants() -> None:
    assert norm_clause("п. 3") == norm_clause("пункт 3") == norm_clause("п.3") == "п.3"
    assert norm_clause("Статья 5") == "ст.5"
    assert norm_clause("пп. 2)") == "пп.2"
    assert norm_clause("12.") == "п.12"
    assert norm_url("https://Adilet.zan.kz/rus/docs/V123/#p5") == "adilet.zan.kz/rus/docs/v123"


def test_chunk_markdown_detects_document_clauses_and_url() -> None:
    chunks = chunk_markdown(TOY_MD, source_file="toy.md")
    assert chunks, "чанки должны быть"
    assert {c.document for c in chunks} == {"Тестовые правила кассы"}
    assert all(c.url == "https://example.invalid/toy" for c in chunks)
    clauses = [c.clause for c in chunks]
    assert "п. 1" in clauses and "п. 2" in clauses and "п. 3" in clauses
    p2 = next(c for c in chunks if c.clause == "п. 2")
    assert "окно" in p2.text, "подпункты 1)/2) остаются внутри пункта"
    assert "Глава 1" in p2.heading


def test_chunk_markdown_splits_long_sections() -> None:
    long = "# Док\n\n5. " + " ".join(["Очень длинное предложение про кассу."] * 80)
    chunks = chunk_markdown(long, max_chars=300, overlap=50)
    assert len(chunks) > 3
    assert all(len(c.text) <= 300 for c in chunks)
    assert all(c.clause == "п. 5" for c in chunks)


def test_catalog_and_sources_loaded(corpus) -> None:
    assert corpus.is_fixture  # файл-фикстура помечен fixture: true -> в ответах будет предупреждение
    kinds = {c.kind for c in corpus.chunks}
    assert kinds == {"catalog_rule", "source"}
    rules = corpus.rules_for_type("V05")
    assert rules and rules[0].clause == "п. 9"
    # источник с тем же пунктом привязан к типу
    src = [c for c in corpus.chunks if c.kind == "source" and c.clause == "п. 9"]
    assert src and "V05" in src[0].type_ids


def test_find_citation(corpus) -> None:
    assert corpus.find("ФИКСТУРА: Тестовый регламент кассы", "пункт 9") is not None
    assert corpus.find(None, "п. 9", "https://example.invalid/fixture/reglament") is not None
    assert corpus.find("ФИКСТУРА: Тестовый регламент кассы", "п. 999") is None
    assert corpus.find("Выдуманный закон", "п. 9") is None


def test_retriever_top1_is_relevant(cfg, corpus, embedder) -> None:
    r = Retriever(corpus, cfg, embedder=embedder)
    cases = {
        "можно ли кассиру разговаривать по мобильному телефону": "п. 9",
        "посторонний человек зашёл в кассовую зону": "п. 3",
        "инкассаторская сумка оставлена без присмотра": "п. 4",
        "сейф открыт и кассир ушёл": "п. 7",
    }
    for q, clause in cases.items():
        hits = r.search(q, top_k=3)
        assert hits, q
        assert hits[0].chunk.clause == clause, (q, [(h.chunk.clause, round(h.score, 3)) for h in hits])
        top = hits[0].chunk
        assert top.document and top.url  # у каждого результата есть источник


def test_retriever_dedupes_same_clause(cfg, corpus, embedder) -> None:
    hits = Retriever(corpus, cfg, embedder=embedder).search("мобильный телефон ценности", top_k=6)
    keys = [h.chunk.key for h in hits if h.chunk.clause]
    assert len(keys) == len(set(keys))


def test_type_boost(cfg, corpus, embedder) -> None:
    hits = Retriever(corpus, cfg, embedder=embedder).search("что нарушено", top_k=2, type_ids=["V07"])
    assert hits and "V07" in hits[0].chunk.type_ids


def test_empty_corpus_is_safe(cfg, embedder) -> None:
    c = build_corpus({"types": {}}, [], cfg=cfg)
    assert Retriever(c, cfg, embedder=embedder).search("телефон") == []


# ----------------------------------------------------------------------------- формат каталога A13 v2

A13_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "norms_a13"


def _a13_corpus(cfg):
    from copilot.corpus import load_corpus

    c2 = load_config({**cfg, "corpus": {**cfg["corpus"], "catalog_path": str(A13_DIR / "catalog.yaml"),
                                        "sources_glob": str(A13_DIR / "sources" / "*.md")}})
    return load_corpus(c2)


def test_a13_sources_skip_non_norm_sections(cfg) -> None:
    corpus = _a13_corpus(cfg)
    texts = " ".join(c.text for c in corpus.chunks)
    assert "Пересказ" not in texts and "TODO" not in texts and "Сверить пункт 7" not in texts
    assert "Статус: действует" not in texts and "Провенанс" not in texts and "НЕ полный текст" not in texts
    assert "Дословность" not in texts  # преамбула под H1 не цитируется


def test_a13_fragment_clauses_and_honest_flags(cfg) -> None:
    corpus = _a13_corpus(cfg)
    src = {c.heading.split(".")[0]: c for c in corpus.chunks if c.kind == "source"}
    assert src["Ф1"].clause == "п. 20" and src["Ф1"].note.startswith("фрагмент поисковой выдачи")
    assert src["Ф2"].clause is None                      # «номер пункта не установлен»
    assert src["Ф3"].clause == "п. 8" and "номер пункта не сверен" in src["Ф3"].note
    assert all(not c.verified for c in corpus.chunks)    # search_snippets — ничего не сверено
    assert src["Ф1"].citation()["fragment"].startswith("Ф1. Пункт 20")


def test_a13_disabled_type_not_cited_and_url_linking(cfg) -> None:
    corpus = _a13_corpus(cfg)
    assert corpus.rules_for_type("V01") == []            # legal_status: disabled -> «без пункта»
    v08 = corpus.rules_for_type("V08")
    assert len(v08) == 1 and v08[0].status == "candidate" and not v08[0].verified
    # источник с тем же URL и пунктом связан с V08, хотя название документа в rules другое
    src20 = [c for c in corpus.chunks if c.kind == "source" and c.clause == "п. 20"]
    assert src20 and src20[0].type_ids == ["V08"]
    assert corpus.short_name(src20[0].document, src20[0].url) == "ФИКСТУРА: Правила кассы (№ 1)"


REAL_NORMS = Path(__file__).resolve().parents[2] / "norms"


@pytest.mark.skipif(not (REAL_NORMS / "catalog.yaml").is_file(), reason="norms/ (A13) нет в этой ветке")
def test_real_a13_norms_load_without_non_norm_text(cfg) -> None:
    from copilot.corpus import load_corpus

    c2 = load_config({**cfg, "corpus": {**cfg["corpus"], "catalog_path": str(REAL_NORMS / "catalog.yaml"),
                                        "sources_glob": str(REAL_NORMS / "sources" / "*.md"),
                                        "fixture_fallback": False}})
    corpus = load_corpus(c2)
    assert corpus.chunks and not corpus.is_fixture
    for c in corpus.chunks:
        assert "Провенанс" not in c.text and "пересказ" not in c.heading.lower()
