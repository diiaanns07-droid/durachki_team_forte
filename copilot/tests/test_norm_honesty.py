"""Честные нормы (формат каталога A13 v2): каталог vs найдено поиском, кандидат/не сверено, «без пункта»."""
from __future__ import annotations

from pathlib import Path

import pytest

from copilot.config import load_config
from copilot.corpus import load_corpus
from copilot.llm import ScriptedLLM
from copilot.prompts import build_messages, normalize_result
from copilot.report import report
from copilot.service import CopilotService

A13_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "norms_a13"
NO_DATA = "По имеющимся данным это установить нельзя"


@pytest.fixture()
def a13_cfg(cfg):
    return load_config({**cfg, "corpus": {**cfg["corpus"], "catalog_path": str(A13_DIR / "catalog.yaml"),
                                          "sources_glob": str(A13_DIR / "sources" / "*.md")}})


@pytest.fixture()
def a13_corpus(a13_cfg):
    return load_corpus(a13_cfg)


@pytest.fixture()
def job_v08_v01():
    return {"job_id": "a13", "status": "done", "verdict": "violation",
            "meta": {"completeness": {"complete": True}},
            "violations": [
                {"type": "Пересчёт наличных не в присутствии клиента", "type_id": "V08", "timecode_sec": 12.0,
                 "t_start": 11.0, "t_end": 15.0, "confidence": 0.7,
                 "description": "Кассир пересчитывает наличные, клиента у окна нет", "evidence_frames": []},
                {"type": "Посторонний в зоне кассира", "type_id": "V01", "timecode_sec": 30.0,
                 "t_start": 29.0, "t_end": 33.0, "confidence": 0.6,
                 "description": "Человек не из персонала на стороне кассира", "evidence_frames": []}]}


def test_candidate_norm_is_labeled_and_linked_to_event(a13_cfg, a13_corpus, job_v08_v01) -> None:
    r = CopilotService(cfg=a13_cfg, corpus=a13_corpus).answer(job_v08_v01, "Почему отмечен пересчёт?", [])
    assert "Норма: [1] (требует сверки)" in r["answer"]
    c = r["citations"][0]
    assert c["relation"] == "catalog" and c["status"] == "candidate" and c["verified"] is False
    assert c["events"] == [1] and c["clause"] == "п. 20"
    assert "норма каталога" in r["sources"][0] and "кандидат" in r["sources"][0] and "не сверено" in r["sources"][0]
    assert c.get("short") is None or c["short"].startswith("ФИКСТУРА")


def test_disabled_type_says_without_clause(a13_cfg, a13_corpus, job_v08_v01) -> None:
    svc = CopilotService(cfg=a13_cfg, corpus=a13_corpus)
    r = svc.answer(job_v08_v01, "Почему отмечен посторонний?", [])
    assert "без пункта" in r["answer"]
    assert all(c["relation"] != "catalog" or 2 not in c["events"] for c in r["citations"])
    r2 = svc.answer(job_v08_v01, "Какой пункт нарушен посторонним?", [])
    assert r2["answer"].startswith(NO_DATA) and "без пункта" in r2["answer"]
    for c in r2["citations"]:          # если что-то найдено поиском — помечено честно
        assert c["relation"] == "retrieved" and c["verified"] is False


def test_prompt_marks_norm_status_and_missing_clause(a13_cfg, a13_corpus, job_v08_v01) -> None:
    svc = CopilotService(cfg=a13_cfg, corpus=a13_corpus, llm=ScriptedLLM(["ok"]))
    res = normalize_result(job_v08_v01)
    p = svc._prepare(job_v08_v01, "Почему?", [])
    msgs = build_messages(res, p["refs"], a13_corpus, "Почему?", [])
    ctx = msgs[0]["content"]
    assert "Пометки: норма каталога для этого типа; статус в каталоге: кандидат, требует сверки" in ctx
    assert "Норма: без пункта" in ctx and "Пояснение каталога (не текст нормы)" in ctx


def test_llm_cannot_upgrade_candidate_to_confirmed(a13_cfg, a13_corpus, job_v08_v01) -> None:
    """Модель может написать «подтверждено», но цитата в ответе всё равно несёт verified=false/candidate."""
    llm = ScriptedLLM(["Пересчёт без клиента запрещён [N1]."])
    r = CopilotService(cfg=a13_cfg, corpus=a13_corpus, llm=llm).answer(job_v08_v01, "Почему?", [])
    assert r["citations"] and r["citations"][0]["verified"] is False and r["citations"][0]["status"] == "candidate"


def test_report_shows_status_note_exclusions_and_without_clause(a13_cfg, a13_corpus, job_v08_v01) -> None:
    md = report(job_v08_v01, corpus=a13_corpus, cfg=a13_cfg)
    assert "норма-кандидат" in md and "цитата требует сверки с первоисточником" in md
    assert "без пункта" in md
    assert "Пояснение каталога (не текст нормы)" in md
    assert "Когда это не нарушение (каталог):" in md and "руки перекрыты счётчиком" in md


def test_irrelevant_search_hits_not_offered(a13_cfg, a13_corpus) -> None:
    """Нет нормы про телефон — нельзя подсовывать нерелевантные фрагменты как «возможно относится»."""
    job = {"job_id": "p", "status": "done", "verdict": "violation", "meta": {"completeness": {"complete": True}},
           "violations": [{"type": "Использование мобильного телефона при работе с ценностями", "type_id": "V05",
                           "timecode_sec": 5.0, "description": "телефон у уха", "evidence_frames": []}]}
    r = CopilotService(cfg=a13_cfg, corpus=a13_corpus).answer(job, "Какая норма применяется к телефону?", [])
    assert r["answer"].startswith(NO_DATA) and "без пункта" in r["answer"] and r["citations"] == []


def test_relevant_search_hit_still_offered(a13_cfg, a13_corpus) -> None:
    job = {"job_id": "p", "status": "done", "verdict": "violation", "meta": {"completeness": {"complete": True}},
           "violations": [{"type": "Иное", "type_id": "V99", "timecode_sec": 5.0, "description": "x",
                           "evidence_frames": []}]}
    r = CopilotService(cfg=a13_cfg, corpus=a13_corpus).answer(job, "Где норма про изоляцию кассы от других помещений?", [])
    assert any(c["relation"] == "retrieved" and c["clause"] == "п. 8" for c in r["citations"])


def test_exclusions_answer_keeps_catalog_clause_mentions(a13_cfg, a13_corpus, job_v08_v01) -> None:
    r = CopilotService(cfg=a13_cfg, corpus=a13_corpus).answer(job_v08_v01, "Когда пересчёт не нарушение?", [])
    assert "не считается нарушением" in r["answer"] and "руки перекрыты счётчиком" in r["answer"]
    assert r["removed_citations"] == []
