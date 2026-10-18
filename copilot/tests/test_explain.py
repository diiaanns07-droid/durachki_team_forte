"""Прозрачное «почему отмечено»: только данные пайплайна (A12 overlays / A01 violation_extra), без выдумок."""
from __future__ import annotations

from copilot.llm import ScriptedLLM
from copilot.prompts import build_messages, explain_for, normalize_result
from copilot.report import report
from copilot.service import CopilotService


def _job(with_explain: bool = True) -> dict:
    job = {"job_id": "e", "status": "done", "verdict": "violation",
           "violations": [{"type": "Использование мобильного телефона при работе с ценностями", "type_id": "V05",
                           "timecode_sec": 22.0, "t_start": 20.0, "t_end": 25.0, "confidence": 0.81,
                           "description": "Телефон у уха кассира", "evidence_frames": []}],
           "meta": {"completeness": {"complete": True}}}
    if with_explain:
        job["meta"]["ui"] = {
            "evidence_overlays": [{"violation_index": 0, "facts": ["телефон в руке кассира", "наличные на столе"],
                                   "sources": {"rules": 0.72, "vlm": 0.6}, "vlm": "supports", "frames": ["QQ=="]}],
            "review_hints": [{"type_id": "V03", "type": "Сейф открыт", "timecode_sec": 41.0, "score": 0.3,
                              "threshold": 0.5, "facts": ["дверца сейфа"], "reasons_ru": ["ниже порога"]}]}
    return job


def test_explain_reads_only_pipeline_data() -> None:
    res = normalize_result(_job())
    ex = explain_for(res, res["violations"][0])
    assert ex == {"facts": ["телефон в руке кассира", "наличные на столе"],
                  "sources": [("rules", 0.72), ("vlm", 0.6)], "vlm": "supports"}
    empty = normalize_result(_job(with_explain=False))
    assert explain_for(empty, empty["violations"][0]) == {"facts": [], "sources": [], "vlm": None}


def test_violation_extra_from_contract_is_used() -> None:
    job = _job(with_explain=False)
    job["meta"]["violation_extra"] = [{"facts": ["факт из extra"], "sources": ["rules", "temporal"],
                                       "vlm": {"agreement": "uncertain"}}]
    res = normalize_result(job)
    ex = explain_for(res, res["violations"][0])
    assert ex["facts"] == ["факт из extra"] and ex["sources"] == [("rules", None), ("temporal", None)]
    assert ex["vlm"] == "uncertain"


def test_why_answer_shows_facts_detectors_vlm(cfg, corpus) -> None:
    r = CopilotService(cfg=cfg, corpus=corpus).answer(_job(), "Почему это отмечено?", [])
    a = r["answer"]
    assert "Наблюдаемые факты (от детекторов): телефон в руке кассира; наличные на столе" in a
    assert "правила по сигналам сцены (0.72)" in a and "VLM по кадрам согласна с детектором" in a
    assert r["grounded"] is True and r["removed_citations"] == []


def test_why_without_explain_data_invents_nothing(cfg, corpus) -> None:
    a = CopilotService(cfg=cfg, corpus=corpus).answer(_job(False), "Почему это отмечено?", [])["answer"]
    assert "Наблюдаемые факты" not in a and "Кто отметил" not in a and "VLM" not in a


def test_hints_are_reported_as_not_in_verdict(cfg, corpus) -> None:
    svc = CopilotService(cfg=cfg, corpus=corpus)
    a = svc.answer(_job(), "Какие спорные места?", [])["answer"]
    assert "в вердикт не вошли" in a and "Сейф открыт, 00:41: ниже порога" in a
    a2 = svc.answer(_job(False), "Какие спорные места?", [])["answer"]
    assert "Сведений о спорных местах" in a2 and "не означает" in a2   # unknown ≠ «нет»


def test_hint_timecode_allowed_in_llm_answer(cfg, corpus) -> None:
    llm = ScriptedLLM(["На 41 с был кандидат «сейф открыт», но он ниже порога и в вердикт не вошёл."])
    r = CopilotService(cfg=cfg, corpus=corpus, llm=llm).answer(_job(), "Что ещё было?", [])
    assert "41 с" in r["answer"]


def test_context_and_report_include_explanation(cfg, corpus) -> None:
    svc = CopilotService(cfg=cfg, corpus=corpus, llm=ScriptedLLM(["ok"]))
    p = svc._prepare(_job(), "Почему?", [])
    ctx = build_messages(p["result"], p["refs"], corpus, "Почему?", [])[0]["content"]
    assert "Кто отметил: правила по сигналам сцены (0.72), проверка VLM (0.60)" in ctx
    assert "Спорные места (кандидаты ниже порога, в вердикт НЕ вошли)" in ctx and "QQ==" not in ctx
    md = report(_job(), corpus=corpus, cfg=cfg)
    assert "Почему отмечено (данные системы)" in md and "Спорные места (не вошли в вердикт)" in md
