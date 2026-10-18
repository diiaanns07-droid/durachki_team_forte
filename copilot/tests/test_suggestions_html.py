"""Вопросы-подсказки для UI и HTML-заключение (безопасное, с дисклеймером)."""
from __future__ import annotations

import base64
import copy

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from copilot.extractive import classify
from copilot.report import report
from copilot.router import make_router
from copilot.service import CopilotService
from copilot.suggestions import _load, suggestions

JPEG = base64.b64encode(b"\xff\xd8\xff\xe0" + b"\x00" * 64 + b"\xff\xd9").decode()


def _all_suggested() -> list[str]:
    d = _load()
    out = list(d.get("general") or []) + list(d.get("no_violation") or []) + list(d.get("incomplete") or [])
    for qs in (d.get("types") or {}).values():
        out.extend(qs)
    return list(dict.fromkeys(out))


def test_three_to_four_per_type_including_v09() -> None:
    types = _load()["types"]
    for tid in [f"V0{i}" for i in range(1, 10)] + ["V99"]:
        assert 3 <= len(types[tid]) <= 4, tid


@pytest.mark.parametrize("q", _all_suggested())
def test_suggestions_are_answerable_not_refusals(q) -> None:
    assert classify(q) not in {"identity", "unobservable", "offtopic"}, q


@pytest.mark.parametrize("q", _all_suggested())
def test_suggestions_get_grounded_answers(q, cfg, corpus, job_violation, job_clean) -> None:
    svc = CopilotService(cfg=cfg, corpus=corpus)
    for job in (job_violation, job_clean):
        r = svc.answer(job, q, [])
        assert r["error"] is None and r["answer"] and r["removed_citations"] == [], (q, r["answer"])


def test_suggestions_follow_result(job_violation, job_clean) -> None:
    s = suggestions(job_violation)
    assert len(s["suggestions"]) == 4 and set(s["by_type"]) == {"V05", "V03"}
    assert s["suggestions"][0] == "Какие кадры показывают телефон?"
    clean = suggestions(job_clean)
    assert "Почему нарушений не найдено?" in clean["suggestions"]
    inc = suggestions({**job_clean, "meta": {"completeness": {"complete": False}}})
    assert inc["suggestions"][0] == "Насколько полон анализ?"


def test_coverage_and_exclusions_intents(cfg, corpus, job_clean) -> None:
    svc = CopilotService(cfg=cfg, corpus=corpus)
    a = svc.answer({**job_clean, "meta": {"completeness": {"complete": False, "analyzed_until_sec": 30,
                                                           "duration_sec": 60}}}, "Насколько полон анализ?", [])
    assert "НЕПОЛНЫЙ" in a["answer"] and "не означает" in a["answer"]
    u = svc.answer(job_clean, "Насколько полон анализ?", [])["answer"]
    assert "сведений о полноте анализа нет" in u


def test_router_suggestions_endpoint(cfg, corpus, job_violation) -> None:
    app = FastAPI()
    app.include_router(make_router(get_job={"job-test-1": job_violation}.get,
                                   service=CopilotService(cfg=cfg, corpus=corpus)))
    c = TestClient(app)
    r = c.get("/api/v1/jobs/job-test-1/suggestions?limit=3")
    assert r.status_code == 200 and len(r.json()["suggestions"]) == 3
    assert c.get("/api/v1/jobs/none/suggestions").status_code == 404


# ----------------------------------------------------------------------------- HTML

def test_html_report_is_escaped_and_has_disclaimer(cfg, corpus, job_violation) -> None:
    job = copy.deepcopy(job_violation)
    job["violations"][0]["description"] = "<script>alert(1)</script> [x](javascript:alert(2)) **жирно**"
    job["violations"][0]["evidence_frames"] = [{"timestamp_sec": 22.0, "image_base64": JPEG},
                                               {"timestamp_sec": 23.0, "image_base64": "PHN2Zz48L3N2Zz4="}]
    h = report(job, corpus=corpus, cfg=cfg, fmt="html")
    assert h.startswith("<!doctype html>") and "Content-Security-Policy" in h and "default-src 'none'" in h
    assert "<script>" not in h and "javascript:" not in h and "&lt;script&gt;" in h
    assert "не юридическое заключение" in h
    assert h.count("<img ") == 1 and f"data:image/jpeg;base64,{JPEG}" in h   # не-JPEG (svg) отброшен
    assert "@@FRAMES" not in h


def test_html_report_without_frames_option(cfg, corpus, job_violation) -> None:
    from copilot.config import load_config

    job = copy.deepcopy(job_violation)
    job["violations"][0]["evidence_frames"] = [{"timestamp_sec": 22.0, "image_base64": JPEG}]
    cfg2 = load_config({**cfg, "report": {**cfg["report"], "html_include_frames": False}})
    assert "<img " not in report(job, corpus=corpus, cfg=cfg2, fmt="html")


def test_router_report_html(cfg, corpus, job_violation) -> None:
    app = FastAPI()
    app.include_router(make_router(get_job={"job-test-1": job_violation}.get,
                                   service=CopilotService(cfg=cfg, corpus=corpus)))
    c = TestClient(app)
    r = c.get("/api/v1/jobs/job-test-1/report?format=html")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "default-src 'none'" in r.headers["content-security-policy"]
    assert c.get("/api/v1/jobs/job-test-1/report").headers["content-type"].startswith("text/markdown")
    assert c.get("/api/v1/jobs/job-test-1/report?format=pdf").status_code == 422
