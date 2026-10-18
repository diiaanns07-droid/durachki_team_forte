from __future__ import annotations

import json

import pytest
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from copilot.config import load_config
from copilot.router import make_router
from copilot.service import CopilotService


@pytest.fixture()
def jobs(job_violation, job_clean):
    return {"job-test-1": job_violation, "job-test-2": job_clean,
            "busy": {"job_id": "busy", "status": "processing"}}


def _client(service, jobs, async_provider=False, deps=None) -> TestClient:
    app = FastAPI()
    if async_provider:
        async def get_job(job_id):
            return jobs.get(job_id)
    else:
        get_job = jobs.get
    app.include_router(make_router(get_job=get_job, service=service), dependencies=deps or [])
    return TestClient(app)


def test_ask_ok(cfg, corpus, jobs) -> None:
    c = _client(CopilotService(cfg=cfg, corpus=corpus), jobs)
    r = c.post("/api/v1/jobs/job-test-1/ask", json={"question": "Почему это нарушение с телефоном?", "history": []})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) >= {"answer", "citations", "grounded"}
    assert body["citations"][0]["clause"] == "п. 9"


def test_ask_async_provider_and_history(cfg, corpus, jobs) -> None:
    c = _client(CopilotService(cfg=cfg, corpus=corpus), jobs, async_provider=True)
    r = c.post("/api/v1/jobs/job-test-1/ask", json={
        "question": "А когда?", "history": [{"role": "user", "content": "Что с телефоном?"},
                                            {"role": "assistant", "content": "Телефон у уха."}]})
    assert r.status_code == 200 and "00:22" in r.json()["answer"]


def test_ask_errors(cfg, corpus, jobs) -> None:
    c = _client(CopilotService(cfg=cfg, corpus=corpus), jobs)
    assert c.post("/api/v1/jobs/nope/ask", json={"question": "Почему?"}).status_code == 404
    assert c.post("/api/v1/jobs/busy/ask", json={"question": "Почему?"}).status_code == 409
    assert c.post("/api/v1/jobs/job-test-1/ask", json={"question": ""}).status_code == 422
    assert c.post("/api/v1/jobs/job-test-1/ask", json={"history": []}).status_code == 422
    bad_role = {"question": "Почему?", "history": [{"role": "system", "content": "x"}]}
    assert c.post("/api/v1/jobs/job-test-1/ask", json=bad_role).status_code == 422


def test_disabled_is_501_and_status(cfg, corpus, jobs) -> None:
    svc = CopilotService(cfg=load_config({**cfg, "enabled": False}), corpus=corpus)
    c = _client(svc, jobs)
    r = c.post("/api/v1/jobs/job-test-1/ask", json={"question": "Почему?"})
    assert r.status_code == 501
    assert r.json()["detail"]["error"] == "disabled" and r.json()["detail"]["answer"] == ""
    st = c.get("/api/v1/copilot/status").json()
    assert st["enabled"] is False and st["available"] is False


def test_unavailable_is_501(cfg, corpus, jobs) -> None:
    svc = CopilotService(cfg=load_config({**cfg, "llm": {**cfg["llm"], "backend": "auto", "no_model_mode": "unavailable",
                                                          "model_candidates": []}}), corpus=corpus)
    r = _client(svc, jobs).post("/api/v1/jobs/job-test-1/ask", json={"question": "Почему?"})
    assert r.status_code == 501 and r.json()["detail"]["error"] == "unavailable"


def test_sse_stream(cfg, corpus, jobs) -> None:
    c = _client(CopilotService(cfg=cfg, corpus=corpus), jobs)
    with c.stream("POST", "/api/v1/jobs/job-test-1/ask?stream=1", json={"question": "Почему?"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())
    events = [blk for blk in text.split("\n\n") if blk.strip()]
    assert events[0].startswith("event: token")
    last = events[-1]
    assert last.startswith("event: done")
    payload = json.loads(last.split("data: ", 1)[1])
    assert payload["grounded"] is True and payload["citations"]


def test_report_markdown(cfg, corpus, jobs) -> None:
    c = _client(CopilotService(cfg=cfg, corpus=corpus), jobs)
    r = c.get("/api/v1/jobs/job-test-1/report")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/markdown")
    assert "не юридическое заключение" in r.text
    assert c.get("/api/v1/jobs/busy/report").status_code == 409
    assert c.get("/api/v1/jobs/nope/report").status_code == 404


def test_report_available_when_llm_disabled(cfg, corpus, jobs) -> None:
    svc = CopilotService(cfg=load_config({**cfg, "enabled": False}), corpus=corpus)
    assert _client(svc, jobs).get("/api/v1/jobs/job-test-1/report").status_code == 200


def test_api_key_dependency_from_a02(cfg, corpus, jobs) -> None:
    def require_key(x_api_key: str = Header(default="")) -> None:
        if x_api_key != "secret-test":
            raise HTTPException(status_code=401)

    c = _client(CopilotService(cfg=cfg, corpus=corpus), jobs, deps=[Depends(require_key)])
    assert c.post("/api/v1/jobs/job-test-1/ask", json={"question": "Почему?"}).status_code == 401
    assert c.get("/api/v1/copilot/status").status_code == 401
    ok = c.post("/api/v1/jobs/job-test-1/ask", json={"question": "Почему?"}, headers={"X-API-Key": "secret-test"})
    assert ok.status_code == 200


def test_module_router_without_provider_is_503(cfg, corpus) -> None:
    from copilot import router as router_mod

    router_mod.set_job_provider(None)
    app = FastAPI()
    app.include_router(router_mod.router)
    r = TestClient(app).post("/api/v1/jobs/x/ask", json={"question": "Почему?"})
    assert r.status_code == 503


def test_sse_disabled_returns_http_501_not_200(cfg, corpus, jobs) -> None:
    svc = CopilotService(cfg=load_config({**cfg, "enabled": False}), corpus=corpus)
    r = _client(svc, jobs).post("/api/v1/jobs/job-test-1/ask?stream=1", json={"question": "Почему?"})
    assert r.status_code == 501 and r.json()["detail"]["error"] == "disabled"


def test_provider_returning_foreign_job_is_404(cfg, corpus, job_violation) -> None:
    """Провайдер по ошибке отдал другую задачу — по ней не отвечаем и её не раскрываем."""
    app = FastAPI()
    app.include_router(make_router(get_job=lambda jid: job_violation, service=CopilotService(cfg=cfg, corpus=corpus)))
    c = TestClient(app)
    assert c.post("/api/v1/jobs/other-job/ask", json={"question": "Почему?"}).status_code == 404
    assert c.get("/api/v1/jobs/other-job/report").status_code == 404
    assert c.post("/api/v1/jobs/job-test-1/ask", json={"question": "Почему?"}).status_code == 200
