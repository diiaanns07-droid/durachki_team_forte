"""Готовность к интеграции (синтетика, без GPU/весов/сети).

* импорт и запуск основного API не требуют весов copilot;
* нет модели -> понятный unavailable/501, без фальшивого ответа и без облака;
* ошибка/таймаут copilot не ломают остальной API (анализ видео);
* веса не скачиваются автоматически;
* ответы опираются только на переданные факты и источники, неподтверждённое не выдаётся;
* нехватка данных -> «По имеющимся данным это установить нельзя».
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from copilot.config import load_config
from copilot.corpus import build_corpus
from copilot.guard import UNVERIFIED_ANSWER
from copilot.llm import LLMBusy, LlamaCppBackend, ScriptedLLM
from copilot.router import make_router
from copilot.service import CopilotService

ROOT = Path(__file__).resolve().parents[2]
NO_DATA = "По имеющимся данным это установить нельзя"


def _analysis_app(copilot_router) -> TestClient:
    """Имитация основного API (A02): свой эндпоинт анализа + подключённый роутер copilot."""
    app = FastAPI()

    @app.post("/api/v1/analyze")
    def analyze() -> dict:
        return {"job_id": "job-x"}

    @app.get("/api/v1/jobs/{job_id}")
    def job(job_id: str) -> dict:
        return {"job_id": job_id, "status": "done", "verdict": "no_violation", "violations": []}

    app.include_router(copilot_router)
    return TestClient(app)


class _Boom:
    name = "llama_cpp"
    is_generative = True

    def stream_chat(self, *a, **k):
        raise RuntimeError("CUDA error: out of memory")

    def warmup(self):
        raise RuntimeError("cannot load model")

    def release(self):
        raise RuntimeError("cannot free")

    def info(self):
        raise RuntimeError("info failed")


# ----------------------------------------------------------------------------- старт без весов

def test_main_api_starts_without_copilot_weights(monkeypatch, tmp_path, job_violation) -> None:
    from copilot import router as router_mod
    from copilot import service as service_mod

    monkeypatch.setenv("COPILOT_MODEL_PATH", str(tmp_path / "missing.gguf"))
    service_mod.set_service(None)
    router_mod.set_job_provider({"job-test-1": job_violation}.get)
    try:
        with _analysis_app(router_mod.router) as client:  # startup приложения
            assert client.post("/api/v1/analyze").status_code == 200
            st = client.get("/api/v1/copilot/status").json()
            assert st == {**st, "enabled": False, "available": False, "reason": "disabled"}
            r = client.post("/api/v1/jobs/job-test-1/ask", json={"question": "Почему?"})
            assert r.status_code == 501 and r.json()["detail"]["answer"] == ""
            assert client.get("/api/v1/jobs/job-test-1/report").status_code == 200  # отчёт без LLM

            monkeypatch.setenv("COPILOT_ENABLED", "1")  # включили, но модели (ml.vlm.service) нет
            service_mod.set_service(None)
            st = client.get("/api/v1/copilot/status").json()
            assert st["enabled"] is True and st["available"] is True and st["generative"] is False
            assert st["mode"] == "retrieval" and st["reason"] == "generative_unavailable"
            r = client.post("/api/v1/jobs/job-test-1/ask", json={"question": "Почему?"})
            assert r.status_code == 200
            body = r.json()
            assert body["mode"] == "retrieval" and body["generative"] is False and body["backend"] != "mock"
            assert body["answer"].startswith("(Генеративный режим недоступен")
    finally:
        service_mod.set_service(None)
        router_mod.set_job_provider(None)


def test_enabled_without_weights_subprocess_no_heavy_imports_no_network(tmp_path) -> None:
    code = r"""
import json, socket, sys
def _blocked(*a, **k):
    raise AssertionError("network access attempted")
socket.socket.connect = _blocked
socket.create_connection = _blocked
import copilot
from copilot.router import make_router
st = copilot.status()
r = copilot.answer({"job_id": "1", "status": "done", "verdict": "violation", "violations": [
    {"type": "t", "type_id": "V05", "timecode_sec": 3.0, "description": "d", "evidence_frames": []}]}, "Почему?")
heavy = [m for m in ("torch", "transformers", "llama_cpp", "sentence_transformers", "huggingface_hub") if m in sys.modules]
print(json.dumps({"available": st["available"], "reason": st["reason"], "error": r["error"],
                  "answer": r["answer"], "heavy": heavy}))
"""
    env = {"PATH": "/usr/bin:/bin", "COPILOT_ENABLED": "1", "COPILOT_MODEL_PATH": str(tmp_path / "none.gguf"),
           "COPILOT_CACHE_DIR": str(tmp_path / "cache"), "HF_HUB_OFFLINE": "1", "COPILOT_EMBED_BACKEND": "hashing",
           "COPILOT_VLM_MODULE": "copilot._no_vlm_service_for_test"}  # «модели нет» и после мерджа A11
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60, env=env)
    assert out.returncode == 0, out.stderr
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["available"] is True and res["reason"] == "generative_unavailable" and res["error"] is None
    assert res["answer"].startswith("(Генеративный режим недоступен") and res["heavy"] == []


# ----------------------------------------------------------------------------- сбои copilot не ломают API

def test_copilot_errors_and_timeouts_do_not_break_api(cfg, corpus, job_violation) -> None:
    jobs = {"job-test-1": job_violation}
    boom = CopilotService(cfg=cfg, corpus=corpus, llm=_Boom())
    client = _analysis_app(make_router(get_job=jobs.get, service=boom))
    r = client.post("/api/v1/jobs/job-test-1/ask", json={"question": "Почему это нарушение?"})
    assert r.status_code == 500 and r.json()["detail"]["error"] == "llm_error" and r.json()["detail"]["answer"] == ""
    assert client.post("/api/v1/analyze").status_code == 200

    slow_cfg = load_config({**cfg, "llm": {**cfg["llm"], "timeout_s": 0.2}})
    slow = CopilotService(cfg=slow_cfg, corpus=corpus, llm=ScriptedLLM([" ".join(["слово"] * 100)], delay_s=0.02))
    client2 = _analysis_app(make_router(get_job=jobs.get, service=slow))
    t0 = time.monotonic()
    r = client2.post("/api/v1/jobs/job-test-1/ask", json={"question": "Почему?"})
    assert r.status_code == 504 and time.monotonic() - t0 < 3
    assert client2.post("/api/v1/analyze").status_code == 200
    assert client2.get("/api/v1/jobs/job-test-1").status_code == 200


def test_warmup_release_status_never_raise(cfg, corpus) -> None:
    svc = CopilotService(cfg=cfg, corpus=corpus, llm=_Boom())
    assert svc.warmup() is False
    svc.release()
    st = svc.status()
    assert st["available"] is False and st["reason"] == "unavailable"


def test_internal_errors_are_reported_not_raised(cfg, job_violation, monkeypatch) -> None:
    import copilot.service as service_mod

    def _broken(_cfg):
        raise OSError("disk unavailable")

    monkeypatch.setattr(service_mod, "load_corpus", _broken)
    svc = CopilotService(cfg=cfg)
    r = svc.answer(job_violation, "Почему?", [])
    assert r["error"] == "internal_error" and r["answer"] == ""
    events = list(svc.stream(job_violation, "Почему?", []))
    assert events[-1]["event"] == "error" and events[-1]["error"] == "internal_error"
    client = TestClient(_app_with(svc, {"job-test-1": job_violation}))
    resp = client.post("/api/v1/jobs/job-test-1/ask", json={"question": "Почему?"})
    assert resp.status_code == 500 and resp.json()["detail"]["error"] == "internal_error"


def _app_with(svc, jobs) -> FastAPI:
    app = FastAPI()
    app.include_router(make_router(get_job=jobs.get, service=svc))
    return app


def test_model_busy_does_not_block_api_threads(tmp_path, monkeypatch) -> None:
    mod = types.ModuleType("llama_cpp")

    class _Llama:
        def __init__(self, **kw):
            pass

        def create_chat_completion(self, **kw):
            yield {"choices": [{"delta": {"content": "x"}}]}

    mod.Llama = _Llama
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"\0")
    be = LlamaCppBackend(gguf, load_config({"llm": {"gpu_layers": "0", "idle_unload_s": 0}}))
    be._lock.acquire()  # другой запрос держит модель
    try:
        t0 = time.monotonic()
        with pytest.raises(LLMBusy):
            "".join(be.stream_chat([{"role": "user", "content": "q"}], deadline=time.monotonic() + 0.2))
        assert time.monotonic() - t0 < 1.0
    finally:
        be._lock.release()


# ----------------------------------------------------------------------------- без автоскачивания

def _block_network(monkeypatch) -> None:
    def _boom(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", _boom)
    monkeypatch.setattr(socket.socket, "connect", _boom)


def test_e5_never_downloads_falls_back_to_hashing(monkeypatch) -> None:
    from copilot.embed import HashingEmbedder, make_embedder

    calls = []
    hub = types.ModuleType("huggingface_hub")

    def snapshot_download(repo_id, local_files_only=False, **kw):
        calls.append({"repo_id": repo_id, "local_files_only": local_files_only})
        raise FileNotFoundError("not in local cache")

    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    _block_network(monkeypatch)
    warnings: list[str] = []
    emb = make_embedder(load_config({"embed": {"backend": "auto"}}), warnings)
    assert isinstance(emb, HashingEmbedder) and warnings
    assert calls == [{"repo_id": "intfloat/multilingual-e5-small", "local_files_only": True}]


def test_e5_loads_only_from_local_snapshot(monkeypatch, tmp_path) -> None:
    from copilot.embed import E5Embedder

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda repo_id, local_files_only=False, **kw: (
        str(tmp_path) if local_files_only else (_ for _ in ()).throw(AssertionError("download attempted")))
    st_mod = types.ModuleType("sentence_transformers")
    loaded = {}

    class _ST:
        def __init__(self, path, device="cpu", **kw):
            loaded["path"], loaded["kw"] = path, kw

    st_mod.SentenceTransformer = _ST
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "sentence_transformers", st_mod)
    E5Embedder("intfloat/multilingual-e5-small")
    assert loaded["path"] == str(tmp_path)


def test_llama_backend_never_uses_from_pretrained(monkeypatch, tmp_path) -> None:
    mod = types.ModuleType("llama_cpp")

    class _Llama:
        def __init__(self, **kw):
            assert "repo_id" not in kw and Path(kw["model_path"]).is_file()

        @classmethod
        def from_pretrained(cls, *a, **k):
            raise AssertionError("download attempted")

        def create_chat_completion(self, **kw):
            yield {"choices": [{"delta": {"content": "ok"}}]}

    mod.Llama = _Llama
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    _block_network(monkeypatch)
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"\0")
    be = LlamaCppBackend(gguf, load_config({"llm": {"gpu_layers": "0", "idle_unload_s": 0}}))
    assert "".join(be.stream_chat([{"role": "user", "content": "q"}])) == "ok"


# ----------------------------------------------------------------------------- только факты и источники

MIXED = ("Пользоваться телефоном при работе с ценностями запрещено [N1]. "
         "Также нарушен п. 999 регламента. "
         "Кассир пересчитал 500 000 тенге. "
         "На 95 с кассир передал пакет клиенту. "
         "Кассир, судя по всему, украл деньги и скрылся через чёрный ход. "
         "Подробнее: https://adilet.zan.kz/rus/docs/V0000000. "
         "Кассир держит телефон у уха [V1].")


def test_only_given_facts_and_sources_reach_user(cfg, corpus, job_violation) -> None:
    r = CopilotService(cfg=cfg, corpus=corpus, llm=ScriptedLLM([MIXED])).answer(job_violation, "Почему?", [])
    ans = r["answer"]
    assert "запрещено [1]" in ans and "телефон у уха (№1, 00:22)" in ans
    for bad in ("999", "500 000", "95 с", "украл", "V0000000", "[N"):
        assert bad not in ans, bad
    assert r["grounded"] is False and len(r["removed_citations"]) >= 5
    assert [c["clause"] for c in r["citations"]] == ["п. 9"]
    for c in r["citations"]:
        assert corpus.find(c["document"], c["clause"], c["url"]) is not None


def test_all_unverified_becomes_honest_no_data(cfg, corpus, job_violation) -> None:
    llm = ScriptedLLM(["Согласно п. 999 Закона «О тайных кассах» кассир украл 1 000 000 тенге."])
    r = CopilotService(cfg=cfg, corpus=corpus, llm=llm).answer(job_violation, "Что случилось?", [])
    assert r["answer"] == UNVERIFIED_ANSWER and r["answer"].startswith(NO_DATA)
    assert r["citations"] == [] and r["grounded"] is False and r["refusal"] is True


def test_empty_llm_output_is_honest(cfg, corpus, job_violation) -> None:
    r = CopilotService(cfg=cfg, corpus=corpus, llm=ScriptedLLM([""])).answer(job_violation, "Почему?", [])
    assert r["answer"].startswith(NO_DATA) and r["error"] is None


def test_stream_never_emits_unverified_text(cfg, corpus, job_violation) -> None:
    svc = CopilotService(cfg=cfg, corpus=corpus, llm=ScriptedLLM([MIXED], delay_s=0.001))
    events = list(svc.stream(job_violation, "Почему?", []))
    streamed = "".join(e["text"] for e in events if e["event"] == "token")
    for bad in ("999", "500 000", "95 с", "украл", "V0000000", "[N"):
        assert bad not in streamed, bad
    done = events[-1]
    assert done["event"] == "done" and streamed == done["answer"]


def test_stream_all_unverified_emits_only_honest_text(cfg, corpus, job_violation) -> None:
    llm = ScriptedLLM(["Кассир украл 1 000 000 тенге согласно п. 999."])
    events = list(CopilotService(cfg=cfg, corpus=corpus, llm=llm).stream(job_violation, "Что?", []))
    streamed = "".join(e["text"] for e in events if e["event"] == "token")
    assert streamed == UNVERIFIED_ANSWER and events[-1]["answer"] == UNVERIFIED_ANSWER


# ----------------------------------------------------------------------------- нехватка данных

def test_time_without_events_is_no_data(cfg, corpus, job_violation, job_clean) -> None:
    svc = CopilotService(cfg=cfg, corpus=corpus)
    for job in (job_violation, job_clean):
        r = svc.answer(job, "Что происходит на 35 секунде?", [])
        assert r["answer"].startswith(NO_DATA) and "00:35" in r["answer"], r["answer"]
        assert r["grounded"] is True and r["citations"] == []


def test_time_with_event_answers_from_result(cfg, corpus, job_violation) -> None:
    r = CopilotService(cfg=cfg, corpus=corpus).answer(job_violation, "Что на 47 секунде?", [])
    assert "(№2, 00:48)" in r["answer"] or "(№2, 00:47)" in r["answer"]
    assert not r["answer"].startswith(NO_DATA)


def test_no_norms_available_is_no_data(cfg, job_violation) -> None:
    empty = build_corpus({"version": 1, "types": {}}, [], cfg=cfg)
    r = CopilotService(cfg=cfg, corpus=empty).answer(job_violation, "Какой пункт нарушен?", [])
    assert r["answer"].startswith(NO_DATA) and r["citations"] == []


def test_unobservable_is_no_data(cfg, corpus, job_violation) -> None:
    r = CopilotService(cfg=cfg, corpus=corpus).answer(job_violation, "Сколько денег было в сумке?", [])
    assert r["answer"].startswith(NO_DATA) and r["refusal"] is True and r["citations"] == []


def test_failed_job_is_no_data(cfg, corpus) -> None:
    r = CopilotService(cfg=cfg, corpus=corpus).answer({"job_id": "f", "status": "failed"}, "Что нашли?", [])
    assert r["answer"].startswith(NO_DATA)


# ----------------------------------------------------------------------------- провайдер результатов

def test_router_with_test_job_provider_variants(cfg, corpus, job_violation) -> None:
    from copilot._contract import AnalysisResult, EvidenceFrame, Violation

    dc = AnalysisResult(job_id="dc", status="done", verdict="violation", violations=[
        Violation(type="Использование мобильного телефона при работе с ценностями", timecode_sec=12.0,
                  description="Кассир держит телефон", evidence_frames=[EvidenceFrame(12.0, "QQ==")], type_id="V05")])
    store = {"dict": {**job_violation, "job_id": "dict"}, "dc": dc,
             "json": json.dumps({**job_violation, "job_id": "json"}, ensure_ascii=False)}

    def provider(job_id: str):
        if job_id == "explode":
            raise ConnectionError("db down")
        return store.get(job_id)

    client = TestClient(_provider_app(cfg, corpus, provider))
    for jid in ("dict", "dc", "json"):
        r = client.post(f"/api/v1/jobs/{jid}/ask", json={"question": "Когда?"})
        assert r.status_code == 200, (jid, r.text)
        assert r.json()["answer"] and r.json()["error"] is None
    assert client.post("/api/v1/jobs/missing/ask", json={"question": "Когда?"}).status_code == 404
    r = client.post("/api/v1/jobs/explode/ask", json={"question": "Когда?"})
    assert r.status_code == 503 and r.json()["detail"]["code"] == "job_provider_error"
    assert client.get("/api/v1/jobs/explode/report").status_code == 503


def _provider_app(cfg, corpus, provider) -> FastAPI:
    app = FastAPI()
    app.include_router(make_router(get_job=provider, service=CopilotService(cfg=cfg, corpus=corpus)))
    return app


def test_malformed_result_does_not_crash(cfg, corpus) -> None:
    from copilot.report import report

    bad = {"job_id": "bad", "status": "done", "verdict": "violation", "meta": {"duration": "n/a"},
           "violations": [{"type": None, "type_id": None, "timecode_sec": "abc", "confidence": "high",
                           "t_start": None, "t_end": "?", "description": 42,
                           "evidence_frames": [{"timestamp_sec": "x"}, "junk"]}, "junk"]}
    svc = CopilotService(cfg=cfg, corpus=corpus)
    for q in ("Почему?", "Когда?", "Насколько уверен?", "Что проверить?", "Какие кадры?"):
        r = svc.answer(bad, q, [])
        assert r["error"] is None, (q, r)
    md = report(bad, corpus=corpus, cfg=cfg)
    assert md.startswith("# Заключение для проверяющего")
    r = svc.answer("not json at all", "Почему?", [])
    assert r["error"] is None


# ----------------------------------------------------------------------------- совместимость с A02/A15

def test_a15_request_shape_history_with_text_and_sources(cfg, corpus, job_violation) -> None:
    """A15 шлёт history=[{role, text}] и читает answer + sources (строки)."""
    client = TestClient(_provider_app(cfg, corpus, {"job-test-1": job_violation}.get))
    body = {"question": "А какая норма?", "history": [{"role": "user", "text": "Что с телефоном?"},
                                                     {"role": "assistant", "text": "Кассир держит телефон у уха."}]}
    r = client.post("/api/v1/jobs/job-test-1/ask", json=body)
    assert r.status_code == 200, r.text
    j = r.json()
    assert isinstance(j["answer"], str) and j["answer"]
    assert j["sources"] and all(isinstance(s, str) for s in j["sources"])
    assert j["sources"][0].startswith("[1] ") and "п. 9" in j["sources"][0]
    assert len(j["sources"]) == len(j["citations"])


def test_a02_style_double_mount_api_and_ui(cfg, corpus, job_violation) -> None:
    """A02 монтирует make_router дважды: /api/v1 (X-API-Key) и /ui/v1 (cookie+CSRF)."""
    from fastapi import Depends, Header, HTTPException

    def api_key(x_api_key: str = Header(default="")) -> None:
        if x_api_key != "k":
            raise HTTPException(401)

    def ui_csrf(x_csrf_token: str = Header(default="")) -> None:
        if x_csrf_token != "t":
            raise HTTPException(403)

    svc = CopilotService(cfg=cfg, corpus=corpus)
    jobs = {"job-test-1": job_violation, "run": {"job_id": "run", "status": "processing", "stage": "detect"}}
    app = FastAPI()
    app.include_router(make_router(get_job=jobs.get, service=svc, prefix="/api/v1", dependencies=[Depends(api_key)]))
    app.include_router(make_router(get_job=jobs.get, service=svc, prefix="/ui/v1", dependencies=[Depends(ui_csrf)]))
    c = TestClient(app)
    q = {"question": "Почему?"}
    assert c.post("/api/v1/jobs/job-test-1/ask", json=q).status_code == 401
    assert c.post("/ui/v1/jobs/job-test-1/ask", json=q).status_code == 403
    assert c.post("/api/v1/jobs/job-test-1/ask", json=q, headers={"X-API-Key": "k"}).status_code == 200
    assert c.post("/ui/v1/jobs/job-test-1/ask", json=q, headers={"X-CSRF-Token": "t"}).status_code == 200
    assert c.get("/ui/v1/copilot/status", headers={"X-CSRF-Token": "t"}).json()["available"] is True
    assert c.post("/api/v1/jobs/run/ask", json=q, headers={"X-API-Key": "k"}).status_code == 409


def test_openapi_builds_with_double_mount() -> None:
    """A02 отдаёт /docs: схема должна собираться без дублей operationId при двух префиксах."""
    import warnings

    app = FastAPI()
    app.include_router(make_router(get_job=lambda j: None, prefix="/api/v1"))
    app.include_router(make_router(get_job=lambda j: None, prefix="/ui/v1"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec = app.openapi()
    assert not [w for w in caught if "operation" in str(w.message).lower()]
    ops = [op["operationId"] for p in spec["paths"].values() for op in p.values()]
    assert len(ops) == len(set(ops)) == 8  # status, ask, report, suggestions × 2 префикса
