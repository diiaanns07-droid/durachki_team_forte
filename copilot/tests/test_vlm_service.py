"""Общая VLM через ml.vlm.service.chat (A11): низкий приоритет, честный негенеративный режим."""
from __future__ import annotations

import sys
import threading
import time
import types
from typing import Any

import pytest

from copilot.config import load_config
from copilot.llm import RetrievalBackend, VLMRuntimeAdapter, VLMServiceBackend, make_llm
from copilot.service import NOTICE_BUSY, NOTICE_NO_MODEL, CopilotService


class _Gen:
    def __init__(self, text: str, timed_out: bool = False, error: str = "") -> None:
        self.text, self.timed_out, self.error, self.latency_s = text, timed_out, error, 0.01


def _install(monkeypatch, chat: Any, available: Any = None) -> types.ModuleType:
    mod = types.ModuleType("ml.vlm.service")
    mod.chat = chat
    if available is not None:
        mod.available = available
    monkeypatch.setitem(sys.modules, "ml.vlm.service", mod)
    return mod


@pytest.fixture()
def vcfg(cfg):
    return load_config({**cfg, "llm": {**cfg["llm"], "backend": "auto"}})


def test_messages_signature_low_priority(monkeypatch, vcfg, corpus, job_violation) -> None:
    seen: dict[str, Any] = {}

    def chat(messages, max_new_tokens=256, timeout_s=10.0, priority="normal", temperature=0.0):
        seen.update(messages=messages, priority=priority, timeout_s=timeout_s, max_new_tokens=max_new_tokens)
        return _Gen("Кассир держит телефон у уха [V1]. Пользоваться телефоном запрещено [N1].")

    _install(monkeypatch, chat)
    svc = CopilotService(cfg=vcfg, corpus=corpus)
    r = svc.answer(job_violation, "Почему это нарушение?", [])
    assert r["backend"] == "vlm_service" and r["generative"] is True and r["mode"] == "generative"
    assert r["notice"] is None and not r["answer"].startswith("(")
    assert seen["priority"] == "low" and 0 < seen["timeout_s"] <= vcfg["llm"]["timeout_s"]
    assert seen["messages"][0]["role"] == "system" and "<<<ДАННЫЕ" in seen["messages"][0]["content"]
    assert r["citations"] and r["grounded"] is True


def test_system_prompt_signature_is_flattened(monkeypatch, vcfg, corpus, job_violation) -> None:
    seen: dict[str, Any] = {}

    def chat(images, system, prompt, max_tokens=128, timeout=5.0):
        seen.update(images=images, system=system, prompt=prompt, max_tokens=max_tokens)
        return "Кассир держит телефон у уха [V1]."

    _install(monkeypatch, chat)
    r = CopilotService(cfg=vcfg, corpus=corpus).answer(job_violation, "Что видно?", [])
    assert r["error"] is None and "телефон у уха" in r["answer"]
    assert seen["images"] == [] and "Проверяющий: Вопрос проверяющего: Что видно?" in seen["prompt"]
    assert seen["max_tokens"] == vcfg["llm"]["max_tokens"]


def test_service_not_available_means_retrieval_mode(monkeypatch, vcfg, corpus, job_violation) -> None:
    calls = []
    _install(monkeypatch, lambda **kw: calls.append(kw) or "x", available=lambda: (False, "нет весов"))
    warnings: list[str] = []
    assert make_llm(vcfg, warnings) is None and any("нет весов" in w for w in warnings)
    svc = CopilotService(cfg=vcfg, corpus=corpus)
    st = svc.status()
    assert st["available"] is True and st["generative"] is False and st["mode"] == "retrieval"
    assert st["message"] == NOTICE_NO_MODEL
    r = svc.answer(job_violation, "Почему это нарушение?", [])
    assert r["answer"].startswith(f"({NOTICE_NO_MODEL})") and r["mode"] == "retrieval" and not calls
    assert r["backend"] == "retrieval" and r["grounded"] is True


def test_no_module_at_all_is_retrieval(monkeypatch, vcfg, corpus, job_violation) -> None:
    monkeypatch.setitem(sys.modules, "ml.vlm.service", None)  # import -> ImportError
    r = CopilotService(cfg=vcfg, corpus=corpus).answer(job_violation, "Когда?", [])
    assert r["mode"] == "retrieval" and r["error"] is None and "00:22" in r["answer"]
    assert isinstance(CopilotService(cfg=vcfg, corpus=corpus).llm, RetrievalBackend)


def test_chat_timeout_is_504_not_fake(monkeypatch, vcfg, corpus, job_violation) -> None:
    def chat(messages, timeout_s=1.0, **kw):
        raise TimeoutError("vlm timeout")

    _install(monkeypatch, chat)
    r = CopilotService(cfg=vcfg, corpus=corpus).answer(job_violation, "Почему?", [])
    assert r["error"] == "timeout" and r["answer"] == ""


def test_service_busy_falls_back_to_retrieval(monkeypatch, vcfg, corpus, job_violation) -> None:
    _install(monkeypatch, lambda messages, **kw: _Gen("", error="GPU busy"))
    r = CopilotService(cfg=vcfg, corpus=corpus).answer(job_violation, "Почему?", [])
    assert r["mode"] == "retrieval" and r["notice"] == NOTICE_BUSY and r["answer"].startswith(f"({NOTICE_BUSY})")
    cfg_err = load_config({**vcfg, "llm": {**vcfg["llm"], "busy_policy": "error"}})
    assert CopilotService(cfg=cfg_err, corpus=corpus).answer(job_violation, "Почему?", [])["error"] == "busy"


def test_busy_probe_keeps_gpu_for_analysis(monkeypatch, vcfg, corpus, job_violation) -> None:
    calls = []
    _install(monkeypatch, lambda messages, **kw: calls.append(1) or _Gen("ok"))
    svc = CopilotService(cfg=vcfg, corpus=corpus)
    svc.busy_probe = lambda: True
    r = svc.answer(job_violation, "Почему?", [])
    assert not calls and r["notice"] == NOTICE_BUSY and r["generative"] is False
    events = list(svc.stream(job_violation, "Почему?", []))
    assert [e["event"] for e in events] == ["token", "done"] and events[0]["text"].startswith(f"({NOTICE_BUSY})")
    svc.busy_probe = lambda: False
    assert svc.answer(job_violation, "Почему?", [])["mode"] == "generative" and calls


def test_streaming_with_vlm_service(monkeypatch, vcfg, corpus, job_violation) -> None:
    _install(monkeypatch, lambda messages, **kw: _Gen("Кассир держит телефон у уха [V1]. Также п. 999 нарушен."))
    events = list(CopilotService(cfg=vcfg, corpus=corpus).stream(job_violation, "Почему?", []))
    text = "".join(e["text"] for e in events if e["event"] == "token")
    assert "999" not in text and events[-1]["event"] == "done" and events[-1]["mode"] == "generative"


# ----------------------------------------------------------------------------- временный адаптер к ml.vlm.runtime

def _install_runtime(monkeypatch, backend: Any, lock: Any) -> None:
    runtime = types.ModuleType("ml.vlm.runtime")
    runtime.get_backend = lambda cfg: (backend, "" if backend else "no backend")
    runtime.gpu_guard = lambda: lock
    vcfgmod = types.ModuleType("ml.vlm.config")
    vcfgmod.load_config = lambda: {}
    monkeypatch.setitem(sys.modules, "ml.vlm.runtime", runtime)
    monkeypatch.setitem(sys.modules, "ml.vlm.config", vcfgmod)
    monkeypatch.setitem(sys.modules, "ml.vlm.service", None)


class _WarmBackend:
    loaded = True

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, images, system, prompt, max_new_tokens=256, timeout_s=10.0, temperature=0.0):
        self.calls.append({"images": images, "system": system, "prompt": prompt})
        return _Gen("Кассир держит телефон у уха [V1].")


def test_runtime_adapter_is_off_by_default(monkeypatch, vcfg) -> None:
    _install_runtime(monkeypatch, _WarmBackend(), threading.Lock())
    assert make_llm(vcfg) is None


def test_runtime_adapter_uses_warm_vlm_and_yields_to_analysis(monkeypatch, vcfg, corpus, job_violation) -> None:
    be, lock = _WarmBackend(), threading.Lock()
    _install_runtime(monkeypatch, be, lock)
    cfg_ad = load_config({**vcfg, "vlm_service": {**vcfg["vlm_service"], "runtime_adapter": True, "wait_s": 0.2}})
    llm = make_llm(cfg_ad)
    assert isinstance(llm, VLMRuntimeAdapter)
    svc = CopilotService(cfg=cfg_ad, corpus=corpus)
    r = svc.answer(job_violation, "Что видно?", [])
    assert r["backend"] == "vlm_runtime" and be.calls and be.calls[0]["images"] == []
    lock.acquire()  # анализ видео держит GPU
    try:
        t0 = time.monotonic()
        r2 = CopilotService(cfg=cfg_ad, corpus=corpus).answer(job_violation, "Что видно?", [])
        assert r2["notice"] == NOTICE_BUSY and time.monotonic() - t0 < 2.0
    finally:
        lock.release()


def test_runtime_adapter_never_cold_loads(monkeypatch, vcfg) -> None:
    cold = _WarmBackend()
    cold.loaded = False
    _install_runtime(monkeypatch, cold, threading.Lock())
    cfg_ad = load_config({**vcfg, "vlm_service": {**vcfg["vlm_service"], "runtime_adapter": True}})
    warnings: list[str] = []
    assert make_llm(cfg_ad, warnings) is None and any("не загружена" in w for w in warnings)


def test_vlm_backend_kwargs_filtered_by_signature() -> None:
    seen = {}

    def chat(messages):  # минимальная подпись — лишние параметры не передаются
        seen["n"] = len(messages)
        return "ok"

    be = VLMServiceBackend(chat, load_config({}))
    assert "".join(be.stream_chat([{"role": "user", "content": "q"}], deadline=time.monotonic() + 5)) == "ok"
    assert seen["n"] == 1


def test_e5_uses_l1_offline_cache_dir(monkeypatch, tmp_path) -> None:
    from copilot.embed import E5Embedder

    hub_dir = tmp_path / "hub"
    hub_dir.mkdir()
    seen = {}
    hub = types.ModuleType("huggingface_hub")

    def snapshot_download(repo_id, local_files_only=False, cache_dir=None):
        seen.update(local_files_only=local_files_only, cache_dir=cache_dir)
        return str(tmp_path)

    hub.snapshot_download = snapshot_download
    st_mod = types.ModuleType("sentence_transformers")
    st_mod.SentenceTransformer = lambda path, device="cpu": object()
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "sentence_transformers", st_mod)
    E5Embedder("intfloat/multilingual-e5-small", hf_cache_dir=str(hub_dir))
    assert seen == {"local_files_only": True, "cache_dir": str(hub_dir)}
