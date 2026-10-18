"""Граница с A11 (ml.vlm.service): реальный контракт ChatResult, без второй загрузки VLM, приоритет анализа.

* Синтетический двойник повторяет НАСТОЯЩИЙ API A11 (module.chat(messages, **kw) -> VLMService.chat(messages, *,
  images, max_new_tokens, timeout_s, wait_s, temperature) -> ChatResult(ok, text, error, truncated)).
* Тесты с настоящим ml.vlm.service (A11) запускаются, только если модуль импортируется (ветка интеграции L2).
"""
from __future__ import annotations

import importlib
import sys
import threading
import time
import types
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

from copilot.config import load_config
from copilot.llm import VLMServiceBackend, make_llm
from copilot.service import NOTICE_BUSY, NOTICE_NO_MODEL, CopilotService


@dataclass
class ChatResult:  # как ml.vlm.service.ChatResult
    ok: bool
    text: str = ""
    error: str = ""
    finish_reason: str = ""
    truncated: bool = False
    timed_out: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


class _FakeBackend:
    name = "fake"

    def __init__(self, loaded: bool = True, available: bool = True) -> None:
        self.loaded = loaded
        self._available = available

    def available(self) -> tuple[bool, str]:
        return (self._available, "" if self._available else "нет весов")


class _FakeService:
    """Повторяет подпись VLMService.chat у A11 (keyword-only, без priority)."""

    def __init__(self, backend: Optional[_FakeBackend], reply: Any) -> None:
        self._be = backend
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def backend(self) -> tuple[Optional[_FakeBackend], str]:
        return self._be, "" if self._be else "no weights"

    def chat(self, messages: list[dict[str, Any]], *, images: Optional[list[Any]] = None,
             max_new_tokens: Optional[int] = None, timeout_s: Optional[float] = None,
             wait_s: Optional[float] = None, temperature: float = 0.0) -> ChatResult:
        self.calls.append({"messages": messages, "max_new_tokens": max_new_tokens, "timeout_s": timeout_s,
                           "wait_s": wait_s, "chars": sum(len(m["content"]) for m in messages)})
        return self.reply(messages) if callable(self.reply) else self.reply


def _install(monkeypatch, svc: _FakeService) -> types.ModuleType:
    mod = types.ModuleType("ml.vlm.service")
    mod.get_service = lambda: svc
    mod.chat = lambda messages, **kw: svc.chat(messages, **kw)   # как у A11: **kw
    monkeypatch.setitem(sys.modules, "ml.vlm.service", mod)
    return mod


@pytest.fixture()
def vcfg(cfg):
    return load_config({**cfg, "llm": {**cfg["llm"], "backend": "vlm_service", "timeout_s": 30}})


GOOD = ChatResult(ok=True, text="Кассир держит телефон у уха [V1]. Пользоваться телефоном запрещено [N1].")


def test_real_signature_no_priority_kwarg_and_bounded_time(monkeypatch, vcfg, corpus, job_violation) -> None:
    svc = _FakeService(_FakeBackend(), GOOD)
    _install(monkeypatch, svc)
    r = CopilotService(cfg=vcfg, corpus=corpus).answer(job_violation, "Почему?", [])
    assert r["error"] is None and r["mode"] == "generative" and r["citations"]
    call = svc.calls[0]
    assert set(call) == {"messages", "max_new_tokens", "timeout_s", "wait_s", "chars"}   # нет priority
    assert call["wait_s"] + call["timeout_s"] <= vcfg["llm"]["timeout_s"]               # ожидание + генерация ≤ бюджета


def test_chatresult_unavailable_is_retrieval_not_500(monkeypatch, vcfg, corpus, job_violation) -> None:
    _install(monkeypatch, _FakeService(_FakeBackend(), ChatResult(ok=False, error="unavailable: model not warm")))
    r = CopilotService(cfg=vcfg, corpus=corpus).answer(job_violation, "Почему?", [])
    assert r["error"] is None and r["mode"] == "retrieval" and r["notice"] == NOTICE_NO_MODEL


def test_chatresult_busy_and_timeout(monkeypatch, vcfg, corpus, job_violation) -> None:
    _install(monkeypatch, _FakeService(_FakeBackend(), ChatResult(ok=False, error="busy")))
    r = CopilotService(cfg=vcfg, corpus=corpus).answer(job_violation, "Почему?", [])
    assert r["mode"] == "retrieval" and r["notice"] == NOTICE_BUSY
    _install(monkeypatch, _FakeService(_FakeBackend(), ChatResult(ok=False, error="timeout")))
    r2 = CopilotService(cfg=vcfg, corpus=corpus).answer(job_violation, "Почему?", [])
    assert r2["error"] == "timeout" and r2["answer"] == ""


def test_model_not_loaded_here_is_never_cold_loaded(monkeypatch, vcfg, corpus, job_violation) -> None:
    """В проде VLM живёт в процессе-воркере A03: копилот в процессе API не грузит вторую копию."""
    svc = _FakeService(_FakeBackend(loaded=False), GOOD)
    _install(monkeypatch, svc)
    warnings: list[str] = []
    assert make_llm(vcfg, warnings) is None and any("не загружена" in w for w in warnings)
    copilot = CopilotService(cfg=vcfg, corpus=corpus)
    st = copilot.status()
    assert st["available"] is True and st["generative"] is False and st["mode"] == "retrieval"
    r = copilot.answer(job_violation, "Почему?", [])
    assert r["mode"] == "retrieval" and not svc.calls


def test_model_unloaded_between_questions(monkeypatch, vcfg, corpus, job_violation) -> None:
    be = _FakeBackend(loaded=True)
    svc = _FakeService(be, GOOD)
    _install(monkeypatch, svc)
    copilot = CopilotService(cfg=vcfg, corpus=corpus)
    assert copilot.answer(job_violation, "Почему?", [])["mode"] == "generative"
    be.loaded = False  # A03 выгрузил VLM
    r = copilot.answer(job_violation, "Почему?", [])
    assert r["mode"] == "retrieval" and r["notice"] == NOTICE_NO_MODEL and len(svc.calls) == 1


def test_no_backend_or_unavailable_backend(monkeypatch, vcfg) -> None:
    _install(monkeypatch, _FakeService(None, GOOD))
    assert make_llm(vcfg) is None
    _install(monkeypatch, _FakeService(_FakeBackend(available=False), GOOD))
    w: list[str] = []
    assert make_llm(vcfg, w) is None and any("нет весов" in x for x in w)


def test_prompt_fits_a11_input_limit(monkeypatch, vcfg, corpus, job_violation) -> None:
    svc = _FakeService(_FakeBackend(), GOOD)
    _install(monkeypatch, svc)
    cfg2 = load_config({**vcfg, "vlm_service": {**vcfg["vlm_service"], "max_input_chars": 5000}})
    hist = [{"role": "user" if i % 2 == 0 else "assistant", "content": "длинная реплика " * 60} for i in range(8)]
    r = CopilotService(cfg=cfg2, corpus=corpus).answer(job_violation, "Почему?", hist)
    assert r["error"] is None and svc.calls[0]["chars"] <= 5000


def test_truncated_answer_keeps_only_complete_sentences(monkeypatch, vcfg, corpus, job_violation) -> None:
    _install(monkeypatch, _FakeService(_FakeBackend(), ChatResult(
        ok=True, truncated=True, text="Кассир держит телефон у уха [V1]. Пользоваться телефоном при работе с")))
    r = CopilotService(cfg=vcfg, corpus=corpus).answer(job_violation, "Почему?", [])
    assert r["answer"].endswith("(№1, 00:22).") and "при работе с" not in r["answer"]


def test_timed_out_partial_text_keeps_only_complete_sentences(monkeypatch, vcfg, corpus, job_violation) -> None:
    """A11 отдаёт ok=True + timed_out=True, если упёрлись во время: хвост оборван — его не показываем."""
    _install(monkeypatch, _FakeService(_FakeBackend(), ChatResult(
        ok=True, timed_out=True, text="Кассир держит телефон у уха [V1]. Пользоваться телефоном при работе с")))
    r = CopilotService(cfg=vcfg, corpus=corpus).answer(job_violation, "Почему?", [])
    assert r["answer"].endswith("(№1, 00:22).") and "при работе с" not in r["answer"]


def test_explicit_signature_with_priority_param(monkeypatch, vcfg, corpus, job_violation) -> None:
    """Новая подпись A11: chat(messages, *, …, priority="low") — priority передаём, бюджет тот же."""
    seen: dict[str, Any] = {}

    def chat(messages, *, images=None, max_new_tokens=None, timeout_s=None, wait_s=None, temperature=0.0,
             priority="low"):
        seen.update(priority=priority, timeout_s=timeout_s, wait_s=wait_s)
        return GOOD

    mod = _install(monkeypatch, _FakeService(_FakeBackend(), GOOD))
    mod.chat = chat
    r = CopilotService(cfg=vcfg, corpus=corpus).answer(job_violation, "Почему?", [])
    assert r["mode"] == "generative" and seen["priority"] == "low"
    assert seen["wait_s"] + seen["timeout_s"] <= vcfg["llm"]["timeout_s"]


def _forbidden(*_a: Any, **_k: Any) -> Any:
    raise AssertionError("в процессе без модели A11 нельзя звать backend()/available()/chat()")


def test_api_process_without_a11_runtime_does_not_probe_model(monkeypatch, vcfg, corpus, job_violation) -> None:
    """Процесс API (A02): ml.vlm.runtime не импортирован — модели здесь нет. backend()/available() A11 не зовём:
    они импортируют torch/transformers и проверяют CUDA в процессе, где модели быть не должно."""
    mod = types.ModuleType("ml.vlm.service")
    mod.__package__, mod.__file__ = "ml.vlm", "/synthetic/ml/vlm/service.py"
    mod.get_service = lambda: types.SimpleNamespace(backend=_forbidden, chat=_forbidden)
    mod.chat, mod.available = _forbidden, _forbidden
    monkeypatch.setitem(sys.modules, "ml.vlm.service", mod)
    monkeypatch.delitem(sys.modules, "ml.vlm.runtime", raising=False)
    w: list[str] = []
    assert make_llm(vcfg, w) is None and any("не загружена в этом процессе" in x for x in w)
    copilot = CopilotService(cfg=vcfg, corpus=corpus)
    assert copilot.status()["mode"] == "retrieval"
    assert copilot.answer(job_violation, "Почему?", [])["notice"] == NOTICE_NO_MODEL


def test_set_chat_fn_payload_style_ipc(cfg, corpus, job_violation) -> None:
    """A03 может пробросить чат воркера: fn(payload) -> dict как у ml.vlm.service.chat_request."""
    seen = {}

    def chat_request(payload: dict[str, Any]) -> dict[str, Any]:
        seen.update(payload)
        return {"ok": True, "text": "Кассир держит телефон у уха [V1].", "error": "", "truncated": False}

    copilot = CopilotService(cfg=cfg, corpus=corpus)
    copilot.set_chat_fn(chat_request)
    r = copilot.answer(job_violation, "Что видно?", [])
    assert r["mode"] == "generative" and "телефон у уха" in r["answer"]
    assert set(seen) >= {"messages", "max_new_tokens", "timeout_s", "wait_s"} and "priority" not in seen
    copilot.set_chat_fn(lambda payload: {"ok": False, "error": "busy"})
    assert copilot.answer(job_violation, "Что видно?", [])["notice"] == NOTICE_BUSY


def test_backend_info_reports_where_model_lives(monkeypatch, vcfg) -> None:
    mod = _install(monkeypatch, _FakeService(_FakeBackend(loaded=True), GOOD))
    info = VLMServiceBackend(mod.chat, vcfg, mod).info()
    assert info["model_loaded_here"] is True


# ----------------------------------------------------------------------------- настоящий A11 (ветка интеграции)

def _real_a11(monkeypatch):
    monkeypatch.delitem(sys.modules, "ml.vlm.service", raising=False)
    try:
        service = importlib.import_module("ml.vlm.service")
        runtime = importlib.import_module("ml.vlm.runtime")
        gpu = importlib.import_module("ml.vlm.gpu")
        base = importlib.import_module("ml.vlm.backends.base")
    except Exception as exc:  # в ветке A14 модулей A11 нет
        pytest.skip(f"ml.vlm (A11) недоступен: {type(exc).__name__}")
    monkeypatch.setattr(service, "_service", None)
    return service, runtime, gpu, base


def _real_backend(base: Any, loaded: bool, text: str) -> Any:
    class Fake(base.VLMBackend):
        name = "fake-a14"
        model_id = "synthetic"

        def __init__(self) -> None:
            self._loaded = loaded
            self.chat_calls = 0
            self.load_calls = 0

        @property
        def loaded(self) -> bool:
            return self._loaded

        def load(self) -> None:
            self.load_calls += 1
            self._loaded = True

        def chat(self, messages, images=None, max_new_tokens=256, timeout_s=10.0, temperature=0.0, **kw):
            self.chat_calls += 1
            return base.GenResult(text=text, latency_s=0.01, finish_reason="eos")

    return Fake()


def test_real_a11_chat_generative(monkeypatch, vcfg, corpus, job_violation) -> None:
    service, runtime, gpu, base = _real_a11(monkeypatch)
    be = _real_backend(base, True, "Кассир держит телефон у уха [V1].")
    runtime.set_backend(be)
    try:
        r = CopilotService(cfg=vcfg, corpus=corpus).answer(job_violation, "Что видно?", [])
        assert r["error"] is None and r["mode"] == "generative" and be.chat_calls == 1
        assert "телефон у уха (№1, 00:22)" in r["answer"]
    finally:
        runtime.set_backend(None)


def test_real_a11_not_warm_is_not_loaded_by_copilot(monkeypatch, vcfg, corpus, job_violation) -> None:
    service, runtime, gpu, base = _real_a11(monkeypatch)
    be = _real_backend(base, False, "x")
    runtime.set_backend(be)
    try:
        r = CopilotService(cfg=vcfg, corpus=corpus).answer(job_violation, "Что видно?", [])
        assert r["mode"] == "retrieval" and be.load_calls == 0 and be.chat_calls == 0
    finally:
        runtime.set_backend(None)


def test_real_a11_analysis_has_priority(monkeypatch, vcfg, corpus, job_violation) -> None:
    service, runtime, gpu, base = _real_a11(monkeypatch)
    be = _real_backend(base, True, "Кассир держит телефон у уха [V1].")
    runtime.set_backend(be)
    cfg2 = load_config({**vcfg, "vlm_service": {**vcfg["vlm_service"], "wait_s": 0.3}})
    held, release = threading.Event(), threading.Event()

    def analysis() -> None:
        with gpu.guard(gpu.PRIO_ANALYSIS):
            held.set()
            release.wait(5)

    th = threading.Thread(target=analysis)
    th.start()
    held.wait(2)
    try:
        t0 = time.monotonic()
        r = CopilotService(cfg=cfg2, corpus=corpus).answer(job_violation, "Что видно?", [])
        assert r["notice"] == NOTICE_BUSY and be.chat_calls == 0 and time.monotonic() - t0 < 3
    finally:
        release.set()
        th.join(5)
        runtime.set_backend(None)


def test_real_a11_api_process_status_does_not_import_runtime(monkeypatch, vcfg, corpus, job_violation) -> None:
    """Настоящий A11, процесс без пайплайна (как процесс API A02): статус и ответ без импорта ml.vlm.runtime."""
    _real_a11(monkeypatch)
    monkeypatch.delitem(sys.modules, "ml.vlm.runtime")
    copilot = CopilotService(cfg=vcfg, corpus=corpus)
    assert copilot.status()["mode"] == "retrieval"
    assert copilot.answer(job_violation, "Почему?", [])["mode"] == "retrieval"
    assert "ml.vlm.runtime" not in sys.modules
