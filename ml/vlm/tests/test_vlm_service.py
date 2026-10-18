import threading
import time

import pytest

from ml.vlm import gpu, runtime, service
from ml.vlm.backends.mock import MockBackend


@pytest.fixture(autouse=True)
def _fresh_service(monkeypatch):
    monkeypatch.setattr(service, "_service", None)
    gpu.set_external_lock(None)
    yield
    gpu.set_external_lock(None)


def test_priority_lock_serves_analysis_before_chat():
    lk = gpu.PriorityLock()
    order = []
    lk.acquire(gpu.PRIO_ANALYSIS)  # держит «текущий» вызов

    def waiter(prio, name):
        lk.acquire(prio)
        order.append(name)
        lk.release()

    t_chat = threading.Thread(target=waiter, args=(gpu.PRIO_CHAT, "chat"))
    t_chat.start()
    time.sleep(0.05)  # чат встал в очередь раньше анализа...
    t_an = threading.Thread(target=waiter, args=(gpu.PRIO_ANALYSIS, "analysis"))
    t_an.start()
    time.sleep(0.05)
    assert lk.analysis_waiting()
    lk.release()
    t_chat.join(2)
    t_an.join(2)
    assert order == ["analysis", "chat"]  # ...но анализ получает GPU первым


def test_priority_lock_timeout_and_reentrancy():
    lk = gpu.PriorityLock()
    assert lk.acquire(0) and lk.acquire(0)  # реентерабельность в одном потоке
    res = []
    th = threading.Thread(target=lambda: res.append(lk.acquire(gpu.PRIO_CHAT, timeout=0.05)))
    th.start()
    th.join(1)
    assert res == [False] and lk.stats()["waiting"] == {}
    lk.release()
    lk.release()
    with pytest.raises(RuntimeError):
        lk.release()


def test_chat_with_mock_backend():
    runtime.set_backend(MockBackend())
    r = service.chat([{"role": "system", "content": "Ты помощник."}, {"role": "user", "content": "Привет"}])
    assert r.ok and r.text.startswith("[mock]") and r.finish_reason == "eos" and not r.truncated
    assert service.stats()["counters"]["chat_ok"] == 1


def test_chat_validation_and_unavailable(monkeypatch):
    runtime.set_backend(MockBackend())
    assert service.chat([]).error.startswith("invalid")
    assert service.chat([{"role": "robot", "content": "x"}]).error.startswith("invalid")
    assert service.chat([{"role": "system", "content": "x"}]).error.startswith("invalid")  # нет user
    runtime.set_backend(None)
    monkeypatch.setenv("SENTINEL_MOCK", "0")
    monkeypatch.setenv("SENTINEL_VLM_BACKEND", "off")
    r = service.chat([{"role": "user", "content": "x"}])
    assert not r.ok and r.error.startswith("unavailable")  # production не подменяет модель mock'ом


def test_chat_busy_while_analysis_holds_gpu():
    runtime.set_backend(MockBackend())
    entered = threading.Event()
    release = threading.Event()

    def analysis():
        with gpu.guard(gpu.PRIO_ANALYSIS):
            entered.set()
            release.wait(2)

    th = threading.Thread(target=analysis)
    th.start()
    entered.wait(1)
    r = service.chat([{"role": "user", "content": "Как дела?"}], wait_s=0.05)
    release.set()
    th.join(2)
    assert not r.ok and r.error == "busy"
    assert service.stats()["counters"]["chat_busy"] == 1


def test_chat_respects_external_lock_timeout():
    runtime.set_backend(MockBackend())
    ext = threading.Lock()
    gpu.set_external_lock(ext)
    ext.acquire()  # A03 держит GPU (детектор)
    try:
        r = service.chat([{"role": "user", "content": "x"}], wait_s=0.05)
    finally:
        ext.release()
    assert r.error == "busy"
    assert service.chat([{"role": "user", "content": "x"}], wait_s=0.5).ok


def test_chat_request_json_roundtrip():
    runtime.set_backend(MockBackend(responses=["<think>скрыто</think>Ответ"]))
    out = service.chat_request({"messages": [{"role": "user", "content": "q"}], "max_new_tokens": 10_000})
    assert out["ok"] and out["text"] == "Ответ"
    assert service.chat_request("bad")["error"].startswith("invalid")


def test_analysis_uses_same_backend_instance(clip):
    from ml.vlm._contract import Event
    import ml.vlm as vlm
    be = MockBackend(mode="confirm")
    runtime.set_backend(be)
    vlm.verify(clip, [Event("V05", 1, 2, 1.5, 0.5, "rules")], 30.0)
    service.chat([{"role": "user", "content": "x"}])
    kinds = [c.get("kind", "generate") for c in be.calls]
    assert "chat" in kinds and "generate" in kinds  # один экземпляр модели на процесс


def _a14_call(fn, messages, *, max_tokens=64, temperature=0.2, remaining=5.0, wait_s=10.0):
    """Повтор соглашения A14 (copilot/llm.py VLMServiceBackend.stream_chat @2055fc4): параметры по сигнатуре."""
    import inspect
    params = inspect.signature(fn).parameters
    varkw = any(p.kind == p.VAR_KEYWORD for p in params.values())
    accepts = lambda n: varkw or n in params
    kwargs = {"messages": list(messages)} if accepts("messages") else {}
    for name, value in (("max_new_tokens", int(max_tokens)), ("timeout_s", remaining),
                        ("temperature", float(temperature)), ("priority", "low"),
                        ("wait_s", min(remaining, wait_s))):
        if accepts(name):
            kwargs[name] = value
    res = fn(**kwargs)
    return (str(getattr(res, "text", "") or ""), bool(getattr(res, "timed_out", False)),
            str(getattr(res, "error", "") or ""))


def test_a14_calling_convention_and_priority_kwarg():
    runtime.set_backend(MockBackend())
    text, timed_out, error = _a14_call(service.chat, [{"role": "user", "content": "Вопрос"}])
    assert text.startswith("[mock]") and not timed_out and error == ""  # priority="low" не роняет вызов


def test_a14_sees_timeout_and_busy():
    from ml.vlm.budget import FakeClock
    runtime.set_backend(MockBackend(latency_s=5.0, clock=FakeClock()))
    text, timed_out, error = _a14_call(service.chat, [{"role": "user", "content": "x"}], remaining=1.0)
    assert text == "" and timed_out and error == "timeout"  # A14 -> LLMTimeout
    runtime.set_backend(MockBackend())
    entered, release = threading.Event(), threading.Event()

    def analysis():
        with gpu.guard(gpu.PRIO_ANALYSIS):
            entered.set()
            release.wait(2)

    th = threading.Thread(target=analysis)
    th.start()
    entered.wait(1)
    _t, _to, error = _a14_call(service.chat, [{"role": "user", "content": "x"}], remaining=0.05, wait_s=0.05)
    release.set()
    th.join(2)
    assert "busy" in error  # A14 -> LLMBusy


def test_available_without_loading(monkeypatch):
    be = MockBackend()
    runtime.set_backend(be)
    assert service.available() == (True, "")
    runtime.set_backend(None)
    monkeypatch.setenv("SENTINEL_MOCK", "0")
    monkeypatch.setenv("SENTINEL_VLM_BACKEND", "off")
    ok, why = service.available()
    assert not ok and why
