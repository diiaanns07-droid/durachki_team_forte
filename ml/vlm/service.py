"""ml.vlm.service — одна модель на процесс: анализ (verify/discover) и чат копилота A14 на одном экземпляре.

    from ml.vlm import service
    r = service.chat([{"role": "system", "content": "..."}, {"role": "user", "content": "..."}])
    r.ok, r.text, r.truncated, r.error   # error: unavailable | busy | invalid | timeout | <тип исключения>

* Модель — та же, что у анализа (runtime.get_backend): второй экземпляр не грузится.
* Приоритет: ожидающий анализ всегда получает GPU раньше чата (ml.vlm.gpu.PriorityLock). Чат ждёт не дольше
  service.chat.wait_s, иначе честно отвечает busy; токены/время чата ограничены — анализ не вытесняется надолго.
* Production никогда не подменяет модель mock'ом: нет модели -> ok=False, error="unavailable".
* chat_request(payload) — JSON-вход/выход для вызова через IPC (например, из воркера A03), если копилот
  живёт в другом процессе: так на машине остаётся одна загруженная VLM.
* Тексты диалога не логируются; stats() — только счётчики и задержки.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

log = logging.getLogger(__name__)

_ROLES = {"system", "user", "assistant"}


@dataclass
class ChatResult:
    ok: bool
    text: str = ""
    error: str = ""
    finish_reason: str = ""
    truncated: bool = False
    timed_out: bool = False        # упёрлись в timeout_s (A14 переводит в LLMTimeout)
    latency_s: float = 0.0
    wait_s: float = 0.0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    meta: dict[str, Any] = field(default_factory=dict)


class VLMService:
    def __init__(self, cfg: Optional[dict[str, Any]] = None) -> None:
        self._cfg = cfg
        self._lock = threading.Lock()
        self.counters: dict[str, int] = {"chat_ok": 0, "chat_busy": 0, "chat_unavailable": 0, "chat_invalid": 0,
                                         "chat_error": 0, "chat_truncated": 0}
        self.latencies: list[float] = []

    # ------------------------------------------------------------------ конфиг и модель
    def cfg(self) -> dict[str, Any]:
        if self._cfg is None:
            from .config import load_config

            return load_config()
        return self._cfg

    def backend(self) -> tuple[Any, str]:
        from .runtime import get_backend

        return get_backend(self.cfg())

    def warmup(self) -> bool:
        from .runtime import ensure_loaded

        be, _ = self.backend()
        return bool(be is not None and ensure_loaded(be)[0])

    # ------------------------------------------------------------------ чат
    def _validate(self, messages: Any, max_chars: int) -> tuple[Optional[list[dict[str, Any]]], str]:
        if not isinstance(messages, list) or not messages:
            return None, "messages must be a non-empty list"
        out = []
        total = 0
        for m in messages:
            if not isinstance(m, dict) or str(m.get("role")) not in _ROLES:
                return None, "each message needs role in system|user|assistant"
            content = m.get("content", "")
            if not isinstance(content, str):
                return None, "content must be a string"
            total += len(content)
            out.append({"role": str(m["role"]), "content": content})
        if not any(m["role"] == "user" for m in out):
            return None, "no user message"
        if total > max_chars:
            return None, f"messages too long ({total} > {max_chars} chars)"
        return out, ""

    def chat(self, messages: list[dict[str, Any]], *, images: Optional[list[Any]] = None,
             max_new_tokens: Optional[int] = None, timeout_s: Optional[float] = None,
             wait_s: Optional[float] = None, temperature: float = 0.0, priority: Optional[str] = "low"
             ) -> ChatResult:
        """priority принимается для совместимости с A14 (priority="low"); чат ВСЕГДА ниже анализа,
        поднять его над verify/discover нельзя."""
        from . import gpu
        from .runtime import ensure_loaded

        ccfg = (self.cfg().get("service") or {}).get("chat") or {}
        if not ccfg.get("enabled", True):
            return self._count(ChatResult(ok=False, error="unavailable: chat disabled in config"), "chat_unavailable")
        msgs, why = self._validate(messages, int(ccfg.get("max_input_chars", 12000)))
        if msgs is None:
            return self._count(ChatResult(ok=False, error=f"invalid: {why}"), "chat_invalid")
        mnt = min(int(max_new_tokens or ccfg.get("max_new_tokens", 384)), int(ccfg.get("max_new_tokens_cap", 768)))
        tmo = min(float(timeout_s or ccfg.get("timeout_s", 20.0)), float(ccfg.get("timeout_cap_s", 45.0)))
        wait = float(ccfg.get("wait_s", 10.0) if wait_s is None else wait_s)
        be, reason = self.backend()
        if be is None:
            return self._count(ChatResult(ok=False, error=f"unavailable: {reason[:160]}"), "chat_unavailable")
        if not be.loaded and not ccfg.get("allow_cold_load", True):
            return self._count(ChatResult(ok=False, error="unavailable: model not warm"), "chat_unavailable")
        t0 = time.monotonic()
        try:
            with gpu.guard(gpu.PRIO_CHAT, timeout=wait):
                waited = time.monotonic() - t0
                ok, why = ensure_loaded(be)
                if not ok:
                    return self._count(ChatResult(ok=False, error=f"unavailable: {why[:160]}", wait_s=waited),
                                       "chat_unavailable")
                res = be.chat(msgs, images=images, max_new_tokens=mnt, timeout_s=tmo, temperature=temperature)
        except TimeoutError:
            return self._count(ChatResult(ok=False, error="busy", wait_s=round(time.monotonic() - t0, 3)),
                               "chat_busy")
        except Exception as e:  # noqa: BLE001 — копилот не должен ронять процесс анализа
            log.warning("vlm chat failed: %s", type(e).__name__)
            return self._count(ChatResult(ok=False, error=type(e).__name__), "chat_error")
        timed_out = bool(res.timed_out or res.finish_reason == "time" or res.error == "timeout")
        if res.error and not res.text:
            return self._count(ChatResult(ok=False, error=res.error, wait_s=round(waited, 3), timed_out=timed_out,
                                          latency_s=round(res.latency_s, 3)), "chat_error")
        from .parsing import _THINK_RE  # убираем возможные <think>…</think>

        text = _THINK_RE.sub("", res.text or "").strip()
        out = ChatResult(ok=True, text=text, finish_reason=res.finish_reason, truncated=res.truncated,
                         timed_out=timed_out, latency_s=round(res.latency_s, 3), wait_s=round(waited, 3),
                         prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens,
                         meta={"model": be.info().get("model"), "backend": be.name})
        if out.truncated:
            self.counters["chat_truncated"] += 1
        self.latencies.append(out.latency_s)
        return self._count(out, "chat_ok")

    def _count(self, r: ChatResult, key: str) -> ChatResult:
        with self._lock:
            self.counters[key] = self.counters.get(key, 0) + 1
        return r

    def stats(self) -> dict[str, Any]:
        from . import gpu

        lat = sorted(self.latencies)
        return {"counters": dict(self.counters), "gpu": gpu.priority_lock().stats(),
                "chat_latency_p50_s": lat[len(lat) // 2] if lat else None}


_service: Optional[VLMService] = None
_service_lock = threading.Lock()


def get_service() -> VLMService:
    """Один сервис (и одна модель) на процесс."""
    global _service
    with _service_lock:
        if _service is None:
            _service = VLMService()
        return _service


def chat(messages: list[dict[str, Any]], *, images: Optional[list[Any]] = None,
         max_new_tokens: Optional[int] = None, timeout_s: Optional[float] = None, wait_s: Optional[float] = None,
         temperature: float = 0.0, priority: Optional[str] = "low") -> ChatResult:
    """Явная сигнатура: A14 (copilot/llm.py VLMServiceBackend) передаёт только принимаемые параметры."""
    return get_service().chat(messages, images=images, max_new_tokens=max_new_tokens, timeout_s=timeout_s,
                              wait_s=wait_s, temperature=temperature, priority=priority)


def available() -> tuple[bool, str]:
    """Можно ли звать chat, БЕЗ загрузки модели (A14 проверяет перед подключением)."""
    svc = get_service()
    ccfg = (svc.cfg().get("service") or {}).get("chat") or {}
    if not ccfg.get("enabled", True):
        return False, "chat disabled in config"
    be, reason = svc.backend()
    if be is None:
        return False, reason[:160]
    if not be.loaded and not ccfg.get("allow_cold_load", True):
        return False, "model not warm"
    return True, ""


def chat_request(payload: dict[str, Any]) -> dict[str, Any]:
    """JSON-совместимый вход/выход для IPC: {"messages": [...], "max_new_tokens"?, "timeout_s"?, "wait_s"?}."""
    if not isinstance(payload, dict):
        return asdict(ChatResult(ok=False, error="invalid: payload must be an object"))
    kw = {k: payload[k] for k in ("max_new_tokens", "timeout_s", "wait_s") if k in payload}
    return asdict(get_service().chat(payload.get("messages"), **kw))


def stats() -> dict[str, Any]:
    return get_service().stats()
