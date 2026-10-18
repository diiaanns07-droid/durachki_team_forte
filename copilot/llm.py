"""LLM-бэкенды Sentinel Copilot. Всё локально: никаких облачных API (правило 1).

* ``mock`` (SENTINEL_MOCK=1) / ``extractive`` (только явно в конфиге) — без модели: детерминированный
  ответ из фактов результата и норм. НЕ подставляется автоматически вместо LLM: если модели нет,
  ассистент честно сообщает «недоступен» (HTTP 501).
* ``vlm_service`` — общая Qwen3-VL-4B через ``ml.vlm.service.chat`` (A11), низкий приоритет (по умолчанию).
* ``retrieval`` — негенеративный режим, когда модели нет или GPU занят анализом (не mock, честно помечен).
* ``llama_cpp`` — llama-cpp-python (MIT) в процессе, GGUF Q4_K_M, загрузка по требованию,
  частичный offload на GPU по свободной VRAM, выгрузка по простою (VRAM нужна VLM).
* ``llama_server`` — локальный llama.cpp server (OpenAI-совместимый /v1/chat/completions),
  разрешены только loopback/частные адреса.
* ``scripted`` — заранее заданные ответы (для тестов и оценки пост-проверки).
"""
from __future__ import annotations

import gc
import importlib.util
import ipaddress
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence
from urllib.parse import urlparse

from .config import default_config, is_mock, resolve_path

log = logging.getLogger(__name__)

Message = dict[str, str]


class LLMTimeout(RuntimeError):
    """Генерация не уложилась в бюджет."""


class LLMBusy(RuntimeError):
    """Модель/GPU заняты и не освободились до дедлайна."""


# ----------------------------------------------------------------------------- think-фильтр

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


def strip_think(text: str) -> str:
    """Убрать рассуждения Qwen3 (<think>…</think>), в т.ч. незакрытый блок."""
    text = _THINK_RE.sub("", text or "")
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


# ----------------------------------------------------------------------------- VRAM

def free_vram_mb() -> Optional[float]:
    """Свободная VRAM (МБ) без обязательного импорта torch; None — GPU не найден."""
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                free, _total = torch.cuda.mem_get_info()
                return free / 2**20
        except Exception:
            pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        vals = [float(x) for x in out.stdout.split() if x.strip().replace(".", "", 1).isdigit()]
        return max(vals) if vals else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def plan_gpu_layers(free_mb: Optional[float], model_mb: float, n_layers: int,
                    reserve_mb: float = 1500, kv_mb: float = 400) -> int:
    """Сколько слоёв отдать на GPU: -1 — все, 0 — CPU, N — частичный offload."""
    if free_mb is None or model_mb <= 0 or n_layers <= 0:
        return 0
    budget = free_mb - reserve_mb - kv_mb
    if budget <= 0:
        return 0
    if budget >= model_mb:
        return -1
    per_layer = model_mb / n_layers
    return max(0, min(n_layers, int(budget // per_layer)))


# ----------------------------------------------------------------------------- бэкенды

class ExtractiveBackend:
    """Ответ собирает copilot.extractive (без генерации); stream_chat не используется."""

    name = "extractive"
    is_generative = False

    def stream_chat(self, messages: Sequence[Message], **kw: Any) -> Iterator[str]:
        raise NotImplementedError("extractive backend composes answers in copilot.extractive")

    def release(self) -> None:
        return None

    def info(self) -> dict[str, Any]:
        return {"backend": self.name}


class ScriptedLLM:
    """Тестовый «LLM»: возвращает заданные ответы по кругу (или функцию от messages)."""

    name = "scripted"
    is_generative = True

    def __init__(self, responses: Iterable[str] | Any, delay_s: float = 0.0) -> None:
        self._fn = responses if callable(responses) else None
        self._responses = [] if callable(responses) else list(responses)
        self._i = 0
        self.delay_s = delay_s
        self.calls: list[list[Message]] = []

    def stream_chat(self, messages: Sequence[Message], max_tokens: int = 512, temperature: float = 0.2,
                    deadline: Optional[float] = None, **kw: Any) -> Iterator[str]:
        self.calls.append(list(messages))
        if self._fn is not None:
            text = str(self._fn(list(messages)))
        else:
            text = self._responses[self._i % len(self._responses)] if self._responses else ""
            self._i += 1
        for piece in re.findall(r"\S+\s*", text):
            if self.delay_s:
                time.sleep(self.delay_s)
            if deadline is not None and time.monotonic() > deadline:
                raise LLMTimeout("scripted timeout")
            yield piece

    def release(self) -> None:
        return None

    def info(self) -> dict[str, Any]:
        return {"backend": self.name}


class LlamaCppBackend:
    """llama-cpp-python: ленивое создание Llama, один экземпляр, генерации сериализованы."""

    name = "llama_cpp"
    is_generative = True

    def __init__(self, model_path: str | os.PathLike[str], cfg: Optional[dict[str, Any]] = None) -> None:
        self.cfg = (cfg or default_config()).get("llm", {})
        self.model_path = str(model_path)
        self._llm: Any = None
        # обычный Lock (не RLock): допускает освобождение из другого потока
        self._lock = threading.Lock()
        self._last_used = 0.0
        self._timer: Optional[threading.Timer] = None
        self.n_gpu_layers: Optional[int] = None

    def _resolve_gpu_layers(self) -> int:
        raw = str(self.cfg.get("gpu_layers", "auto")).strip().lower()
        if raw not in {"auto", ""}:
            try:
                return int(raw)
            except ValueError:
                return 0
        try:
            model_mb = Path(self.model_path).stat().st_size / 2**20
        except OSError:
            model_mb = 0.0
        return plan_gpu_layers(free_vram_mb(), model_mb, int(self.cfg.get("model_n_layers", 36)),
                               float(self.cfg.get("vram_reserve_mb", 1500)))

    def _ensure_loaded(self) -> Any:
        if self._llm is not None:
            return self._llm
        from llama_cpp import Llama  # type: ignore  # ленивый импорт

        self.n_gpu_layers = self._resolve_gpu_layers()
        kwargs: dict[str, Any] = {
            "model_path": self.model_path,
            "n_ctx": int(self.cfg.get("n_ctx", 4096)),
            "n_gpu_layers": self.n_gpu_layers,
            "verbose": False,
        }
        if self.cfg.get("n_threads"):
            kwargs["n_threads"] = int(self.cfg["n_threads"])
        lora = resolve_path(self.cfg.get("lora_path"))
        if lora and lora.is_file():
            kwargs["lora_path"] = str(lora)
        t0 = time.monotonic()
        try:
            self._llm = Llama(**kwargs)
        except Exception:
            if self.n_gpu_layers != 0:  # нет VRAM / CUDA — пробуем CPU
                log.warning("llama.cpp GPU load failed, retry on CPU", exc_info=True)
                kwargs["n_gpu_layers"] = self.n_gpu_layers = 0
                self._llm = Llama(**kwargs)
            else:
                raise
        log.info("copilot LLM loaded %s (gpu_layers=%s) in %.1fs", Path(self.model_path).name,
                 self.n_gpu_layers, time.monotonic() - t0)
        return self._llm

    def _schedule_unload(self) -> None:
        idle = float(self.cfg.get("idle_unload_s", 300) or 0)
        if idle <= 0:
            return
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(idle, self._unload_if_idle, args=(idle,))
        self._timer.daemon = True
        self._timer.start()

    def _unload_if_idle(self, idle: float) -> None:
        if time.monotonic() - self._last_used >= idle * 0.99:
            self.release()

    def release(self) -> None:
        """Выгрузить модель (освободить VRAM для VLM/детектора)."""
        acquired = self._lock.acquire(timeout=5)
        if not acquired:  # идёт генерация — не освобождаем модель из-под неё, повторим по таймеру
            log.warning("copilot LLM busy, unload postponed")
            self._schedule_unload()
            return
        try:
            if self._llm is not None:
                try:
                    close = getattr(self._llm, "close", None)
                    if callable(close):
                        close()
                finally:
                    self._llm = None
                    gc.collect()
        finally:
            self._lock.release()

    def warmup(self) -> None:
        with self._lock:
            self._ensure_loaded()
            self._last_used = time.monotonic()
        self._schedule_unload()

    def stream_chat(self, messages: Sequence[Message], max_tokens: int = 512, temperature: float = 0.2,
                    deadline: Optional[float] = None, **kw: Any) -> Iterator[str]:
        # ждём модель не дольше дедлайна: зависшие запросы не должны занимать потоки API
        wait = -1 if deadline is None else max(0.0, deadline - time.monotonic())
        if not self._lock.acquire(timeout=wait):
            raise LLMBusy("copilot model is busy with another request")
        try:
            llm = self._ensure_loaded()
            try:
                stream = llm.create_chat_completion(
                    messages=list(messages), stream=True, max_tokens=int(max_tokens),
                    temperature=float(temperature), top_p=float(self.cfg.get("top_p", 0.9)),
                    repeat_penalty=float(self.cfg.get("repeat_penalty", 1.05)),
                )
                for chunk in stream:
                    try:
                        delta = chunk["choices"][0].get("delta", {}).get("content")
                    except (KeyError, IndexError, AttributeError):
                        delta = None
                    if delta:
                        yield delta
                    if deadline is not None and time.monotonic() > deadline:
                        raise LLMTimeout("llama.cpp generation exceeded budget")
            finally:
                self._last_used = time.monotonic()
                self._schedule_unload()
        finally:
            self._lock.release()

    def info(self) -> dict[str, Any]:
        return {"backend": self.name, "model": Path(self.model_path).name, "gpu_layers": self.n_gpu_layers,
                "loaded": self._llm is not None}


def check_local_url(url: str, allow_private: bool = True) -> str:
    """Разрешаем только локальный инференс: loopback, (опц.) частные сети и docker-имена без точки."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"bad LLM server url: {url!r}")
    host = parsed.hostname.lower()
    if host == "localhost" or host.endswith(".localhost"):
        return url.rstrip("/")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        if allow_private and "." not in host:  # имя сервиса docker-compose
            return url.rstrip("/")
        raise ValueError(f"LLM server must be local (rule 1: no cloud APIs), got host {host!r}") from None
    if ip.is_loopback or (allow_private and (ip.is_private or ip.is_link_local)):
        return url.rstrip("/")
    raise ValueError(f"LLM server must be local (rule 1: no cloud APIs), got {host!r}")


class LlamaServerBackend:
    """Локальный llama.cpp server (`llama-server -m model.gguf --port 8081`)."""

    name = "llama_server"
    is_generative = True

    def __init__(self, url: str, cfg: Optional[dict[str, Any]] = None) -> None:
        self.cfg = (cfg or default_config()).get("llm", {})
        self.url = check_local_url(url, bool(self.cfg.get("allow_private_hosts", True)))

    def _opener(self) -> Any:
        import urllib.request

        return urllib.request.build_opener(urllib.request.ProxyHandler({}))  # никаких прокси наружу

    def healthy(self, timeout: float = 1.5) -> bool:
        try:
            with self._opener().open(f"{self.url}/health", timeout=timeout) as resp:
                return 200 <= resp.status < 300
        except Exception:
            return False

    def stream_chat(self, messages: Sequence[Message], max_tokens: int = 512, temperature: float = 0.2,
                    deadline: Optional[float] = None, **kw: Any) -> Iterator[str]:
        import urllib.request

        body = json.dumps({
            "messages": list(messages), "stream": True, "max_tokens": int(max_tokens),
            "temperature": float(temperature), "top_p": float(self.cfg.get("top_p", 0.9)),
        }).encode("utf-8")
        req = urllib.request.Request(f"{self.url}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        timeout = max(1.0, (deadline - time.monotonic()) if deadline else float(self.cfg.get("timeout_s", 45)))
        with self._opener().open(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0].get("delta", {}).get("content")
                except (ValueError, KeyError, IndexError):
                    delta = None
                if delta:
                    yield delta
                if deadline is not None and time.monotonic() > deadline:
                    raise LLMTimeout("llama server generation exceeded budget")

    def release(self) -> None:
        return None

    def info(self) -> dict[str, Any]:
        return {"backend": self.name, "url": self.url}


# ----------------------------------------------------------------------------- выбор

def find_model(cfg: dict[str, Any]) -> Optional[Path]:
    for cand in cfg.get("llm", {}).get("model_candidates") or []:
        p = resolve_path(cand)
        if p and p.is_file():
            return p
    return None


class RetrievalBackend(ExtractiveBackend):
    """Негенеративный режим: модели нет/GPU занят — ответ из фактов задачи и найденных норм (не mock)."""

    name = "retrieval"


def _text_of(res: Any) -> tuple[str, bool, str]:
    """(text, timed_out, error) из str / GenResult-подобного объекта / dict."""
    if res is None:
        return "", False, "empty"
    if isinstance(res, str):
        return res, False, ""
    if isinstance(res, dict):
        return (str(res.get("text") or res.get("content") or ""), bool(res.get("timed_out")),
                str(res.get("error") or ""))
    return str(getattr(res, "text", "") or ""), bool(getattr(res, "timed_out", False)), str(getattr(res, "error", "") or "")


def chat_outcome(res: Any) -> tuple[str, str, str, bool]:
    """(text, kind, error, truncated). kind: "" | unavailable | busy | timeout | invalid | error.

    Понимает ``ml.vlm.service.ChatResult`` (A11: ok/text/error/truncated/timed_out), dict из ``chat_request``,
    GenResult и str. Текст, оборванный по времени (timed_out при ok=True), считается обрезанным.
    """
    text, timed_out, error = _text_of(res)
    get = res.get if isinstance(res, dict) else (lambda k, d=None: getattr(res, k, d))
    ok = None if isinstance(res, str) or res is None else get("ok", None)
    truncated = bool(get("truncated", False)) if not isinstance(res, str) and res is not None else False
    truncated = truncated or (timed_out and bool(text))
    low = error.lower().strip()
    if text and ok is not False:
        return text, "", error, truncated
    if low.startswith("unavailable"):
        return "", "unavailable", error, False
    if "busy" in low:
        return "", "busy", error, False
    if timed_out or low.startswith("timeout"):
        return "", "timeout", error or "timeout", False
    if low.startswith("invalid"):
        return "", "invalid", error, False
    if ok is False or error or not text:
        return "", "error", error or "empty answer", False
    return text, "", "", truncated


class LLMUnavailable(RuntimeError):
    """Генеративной модели нет или она не загружена в этом процессе — отвечаем без генерации."""


def _flatten(messages: Sequence[Message]) -> tuple[str, str]:
    """messages -> (system, prompt) для API вида generate(system, prompt)."""
    system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
    turns = [f"{'Проверяющий' if m['role'] == 'user' else 'Ассистент'}: {m['content']}"
             for m in messages if m.get("role") in {"user", "assistant"}]
    return system, "\n\n".join(turns)


NOT_LOADED_HERE = "модель анализа не загружена в этом процессе (копилот её не загружает)"


def _vlm_runtime_absent(module: Any) -> bool:
    """Настоящий A11 (пакет ml.vlm) держит модель в ``ml.vlm.runtime``. Если он в этом процессе не
    импортирован, модели здесь нет: так процесс API не тянет torch/transformers ради статуса копилота."""
    return (str(getattr(module, "__package__", "") or "") == "ml.vlm" and bool(getattr(module, "__file__", None))
            and "ml.vlm.runtime" not in sys.modules)


def _params_of(fn: Any) -> tuple[set[str], bool]:
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return set(), True
    return set(sig.parameters), any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())


class VLMServiceBackend:
    """Общая модель Qwen3-VL-4B через ``ml.vlm.service.chat`` (A11) — отдельная LLM на 8 ГБ не влезает.

    * Приоритет анализа обеспечивает A11 (``gpu.PriorityLock``: ожидающий анализ всегда раньше чата);
      копилот лишь ограничивает ожидание (``wait_s``) и общий бюджет (ожидание + генерация ≤ дедлайна).
    * Модель копилот НЕ загружает: если в этом процессе VLM не загружена (в проде она живёт в
      процессе-воркере A03), генеративный режим честно недоступен (LLMUnavailable -> retrieval).
      Для доступа к модели воркера — ``copilot.service.set_chat_fn(<RPC к ml.vlm.service.chat_request>)``.
    * Подпись: ``chat(messages, **kw)`` (A11) — допустимые параметры берутся из ``get_service().chat``;
      ``chat(payload: dict)`` (IPC) — один словарь; ``chat(system=, prompt=)`` — старый вариант.
    * Ответ: ChatResult/dict/GenResult/str -> chat_outcome(); unavailable/busy/timeout различаются.
    """

    name = "vlm_service"
    is_generative = True

    def __init__(self, chat_fn: Any, cfg: Optional[dict[str, Any]] = None, module: Any = None) -> None:
        self.cfg = (cfg or default_config())
        self.vcfg = self.cfg.get("vlm_service", {}) or {}
        self._chat = chat_fn
        self._module = module
        self._params, self._varkw = _params_of(chat_fn)
        self._payload_style = self._params == {"payload"}
        if self._varkw and module is not None and callable(getattr(module, "get_service", None)):
            try:  # A11: chat(messages, **kw) -> VLMService.chat(messages, *, images, max_new_tokens, ...)
                inner, inner_varkw = _params_of(module.get_service().chat)
                if inner and not inner_varkw:
                    self._params, self._varkw = inner | {"messages"}, False
            except Exception:
                pass
        self.max_input_chars = int(self.vcfg.get("max_input_chars", 11000))

    # ------------------------------------------------------------------ доступность
    def backend_state(self) -> tuple[Optional[bool], str]:
        """(загружена ли модель в этом процессе, причина). None — неизвестно (IPC/сторонний сервис)."""
        gs = getattr(self._module, "get_service", None) if self._module is not None else None
        if not callable(gs):
            return None, ""
        if _vlm_runtime_absent(self._module):
            return False, NOT_LOADED_HERE
        try:
            be, reason = gs().backend()
        except Exception as exc:
            return False, f"VLM-сервис: ошибка ({type(exc).__name__})"
        if be is None:
            return False, f"нет модели: {reason}" if reason else "нет модели"
        av = getattr(be, "available", None)
        if callable(av):
            try:
                ok = av()
                ok, why = ok if isinstance(ok, tuple) else (bool(ok), "")
            except Exception as exc:
                ok, why = False, type(exc).__name__
            if not ok:
                return False, f"модель недоступна: {why}" if why else "модель недоступна"
        if getattr(be, "loaded", True):
            return True, ""
        return False, NOT_LOADED_HERE

    def usable(self) -> tuple[bool, str]:
        loaded, why = self.backend_state()
        if loaded is False and not (why == NOT_LOADED_HERE and bool(self.vcfg.get("allow_cold_load", False))):
            return False, why
        return True, ""

    @classmethod
    def resolve(cls, cfg: dict[str, Any], warnings: list[str]) -> Optional["VLMServiceBackend"]:
        vcfg = cfg.get("vlm_service", {}) or {}
        mod_name = str(vcfg.get("module", "ml.vlm.service"))
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:  # нет модуля / сломан импорт — генеративный режим недоступен
            warnings.append(f"{mod_name} недоступен ({type(exc).__name__})")
            return None
        fn = getattr(mod, str(vcfg.get("function", "chat")), None)
        if not callable(fn):
            warnings.append(f"{mod_name}.{vcfg.get('function', 'chat')} не найден")
            return None
        be = cls(fn, cfg, mod)
        ok, why = be.usable()  # сначала дешёвая проверка: модель не загружена здесь — available() A11 не зовём
        if not ok:
            warnings.append(f"генеративный режим недоступен: {why}")
            return None
        avail = getattr(mod, "available", None)
        if callable(avail):
            try:
                ok = avail()
                ok, why = (ok if isinstance(ok, tuple) else (bool(ok), ""))
                if not ok:
                    warnings.append(f"VLM-сервис недоступен: {why or 'нет модели'}")
                    return None
            except Exception as exc:
                warnings.append(f"VLM-сервис: ошибка проверки ({type(exc).__name__})")
                return None
        return be

    # ------------------------------------------------------------------ вызов
    def _accepts(self, name: str) -> bool:
        return self._varkw or name in self._params

    def _call(self, messages: Sequence[Message], kwargs: dict[str, Any]) -> Any:
        if self._payload_style:
            return self._chat({"messages": list(messages), **kwargs})
        if self._accepts("messages"):
            call_kw = {"messages": list(messages), **kwargs}
        else:
            system, prompt = _flatten(messages)
            call_kw = {"system": system, "prompt": prompt, **kwargs}
            if self._accepts("images"):
                call_kw["images"] = []
        for _ in range(6):  # сервис отверг незнакомый параметр — убираем его и повторяем
            try:
                return self._chat(**call_kw)
            except TypeError as exc:
                m = re.search(r"unexpected keyword argument '(\w+)'", str(exc))
                if not m or m.group(1) not in call_kw or m.group(1) in {"messages", "system", "prompt"}:
                    raise
                call_kw.pop(m.group(1))
        return self._chat(**call_kw)

    def stream_chat(self, messages: Sequence[Message], max_tokens: int = 512, temperature: float = 0.2,
                    deadline: Optional[float] = None, **kw: Any) -> Iterator[str]:
        remaining = (deadline - time.monotonic()) if deadline is not None else float(self.cfg.get("llm", {}).get("timeout_s", 45))
        if remaining <= 1.0:
            raise LLMTimeout("no time left for VLM chat")
        ok, why = self.usable()  # состояние могло измениться (A03 выгрузил VLM между видео)
        if not ok:
            raise LLMUnavailable(why)
        wait = max(0.0, min(float(self.vcfg.get("wait_s", 10)), remaining * 0.5))
        gen_timeout = max(1.0, remaining - wait - 0.5)  # ожидание GPU + генерация укладываются в дедлайн
        kwargs: dict[str, Any] = {}
        alts = {"max_new_tokens": "max_tokens", "timeout_s": "timeout"}
        for name, value in (("max_new_tokens", int(max_tokens)), ("timeout_s", gen_timeout),
                            ("wait_s", wait), ("temperature", float(temperature))):
            if self._payload_style or self._accepts(name):
                kwargs[name] = value
            elif name in alts and self._accepts(alts[name]):
                kwargs[alts[name]] = value
        if "priority" in self._params:  # только если сервис явно принимает (A11 ставит приоритет сам)
            kwargs["priority"] = str(self.vcfg.get("priority", "low"))
        try:
            res = self._call(messages, kwargs)
        except TimeoutError as exc:
            raise LLMTimeout(str(exc)) from exc
        except Exception as exc:
            if type(exc).__name__ in {"Busy", "GPUBusy", "QueueFull", "BackendBusy"}:
                raise LLMBusy(str(exc)) from exc
            raise
        if hasattr(res, "__iter__") and not isinstance(res, (str, dict, bytes)) and not hasattr(res, "text"):
            for piece in res:  # потоковый chat
                text, _, _ = _text_of(piece)
                if text:
                    yield text
                if deadline is not None and time.monotonic() > deadline:
                    raise LLMTimeout("VLM chat exceeded budget")
            return
        text, kind, error, truncated = chat_outcome(res)
        if kind == "unavailable":
            raise LLMUnavailable(error)
        if kind == "busy":
            raise LLMBusy(error or "busy")
        if kind == "timeout":
            raise LLMTimeout(error or "VLM chat timed out")
        if kind:
            raise RuntimeError(f"VLM chat {kind}: {error}")
        if truncated:  # обрезано по лимиту токенов/времени — оставляем только законченные предложения
            from .guard import complete_prefix

            text = complete_prefix(text.rstrip() + " ") or text
        yield text

    def release(self) -> None:
        return None  # моделью владеет A11/A03

    def info(self) -> dict[str, Any]:
        loaded, why = self.backend_state()
        return {"backend": self.name, "module": getattr(self._module, "__name__", "") if self._module else "injected",
                "model_loaded_here": loaded, "note": why}


class VLMRuntimeAdapter(VLMServiceBackend):
    """ВРЕМЕННЫЙ адаптер к внутренностям A11 (``ml.vlm.runtime``), пока нет ``ml.vlm.service.chat``.

    Выключен по умолчанию (``vlm_service.runtime_adapter: false``). Модель сам не загружает: если VLM
    не «тёплая» (не загружена пайплайном), генеративный режим недоступен. GPU-замок A11 берётся с
    таймаутом ``wait_s``: не дождались — LLMBusy (приоритет у анализа видео).
    """

    name = "vlm_runtime"

    @classmethod
    def resolve(cls, cfg: dict[str, Any], warnings: list[str]) -> Optional["VLMRuntimeAdapter"]:
        try:
            runtime = importlib.import_module("ml.vlm.runtime")
            vlm_config = importlib.import_module("ml.vlm.config")
            be, reason = runtime.get_backend(vlm_config.load_config())
        except Exception as exc:
            warnings.append(f"ml.vlm.runtime недоступен ({type(exc).__name__})")
            return None
        if be is None:
            warnings.append(f"VLM не настроена: {reason}")
            return None
        if not getattr(be, "loaded", False) and not (cfg.get("vlm_service", {}) or {}).get("allow_cold_load", False):
            warnings.append("VLM не загружена пайплайном — копилот её сам не грузит")
            return None
        adapter = cls(lambda **kw: None, cfg, runtime)
        adapter._backend = be
        return adapter

    def stream_chat(self, messages: Sequence[Message], max_tokens: int = 512, temperature: float = 0.2,
                    deadline: Optional[float] = None, **kw: Any) -> Iterator[str]:
        remaining = (deadline - time.monotonic()) if deadline is not None else 30.0
        system, prompt = _flatten(messages)
        lock = self._module.gpu_guard()
        wait = min(max(0.0, remaining), float(self.vcfg.get("wait_s", 10)))

        def _gen() -> Any:
            if not getattr(self._backend, "loaded", False):
                raise LLMBusy("VLM выгружена")
            return self._backend.generate([], system, prompt, max_new_tokens=int(max_tokens),
                                          timeout_s=max(1.0, (deadline - time.monotonic()) if deadline else 30.0),
                                          temperature=float(temperature))

        if hasattr(lock, "acquire") and hasattr(lock, "release"):
            if not lock.acquire(timeout=wait):
                raise LLMBusy("GPU занят анализом видео")
            try:
                res = _gen()
            finally:
                lock.release()
        else:  # замок без acquire(timeout) — только как context manager (ждём без таймаута)
            with lock:
                res = _gen()
        text, timed_out, error = _text_of(res)
        if timed_out and not text:
            raise LLMTimeout(error or "VLM generate timed out")
        if error and not text:
            raise RuntimeError(f"VLM generate error: {error}")
        yield text


def make_llm(cfg: Optional[dict[str, Any]] = None, warnings: Optional[list[str]] = None) -> Optional[Any]:
    """Выбор генеративного бэкенда. Модель здесь НЕ загружается.

    auto: ``ml.vlm.service.chat`` (общая Qwen3-VL-4B) -> [временный адаптер ml.vlm.runtime, если включён]
    -> локальный llama.cpp server (если явно задан ``server_url``). Отдельная llama.cpp-модель в процессе
    (``llama_cpp``) — только явно: на 8 ГБ рядом с VLM она не помещается.
    Нет модели -> None: сервис отвечает в негенеративном режиме (retrieval) или 501 — по конфигу.
    ``mock`` — только при SENTINEL_MOCK=1 (auto) или явно; ``extractive`` — только явно.
    """
    cfg = cfg or default_config()
    lcfg = cfg.get("llm", {})
    backend = str(lcfg.get("backend", "auto")).lower()
    warn = warnings if warnings is not None else []
    if backend == "extractive":
        return ExtractiveBackend()
    if backend == "mock" or (is_mock() and backend == "auto"):
        mock = ExtractiveBackend()
        mock.name = "mock"
        return mock
    if backend in {"auto", "vlm_service"}:
        be = VLMServiceBackend.resolve(cfg, warn)
        if be is not None:
            return be
        if (cfg.get("vlm_service", {}) or {}).get("runtime_adapter", False):
            ad = VLMRuntimeAdapter.resolve(cfg, warn)
            if ad is not None:
                return ad
    if backend in {"auto", "llama_server"} and lcfg.get("server_url"):
        try:
            srv = LlamaServerBackend(str(lcfg["server_url"]), cfg)
            if srv.healthy():
                return srv
            warn.append("llama.cpp server не отвечает")
        except ValueError as exc:
            warn.append(str(exc))
    if backend == "llama_cpp":
        path = find_model(cfg)
        if path is None:
            warn.append("GGUF-модель не найдена (llm.model_candidates)")
        elif importlib.util.find_spec("llama_cpp") is None:
            warn.append("llama-cpp-python не установлен")
        else:
            return LlamaCppBackend(path, cfg)
    if backend not in {"auto", "llama_cpp", "llama_server", "vlm_service"}:
        warn.append(f"неизвестный llm.backend: {backend!r}")
    return None
