"""llama.cpp-бэкенды (MIT): (1) локальный llama-server по HTTP (OpenAI-совместимый /v1/chat/completions,
картинки — data:image/jpeg;base64, JSON-схема -> грамматика); (2) llama-cpp-python в процессе (экспериментально).

Сервер запускается отдельно (A17): llama-server -m <model.gguf> --mmproj <mmproj.gguf> --port 8081 -ngl 99 -c 8192
Нелокальные URL запрещены (local_only) — облачных вызовов нет и быть не может.
"""
from __future__ import annotations

import importlib.util
import ipaddress
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Optional
from urllib.parse import urlparse

from ..frames import make_grid, pil_to_jpeg_b64
from .base import BackendUnavailable, GenResult, VLMBackend

log = logging.getLogger(__name__)

_LOCAL_NAMES = {"localhost", "host.docker.internal", "vlm", "llama", "llama-server"}


def is_local_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").strip("[]").lower()
    if not host:
        return False
    if host in _LOCAL_NAMES:
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_loopback or ip.is_private or ip.is_link_local
    except ValueError:
        return "." not in host  # имя сервиса docker-compose без домена


def _opener() -> urllib.request.OpenerDirector:
    # локальный сервер — мимо любых HTTP(S)_PROXY
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


class LlamaCppServerBackend(VLMBackend):
    name = "llamacpp_server"

    def __init__(self, url: str = "http://127.0.0.1:8081", model_file: str = "", health_timeout_s: float = 1.5,
                 use_json_schema: bool = True, local_only: bool = True, jpeg_quality: int = 85, **_: Any) -> None:
        self.url = url.rstrip("/")
        self.model_id = model_file or "llama-server"
        self.health_timeout_s = float(health_timeout_s)
        self.use_json_schema = bool(use_json_schema)
        self.local_only = bool(local_only)
        self.jpeg_quality = int(jpeg_quality)
        self._schema_mode = "json_schema" if self.use_json_schema else "json_object"
        self._opener = _opener()

    def available(self) -> tuple[bool, str]:
        if self.local_only and not is_local_url(self.url):
            return False, f"non-local llama.cpp URL refused: {self.url}"
        try:
            with self._opener.open(self.url + "/health", timeout=self.health_timeout_s) as r:
                if r.status == 200:
                    return True, ""
                return False, f"health status {r.status}"
        except urllib.error.HTTPError as e:  # 503 — модель ещё грузится
            return False, f"health HTTP {e.code}"
        except Exception as e:  # noqa: BLE001
            return False, f"llama-server not reachable at {self.url}: {e}"

    def _post(self, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with self._opener.open(req, timeout=max(0.5, timeout_s)) as r:
            return json.loads(r.read().decode("utf-8"))

    def generate(self, images: list[Any], system: str, prompt: str, max_new_tokens: int = 256,
                 timeout_s: float = 10.0, json_schema: Optional[dict[str, Any]] = None,
                 temperature: float = 0.0) -> GenResult:
        if self.local_only and not is_local_url(self.url):
            raise BackendUnavailable(f"non-local llama.cpp URL refused: {self.url}")
        t0 = time.perf_counter()
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + pil_to_jpeg_b64(im, self.jpeg_quality)}}
            for im in images]
        content.append({"type": "text", "text": prompt})
        payload: dict[str, Any] = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "max_tokens": int(max_new_tokens), "temperature": float(temperature), "stream": False,
            "cache_prompt": True,
        }
        if json_schema is not None:
            if self._schema_mode == "json_schema":
                payload["response_format"] = {"type": "json_schema",
                                              "json_schema": {"name": "answer", "schema": json_schema, "strict": True}}
            else:
                payload["response_format"] = {"type": "json_object"}
        try:
            data = self._post(payload, timeout_s)
        except urllib.error.HTTPError as e:
            if e.code == 400 and "response_format" in payload and self._schema_mode == "json_schema":
                log.warning("llama-server rejected json_schema (%s); falling back to json_object", e)
                self._schema_mode = "json_object"
                payload["response_format"] = {"type": "json_object"}
                data = self._post(payload, max(0.5, timeout_s - (time.perf_counter() - t0)))
            else:
                return GenResult(text="", latency_s=time.perf_counter() - t0, error=f"HTTP {e.code}")
        except TimeoutError:
            return GenResult(text="", latency_s=time.perf_counter() - t0, timed_out=True, error="timeout")
        except urllib.error.URLError as e:
            timed_out = "timed out" in str(e.reason).lower()
            return GenResult(text="", latency_s=time.perf_counter() - t0, timed_out=timed_out, error=str(e.reason))
        lat = time.perf_counter() - t0
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage = data.get("usage") or {}
        fr = str(choice.get("finish_reason") or "")
        return GenResult(text=str(msg.get("content") or ""), latency_s=lat,
                         prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                         finish_reason={"stop": "eos", "length": "length"}.get(fr, fr),
                         meta={"finish_reason": fr, "timings": data.get("timings")})

    def info(self) -> dict[str, Any]:
        return {"backend": self.name, "model": self.model_id, "url": self.url, "schema_mode": self._schema_mode}


class LlamaCppPythonBackend(VLMBackend):
    """llama-cpp-python в процессе. Одна картинка-коллаж на вызов (multi-image поддержан не всеми хендлерами)."""

    name = "llamacpp_python"
    supports_multi_image = False

    def __init__(self, models_dir: str = "weights/vlm", model_file: str = "", mmproj_file: str = "",
                 chat_handler: str = "Qwen25VLChatHandler", n_gpu_layers: int = -1, n_ctx: int = 8192,
                 jpeg_quality: int = 85, **_: Any) -> None:
        self.model_path = os.path.join(models_dir, model_file) if model_file else ""
        self.mmproj_path = os.path.join(models_dir, mmproj_file) if mmproj_file else ""
        self.model_id = model_file or "llama-cpp-python"
        self.handler_name = chat_handler
        self.n_gpu_layers = int(n_gpu_layers)
        self.n_ctx = int(n_ctx)
        self.jpeg_quality = int(jpeg_quality)
        self._llm: Any = None
        self._lock = threading.Lock()

    def available(self) -> tuple[bool, str]:
        if importlib.util.find_spec("llama_cpp") is None:
            return False, "llama-cpp-python not installed"
        for p in (self.model_path, self.mmproj_path):
            if not p or not os.path.exists(p):
                return False, f"GGUF not found: {p or '(not configured)'}"
        return True, ""

    @property
    def loaded(self) -> bool:
        return self._llm is not None

    def load(self) -> None:
        with self._lock:
            if self._llm is not None:
                return
            ok, why = self.available()
            if not ok:
                raise BackendUnavailable(why)
            from llama_cpp import Llama
            import llama_cpp.llama_chat_format as fmt

            cls = getattr(fmt, self.handler_name, None) or fmt.Llava15ChatHandler
            handler = cls(clip_model_path=self.mmproj_path, verbose=False)
            self._llm = Llama(model_path=self.model_path, chat_handler=handler, n_ctx=self.n_ctx,
                              n_gpu_layers=self.n_gpu_layers, verbose=False)

    def unload(self) -> None:
        with self._lock:
            self._llm = None

    def generate(self, images: list[Any], system: str, prompt: str, max_new_tokens: int = 256,
                 timeout_s: float = 10.0, json_schema: Optional[dict[str, Any]] = None,
                 temperature: float = 0.0) -> GenResult:
        if self._llm is None:
            self.load()
        imgs = [make_grid(list(images), cols=3)] if len(images) > 1 else list(images)
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + pil_to_jpeg_b64(im, self.jpeg_quality)}}
            for im in imgs]
        content.append({"type": "text", "text": prompt})
        kw: dict[str, Any] = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                              "max_tokens": int(max_new_tokens), "temperature": float(temperature)}
        if json_schema is not None:
            kw["response_format"] = {"type": "json_object", "schema": json_schema}
        t0 = time.perf_counter()
        deadline = t0 + max(0.5, timeout_s)
        with self._lock:
            try:
                from llama_cpp import StoppingCriteriaList

                kw["stopping_criteria"] = StoppingCriteriaList([lambda *_a: time.perf_counter() > deadline])
                data = self._llm.create_chat_completion(**kw)
            except TypeError:
                kw.pop("stopping_criteria", None)
                data = self._llm.create_chat_completion(**kw)
        lat = time.perf_counter() - t0
        msg = ((data.get("choices") or [{}])[0].get("message") or {})
        usage = data.get("usage") or {}
        return GenResult(text=str(msg.get("content") or ""), latency_s=lat, prompt_tokens=usage.get("prompt_tokens"),
                         completion_tokens=usage.get("completion_tokens"), timed_out=time.perf_counter() > deadline)
