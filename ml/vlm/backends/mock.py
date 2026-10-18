"""Mock-бэкенд (SENTINEL_MOCK=1): детерминированный, без весов. Для тестов и прогонов пайплайна.

Режимы (config mock.mode / env SENTINEL_VLM_MOCK_MODE):
  neutral — verify: visible=true, confidence=0.5 (ничего не меняет); discover: пусто.
  confirm — verify подтверждает кандидата (0.85); discover пусто.
  reject  — verify отвергает (0.85); discover пусто.
  garbage — невалидный ответ (проверка ретрая/фолбэка).
  phone   — verify подтверждает; discover находит «phone» (V05) в каждом окне.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Optional

from .base import GenResult, OptionScores, VLMBackend

Responder = Callable[[list[Any], str, str], str]


class MockBackend(VLMBackend):
    """ТОЛЬКО для тестов и демо (SENTINEL_MOCK=1 или явный backend: mock). Ответы помечены как mock."""

    name = "mock"
    model_id = "mock-vlm"
    supports_option_probs = True

    def __init__(self, mode: str = "neutral", latency_s: float = 0.0, responder: Optional[Responder] = None,
                 responses: Optional[list[str]] = None, clock: Any = None,
                 option_responder: Optional[Callable[[list[Any], str, str, dict[str, list[str]]], dict[str, float]]] = None
                 ) -> None:
        self.mode = mode
        self.option_responder = option_responder
        self.latency_s = float(latency_s)
        self.responder = responder
        self.responses = list(responses or [])
        self.clock = clock  # FakeClock для тестов бюджета: advance() вместо sleep()
        self.calls: list[dict[str, Any]] = []

    def _wait(self, timeout_s: float) -> bool:
        d = min(self.latency_s, timeout_s) if timeout_s > 0 else self.latency_s
        if d > 0:
            if self.clock is not None and hasattr(self.clock, "advance"):
                self.clock.advance(d)
            else:
                time.sleep(d)
        return self.latency_s > timeout_s > 0

    def _default(self, prompt: str) -> str:
        is_discover = '"findings"' in prompt
        m = re.search(r"(?:тип|type):\s*(V\d{2})", prompt)
        cand = m.group(1) if m else "V99"
        if self.mode == "garbage":
            return "Sorry, I cannot help with that"
        if is_discover:
            if self.mode == "phone":
                return json.dumps({"findings": [{"item": "phone", "observed_facts": [
                    "Кассир держит в руке предмет, похожий на мобильный телефон"], "confidence": 0.8, "frame": 2}],
                    "uncertain_because": ""}, ensure_ascii=False)
            return '{"findings": [], "uncertain_because": ""}'
        if self.mode in ("confirm", "phone"):
            return json.dumps({"visible": True, "type_id": cand, "observed_facts": [
                "На кадрах виден описанный признак"], "confidence": 0.85, "best_frame": 3,
                "uncertain_because": ""}, ensure_ascii=False)
        if self.mode == "reject":
            return json.dumps({"visible": False, "type_id": cand, "observed_facts": [], "confidence": 0.85,
                               "best_frame": 1, "uncertain_because": "признак не виден на кадрах"}, ensure_ascii=False)
        return json.dumps({"visible": True, "type_id": cand, "observed_facts": [], "confidence": 0.5,
                           "best_frame": 1, "uncertain_because": "mock backend"}, ensure_ascii=False)

    def generate(self, images: list[Any], system: str, prompt: str, max_new_tokens: int = 256,
                 timeout_s: float = 10.0, json_schema: Optional[dict[str, Any]] = None,
                 temperature: float = 0.0) -> GenResult:
        t0 = time.perf_counter()
        self.calls.append({"n_images": len(images), "system": system, "prompt": prompt,
                           "sizes": [getattr(i, "size", None) for i in images]})
        timed_out = self._wait(timeout_s)
        if timed_out:
            return GenResult(text="", latency_s=min(self.latency_s, timeout_s), timed_out=True, error="timeout")
        if self.responses:
            text = self.responses.pop(0)
        elif self.responder is not None:
            text = self.responder(images, system, prompt)
        else:
            text = self._default(prompt)
        lat = self.latency_s if self.clock is not None else time.perf_counter() - t0
        return GenResult(text=text, latency_s=lat, prompt_tokens=None, completion_tokens=len(text) // 4)

    def option_probs(self, images: list[Any], system: str, prompt: str,
                     options: dict[str, list[str]], timeout_s: float = 10.0) -> OptionScores:
        """neutral — равномерно; confirm/phone — «да»; reject — «нет»; garbage — «не видно»."""
        self.calls.append({"n_images": len(images), "system": system, "prompt": prompt, "options": list(options),
                           "sizes": [getattr(i, "size", None) for i in images], "kind": "option_probs"})
        self._wait(timeout_s)
        keys = list(options)
        if self.option_responder is not None:
            raw = self.option_responder(images, system, prompt, options)
        else:
            raw = {k: 1.0 for k in keys}
            pick = {"confirm": "yes", "phone": "yes", "reject": "no", "garbage": "unknown"}.get(self.mode)
            if pick in raw:
                raw = {k: (17.0 if k == pick else 1.0) for k in keys}
        tot = sum(max(0.0, float(raw.get(k, 0.0))) for k in keys) or 1.0
        probs = {k: max(0.0, float(raw.get(k, 0.0))) / tot for k in keys}
        return OptionScores(probs=probs, mass=0.9, latency_s=self.latency_s, mode="words", meta={"mock": True})

    def chat(self, messages: list[dict[str, Any]], images: Optional[list[Any]] = None, max_new_tokens: int = 256,
             timeout_s: float = 20.0, temperature: float = 0.0) -> GenResult:
        self.calls.append({"kind": "chat", "n_messages": len(messages), "n_images": len(images or [])})
        timed_out = self._wait(timeout_s)
        if timed_out:
            return GenResult(text="", latency_s=min(self.latency_s, timeout_s), timed_out=True, error="timeout",
                             finish_reason="time")
        if self.responses:
            text = self.responses.pop(0)
        else:
            last = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
            text = f"[mock] получено сообщение из {len(str(last))} символов"
        return GenResult(text=text, latency_s=self.latency_s, completion_tokens=len(text) // 4, finish_reason="eos")
