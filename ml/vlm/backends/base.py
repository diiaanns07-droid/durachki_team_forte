"""Интерфейс VLM-бэкенда. Все бэкенды — локальные (правило 1: никаких облачных API)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# Причины окончания генерации: eos — модель сама закончила (ответ полный); length — упёрлись в max_new_tokens
# (ответ обрезан); time — упёрлись в max_time (обрезан); stop — иной стоп; "" — бэкенд не сообщает.
FINISH_TRUNCATED = {"length", "time"}


@dataclass
class GenResult:
    text: str
    latency_s: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    timed_out: bool = False
    error: str = ""
    meta: dict[str, Any] = field(default_factory=dict)  # timings {prep_s, gen_s, decode_s}, vram_peak_mb, …
    finish_reason: str = ""

    @property
    def truncated(self) -> bool:
        return self.timed_out or self.finish_reason in FINISH_TRUNCATED


@dataclass
class OptionScores:
    """Вероятности вариантов ответа из логитов ПЕРВОГО токена ответа (softmax по вариантам).

    probs — нормированы по вариантам (сумма 1); mass — доля полной вероятности следующего токена, пришедшаяся
    на варианты (низкая => модель «хотела» ответить иначе, ответу доверять меньше); mode — words|letters.
    """
    probs: dict[str, float]
    mass: float
    latency_s: float
    mode: str = "words"
    meta: dict[str, Any] = field(default_factory=dict)


class BackendUnavailable(RuntimeError):
    """Бэкенд нельзя использовать (нет весов/CUDA/сервера) — стадию VLM пропускаем."""


class VLMBackend:
    name: str = "base"
    model_id: str = ""
    supports_multi_image: bool = True
    supports_option_probs: bool = False

    def available(self) -> tuple[bool, str]:
        """Дешёвая проверка без загрузки весов: (можно_ли, причина)."""
        return True, ""

    def load(self) -> None:  # лениво, один раз; держим «тёплым»
        return None

    def unload(self) -> None:
        return None

    @property
    def loaded(self) -> bool:
        return True

    def generate(self, images: list[Any], system: str, prompt: str, max_new_tokens: int = 256,
                 timeout_s: float = 10.0, json_schema: Optional[dict[str, Any]] = None,
                 temperature: float = 0.0) -> GenResult:
        raise NotImplementedError

    def option_probs(self, images: list[Any], system: str, prompt: str,
                     options: dict[str, list[str]], timeout_s: float = 10.0) -> OptionScores:
        """Вероятности вариантов (ключ -> варианты написания) по логитам первого токена ответа."""
        raise NotImplementedError(f"{self.name}: option_probs not supported")

    def chat(self, messages: list[dict[str, Any]], images: Optional[list[Any]] = None, max_new_tokens: int = 256,
             timeout_s: float = 20.0, temperature: float = 0.0) -> GenResult:
        """Многоходовой диалог (для копилота A14). По умолчанию — через generate по последней реплике."""
        system = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
        turns = [m for m in messages if m.get("role") in ("user", "assistant")]
        history = "\n".join(f"{m['role']}: {m.get('content', '')}" for m in turns[:-1])
        last = str(turns[-1].get("content", "")) if turns else ""
        prompt = f"{history}\n{last}".strip() if history else last
        return self.generate(list(images or []), system, prompt, max_new_tokens=max_new_tokens,
                             timeout_s=timeout_s, json_schema=None, temperature=temperature)

    def info(self) -> dict[str, Any]:
        return {"backend": self.name, "model": self.model_id}
