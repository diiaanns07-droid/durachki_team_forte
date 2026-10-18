"""Общий бюджет VLM-стадии: время (budget_s от A03) + лимиты вызовов/кадров/визуальных токенов/выходных
токенов/ретраев. Любой лимит исчерпан -> новые вызовы не начинаются (события остаются как были)."""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


class FakeClock:
    """Управляемые часы для тестов бюджета (без sleep)."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = float(t)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


@dataclass
class VLMLimits:
    max_calls: int = 6
    max_frames_per_call: int = 6
    max_visual_tokens_per_call: int = 1200
    max_visual_tokens_total: int = 6000
    max_output_tokens_per_call: int = 220
    max_output_tokens_total: int = 1500
    max_retries_per_item: int = 1
    max_retries_total: int = 2
    est_call_s: float = 4.0
    safety: float = 1.15
    call_timeout_s: float = 12.0

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any], stage: str, overrides: Optional[dict[str, Any]] = None) -> "VLMLimits":
        b = dict((cfg.get("budget") or {}).get("common") or {})
        b.update((cfg.get("budget") or {}).get(stage) or {})
        g = cfg.get("generation") or {}
        b.setdefault("est_call_s", g.get("est_call_s", 4.0))
        b.setdefault("safety", g.get("budget_safety", 1.15))
        b.setdefault("call_timeout_s", g.get("call_timeout_s", 12.0))
        if overrides:
            b.update(overrides)
        names = cls.__dataclass_fields__
        return cls(**{k: type(getattr(cls, k))(v) for k, v in b.items() if k in names})


def visual_tokens(size: tuple[int, int], px_per_token: int = 28) -> int:
    """Оценка числа визуальных токенов Qwen-VL: (H/px)*(W/px), px = patch*merge (28 у Qwen2.5-VL, 32 у Qwen3-VL)."""
    w, h = size
    return max(1, math.ceil(w / px_per_token)) * max(1, math.ceil(h / px_per_token))


@dataclass
class Budget:
    total_s: float
    limits: VLMLimits = field(default_factory=VLMLimits)
    clock: Callable[[], float] = time.monotonic
    px_per_token: int = 28

    def __post_init__(self) -> None:
        self.t0 = self.clock()
        self.calls = 0
        self.retries = 0
        self.visual_tokens = 0
        self.output_tokens = 0
        self.latencies: list[float] = []
        self.stop_reason = ""
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- время
    def elapsed(self) -> float:
        return self.clock() - self.t0

    def remaining(self) -> float:
        return max(0.0, float(self.total_s) - self.elapsed())

    def est_call_s(self) -> float:
        if not self.latencies:
            return self.limits.est_call_s
        # консервативно: максимум из EMA и последнего замера
        ema = self.latencies[0]
        for x in self.latencies[1:]:
            ema = 0.6 * ema + 0.4 * x
        return max(ema, self.latencies[-1])

    def call_timeout(self) -> float:
        return max(0.0, min(self.limits.call_timeout_s, self.remaining()))

    # ----------------------------------------------------------------- допуски
    def can_call(self, n_frames: int = 0, vis_tokens: int = 0, retry: bool = False) -> bool:
        lim = self.limits
        reason = ""
        if self.calls >= lim.max_calls:
            reason = "max_calls"
        elif retry and self.retries >= lim.max_retries_total:
            reason = "max_retries_total"
        elif self.visual_tokens + vis_tokens > lim.max_visual_tokens_total:
            reason = "max_visual_tokens_total"
        elif self.output_tokens >= lim.max_output_tokens_total:
            reason = "max_output_tokens_total"
        elif self.remaining() < self.est_call_s() * lim.safety:
            reason = "time"
        if reason:
            self.stop_reason = self.stop_reason or reason
            return False
        return True

    def output_tokens_for_call(self, wanted: int) -> int:
        left = self.limits.max_output_tokens_total - self.output_tokens
        return max(0, min(int(wanted), self.limits.max_output_tokens_per_call, left))

    def record(self, latency_s: float, vis_tokens: int = 0, out_tokens: Optional[int] = None,
               max_new_tokens: int = 0, retry: bool = False) -> None:
        with self._lock:
            self.calls += 1
            self.retries += int(retry)
            self.visual_tokens += int(vis_tokens)
            self.output_tokens += int(out_tokens if out_tokens is not None else max_new_tokens)
            self.latencies.append(float(latency_s))

    def summary(self) -> dict[str, Any]:
        return {
            "budget_s": round(float(self.total_s), 2),
            "elapsed_s": round(self.elapsed(), 3),
            "calls": self.calls,
            "retries": self.retries,
            "visual_tokens": self.visual_tokens,
            "output_tokens": self.output_tokens,
            "stop_reason": self.stop_reason,
            "mean_call_s": round(sum(self.latencies) / len(self.latencies), 3) if self.latencies else None,
        }
