"""Арбитраж GPU внутри процесса: анализ (verify/discover) всегда раньше чата копилота.

PriorityLock — реентерабельный lock с приоритетами: среди ожидающих первым получает тот, у кого приоритет
меньше (0 — анализ, 10 — чат), при равенстве — кто раньше пришёл. Идущий вызов не прерывается (GPU-операцию
не вытеснить), поэтому у чата короткие лимиты токенов/времени и ограниченное ожидание (ответ busy).
Поверх него берётся внешний lock A03 (set_gpu_lock), чтобы VLM не шла параллельно с детектором/энкодерами.
"""
from __future__ import annotations

import contextlib
import heapq
import itertools
import threading
import time
from typing import Any, Iterator, Optional

PRIO_ANALYSIS = 0
PRIO_CHAT = 10


class PriorityLock:
    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._owner: Optional[int] = None
        self._depth = 0
        self._heap: list[tuple[int, int, int]] = []  # (priority, seq, thread id)
        self._seq = itertools.count()
        self.waits: dict[int, list[float]] = {}

    def acquire(self, priority: int = PRIO_ANALYSIS, timeout: Optional[float] = None) -> bool:
        me = threading.get_ident()
        t0 = time.monotonic()
        with self._cond:
            if self._owner == me:  # реентерабельность
                self._depth += 1
                return True
            ticket = (int(priority), next(self._seq), me)
            heapq.heappush(self._heap, ticket)
            deadline = None if timeout is None else t0 + max(0.0, float(timeout))
            while not (self._owner is None and self._heap[0] == ticket):
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    self._heap.remove(ticket)
                    heapq.heapify(self._heap)
                    self._cond.notify_all()
                    return False
                self._cond.wait(left)
            heapq.heappop(self._heap)
            self._owner, self._depth = me, 1
            self.waits.setdefault(int(priority), []).append(time.monotonic() - t0)
            return True

    def release(self) -> None:
        with self._cond:
            if self._owner != threading.get_ident():
                raise RuntimeError("PriorityLock released by non-owner")
            self._depth -= 1
            if self._depth == 0:
                self._owner = None
                self._cond.notify_all()

    @contextlib.contextmanager
    def hold(self, priority: int = PRIO_ANALYSIS, timeout: Optional[float] = None) -> Iterator[None]:
        if not self.acquire(priority, timeout):
            raise TimeoutError("gpu busy")
        try:
            yield
        finally:
            self.release()

    def stats(self) -> dict[str, Any]:
        with self._cond:
            waiting: dict[int, int] = {}
            for p, _s, _t in self._heap:
                waiting[p] = waiting.get(p, 0) + 1
            return {"held": self._owner is not None, "waiting": waiting,
                    "mean_wait_s": {p: round(sum(w) / len(w), 4) for p, w in self.waits.items() if w}}

    def analysis_waiting(self) -> bool:
        with self._cond:
            return any(p <= PRIO_ANALYSIS for p, _s, _t in self._heap)


_PRIO_LOCK = PriorityLock()
_external: Any = None


def priority_lock() -> PriorityLock:
    return _PRIO_LOCK


def set_external_lock(lock: Optional[Any]) -> None:
    """Общий GPU-lock A03 (любой context manager; Lock/RLock — с таймаутом для чата)."""
    global _external
    _external = lock


def _enter_external(timeout: Optional[float]) -> Optional[Any]:
    """Берёт внешний lock; -> объект для выхода или None (не удалось за timeout). Без внешнего — nullcontext."""
    lk = _external
    if lk is None:
        return contextlib.nullcontext()
    if timeout is not None and hasattr(lk, "acquire") and hasattr(lk, "release"):
        try:
            ok = lk.acquire(timeout=max(0.0, timeout))
        except TypeError:  # acquire без timeout
            ok = lk.acquire()
        if not ok:
            return None

        class _Rel:
            def __enter__(self) -> None:
                return None

            def __exit__(self, *a: Any) -> bool:
                lk.release()
                return False

        return _Rel()
    if hasattr(lk, "__enter__"):
        lk.__enter__()

        class _Exit:
            def __enter__(self) -> None:
                return None

            def __exit__(self, *a: Any) -> bool:
                lk.__exit__(None, None, None)
                return False

        return _Exit()
    return contextlib.nullcontext()


@contextlib.contextmanager
def guard(priority: int = PRIO_ANALYSIS, timeout: Optional[float] = None) -> Iterator[None]:
    """Приоритетный lock процесса + внешний lock A03. TimeoutError — не дождались за timeout (только чат)."""
    t0 = time.monotonic()
    if not _PRIO_LOCK.acquire(priority, timeout):
        raise TimeoutError("gpu busy")
    try:
        left = None if timeout is None else max(0.0, timeout - (time.monotonic() - t0))
        ext = _enter_external(left)
        if ext is None:
            raise TimeoutError("gpu busy (external lock)")
        with ext:
            yield
    finally:
        _PRIO_LOCK.release()
