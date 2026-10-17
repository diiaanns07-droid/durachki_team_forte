"""Жизненный цикл бэкенда (ленивая загрузка, тёплая модель, unload), общий GPU-lock и вызов с JSON-ретраем.

Один GPU-владелец: все обращения к GPU (load + generate) идут под lock, который A03 может заменить своим
(ml.vlm.set_gpu_lock) — тогда VLM не работает параллельно с детектором/энкодерами.
"""
from __future__ import annotations

import contextlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, ContextManager, Optional

from .backends import BackendUnavailable, VLMBackend, resolve_backend
from .budget import Budget, visual_tokens
from .parsing import extract_json

log = logging.getLogger(__name__)

_state_lock = threading.Lock()
_backend: Optional[VLMBackend] = None
_backend_key: Optional[tuple[Any, ...]] = None
_backend_reason = ""
_injected: Optional[VLMBackend] = None
def set_gpu_lock(lock: Optional[Any]) -> None:
    """A03 передаёт общий lock/семафор GPU (любой context manager). None — без внешнего lock.
    Внутри процесса VLM дополнительно арбитрирует анализ и чат приоритетным lock (ml.vlm.gpu)."""
    from . import gpu

    gpu.set_external_lock(lock)


def gpu_guard() -> ContextManager[Any]:
    """GPU для анализа (приоритет выше чата копилота)."""
    from . import gpu

    return gpu.guard(gpu.PRIO_ANALYSIS)


def set_backend(backend: Optional[VLMBackend]) -> None:
    """Явная подстановка бэкенда (тесты, интеграция). None — снова из конфига."""
    global _injected
    _injected = backend


def _key(cfg: dict[str, Any]) -> tuple[Any, ...]:
    preset = (cfg.get("presets") or {}).get(cfg.get("preset", ""), {})
    return (cfg.get("backend"), cfg.get("preset"), repr(preset), repr(cfg.get("llamacpp_server")),
            repr((cfg.get("mock") or {})))


def get_backend(cfg: dict[str, Any]) -> tuple[Optional[VLMBackend], str]:
    global _backend, _backend_key, _backend_reason
    if _injected is not None:
        return _injected, ""
    rc = cfg.get("runtime") or {}
    configure_registry(bool(rc.get("use_registry", True)), float(rc.get("registry_vram_gb", 3.5)))
    k = _key(cfg)
    with _state_lock:
        if _backend_key != k:
            if _backend is not None:
                with contextlib.suppress(Exception):
                    _backend.unload()
            _backend, _backend_reason = resolve_backend(cfg)
            _backend_key = k
            if _backend is None:
                log.warning("VLM stage disabled: %s", _backend_reason)
        return _backend, _backend_reason


_load_failed: dict[int, tuple[float, str]] = {}  # id(backend) -> (monotonic time, причина)
REGISTRY_NAME = "ml.vlm:qwen"
_registry_vram_gb = 3.5   # оценка резерва: 2.7 ГБ после загрузки (замер L1) + активации <=6 кадров по 448 px
_use_registry = True


def configure_registry(use: bool = True, vram_gb: float = 3.5) -> None:
    global _use_registry, _registry_vram_gb
    _use_registry, _registry_vram_gb = bool(use), float(vram_gb)


def _register(reg: Any, be: VLMBackend) -> Any:
    return reg.get_or_load(REGISTRY_NAME, lambda: (be.load(), be)[1], vram_gb=_registry_vram_gb,
                           unload=lambda obj: obj.unload())


def _registry() -> Optional[Any]:
    """ModelRegistry A03, только если он уже создан в этом процессе (воркер пайплайна); сами его не создаём."""
    if not _use_registry:
        return None
    try:
        from backend.pipeline.registry import ModelRegistry  # type: ignore

        return ModelRegistry.peek()
    except Exception:  # noqa: BLE001 — нет A03 (CLI, тесты) — грузим сами
        return None


def ensure_loaded(be: VLMBackend, budget: Optional[Budget] = None, min_budget_s: float = 0.0,
                  retry_cooldown_s: float = 300.0) -> tuple[bool, str]:
    """Ленивая загрузка. Если бюджет передан: время загрузки съедает его же (модель держите тёплой через
    warmup()); при остатке < min_budget_s холодная загрузка не начинается. Неудачная загрузка не
    повторяется retry_cooldown_s секунд (иначе каждое видео платило бы за неё заново)."""
    if be.loaded:
        return True, ""
    failed = _load_failed.get(id(be))
    if failed is not None and time.monotonic() - failed[0] < retry_cooldown_s:
        return False, f"load failed recently: {failed[1]}"
    if budget is not None and budget.remaining() < min_budget_s:
        return False, f"model not warm and budget {budget.remaining():.1f}s < lazy_load_min_budget_s {min_budget_s}"
    try:
        with gpu_guard():
            reg = _registry()
            if reg is not None and be.name not in ("mock",):
                # A03 освобождает LRU-модели под нужный объём и сможет выгрузить VLM через be.unload()
                obj = _register(reg, be)
                if obj is not be:  # запись держит прежний экземпляр (после unload/смены конфига): заменить,
                    with contextlib.suppress(Exception):  # иначе новый экземпляр жил бы вне учёта VRAM A03
                        reg.release(REGISTRY_NAME)
                    _register(reg, be)
                if not be.loaded:  # запись есть, но модель выгружали напрямую через be.unload()
                    be.load()
            else:
                be.load()
        _load_failed.pop(id(be), None)
        return True, ""
    except BackendUnavailable as e:
        reason = str(e)
    except Exception as e:  # noqa: BLE001 — OOM, битые веса и т.п.: стадия пропускается
        log.warning("VLM load failed: %s", type(e).__name__)
        reason = f"{type(e).__name__}: {str(e)[:200]}"
        _free_cuda()
    _load_failed[id(be)] = (time.monotonic(), reason)
    return False, reason


def _free_cuda() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def unload_backend() -> None:
    """Выгрузить модель и снять её запись в ModelRegistry A03 (иначе реестр считал бы VRAM занятой)."""
    global _backend, _backend_key
    _load_failed.clear()
    reg = _registry()
    if reg is not None:
        with contextlib.suppress(Exception):
            reg.release(REGISTRY_NAME)
    with _state_lock:
        if _backend is not None:
            with contextlib.suppress(Exception):
                _backend.unload()
        _backend, _backend_key = None, None


_last_stage: dict[str, dict[str, Any]] = {}


def record_stage(stage: str, **info: Any) -> None:
    """Последний итог стадии (verify|discover) — только технические поля и счётчики, без содержимого."""
    _last_stage[stage] = {"t": round(time.time(), 3), **info}


def backend_status() -> dict[str, Any]:
    be = _injected or _backend
    info = be.info() if be is not None else {"backend": None}
    info["reason"] = _backend_reason if be is None else ""
    info["last"] = {k: dict(v) for k, v in _last_stage.items()}
    return info


# --------------------------------------------------------------------------- вызов с ретраем


@dataclass
class CallOutcome:
    ok: bool = False
    status: str = "not_called"   # ok | parse_failed | timeout | error | skipped_budget
    data: dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    latency_s: float = 0.0
    repaired: bool = False
    error: str = ""
    finish_reason: str = ""        # последнего вызова: eos | length | time | stop | ""
    truncated: bool = False        # ответ обрезан по длине/времени (complete=False)
    raw: list[str] = field(default_factory=list)  # только для локального дампа (CLI), не для логов


def fit_images(images: list[Any], max_frames: int, max_vis_tokens: int, px_per_token: int,
               keep_index: Optional[int] = None) -> list[Any]:
    """Урезает число кадров (сохраняя keep_index) и при необходимости уменьшает их, чтобы влезть в лимит токенов."""
    from .frames import resize_max_side

    imgs = list(images)
    if max_frames > 0 and len(imgs) > max_frames:
        keep = keep_index if keep_index is not None and 0 <= keep_index < len(imgs) else len(imgs) // 2
        step = (len(imgs) - 1) / max(1, max_frames - 1)
        idx = sorted({int(round(i * step)) for i in range(max_frames)})
        if keep not in idx:
            idx[min(range(len(idx)), key=lambda j: abs(idx[j] - keep))] = keep
            idx = sorted(set(idx))
        imgs = [imgs[i] for i in idx]
    for _ in range(4):
        tot = sum(visual_tokens(im.size, px_per_token) for im in imgs)
        if max_vis_tokens <= 0 or tot <= max_vis_tokens:
            break
        k = (max_vis_tokens / float(tot)) ** 0.5 * 0.97
        imgs = [resize_max_side(im, max(2 * px_per_token, int(max(im.size) * k))) for im in imgs]
    return imgs


def call_json(be: VLMBackend, images: list[Any], system: str, prompt: str, budget: Budget,
              normalize: Callable[[dict[str, Any]], tuple[Optional[dict[str, Any]], str]],
              max_new_tokens: int, schema: Optional[dict[str, Any]] = None, retry_suffix: str = "",
              temperature: float = 0.0,
              fallback: Optional[Callable[[str], Optional[dict[str, Any]]]] = None) -> CallOutcome:
    """Вызов VLM -> JSON -> нормализация.

    1 ретрай (в пределах бюджета) — только при невалидном JSON; таймаут не ретраится. Регэксп-fallback
    (без фактов) — лишь последний шанс, если ретрай не дал валидного JSON или не поместился в бюджет.
    """
    out = CallOutcome()
    lim = budget.limits
    vis = sum(visual_tokens(getattr(im, "size", (448, 448)), budget.px_per_token) for im in images)
    fb_data: Optional[dict[str, Any]] = None

    def _finish_with_fallback() -> CallOutcome:
        # регэксп-fallback только для НЕобрезанного, но невалидного текста; обрезанный ответ им не «спасаем»
        if fb_data is not None and out.status == "parse_failed":
            out.ok, out.status, out.data, out.repaired = True, "ok", fb_data, True
        return out

    for attempt in range(1 + max(0, lim.max_retries_per_item)):
        retry = attempt > 0
        if not budget.can_call(len(images), vis, retry=retry):
            if not retry:
                out.status = "skipped_budget"
            break
        mnt = budget.output_tokens_for_call(max_new_tokens)
        if mnt < 24:
            budget.stop_reason = budget.stop_reason or "max_output_tokens_total"
            out.status = out.status if retry else "skipped_budget"
            break
        p = prompt if not retry else f"{prompt}\n\n{retry_suffix}".strip()
        res = None
        try:
            with gpu_guard():
                # пока ждали общий GPU-lock (другое видео/детектор), бюджет мог закончиться
                if budget.remaining() < budget.est_call_s() * lim.safety:
                    budget.stop_reason = budget.stop_reason or "time_after_gpu_wait"
                else:
                    res = be.generate(images, system, p, max_new_tokens=mnt, timeout_s=budget.call_timeout(),
                                      json_schema=schema, temperature=temperature)
        except Exception as e:  # noqa: BLE001 — сбой бэкенда не роняет пайплайн
            budget.record(0.0, vis, 0, 0, retry)
            out.attempts += 1
            out.status, out.error = "error", f"{type(e).__name__}"
            log.warning("VLM generate failed: %s", type(e).__name__)
            break
        if res is None:
            out.status = out.status if retry else "skipped_budget"
            break
        out.attempts += 1
        out.latency_s += res.latency_s
        out.finish_reason, out.truncated = res.finish_reason, bool(res.truncated)
        budget.record(res.latency_s, vis, res.completion_tokens, mnt, retry)
        out.raw.append(res.text)
        if not res.text.strip():
            out.status, out.error = ("timeout", res.error or "timeout") if res.timed_out else ("error", res.error or "empty")
            break
        if res.truncated:
            # Ответ обрезан (max_new_tokens -> length, max_time -> time): даже «отремонтированный» JSON из него —
            # недосказанный ответ, ему нельзя доверять подтверждение/отказ. length — 1 ретрай со строгим
            # суффиксом (короче), если позволяют лимиты; time — без ретрая (дорого).
            by_time = res.timed_out or res.finish_reason == "time"
            out.status, out.error = ("timeout", "time") if by_time else ("truncated", "length")
            if by_time:
                break
            continue
        pr = extract_json(res.text)
        if pr.ok:
            data, err = normalize(pr.data)
            if data is not None:
                out.ok, out.status, out.data, out.repaired = True, "ok", data, pr.repaired
                return out
            out.error = err
        else:
            out.error = pr.error
            if fallback is not None:
                fb_data = fallback(res.text) or fb_data
        out.status = "parse_failed"
        if res.timed_out:  # обрезанный по времени ответ не ретраим — дорого
            out.status = "timeout"
            break
    return _finish_with_fallback()
