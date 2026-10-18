"""ml.vlm — локальная мультимодальная модель (A11): верификация событий и поиск новых типов нарушений.

Публичные функции (сигнатуры контракта v1 + необязательные keyword-only расширения):
    verify(clip, events, budget_s, *, limits=None) -> list[Event]
    discover(clip, budget_s, *, limits=None, motion=None) -> list[Event]
Служебные: warmup(), unload(), status(), set_gpu_lock(lock), set_catalog(catalog), set_backend(backend).

Импорт пакета лёгкий: torch/transformers/cv2 подгружаются только при первом реальном вызове.
Ни verify, ни discover не бросают исключений наружу — при сбое стадия деградирует (события как были / []).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

log = logging.getLogger(__name__)

__version__ = "2.0.0"

__all__ = ["verify", "discover", "warmup", "unload", "status", "set_gpu_lock", "set_catalog", "set_backend",
           "model_info", "__version__"]


def verify(clip: Any, events: list[Any], budget_s: float, *, limits: Optional[dict[str, Any]] = None,
           zones: Optional[list[Any]] = None) -> list[Any]:
    """Проверка событий-кандидатов. Возвращает те же события (1:1, тот же порядок) с обновлёнными
    confidence/facts и extra.vlm: verdict/score (для ml.fusion), agreement, coverage, …
    zones — зоны A06 (для кропа стола/сейфа в режиме qa); можно и в event.extra["zones"]."""
    from .verifier import _copy_event, verify_events

    try:
        return verify_events(clip, list(events or []), float(budget_s), limits=limits, zones=zones)
    except Exception as e:  # noqa: BLE001 — стадия не должна ронять пайплайн
        log.warning("ml.vlm.verify failed: %s", type(e).__name__)
        out = [_copy_event(ev) for ev in (events or [])]
        for ev in out:
            ev.extra["vlm"] = {"status": "error", "agreement": "not_checked", "error": type(e).__name__}
        return out


def discover(clip: Any, budget_s: float, *, limits: Optional[dict[str, Any]] = None,
             motion: Optional[list[tuple[float, float]]] = None) -> list[Any]:
    """Поиск нарушений вне правил (включая V99). По умолчанию выключен (config: discover.enabled=false).
    motion — необязательная кривая [(t, 0..1)] (сигнал motion_energy из A08), чтобы не считать её заново."""
    from .config import load_config
    from .discoverer import discover_events

    try:
        cfg = load_config()
        dcfg = cfg.get("discover") or {}
        if dcfg.get("enabled", False) and str(dcfg.get("mode", "windows")) == "states":
            from .states import states_events

            return states_events(clip, float(budget_s), cfg=cfg, limits=limits)
        return discover_events(clip, float(budget_s), limits=limits, motion=motion)
    except Exception as e:  # noqa: BLE001
        log.warning("ml.vlm.discover failed: %s", type(e).__name__)
        return []


def warmup() -> None:
    """Загрузить модель заранее (держим тёплой). Без бэкенда — тихо ничего не делает."""
    from .config import load_config
    from .runtime import ensure_loaded, get_backend

    be, _ = get_backend(load_config())
    if be is not None:
        ensure_loaded(be)


def unload() -> None:
    """Освободить VRAM (A03 может вызывать между видео/перед тяжёлыми стадиями)."""
    from .runtime import unload_backend

    unload_backend()


def status() -> dict[str, Any]:
    """Технический статус (без содержимого кадров/ответов): бэкенд, модель, загружена ли, причина отказа."""
    from .runtime import backend_status

    return backend_status()


def set_gpu_lock(lock: Any) -> None:
    from .runtime import set_gpu_lock as _set

    _set(lock)


def set_catalog(catalog: Optional[dict[str, Any]]) -> None:
    from .catalog import set_catalog as _set

    _set(catalog)


def set_backend(backend: Any) -> None:
    from .runtime import set_backend as _set

    _set(backend)


def model_info() -> dict[str, Any]:
    """Хук A03 для ключа кэша: чекпойнт, ревизия снапшота, версия промптов (+ хэш .md-промптов, qa_spec.yaml и
    config.yaml — A03 сам хэширует только .py/.yaml/.json/.txt/.jinja, а промпты здесь в .md)."""
    import hashlib
    from pathlib import Path

    from .config import load_config, preset_cfg

    cfg = load_config()
    here = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(list((here / "prompts").glob("*.md")) + [here / "qa_spec.yaml", here / "config.yaml"]):
        if p.exists():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    model = preset_cfg(cfg, "transformers").get("model")
    revision = None
    try:
        from huggingface_hub import try_to_load_from_cache

        from .backends.transformers_backend import resolve_cache_dir

        r = try_to_load_from_cache(str(model), "config.json",
                                   cache_dir=resolve_cache_dir((cfg.get("transformers") or {}).get("cache_dir")))
        if isinstance(r, str):
            revision = Path(r).parent.name  # имя каталога снапшота = commit hash
    except Exception:  # noqa: BLE001
        pass
    return {"checkpoint": model, "revision": revision,
            "prompt_version": f"{cfg.get('prompt_version', 'v1')}+{h.hexdigest()[:12]}",
            "verify_mode": (cfg.get("verify") or {}).get("mode", "json"), "backend": cfg.get("backend"),
            "version": __version__}
