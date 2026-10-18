"""Честная доступность стадии позы — без загрузки моделей и без побочных эффектов.

Поза «доступна», только если для выбранного бэкенда одновременно: файл весов найден локально И нужный
пакет установлен (importlib.util.find_spec, без импорта). Импортируемость самого ml.pose ничего не значит.
`verified` становится True лишь после реальной загрузки бэкенда в этом процессе (estimate()/warmup()).

model_info() — необязательный хук A03 (backend.pipeline.stages.module_identity): A03 кладёт его в identity
стадии и в ключ кэша, а `checkpoint` превращает в подпись файла (имя:размер:mtime). Поэтому появление весов
или смена SENTINEL_POSE_* меняет ключ кэша, и пустой результат «без весов» не переиспользуется.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from typing import Any, Optional

from .config import backend_override, enabled_mode, is_mock_mode, load_config

POSE_ENV = ("SENTINEL_POSE_ENABLED", "SENTINEL_POSE_BACKEND", "SENTINEL_POSE_CONFIG", "SENTINEL_MOCK",
            "SENTINEL_POSE_ONNX", "SENTINEL_POSE_MP_TASK", "SENTINEL_WEIGHTS_DIR")
_PACKAGE = {"rtmpose_onnx": "onnxruntime", "mediapipe": "mediapipe"}


def _has_package(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _weights(name: str, cfg: dict[str, Any]) -> Optional[str]:
    if name == "rtmpose_onnx":
        from .backends.rtmpose_onnx import find_weights

        return find_weights(cfg)
    if name == "mediapipe":
        from .backends.mediapipe_backend import find_task

        return find_task(cfg)
    return None


def _loaded() -> dict[str, Any]:
    from .backends import _CACHE

    return dict(_CACHE)


def backend_report(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Состояние одного реального бэкенда: веса, пакет, готовность, причина."""
    weights = _weights(name, cfg)
    package = _has_package(_PACKAGE[name])
    loaded = name in _loaded()
    reason = None
    if not weights:
        reason = "веса не найдены локально"
    elif not package:
        reason = f"пакет {_PACKAGE[name]} не установлен"
    return {"weights": weights, "package": package, "ready": bool(weights) and package, "verified": loaded,
            "reason": reason}


def availability(cfg: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Что будет, если сейчас вызвать estimate(): доступен ли реальный бэкенд и почему нет.

    stage_status — прогноз в терминах contracts.extensions.STAGE_STATUSES: ok | skipped | failed | mock.
    """
    cfg = cfg or load_config()
    mode = enabled_mode(cfg)
    mock = is_mock_mode()
    order = ["mock"] if mock else ([backend_override()] if backend_override() else
                                   list(cfg.get("backend", {}).get("order", ["rtmpose_onnx"])))
    backends = {name: backend_report(name, cfg) for name in ("rtmpose_onnx", "mediapipe")}
    info: dict[str, Any] = {"enabled": mode, "order": order, "mock": mock, "backends": backends,
                            "available": False, "backend": None, "checkpoint": [], "stage_status": None,
                            "reason": None}
    if mode == "false":
        info.update(stage_status="skipped", reason="стадия позы выключена (enabled: false / SENTINEL_POSE_ENABLED=0)")
        return info
    if mock:
        info.update(available=True, backend="mock", stage_status="mock",
                    reason="MOCK: синтетические позы (только тесты/демо), не настоящие")
        return info
    for name in order:
        rep = backends.get(name)
        if rep and rep["ready"]:
            info.update(available=True, backend=name, checkpoint=[rep["weights"]], stage_status="ok")
            return info
    why = "; ".join(f"{n}: {backends[n]['reason']}" for n in order if n in backends) or "нет бэкендов в order"
    info.update(stage_status="skipped" if mode == "auto" else "failed", reason=why)
    return info


def _config_digest(cfg: dict[str, Any]) -> str:
    raw = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def model_info() -> dict[str, Any]:
    """Хук A03: identity стадии позы (входит в ключ кэша и в provenance). Без загрузки моделей."""
    cfg = load_config()
    av = availability(cfg)
    return {
        "checkpoint": list(av["checkpoint"]),          # A03 → checkpoint_sig (имя:размер:mtime)
        "available": av["available"],
        "backend": av["backend"],
        "stage_status": av["stage_status"],
        "reason": av["reason"],
        "enabled": av["enabled"],
        "mock": av["mock"],
        "models_loaded": [av["backend"]] if av["available"] and av["backend"] else [],
        "config_sha": _config_digest(cfg),            # пресеты SENTINEL_POSE_CONFIG меняют ключ кэша
        "env": {k: os.environ[k] for k in POSE_ENV if os.environ.get(k)},
    }
