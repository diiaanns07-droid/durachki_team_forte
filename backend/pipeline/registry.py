"""ModelRegistry: тёплые модели между задачами (внутри процесса-воркера), устройство, учёт VRAM (8 ГБ).

Один владелец GPU обеспечивается исполнителем (worker.py): все вызовы моделей идут последовательно
в одном процессе. Реестр — необязательное хранилище моделей для модулей + политика VLM.
"""
from __future__ import annotations

import gc
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from .config import cfg_get

log = logging.getLogger("sentinel.pipeline")


@dataclass
class _Entry:
    name: str
    obj: Any
    vram_gb: float
    keep_warm: bool
    unload: Optional[Callable[[Any], None]]
    last_used: float = field(default_factory=time.monotonic)


class ModelRegistry:
    """Синглтон процесса. Модули могут (необязательно) держать модели здесь через ``get_or_load``;
    иначе оркестратор управляет памятью через необязательные хуки модулей ``warmup()``/``unload()``.
    """

    _instance: Optional["ModelRegistry"] = None
    _instance_lock = threading.Lock()

    def __init__(self, cfg: Optional[Mapping] = None) -> None:
        self._lock = threading.RLock()
        self._models: dict[str, _Entry] = {}
        self._device: Optional[str] = None
        self.cfg: Mapping = cfg or {}
        self.warm = False                      # True только после успешного warmup()
        self.warmup_report: dict[str, Any] = {}
        self.warmup_error: Optional[str] = None
        self.vram_measured_mb: dict[str, float] = {}

    # ---------------------------------------------------------------- синглтон
    @classmethod
    def instance(cls, cfg: Optional[Mapping] = None) -> "ModelRegistry":
        with cls._instance_lock:
            if cls._instance is None:
                if cfg is None:
                    from .config import load_config
                    cfg = load_config()
                cls._instance = cls(cfg)
            elif cfg is not None:
                cls._instance.cfg = cfg
            return cls._instance

    @classmethod
    def peek(cls) -> Optional["ModelRegistry"]:
        """Синглтон, если уже создан (без создания)."""
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Только для тестов: забыть синглтон (модели освобождаются)."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.release_all()
            cls._instance = None

    # ---------------------------------------------------------------- устройство
    def device(self) -> str:
        if self._device is not None:
            return self._device
        want = str(cfg_get(self.cfg, "gpu.device", "auto") or "auto").lower()
        if want == "cpu":
            self._device = "cpu"
        else:
            self._device = "cuda" if _cuda_available() else "cpu"
            if want == "cuda" and self._device != "cuda":
                log.warning("gpu.device=cuda, но CUDA недоступна — работаем на CPU")
        return self._device

    def set_device(self, device: Optional[str]) -> None:
        self._device = device

    # ---------------------------------------------------------------- модели
    def get_or_load(self, name: str, loader: Callable[[], Any], *, vram_gb: float = 0.0,
                    keep_warm: bool = True, unload: Optional[Callable[[Any], None]] = None) -> Any:
        with self._lock:
            entry = self._models.get(name)
            if entry is not None:
                entry.last_used = time.monotonic()
                return entry.obj
            if vram_gb > 0 and self.device() == "cuda":
                self.ensure_vram(vram_gb, exclude=(name,))
            obj = loader()
            self._models[name] = _Entry(name, obj, float(vram_gb), keep_warm, unload)
            return obj

    def loaded(self) -> dict[str, float]:
        with self._lock:
            return {n: e.vram_gb for n, e in self._models.items()}

    def release(self, name: str) -> bool:
        with self._lock:
            entry = self._models.pop(name, None)
        if entry is None:
            return False
        if entry.unload is not None:
            try:
                entry.unload(entry.obj)
            except Exception as exc:  # noqa: BLE001
                log.warning("unload(%s) упал: %s", name, exc)
        del entry
        self.empty_cache()
        return True

    def release_all(self, keep: tuple[str, ...] = ()) -> None:
        for name in list(self._models):
            if name not in keep:
                self.release(name)

    def ensure_vram(self, need_gb: float, exclude: tuple[str, ...] = ()) -> bool:
        """Освобождает LRU-модели реестра, пока не освободится need_gb (или нечего освобождать)."""
        budget = float(cfg_get(self.cfg, "gpu.vram_total_gb", 8.0)) - float(cfg_get(self.cfg, "gpu.vram_reserve_gb", 0.7))
        while True:
            free = self.free_vram_gb()
            used = sum(e.vram_gb for e in self._models.values())
            fits = (free is None or free >= need_gb) and used + need_gb <= budget
            if fits:
                return True
            victims = sorted((e for e in self._models.values() if e.name not in exclude),
                             key=lambda e: (e.keep_warm, e.last_used))
            if not victims:
                return free is None or free >= need_gb
            self.release(victims[0].name)

    # ---------------------------------------------------------------- VRAM-утилиты
    @staticmethod
    def free_vram_gb() -> Optional[float]:
        torch = sys.modules.get("torch")
        if torch is None:
            return None
        try:
            if not torch.cuda.is_available():
                return None
            free, _total = torch.cuda.mem_get_info()
            return free / 1024 ** 3
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def empty_cache() -> None:
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is None:
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def stats(self) -> dict:
        return {"device": self.device(), "loaded": self.loaded(),
                "free_vram_gb": _round(self.free_vram_gb()), "warm": self.warm,
                "vram_measured_mb": dict(self.vram_measured_mb)}

    # ---------------------------------------------------------------- политика VLM
    def vlm_policy(self) -> str:
        policy = str(cfg_get(self.cfg, "gpu.vlm_policy", "auto") or "auto").lower()
        if policy not in ("auto", "keep", "on_demand", "off"):
            policy = "auto"
        if policy == "auto":
            if self.device() != "cuda":
                return "on_demand"
            need = float(cfg_get(self.cfg, "gpu.vram_gb.vlm_verify", 5.0))
            others = sum(float(v) for k, v in (cfg_get(self.cfg, "gpu.vram_gb", {}) or {}).items()
                         if not str(k).startswith("vlm"))
            total = float(cfg_get(self.cfg, "gpu.vram_total_gb", 8.0)) - float(cfg_get(self.cfg, "gpu.vram_reserve_gb", 0.7))
            return "keep" if need + others <= total else "on_demand"
        return policy


def _round(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(v, 3)


def _cuda_available() -> bool:
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        return False
    try:
        import torch  # noqa: WPS433 — лениво, только при первом вопросе об устройстве
    except Exception:  # noqa: BLE001
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False
