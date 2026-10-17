"""Фабрика бэкендов с кэшем (модели держим тёплыми между видео)."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from ..config import env_flag
from ..labels import COCO_TO_LABEL
from ..weights import is_cached, missing_message, resolve
from .base import FrameDetector, resolve_device
from .mock import MockColorDetector

log = logging.getLogger("ml.detect.backends")

CLOSED_SET_LABELS = frozenset(COCO_TO_LABEL.values())


@dataclass
class Backends:
    closed: list[FrameDetector] = field(default_factory=list)
    open: Optional[FrameDetector] = None
    mock: bool = False
    warnings: list[str] = field(default_factory=list)
    fallback_used: bool = False      # загружен не основной closed-set (только если явно задан hf_fallback_ids)

    @property
    def names(self) -> list[str]:
        out = [getattr(b, "name", type(b).__name__) for b in self.closed]
        if self.open is not None:
            out.append(getattr(self.open, "name", type(self.open).__name__))
        return out


_LOCK = threading.Lock()
_CACHE: dict[tuple, Backends] = {}


def is_mock() -> bool:
    return env_flag("SENTINEL_MOCK")


class DetectorUnavailable(RuntimeError):
    """Нет ни одного рабочего бэкенда (нет локальных весов / зависимостей). Mock автоматически НЕ включается."""


def _mock_backends(cfg: dict[str, Any]) -> Backends:
    """Только явный тестовый режим SENTINEL_MOCK=1."""
    m = cfg["mock"]
    common = dict(palette=m["palette"], tolerance=m["tolerance"], min_area_px=m["min_area_px"], score=m["score"])
    closed = MockColorDetector(**common, motion_person=m.get("motion_person", True),
                               motion_person_min_area=m.get("motion_person_min_area", 0.02), name="mock")
    open_labels = {q["label"] for q in cfg["open_vocab"]["queries"]}
    opn = MockColorDetector(**common, labels=open_labels, name="mock:open") if cfg["open_vocab"].get("enabled") else None
    return Backends(closed=[closed], open=opn, mock=True)


def _load_closed(cfg: dict[str, Any], warnings: list[str], flags: Optional[dict[str, bool]] = None) -> list[FrameDetector]:
    cs = cfg["closed_set"]
    rt = cfg["runtime"]
    out: list[FrameDetector] = []
    for kind in cs.get("backends", ["hf"]):
        if kind == "onnx_finetuned":
            p, lp = resolve(cs["onnx_path"]), resolve(cs["onnx_labels"])
            if not (p.exists() and lp.exists()):
                continue
            try:
                from .onnx_det import OnnxDetrBackend

                out.append(OnnxDetrBackend(p, lp, device=rt["device"], input_size=cs["input_size"],
                                           min_score=cs["min_score"], max_dets=cs["max_dets_per_frame"]))
            except Exception as e:  # noqa: BLE001
                warnings.append(f"detect: finetuned ONNX not loaded: {e}")
                continue
            if not cs.get("onnx_merge_with_hf", True):
                break
        elif kind == "hf":
            from .hf_detr import HFDetrBackend

            primary = cs["hf_model_id"]
            # резервные модели — только явно перечисленные в конфиге (по умолчанию нет): без тихой подмены детектора
            for mid in [primary, *cs.get("hf_fallback_ids", [])]:
                if not is_cached(mid):
                    warnings.append(f"detect: {missing_message(mid)}")
                    continue
                try:
                    out.append(HFDetrBackend(mid, device=rt["device"], fp16=rt["fp16"], input_size=cs["input_size"],
                                             min_score=cs["min_score"], max_dets=cs["max_dets_per_frame"],
                                             allow_proxy=cs.get("proxy_labels", True)))
                except Exception as e:  # noqa: BLE001
                    warnings.append(f"detect: closed-set {mid} not loaded: {type(e).__name__}: {e}")
                    continue
                if mid != primary:
                    warnings.append(f"detect: основной детектор {primary} не загружен — используется явно заданный "
                                    f"резервный {mid} (пороги не калиброваны под него; status=degraded)")
                    if flags is not None:
                        flags["fallback_used"] = True
                break
    return out


def _load_open(cfg: dict[str, Any], warnings: list[str]) -> Optional[FrameDetector]:
    ov = cfg["open_vocab"]
    if not ov.get("enabled", True):
        return None
    if not is_cached(ov["hf_model_id"]):
        warnings.append(f"detect: open-vocab {missing_message(ov['hf_model_id'])} — stage skipped")
        return None
    try:
        from .owlv2 import OwlV2Backend

        rt = cfg["runtime"]
        return OwlV2Backend(ov["hf_model_id"], ov["queries"], device=rt["device"], fp16=rt["fp16"],
                            input_size=ov["input_size"], nms_iou=ov["nms_iou"], max_dets=ov["max_dets_per_frame"],
                            min_area=ov.get("min_area", 0.0), max_area=ov.get("max_area"))
    except Exception as e:  # noqa: BLE001
        warnings.append(f"detect: open-vocab {ov['hf_model_id']} not loaded: {type(e).__name__}: {e}")
        return None


def _key(cfg: dict[str, Any]) -> tuple:
    import json

    sub = {k: cfg.get(k) for k in ("runtime", "closed_set", "open_vocab", "mock")}
    sub["runtime"] = {k: v for k, v in (sub["runtime"] or {}).items() if k in ("device", "fp16")}
    return (is_mock(), json.dumps(sub, sort_keys=True, default=str))


def get_backends(cfg: dict[str, Any]) -> Backends:
    key = _key(cfg)
    with _LOCK:
        if key in _CACHE:
            be = _CACHE[key]
            for b in [*be.closed, be.open]:
                if hasattr(b, "reset"):
                    b.reset()
            return be
        if is_mock():
            be = _mock_backends(cfg)
        else:
            warnings: list[str] = []
            flags: dict[str, bool] = {}
            closed = _load_closed(cfg, warnings, flags)
            opn = _load_open(cfg, warnings)
            for w in warnings:
                log.warning(w)
            if not closed:
                # не кэшируем: после setup (скачивания весов) следующий вызов подхватит модели
                raise DetectorUnavailable("; ".join(warnings) or "no closed-set detector configured")
            be = Backends(closed=closed, open=opn, mock=False, warnings=warnings,
                          fallback_used=flags.get("fallback_used", False))
        _CACHE[key] = be
        return be


# mock: группа запросов inventory → цвета палитры (mock.palette + mock.extra_palette)
MOCK_GROUP_COLORS = {"cash": ["cash"], "cash_bag": ["cash_bag"], "generic_bag": ["bag"], "personal_item": ["jacket"],
                     "phone": ["cell_phone"], "landline": ["landline"], "cup": ["cup", "bottle"], "food": ["food"],
                     "safe_open": ["safe_open"], "safe_closed": ["safe"]}

_SCORERS: dict[tuple, Any] = {}


def inventory_texts(cfg: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for qs in cfg["inventory"]["rare_open_vocab"].get("queries", {}).values():
        for q in qs:
            if q not in out:
                out.append(q)
    return out


def get_inventory_scorer(cfg: dict[str, Any], warnings: list[str]) -> Any:
    """Сырой open-vocab скорер для inventory (mode=rare_open_vocab). Нет весов → None + warning (не исключение:
    closed-set часть inventory работает, OWL-виды остаются unknown)."""
    texts = inventory_texts(cfg)
    ov, rt = cfg["open_vocab"], cfg["runtime"]
    key = (is_mock(), tuple(texts), ov["hf_model_id"], rt.get("device"), rt.get("fp16"), ov.get("input_size"))
    with _LOCK:
        if key in _SCORERS:
            return _SCORERS[key]
        if is_mock():
            from .mock import MockOpenVocabScorer

            m = cfg["mock"]
            pal = {**m["palette"], **m.get("extra_palette", {})}
            groups = cfg["inventory"]["rare_open_vocab"].get("queries", {})
            tc = {q: [pal[c] for c in MOCK_GROUP_COLORS.get(g, []) if c in pal] for g, qs in groups.items() for q in qs}
            sc: Any = MockOpenVocabScorer(texts, tc, tolerance=m["tolerance"], min_area_px=m["min_area_px"])
        else:
            if not is_cached(ov["hf_model_id"]):
                warnings.append(f"inventory: open-vocab {missing_message(ov['hf_model_id'])} — "
                                "rare_open_vocab недоступен, работаю как baseline")
                return None
            try:
                from .owlv2 import OwlV2Backend

                sc = OwlV2Backend(ov["hf_model_id"], [{"text": q} for q in texts], device=rt["device"], fp16=rt["fp16"],
                                  input_size=ov["input_size"], name="owlv2")
            except Exception as e:  # noqa: BLE001
                warnings.append(f"inventory: open-vocab not loaded: {type(e).__name__}: {e} — работаю как baseline")
                return None
        _SCORERS[key] = sc
        return sc


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()
        _SCORERS.clear()


__all__ = ["Backends", "CLOSED_SET_LABELS", "DetectorUnavailable", "get_backends", "get_inventory_scorer",
           "clear_cache", "is_mock", "resolve_device"]
