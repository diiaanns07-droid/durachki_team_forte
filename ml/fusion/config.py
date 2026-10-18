"""Загрузка конфигов ml.fusion: config.yaml (параметры слияния) и thresholds.yaml (пороги вердикта).

Отсутствующие ключи берутся из встроенных значений по умолчанию, поэтому модуль
работает даже без YAML-файлов и без PyYAML. Пути можно переопределить env-переменными
``SENTINEL_FUSION_CONFIG`` и ``SENTINEL_FUSION_THRESHOLDS``.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"
THRESHOLDS_PATH = HERE / "thresholds.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    "version": 1,
    "grouping": {
        "merge_gap_s": 3.0,
        "cross_type_nms": True,
        "cross_type_ratio": 0.5,
        "cross_type_min_overlap": 0.5,
    },
    "sources": {
        "rules": 1.0,
        "temporal": 0.8,
        "vlm": 0.9,
        "vlm_discover": 0.7,
        "fusion": 0.8,
        "other": 0.6,
    },
    "combine": "max",
    "temporal_mode": "off",
    "agreement": {"min_conf": 0.3, "bonus_per_source": 0.05, "max_bonus": 0.10},
    "vlm": {"reject_factor": 0.6, "reject_min_score": 0.5, "unsure_factor": 1.0},
    "risk": {
        "support_weight": 0.6,
        "support_min": 0.5,
        "support_pad_s": 1.0,
        "high": 0.65,
        "min_duration_s": 0.4,
        "only_factor": 0.7,
        "fallback_type": "V99",
        "max_events": 3,
    },
    "report_gate": {
        "require_fact": True,
        "fact_sources": ["rules", "vlm"],
        "require_norm_types": ["V99"],
        "norm_missing_if_no_catalog": "report",
        "temporal_only_types": False,
        "upstream_assessment": True,
        "observation_policy": "hint",
        "insufficient_policy": "hint",
        "uncap_norm_review": True,
    },
    "policies": {
        "v02_after_window": {"enabled": False, "remain_min": 0.5, "removed_max": 0.2, "removed_factor": 0.35},
        "v01_client_side": {"enabled": False},
        "negative_evidence": {"enabled": False, "weight": 0.5},
    },
    "timecode": {"policy": "peak"},
    "output": {
        "max_violations": 10,
        "timeline_step_s": 0.5,
        "timecode_round": 1,
        "candidate_floor": 0.15,
    },
    "evidence": {
        "enabled": True,
        "ui_overlays": True,
        "ui_overlay_frames": 1,
        "policy": "sharpest",
        "action": {"before_s": 1.0, "after_s": 2.0, "local_s": 0.2},
        "ui_label_mode": "legend",
        "max_frames": 3,
        "window_s": 1.0,
        "min_sep_s": 0.4,
        "dup_block": 4.0,
        "context_frame": True,
        "context_min_gap_s": 1.5,
        "box_max_dt_s": 1.5,
        "max_side": 1280,
        "max_bytes": 200_000,
        "jpeg_quality": [88, 80, 72, 64, 56, 48, 40, 32],
        "budget_s": 20.0,
        "blur_heads": True,
        "head_fallback": "none",
        "head_frac": 0.18,
        "head_kp_min_score": 0.3,
        "draw_boxes": True,
        "font_paths": [],
    },
    "description": {
        "cite_norm": "verified_only",
        "max_facts": 3,
        "max_chars": 420,
        "check_sentence": "Требуется проверка человеком.",
        "check_sentence_possible": "Это признаки возможного нарушения — требуется проверка человеком.",
        "forbidden": [
            r"укра[лд]\w*",
            r"краж\w*",
            r"вор(?:ов\w*|ует|уют|овал\w*|а|ы)?\b",
            r"похи[тщ]\w*",
            r"хищени\w*",
            r"мошенни\w*",
            r"умышленн\w*",
            r"намеренн\w*",
            r"преднамеренн\w*",
            r"специально",
            r"с целью",
            r"преступ\w*",
            r"злоумышл\w*",
            r"виновн\w*",
            r"сговор\w*",
            r"обман\w*",
            r"фальшив\w*",
            r"поддел\w*",
            r"подделк\w*",
            r"недостач\w*",
            r"растрат\w*",
            r"присвоил\w*",
            r"присвоени\w*",
            r"нарушил\w*",
            r"\d[\d\s.,]*\s*(?:тенге|тг\b|₸|руб\w*|долл\w*|евро|usd|kzt)",
            r"сумм(?:а|ы|у|ой|е)\b",
            r"подлинн\w*",
            r"без\s+допуска",
            r"не\s+име\w+\s+(?:допуска|права)",
            r"неуполномоч\w*",
            r"не\s+допущ\w*",
            r"с\s+умыслом",
            r"намерен\w*",
        ],
    },
}

DEFAULT_CONFIG["completeness"] = {
    "required": ["ingest", "detect", "rules"],
    "optional": ["track", "pose", "scene", "temporal", "vlm"],
    "ok_values": ["ok", "done", "success", "true", "skipped_by_config"],
    "coverage_gate": True,
    "min_decoded": 0.98,
    "min_sampled": 0.9,
    "max_gap_s": 2.0,
    "unknown_coverage": "no_verdict",
}

DEFAULT_THRESHOLDS: dict[str, Any] = {
    "version": 1,
    "tuned": False,
    "default": 0.45,
    "per_type": {
        "V01": 0.45,
        "V02": 0.45,
        "V03": 0.45,
        "V04": 0.50,
        "V05": 0.50,
        "V06": 0.50,
        "V07": 0.45,
        "V08": 0.50,
        "V99": 0.60,
    },
}


def deep_merge(base: dict[str, Any], override: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Рекурсивно накладывает ``override`` на копию ``base`` (списки заменяются целиком)."""
    out = copy.deepcopy(base)
    if not override:
        return out
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> Optional[dict[str, Any]]:
    try:
        import yaml  # lazy
    except ImportError:
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, ValueError):
        return None
    except Exception:  # noqa: BLE001 — битый YAML не должен ронять пайплайн
        return None
    return data if isinstance(data, dict) else None


def load_config(path: Optional[str | os.PathLike[str]] = None) -> dict[str, Any]:
    """config.yaml поверх встроенных значений по умолчанию."""
    p = Path(path or os.environ.get("SENTINEL_FUSION_CONFIG") or CONFIG_PATH)
    return deep_merge(DEFAULT_CONFIG, _read_yaml(p))


def load_thresholds(path: Optional[str | os.PathLike[str]] = None) -> dict[str, Any]:
    """thresholds.yaml поверх встроенных значений по умолчанию."""
    p = Path(path or os.environ.get("SENTINEL_FUSION_THRESHOLDS") or THRESHOLDS_PATH)
    return deep_merge(DEFAULT_THRESHOLDS, _read_yaml(p))


def threshold_for(thresholds: dict[str, Any], type_id: str) -> float:
    """τ для типа: per_type[type_id] → default."""
    per_type = thresholds.get("per_type") or {}
    value = per_type.get(type_id, thresholds.get("default", 0.5))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(DEFAULT_THRESHOLDS["default"])
