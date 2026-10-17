"""Маппинг классов внешних моделей (COCO / Objects365 / дообученная) → LABELS контракта."""
from __future__ import annotations

import re
from typing import Iterable, Mapping, Optional

try:
    from contracts.sentinel_contracts import LABELS  # noqa
except ImportError:  # пока ветка A01 не смерджена
    from contracts.sentinel_contracts import LABELS  # noqa

# COCO-80 (имена как в id2label HF-моделей RT-DETR / D-FINE) → LABELS
COCO_TO_LABEL: dict[str, str] = {
    "person": "person",
    "cell phone": "cell_phone",
    "cup": "cup",
    "wine glass": "cup",
    "bowl": "cup",
    "bottle": "bottle",
    "banana": "food",
    "apple": "food",
    "sandwich": "food",
    "orange": "food",
    "broccoli": "food",
    "carrot": "food",
    "hot dog": "food",
    "pizza": "food",
    "donut": "food",
    "cake": "food",
    "laptop": "laptop",
    "handbag": "bag",
    "backpack": "bag",
    "suitcase": "bag",
    "book": "document",
}

# Синонимы (Objects365 / свои датасеты / опечатки) → LABELS
ALIASES: dict[str, str] = {
    "mobile phone": "cell_phone",
    "cellphone": "cell_phone",
    "phone": "cell_phone",
    "smartphone": "cell_phone",
    "handbag/satchel": "bag",
    "briefcase": "bag",
    "wallet/purse": "bag",
    "banknote": "cash",
    "banknotes": "cash",
    "money": "cash",
    "cash money": "cash",
    "money counter": "money_counter",
    "counting machine": "money_counter",
    "open safe": "safe_open",
    "collector bag": "cash_bag",
    "cash bag": "cash_bag",
    "key": "keys",
    "paper": "document",
    "papers": "document",
    "documents": "document",
    "glass door": "door",
    "open door": "door_open",
    "drink": "cup",
    "mug": "cup",
    "coffee cup": "cup",
}

_LABEL_SET = frozenset(LABELS)

# Метки, которые может дать общий COCO-детектор (RT-DETRv2 / D-FINE): только эти.
STANDARD_LABELS = frozenset(COCO_TO_LABEL.values())
# Нестандартные метки: НИКОГДА не берутся из общего детектора — только из OWLv2 (опция, сходство с запросом)
# или из модели, дообученной на проверенной разметке (finetuned ONNX).
NONSTANDARD_LABELS = frozenset(_LABEL_SET - STANDARD_LABELS)

# Отображения-«заместители»: исходный класс НЕ тождественен метке (книга ≠ документ, миска ≠ чашка).
# Такие детекции помечаются в Detection.source суффиксом «~proxy»; отключаются closed_set.proxy_labels=false.
# Важно: «bag» = любая сумка/рюкзак/чемодан (COCO), НЕ инкассаторская сумка; детектор не выводит
# роли/полномочия людей из предметов — это делает (или не делает) downstream с учётом неопределённости.
PROXY_SOURCES = frozenset({"book", "bowl", "paper", "papers", "documents", "drink", "key", "money",
                           "banknote", "banknotes", "cash money", "collector bag", "cash bag"})


def normalize_name(name: str) -> str:
    """'Cell_Phone ' → 'cell phone' (нижний регистр, '_'/'-' → пробел, схлопывание пробелов)."""
    s = re.sub(r"[_\-]+", " ", str(name).strip().lower())
    return re.sub(r"\s+", " ", s)


def map_label(name: str) -> Optional[str]:
    """Имя класса внешней модели → метка из LABELS или None (класс нам не нужен)."""
    if name in _LABEL_SET:
        return name
    n = normalize_name(name)
    as_label = n.replace(" ", "_")
    if as_label in _LABEL_SET:
        return as_label
    if n in COCO_TO_LABEL:
        return COCO_TO_LABEL[n]
    return ALIASES.get(n)


def is_proxy(name: str) -> bool:
    """True, если имя класса модели лишь похоже на метку (а не является ею/её подвидом)."""
    return normalize_name(name) in PROXY_SOURCES


def build_index_map(id2label: Mapping[int, str] | Iterable[str], allow_proxy: bool = True,
                    allowed: Optional[Iterable[str]] = None) -> dict[int, str]:
    """id2label модели → {class_index: label из LABELS} только для нужных классов.
    allowed — допустимые метки для этого бэкенда (общий детектор → STANDARD_LABELS)."""
    items = id2label.items() if isinstance(id2label, Mapping) else enumerate(id2label)
    allow = frozenset(allowed) if allowed is not None else None
    out: dict[int, str] = {}
    for idx, name in items:
        lab = map_label(name)
        if lab is None or (not allow_proxy and is_proxy(name)) or (allow is not None and lab not in allow):
            continue
        out[int(idx)] = lab
    return out


def source_tag(backend: str, orig_name: str) -> str:
    """'rtdetr_v2_r18vd:book~proxy' — исходный класс модели сохраняется в Detection.source."""
    n = normalize_name(orig_name)
    return f"{backend}:{n}" + ("~proxy" if is_proxy(n) else "")


def validate_label(label: str) -> str:
    if label not in _LABEL_SET:
        raise ValueError(f"label {label!r} not in contract LABELS")
    return label
