"""Политика лицензий Kassa Sentinel (правило 2 ТЗ). Только stdlib, без побочных эффектов.

Разрешены ТОЛЬКО Apache-2.0 / MIT / BSD. Всё остальное (в т.ч. «вроде бы коммерческие»
CC-BY, OpenRAIL, Llama/Gemma community) — красный флаг до ручного решения.

Решение принимается по КОНКРЕТНОЙ модели (repo@revision): заявленная в models.json лицензия
и лицензия из метаданных хаба для этой ревизии обе должны быть в allowlist и совпадать.
Шаблоны имён ниже — только ДОПОЛНИТЕЛЬНЫЙ красный флаг (deny), никогда не основание «разрешить».
"""
from __future__ import annotations

import re

ALLOWED = ("apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "bsd")

_ALIASES = {
    "apache-2.0": "apache-2.0", "apache2.0": "apache-2.0", "apache-2": "apache-2.0",
    "apache2": "apache-2.0", "apache 2.0": "apache-2.0", "apache 2": "apache-2.0",
    "apache license 2.0": "apache-2.0", "apache license, version 2.0": "apache-2.0",
    "apache software license": "apache-2.0", "apache license version 2.0": "apache-2.0",
    "mit": "mit", "mit license": "mit", "the mit license": "mit", "expat": "mit",
    "bsd-2-clause": "bsd-2-clause", "bsd 2-clause": "bsd-2-clause", "simplified bsd": "bsd-2-clause",
    "bsd-3-clause": "bsd-3-clause", "bsd 3-clause": "bsd-3-clause", "new bsd": "bsd-3-clause",
    "bsd-3": "bsd-3-clause", "modified bsd": "bsd-3-clause", "bsd-3-clause-clear": "bsd-3-clause",
    "bsd": "bsd", "bsd license": "bsd",
}

# Дополнительные красные флаги по имени (НЕ allowlist). Причина — для отчёта.
BANNED_MODEL_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"ultralytics", "ultralytics — AGPL-3.0"),
    (r"yolo-?v?(5|8|9|10|11)\b|yolov(5|6|7|8|9|10|11)|yolo11", "YOLOv5–v11 (AGPL/GPL)"),
    (r"yolo[-_]?world", "YOLO-World — GPL-3.0"),
    (r"videomae", "VideoMAE — CC-BY-NC"),
    (r"qwen2\.5-vl-3b", "Qwen2.5-VL-3B — research license"),
    (r"qwen2\.5-3b", "Qwen2.5-3B — research license"),
    (r"(^|[/_-])nc([/_-]|$)|non[-_ ]?commercial|cc-by-nc", "non-commercial"),
)

# pip-пакеты с лицензиями, несовместимыми с правилом 2.
BANNED_PACKAGES = {
    "ultralytics": "AGPL-3.0",
    "ultralytics-thop": "AGPL-3.0",
    "yolov5": "GPL-3.0",
    "mmyolo": "GPL-3.0",
    "yolov10": "AGPL-3.0",
}


def normalize_license(value: object) -> str:
    """'Apache 2.0' → 'apache-2.0'; неизвестное → как есть в нижнем регистре; пусто → ''."""
    if value is None:
        return ""
    s = str(value).strip().lower()
    s = re.sub(r"^license:", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return _ALIASES.get(s, s)


def is_allowed(value: object) -> bool:
    """SPDX-подобные выражения: 'A OR B' — достаточно одной разрешённой; 'A AND B' — нужны все."""
    s = normalize_license(value)
    if not s:
        return False
    s = s.strip("() ")
    if " or " in s:
        return any(is_allowed(p) for p in s.split(" or "))
    if " and " in s:
        return all(is_allowed(p) for p in s.split(" and "))
    return s in ALLOWED


def same_license(a: object, b: object) -> bool:
    na, nb = normalize_license(a), normalize_license(b)
    if na == nb:
        return True
    # 'bsd' обобщённо совпадает с любым BSD-вариантом
    bsd = {"bsd", "bsd-2-clause", "bsd-3-clause"}
    return na in bsd and nb in bsd


def banned_reasons(model_id: str) -> list[str]:
    s = (model_id or "").lower()
    return [why for pat, why in BANNED_MODEL_PATTERNS if re.search(pat, s)]
