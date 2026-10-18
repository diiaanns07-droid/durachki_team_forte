"""Доступ к каталогу нарушений (norms/catalog.yaml, владелец A13) + встроенный fallback.

Fallback содержит ТОЛЬКО названия, шаблоны описаний и серьёзность — без ссылок на нормы:
ссылки на нормативные акты берутся исключительно из norms/catalog.yaml (правило 5),
поэтому при отсутствии каталога ``rules`` у нарушений пустые, а в meta пишется предупреждение.
"""
from __future__ import annotations

from typing import Any, Optional

from ._types import RuleRef

FALLBACK_VERSION = "fusion-fallback-1"

# Ключи совпадают с norms/catalog.yaml. Поле rules намеренно пустое (см. docstring).
FALLBACK_TYPES: dict[str, dict[str, Any]] = {
    "V01": {
        "name": "Посторонний в зоне кассира",
        "why": "Нахождение лица, не являющегося кассиром, в зоне кассира или у ценностей — "
               "признак возможного нарушения порядка доступа в кассовое помещение.",
        "severity": "high",
    },
    "V02": {
        "name": "Кассир покинул рабочее место, оставив наличные без присмотра",
        "why": "Наличные или ценности на рабочем месте без присмотра кассира — "
               "признак возможного нарушения порядка хранения ценностей.",
        "severity": "high",
    },
    "V03": {
        "name": "Сейф открыт и оставлен без присмотра",
        "why": "Открытый сейф без присмотра кассира или в присутствии посторонних — "
               "признак возможного нарушения порядка хранения ценностей.",
        "severity": "high",
    },
    "V04": {
        "name": "Дверь кассового помещения открыта во время операций",
        "why": "Открытая дверь кассового помещения во время работы с ценностями — "
               "признак возможного нарушения режима доступа.",
        "severity": "medium",
    },
    "V05": {
        "name": "Использование мобильного телефона при работе с ценностями",
        "why": "Использование мобильного телефона при работе с наличными — "
               "признак возможного нарушения порядка совершения кассовых операций.",
        "severity": "medium",
    },
    "V06": {
        "name": "Приём пищи или напитков на рабочем месте с наличными",
        "why": "Приём пищи или напитков рядом с наличными — "
               "признак возможного нарушения порядка работы с ценностями.",
        "severity": "low",
    },
    "V07": {
        "name": "Нарушение порядка приёма-передачи инкассаторской сумки",
        "why": "Действия с инкассаторской сумкой вне штатной процедуры приёма-передачи — "
               "признак возможного нарушения порядка инкассации.",
        "severity": "high",
    },
    "V08": {
        "name": "Пересчёт наличных не в присутствии клиента",
        "why": "Операции с наличными вне обзора камеры или с сокрытием рук — "
               "признак возможного нарушения порядка совершения кассовых операций.",
        "severity": "medium",
    },
    "V09": {
        "name": "Личные вещи на рабочем месте кассира",
        "why": "Личные вещи (сумка, одежда) на кассовом столе рядом с ценностями — "
               "признак возможного нарушения порядка организации рабочего места.",
        "severity": "low",
    },
    "V99": {
        "name": "Иное видимое подозрительное действие",
        "why": "Наблюдаемое действие нетипично для штатной работы кассы — "
               "признак возможного нарушения.",
        "severity": "medium",
    },
}


def _types_of(catalog: Any) -> dict[str, dict[str, Any]]:
    """Нормализует разные формы каталога к {type_id: {...}}."""
    if not isinstance(catalog, dict):
        return {}
    types = catalog.get("types", None)
    if types is None and any(str(k).upper().startswith("V") for k in catalog.keys()):
        types = catalog  # передали сразу словарь типов
    if isinstance(types, list):
        out: dict[str, dict[str, Any]] = {}
        for item in types:
            if isinstance(item, dict):
                tid = item.get("type_id") or item.get("id") or item.get("key")
                if tid:
                    out[str(tid)] = item
        return out
    if isinstance(types, dict):
        return {str(k): v for k, v in types.items() if isinstance(v, dict)}
    return {}


def load_norms_catalog() -> Any:
    """Каталог A13 (``norms.load_catalog()``). Единственная точка загрузки — тесты подменяют её monkeypatch-ем."""
    from norms import load_catalog  # type: ignore[import-not-found]  # lazy, модуль A13

    return load_catalog()


def resolve_catalog(catalog: Optional[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Возвращает (types, warnings).

    * Каталог с непустым ``types`` используется как есть (без перезагрузки и без слияния с A13).
    * ``None`` / ``{}`` / ``{"types": {}}`` (так A03 передаёт недоступный каталог) / мусор → ``norms.load_catalog()``.
    * A13 недоступен или пуст → встроенный fallback: только названия, шаблоны и серьёзность, ``rules: []``
      (нормы не выдумываются) + предупреждение ``norms catalog unavailable``.
    """
    warnings: list[str] = []
    types = _types_of(catalog)
    if not types:
        try:
            types = _types_of(load_norms_catalog())
        except Exception:  # noqa: BLE001 — norms может отсутствовать в окружении
            types = {}
        if not types:
            warnings.append("norms catalog unavailable: используются встроенные названия типов, "
                            "ссылки на нормы не подставлены")
            return {k: dict(v, rules=[]) for k, v in FALLBACK_TYPES.items()}, warnings
    return types, warnings


def known_type(types: dict[str, dict[str, Any]], type_id: str) -> bool:
    return type_id in types or type_id in FALLBACK_TYPES


def type_info(types: dict[str, dict[str, Any]], type_id: str) -> dict[str, Any]:
    """Сведения о типе: каталог поверх fallback (имя, шаблон, серьёзность)."""
    base = dict(FALLBACK_TYPES.get(type_id) or FALLBACK_TYPES["V99"])
    entry = types.get(type_id)
    if entry:
        base.update({k: v for k, v in entry.items() if v not in (None, "", [])})
        if "rules" not in entry:
            base["rules"] = []
    else:
        base["rules"] = []
    if not base.get("name"):
        base["name"] = FALLBACK_TYPES["V99"]["name"]
    return base


def type_name(types: dict[str, dict[str, Any]], type_id: str) -> str:
    return str(type_info(types, type_id).get("name") or type_id)


def severity(types: dict[str, dict[str, Any]], type_id: str) -> str:
    sev = str(type_info(types, type_id).get("severity") or "medium").lower()
    return sev if sev in {"low", "medium", "high"} else "medium"


def is_verified(value: Any) -> bool:
    """Строго: только YAML true / "true" / "yes" / 1. Строка "false" или мусор → False (не выдаём за сверенную)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    return str(value).strip().lower() in {"true", "yes", "да", "1", "verified"}


def rule_refs(types: dict[str, dict[str, Any]], type_id: str) -> list[RuleRef]:
    """Нормы ТОЛЬКО из каталога: поля document/clause/url/quote/verified как в нём."""
    entry = types.get(type_id) or {}
    out: list[RuleRef] = []
    for raw in entry.get("rules") or []:
        if not isinstance(raw, dict):
            continue
        document = str(raw.get("document") or "").strip()
        url = str(raw.get("url") or "").strip()
        if not document or not url:
            continue
        clause = raw.get("clause")
        out.append(
            RuleRef(
                document=document,
                clause=str(clause) if clause not in (None, "") else None,
                url=url,
                quote=str(raw.get("quote") or ""),
                verified=is_verified(raw.get("verified", False)),
            )
        )
    return out
