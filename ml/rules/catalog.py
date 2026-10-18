"""Каталог типов нарушений для движка правил.

Источник истины — ``norms/catalog.yaml`` (владелец A13), сюда он приходит аргументом
``catalog`` в :func:`ml.rules.run_rules`. Если каталог пустой/отсутствует — пробуем
``norms.load_catalog()`` (лениво), иначе используем встроенный fallback **только с именами
типов**. Ссылок на нормы здесь нет и быть не должно (правило 5: нормы — только из
``norms/catalog.yaml``; RuleRef подставляет A12/fusion).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

__all__ = ["FALLBACK_CATALOG", "resolve_catalog", "type_info", "norm_status", "legal_basis_of", "SEVERITY_DEFAULT"]

SEVERITY_DEFAULT = "medium"

FALLBACK_CATALOG: dict[str, Any] = {
    "version": 1,
    "source": "ml.rules fallback (норм нет — только имена типов; ничего не верифицировано)",
    "types": {
        "V01": {"name": "Посторонний в кассовом помещении", "severity": "high"},
        "V02": {"name": "Кассир покинул рабочее место, оставив ценности без присмотра", "severity": "high"},
        "V03": {"name": "Сейф открыт без присмотра или при посторонних", "severity": "high"},
        "V04": {"name": "Дверь кассового помещения открыта во время операций", "severity": "medium"},
        "V05": {"name": "Использование мобильного телефона при работе с ценностями", "severity": "medium"},
        "V06": {"name": "Приём пищи или напитков на рабочем месте с наличными", "severity": "low"},
        "V07": {"name": "Нарушение порядка приёма-передачи инкассаторской сумки", "severity": "high"},
        "V08": {"name": "Пересчёт или передача наличных вне обзора камеры", "severity": "medium"},
        "V09": {"name": "Личные вещи на кассовом столе", "severity": "low"},
        "V99": {"name": "Иное подозрительное действие", "severity": "low"},
    },
}


def _try_norms_catalog() -> Optional[dict]:
    try:  # ленивый импорт: модуль A13 может отсутствовать в ветке
        from norms import load_catalog  # type: ignore
    except Exception:
        return None
    try:
        cat = load_catalog()
    except Exception:
        return None
    return cat if isinstance(cat, Mapping) and cat.get("types") else None


def resolve_catalog(catalog: Optional[Mapping[str, Any]]) -> tuple[dict, str]:
    """Вернуть (каталог, источник). Источник: ``arg`` | ``norms`` | ``fallback``."""
    if isinstance(catalog, Mapping) and isinstance(catalog.get("types"), Mapping) and catalog["types"]:
        return dict(catalog), "arg"
    cat = _try_norms_catalog()
    if cat is not None:
        return dict(cat), "norms"
    return FALLBACK_CATALOG, "fallback"


def type_info(catalog: Mapping[str, Any], type_id: str) -> dict[str, Any]:
    """Имя/severity типа; если в переданном каталоге типа нет — из fallback (с пометкой)."""
    types = catalog.get("types") or {}
    entry = types.get(type_id)
    in_catalog = isinstance(entry, Mapping)
    if not in_catalog:
        entry = FALLBACK_CATALOG["types"].get(type_id, {})
    name = str(entry.get("name") or type_id)
    severity = str(entry.get("severity") or SEVERITY_DEFAULT)
    return {"name": name, "severity": severity, "in_catalog": in_catalog}


def norm_status(catalog: Mapping[str, Any], type_id: str) -> dict[str, Any]:
    """Есть ли у типа подтверждённая норма в каталоге A13.

    ``status``:
    ``verified`` — запись ``rules`` с ``verified: true`` и непустым URL, а ``legal_status`` (если A13 его
    задал) равен ``confirmed``; ``candidate`` — ``legal_status: candidate`` (пункт найден, «требует
    сверки»); ``disabled`` — ``legal_status: disabled``; ``unverified`` — нормы есть, но не проверены;
    ``absent`` — типа/норм нет (в т.ч. fallback). Нормы здесь не используются и не выдумываются.
    """
    types = catalog.get("types") or {}
    entry = types.get(type_id)
    if not isinstance(entry, Mapping):
        return {"status": "absent", "n_rules": 0, "n_verified": 0, "legal_status": None}
    legal = str(entry.get("legal_status") or "").strip().lower() or None
    refs = [r for r in (entry.get("rules") or []) if isinstance(r, Mapping)]
    verified = [r for r in refs if r.get("verified") is True and str(r.get("url") or "").strip()]
    if legal == "disabled":
        status = "disabled"
    elif verified and legal in (None, "confirmed"):
        status = "verified"
    elif legal == "candidate":
        status = "candidate"
    elif refs:
        status = "unverified"
    else:
        status = "absent"
    return {"status": status, "n_rules": len(refs), "n_verified": len(verified), "legal_status": legal}


def legal_basis_of(status: str) -> str:
    """Правовая опора события: ``confirmed`` | ``candidate`` (требует сверки) | ``none``."""
    return {"verified": "confirmed", "candidate": "candidate"}.get(status, "none")
