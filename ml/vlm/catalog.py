"""Каталог типов нарушений для промптов.

Источник истины — norms/catalog.yaml (A13) через ``norms.load_catalog()``. Пока его нет в ветке —
встроенный fallback с теми же ключами V01..V08, V99, БЕЗ ссылок на нормы (нормы выдаёт только A13;
VLM нормы не цитирует вообще).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

log = logging.getLogger(__name__)

TYPE_ID_RE = re.compile(r"^V\d{2}$")

FALLBACK_CATALOG: dict[str, Any] = {
    "version": 1,
    "types": {
        "V01": {
            "name": "Посторонний в кассовом помещении",
            "observable": ["человек, не являющийся кассиром, находится за кассовым столом, в зоне кассира или у ценностей"],
            "not_observable": ["наличие допуска у лица", "личность человека"],
        },
        "V02": {
            "name": "Рабочее место кассира оставлено без присмотра с ценностями",
            "observable": ["кассира нет на рабочем месте, при этом на столе видны наличные или ценности"],
            "not_observable": ["сумма наличных", "причина отсутствия кассира"],
        },
        "V03": {
            "name": "Сейф открыт без присмотра или при посторонних",
            "observable": ["дверца сейфа открыта, рядом нет кассира или рядом находятся посторонние"],
            "not_observable": ["содержимое сейфа", "кто имеет ключи/код"],
        },
        "V04": {
            "name": "Дверь кассового помещения открыта во время операций",
            "observable": ["дверь или проход в кассовое помещение открыты, через них проходят люди во время работы с наличными"],
            "not_observable": ["кто разрешил проход"],
        },
        "V05": {
            "name": "Использование мобильного телефона при работе с ценностями",
            "observable": ["кассир держит мобильный телефон в руке, смотрит в него или подносит к уху"],
            "not_observable": ["с кем разговор", "содержание экрана"],
        },
        "V06": {
            "name": "Приём пищи или напитков на рабочем месте с наличными",
            "observable": ["кассир ест или пьёт, на столе еда, чашка или бутылка рядом с наличными"],
            "not_observable": ["что именно в чашке"],
        },
        "V07": {
            "name": "Нарушение порядка приёма-передачи инкассаторской сумки",
            "observable": ["инкассаторская сумка оставлена без присмотра, передаётся вне стола/процедуры или вскрывается при посторонних"],
            "not_observable": ["содержимое сумки", "пломбы и документы", "полномочия инкассатора"],
        },
        "V08": {
            "name": "Пересчёт или передача наличных вне обзора камеры",
            "observable": ["руки с наличными уходят из обзора камеры или закрыты телом/предметами во время пересчёта или передачи"],
            "not_observable": ["сумма", "правильность пересчёта"],
        },
        "V09": {
            "name": "Личные вещи на кассовом столе",
            "observable": ["на кассовом столе лежат личные вещи (куртка, сумка, пакет), не относящиеся к работе"],
            "not_observable": ["кому принадлежат вещи", "их содержимое"],
        },
        "V99": {
            "name": "Иное видимое подозрительное действие",
            "observable": ["конкретное видимое действие, нарушающее кассовую дисциплину, не подходящее под V01–V08"],
            "not_observable": ["умысел", "личность", "суммы и проводки"],
        },
    },
}

_override: Optional[dict[str, Any]] = None


def set_catalog(catalog: Optional[dict[str, Any]]) -> None:
    """Интегратор может передать каталог явно (verify/discover его в сигнатуре не получают)."""
    global _override
    _override = catalog


def load_catalog() -> dict[str, Any]:
    if _override:
        return _override
    try:
        from norms import load_catalog as _norms_load  # type: ignore

        cat = _norms_load()
        if isinstance(cat, dict) and cat.get("types"):
            return cat
    except Exception as e:  # noqa: BLE001 — норм ещё нет в ветке / битый yaml
        log.debug("norms.load_catalog unavailable (%s); using fallback catalog", e)
    return FALLBACK_CATALOG


def catalog_types(catalog: Optional[dict[str, Any]] = None) -> dict[str, dict[str, Any]]:
    cat = catalog or load_catalog()
    types = cat.get("types") or {}
    out: dict[str, dict[str, Any]] = {}
    for tid, spec in types.items():
        if TYPE_ID_RE.match(str(tid)) and isinstance(spec, dict):
            out[str(tid)] = spec
    # гарантируем V99 (catch-all) даже если в каталоге его нет
    out.setdefault("V99", FALLBACK_CATALOG["types"]["V99"])
    return dict(sorted(out.items()))


def type_name(type_id: str, catalog: Optional[dict[str, Any]] = None) -> str:
    spec = catalog_types(catalog).get(type_id) or {}
    return str(spec.get("name") or type_id)


def _as_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if str(x).strip()]
    return [str(v)]


def describe_types(catalog: Optional[dict[str, Any]] = None, only: Optional[list[str]] = None) -> str:
    """Текстовый блок с описаниями типов для подстановки в промпт."""
    lines = []
    for tid, spec in catalog_types(catalog).items():
        if only and tid not in only:
            continue
        obs = "; ".join(_as_list(spec.get("observable")))
        nobs = "; ".join(_as_list(spec.get("not_observable")))
        line = f"- {tid} «{spec.get('name', tid)}»: видно — {obs or 'см. название'}"
        if nobs:
            line += f". НЕ видно по камере (не утверждать): {nobs}"
        lines.append(line)
    return "\n".join(lines)
