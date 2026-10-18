"""Kassa Sentinel — нормативная база РК и каталог типов нарушений (владелец A13).

Публичный API (сигнатура заморожена контрактом):
    load_catalog() -> dict          # norms/catalog.yaml

Доп. хелперы для fusion / copilot / frontend (не обязательны):
    get_type(type_id)               -> dict типа (fallback на V99)
    type_name(type_id)              -> русское название типа
    rule_refs(type_id)              -> list[dict] ровно с ключами RuleRef ([] для disabled)
    legal_status(type_id)           -> "confirmed" | "candidate" | "disabled"
    exclusions(type_id=None)        -> что само по себе НЕ нарушение (общие + типа)
    legal_basis(type_id)            -> блок для UI/fusion: статус, пояснение, нормы, уровни проверки
    examples(type_id=None)          -> синтетические примеры positive/negative/unknown (для тестов A09/A12)
    render_description(type_id, t, facts) -> str по description_template
    catalog_source()                -> "yaml" | "fallback" (для meta.warnings)

Импорт без побочных эффектов. PyYAML импортируется лениво; если его нет или
файл каталога не найден/битый — используется встроенная копия norms/_fallback.py
(та же структура, синхронизируется командой ``python -m norms.validate --write-fallback``).
"""
from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Optional, Union

__all__ = [
    "CATALOG_PATH",
    "NORMS_DIR",
    "RULE_KEYS",
    "SEVERITIES",
    "SOURCES_DIR",
    "TYPE_IDS",
    "catalog_source",
    "examples",
    "exclusions",
    "legal_basis",
    "legal_status",
    "format_timecode",
    "get_type",
    "load_catalog",
    "render_description",
    "rule_refs",
    "type_name",
]

log = logging.getLogger(__name__)

NORMS_DIR = Path(__file__).resolve().parent
CATALOG_PATH = NORMS_DIR / "catalog.yaml"
SOURCES_DIR = NORMS_DIR / "sources"

#: Ключи нормы — ровно поля contracts.RuleRef (можно делать RuleRef(**rule)).
RULE_KEYS: tuple[str, ...] = ("document", "clause", "url", "quote", "verified")
#: Обязательные типы нарушений (ключи каталога = Event.type_id).
TYPE_IDS: tuple[str, ...] = ("V01", "V02", "V03", "V04", "V05", "V06", "V07", "V08", "V09", "V99")
SEVERITIES: tuple[str, ...] = ("low", "medium", "high", "critical")
#: Переменная окружения для подмены пути к каталогу (тесты, интегратор).
ENV_CATALOG = "SENTINEL_NORMS_CATALOG"

_cache: dict[str, dict[str, Any]] = {}
_last_source: str = "none"


def _catalog_file(path: Union[str, os.PathLike, None]) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get(ENV_CATALOG)
    return Path(env) if env else CATALOG_PATH


def _read_yaml(path: Path) -> Optional[dict[str, Any]]:
    """Читает YAML-каталог. None — если файла/PyYAML нет или структура битая."""
    if not path.is_file():
        log.warning("norms: каталог %s не найден — используется встроенный fallback", path)
        return None
    try:
        import yaml  # lazy: PyYAML может отсутствовать в урезанном окружении
    except ImportError:
        log.warning("norms: PyYAML не установлен — используется встроенный fallback")
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        log.warning("norms: не удалось прочитать %s (%s) — используется fallback", path, exc)
        return None
    if not isinstance(data, dict) or not isinstance(data.get("types"), dict):
        log.warning("norms: %s не содержит словаря 'types' — используется fallback", path)
        return None
    return data


def load_catalog(path: Union[str, os.PathLike, None] = None, *, use_cache: bool = True) -> dict[str, Any]:
    """Каталог типов нарушений и норм РК (norms/catalog.yaml).

    Возвращает глубокую копию — потребители могут её мутировать без влияния на других.
    Форма: ``{"version": 1, "types": {"V01": {...}, ...}, "documents": {...}, ...}``.
    Каждый тип: name, observable, not_observable, description_template ({t}, {facts}),
    severity, rules[] — элементы rules имеют ровно ключи :data:`RULE_KEYS`.
    """
    global _last_source
    file = _catalog_file(path)
    key = str(file.resolve()) if file.exists() else str(file)
    if use_cache and key in _cache:
        _last_source = _cache[key].get("_source", "yaml")
        return copy.deepcopy(_cache[key]["catalog"])
    data = _read_yaml(file)
    source = "yaml"
    if data is None:
        from ._fallback import CATALOG

        data = copy.deepcopy(CATALOG)
        source = "fallback"
    _cache[key] = {"catalog": data, "_source": source}
    _last_source = source
    return copy.deepcopy(data)


def catalog_source() -> str:
    """Откуда взят последний загруженный каталог: "yaml" | "fallback" | "none"."""
    return _last_source


def clear_cache() -> None:
    _cache.clear()


def get_type(type_id: str, catalog: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Описание типа; неизвестный type_id → V99 (catch-all), чтобы fusion не падал."""
    types = (catalog or load_catalog())["types"]
    if type_id in types:
        return types[type_id]
    log.warning("norms: неизвестный type_id %r — используется V99", type_id)
    return types["V99"]


def type_name(type_id: str, catalog: Optional[dict[str, Any]] = None) -> str:
    return str(get_type(type_id, catalog).get("name", type_id))


def legal_status(type_id: str, catalog: Optional[dict[str, Any]] = None) -> str:
    """confirmed | candidate | disabled. disabled — норма к детекции не прикрепляется (только наблюдение)."""
    return str(get_type(type_id, catalog).get("legal_status") or "disabled")


def rule_refs(type_id: str, catalog: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    """Нормы типа в форме contracts.RuleRef: ``RuleRef(**r) for r in rule_refs("V08")``.

    Для legal_status=disabled всегда [] — неподтверждённая норма не выдаётся как основание.
    """
    t = get_type(type_id, catalog)
    if str(t.get("legal_status") or "disabled") == "disabled":
        return []
    return [{k: r.get(k) for k in RULE_KEYS} for r in (t.get("rules") or [])]


def exclusions(type_id: Optional[str] = None, catalog: Optional[dict[str, Any]] = None) -> list[str]:
    """Ситуации, которые сами по себе НЕ являются нарушением: общие + специфичные для типа."""
    cat = catalog or load_catalog()
    out = [str(x) for x in (cat.get("exclusions") or [])]
    if type_id is not None:
        out += [str(x) for x in (get_type(type_id, cat).get("exclusions") or [])]
    return out


def format_timecode(t: Union[float, int, str]) -> str:
    """22.4 → "00:22"; 125 → "02:05". Строка возвращается как есть."""
    if isinstance(t, str):
        return t
    total = max(0, int(round(float(t))))
    return f"{total // 60:02d}:{total % 60:02d}"


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:  # неизвестный плейсхолдер не роняет fusion
        return "{" + key + "}"


def render_description(
    type_id: str,
    t: Union[float, int, str],
    facts: Iterable[str] = (),
    catalog: Optional[dict[str, Any]] = None,
) -> str:
    """Текст описания по description_template типа: {t} — таймкод, {facts} — факты через '; '."""
    tpl = str(get_type(type_id, catalog).get("description_template") or "{t}: {facts}")
    facts_list = [str(f).strip().rstrip(".") for f in facts if str(f).strip()]
    facts_str = "; ".join(facts_list) if facts_list else "см. кадры-доказательства"
    return tpl.format_map(_SafeDict(t=format_timecode(t), facts=facts_str)).strip()


def legal_basis(type_id: str, catalog: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Всё о нормативном основании типа одним блоком (для fusion.meta, frontend, copilot).

    ``rules``/``checks`` пусты для disabled; ``checks[i]`` описывает уровень проверки ``rules[i]``
    (точность цитаты, актуальность, применимость, видимость условий). Не юридическое заключение.
    """
    cat = catalog or load_catalog()
    t = get_type(type_id, cat)
    status = legal_status(type_id, cat)
    enabled = status != "disabled"
    return {
        "type_id": type_id if type_id in cat["types"] else "V99",
        "status": status,
        "note": str(t.get("legal_note") or "").strip(),
        "rules": rule_refs(type_id, cat),
        "checks": list(t.get("rule_checks") or []) if enabled else [],
        "exclusions": exclusions(type_id, cat),
        "disclaimer": str(cat.get("disclaimer") or "").strip(),
    }


def examples(type_id: Optional[str] = None, catalog: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    """Синтетические примеры спецификации (не реальные сцены): kind, facts, signals (null = неизвестно), expected.

    Для одного типа — его примеры; без аргумента — все, с полем ``type_id``.
    """
    cat = catalog or load_catalog()
    items = [(type_id, get_type(type_id, cat))] if type_id else list(cat["types"].items())
    return [{"type_id": tid, **e} for tid, t in items for e in (t.get("examples") or [])]
