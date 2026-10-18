"""Список типов нарушений для валидатора и страницы подтверждения — из актуального norms/catalog.yaml.

Порядок поиска: явный путь → <root>/norms/catalog.yaml → снимок ниже (integration-cloud @ 2055fc4, A13).
Источник всегда возвращается вместе со списком, чтобы было видно, с чем сверялись.
"""
from __future__ import annotations

from pathlib import Path

# Снимок types.<ID>.name из norms/catalog.yaml (integration-cloud @ 2055fc4). Обновлять вместе с каталогом.
FALLBACK: dict[str, str] = {
    "V01": "Посторонний на стороне кассира / в кассовой зоне",
    "V02": "Кассир покинул рабочее место, оставив ценности без присмотра",
    "V03": "Сейф открыт без присмотра или за спиной работающего кассира",
    "V04": "Вход в кассовое помещение при открытых ценностях",
    "V05": "Мобильный телефон в руке кассира при работе с ценностями",
    "V06": "Еда или напитки на рабочем месте с ценностями",
    "V07": "Нарушение порядка передачи инкассаторской сумки",
    "V08": "Пересчёт наличных не в присутствии клиента",
    "V09": "Личные вещи на рабочем месте кассира",
    "V99": "Иное видимое подозрительное действие",
}
FALLBACK_SOURCE = "встроенный снимок norms/catalog.yaml (integration-cloud @ 2055fc4)"


def missing_vs_snapshot(types: dict[str, str]) -> list[str]:
    """id из снимка, которых нет в загруженном каталоге (признак устаревшего каталога, например без V09)."""
    return [k for k in FALLBACK if k not in types]


def parse_catalog(data: dict) -> dict[str, str]:
    types = (data or {}).get("types") or {}
    out = {str(k).upper(): str((v or {}).get("name", "")).strip() for k, v in types.items()}
    if not out:
        raise ValueError("в каталоге нет раздела types")
    return dict(sorted(out.items()))


def load_types(catalog: Path | None = None, root: Path | None = None) -> tuple[dict[str, str], str]:
    """{type_id: название} и описание источника."""
    if catalog is not None:  # явно указанный путь обязан существовать — без тихого отката
        if not Path(catalog).is_file():
            raise FileNotFoundError(f"каталог не найден: {catalog}")
        p = Path(catalog)
    elif root is not None and (Path(root) / "norms" / "catalog.yaml").is_file():
        p = Path(root) / "norms" / "catalog.yaml"
    else:
        return dict(FALLBACK), FALLBACK_SOURCE
    import yaml  # pyyaml есть в локе L1 и в ядре pyproject

    with open(p, encoding="utf-8") as f:
        return parse_catalog(yaml.safe_load(f)), str(p.resolve())
