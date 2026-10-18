"""Библиотека промптов ml/vlm/prompts/<id>.<version>.<lang>.md с front-matter и {{плейсхолдерами}}.

Описания V0x берутся из каталога (norms/catalog.yaml или fallback) и подставляются в {{type_catalog}}.
Few-shot кадров из тестовых видео нет и не будет — только текстовые инструкции.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from .catalog import catalog_types, describe_types
from .config import PROMPTS_DIR

_PH_RE = re.compile(r"\{\{\s*([a-z_][a-z0-9_]*)\s*\}\}")

# Пункты чек-листа discover: ключ -> (RU, EN). Ключи совпадают с discover.checklist в config.yaml.
CHECKLIST_TEXT: dict[str, tuple[str, str]] = {
    "phone": ("кассир держит мобильный телефон в руке, смотрит в него или подносит к уху",
              "the cashier holds a mobile phone, looks at it or holds it to the ear"),
    "food_drink": ("кассир ест или пьёт; еда, чашка или бутылка на рабочем столе рядом с наличными",
                   "the cashier eats or drinks; food, a cup or a bottle on the desk next to cash"),
    "safe_open_unattended": ("дверца сейфа открыта, а кассира рядом нет или рядом посторонний",
                             "the safe door is open with no cashier nearby or with an outsider nearby"),
    "outsider_in_cashier_area": ("человек, не являющийся кассиром, находится за кассовым столом / в зоне кассира / у сейфа",
                                 "a person who is not the cashier is behind the cash desk / in the cashier area / at the safe"),
    "cash_unattended": ("на столе лежат наличные, а кассира на рабочем месте нет",
                        "cash is lying on the desk while the cashier is away from the workplace"),
    "bag_handover_violation": ("инкассаторская сумка оставлена без присмотра, передаётся вне стола или вскрывается при посторонних",
                               "a cash-collection bag is left unattended, handed over away from the desk or opened in front of outsiders"),
    "door_open": ("дверь кассового помещения открыта и через неё проходят люди во время работы с наличными",
                  "the cash room door is open and people pass through while cash is handled"),
    "hands_hidden": ("пересчёт или передача наличных происходит вне обзора камеры или руки с деньгами закрыты",
                     "counting or handing over cash happens outside the camera view or hands with cash are concealed"),
    "other": ("иное конкретное видимое действие, нарушающее кассовую дисциплину",
              "another concrete visible action that breaks cash-handling discipline"),
}


@dataclass(frozen=True)
class PromptTemplate:
    id: str
    version: str
    lang: str
    placeholders: tuple[str, ...]
    body: str
    path: str

    @property
    def key(self) -> str:
        return f"{self.id}.{self.version}.{self.lang}"

    def render(self, **values: Any) -> str:
        missing = [p for p in self.placeholders if p not in values]
        if missing:
            raise KeyError(f"prompt {self.key}: missing placeholders {missing}")

        def sub(m: re.Match[str]) -> str:
            name = m.group(1)
            if name not in values:
                raise KeyError(f"prompt {self.key}: unknown placeholder {{{{{name}}}}}")
            return str(values[name])

        return _PH_RE.sub(sub, self.body).strip()


def _parse(path: Path) -> PromptTemplate:
    text = path.read_text(encoding="utf-8")
    meta: dict[str, Any] = {}
    body = text
    if text.startswith("---"):
        _, fm, body = text.split("---", 2)
        import yaml  # лениво

        meta = yaml.safe_load(fm) or {}
    stem = path.name[:-3].split(".")  # id.version.lang
    pid = str(meta.get("id") or stem[0])
    version = str(meta.get("version") or (stem[1] if len(stem) > 1 else "v1"))
    lang = str(meta.get("lang") or (stem[2] if len(stem) > 2 else "ru"))
    phs = tuple(meta.get("placeholders") or [])
    used = set(_PH_RE.findall(body))
    if set(phs) != used:
        raise ValueError(f"{path.name}: declared placeholders {sorted(phs)} != used {sorted(used)}")
    return PromptTemplate(pid, version, lang, phs, body.strip("\n"), str(path))


@lru_cache(maxsize=None)
def _library(prompts_dir: str) -> dict[str, PromptTemplate]:
    lib: dict[str, PromptTemplate] = {}
    for p in sorted(Path(prompts_dir).glob("*.md")):
        if p.name.lower() == "readme.md":
            continue
        t = _parse(p)
        lib[t.key] = t
    return lib


def list_prompts(prompts_dir: Optional[str] = None) -> list[str]:
    return sorted(_library(str(prompts_dir or PROMPTS_DIR)))


def get_prompt(pid: str, version: str = "v1", lang: str = "ru", prompts_dir: Optional[str] = None) -> PromptTemplate:
    lib = _library(str(prompts_dir or PROMPTS_DIR))
    for key in (f"{pid}.{version}.{lang}", f"{pid}.{version}.ru", f"{pid}.v1.{lang}", f"{pid}.v1.ru"):
        if key in lib:
            return lib[key]
    raise KeyError(f"prompt {pid}.{version}.{lang} not found in {prompts_dir or PROMPTS_DIR}")


def _fmt_times(times: list[float]) -> str:
    return ", ".join(f"{i + 1}: {t:.1f}" for i, t in enumerate(times))


def checklist_block(checklist: dict[str, str], lang: str = "ru") -> str:
    lines = []
    for key, tid in checklist.items():
        ru, en = CHECKLIST_TEXT.get(key, (key, key))
        lines.append(f'- "{key}" ({tid}): {en if lang == "en" else ru}')
    return "\n".join(lines)


BOXES_NOTE = {
    "ru": ("Красными рамками подсвечены объекты, на которые указал автоматический детектор "
           "(подпись — что это по его мнению; детектор может ошибаться).",
           "Рамок на кадрах нет — оцени кадр целиком."),
    "en": ("Red boxes highlight objects flagged by an automatic detector (the label is the detector's guess; "
           "it can be wrong).", "There are no boxes on the frames — judge the whole frame."),
}


def render_verify(event_type: str, candidate_facts: list[str], frame_times: list[float],
                  catalog: Optional[dict[str, Any]] = None, lang: str = "ru", version: str = "v1",
                  has_boxes: bool = True) -> tuple[str, str]:
    """-> (system, user) для verify."""
    types = catalog_types(catalog)
    spec = types.get(event_type) or types["V99"]
    facts = "; ".join(candidate_facts) if candidate_facts else ("нет" if lang == "ru" else "none")
    note = BOXES_NOTE.get(lang, BOXES_NOTE["ru"])
    user = get_prompt("verify", version, lang).render(
        n_frames=len(frame_times),
        frame_times=_fmt_times(frame_times),
        boxes_note=note[0] if has_boxes else note[1],
        candidate_type=event_type,
        candidate_name=spec.get("name", event_type),
        candidate_facts=facts,
        type_catalog=describe_types(catalog),
    )
    return get_prompt("system", version, lang).render(), user


def render_discover(checklist: dict[str, str], frame_times: list[float], catalog: Optional[dict[str, Any]] = None,
                    lang: str = "ru", version: str = "v1") -> tuple[str, str]:
    user = get_prompt("discover", version, lang).render(
        n_frames=len(frame_times),
        frame_times=_fmt_times(frame_times),
        checklist=checklist_block(checklist, lang),
        type_catalog=describe_types(catalog),
    )
    return get_prompt("system", version, lang).render(), user


def retry_suffix(lang: str = "ru", version: str = "v1") -> str:
    return get_prompt("retry_suffix", version, lang).render()
