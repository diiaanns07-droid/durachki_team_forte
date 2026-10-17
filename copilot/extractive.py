"""Детерминированный составитель ответа (без генерации) + классификатор намерений вопроса.

Используется: (1) как ``mock``-бэкенд при SENTINEL_MOCK=1; (2) как явно включаемый бэкенд
``extractive`` (только по конфигу, НЕ автоматический fallback); (3) для обязательных отказов
на вопросы о невидимом (сумма, подлинность, умысел, личность…) — до вызова LLM.
Ответ собирается только из фактов результата и найденных норм, с маркерами [Nk]/[Vk].
"""
from __future__ import annotations

import re
from typing import Any, Optional, Sequence

from .corpus import Corpus
from .prompts import (NormRef, REFUSAL_NO_DATA, REFUSAL_SCOPE, REFUSAL_SOURCES, REFUSAL_VIDEO, completeness,
                      completeness_text, explain_for, explain_lines, fmt_tc, review_hints, to_float,
                      unobservable_items, verdict_text)

# порядок важен: первое совпадение побеждает
_INTENT_PATTERNS: list[tuple[str, str]] = [
    ("identity", r"кто\s+(это|этот|эта|эти|он|она|они|такой|такая|таков\w*)\b|кем\s+явля|как\s+(его|её|ее|их)?\s*зовут|фамили|\bфио\b|личност|"
                 r"опозна|распозна\w*\s+лиц|идентифиц|по\s+лицу|узна(ть|й)\s+(человека|лицо)"),
    ("unobservable", r"сколько\s+(денег|наличн|купюр|тенге|средств)|как(ую|ая)\s+сумм|сумм[ауы]\s+(денег|наличн|в\s+сумке)|"
                     r"подлинн|фальшив|поддел|(купюр|банкнот|деньг)\w*\s+(ли\s+)?настоящ|настоящ\w*\s+(ли\s+)?(купюр|банкнот|деньг)|умыс|умышлен|намеренн|нарочно|специально|хотел[аи]?\s+(ли\s+)?(укра|похит|скрыть)|"
                     r"собирал(ся|ась)|укра(л|ла|сть)|похитил|кража|ворует|проводк|допуск|уполномоч|имеет\s+ли\s+(он|она)?\s*право|"
                     r"что\s+(?:\w+\s+){0,2}(?:говорил|сказал|обсуждал)|о\s+ч[её]м\s+(?:\w+\s+){0,2}(?:говор|разговар|беседовал)|"
                     r"содержани\w*\s+разговор"),
    ("offtopic", r"\bстих|анекдот|погод|рецепт|переведи|напиши\s+(код|программ|письмо|эссе)|сочини|курс\s+(доллар|валют)"),
    ("fine", r"штраф|наказан|санкци|ответственност|уволь|выговор"),
    ("exclusions", r"когда\s+(?:\w+\s+){0,3}не\s+(?:является\s+|считается\s+)?нарушени|не\s+считается\s+нарушени|"
                   r"исключени|в\s+каких\s+случаях"),
    ("coverage", r"\bполн(ый|ая|ое|ота|оту|ого|ом|ым|ые)\b|\bполон\b|неполн|покрыт|весь\s+ролик|вс[её]\s+видео"),
    ("evidence", r"кадр|доказательств|скриншот|фото|изображени"),
    ("checklist", r"что\s+(мне\s+)?(проверить|делать|сделать)|как\s+проверить|рекомендац|следующ\w+\s+шаг|"
                  r"на\s+что\s+обратить|просмотреть|вручную"),
    ("hints", r"спорн|сомнительн|кандидат|ниже\s+порога|не\s+вошл|отсеян|отброшен"),
    ("confidence", r"уверен|вероятност|точност|насколько\s+(надёжн|надежн|точн)|ошиб(ка|ся|лась)|ложн"),
    ("when", r"когда|в\s+какой\s+момент|на\s+какой\s+секунд|таймкод|в\s+какое\s+время|время\s+нарушени"),
    ("why", r"почему|на\s+каком\s+основании|за\s+что|объясни|поясни|что\s+не\s+так|в\s+ч[её]м\s+(нарушение|проблема)|чем\s+нарушен"),
    ("norm", r"норм|пункт|правил|закон|документ|стать[яи]|требовани|регламент|инструкц|запрещ|разрешен|можно\s+ли|"
             r"допускается|ссылк|основани"),
    ("summary", r"сколько\s+нарушени|какие\s+нарушени|есть\s+ли\s+нарушени|вердикт|итог|что\s+(найдено|обнаружено|видно)|"
                r"резюм|кратко|опиши|что\s+произошло|нарушени"),
]

_TYPE_HINTS: dict[str, str] = {
    "V01": r"посторон|клиент\w*\s+за|чуж|неуполномоч",
    "V02": r"покинул|отош[её]л|ушл?[аи]?\b|без\s+присмотр|рабоч\w+\s+мест",
    "V03": r"сейф",
    "V04": r"двер|перегородк",
    "V05": r"телефон|смартфон|мобильн",
    "V06": r"\bе(ст|л|да)\b|пь[её]т|напит|кофе|чай|еда|пищ",
    "V07": r"инкасс|сумк",
    "V08": r"пересч|скрыт|вне\s+обзора|руки",
    "V09": r"личн\w*\s+вещ|куртк|рюкзак|одежд|пакет",
}

_ORDINALS = {"перв": 1, "втор": 2, "трет": 3, "четв": 4, "пят": 5, "шест": 6}


def _asked_times(question: str) -> list[float]:
    from .guard import _times_in

    return _times_in(question or "")


def _near(v: dict[str, Any], times: Sequence[float], pad: float = 2.0) -> bool:
    t0 = to_float(v.get("t_start"))
    t1 = to_float(v.get("t_end"))
    tc = to_float(v.get("timecode_sec"), 0.0) or 0.0
    lo = (t0 if t0 is not None else tc) - pad
    hi = (t1 if t1 is not None else tc) + pad
    return any(lo <= t <= hi for t in times)


def classify(question: str) -> str:
    q = (question or "").lower().replace("ё", "е")
    for intent, pattern in _INTENT_PATTERNS:
        if re.search(pattern.replace("ё", "е"), q):
            return intent
    return "general"


def target_violations(question: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    """Какие нарушения имеет в виду вопрос: по номеру («второе», «№2») или по типу (телефон -> V05)."""
    viols = result.get("violations") or []
    q = (question or "").lower()
    nums = {int(n) for n in re.findall(r"(?:№|номер|нарушени\w*)\s*(\d{1,2})", q)}
    for stem_, n in _ORDINALS.items():
        if re.search(stem_ + r"\w*\s+нарушени", q):
            nums.add(n)
    picked = [v for v in viols if v.get("_n") in nums]
    if picked:
        return picked
    types = {tid for tid, pat in _TYPE_HINTS.items() if re.search(pat, q)}
    picked = [v for v in viols if v.get("type_id") in types]
    return picked or list(viols)


def refusal_for(intent: str, result: dict[str, Any], corpus: Optional[Corpus]) -> Optional[str]:
    if intent == "identity":
        return (f"{REFUSAL_VIDEO}. Система не распознаёт и не идентифицирует людей по лицу. "
                "Роли (кассир, посетитель, инкассатор) определяются только по зоне, поведению и предметам.")
    if intent == "unobservable":
        items = "; ".join(unobservable_items(result, corpus))
        return (f"{REFUSAL_VIDEO}. Фиксируются только видимые действия; не устанавливаются: {items}. "
                "Это проверяется по документам и учётной системе.")
    if intent == "offtopic":
        return f"{REFUSAL_SCOPE}."
    return None


def _catalog_refs(v: dict[str, Any], refs: Sequence[NormRef]) -> list[NormRef]:
    return [r for r in refs if r.relation == "catalog" and v.get("_n") in r.events]


def _norm_phrase(r: NormRef) -> str:
    tags = []
    if r.chunk.status == "candidate" or not r.chunk.verified:
        tags.append("требует сверки")
    return f"{r.marker}" + (f" ({', '.join(tags)})" if tags else "")


def _viol_line(v: dict[str, Any], refs: Sequence[NormRef], corpus: Optional[Corpus],
               result: Optional[dict[str, Any]] = None) -> str:
    tid = v.get("type_id") or ""
    name = v.get("type") or (corpus.type_name(tid) if corpus and tid else "") or "возможное нарушение"
    own = _catalog_refs(v, refs)
    conf = v.get("confidence")
    conf_s = f", уверенность {float(conf):.2f}" if conf not in (None, "") else ""
    desc = str(v.get("description") or "").strip().rstrip(".")
    line = f"- [V{v.get('_n')}] {name}{conf_s}"
    if desc:
        line += f". Наблюдение системы: {desc}"
    if result is not None:  # «почему отмечено» — только то, что прислал пайплайн
        for ex in explain_lines(explain_for(result, v)):
            line += f". {ex}"
    if own:
        line += ". Норма: " + ", ".join(_norm_phrase(r) for r in own)
    elif corpus is not None and tid:
        line += ". Норма: без пункта (в каталоге нет подтверждённого пункта для этого типа)"
    return line + "."


def compose(intent: str, question: str, result: dict[str, Any], refs: Sequence[NormRef],
            corpus: Optional[Corpus]) -> str:
    """Собрать ответ с маркерами [Nk]/[Vk] (пройдёт ту же пост-проверку, что и ответ LLM)."""
    refusal = refusal_for(intent, result, corpus)
    if refusal:
        return refusal
    viols = result.get("violations") or []
    targets = target_violations(question, result)
    status = result.get("status")
    if status in {"processing", "failed"}:
        return f"{REFUSAL_NO_DATA}: {verdict_text(result)}. Нет данных для ответа."
    asked = _asked_times(question)
    if asked and intent not in {"norm", "fine"}:
        near = [v for v in viols if _near(v, asked)]
        if not near:
            when = ", ".join(fmt_tc(t) for t in asked)
            return (f"{REFUSAL_NO_DATA}: в результате анализа нет событий около {when}. "
                    "Анализ сообщает только о перечисленных в результате эпизодах; этот момент стоит просмотреть вручную.")
        targets = near

    if intent == "fine":
        hits = [r for r in refs if re.search(r"штраф|ответствен|санкц|наказан", r.chunk.text.lower())]
        if not hits:
            return f"{REFUSAL_SOURCES}. В найденных нормах не указаны штрафы или ответственность."
        return "В источниках: " + " ".join(f"«{r.chunk.text}» {r.marker}" for r in hits[:2])

    if intent == "norm":
        cat = [r for r in refs if r.relation == "catalog" and (not targets or any(t.get("_n") in r.events for t in targets))]
        found = [r for r in refs if r.relation != "catalog"]
        missing = [str(t.get("type") or (corpus.type_name(str(t.get("type_id"))) if corpus else "") or t.get("type_id"))
                   for t in targets if t.get("type_id") and not _catalog_refs(t, refs)]
        lines: list[str] = []
        if missing and not cat:
            lines.append(f"{REFUSAL_NO_DATA}: " + "; ".join(
                f"для «{m}» в каталоге нет подтверждённого пункта — без пункта" for m in missing) + ".")
        if cat:
            lines.append("Нормы каталога для этого типа:")
            for r in cat[:3]:
                lines.append(f"- {r.chunk.document}" + (f", {r.chunk.clause}" if r.chunk.clause else "")
                             + f": «{r.chunk.text}» {_norm_phrase(r)}")
            for m in missing:
                lines.append(f"Для «{m}» в каталоге нет подтверждённого пункта — без пункта.")
        if found and not cat:
            lines.append("Найдено поиском по текстам норм (связь с нарушением каталогом не подтверждена):")
            for r in found[:2]:
                lines.append(f"- {r.chunk.document}" + (f", {r.chunk.clause}" if r.chunk.clause else "")
                             + f": «{r.chunk.text}» {r.marker} (возможно относится, требует сверки)")
        if not lines:
            return f"{REFUSAL_SOURCES}."
        return "\n".join(lines)

    if intent == "coverage":
        c = completeness(result)
        text = f"Полнота анализа: {completeness_text(result)}."
        if c["state"] == "incomplete":
            text += " Отсутствие находок в непроанализированной части не означает, что нарушений нет."
        elif c["state"] == "unknown":
            text += " Пайплайн не сообщил, проанализировано ли видео целиком."
        return text

    if intent == "exclusions":
        if corpus is None:
            return f"{REFUSAL_SOURCES}."
        parts = []
        for t in targets or viols:
            tid = str(t.get("type_id") or "")
            name = t.get("type") or corpus.type_name(tid)
            excl = [str(x) for x in corpus.type_info(tid).get("exclusions") or [] if str(x).strip()]
            if excl:
                parts.append(f"«{name}» по каталогу не считается нарушением, если: " + "; ".join(excl) + ".")
            else:
                parts.append(f"Для «{name}» исключения в каталоге не описаны.")
        return "\n".join(parts) if parts else f"{REFUSAL_SOURCES}."

    if intent == "hints":
        hints = review_hints(result)
        if not hints:
            return ("Сведений о спорных местах (кандидатах ниже порога) в результате нет. "
                    "Это не означает, что сомнительных моментов не было.")
        parts = [f"- {h.get('type') or h.get('type_id') or 'кандидат'}, {fmt_tc(h.get('timecode_sec'))}: "
                 + (", ".join(str(x) for x in h.get("reasons_ru") or []) or "ниже порога") for h in hints]
        return ("Спорные места (в вердикт не вошли, решение за проверяющим):\n" + "\n".join(parts))

    comp = completeness(result)
    if not viols and comp["state"] == "incomplete":
        return (f"{REFUSAL_NO_DATA}: {completeness_text(result)}. Нарушений в проанализированной части не найдено, "
                "но отсутствие находок при неполном анализе не означает, что нарушений нет — нужен просмотр видео.")
    if not viols:
        base = ("Признаков нарушения по результату анализа не обнаружено."
                + (" Полнота анализа не подтверждена." if comp["state"] == "unknown" else ""))
        if intent in {"why", "when", "evidence"}:
            return base + " Нет данных о нарушениях, поэтому таймкодов и кадров-доказательств нет."
        if intent == "checklist":
            return base + " Рекомендуется выборочно просмотреть видео: автоматический анализ может пропустить нарушение."
        return base + " Это подсказка, а не заключение: окончательное решение принимает проверяющий."

    if intent == "when":
        parts = []
        for v in targets:
            span = f" (интервал {fmt_tc(v.get('t_start'))}–{fmt_tc(v.get('t_end'))})" if v.get("t_end") else ""
            parts.append(f"- [V{v.get('_n')}] {v.get('type') or v.get('type_id')}{span}.")
        return "Таймкоды возможных нарушений:\n" + "\n".join(parts)

    if intent == "evidence":
        what = bool(re.search(r"что\s+(там\s+)?видн|что\s+на\s+кадре|что\s+происходит|что\s+изображ", question.lower()))
        parts = []
        for v in targets:
            frames = [fmt_tc(f.get("timestamp_sec")) for f in v.get("evidence_frames") or []]
            line = f"- [V{v.get('_n')}]: " + (f"кадры {', '.join(frames)}." if frames else "кадров-доказательств нет.")
            if what:  # «что видно на кадре» — описание и факты из результата, без домысливания картинки
                desc = str(v.get("description") or "").strip().rstrip(".")
                if desc:
                    line += f" Что видно (по описанию системы): {desc}."
                for ex in explain_lines(explain_for(result, v)):
                    if ex.startswith("Наблюдаемые факты"):
                        line += f" {ex}."
            parts.append(line)
        return "Кадры-доказательства:\n" + "\n".join(parts)

    if intent == "confidence":
        parts = [f"- [V{v.get('_n')}]: уверенность {float(v.get('confidence') or 0):.2f}." for v in targets]
        return ("Уверенность модели по нарушениям:\n" + "\n".join(parts) +
                "\nЭто оценка алгоритма, а не вероятность нарушения: требуется просмотр фрагмента человеком.")

    if intent == "checklist":
        parts = []
        for v in targets:
            span = (f"{fmt_tc(v.get('t_start'))}–{fmt_tc(v.get('t_end'))}" if v.get("t_end")
                    else fmt_tc(v.get("timecode_sec")))
            parts.append(f"- Просмотреть фрагмент {span} с запасом до и после [V{v.get('_n')}] и сверить с описанием.")
        parts.append("- Сверить с документами то, что по видео не устанавливается: "
                     + "; ".join(unobservable_items(result, corpus)) + ".")
        return "Что проверить вручную:\n" + "\n".join(parts)

    # why / summary / general
    head = (f"Вердикт: {verdict_text(result)}; возможных нарушений: {len(viols)}."
            if intent in {"summary", "general"} else "Основания (только наблюдаемые факты):")
    lines = [head] + [_viol_line(v, refs, corpus, result if intent == "why" else None) for v in targets]
    if corpus is None and not any(_catalog_refs(t, refs) for t in targets):
        lines.append("Норма для этого случая в источниках не найдена.")
    lines.append("Это признаки возможного нарушения, решение принимает проверяющий.")
    return "\n".join(lines)
