"""Пост-проверка ответа: пользователю уходит только то, что подтверждено контекстом.

Проверяем (по предложениям):
* маркеры [Nk] — только из выданного контекста; превращаются в сноски [1], [2] по порядку citations;
* упоминания «п. 15», «пункт 15», «статья 5» — пункт должен быть среди норм, переданных в контексте;
* названия актов («Правила «…»», «Закон «…»») — документ должен быть в корпусе;
* URL — только известные корпусу;
* таймкоды — только из результата анализа или из вопроса/истории;
* денежные суммы — только встречающиеся в результате/вопросе (по видео сумма не устанавливается);
* опора на контекст — предложение без лексической опоры на результат/нормы считается домыслом.
Предложения, не прошедшие проверку, вырезаются; если не осталось ничего — честный отказ.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .corpus import Corpus, norm_clause, norm_text, norm_url, _CLAUSE_RE
from .llm import strip_think
from .prompts import NormRef, REFUSAL_NO_DATA, REFUSAL_SCOPE, REFUSAL_SOURCES, REFUSAL_VIDEO, fmt_tc
from .safety import strip_output_artifacts

_MARKER_RE = re.compile(r"[\[(]\s*([NnНн])\s*(\d{1,2})\s*[\])]")
_VMARK_RE = re.compile(r"\[\s*[Vv]\s*(\d{1,2})\s*\]")
_URL_RE = re.compile(r"https?://[^\s)\]>»\"']+")
_ACT_RE = re.compile(
    r"(?P<kind>Правил\w*|Закон\w*|Постановлени\w*|Приказ\w*|Инструкци\w*|Кодекс\w*|Положени\w*|Регламент\w*)"
    r"\s*(?:РК|Республики Казахстан|НБРК|Национального Банка[^«\"]{0,40})?\s*[«\"](?P<title>[^»\"]{5,200})[»\"]",
)
_TIME_RE = re.compile(
    r"(?<![\d:.])(?:(?P<h>\d{1,2}):)?(?P<m>\d{1,2}):(?P<s>\d{2})(?![\d:])"
    r"|(?<![\d.,])(?P<sec>\d{1,4}(?:[.,]\d)?)\s*(?:-?(?:й|я|ой|ая|ую))?\s*(?:с\b|сек\b|сек\.|секунд\w*)",
    re.I,
)
_AMOUNT_RE = re.compile(
    r"(?:[$€₸₽]\s*\d[\d\s\u00a0]*(?:[.,]\d+)?)"
    r"|(?:(?<![\w.])\d[\d\s\u00a0]*(?:[.,]\d+)?\s*(?:тыс\.?|млн\.?|млрд\.?)?\s*"
    r"(?:тенге|тг\b|₸|руб\w*|₽|доллар\w*|\$|usd\b|евро|€|eur\b))",
    re.I,
)
_BOUNDARY_RE = re.compile(r"([.!?…]+)(\s+)")
_ITEM_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
# сокращения, после точки которых предложение НЕ заканчивается («п. 9», «ст. 5», «22 с. кассир»)
_ABBREV = {"п", "пп", "ст", "ч", "гл", "разд", "прил", "т.е", "т.д", "т.п", "т.к", "др", "г", "гг", "с",
           "сек", "мин", "им", "рис", "см", "напр", "тыс", "млн", "млрд", "руб", "тг", "no", "№", "стр", "абз"}
_WORD_RE = re.compile(r"[a-zа-яё]{4,}", re.I)
_STOP = set("""это этот эта эти этого этой того тому также если или либо чтобы тоже только уже ещё еще
его её ее их она они оно был была были быть есть может могут можно нужно должен должна должны который
которая которые которое чем где когда тогда здесь там весь вся всего всему всех очень более менее через
после перед между около какой какая какие такой такая такие свой своя свои себя этом этих было будет будут""".split())
# служебная лексика ответов ассистента (не факты) — не требует опоры на контекст
_ANSWER_VOCAB = {w[:4] for w in """признак возможного нарушение проверяющий решение принимает подсказка заключение
видео основание норма источник пункт данные установить нельзя факты наблюдаемые кадры таймкод уверенность
анализ результат система фрагмент просмотреть проверить вручную документ сноска интервал секунда минута
согласно требование запрещено допускается указано видно отмечено зафиксировано обнаружено рекомендуется
сверить подтвердить отклонить алгоритм оценка вероятность человек определяется является случае ответ
информация относящиеся найденные данных имеющимся момент означает пайплайн сообщил проанализировано целиком
полнота неполный неизвестно каталог исключения описаны поиском пункта сверки роли кассир посетитель
инкассатор зоне поведению предметам""".split()}

# утверждения «нарушений нет» / «нарушение есть» — сверяются с вердиктом и полнотой анализа
_CLAIM_NONE_RE = re.compile(
    r"нарушени\w*\s+(?:нет|не\s+(?:было|обнаружен\w*|найден\w*|выявлен\w*)|отсутству\w*)"
    r"|(?:не\s+(?:обнаружен\w*|найден\w*|выявлен\w*)|нет|отсутству\w*)\s+(?:никаких\s+)?нарушени\w*"
    r"|вс[её]\s+в\s+порядке|без\s+нарушений|касса\s+чист\w*",
    re.I,
)
_CLAIM_SOME_RE = re.compile(r"(?:обнаружен\w*|выявлен\w*|зафиксирован\w*|есть)\s+(?:признак\w*\s+)?(?:возможного\s+)?нарушени", re.I)
_CLAIM_STRONG_RE = re.compile(
    r"вс[её]\s+в\s+порядке|касса\s+чист\w*|без\s+нарушений|нарушени\w*\s+нет\b|нет\s+(?:никаких\s+)?нарушени", re.I)
_BAD = "\u2063BAD\u2063"  # внутренняя метка «предложение ссылается на несуществующее»
_CAVEAT_RE = re.compile(r"не\s+означает|неполн|не\s+подтвержд|полнота|в\s+проанализированной\s+части", re.I)


REFUSAL_PATTERNS = [
    r"установить нельзя", r"нельзя установить", r"не\s+устанавлива", r"в источниках нет", r"нет информации",
    r"нет данных", r"не могу ответить", r"не выполняется", r"не определяется", r"невозможно определить",
    r"не\s+распозна", r"не\s+идентифицир",
    re.escape(REFUSAL_SCOPE.lower()),
]
UNVERIFIED_ANSWER = f"{REFUSAL_NO_DATA}: ответ модели не подтверждён результатом анализа и источниками."


def is_refusal(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in REFUSAL_PATTERNS)


@dataclass
class GuardResult:
    answer: str
    citations: list[dict[str, Any]]
    grounded: bool
    refusal: bool
    removed: list[str] = field(default_factory=list)
    raw_invalid: bool = False
    warnings: list[str] = field(default_factory=list)


def _times_in(text: str) -> list[float]:
    out: list[float] = []
    for m in _TIME_RE.finditer(text or ""):
        if m.group("sec"):
            try:
                out.append(float(m.group("sec").replace(",", ".")))
            except ValueError:
                pass
        elif m.group("m") is not None:
            h = int(m.group("h") or 0)
            out.append(h * 3600 + int(m.group("m")) * 60 + int(m.group("s")))
    return out


def _is_boundary(line: str, start: int, m: re.Match[str]) -> bool:
    """Конец предложения? Не режем после сокращений («п. 9») и перед строчной буквой/цифрой."""
    if m.group(1) != ".":
        return True
    before = line[start:m.start(1)]
    lw = re.search(r"([\w.№]+)$", before)
    last = lw.group(1).lower() if lw else ""
    nxt = line[m.end():m.end() + 1]
    if last in _ABBREV or (len(last) == 1 and last.isalpha()):
        return False
    if nxt and (nxt.isdigit() or nxt.islower()):
        return False
    return True


def _split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if _ITEM_RE.match(line):
            parts.append(line)
            continue
        start = 0
        for m in _BOUNDARY_RE.finditer(line):
            if _is_boundary(line, start, m):
                parts.append(line[start:m.end(1)].strip())
                start = m.end()
        tail = line[start:].strip()
        if tail:
            parts.append(tail)
    return parts


def complete_prefix(text: str) -> str:
    """Часть текста до последнего завершённого предложения (для потоковой выдачи проверенного текста)."""
    text = text or ""
    cut = text.rfind("\n") + 1
    line_start = cut
    line = text[line_start:]
    start = 0
    for m in _BOUNDARY_RE.finditer(line):
        if _is_boundary(line, start, m):
            cut = line_start + m.end(1)
            start = m.end()
    return text[:cut]


def _stems(text: str) -> set[str]:
    return {w.lower().replace("ё", "е")[:4] for w in _WORD_RE.findall(text or "") if w.lower() not in _STOP}


def _doc_known(title: str, corpus: Optional[Corpus], refs: Sequence[NormRef]) -> bool:
    t = norm_text(title)
    docs = {norm_text(r.chunk.document) for r in refs}
    if corpus is not None:
        docs |= corpus.documents()
    if not t:
        return True
    t_words = set(re.findall(r"\w{4,}", t))
    for d in docs:
        if t in d or d in t:
            return True
        d_words = set(re.findall(r"\w{4,}", d))
        if t_words and len(t_words & d_words) / len(t_words) >= 0.6:
            return True
    return False


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def check_answer(raw: str, refs: Sequence[NormRef], corpus: Optional[Corpus], result: dict[str, Any],
                 question: str = "", history: Sequence[dict[str, str]] = (),
                 tolerance_s: float = 1.5, min_support: float = 0.3) -> GuardResult:
    text = strip_output_artifacts(strip_think(raw))
    removed: list[str] = []
    by_n = {r.n: r for r in refs}
    viols = {int(v.get("_n", i + 1)): v for i, v in enumerate(result.get("violations") or [])}

    # --- 1. маркеры норм [Nk] -> сноски по порядку появления
    order: list[int] = []

    def _marker(m: re.Match[str]) -> str:
        n = int(m.group(2))
        if n not in by_n:
            removed.append(m.group(0))
            return _BAD  # утверждение со ссылкой на несуществующую норму уходит вместе с предложением
        if n not in order:
            order.append(n)
        return f"[{order.index(n) + 1}]"

    text = _MARKER_RE.sub(_marker, text)

    # --- 2. маркеры нарушений [Vk] -> «(№k, мм:сс)»
    def _vmark(m: re.Match[str]) -> str:
        n = int(m.group(1))
        v = viols.get(n)
        if v is None:
            removed.append(m.group(0))
            return _BAD  # утверждение о несуществующем событии уходит вместе с предложением
        return f"(№{n}, {fmt_tc(v.get('timecode_sec'))})"

    text = _VMARK_RE.sub(_vmark, text)

    # --- 3. контекст, на который ответ имеет право опираться: ТОЛЬКО текущая задача, каталог и текущий вопрос.
    # История диалога — не источник фактов (может быть устаревшей, от другой задачи или поддельной).
    user_text = question
    allowed_times = [float(t) for t in _result_times(result)] + _times_in(user_text)
    ref_clauses = {norm_clause(r.chunk.clause) for r in refs if r.chunk.clause}
    if corpus is not None:  # пункты, упомянутые в пояснениях/исключениях каталога для типов событий (тоже контекст)
        for v in viols.values():
            info = corpus.type_info(str(v.get("type_id") or ""))
            for txt in [str(info.get("legal_note") or "")] + [str(x) for x in info.get("exclusions") or []]:
                ref_clauses |= {norm_clause(cm.group(0)) for cm in _CLAUSE_RE.finditer(txt)}
    known_urls = {norm_url(r.chunk.url) for r in refs if r.chunk.url}
    if corpus is not None:
        known_urls |= corpus.known_urls()
    result_text = json.dumps(result, ensure_ascii=False, default=str)
    context_text = " ".join([result_text, user_text] + [f"{r.chunk.document} {r.chunk.heading} {r.chunk.text}" for r in refs])
    if corpus is not None:
        context_text += " " + " ".join(corpus.type_name(str(v.get("type_id") or "")) for v in viols.values())
    # суммы, явно упомянутые в контексте (результат/вопрос/нормы); совпадение с таймкодом не считается
    context_amounts = {_digits(m.group(0)) for m in _AMOUNT_RE.finditer(context_text)}
    context_stems = _stems(context_text) | _ANSWER_VOCAB

    kept: list[str] = []
    for sent in _split_sentences(text):
        bad = _BAD in sent
        url_cut = False
        for um in _URL_RE.finditer(sent):
            if norm_url(um.group(0)) not in known_urls:
                removed.append(um.group(0))
                sent = sent.replace(um.group(0), "").strip()
                url_cut = True
        if url_cut and len(re.findall(r"[а-яёa-z]{2,}", sent, re.I)) < 3:
            bad = True  # после удаления ссылки осталась «Подробнее:» и т.п.
        for am in _ACT_RE.finditer(sent):
            if not _doc_known(am.group("title"), corpus, refs):
                removed.append(am.group(0))
                bad = True
        for cm in _CLAUSE_RE.finditer(sent):
            # только пункты из переданных модели норм: модель не видела остальной корпус,
            # а «п. 12» почти наверняка найдётся в каком-нибудь документе каталога
            if norm_clause(cm.group(0)) in ref_clauses:
                continue
            removed.append(cm.group(0))
            bad = True
        for tsec in _times_in(sent):
            if not any(abs(tsec - a) <= tolerance_s for a in allowed_times):
                removed.append(f"{tsec:g} с")
                bad = True
        for mm in _AMOUNT_RE.finditer(sent):
            d = _digits(mm.group(0))
            if d and d not in context_amounts:
                removed.append(f"сумма: {mm.group(0).strip()}")
                bad = True
        if not bad and _contradicts(sent, result):
            removed.append(f"противоречит результату: {sent[:60]}")
            bad = True
        if not bad and _invented_violation(sent, result):
            removed.append(f"нарушение не из результата: {sent[:60]}")
            bad = True
        if not bad and _claims_verified_norm(sent, order, by_n):
            removed.append(f"непроверенная норма выдана за проверенную: {sent[:60]}")
            bad = True
        if not bad and not is_refusal(sent):
            stems = _stems(re.sub(r"\(№\d+, [\d:]+\)|\[\d+\]", " ", sent))
            if len(stems) >= 4 and len(stems & context_stems) / len(stems) < min_support:
                removed.append(f"без опоры на контекст: {sent[:60]}")
                bad = True
        if not bad and sent.strip():
            kept.append(sent.strip())

    answer = ""
    prev_item = False
    for part in kept:
        is_item = bool(_ITEM_RE.match(part))
        sep = "\n" if (is_item or prev_item or answer.endswith(":")) else " "
        answer = f"{answer}{sep}{part}" if answer else part
        prev_item = is_item
    answer = answer.strip()
    answer = re.sub(r"[ \t]+([.,;:!?])", r"\1", answer)
    answer = re.sub(r"[ \t]{2,}", " ", answer)

    citations: list[dict[str, Any]] = []
    for n in order:
        r = by_n[n]
        if f"[{order.index(n) + 1}]" not in answer:
            continue  # сноска ушла вместе с вырезанным предложением
        cite = r.chunk.citation()
        cite.update({"ref": order.index(n) + 1, "quote": r.chunk.text, "verified": r.chunk.verified,
                     "relation": r.relation, "status": r.chunk.status, "note": r.status_ru,
                     "events": list(r.events)})
        if corpus is not None:
            short = corpus.short_name(r.chunk.document, r.chunk.url)
            if short != r.chunk.document:
                cite["short"] = short
        citations.append(cite)

    raw_invalid = bool(removed)
    if not answer and raw_invalid:
        answer = UNVERIFIED_ANSWER
    refusal = is_refusal(answer)
    refers_result = bool(re.search(r"№\d+, \d{2}:\d{2}", answer)) or bool(_times_in(answer))
    supported = bool(citations) or refusal or refers_result or bool(
        re.search(r"признак\w* (возможного )?нарушени|не обнаружен", answer.lower()))
    grounded = bool(answer) and not raw_invalid and supported
    warnings = [f"удалены неподтверждённые фрагменты: {', '.join(dict.fromkeys(removed))}"] if removed else []
    return GuardResult(answer=answer, citations=citations, grounded=grounded, refusal=refusal,
                       removed=list(dict.fromkeys(removed)), raw_invalid=raw_invalid, warnings=warnings)


# сильные признаки типа нарушения в тексте ответа (для проверки «нарушение не из результата»)
_TYPE_WORDS = {
    "V01": r"посторонн", "V02": r"покинул|отош[её]л\s+от\s+рабоч", "V03": r"сейф", "V04": r"двер",
    "V05": r"телефон|смартфон", "V06": r"\bеда\b|\bпищ|напит|\bкофе\b|\bчай\b", "V07": r"инкасс",
    "V08": r"пересч[её]т|пересчит", "V09": r"личн\w*\s+вещ|куртк|рюкзак",
}
_NOT_A_DETECTION_RE = re.compile(r"кандидат|ниже\s+порога|спорн|не\s+вошл|не\s+(?:обнаружен|выявлен|зафиксирован)", re.I)
_VERIFIED_CLAIM_RE = re.compile(
    r"(?:подтвержд[её]нн\w*|сверенн\w*|проверенн\w*|действующ\w*)\s+(?:норм|пункт)\w*"
    r"|(?:норм|пункт)\w*\s+(?:\w+\s+){0,2}(?:подтвержден\w*|сверен\w*|проверен\w*)", re.I)


def _invented_violation(sent: str, result: dict[str, Any]) -> bool:
    """Утверждение «обнаружено нарушение X», где типа X нет ни в нарушениях, ни в спорных местах."""
    from .prompts import review_hints

    if not _CLAIM_SOME_RE.search(sent) or _NOT_A_DETECTION_RE.search(sent):
        return False
    present = {str(v.get("type_id") or "") for v in result.get("violations") or []}
    present |= {str(h.get("type_id") or "") for h in review_hints(result)}
    low = sent.lower().replace("ё", "е")
    mentioned = {tid for tid, pat in _TYPE_WORDS.items() if re.search(pat.replace("ё", "е"), low)}
    return bool(mentioned - present)


def _claims_verified_norm(sent: str, order: list[int], by_n: dict[int, NormRef]) -> bool:
    """«Проверенная/подтверждённая норма», хотя процитированная норма не сверена (или нормы нет)."""
    if not _VERIFIED_CLAIM_RE.search(sent) or re.search(
            r"(?:\bне|\bнет|\bбез|отсутству\w*|требует)\s+(?:\w+\s+)?(?:подтвержд|сверен|провер|сверки)", sent, re.I):
        return False
    refs = [by_n[order[int(k) - 1]] for k in re.findall(r"\[(\d+)\]", sent) if 0 < int(k) <= len(order)]
    return not any(r.chunk.verified and r.chunk.status in {"", "confirmed"} for r in refs)


def _contradicts(sent: str, result: dict[str, Any]) -> bool:
    """«Нарушений нет» при вердикте violation или при неполном/упавшем анализе; «нарушение есть» при чистом."""
    from .prompts import completeness

    if _CAVEAT_RE.search(sent) or is_refusal(sent):
        return False
    has_viols = bool(result.get("violations")) or result.get("verdict") == "violation"
    state = completeness(result)["state"]
    if _CLAIM_NONE_RE.search(sent):
        if _times_in(sent):
            return False  # локальное утверждение о моменте — его проверяет логика таймкодов
        scoped = re.search(r"\bдруг\w+|\bин(?:ых|ые|ой)\b|\bкроме\b|\bещ[её]\b", sent, re.I)
        if state in {"incomplete", "pending"} or result.get("status") == "failed":
            return True
        if has_viols and not scoped:
            return True
        if state == "unknown" and _CLAIM_STRONG_RE.search(sent):
            return True  # полнота не подтверждена — «всё в порядке» без оговорки не говорим
    elif _CLAIM_SOME_RE.search(sent) and not has_viols and result.get("status") == "done":
        return not re.search(r"\bне\s+(?:обнаружен|выявлен|зафиксирован)", sent, re.I)
    return False


def _result_times(result: dict[str, Any]) -> list[float]:
    from .prompts import result_times

    return result_times(result)


__all__ = ["GuardResult", "UNVERIFIED_ANSWER", "check_answer", "complete_prefix", "is_refusal",
           "REFUSAL_NO_DATA", "REFUSAL_SOURCES", "REFUSAL_VIDEO"]
