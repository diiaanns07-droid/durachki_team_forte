"""Фильтр «невидимых» фактов (правило 4): сумма, проводка, личность, подлинность, умысел, домыслы.

Факт, содержащий такой признак, отбрасывается целиком (не редактируется) — с причиной, для трейса.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# (категория, регэксп). Регэкспы — по основам слов, регистронезависимо, RU + EN.
_RULES: list[tuple[str, str]] = [
    # суммы / деньги в количественном выражении
    ("amount", r"\bсумм\w*|\b\d[\d\s.,]*\s*(тг|тенге|₸|kzt|руб\w*|₽|\$|usd|долл\w*|евро|eur|€)"),
    ("amount", r"\b(amount|sum of|total of|\d+\s*(tenge|dollars|rubles|euros))\b"),
    ("amount", r"\b(тысяч\w*|миллион\w*)\s+(тенге|руб\w*|долл\w*)"),
    ("amount", r"\bкупюр\w*\s+(номинал\w*|достоинств\w*)|\bноминал\w*|\bdenomination"),
    # проводка / учёт / документы, которые камера не подтверждает
    ("accounting", r"\bпровод(к|ит|ят|ил)\w*|\bтранзакц\w*|\bучёт\w*|\bучет\w*|\bбаланс\w*|\bнедостач\w*|\bизлиш\w*"),
    ("accounting", r"\b(transaction|ledger|posting|accounting|shortage|surplus|balance)\w*"),
    # личность / идентификация (распознавание лиц запрещено)
    ("identity", r"\bзовут\b|\bпо имени\b|\bимя\b|\bфамили\w*|\bличност\w*|\bопознан\w*|\bидентифицир\w*|\bузна(л|ть|ется)\w*\s+(как|в)\b"),
    ("identity", r"\b(named|his name|her name|identity|identified as|recogni[sz]ed as)\b"),
    ("identity", r"\b(кассир|клиент|инкассатор|сотрудник)\w*\s+[А-ЯЁ][а-яё]+(ов|ев|ин|ина|ова|ева|ская|ский)\b"),
    ("identity", r"\b(возраст|национальн|этни)\w*|\b\d{2}\s*(лет|года)\b|\b(age|ethnic)\w*"),
    # подлинность / фальшивки
    ("authenticity", r"\bподлинн\w*|\bфальшив\w*|\bподдел\w*|\bнастоящ\w*\s+(купюр|деньг|банкнот)|\bсчита(ет|ют)\s+фальш"),
    ("authenticity", r"\b(counterfeit|fake (notes|money|bills)|forged|genuine|authentic)\w*"),
    # умысел / мотивы / квалификация
    ("intent", r"\bумысел\w*|\bумышлен\w*|\bнамерен\w*|\bнарочно\b|\bспециально\b|\bс целью\b|\bчтобы\s+(скрыть|украсть|спрятать|вынести|обмануть)"),
    ("intent", r"\bпыта(ет|ют|лся|лась)ся\s+(скрыть|украсть|спрятать|вынести|обмануть)|\bхочет\b|\bсобира(ет|ют)ся\b"),
    ("intent", r"\bкраж\w*|\bворует\b|\bукрал\w*|\bпохити\w*|\bхищени\w*|\bмошенни\w*|\bсговор\w*|\bпреступ\w*|\bвинов\w*"),
    ("intent", r"\b(intent\w*|on purpose|deliberately|in order to|trying to (steal|hide)|theft|steal\w*|fraud\w*|collusion|guilty|crime)\b"),
    # нормы / пункты / полномочия людей — VLM их не знает и не придумывает (нормы — только norms/catalog.yaml)
    ("norm_or_authority", r"\bпункт\w*|\bп\.\s*\d|\bстать(я|и|ей|ю)\s+\d|\bзакон\w*|\bпостановлени\w*|\bнорм(а|ы|ой|е|у|ам|ами|ах|атив\w*)\b|\bправил(а|ам|ами|ах)?\b|\bинструкци\w*|\bрегламент\w*|\bнбрк\b"),
    ("norm_or_authority", r"\bуполномоч\w*|\bполномочи\w*|\bдопуск\w*|\bимеет\s+прав\w*|\bимеют\s+прав\w*|\bразрешени\w*|\bзапрещ\w*|\bне\s+положено\b"),
    ("norm_or_authority", r"\b(regulation|clause|article \d+|law|authori[sz]ed|unauthori[sz]ed|permission|not allowed|prohibited)\b"),
    # квалификация вместо наблюдения
    ("conclusion", r"\bнаруша\w*|\bнарушени\w*|\bviolat\w*"),
    # эмоции / внутреннее состояние — не наблюдаемо
    ("mental_state", r"\bнервнича\w*|\bволну(ет|ют)ся\b|\bбоит\w*|\bзл(ой|ится)\b|\bнедовол\w*|\bспешит\b"),
    ("mental_state", r"\b(nervous|anxious|angry|afraid|wants to|intends)\b"),
    # содержимое закрытых объектов / экранов / разговоров
    ("hidden_content", r"\bв сумке (лежат|находятся|были)|\bв сейфе (лежат|находятся|хранятся)|\bна экране (видно|написано)|\bразговарива\w*\s+с\s+\w+"),
    ("hidden_content", r"\b(inside the (bag|safe) (are|is|were)|on the screen (it says|shows)|talking to (his|her|a) )"),
]

_HEDGES = r"\bвероятно\b|\bскорее всего\b|\bпо-видимому\b|\bвидимо\b|\bнаверное\b|\bпредположительно\b|\bкажется\b|\b(probably|likely|presumably|apparently|seems to)\b"

# «Якоря» наблюдаемости: факт должен называть видимую роль/объект/часть тела (иначе это не наблюдение).
_ANCHORS = (r"кассир|человек|люд(и|ей|ьми)|посетител|клиент|инкассатор|сотрудник|рук(а|и|у|ой|ами|ах)?\b|ладон|пальц|"
            r"телефон|смартфон|мобильн|наличн|купюр|банкнот|деньг|денег|пачк|сейф|дверц|двер|сумк|мешок|пакет|"
            r"стол|стул|кресл|окн|стекл|перегородк|касс|ноутбук|компьютер|монитор|клавиатур|сч[её]тчик|машинк|"
            r"чашк|кружк|стакан|бутылк|напит|ед(а|у|ы|ой)\b|пищ|бутерброд|документ|бумаг|ключ|рабоч|зон|помещени|проход|"
            r"голов|плеч|спин|лиц(о|а|ом)\b|"
            r"cashier|person|people|visitor|client|collector|hand|phone|cash|banknote|money|safe|door|bag|desk|"
            r"table|chair|window|laptop|counter|cup|bottle|food|drink|document|key")

_COMPILED = [(cat, re.compile(rx, re.I | re.U)) for cat, rx in _RULES]
_ANCHOR_RE = re.compile(_ANCHORS, re.I | re.U)
_HEDGE_RE = re.compile(_HEDGES, re.I | re.U)
_LATIN_ONLY_RE = re.compile(r"^[^А-Яа-яЁё]*$")


@dataclass
class FilterResult:
    kept: list[str] = field(default_factory=list)
    dropped: list[tuple[str, str]] = field(default_factory=list)  # (факт, категория)


def check_fact(fact: str, drop_hedges: bool = True, require_anchor: bool = True) -> str:
    """'' — факт допустим; иначе категория нарушения. Это грубая страховка, НЕ гарантия от галлюцинаций."""
    for cat, rx in _COMPILED:
        if rx.search(fact):
            return cat
    if drop_hedges and _HEDGE_RE.search(fact):
        return "hedge"
    if require_anchor and not _ANCHOR_RE.search(fact):
        return "no_anchor"
    return ""


def _clean(fact: str, max_len: int) -> str:
    f = re.sub(r"\s+", " ", str(fact)).strip().strip("\"'«»").strip()
    f = re.sub(r"^(?:[-•*]\s*|\d{1,2}[.)]\s+)", "", f)
    if len(f) > max_len:
        cut = f[:max_len]
        f = cut[: cut.rfind(" ")] if " " in cut else cut
        f = f.rstrip(",;:") + "…"
    if f and f[0].islower():
        f = f[0].upper() + f[1:]
    return f


def filter_facts(facts: list[str], max_len: int = 220, max_items: int = 4, drop_hedges: bool = True,
                 require_cyrillic: bool = True, require_anchor: bool = True) -> FilterResult:
    """Чистит, дедуплицирует и отбрасывает недопустимые факты.

    require_cyrillic: факты для жюри — по-русски; английский ответ модели отбрасываем
    (вызывающий код подставит шаблонный факт из каталога).
    """
    res = FilterResult()
    seen: set[str] = set()
    for raw in facts or []:
        f = _clean(raw, max_len)
        if len(f) < 4:
            continue
        key = f.lower()
        if key in seen:
            continue
        seen.add(key)
        cat = check_fact(f, drop_hedges=drop_hedges, require_anchor=require_anchor)
        if not cat and require_cyrillic and _LATIN_ONLY_RE.match(f):
            cat = "not_russian"
        if cat:
            res.dropped.append((f, cat))
            continue
        if len(res.kept) < max_items:
            res.kept.append(f)
    return res
