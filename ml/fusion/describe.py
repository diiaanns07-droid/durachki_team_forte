"""Описания нарушений: что видно → почему это может быть нарушением → «требуется проверка человеком».

Только наблюдаемые факты (правило 4): предложения и факты со словами-домыслами
(«украл», «мошенничество», «умышленно», …) вычищаются.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Iterable, Optional

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_WS = re.compile(r"\s+")


@lru_cache(maxsize=8)
def _compile(patterns: tuple[str, ...]) -> Optional[re.Pattern[str]]:
    valid: list[str] = []
    for p in patterns:
        try:
            re.compile(p)
        except re.error:
            continue
        valid.append(f"(?:{p})")
    if not valid:
        return None
    return re.compile(r"\b(?:" + "|".join(valid) + ")", re.IGNORECASE)


def forbidden_regex(cfg: dict[str, Any]) -> Optional[re.Pattern[str]]:
    patterns = (cfg.get("description") or {}).get("forbidden") or []
    return _compile(tuple(str(p) for p in patterns))


def has_forbidden(text: str, cfg: dict[str, Any]) -> bool:
    rx = forbidden_regex(cfg)
    return bool(rx and rx.search(text or ""))


def format_time(t: float) -> str:
    """22.4 → «00:22,4»; 65.0 → «01:05»."""
    t = max(0.0, float(t))
    minutes = int(t // 60)
    sec = t - minutes * 60
    if abs(sec - round(sec)) < 0.05:
        return f"{minutes:02d}:{int(round(sec)):02d}"
    return f"{minutes:02d}:{sec:04.1f}".replace(".", ",")


def _norm_fact(fact: str) -> str:
    return _WS.sub(" ", str(fact)).strip().strip(" .;,").strip()


def clean_facts(facts: Iterable[Any], cfg: dict[str, Any], max_facts: Optional[int] = None) -> list[str]:
    """Убирает пустые, дубли и факты с домыслами; сохраняет порядок."""
    rx = forbidden_regex(cfg)
    out: list[str] = []
    seen: set[str] = set()
    for raw in facts or []:
        fact = _norm_fact(raw)
        if not fact:
            continue
        key = fact.lower()
        if key in seen:
            continue
        if rx and rx.search(fact):
            continue
        seen.add(key)
        out.append(fact)
    limit = max_facts if max_facts is not None else int((cfg.get("description") or {}).get("max_facts", 3))
    return out[: max(0, limit)]


class _SafeDict(dict):  # type: ignore[type-arg]
    def __missing__(self, key: str) -> str:
        return ""


def _render_template(template: str, values: dict[str, str]) -> str:
    try:
        return template.format_map(_SafeDict(values))
    except (ValueError, IndexError, AttributeError, KeyError):
        return ""


def _sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(_WS.sub(" ", text or "").strip())]
    return [p for p in parts if p]


def _ensure_period(sentence: str) -> str:
    sentence = sentence.strip()
    if sentence and sentence[-1] not in ".!?…":
        sentence += "."
    return sentence


def _capitalize(sentence: str) -> str:
    return sentence[:1].upper() + sentence[1:] if sentence else sentence


def build_description(
    type_id: str,
    info: dict[str, Any],
    facts: list[str],
    t_peak: float,
    cfg: dict[str, Any],
    confidence: Optional[float] = None,
    norm_ref: Optional[str] = None,
) -> str:
    """1–3 предложения: что видно → почему это может быть нарушением → проверка человеком.

    ``norm_ref`` — ТОЛЬКО сверенная норма из каталога (verified=true); дописывается к «почему».
    """
    dcfg = cfg.get("description") or {}
    check = _ensure_period(str(dcfg.get("check_sentence") or "Требуется проверка человеком."))
    max_chars = int(dcfg.get("max_chars", 420))
    rx = forbidden_regex(cfg)
    name = str(info.get("name") or type_id)
    t_str = format_time(t_peak)
    facts = clean_facts(facts, cfg)
    facts_str = "; ".join(f[:1].lower() + f[1:] for f in facts) if facts else ""

    seen_sentence = (
        f"На {t_str} видно: {facts_str}" if facts_str
        else f"На {t_str} зафиксированы визуальные признаки ситуации «{name}»"
    )

    sentences: list[str] = []
    template = str(info.get("description_template") or "").strip()
    if template:
        rendered = _render_template(
            template,
            {
                "t": t_str,
                "t_sec": f"{t_peak:.1f}",
                "facts": facts_str or f"признаки ситуации «{name}»",
                "name": name,
                "type": name,
                "type_id": type_id,
                "conf": f"{confidence:.2f}" if confidence is not None else "",
            },
        )
        sentences = _sentences(rendered)
        if facts_str and "{facts}" not in template:
            sentences.insert(0, seen_sentence)
    if not sentences:
        why = str(info.get("why") or "").strip()
        sentences = [seen_sentence]
        if why:
            sentences.append(why)

    # Чистка домыслов на уровне предложений.
    sentences = [_capitalize(_ensure_period(s)) for s in sentences if s and not (rx and rx.search(s))]
    sentences = [s for s in sentences if s.lower() != check.lower()]
    if not sentences:
        sentences = [_ensure_period(f"На {t_str} зафиксированы визуальные признаки ситуации «{name}»")]

    body = sentences[:2]
    if not any("возможн" in s.lower() for s in body):
        if len(body) == 1:
            body.append(f"Это может быть признаком возможного нарушения: «{name}».")
        else:  # не вытесняем второе предложение шаблона (например, честное «норма не подтверждена»)
            check = _ensure_period(str(dcfg.get("check_sentence_possible")
                                       or "Это признаки возможного нарушения — требуется проверка человеком."))
    if norm_ref and not (rx and rx.search(norm_ref)):
        last = body[-1].rstrip(".!?… ")
        body[-1] = f"{last} (норма: {norm_ref.strip().rstrip('.')})."
    text = " ".join(body + [check])
    if len(text) > max_chars:
        head = " ".join(body)
        room = max(0, max_chars - len(check) - 2)
        head = head[:room].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
        text = f"{head} {check}"
    return text
