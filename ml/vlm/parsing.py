"""Надёжное извлечение JSON из ответа VLM: code fences, <think>, мусор вокруг, одинарные кавычки,
висячие запятые, Python-литералы, обрезанный хвост. Затем — нормализация полей под схему."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

def verify_schema(type_ids: Optional[list[str]] = None, n_frames: int = 6) -> dict[str, Any]:
    """JSON-схема ответа verify (для llama.cpp превращается в грамматику: enum и диапазоны соблюдаются)."""
    n = max(1, int(n_frames))
    tid: dict[str, Any] = {"type": "string", "enum": list(type_ids)} if type_ids else {"type": "string", "pattern": "^V[0-9]{2}$"}
    return {
        "type": "object",
        "properties": {
            "visible": {"type": "boolean"},
            "type_id": tid,
            "observed_facts": {"type": "array", "items": {"type": "string", "maxLength": 200}, "maxItems": 3},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "best_frame": {"type": "integer", "minimum": 1, "maximum": n},
            "visible_frames": {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": n}, "maxItems": n},
            "uncertain_because": {"type": "string", "maxLength": 200},
        },
        "required": ["visible", "type_id", "observed_facts", "confidence", "best_frame", "visible_frames",
                     "uncertain_because"],
        "additionalProperties": False,
    }


VERIFY_SCHEMA: dict[str, Any] = verify_schema()


def discover_schema(items: list[str], n_frames: int = 4) -> dict[str, Any]:
    n = max(1, int(n_frames))
    return {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string", "enum": list(items)},
                        "observed_facts": {"type": "array", "items": {"type": "string", "maxLength": 200}, "maxItems": 2},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "frame": {"type": "integer", "minimum": 1, "maximum": n},
                    },
                    "required": ["item", "observed_facts", "confidence", "frame"],
                    "additionalProperties": False,
                },
            },
            "uncertain_because": {"type": "string", "maxLength": 200},
        },
        "required": ["findings", "uncertain_because"],
        "additionalProperties": False,
    }


@dataclass
class ParseResult:
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    repaired: bool = False
    raw: str = ""


_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.S | re.I)
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)(```|$)", re.S)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_UNQUOTED_KEY_RE = re.compile(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)')
_PY_LITERALS = ((r"\bTrue\b", "true"), (r"\bFalse\b", "false"), (r"\bNone\b", "null"))


def _strip_wrappers(text: str) -> str:
    text = _THINK_RE.sub("", text or "")
    m = _FENCE_RE.search(text)
    if m and "{" in m.group(1):
        return m.group(1)
    return text


def _balanced_candidates(text: str) -> list[str]:
    """Все top-level {...} блоки с учётом строк; незакрытый хвостовой блок — тоже (для ремонта)."""
    out: list[str] = []
    depth = 0
    start = -1
    in_str: Optional[str] = None
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == in_str:
                in_str = None
            continue
        if ch in ('"', "'") and depth > 0:
            in_str = ch
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(text[start:i + 1])
                start = -1
    if depth > 0 and start >= 0:
        out.append(text[start:])  # обрезанный ответ (max_new_tokens)
    return out


def _close_truncated(s: str) -> str:
    """Дописывает недостающие кавычки/скобки у обрезанного JSON."""
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    s2 = s + ('"' if in_str else "")
    s2 = re.sub(r",\s*$", "", s2.rstrip())
    s2 = re.sub(r'(,\s*"[^"]*"\s*:?\s*)$', "", s2) if not in_str else s2  # висячий ключ без значения
    s2 = re.sub(r":\s*$", ": null", s2)
    return s2 + "".join(reversed(stack))


def _repair(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("«", '"').replace("»", '"')
    for pat, rep in _PY_LITERALS:
        s = re.sub(pat, rep, s)
    if '"' not in s and "'" in s:
        s = s.replace("'", '"')
    else:
        # одинарные кавычки вокруг ключей/значений: 'key': 'value'
        s = re.sub(r"(?<=[{,\[:\s])'([^'\"]*)'(?=\s*[:,}\]])", r'"\1"', s)
    s = _UNQUOTED_KEY_RE.sub(r'\1"\2"\3', s)
    s = _TRAILING_COMMA_RE.sub(r"\1", s)
    return s


def extract_json(text: str) -> ParseResult:
    raw = text or ""
    body = _strip_wrappers(raw)
    cands = _balanced_candidates(body)
    if not cands:
        return ParseResult(ok=False, error="no_json_object", raw=raw)
    # предпочитаем самый длинный валидный объект
    for cand in sorted(cands, key=len, reverse=True):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return ParseResult(ok=True, data=obj, raw=raw)
        except json.JSONDecodeError:
            pass
    for cand in sorted(cands, key=len, reverse=True):
        for attempt in (_repair(cand), _close_truncated(_repair(cand))):
            try:
                obj = json.loads(attempt)
                if isinstance(obj, dict):
                    return ParseResult(ok=True, data=obj, repaired=True, raw=raw)
            except json.JSONDecodeError:
                continue
    return ParseResult(ok=False, error="invalid_json", raw=raw)


# ---------------------------------------------------------------- нормализация полей

_YES = {"true", "yes", "да", "1", "visible", "видно", "y", "истина"}
_NO = {"false", "no", "нет", "0", "not visible", "не видно", "n", "ложь", "none", "null"}
_CONF_WORDS = {"very high": 0.9, "high": 0.8, "высокая": 0.8, "medium": 0.55, "moderate": 0.55,
               "средняя": 0.55, "low": 0.3, "низкая": 0.3, "very low": 0.15}


def coerce_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in _YES:
            return True
        if s in _NO:
            return False
    return None


def coerce_conf(v: Any) -> Optional[float]:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        x = float(v)
    elif isinstance(v, str):
        s = v.strip().lower().replace(",", ".")
        if s in _CONF_WORDS:
            return _CONF_WORDS[s]
        m = re.search(r"-?\d+(\.\d+)?", s)
        if not m:
            return None
        x = float(m.group(0))
        if "%" in s:
            x /= 100.0
    else:
        return None
    if x != x or x in (float("inf"), float("-inf")):  # NaN/inf
        return None
    if 1.0 < x <= 100.0:  # проценты
        x /= 100.0
    if x < 0.0 or x > 1.0:
        return None  # вне диапазона — считаем, что уверенности нет (флаг confidence_invalid)
    return x


def coerce_type_id(v: Any, allowed: Optional[set[str]] = None) -> Optional[str]:
    if v is None:
        return None
    m = re.search(r"V\s*-?\s*(\d{1,2})", str(v).upper())
    if not m:
        return None
    tid = f"V{int(m.group(1)):02d}"
    if allowed is not None and tid not in allowed:
        return None  # вне enum каталога — флаг type_id_invalid
    return tid


def coerce_str_list(v: Any, max_items: int = 6) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        parts = [p.strip(" -•\t") for p in re.split(r"[\n;]+", v)]
    elif isinstance(v, (list, tuple)):
        parts = []
        for x in v:
            if isinstance(x, dict):  # {"fact": "..."} — бывает у маленьких моделей
                x = x.get("fact") or x.get("text") or next(iter(x.values()), "")
            parts.append(str(x).strip())
    else:
        parts = [str(v).strip()]
    out: list[str] = []
    for p in parts:
        if p and p not in out:
            out.append(p)
    return out[:max_items]


def coerce_int(v: Any) -> Optional[int]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        m = re.search(r"\d+", v)
        return int(m.group(0)) if m else None
    return None


def coerce_int_list(v: Any) -> list[int]:
    if v is None:
        return []
    if isinstance(v, (int, float, str)) and not isinstance(v, bool):
        v = [v] if not isinstance(v, str) else re.findall(r"\d+", v)
    out: list[int] = []
    if isinstance(v, (list, tuple)):
        for x in v:
            i = coerce_int(x)
            if i is not None and i not in out:
                out.append(i)
    return out


def normalize_verify(data: dict[str, Any], allowed_types: Optional[set[str]] = None,
                     n_frames: Optional[int] = None) -> tuple[Optional[dict[str, Any]], str]:
    """Приводит ответ verify к схеме и валидирует enum/числа/номера кадров.

    (None, причина) — если нет обязательного 'visible'. Невалидные поля -> None + флаг в 'flags'.
    """
    d = {str(k).strip().lower(): v for k, v in data.items()}
    visible = coerce_bool(d.get("visible", d.get("is_visible", d.get("violation"))))
    if visible is None:
        return None, "missing_visible"
    flags: list[str] = []
    raw_conf = d.get("confidence", d.get("conf"))
    conf = coerce_conf(raw_conf)
    if conf is None:
        flags.append("confidence_missing" if raw_conf is None else "confidence_invalid")
    raw_tid = d.get("type_id", d.get("type"))
    tid = coerce_type_id(raw_tid, allowed_types)
    if tid is None and raw_tid not in (None, ""):
        flags.append("type_id_invalid")
    best = coerce_int(d.get("best_frame", d.get("frame")))
    vis_frames = coerce_int_list(d.get("visible_frames"))
    if n_frames is not None:
        if best is not None and not 1 <= best <= n_frames:
            flags.append("best_frame_out_of_range")
            best = None
        bad = [i for i in vis_frames if not 1 <= i <= n_frames]
        if bad:
            flags.append("visible_frames_out_of_range")
            vis_frames = [i for i in vis_frames if 1 <= i <= n_frames]
    out = {
        "visible": visible,
        "type_id": tid,
        "observed_facts": coerce_str_list(d.get("observed_facts", d.get("facts"))),
        "confidence": 0.5 if conf is None else conf,
        "confidence_missing": conf is None,
        "best_frame": best,
        "visible_frames": sorted(vis_frames),
        "uncertain_because": str(d.get("uncertain_because") or "").strip()[:300],
        "flags": flags,
    }
    return out, ""


def normalize_discover(data: dict[str, Any], items: list[str],
                       n_frames: Optional[int] = None) -> tuple[Optional[dict[str, Any]], str]:
    d = {str(k).strip().lower(): v for k, v in data.items()}
    raw = d.get("findings")
    if raw is None:
        # формат «чек-лист»: {"phone": true, ...}
        flat = [{"item": k, "observed_facts": [], "confidence": None} for k in items if coerce_bool(d.get(k))]
        if not flat and not any(k in d for k in items):
            return None, "missing_findings"
        raw = flat
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return None, "findings_not_list"
    findings = []
    flags: list[str] = []
    for f in raw:
        if not isinstance(f, dict):
            flags.append("finding_not_object")
            continue
        fd = {str(k).strip().lower(): v for k, v in f.items()}
        item = str(fd.get("item") or fd.get("name") or "").strip().lower()
        if item not in items:
            flags.append("item_not_in_enum")
            continue
        if coerce_bool(fd.get("present", fd.get("visible", True))) is False:
            continue
        raw_conf = fd.get("confidence")
        conf = coerce_conf(raw_conf)
        if conf is None and raw_conf is not None:
            flags.append("confidence_invalid")
        frame = coerce_int(fd.get("frame", fd.get("best_frame")))
        if frame is not None and n_frames is not None and not 1 <= frame <= n_frames:
            flags.append("frame_out_of_range")
            frame = None
        findings.append({
            "item": item,
            "observed_facts": coerce_str_list(fd.get("observed_facts", fd.get("fact", fd.get("facts")))),
            "confidence": 0.5 if conf is None else conf,
            "confidence_missing": conf is None,
            "frame": frame,
        })
    return {"findings": findings, "uncertain_because": str(d.get("uncertain_because") or "")[:300],
            "flags": flags}, ""


def regex_fallback_verify(text: str, allowed_types: Optional[set[str]] = None) -> Optional[dict[str, Any]]:
    """Последний шанс: вытащить ключевые поля регэкспами из текста, который не стал JSON."""
    t = text or ""
    mv = re.search(r"visible\W{0,4}\s*(true|false|yes|no|да|нет)", t, re.I)
    if not mv:
        return None
    mc = re.search(r"confidence\W{0,4}\s*([0-9.]+%?)", t, re.I)
    mt = re.search(r"type_id\W{0,4}\s*(V\s*-?\d{1,2})", t, re.I)
    conf = coerce_conf(mc.group(1)) if mc else None
    return {
        "visible": coerce_bool(mv.group(1)),
        "type_id": coerce_type_id(mt.group(1), allowed_types) if mt else None,
        "observed_facts": [],  # из нераспарсенного текста факты не берём
        "confidence": 0.5 if conf is None else conf,
        "confidence_missing": conf is None,
        "best_frame": None,
        "visible_frames": [],
        "uncertain_because": "",
        "flags": ["regex_fallback"],
    }
