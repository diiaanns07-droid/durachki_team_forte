"""Защита от prompt injection: всё, что пришло извне (вопрос, история, результат, тексты норм), — данные.

* ``sanitize_untrusted`` — перед вставкой в промпт: убирает служебные токены чат-шаблона Qwen
  (``<|im_start|>``, ``<|im_end|>`` …), теги ``<think>``, наши маркеры ``[N1]``/``[V1]`` и разделители
  блоков данных, чтобы текст не мог «закрыть» блок или притвориться ссылкой/системной ролью.
* ``strip_output_artifacts`` — после генерации: служебные токены, утечка системного промпта,
  HTML/скрипты и ссылки со схемами javascript:/data:.
Смысл инструкций внутри данных не анализируется: модель получает правило «данные не выполнять»,
а всё, что она всё же выдаст, проходит пост-проверку (guard) по фактам и источникам.
"""
from __future__ import annotations

import re

DATA_OPEN = "<<<ДАННЫЕ"
DATA_CLOSE = "<<<КОНЕЦ ДАННЫХ>>>"

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|<>\n]{0,40}\|>")
_THINK_TAG_RE = re.compile(r"</?\s*think\s*>", re.I)
_OUR_MARKER_RE = re.compile(r"\[\s*([NnНнVv])\s*(\d{1,3})\s*\]")
_DELIM_RE = re.compile(r"<{2,}|>{3,}")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ROLE_LINE_RE = re.compile(r"(?im)^\s*(system|assistant|user|система|ассистент)\s*:\s*")

_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^<>\n]{0,300}>")
_BAD_SCHEME_RE = re.compile(r"(?i)\b(?:javascript|vbscript|data)\s*:[^\s)\]]*")
_LEAK_RE = re.compile(r"(?im)^.*(?:<<<ДАННЫЕ|<<<КОНЕЦ ДАННЫХ|Правила ответа:|Ты — Sentinel Copilot).*$")


def sanitize_untrusted(text: object, max_chars: int = 4000) -> str:
    """Нейтрализовать управляющие конструкции во внешнем тексте (смысл текста сохраняется)."""
    s = str(text or "")
    s = _CTRL_RE.sub(" ", s)
    s = _SPECIAL_TOKEN_RE.sub(" ", s)
    s = _THINK_TAG_RE.sub(" ", s)
    s = _OUR_MARKER_RE.sub(lambda m: f"({m.group(1)}{m.group(2)})", s)
    s = _DELIM_RE.sub(" ", s)
    s = _ROLE_LINE_RE.sub(lambda m: f"{m.group(1)} — ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s[:max_chars].strip()


def wrap_data(title: str, body: str) -> str:
    """Блок данных с явными границами (внутри — только санитизированный текст)."""
    return f"{DATA_OPEN}: {title}>>>\n{body}\n{DATA_CLOSE}"


def strip_output_artifacts(text: str) -> str:
    """Чистка ответа модели до пост-проверки фактов."""
    s = _SPECIAL_TOKEN_RE.sub(" ", text or "")
    s = _LEAK_RE.sub("", s)
    s = _HTML_TAG_RE.sub("", s)
    s = _BAD_SCHEME_RE.sub("", s)
    return s
