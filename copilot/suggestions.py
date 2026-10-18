"""Вопросы-подсказки для панели Copilot по типам нарушений в результате (A15: GET …/suggestions)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from .prompts import completeness, normalize_result

SUGGESTIONS_PATH = Path(__file__).resolve().parent / "suggestions.yaml"


@lru_cache(maxsize=1)
def _load() -> dict[str, Any]:
    import yaml

    with open(SUGGESTIONS_PATH, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


def suggestions(job_result: Any, limit: int = 4) -> dict[str, Any]:
    """{"job_id", "suggestions": [...], "by_type": {type_id: [...]}} — без вопросов о невидимом."""
    res = normalize_result(job_result)
    data = _load()
    types = data.get("types") or {}
    by_type: dict[str, list[str]] = {}
    ordered: list[str] = []
    viols = res.get("violations") or []
    if completeness(res)["state"] == "incomplete":
        ordered.extend(data.get("incomplete") or [])
    if not viols:
        ordered.extend(data.get("no_violation") or [])
    for v in viols:
        tid = str(v.get("type_id") or "")
        qs = list(types.get(tid) or types.get("V99") or [])
        if tid and tid not in by_type:
            by_type[tid] = qs
            ordered.extend(qs)
    ordered.extend(data.get("general") or [])
    uniq = list(dict.fromkeys(str(q) for q in ordered if str(q).strip()))
    return {"job_id": str(res.get("job_id") or ""), "suggestions": uniq[:max(1, int(limit))], "by_type": by_type}
