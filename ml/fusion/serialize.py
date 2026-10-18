"""JSON-сериализация для ml.fusion: события ↔ dict, AnalysisResult → ответ API.

``result_to_api`` — локальный fallback точно по схеме ТЗ (+ доп. поля); в проде используется
``contracts.serialize.result_to_api`` (A01), если он есть.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any, Optional

from ._types import AnalysisResult, Event


def to_jsonable(obj: Any, ndigits: Optional[int] = None) -> Any:
    """dataclass / numpy / tuple / set / Path → чистые JSON-типы. NaN/inf → None."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, ndigits) if ndigits is not None else obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name), ndigits) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v, ndigits) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    # numpy без жёсткой зависимости
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return to_jsonable(tolist(), ndigits)
    item = getattr(obj, "item", None)
    if callable(item):
        return to_jsonable(item(), ndigits)
    return str(obj)


def event_to_dict(ev: Any) -> dict[str, Any]:
    """Event (или совместимый объект/словарь) → dict."""
    if isinstance(ev, dict):
        return to_jsonable(ev, 4)
    return {
        "type_id": str(getattr(ev, "type_id", "")),
        "t_start": getattr(ev, "t_start", 0.0),
        "t_end": getattr(ev, "t_end", 0.0),
        "t_peak": getattr(ev, "t_peak", 0.0),
        "confidence": getattr(ev, "confidence", 0.0),
        "source": str(getattr(ev, "source", "")),
        "facts": to_jsonable(list(getattr(ev, "facts", []) or [])),
        "bboxes": to_jsonable(dict(getattr(ev, "bboxes", {}) or {}), 4),
        "track_ids": to_jsonable(list(getattr(ev, "track_ids", []) or [])),
        "extra": to_jsonable(dict(getattr(ev, "extra", {}) or {}), 4),
    }


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(out) or math.isinf(out) else out


def event_from_dict(d: dict[str, Any]) -> Event:
    """dict → Event (bbox-ы → кортежи)."""
    bboxes: dict[str, Any] = {}
    for key, box in (d.get("bboxes") or {}).items():
        try:
            x1, y1, x2, y2 = (float(v) for v in box)
        except (TypeError, ValueError):
            continue
        bboxes[str(key)] = (x1, y1, x2, y2)
    t_peak = _f(d.get("t_peak", d.get("t", 0.0)))
    return Event(
        type_id=str(d.get("type_id", "V99")),
        t_start=_f(d.get("t_start", t_peak), t_peak),
        t_end=_f(d.get("t_end", t_peak), t_peak),
        t_peak=t_peak,
        confidence=_f(d.get("confidence", 0.0)),
        source=str(d.get("source", "")),
        facts=[str(f) for f in d.get("facts") or []],
        bboxes=bboxes,
        track_ids=[int(t) for t in d.get("track_ids") or [] if str(t).lstrip("-").isdigit()],
        extra=dict(d.get("extra") or {}),
    )


JURY_VIOLATION_KEYS = ("type", "timecode_sec", "description", "evidence_frames")
JURY_TOP_KEYS = ("job_id", "status", "verdict", "violations")


def result_to_jury(result: AnalysisResult) -> dict[str, Any]:
    """Строгий ответ жюри. Канон — ``contracts.serialize.result_to_jury_api`` (A01: схема ТЗ, политика вердикта,
    проверка meta.coverage перед no_violation). Старый A01 без него — ``result_to_api``. Без contracts —
    локальный fallback ровно по схеме ТЗ с той же политикой: failed/processing → verdict null; done без
    вердикта и без нарушений → failed («нет нарушения» не выдумываем)."""
    try:
        import contracts.serialize as a01  # type: ignore[import-not-found]
    except ImportError:
        a01 = None
    fn = (getattr(a01, "result_to_jury_api", None) or getattr(a01, "result_to_api", None)) if a01 else None
    if fn is not None:
        return fn(result)
    status = str(result.status)
    violations = list(result.violations or []) if status == "done" else []
    verdict = result.verdict if status == "done" else None
    if status == "done" and verdict is None:
        if violations:
            verdict = "violation"
        else:
            status = "failed"
    if verdict == "no_violation":
        violations = []
    out = {
        "job_id": str(result.job_id),
        "status": status,
        "verdict": verdict,
        "violations": [
            {
                "type": str(v.type),
                "timecode_sec": float(v.timecode_sec),
                "description": str(v.description),
                "evidence_frames": [
                    {"timestamp_sec": float(f.timestamp_sec), "image_base64": str(f.image_base64)}
                    for f in (v.evidence_frames or [])
                ],
            }
            for v in violations
        ],
    }
    return out


def result_to_ui_report(result: AnalysisResult) -> dict[str, Any]:
    """Полный отчёт для UI/проверяющего: всё из API + нормы со статусом сверки, подсказки, оверлеи,
    полнота анализа. Не для автоматической проверки жюри."""
    out = result_to_api(result)
    meta = result.meta or {}
    ui = meta.get("ui") or {}
    out["ui"] = to_jsonable(ui)
    out["analysis_complete"] = meta.get("analysis_complete")
    out["warnings"] = to_jsonable(meta.get("warnings") or [])
    out["disclaimer"] = meta.get("disclaimer", "Результат — подсказка для проверяющего, а не юридическое заключение.")
    for v in out["violations"]:
        for r in v.get("rules") or []:
            r["status_ru"] = ("норма сверена с источником" if r.get("verified")
                              else "НЕ подтверждено: требует сверки юристом")
    return out


def result_to_api(result: AnalysisResult) -> dict[str, Any]:
    """AnalysisResult → JSON ответа ``GET /api/v1/jobs/{job_id}``: обязательные поля ТЗ + доп. поля.

    Fallback до слияния A01 (contracts.serialize). ``rules[].verified`` — как в каталоге, не «повышается».
    """
    violations = []
    for v in result.violations or []:
        violations.append(
            {
                # --- обязательные поля ТЗ ---
                "type": str(v.type),
                "timecode_sec": float(v.timecode_sec),
                "description": str(v.description),
                "evidence_frames": [
                    {"timestamp_sec": float(f.timestamp_sec), "image_base64": str(f.image_base64)}
                    for f in (v.evidence_frames or [])
                ],
                # --- дополнительные поля ---
                "type_id": str(v.type_id),
                "confidence": float(v.confidence),
                "t_start": float(v.t_start),
                "t_end": float(v.t_end),
                "rules": [to_jsonable(r) for r in (v.rules or [])],
            }
        )
    out: dict[str, Any] = {
        "job_id": str(result.job_id),
        "status": str(result.status),
        "verdict": result.verdict,
        "violations": violations,
        "timeline": [[float(t), float(r)] for t, r in (result.timeline or [])],
        "meta": to_jsonable(result.meta or {}),
    }
    return out
