"""Проверка ответа API на соответствие схеме ТЗ (+ мягкие проверки качества ответа).

Обязательное по ТЗ::

    {"job_id", "status": "processing|done|failed", "verdict": "violation|no_violation",
     "violations": [{"type", "timecode_sec", "description",
                     "evidence_frames": [{"timestamp_sec", "image_base64"}]}]}

Если есть ``contracts/api_schema.json`` (A01) и установлен jsonschema — проверяем и по нему.
"""
from __future__ import annotations

import base64
import binascii
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .common import REPO_ROOT

STATUSES = ("processing", "done", "failed")
VERDICTS = ("violation", "no_violation")

TZ_JSON_SCHEMA: dict = {
    "type": "object",
    "required": ["job_id", "status"],
    "properties": {
        "job_id": {"type": "string", "minLength": 1},
        "status": {"enum": list(STATUSES)},
        "verdict": {"enum": list(VERDICTS) + [None]},
        "violations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type", "timecode_sec", "description", "evidence_frames"],
                "properties": {
                    "type": {"type": "string", "minLength": 1},
                    "timecode_sec": {"type": "number", "minimum": 0},
                    "description": {"type": "string", "minLength": 1},
                    "evidence_frames": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["timestamp_sec", "image_base64"],
                            "properties": {"timestamp_sec": {"type": "number", "minimum": 0},
                                           "image_base64": {"type": "string", "minLength": 1}},
                        },
                    },
                },
            },
        },
    },
}


@dataclass
class SchemaReport:
    errors: list[str] = field(default_factory=list)     # нарушение схемы ТЗ — жюри может не принять ответ
    warnings: list[str] = field(default_factory=list)   # качество ответа
    frames_total: int = 0
    frames_valid_jpeg: int = 0

    @property
    def valid(self) -> bool:
        return not self.errors


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def decode_jpeg_b64(s: Any) -> Optional[bytes]:
    """bytes JPEG или None. data:-префикс допускаем (с предупреждением в check_api_response)."""
    if not isinstance(s, str) or not s:
        return None
    if s.startswith("data:"):
        s = s.split(",", 1)[-1]
    try:
        raw = base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError):
        return None
    if len(raw) < 4 or raw[:2] != b"\xff\xd8":
        return None
    return raw


def _contract_schema() -> Optional[dict]:
    p = REPO_ROOT / "contracts" / "api_schema.json"
    if not p.exists():
        return None
    try:
        import json

        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def check_api_response(d: Any, duration: Optional[float] = None, max_frames: int = 3,
                       require_done: bool = True, use_contract_schema: bool = True) -> SchemaReport:
    rep = SchemaReport()
    E, W = rep.errors.append, rep.warnings.append
    if not isinstance(d, dict):
        E("ответ не JSON-объект")
        return rep
    if not isinstance(d.get("job_id"), str) or not d.get("job_id"):
        E("job_id: нет или не строка")
    status = d.get("status")
    if status not in STATUSES:
        E(f"status: {status!r} не из {STATUSES}")
    if require_done and status != "done":
        E(f"status={status!r}, ожидался done")
    if status == "done":
        if d.get("verdict") not in VERDICTS:
            E(f"verdict: {d.get('verdict')!r} не из {VERDICTS}")
        if not isinstance(d.get("violations"), list):
            E("violations: нет или не список")
    vios = d.get("violations") if isinstance(d.get("violations"), list) else []
    if d.get("verdict") == "violation" and not vios:
        W("verdict=violation, но violations пуст")
    if d.get("verdict") == "no_violation" and vios:
        W(f"verdict=no_violation, но violations: {len(vios)}")
    for i, v in enumerate(vios):
        p = f"violations[{i}]"
        if not isinstance(v, dict):
            E(f"{p}: не объект")
            continue
        if not isinstance(v.get("type"), str) or not v.get("type", "").strip():
            E(f"{p}.type: нет или пустая строка")
        tc = v.get("timecode_sec")
        if not _is_num(tc):
            E(f"{p}.timecode_sec: {tc!r} не число")
        elif tc < 0:
            E(f"{p}.timecode_sec < 0")
        elif duration and tc > duration + 0.5:
            W(f"{p}.timecode_sec={tc} больше длительности {duration:.1f}")
        if not isinstance(v.get("description"), str) or not v.get("description", "").strip():
            E(f"{p}.description: нет или пустая строка")
        fr = v.get("evidence_frames")
        if not isinstance(fr, list):
            E(f"{p}.evidence_frames: нет или не список")
            continue
        if not 1 <= len(fr) <= max_frames:
            W(f"{p}.evidence_frames: {len(fr)} кадров (ожидается 1–{max_frames})")
        for j, f in enumerate(fr):
            q = f"{p}.evidence_frames[{j}]"
            if not isinstance(f, dict):
                E(f"{q}: не объект")
                continue
            rep.frames_total += 1
            if not _is_num(f.get("timestamp_sec")):
                E(f"{q}.timestamp_sec: не число")
            elif _is_num(tc) and abs(f["timestamp_sec"] - tc) > 5.0:
                W(f"{q}.timestamp_sec={f['timestamp_sec']} далеко от timecode_sec={tc}")
            b = f.get("image_base64")
            if not isinstance(b, str) or not b:
                E(f"{q}.image_base64: нет или пусто")
                continue
            if b.startswith("data:"):
                W(f"{q}.image_base64 с data:-префиксом (по ТЗ — чистый base64 JPEG)")
            if decode_jpeg_b64(b) is None:
                W(f"{q}.image_base64: не декодируется как JPEG")
            else:
                rep.frames_valid_jpeg += 1
    if use_contract_schema:
        schema = _contract_schema()
        if schema is not None:
            try:
                import jsonschema  # лениво

                for err in jsonschema.Draft202012Validator(schema).iter_errors(d):
                    loc = "/".join(str(x) for x in err.absolute_path)
                    msg = f"api_schema.json: {loc}: {err.message}"[:300]
                    if msg not in rep.errors:
                        E(msg)
            except ImportError:
                pass
            except Exception as e:  # noqa: BLE001 — кривая схема не должна ронять оценку
                W(f"api_schema.json не применена: {e}")
    return rep


def check_with_jsonschema(d: Any) -> list[str]:
    """Только встроенная JSON-схема ТЗ (для jury_sim, если нужна «чистая» проверка)."""
    try:
        import jsonschema
    except ImportError:
        return check_api_response(d, require_done=False, use_contract_schema=False).errors
    return [f"{'/'.join(map(str, e.absolute_path))}: {e.message}"
            for e in jsonschema.Draft202012Validator(TZ_JSON_SCHEMA).iter_errors(d)]


def load_schema_file(path: str | Path) -> dict:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- полнота анализа (канон A01)
def _f(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def min_coverage_ratio(cfg: Optional[dict] = None) -> float:
    """Порог полноты для «нет нарушения»: eval config coverage.min, иначе A01 (contracts/config.yaml), иначе 0.98."""
    v = ((cfg or {}).get("coverage") or {}).get("min")
    if v is not None:
        return float(v)
    try:
        from contracts.serialize import load_config as _a01_config  # type: ignore
    except ImportError:
        return 0.98
    return float(_a01_config()["coverage"]["min_ratio_for_no_violation"])


def coverage_ratio(meta: Optional[dict]) -> Optional[float]:
    """Доля проанализированного видео: канон A01 — meta.coverage.ratio; legacy — число. None — не сообщено."""
    cov = (meta or {}).get("coverage")
    if isinstance(cov, dict):
        return _f(cov.get("ratio"))
    return _f(cov)


def coverage_problem(meta: Optional[dict], cfg: Optional[dict] = None) -> Optional[str]:
    """None — полнота подтверждена и «нет нарушения» допустимо; иначе причина (как contracts.extensions A01).

    Если модуль A01 есть — используется он (одна семантика с production-сериализатором); иначе зеркало его правил.
    coverage.require=false — только для старых прогонов без coverage (отсутствие тогда не считается проблемой).
    """
    meta = meta or {}
    require = bool(((cfg or {}).get("coverage") or {}).get("require", True))
    if meta.get("coverage") is None and not require and meta.get("complete") is not False:
        return None
    min_ratio = min_coverage_ratio(cfg)
    try:
        from contracts.extensions import coverage_problem as _a01_problem, mock_allowed  # type: ignore
    except ImportError:
        _a01_problem = None
    if _a01_problem is not None:
        return _a01_problem(meta, min_ratio, mock_allowed())
    if meta.get("complete") is False:
        return "оркестратор сообщил незавершённый анализ (meta.complete=false)"
    cov = meta.get("coverage")
    if cov is None:
        return "полнота анализа не подтверждена: нет meta.coverage"
    ratio = coverage_ratio(meta)
    if ratio is None:
        return "meta.coverage.ratio не задан"
    if ratio + 1e-9 < min_ratio:
        return f"проанализировано {ratio:.0%} видео (< {min_ratio:.0%})"
    if isinstance(cov, dict) and cov.get("complete") is False:
        return "meta.coverage.complete=false"
    return None
