"""Полнота анализа: пустой список нарушений после падения inference НЕ означает «нарушений нет».

Оркестратор (A03) после ``fuse()`` вызывает::

    result = apply_completeness(result, {"ingest": "ok", "detect": "failed: CUDA OOM", "rules": "ok",
                                         "temporal": "timeout", "vlm": "skipped"})

* упала/не выполнена обязательная стадия (completeness.required) и нарушений нет →
  ``status="failed"``, ``verdict=None``, ``meta.failure`` с причиной (технический отказ, а не «чисто»);
* упала обязательная стадия, но нарушения найдены → ``verdict="violation"`` остаётся,
  ``meta.analysis_complete=False`` (найденное не прячем, неполнота видна в UI);
* упали только необязательные стадии → вердикт остаётся, ``meta.degraded`` для UI.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional

from ._types import AnalysisResult
from .config import load_config
from .timing import clip_times

_ALIASES = {
    "decode": "ingest", "load_video": "ingest", "video": "ingest",
    "detection": "detect", "detector": "detect", "det": "detect",
    "tracking": "track", "tracker": "track",
    "rule": "rules", "rule_engine": "rules", "rules_engine": "rules",
    "scene_states": "scene", "pose_estimation": "pose",
    "vlm_verify": "vlm", "vlm_discover": "vlm",
}


def _status_of(value: Any) -> tuple[str, str]:
    """→ (status, detail). Принимает str / bool / {"status": ..., "error": ...}."""
    if isinstance(value, bool):
        return ("ok" if value else "failed"), ""
    if isinstance(value, Mapping):
        status = str(value.get("status", value.get("state", "")) or "").strip().lower()
        detail = str(value.get("error", value.get("detail", value.get("reason", ""))) or "")
        return status or ("failed" if detail else "unknown"), detail
    text = str(value or "").strip()
    head, _, detail = text.partition(":")
    return head.strip().lower() or "unknown", detail.strip()


def normalize_stages(stages: Optional[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for name, value in (stages or {}).items():
        key = _ALIASES.get(str(name).strip().lower(), str(name).strip().lower())
        status, detail = _status_of(value)
        prev = out.get(key)
        # если стадия пришла под двумя именами — худший статус побеждает
        if prev is None or prev["status"] in {"ok", "done", "success", "true"}:
            out[key] = {"status": status, "detail": detail}
    return out


def coverage_info(clip: Any, duration: float, cfg: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """meta.coverage: насколько видео реально покрыто анализом.

    Источники: ``clip.meta["coverage"]``/``["complete"]`` от A04 (декодированная доля длительности) и
    сетка ``clip.times`` (доля длительности, покрытая сэмплами, максимальный разрыв).
    ``complete``: True — покрыто; False — неполно (``reasons``); None — неизвестно (нет данных).
    """
    cfg = cfg or load_config()
    ccfg = cfg.get("completeness") or {}
    min_decoded = float(ccfg.get("min_decoded", 0.98))
    min_sampled = float(ccfg.get("min_sampled", 0.9))
    max_gap_s = float(ccfg.get("max_gap_s", 2.0))
    out: dict[str, Any] = {"checked": False, "complete": None, "reasons": []}
    meta = getattr(clip, "meta", None)
    meta = meta if isinstance(meta, Mapping) else {}
    if meta.get("coverage") is not None:
        try:
            decoded = float(meta["coverage"])
        except (TypeError, ValueError):
            decoded = None
        if decoded is not None:
            out["decoded"] = round(decoded, 4)
            out["checked"] = True
            if decoded < min_decoded:
                out["reasons"].append(f"декодировано {decoded:.0%} длительности видео (< {min_decoded:.0%})")
    if meta.get("complete") is False:
        out["checked"] = True
        if not out["reasons"]:
            out["reasons"].append("ingest: видео декодировано не полностью")
    times = clip_times(clip)
    if times and duration > 0:
        diffs = sorted(b - a for a, b in zip(times, times[1:]) if b > a)
        step = diffs[len(diffs) // 2] if diffs else min(duration, 0.2)
        gaps = [times[0] - 0.0] + [b - a for a, b in zip(times, times[1:])] + [duration - times[-1]]
        missing = sum(max(0.0, g - max(step, 0.0) * 1.5) for g in gaps if g > max_gap_s)
        sampled = max(0.0, min(1.0, 1.0 - missing / duration))
        max_gap = max(gaps)
        out.update(sampled=round(sampled, 4), max_gap_s=round(max_gap, 3), n_samples=len(times),
                   checked=True)
        if sampled < min_sampled:
            out["reasons"].append(f"сэмплами покрыто {sampled:.0%} длительности (< {min_sampled:.0%})")
        if max_gap > max_gap_s:
            out["reasons"].append(f"разрыв без кадров {max_gap:.1f} с (> {max_gap_s:g} с)")
    elif duration > 0 and not times and out["checked"] is False:
        out["reasons"].append("нет сэмплированных кадров (clip.times пуст)")
        out["checked"] = True
    if out["checked"]:
        out["complete"] = not out["reasons"]
    return out


def apply_completeness(result: AnalysisResult, stages: Optional[Mapping[str, Any]],
                       cfg: Optional[dict[str, Any]] = None) -> AnalysisResult:
    """Помечает неполный анализ. Возвращает НОВЫЙ AnalysisResult (исходный не меняется)."""
    cfg = cfg or load_config()
    ccfg = cfg.get("completeness") or {}
    required = [str(s).lower() for s in ccfg.get("required") or ["ingest", "detect", "rules"]]
    optional = [str(s).lower() for s in ccfg.get("optional") or []]
    ok_values = {str(v).lower() for v in ccfg.get("ok_values") or ["ok", "done", "success", "true"]}

    norm = normalize_stages(stages)
    failed_required = [s for s in required if norm.get(s, {"status": "not_run"})["status"] not in ok_values]
    degraded = [s for s in optional if s in norm and norm[s]["status"] not in ok_values]
    degraded += [s for s in norm if s not in required and s not in optional and norm[s]["status"] not in ok_values]

    meta = dict(result.meta or {})
    meta["stages"] = norm
    meta["analysis_complete"] = not failed_required and result.status != "failed"
    if degraded:
        meta["degraded"] = sorted(set(degraded))
    warnings = list(meta.get("warnings") or [])
    ui = dict(meta.get("ui") or {})

    if result.status == "failed":
        meta["warnings"] = warnings
        return dataclasses.replace(result, meta=meta)

    if not failed_required:
        if degraded:
            warnings.append("анализ выполнен с деградацией: " + ", ".join(sorted(set(degraded))))
        meta["warnings"] = warnings
        return dataclasses.replace(result, meta=meta)

    details = {s: norm.get(s, {"status": "not_run", "detail": ""}) for s in failed_required}
    msg = ("Анализ не завершён: не выполнены стадии " + ", ".join(failed_required) +
           ". Отсутствие нарушений не подтверждено; требуется повторный запуск или ручная проверка.")
    ui["incomplete"] = {"failed_stages": failed_required, "details": details, "message_ru": msg}
    meta["ui"] = ui
    if result.violations:
        warnings.append("анализ неполный (" + ", ".join(failed_required) + "): найденные нарушения показаны, "
                        "но могли быть пропущены другие")
        meta["warnings"] = warnings
        return dataclasses.replace(result, meta=meta, verdict="violation")
    meta["failure"] = {"reason": "incomplete_analysis", "failed_stages": failed_required, "details": details,
                       "message_ru": msg}
    warnings.append(msg)
    meta["warnings"] = warnings
    return dataclasses.replace(result, status="failed", verdict=None, violations=[], meta=meta)
