"""Запасные реализации оркестратора: клип для доказательств, грубые состояния из детекций,
слияние/вердикт без ml.fusion. Описания — только наблюдаемые факты, нормы — только из каталога."""
from __future__ import annotations

import threading
from typing import Any, Iterable, Mapping, Optional

from .config import cfg_get
from .contract import (AnalysisResult, Detection, Event, EvidenceFrame, RuleRef, SceneState, Track,
                       Violation)
from .media import blur_regions, encode_jpeg_b64, read_frame_at_pts

# Только названия (не нормы!) — на случай пустого каталога.
FALLBACK_NAMES = {
    "V01": "Посторонний в кассовом помещении",
    "V02": "Кассир покинул рабочее место, оставив ценности без присмотра",
    "V03": "Сейф открыт без присмотра или при посторонних",
    "V04": "Дверь кассового помещения открыта / вход постороннего",
    "V05": "Использование мобильного телефона при работе с ценностями",
    "V06": "Приём пищи или напитков на рабочем месте с наличными",
    "V07": "Нарушение порядка приёма-передачи инкассаторской сумки",
    "V08": "Пересчёт/передача наличных вне обзора камеры",
    "V99": "Иное подозрительное действие",
}


class EvidenceClip:
    """Минимальный VideoClip в процессе оркестратора: точные кадры по времени через OpenCV seek.
    Нужен для слияния/доказательств, когда воркер недоступен."""

    def __init__(self, path: str, info: Mapping[str, Any]) -> None:
        self.path = str(path)
        self.duration = float(info.get("duration") or 0.0)
        self.fps = float(info.get("fps") or 0.0)
        self.width = int(info.get("width") or 0)
        self.height = int(info.get("height") or 0)
        self.times = [float(t) for t in (info.get("times") or [])]
        self._lock = threading.Lock()

    def frame(self, i: int) -> Any:
        return self.frame_at(self.times[i])

    def frame_at(self, t: float) -> Any:
        """Кадр с PTS, ближайшим к t (переменный fps — время только из PTS)."""
        with self._lock:
            frame, _pts = read_frame_at_pts(self.path, t)
        return frame


def states_from_detections(times: Iterable[float], dets: list[Detection],
                           pose_states: list[SceneState]) -> list[SceneState]:
    """Грубые сигналы сцены прямо из детекций (если ml.scene недоступен). Только то, что видно."""
    by_t: dict[float, list[Detection]] = {}
    for d in dets:
        by_t.setdefault(round(d.t, 3), []).append(d)
    pose_by_t: dict[float, dict[str, float]] = {}
    for s in pose_states:
        pose_by_t.setdefault(round(s.t, 3), {}).update(s.signals)
    out: list[SceneState] = []
    for t in times:
        ds = by_t.get(round(float(t), 3), [])

        def best(label: str) -> float:
            return max((d.score for d in ds if d.label == label), default=0.0)

        persons = [d for d in ds if d.label == "person" and d.score >= 0.3]
        sig = {"n_persons": float(len(persons)), "cashier_present": 1.0 if persons else 0.0,
               "safe_open": best("safe_open"), "door_open": best("door_open"),
               "cash_visible": best("cash"), "cash_bag_visible": best("cash_bag")}
        for k, v in pose_by_t.get(round(float(t), 3), {}).items():
            sig[k] = max(sig.get(k, 0.0), float(v))
        out.append(SceneState(float(t), sig))
    return out


def derive_risk(times: list[float], events: list[Event]) -> list[tuple[float, float]]:
    """Кривая риска для UI из событий (если временной модели нет): max уверенности покрывающих событий."""
    out = []
    for t in times:
        r = 0.0
        for e in events:
            if e.t_start - 0.5 <= t <= e.t_end + 0.5:
                r = max(r, float(e.confidence))
        out.append((float(t), round(r, 4)))
    return out


def _merge(events: list[Event], window: float) -> list[Event]:
    """Слияние событий одного типа, близких по времени. Уверенность — максимум, факты — объединение."""
    merged: list[Event] = []
    for e in sorted(events, key=lambda x: (x.type_id, x.t_peak)):
        last = merged[-1] if merged else None
        if last is not None and last.type_id == e.type_id and e.t_start - last.t_end <= window:
            top = e if e.confidence > last.confidence else last
            facts = list(dict.fromkeys(last.facts + e.facts))
            merged[-1] = Event(e.type_id, min(last.t_start, e.t_start), max(last.t_end, e.t_end), top.t_peak,
                               max(last.confidence, e.confidence), "fusion", facts,
                               dict(top.bboxes), sorted(set(last.track_ids) | set(e.track_ids)),
                               {"sources": sorted(set(last.extra.get("sources", [last.source])) |
                                                  set(e.extra.get("sources", [e.source])))})
        else:
            merged.append(Event(e.type_id, e.t_start, e.t_end, e.t_peak, e.confidence, e.source, list(e.facts),
                                dict(e.bboxes), list(e.track_ids), {"sources": [e.source]}))
    return merged


def _fmt_t(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t // 60)}:{int(t % 60):02d}"


def _description(type_id: str, name: str, ev: Event, catalog: Mapping) -> str:
    facts = "; ".join(f for f in ev.facts if f) or "зафиксированы признаки, требующие проверки"
    tmpl = cfg_get(catalog, f"types.{type_id}.description_template", None)
    if isinstance(tmpl, str) and tmpl.strip():
        try:
            return tmpl.format_map(_SafeDict(t=_fmt_t(ev.t_peak), t_sec=round(ev.t_peak, 1), facts=facts,
                                             name=name))
        except Exception:  # noqa: BLE001
            pass
    return (f"Признаки возможного нарушения «{name}» около {_fmt_t(ev.t_peak)}: {facts}. "
            f"Требуется проверка человеком; это подсказка, а не юридическое заключение.")


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _rules_for(type_id: str, catalog: Mapping) -> list[RuleRef]:
    out = []
    for r in cfg_get(catalog, f"types.{type_id}.rules", []) or []:
        if isinstance(r, Mapping) and r.get("url"):
            out.append(RuleRef(document=str(r.get("document", "")), clause=r.get("clause"), url=str(r["url"]),
                               quote=str(r.get("quote", "")), verified=bool(r.get("verified", False))))
    return out


def _person_boxes_at(tracks: list[Track], t: float, tol: float = 0.25) -> list[tuple]:
    boxes = []
    for tr in tracks:
        if tr.label != "person":
            continue
        best = min(tr.boxes, key=lambda tb: abs(tb[0] - t), default=None)
        if best is not None and abs(best[0] - t) <= tol:
            boxes.append(tuple(best[1]))
    return boxes


def fallback_fuse(clip: Any, events: list[Event], risk: list[tuple[float, float]], catalog: Mapping,
                  job_id: str, cfg: Mapping, tracks: Optional[list[Track]] = None) -> AnalysisResult:
    """Слияние без ml.fusion: порог уверенности, слияние по окну, кадры-доказательства, нормы из каталога."""
    fcfg = cfg_get(cfg, "fallback_fusion", {}) or {}
    thr = float(fcfg.get("event_threshold", 0.5))
    rejected = [e for e in events if str(e.extra.get("vlm_verdict", "")).lower() == "rejected"]
    kept = [e for e in events if e not in rejected and float(e.confidence) >= thr]
    merged = _merge(kept, float(fcfg.get("merge_window_s", 8.0)))
    if not merged and risk:
        t_peak, r_peak = max(risk, key=lambda x: x[1])
        if r_peak >= float(fcfg.get("risk_threshold", 0.75)):
            merged = [Event("V99", max(0.0, t_peak - 2.0), t_peak + 2.0, t_peak, float(r_peak), "temporal",
                            ["временная модель отметила резкий рост риска в этом интервале"], {}, [],
                            {"sources": ["temporal"]})]
    merged.sort(key=lambda e: -e.confidence)
    merged = merged[: int(fcfg.get("max_violations", 5))]
    violations: list[Violation] = []
    for ev in sorted(merged, key=lambda e: e.t_peak):
        name = str(cfg_get(catalog, f"types.{ev.type_id}.name", None) or FALLBACK_NAMES.get(ev.type_id, ev.type_id))
        frames: list[EvidenceFrame] = []
        for t in dict.fromkeys([ev.t_peak, ev.t_start, ev.t_end]):
            if len(frames) >= 3 or clip is None:
                break
            try:
                img = clip.frame_at(t)
            except Exception:  # noqa: BLE001
                img = None
            if img is None:
                continue
            if fcfg.get("blur_heads", True) and tracks:
                img = blur_regions(img, _person_boxes_at(tracks, t))
            b64 = encode_jpeg_b64(img, int(fcfg.get("evidence_max_side", 640)),
                                  int(fcfg.get("evidence_jpeg_quality", 82)))
            if b64:
                frames.append(EvidenceFrame(round(float(t), 2), b64))
        violations.append(Violation(type=name, timecode_sec=round(float(ev.t_peak), 2),
                                    description=_description(ev.type_id, name, ev, catalog),
                                    evidence_frames=frames, type_id=ev.type_id,
                                    confidence=round(float(ev.confidence), 4), t_start=round(ev.t_start, 2),
                                    t_end=round(ev.t_end, 2), rules=_rules_for(ev.type_id, catalog)))
    verdict = "violation" if violations else "no_violation"
    return AnalysisResult(job_id=job_id, status="done", verdict=verdict, violations=violations,
                          timeline=list(risk), meta={"fusion": "fallback"})
