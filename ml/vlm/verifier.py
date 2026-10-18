"""verify(): проверка событий-кандидатов локальной VLM.

На событие — 4–6 кадров (t_start−1 … t_end+1, t_peak включён) с подсветкой боксов, max side 448.
Ответ — строгий JSON; факты фильтруются; уверенность смешивается с исходной (логит-пулинг).
verify НЕ удаляет события и НЕ меняет source: противоречие кадров -> extra.vlm.agreement="contradicts",
extra.vlm.contradiction=true, сниженная confidence — решение за fusion. Порядок событий сохраняется.
"""
from __future__ import annotations

import copy
import logging
import math
from typing import Any, Callable, Optional

from ._contract import Event
from .backends import VLMBackend
from .budget import Budget, VLMLimits
from .catalog import catalog_types, load_catalog
from .config import load_config, preset_cfg
from .facts_filter import filter_facts
from .frames import clip_duration, frames_at, prepare_images, verify_times
from .parsing import normalize_verify, regex_fallback_verify, verify_schema
from .prompting import render_verify, retry_suffix
from .runtime import call_json, ensure_loaded, fit_images, get_backend
from .qa import qa_verify_event
from .scoring import agreement as agreement_of, blend, clamp, fusion_opinion, update_confidence

log = logging.getLogger(__name__)

DumpFn = Callable[[dict[str, Any]], None]


def _copy_event(e: Event) -> Event:
    e2 = copy.copy(e)
    e2.facts = list(e.facts or [])
    e2.bboxes = dict(e.bboxes or {})
    e2.track_ids = list(e.track_ids or [])
    e2.extra = dict(e.extra or {})
    return e2


def _overlap(a: Event, b: Event) -> float:
    inter = max(0.0, min(a.t_end, b.t_end) - max(a.t_start, b.t_start))
    union = max(a.t_end, b.t_end) - min(a.t_start, b.t_start)
    if union <= 1e-6:
        return 1.0 if abs(a.t_peak - b.t_peak) < 0.5 else 0.0
    return inter / union


def _order(events: list[Event], how: str) -> list[int]:
    idx = list(range(len(events)))
    if how == "confidence_desc":
        return sorted(idx, key=lambda i: -float(events[i].confidence))
    if how == "time":
        return sorted(idx, key=lambda i: float(events[i].t_peak))
    # uncertain_first: самые спорные (около 0.5) — первыми, там VLM полезнее всего
    return sorted(idx, key=lambda i: (abs(float(events[i].confidence) - 0.5), -float(events[i].confidence)))


def _px_per_token(cfg: dict[str, Any]) -> int:
    return int(preset_cfg(cfg, "transformers").get("px_per_token", 28))


# VLM спросили, но надёжного ответа нет: для fusion это «не уверен» (score 0 — не перевешивает настоящие мнения),
# а не подтверждение/отказ; уверенность события не меняется, покрытие — неполное.
FAILED_ANSWER_STATUSES = ("parse_failed", "truncated", "timeout", "error")


def _mark(e: Event, status: str, **kw: Any) -> None:
    v = dict(e.extra.get("vlm") or {})
    v.update({"status": status, "agreement": v.get("agreement", "not_checked")})
    if status in FAILED_ANSWER_STATUSES:
        v.update({"agreement": "uncertain", "verdict": "unsure", "score": 0.0, "contradiction": False})
    v.update(kw)
    e.extra["vlm"] = v


def verify_events(clip: Any, events: list[Event], budget_s: float, *, cfg: Optional[dict[str, Any]] = None,
                  backend: Optional[VLMBackend] = None, limits: Optional[dict[str, Any]] = None,
                  catalog: Optional[dict[str, Any]] = None, clock: Any = None,
                  dump: Optional[DumpFn] = None, zones: Optional[list[Any]] = None,
                  mode: Optional[str] = None) -> list[Event]:
    """mode: json (по умолчанию, фаза 1) | qa (фаза 2: нейтральные вопросы + логиты; включается флагом)."""
    cfg = cfg or load_config()
    mode = str(mode or (cfg.get("verify") or {}).get("mode", "json")).lower()
    out = [_copy_event(e) for e in (events or [])]
    if not out:
        return out
    if backend is None:
        backend, reason = get_backend(cfg)
        if backend is None:
            for e in out:
                _mark(e, "unavailable", reason=reason[:160])
            return _with_coverage(out)
    lim = VLMLimits.from_cfg(cfg, "verify_qa" if mode == "qa" else "verify", limits)
    budget = Budget(float(budget_s), lim, **({"clock": clock} if clock else {}))
    rcfg = cfg.get("runtime") or {}
    ok, reason = ensure_loaded(backend, budget, float(rcfg.get("lazy_load_min_budget_s", 0.0)),
                               float(rcfg.get("load_retry_cooldown_s", 300.0)))
    if not ok:
        for e in out:
            _mark(e, "unavailable", reason=reason[:160])
        return _with_coverage(out)

    catalog = catalog or load_catalog()
    types = catalog_types(catalog)
    allowed = set(types)
    fcfg = cfg.get("frames") or {}
    vcfg = cfg.get("verify") or {}
    gcfg = cfg.get("generation") or {}
    ffcfg = cfg.get("facts_filter") or {}
    lang = str(cfg.get("lang", "ru"))
    version = str(cfg.get("prompt_version", "v1"))
    budget.px_per_token = _px_per_token(cfg)
    duration = clip_duration(clip)
    n_frames = max(1, min(int(fcfg.get("verify_n_frames", 5)), lim.max_frames_per_call))
    dedupe_iou = float(vcfg.get("dedupe_iou_s", 0.6))
    qa_spec: dict[str, Any] = {}
    qa_types: set[str] = set()
    qa_system = ""
    if mode == "qa":
        from .prompting import get_prompt
        from .qa import load_spec, supported_types

        qa_spec = load_spec((cfg.get("qa") or {}).get("spec_path"))
        qa_types = supported_types(qa_spec)
        qa_system = get_prompt("system_qa", version, lang).render()
    # (исходный тип, снимок времён, результат) — для дедупликации; тип берём ДО возможной переклассификации
    done: list[tuple[str, Event, dict[str, Any]]] = []

    for i in _order(out, str(vcfg.get("priority", "uncertain_first"))):
        ev = out[i]
        orig_type = ev.type_id
        shared = next((r for (t2, e2, r) in done if t2 == orig_type and _overlap(e2, ev) >= dedupe_iou), None)
        if shared is not None:
            if "_qa" in shared:
                _apply_qa(ev, shared["_qa"], types, cfg, shared_from=True)
            else:
                _apply(ev, shared, types, cfg, ffcfg, shared_from=True)
            continue
        if mode == "qa" and getattr(backend, "supports_option_probs", False) and orig_type in qa_types:
            qres = qa_verify_event(clip, ev, backend, budget, cfg, qa_system,
                                   zones=zones if zones is not None else (ev.extra or {}).get("zones"),
                                   spec=qa_spec, dump=dump)
            if qres is not None:
                if qres.get("status") != "ok":
                    _mark(ev, str(qres.get("status")), mode="qa")
                    continue
                qres["_backend"] = backend.info().get("model", backend.name)
                _apply_qa(ev, qres, types, cfg)
                done.append((orig_type, Event(orig_type, ev.t_start, ev.t_end, ev.t_peak, ev.confidence, ev.source),
                             {"_qa": qres}))
                continue
        times = verify_times(ev.t_start, ev.t_end, ev.t_peak, duration, n_frames,
                             float(fcfg.get("verify_pad_s", 1.0)), float(fcfg.get("verify_max_span_s", 12.0)))
        frames, ftimes = frames_at(clip, times)  # фактические PTS кадров, без дублей
        if not frames:
            _mark(ev, "no_frames")
            continue
        images = prepare_images(frames, ftimes, ev.bboxes, ev.t_peak, fcfg)
        # число кадров уже <= max_frames_per_call; здесь только ужимаем под лимит визуальных токенов
        images = fit_images(images, len(images), lim.max_visual_tokens_per_call, budget.px_per_token)
        has_boxes = bool(ev.bboxes) and bool(fcfg.get("draw_boxes", True))
        system, user = render_verify(orig_type, list(ev.facts or []), ftimes, catalog, lang, version, has_boxes)
        n_ref = len(ftimes)
        outcome = call_json(
            backend, images, system, user, budget,
            normalize=lambda d, n=n_ref: normalize_verify(d, allowed, n),
            max_new_tokens=int(gcfg.get("max_new_tokens_verify", 220)),
            schema=verify_schema(sorted(allowed), n_ref),
            retry_suffix=retry_suffix(lang, version),
            temperature=float(gcfg.get("temperature", 0.0)),
            fallback=lambda txt: regex_fallback_verify(txt, allowed),
        )
        if dump is not None:
            dump({"kind": "verify", "event_index": i, "type_id": orig_type, "times": ftimes, "images": images,
                  "system": system, "prompt": user, "raw": outcome.raw, "status": outcome.status})
        if not outcome.ok:
            _mark(ev, outcome.status, attempts=outcome.attempts, latency_s=round(outcome.latency_s, 3),
                  error=outcome.error[:120], finish_reason=outcome.finish_reason, complete=not outcome.truncated)
            continue
        res = dict(outcome.data)
        res["_times"] = ftimes
        res["_attempts"] = outcome.attempts
        res["_latency_s"] = round(outcome.latency_s, 3)
        res["_repaired"] = outcome.repaired
        res["_finish"] = outcome.finish_reason
        res["_truncated"] = outcome.truncated
        res["_backend"] = backend.info().get("model", backend.name)
        snap = Event(orig_type, ev.t_start, ev.t_end, ev.t_peak, ev.confidence, ev.source)
        _apply(ev, res, types, cfg, ffcfg)
        done.append((orig_type, snap, res))

    drop_below = vcfg.get("drop_below")
    if drop_below is not None:
        out = [e for e in out if float(e.confidence) >= float(drop_below)]
    summary = budget.summary()
    for e in out:
        e.extra.setdefault("vlm", {"status": "skipped_budget", "agreement": "not_checked"})
        e.extra["vlm"].setdefault("budget", summary)
    return _with_coverage(out)


def _with_coverage(out: list[Event]) -> list[Event]:
    cov = coverage(out)
    for e in out:
        e.extra.setdefault("vlm", {"status": "not_checked", "agreement": "not_checked"})["coverage"] = cov
    from .runtime import record_stage

    record_stage("verify", **cov)
    return out


def coverage(events: list[Event]) -> dict[str, Any]:
    """Полнота проверки для meta.coverage: непроверенное событие — «неизвестно», а не «нет нарушения»."""
    st = [str((e.extra.get("vlm") or {}).get("status", "not_checked")) for e in events]
    checked = sum(s == "ok" for s in st)
    return {"checked": checked, "total": len(st), "complete": checked == len(st),
            "statuses": {s: st.count(s) for s in sorted(set(st))}}


def _apply(ev: Event, res: dict[str, Any], types: dict[str, dict[str, Any]], cfg: dict[str, Any],
           ffcfg: dict[str, Any], shared_from: bool = False) -> None:
    vcfg = cfg.get("verify") or {}
    cand = ev.type_id
    vlm_type = res.get("type_id")
    visible = bool(res.get("visible"))
    conf = float(res.get("confidence", 0.5))
    p0 = float(ev.confidence)
    p_new, trace = update_confidence(p0, visible, conf, cand, vlm_type, cfg)
    agreement = trace["agreement"]
    degraded = "regex_fallback" in (res.get("flags") or [])
    if degraded:  # поля вытащены регэкспом из невалидного текста — ни подтверждать, ни отклонять по ним нельзя
        p_new = trace["p_prior"]
        trace = {**trace, "p_post": trace["p_prior"], "agreement": "uncertain"}
        agreement = "uncertain"

    # переклассификация — только для источников, чей тип слабый (temporal/vlm), и при уверенной VLM
    retyped = False
    if (not degraded and agreement == "type_mismatch" and visible and vlm_type in types
            and ev.source in set(vcfg.get("retype_sources") or [])
            and conf >= float(vcfg.get("retype_min_conf", 0.7))):
        ev.extra["vlm_original_type_id"] = cand
        ev.type_id = vlm_type
        p_new, trace = update_confidence(p0, True, conf, vlm_type, vlm_type, cfg)
        agreement = trace["agreement"]
        retyped = True

    fr = filter_facts(res.get("observed_facts") or [], max_len=int(ffcfg.get("max_len", 220)),
                      max_items=int(vcfg.get("max_facts", 4)), drop_hedges=bool(ffcfg.get("drop_hedges", True)),
                      require_anchor=bool(ffcfg.get("require_anchor", True)))
    kept = list(fr.kept)
    if agreement == "supports":
        if not kept:  # факты модели отфильтрованы — подставляем наблюдаемый признак из каталога
            obs = (types.get(ev.type_id) or {}).get("observable") or []
            if obs:
                kept = [f"Локальная VLM-модель отметила видимый признак: {str(obs[0]).rstrip('.')}"]
        for f in kept:
            if f not in ev.facts:
                ev.facts.append(f)

    times: list[float] = list(res.get("_times") or [])
    best = res.get("best_frame")
    t_best = times[best - 1] if best and 1 <= best <= len(times) else None
    vis_t = [times[k - 1] for k in (res.get("visible_frames") or []) if 1 <= k <= len(times)]
    keep_raw = bool((cfg.get("privacy") or {}).get("keep_raw_in_extra", False))
    ev.confidence = float(round(p_new, 4))
    vlm_extra = {
        "status": "ok",
        "agreement": agreement,
        "contradiction": agreement == "contradicts",
        "visible": visible,
        "vlm_confidence": round(conf, 3),
        "confidence_missing": bool(res.get("confidence_missing")),
        "type_id": vlm_type,
        "suggested_type_id": vlm_type if agreement == "type_mismatch" else None,
        "retyped": retyped,
        "facts": kept if agreement == "supports" else list(fr.kept),  # «нет признака»-факты — только здесь
        "dropped_facts": [c for (_f, c) in fr.dropped],              # только категории, без текста
        "uncertain_because": str(res.get("uncertain_because") or "")[:200],
        "t_best": t_best,
        "t_visible": vis_t,
        "frame_times": times,
        "flags": list(res.get("flags") or []),
        "attempts": res.get("_attempts"),
        "latency_s": res.get("_latency_s"),
        "repaired_json": bool(res.get("_repaired")),
        "finish_reason": res.get("_finish", ""),
        "complete": not bool(res.get("_truncated")),  # обрезанный ответ (length/time) — complete=false
        "model": res.get("_backend"),
        "shared": shared_from,
        **trace,
        **fusion_opinion(agreement, trace.get("p_vlm")),  # verdict/score — то, что читает ml.fusion
    }
    if keep_raw:
        vlm_extra["raw"] = res
    ev.extra["vlm"] = vlm_extra
    if vlm_extra.get("verdict") == "reject" and vcfg.get("emit_vlm_verdict_rejected", False):
        ev.extra["vlm_verdict"] = "rejected"  # встроенное слияние A03 такие события отбрасывает — по умолчанию выкл.


def _apply_qa(ev: Event, res: dict[str, Any], types: dict[str, dict[str, Any]], cfg: dict[str, Any],
              shared_from: bool = False) -> None:
    """Обновление события по ответам qa: логит-пулинг с весом, пропорциональным доказательности ответов."""
    vcfg = cfg.get("verify") or {}
    qcfg = cfg.get("qa") or {}
    dec = res.get("decision") or {}
    p0 = float(ev.confidence)
    p0 = 0.5 if not math.isfinite(p0) else clamp(p0, 0.0, 1.0)
    pv = float(dec.get("p_vlm", 0.5))
    evidence = float(dec.get("evidence", 0.0))
    agr = dec.get("agreement") or agreement_of(pv, float(vcfg.get("agree_threshold", 0.6)))
    lo, hi = (float(x) for x in vcfg.get("p_vlm_clip", (0.05, 0.95)))
    w_eff = float(qcfg.get("vlm_weight", vcfg.get("vlm_weight", 0.6))) * min(1.0, evidence)
    p_new = blend(p0, clamp(pv, lo, hi), w_eff) if agr != "uncertain" else p0
    p_new = min(p_new, p0 + float(vcfg.get("max_raise", 0.25)))
    pro = list(res.get("facts_pro") or [])
    con = list(res.get("facts_con") or [])
    if agr == "supports":
        for f in pro:
            if f not in ev.facts:
                ev.facts.append(f)
    times = res.get("times") or {}
    t_tr = res.get("t_transition")
    if t_tr is not None and qcfg.get("refine_t_peak", False):
        ev.extra["vlm_original_t_peak"] = ev.t_peak
        ev.t_peak = float(t_tr)
    ev.confidence = float(round(p_new, 4))
    answers = {f"{q}@{at}": {k: round(v, 3) for k, v in d.items()} for (q, at), d in (res.get("answers") or {}).items()}
    ev.extra["vlm"] = {
        "status": "ok", "mode": "qa", "agreement": agr, "contradiction": agr == "contradicts",
        "p_prior": round(p0, 4), "p_vlm": round(pv, 4), "p_post": round(p_new, 4),
        "p_support": dec.get("p_support"), "p_contra": dec.get("p_contra"), "evidence": round(evidence, 4),
        **fusion_opinion(agr, pv),
        "facts": pro if agr == "supports" else con, "facts_pro": pro, "facts_con": con,
        "answers": answers, "t_transition": t_tr, "frame_times": times,
        "asked": res.get("asked"), "skipped": res.get("skipped"), "low_mass": res.get("low_mass"),
        "crops_used": res.get("crops_used"), "latency_s": res.get("latency_s"),
        "spec_version": res.get("spec_version"), "model": res.get("_backend"), "shared": shared_from,
    }
