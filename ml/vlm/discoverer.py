"""discover(): поиск нарушений, которых не нашли правила (включая V99 — новые типы). ВЫКЛЮЧЕН по умолчанию
(discover.enabled=false) до локального бенчмарка скорости и ложных срабатываний.

Окна ~window_s; порядок обхода — по motion energy (самые «активные» первыми) + минимальное равномерное
покрытие; бюджет (время/вызовы/токены) обрывает хвост. Находка становится событием только при
конкретном видимом факте (после фильтра), с низким приором и потолком уверенности; при reverify —
подтверждается вторым вызовом verify, противоречие -> находка отбрасывается (это гипотеза самой VLM).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from ._contract import Event
from .backends import VLMBackend
from .budget import Budget, VLMLimits
from .catalog import catalog_types, load_catalog
from .config import load_config, preset_cfg
from .facts_filter import filter_facts
from .frames import clip_duration, frames_at, motion_energy, prepare_images
from .parsing import discover_schema, normalize_discover
from .prompting import render_discover, retry_suffix
from .runtime import call_json, ensure_loaded, fit_images, get_backend, record_stage
from .scoring import blend, p_vlm

log = logging.getLogger(__name__)

DumpFn = Callable[[dict[str, Any]], None]


def plan_windows(duration: float, window_s: float, motion: Optional[list[tuple[float, float]]] = None,
                 min_coverage: int = 3) -> list[tuple[float, float, float]]:
    """-> [(t0, t1, score)] в порядке обхода: равномерное покрытие (min_coverage окон) + остальные по motion."""
    if duration <= 0 or window_s <= 0:
        return []
    n = max(1, int(round(duration / window_s)))
    w = duration / n
    wins = []
    for k in range(n):
        t0, t1 = k * w, min(duration, (k + 1) * w)
        vals = [e for (t, e) in (motion or []) if t0 <= t < t1 or (k == n - 1 and t == t1)]
        wins.append((round(t0, 3), round(t1, 3), float(sum(vals) / len(vals)) if vals else 0.0))
    cover_idx: list[int] = []
    if min_coverage > 0:
        m = min(min_coverage, n)
        cover_idx = sorted({int(round((j + 0.5) * n / m - 0.5)) for j in range(m)})
    rest = sorted((i for i in range(n) if i not in cover_idx), key=lambda i: -wins[i][2])
    # покрытие чередуем с самыми активными окнами, чтобы бюджет не ушёл целиком на одно
    order: list[int] = []
    cover = sorted(cover_idx, key=lambda i: -wins[i][2])
    while cover or rest:
        if rest:
            order.append(rest.pop(0))
        if cover:
            order.append(cover.pop(0))
    return [wins[i] for i in order]


def _merge(events: list[Event], gap: float) -> list[Event]:
    events = sorted(events, key=lambda e: (e.type_id, e.t_start))
    merged: list[Event] = []
    for e in events:
        last = merged[-1] if merged else None
        if last is not None and last.type_id == e.type_id and e.t_start - last.t_end <= gap:
            last.t_end = max(last.t_end, e.t_end)
            if e.confidence > last.confidence:
                last.t_peak, last.confidence = e.t_peak, e.confidence
            for f in e.facts:
                if f not in last.facts:
                    last.facts.append(f)
            last.extra["vlm_discover"]["windows"] = last.extra["vlm_discover"].get("windows", 1) + 1
        else:
            merged.append(e)
    return sorted(merged, key=lambda e: e.t_peak)


def discover_events(clip: Any, budget_s: float, *, cfg: Optional[dict[str, Any]] = None,
                    backend: Optional[VLMBackend] = None, limits: Optional[dict[str, Any]] = None,
                    catalog: Optional[dict[str, Any]] = None, clock: Any = None, force: bool = False,
                    dump: Optional[DumpFn] = None, t_range: Optional[tuple[float, float]] = None,
                    motion: Optional[list[tuple[float, float]]] = None) -> list[Event]:
    """motion: готовая кривая [(t, energy 0..1)] (например, сигнал motion_energy из A08) — иначе считаем сами."""
    cfg = cfg or load_config()
    dcfg = cfg.get("discover") or {}
    if not (dcfg.get("enabled", False) or force):
        record_stage("discover", status="disabled", complete=True, events=0)
        return []
    if backend is None:
        backend, _reason = get_backend(cfg)
        if backend is None:  # «ничего не нашли» и «не смотрели» различаются: complete=false
            record_stage("discover", status="unavailable", complete=False, events=0, reason=_reason[:160])
            return []
    lim = VLMLimits.from_cfg(cfg, "discover", limits)
    budget = Budget(float(budget_s), lim, **({"clock": clock} if clock else {}))
    rcfg = cfg.get("runtime") or {}
    ok, _reason = ensure_loaded(backend, budget, float(rcfg.get("lazy_load_min_budget_s", 0.0)),
                                float(rcfg.get("load_retry_cooldown_s", 300.0)))
    if not ok:
        record_stage("discover", status="unavailable", complete=False, events=0, reason=_reason[:160])
        return []
    catalog = catalog or load_catalog()
    types = catalog_types(catalog)
    checklist = {k: v for k, v in (dcfg.get("checklist") or {}).items() if v in types}
    items = list(checklist)
    if not items:
        return []
    fcfg = cfg.get("frames") or {}
    gcfg = cfg.get("generation") or {}
    ffcfg = cfg.get("facts_filter") or {}
    lang = str(cfg.get("lang", "ru"))
    version = str(cfg.get("prompt_version", "v1"))
    budget.px_per_token = int(preset_cfg(cfg, "transformers").get("px_per_token", 28))

    duration = clip_duration(clip)
    lo, hi = (0.0, duration) if t_range is None else (max(0.0, t_range[0]), min(duration, t_range[1]))
    if motion is not None:
        motion = [(float(t), float(e)) for (t, e) in motion if lo <= float(t) <= hi]
    elif dcfg.get("motion_weighting", True):
        try:
            motion = [(t, e) for (t, e) in motion_energy(clip, int(dcfg.get("motion_downscale", 64)))
                      if lo <= t <= hi]
        except Exception as e:  # noqa: BLE001
            log.debug("motion energy failed: %s", type(e).__name__)
    wins = [(t0 + lo, t1 + lo, s) for (t0, t1, s) in
            plan_windows(hi - lo, float(dcfg.get("window_s", 3.0)),
                         [(t - lo, e) for (t, e) in motion] if motion else None,
                         int(dcfg.get("min_coverage_windows", 3)))]
    n_fr = max(1, min(int(dcfg.get("frames_per_window", 4)), lim.max_frames_per_call))
    prior = float(dcfg.get("prior", 0.25))
    w = float(dcfg.get("vlm_weight", 0.6))
    max_conf = float(dcfg.get("max_conf", 0.6))
    min_vlm_conf = float(dcfg.get("min_vlm_conf", 0.55))
    found: list[Event] = []
    windows_done = 0

    do_reverify = bool(dcfg.get("reverify", True))
    for (t0, t1, score) in wins:
        # при наличии находок оставляем вызов(ы) на повторную проверку (иначе окна съедят весь бюджет)
        reserve = min(len(found), int(dcfg.get("reverify_reserve_calls", 1))) if do_reverify else 0
        if reserve and (lim.max_calls - budget.calls <= reserve
                        or budget.remaining() < budget.est_call_s() * lim.safety * (1 + reserve)):
            budget.stop_reason = budget.stop_reason or "reserved_for_reverify"
            break
        if not budget.can_call(n_fr):
            break
        times = [round(t0 + (t1 - t0) * (k + 0.5) / n_fr, 3) for k in range(n_fr)]
        frames, ftimes = frames_at(clip, times)  # фактические PTS кадров, без дублей
        if not frames:
            continue
        images = prepare_images(frames, ftimes, None, None, fcfg)
        images = fit_images(images, len(images), lim.max_visual_tokens_per_call, budget.px_per_token)
        system, user = render_discover(checklist, ftimes, catalog, lang, version)
        n_ref = len(ftimes)
        outcome = call_json(
            backend, images, system, user, budget,
            normalize=lambda d, n=n_ref: normalize_discover(d, items, n),
            max_new_tokens=int(gcfg.get("max_new_tokens_discover", 300)),
            schema=discover_schema(items, n_ref), retry_suffix=retry_suffix(lang, version),
            temperature=float(gcfg.get("temperature", 0.0)),
        )
        windows_done += int(outcome.attempts > 0)
        if dump is not None:
            dump({"kind": "discover", "window": [t0, t1], "times": ftimes, "images": images, "system": system,
                  "prompt": user, "raw": outcome.raw, "status": outcome.status})
        if not outcome.ok:
            continue
        for fnd in outcome.data.get("findings") or []:
            conf = float(fnd.get("confidence", 0.5))
            if fnd.get("confidence_missing") or conf < min_vlm_conf:
                continue
            fr = filter_facts(fnd.get("observed_facts") or [], max_len=int(ffcfg.get("max_len", 220)),
                              max_items=2, drop_hedges=bool(ffcfg.get("drop_hedges", True)),
                              require_anchor=bool(ffcfg.get("require_anchor", True)))
            if not fr.kept:
                continue  # нет конкретного видимого факта — не событие
            tid = checklist[fnd["item"]]
            k = fnd.get("frame")
            t_peak = ftimes[k - 1] if k and 1 <= k <= len(ftimes) else ftimes[len(ftimes) // 2]
            p = min(max_conf, blend(prior, p_vlm(True, conf), w))
            found.append(Event(
                type_id=tid, t_start=float(t0), t_end=float(t1), t_peak=float(t_peak), confidence=round(p, 4),
                source="vlm", facts=list(fr.kept),
                extra={"vlm": {"status": "ok", "stage": "discover", "agreement": "not_checked"},
                       "vlm_discover": {"item": fnd["item"], "vlm_confidence": conf, "prior": prior,
                                        "motion": round(score, 4), "frame_times": ftimes,
                                        "flags": list(outcome.data.get("flags") or []),
                                        "dropped_facts": [c for (_f, c) in fr.dropped],
                                        "model": backend.info().get("model", backend.name)}},
            ))

    events = _merge(found, float(dcfg.get("merge_gap_s", 1.5)))
    rv: Optional[dict[str, Any]] = None
    if events and do_reverify and budget.remaining() > 0:
        from .verifier import verify_events

        # повторная проверка тратит ОСТАТОК того же бюджета discover (общие лимиты не превышаются)
        left = {
            "max_calls": max(0, lim.max_calls - budget.calls),
            "max_visual_tokens_total": max(0, lim.max_visual_tokens_total - budget.visual_tokens),
            "max_output_tokens_total": max(0, lim.max_output_tokens_total - budget.output_tokens),
            "max_retries_total": max(0, lim.max_retries_total - budget.retries),
            "max_frames_per_call": lim.max_frames_per_call,
            "max_visual_tokens_per_call": lim.max_visual_tokens_per_call,
            "max_output_tokens_per_call": lim.max_output_tokens_per_call,
        }
        checked = verify_events(clip, events, budget.remaining(), cfg=cfg, backend=backend, limits=left,
                                catalog=catalog, clock=clock, dump=dump)
        rv = next((e.extra.get("vlm", {}).get("budget") for e in checked if e.extra.get("vlm", {}).get("budget")), None)
        kept = []
        for e in checked:
            agr = (e.extra.get("vlm") or {}).get("agreement")
            if agr == "contradicts":
                continue  # гипотеза самой VLM не подтвердилась
            e.confidence = min(max_conf, float(e.confidence))
            e.extra["vlm"]["stage"] = "discover+verify" if agr not in (None, "not_checked") else "discover"
            kept.append(e)
        events = kept
    events = sorted(events, key=lambda e: -e.confidence)[: int(dcfg.get("max_events", 4))]
    summary = budget.summary()
    if rv:  # учитываем вызовы повторной проверки в общем отчёте
        for k in ("calls", "retries", "visual_tokens", "output_tokens"):
            summary[k] = int(summary.get(k) or 0) + int(rv.get(k) or 0)
        summary["elapsed_s"] = round(budget.elapsed(), 3)
        summary["reverify_calls"] = int(rv.get("calls") or 0)
    summary["windows_planned"] = len(wins)
    summary["windows_done"] = windows_done
    for e in events:
        e.extra["vlm"]["budget"] = summary
    record_stage("discover", status="ok", events=len(events), windows_planned=len(wins), windows_done=windows_done,
                 complete=windows_done >= len(wins), calls=summary.get("calls"), stop_reason=summary.get("stop_reason"))
    return sorted(events, key=lambda e: e.t_peak)
