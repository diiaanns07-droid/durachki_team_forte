"""Точка входа стадии seqrules: (clip, dets, tracks, zones, poses, catalog) → (events, observations)."""
from __future__ import annotations

import time
from typing import Any, Mapping, Optional, Sequence

from .checks import Checker
from .config import config_hash, load_config
from .observe import Frames, PoseIndex, frames_from, get, person_tracks, zone_polygon
from .rules import Ctx, merge_events, obs_hand_cash, rule_entry, rule_phone, rule_safe, rule_storage

RULES = (("phone", rule_phone), ("safe", rule_safe), ("storage", rule_storage), ("entry", rule_entry),
         ("hand_cash", obs_hand_cash))


def _catalog_types(catalog: Any) -> Optional[set[str]]:
    if isinstance(catalog, Mapping):
        types = catalog.get("types")
        if isinstance(types, Mapping) and types:
            return set(types)
    return None


def _thin(frames: list[dict], keep: int) -> list[dict]:
    if keep <= 0 or len(frames) <= keep:
        return frames
    step = len(frames) / float(keep)
    idx = sorted({int(i * step) for i in range(keep)} | {len(frames) - 1})
    return [frames[i] for i in idx]


def _budget_frames(episodes: list[dict], cap: int, min_frames: int) -> None:
    """Кадры эпизодов для UI: события — в первую очередь; общий потолок cap, лишнее прореживается."""
    for e in episodes:
        if len(e.get("frames", [])) < min_frames:
            e["frames_note"] = "кадры не сохранены (эпизод короче порога)"
            e["frames"] = []
    total = sum(len(e.get("frames", [])) for e in episodes)
    if total <= cap:
        return
    ev = [e for e in episodes if e.get("status") == "event"]
    rest = [e for e in episodes if e.get("status") != "event"]
    ev_total = sum(len(e["frames"]) for e in ev)
    ev_cap = min(ev_total, int(cap * 0.8))
    for group, budget in ((ev, ev_cap), (rest, cap - ev_cap)):
        n = sum(len(e["frames"]) for e in group)
        for e in group:
            if n > budget and e["frames"]:
                keep = max(2, int(len(e["frames"]) * budget / max(n, 1)))
                e["frames_note"] = f"прорежено: {keep} из {len(e['frames'])}"
                e["frames"] = _thin(e["frames"], keep)


def run(clip: Any, dets: Sequence[Any], tracks: Sequence[Any], zones: Sequence[Any], poses: Sequence[Any],
        catalog: Any = None, *, config: Optional[Mapping] = None, checker: Any = None) -> tuple[list, list[dict]]:
    """Правила по последовательностям. Возвращает события контракта (source="rules_seq") и наблюдения для UI:
    первый элемент — {"kind": "meta", ...} (покрытие, статусы правил, зоны, проверки), далее эпизоды."""
    t_begin = time.perf_counter()
    cfg = load_config(config)
    fr: Frames = frames_from(clip, list(dets or []))
    persons = person_tracks(tracks)
    pidx = PoseIndex(poses)
    work_name = cfg["person"].get("work_zone", "cashier_area")
    ck = checker if checker is not None else Checker(clip, cfg.get("checks", {}) or {})
    ctx = Ctx(cfg=cfg, fr=fr, persons=persons, poses=pidx, zones=list(zones or []),
              work_zone=zone_polygon(zones, work_name), checker=ck, dets=list(dets or []))
    warnings: list[str] = []
    known = _catalog_types(catalog)
    info: dict[str, Any] = {}
    episodes: list[dict] = []
    events: list = []
    for name, fn in RULES:
        rc = cfg.get(name, {}) or {}
        if not rc.get("enabled", False):
            info[name] = {"status": "disabled"}
            continue
        if known is not None and rc.get("type_id") and rc["type_id"] not in known:
            info[name] = {"status": "unavailable", "reason": f"тип {rc['type_id']} отсутствует в каталоге норм"}
            warnings.append(f"seqrules/{name}: тип {rc['type_id']} нет в каталоге — правило не применено")
            continue
        t0 = time.perf_counter()
        try:
            eps, evs, inf = fn(ctx)
        except Exception as exc:  # noqa: BLE001 — сбой одного правила не роняет остальные, но виден
            eps, evs, inf = [], [], {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            warnings.append(f"seqrules/{name}: ошибка {type(exc).__name__}: {exc}")
        inf["elapsed_s"] = round(time.perf_counter() - t0, 3)
        info[name] = inf
        episodes.extend(eps)
        events.extend(evs)
        if inf.get("status") not in ("ok", "disabled", "no_persons"):
            warnings.append(f"seqrules/{name}: {inf.get('status')} — {inf.get('reason') or inf.get('error') or ''}".rstrip(" —"))
    events = merge_events(events, float(cfg.get("merge_gap_s", 2.0)))
    _budget_frames(episodes, int(cfg.get("max_observation_frames", 2500)), int(cfg.get("episode_min_frames_saved", 2)))
    if work_name and ctx.work_zone is None:
        warnings.append(f"seqrules: зона {work_name} не построена — правила Б и В неприменимы")
    if fr.processed.size and not fr.processed.any():
        warnings.append("seqrules: ни один кадр не обработан детектором — вывод невозможен")
    meta = {
        "kind": "meta", "engine": "seqrules", "version": _version(), "config_hash": config_hash(cfg),
        "frames": {"n": int(len(fr.times)), "processed": int(fr.processed.sum()), "source": fr.source,
                   "t0": float(fr.times[0]) if len(fr.times) else None,
                   "t1": float(fr.times[-1]) if len(fr.times) else None},
        "frame_size": [fr.width, fr.height],
        "persons": {str(tid): {"role": p.role, "role_conf": round(p.role_conf, 3)} for tid, p in persons.items()},
        "pose": {"frames": pidx.n, "tracks": pidx.tracks},
        "zones": [{"name": str(get(z, "name")), "polygon": [[round(float(x), 5), round(float(y), 5)] for x, y in get(z, "polygon")],
                   "conf": float(get(z, "conf", 1.0) or 0.0)} for z in (zones or [])],
        "work_zone": work_name if ctx.work_zone is not None else None,
        "rules": info, "checks": ck.summary() if hasattr(ck, "summary") else {},
        "n_events": len(events), "warnings": warnings,
        "elapsed_s": round(time.perf_counter() - t_begin, 3),
        "note": "наблюдения — что видели модели; события — кандидаты для проверяющего, не юридический вывод",
    }
    return events, [meta, *episodes]


def _version() -> str:
    from . import __version__

    return __version__
