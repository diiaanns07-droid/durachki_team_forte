"""Быстрый цикл тюнинга правил по кэшу состояний (без CV) + сравнение с разметкой.

    python -m ml.rules.replay --cache cache/ --labels data/labels.csv [--names out/summary.json | --videos data/raw]
    python -m ml.rules.replay ... --hypotheses all --json ablation.json      # замер гипотез фазы 2 для L2
    python -m ml.rules.replay ... --profile phase2 --trace                    # профиль + trace по событиям
    python -m ml.rules.replay ... --propose                                   # ≤ 2 правки порогов по данным
    python -m ml.rules.replay --rules-table

Кэш: раскладка A03 ``cache/<sha256>/states.jsonl`` (+ ``tracks.json``, ``zones.json``, ``manifest.json``);
имя ролика по sha — из ``--names`` (``summary.json`` батча A03 или CSV ``sha256,file``) или ``--videos``
(локальный подсчёт sha256). Также ``<cache>/<video>/states.json`` и ``<video>.states.json[l]``.

Вердикт по видео (``--verdict``): ``detection`` (по умолчанию) — есть событие с
``extra.detection == violation_candidate`` (поведение по видео; это меряет жюри в «Детекции»);
``legal`` — только ``candidate_violation`` (кандидат + подтверждённая норма A13). Правовая опора
отчитывается отдельно. Таймкод — ``t_peak`` ближайшего к разметке события (по альтернативам ``t_all``).
Совместная метрика повторяет прокси жюри A16 (eval/config.yaml): детекция с ценой FN=2, FP=1 и
ступенчатая шкала времени 1/2/3/5 с → ``joint = (25·det + 15·time) / 40``.
Пороги, найденные здесь, — только по train/calibration; 42 размеченных ролика уже использованы в
разведке и независимым тестом не являются.
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from .cache_io import hash_videos, load_cache_dir, load_labels, load_name_map, normalize_video_id
from .catalog import FALLBACK_CATALOG, resolve_catalog
from .config import build_config, read_raw_config
from .engine import CANDIDATE, INSUFFICIENT, REVIEW, assessment_of, evaluate

MAX_SWEEP = 500
FN_COST, FP_COST = 2.0, 1.0                                  # как eval/config.yaml (A16)
TIME_STEPS = ((1.0, 1.0), (2.0, 0.75), (3.0, 0.5), (5.0, 0.25))
W_DET, W_TIME = 25.0, 15.0


def _load_catalog(path: Optional[str], assume_verified: bool) -> tuple[dict, str]:
    if path:
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            cat = yaml.safe_load(fh) or {}
        src = path
    else:
        cat, src = resolve_catalog(None)
    cat = copy.deepcopy(dict(cat))
    if assume_verified:
        types = dict(cat.get("types") or {})
        for tid in set(types) | set(FALLBACK_CATALOG["types"]):
            entry = dict(types.get(tid) or FALLBACK_CATALOG["types"].get(tid, {"name": tid}))
            entry["rules"] = [{"document": "CALIBRATION-ONLY", "clause": None, "url": "calibration://assume-verified",
                               "quote": "", "verified": True}]
            entry["legal_status"] = "confirmed"
            types[tid] = entry
        cat["types"] = types
        src += " + assume-verified (калибровка)"
    return cat, src


def _fmt(v: Any, nd: int = 1) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "да" if v else "нет"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _table(rows: list[list[Any]], header: list[str]) -> str:
    cells = [[_fmt(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h) for i, h in enumerate(header)]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    out = [line, "  ".join("-" * w for w in widths)]
    out += ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in cells]
    return "\n".join(out)


def _time_score(abs_dt: Optional[float]) -> float:
    if abs_dt is None:
        return 0.0
    for thr, val in TIME_STEPS:
        if abs_dt <= thr + 1e-9:
            return val
    return 0.0


def _is_positive(ev: Any, verdict: str, count_review: bool) -> bool:
    if verdict == "legal":
        ok = assessment_of(ev) == CANDIDATE
    else:
        ok = ev.extra.get("detection") == "violation_candidate"
    return ok or (count_review and assessment_of(ev) == REVIEW)


def _clause_brief(ev: Any) -> str:
    parts = []
    for c in ev.extra.get("trace", {}).get("clauses", []):
        mark = {"true": "+", "false": "−", "unknown": "?"}.get(c["at_peak"], "?")
        if c["kind"] == "exclusion":
            mark = {"true": "✗", "false": "", "unknown": "?"}.get(c["at_peak"], "?")
            if not mark:
                continue
        parts.append(f"{mark}{c['id']}")
    return " ".join(parts)


def apply_name_map(cache: dict[str, dict[str, Any]], name_map: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Переименовать записи кэша cache/<sha>/ в имена роликов (для сопоставления с разметкой)."""
    if not name_map:
        return cache
    norm_map = {normalize_video_id(k): v for k, v in name_map.items()}
    out: dict[str, dict[str, Any]] = {}
    for key, item in cache.items():
        name = norm_map.get(key) or item.get("video_info", {}).get("file") or item.get("video_info", {}).get("name")
        if name:
            item = dict(item, name=str(name), sha=key)
            key = normalize_video_id(str(name))
        out.setdefault(key, item)
    return out


def _gt_groups(labels: dict[str, list[dict[str, Any]]], dedupe: bool) -> dict[str, str]:
    """id → id-представитель группы дубликатов (md5/group), чтобы считать ролик один раз."""
    rep: dict[str, str] = {}
    seen: dict[str, str] = {}
    for key in sorted(labels):
        g = labels[key][0]
        gkey = (g.get("group") or g.get("md5")) if dedupe else None
        if gkey and gkey in seen:
            rep[key] = seen[gkey]
        else:
            rep[key] = key
            if gkey:
                seen[gkey] = key
    return rep


def run_replay(cache: dict[str, dict[str, Any]], labels: dict[str, list[dict[str, Any]]], cfg: Any, catalog: dict,
               tol: float = 2.0, min_conf: float = 0.0, count_review: bool = False, verdict: str = "detection",
               dedupe: bool = True) -> dict[str, Any]:
    videos = []
    rule_fires: dict[str, dict[str, int]] = {}
    group_rep = _gt_groups(labels, dedupe)
    timings: list[float] = []
    for key in sorted(set(cache) | set(labels)):
        item = cache.get(key)
        gt_list = labels.get(key) or [{}]
        gt = gt_list[0]
        alts = sorted({t for g in gt_list for t in ([g.get("t")] + list(g.get("t_all") or [])) if t is not None})
        row: dict[str, Any] = {"id": key, "name": (item or gt).get("name", key), "gt": gt.get("violation"),
                               "gt_t": gt.get("t"), "gt_all_t": alts, "conflict": bool(gt.get("conflict")),
                               "group_rep": group_rep.get(key, key), "cached": item is not None}
        if item is None:
            row["note"] = "нет кэша"
            videos.append(row)
            continue
        t0 = time.perf_counter()
        rep = evaluate(item["states"], item.get("tracks") or [], item.get("zones") or [], catalog, cfg)
        timings.append((time.perf_counter() - t0) * 1000.0)
        evs = [e for e in rep.events if e.confidence >= min_conf]
        pool = [e for e in evs if _is_positive(e, verdict, count_review)]
        row.update(pred=bool(pool), n_events=len(evs),
                   n_cand=sum(1 for e in evs if e.extra.get("detection") == "violation_candidate"),
                   n_review=sum(1 for e in evs if e.extra.get("detection") == "observation"),
                   n_insuff=sum(1 for e in evs if assessment_of(e) == INSUFFICIENT),
                   n_legal=sum(1 for e in evs if assessment_of(e) == CANDIDATE),
                   coverage=rep.coverage().get("rules", {}), warnings=rep.warnings[:5],
                   events=[{"type_id": e.type_id, "t_start": e.t_start, "t_end": e.t_end, "t_peak": e.t_peak,
                            "confidence": e.confidence, "assessment": assessment_of(e),
                            "detection": e.extra.get("detection"), "legal_basis": e.extra.get("legal_basis"),
                            "rule_ids": e.extra.get("rule_ids", []), "clauses": _clause_brief(e),
                            "evidence": e.extra.get("trace", {}).get("evidence", {}),
                            "trace": e.extra.get("trace")} for e in evs])
        if pool:
            if alts:
                best = min(pool, key=lambda e: (min(abs(e.t_peak - a) for a in alts), -e.confidence))
            else:
                best = max(pool, key=lambda e: e.confidence)
            ev_tr = best.extra.get("trace", {}).get("evidence", {})
            row.update(type_id=best.type_id, t_peak=best.t_peak, t_start=best.t_start, conf=best.confidence,
                       rule=",".join(best.extra.get("rule_ids", [])), detection=best.extra.get("detection"),
                       legal_basis=best.extra.get("legal_basis"), clauses=_clause_brief(best),
                       evidence_kind=ev_tr.get("kind") or ev_tr.get("policy"))
            if alts:
                row["dt"] = round(min((best.t_peak - a for a in alts), key=abs), 2)
                row["dt_onset"] = round(min((best.t_start - a for a in alts), key=abs), 2)
                row["hit"] = abs(row["dt"]) <= tol
        bucket = "violation" if row["gt"] else ("clean" if row["gt"] is False else "unlabeled")
        for e in evs:
            # в источники FP/TP идут только события, которые считаются в вердикте; наблюдения и
            # insufficient_context — отдельными колонками (не «срабатывания на чистых»)
            col = bucket if _is_positive(e, verdict, count_review) else (
                "insufficient" if assessment_of(e) == INSUFFICIENT else "observation")
            for rid in e.extra.get("rule_ids", []):
                d = rule_fires.setdefault(rid, {"violation": 0, "clean": 0, "unlabeled": 0,
                                                "observation": 0, "insufficient": 0})
                d[col] += 1
        videos.append(row)
    return {"videos": videos, "summary": summarize(videos, tol, timings, count_review, verdict),
            "rule_fires": rule_fires}


def summarize(videos: list[dict[str, Any]], tol: float, timings: Sequence[float] = (), count_review: bool = False,
              verdict: str = "detection") -> dict[str, Any]:
    reps = {}
    for v in videos:  # группа дубликатов считается один раз (представитель с кэшем)
        if v.get("cached") and v.get("gt") is not None:
            reps.setdefault(v["group_rep"], v)
    labeled = list(reps.values())
    tp = sum(1 for v in labeled if v["gt"] and v["pred"])
    fn = sum(1 for v in labeled if v["gt"] and not v["pred"])
    fp = sum(1 for v in labeled if not v["gt"] and v["pred"])
    tn = sum(1 for v in labeled if not v["gt"] and not v["pred"])
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
    pos = [v for v in labeled if v["gt"]]
    pos_timed = [v for v in pos if not v.get("conflict") and v.get("gt_all_t")]
    tp_rows = [v for v in pos_timed if v["pred"] and v.get("dt") is not None]
    abs_dt = [abs(v["dt"]) for v in tp_rows]
    denom = FN_COST * len(pos) + FP_COST * (len(labeled) - len(pos))
    det = 1.0 - (FN_COST * fn + FP_COST * fp) / denom if denom else None
    time_sc = (sum(_time_score(abs(v["dt"]) if v["pred"] and v.get("dt") is not None else None) for v in pos_timed)
               / len(pos_timed)) if pos_timed else None
    joint = ((W_DET * det + W_TIME * (time_sc or 0.0)) / (W_DET + W_TIME)) if det is not None else None
    types: dict[str, int] = {}
    for v in labeled:
        if v["gt"] and v["pred"] and v.get("type_id"):
            types[v["type_id"]] = types.get(v["type_id"], 0) + 1
    return {
        "n_videos": len(videos), "n_cached": sum(1 for v in videos if v.get("cached")), "n_labeled": len(labeled),
        "n_conflict": sum(1 for v in labeled if v.get("conflict")),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1,
        "accuracy": (tp + tn) / len(labeled) if labeled else None,
        "det_score": det, "time_score": time_sc, "joint": joint,
        "time_hit_rate": {str(t): (sum(1 for d in abs_dt if d <= t) / len(abs_dt) if abs_dt else None)
                          for t in sorted({0.5, 1.0, 2.0, 3.0, 5.0, tol})},
        "mean_abs_dt": statistics.mean(abs_dt) if abs_dt else None,
        "median_abs_dt": statistics.median(abs_dt) if abs_dt else None,
        "median_gt_minus_t_peak": statistics.median([-v["dt"] for v in tp_rows]) if tp_rows else None,
        "types_on_tp": types,
        "latency_ms": {"mean": statistics.mean(timings) if timings else None, "max": max(timings) if timings else None},
        "vram_mb": 0,
        "count_review": count_review, "verdict": verdict,
    }


def rules_status_table(cfg: Any, width: int = 100) -> str:
    """Компактная таблица правил активного профиля: включено ли и почему."""
    rows = []
    for r in cfg.rules_table():
        reason = " ".join(str(r["reason"]).split())
        rows.append([r["id"], r["type_id"], r["kind"], r["enabled"],
                     reason if len(reason) <= width else reason[: width - 1] + "…"])
    return _table(rows, ["правило", "тип", "вид", "включено", "причина"])


def _print_report(res: dict[str, Any], tol: float, show_events: bool, catalog_src: str, cfg: Any,
                  show_trace: bool = False, show_rules: bool = True) -> None:
    if show_rules:
        print(f"Правила (профиль {cfg.profile}):")
        print(rules_status_table(cfg))
        print()
    rows = []
    for v in res["videos"]:
        gt_t = "/".join(f"{t:g}" for t in v.get("gt_all_t") or []) or None
        rows.append([v["name"], v.get("gt"), (gt_t + (" (конфликт)" if v.get("conflict") else "")) if gt_t else None,
                     v.get("pred"), v.get("type_id"), v.get("detection"), v.get("legal_basis"), v.get("t_peak"),
                     v.get("dt"), v.get("hit"), v.get("evidence_kind"), v.get("rule") or v.get("note"),
                     v.get("clauses")])
    print(_table(rows, ["видео", "GT", "GT_t", "pred", "тип", "обнаружение", "норма", "t_peak", "Δt", f"≤{tol:g}с",
                        "момент", "правило/прим.", "предикаты в t_peak"]))
    if show_events or show_trace:
        for v in res["videos"]:
            for e in v.get("events", []):
                moment = e["evidence"].get("kind") or e["evidence"].get("policy")
                print(f"  {v['name']}: {e['type_id']} {e['detection']}/{e['assessment']} [{e['t_start']:.1f}–"
                      f"{e['t_end']:.1f}] t_peak={e['t_peak']:.1f} ({moment})"
                      f" conf={e['confidence']:.2f} {','.join(e['rule_ids'])} {e['clauses']}")
                if show_trace and e.get("trace"):
                    for c in e["trace"].get("clauses", []):
                        print(f"      {c['kind']:<13} {c['id']:<16} {c['at_peak']:<8} "
                              f"true={c['frac_true']:.2f} unk={c['frac_unknown']:.2f}  {c['label']}")
    s = res["summary"]
    print()
    print(f"rules: {cfg.source_path} | профиль: {cfg.profile} | гипотезы: {', '.join(cfg.active_hypotheses) or '—'}")
    print(f"catalog: {catalog_src} | вердикт: {s['verdict']} | review = нарушение: {_fmt(s['count_review'])}")
    print(f"видео: {s['n_videos']} (кэш: {s['n_cached']}, размечено без дублей: {s['n_labeled']}, "
          f"конфликт таймкода: {s['n_conflict']})")
    print(f"TP={s['tp']} FP={s['fp']} FN={s['fn']} TN={s['tn']}  precision={_fmt(s['precision'], 3)} "
          f"recall={_fmt(s['recall'], 3)} F1={_fmt(s['f1'], 3)} accuracy={_fmt(s['accuracy'], 3)}")
    print(f"обнаружение+время (прокси A16): det={_fmt(s['det_score'], 3)} time={_fmt(s['time_score'], 3)} "
          f"joint={_fmt(s['joint'], 3)}")
    hits = ", ".join(f"≤{k}с: {_fmt(v, 2)}" for k, v in s["time_hit_rate"].items())
    print(f"таймкод (TP, без конфликтов): {hits}; mean|Δt|={_fmt(s['mean_abs_dt'], 2)} "
          f"median|Δt|={_fmt(s['median_abs_dt'], 2)}")
    if s["median_gt_minus_t_peak"] is not None:
        print(f"медиана (GT − t_peak) = {s['median_gt_minus_t_peak'] + 0.0:+.2f} с — справочно "
              f"(сдвиг — только по train/calib)")
    print(f"типы на TP: {s['types_on_tp'] or '—'} | latency правил: mean={_fmt(s['latency_ms']['mean'], 1)} мс, "
          f"max={_fmt(s['latency_ms']['max'], 1)} мс | VRAM: 0 (CPU)")
    fires = s.get("rule_fires") or res.get("rule_fires")
    if fires:
        print(_table([[rid, d["violation"], d["clean"], d["unlabeled"], d.get("observation", 0),
                       d.get("insufficient", 0)] for rid, d in sorted(fires.items())],
                     ["правило", "кандидат: на нарушениях", "кандидат: на чистых (FP)", "кандидат: без разметки",
                      "наблюдения", "insufficient"]))


def _variants(raw: dict, hyp_arg: str) -> list[tuple[str, str, list[str]]]:
    hyps = sorted((raw.get("hypotheses") or {}).keys())
    chosen = hyps if hyp_arg in ("all", "*") else [h.strip() for h in hyp_arg.split(",") if h.strip()]
    out = [("baseline", "baseline", [])]
    out += [(f"+{h}", "baseline", [h]) for h in chosen]
    if hyp_arg in ("all", "*") and "phase2" in (raw.get("profiles") or {}):
        out.append(("profile:phase2", "phase2", []))
    return out


def run_ablation(cache: dict, labels: dict, raw: dict, src: str, catalog: dict, hyp_arg: str, sets: list[str],
                 tol: float, min_conf: float, verdict: str) -> list[dict[str, Any]]:
    results = []
    base = None
    for name, profile, hyps in _variants(raw, hyp_arg):
        cfg = build_config(raw, src, profile, hyps, sets)
        res = run_replay(cache, labels, cfg, catalog, tol, min_conf, False, verdict)
        s = res["summary"]
        row = {"variant": name, **{k: s[k] for k in ("tp", "fp", "fn", "tn", "recall", "precision", "det_score",
                                                     "time_score", "joint", "median_abs_dt")},
               "hit1": s["time_hit_rate"].get("1.0"), "hit2": s["time_hit_rate"].get("2.0"),
               "types": s["types_on_tp"], "latency_ms": s["latency_ms"]["mean"]}
        if base is None:
            base = row
        both = row["joint"] is not None and base["joint"] is not None
        row["d_joint"] = (row["joint"] - base["joint"]) if both else None
        row["d_fp"] = row["fp"] - base["fp"]
        row["d_fn"] = row["fn"] - base["fn"]
        results.append(row)
    return results


def _candidate_edits(cfg: Any) -> list[str]:
    edits = []
    for r in cfg.enabled_rules:
        for f in (0.5, 0.75, 1.5, 2.0):
            edits.append(f"rules.{r.id}.min_duration_s={round(r.min_duration_s * f, 2)}")
        if not r.predicates:
            for on in (0.4, 0.6, 0.7):
                if abs(on - r.on) > 1e-9:
                    edits.append(f"rules.{r.id}.on={on}")
                    edits.append(f"rules.{r.id}.off={min(r.off, on)}")
    for ff in (0.5, 2.0):
        edits.append(f"global.ffill_max_s={ff}")
    return edits


def propose_edits(cache: dict, labels: dict, raw: dict, src: str, catalog: dict, profile: Optional[str],
                  hyps: list[str], sets: list[str], tol: float, verdict: str, max_edits: int = 2,
                  max_fp_increase: int = 1) -> list[dict[str, Any]]:
    """Жадно: ≤ max_edits однопараметрических правок, каждая повышает joint без роста FP > max_fp_increase.

    Рост recall за счёт лавины FP отсекается и ограничением FP, и ценой FP в det_score.
    Это диагностика по тем же данным (не независимый тест): применять только после LOO/отложенных роликов.
    """
    applied: list[str] = []
    proposals: list[dict[str, Any]] = []
    cfg = build_config(raw, src, profile, hyps, sets)
    base = run_replay(cache, labels, cfg, catalog, tol, 0.0, False, verdict)["summary"]
    for _ in range(max_edits):
        best = None
        for edit in _candidate_edits(cfg):
            if any(edit.split("=")[0] == a.split("=")[0] for a in applied):
                continue
            try:
                c2 = build_config(raw, src, profile, hyps, sets + applied + [edit])
            except Exception:
                continue
            s = run_replay(cache, labels, c2, catalog, tol, 0.0, False, verdict)["summary"]
            if s["joint"] is None or base["joint"] is None:
                continue
            if s["fp"] - base["fp"] > max_fp_increase:
                continue
            gain = s["joint"] - base["joint"]
            if gain > 1e-9 and (best is None or gain > best[0]):
                best = (gain, edit, s)
        if best is None:
            break
        gain, edit, s = best
        applied.append(edit)
        proposals.append({"set": edit, "joint_before": base["joint"], "joint_after": s["joint"], "gain": gain,
                          "tp": s["tp"], "fp": s["fp"], "fn": s["fn"], "tn": s["tn"],
                          "median_abs_dt": s["median_abs_dt"]})
        base = s
        cfg = build_config(raw, src, profile, hyps, sets + applied)
    return proposals


def _sweep(cache: dict, labels: dict, raw: dict, src: str, profile: Optional[str], hyps: list[str],
           base_sets: list[str], sweeps: list[str], catalog: dict, tol: float, min_conf: float,
           count_review: bool, verdict: str) -> list[dict[str, Any]]:
    axes = []
    for sw in sweeps:
        key, _, vals = sw.partition("=")
        axes.append([(key.strip(), v.strip()) for v in vals.split(",") if v.strip()])
    combos = list(itertools.product(*axes))[:MAX_SWEEP]
    results = []
    for combo in combos:
        sets = base_sets + [f"{k}={v}" for k, v in combo]
        cfg = build_config(raw, src, profile, hyps, sets)
        s = run_replay(cache, labels, cfg, catalog, tol, min_conf, count_review, verdict)["summary"]
        results.append({"params": dict(combo), **{k: s[k] for k in ("tp", "fp", "fn", "tn", "f1", "recall",
                                                                    "precision", "median_abs_dt", "joint")},
                        "hit": s["time_hit_rate"].get(str(tol))})
    results.sort(key=lambda r: (-(r["joint"] or 0), r["fn"], r["fp"]))
    return results


def rules_table_markdown(cfg: Any) -> str:
    lines = ["| Правило | Тип | Вид | Включено | По бездействию | Причина |", "|---|---|---|---|---|---|"]
    for r in cfg.rules_table():
        lines.append(f"| `{r['id']}` | {r['type_id']} | {r['kind']} | {'да' if r['enabled'] else 'нет'} | "
                     f"{'да' if r['inaction'] else '—'} | {r['reason']} |")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ml.rules.replay", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", help="папка кэша (A03: cache/<sha>/states.jsonl ...)")
    ap.add_argument("--labels", help="CSV разметки (схема L1: video_id,file,label,t_sec,md5,group,t_all,conflict)")
    ap.add_argument("--names", help="sha256 → файл: summary.json батча A03 или CSV sha256,file")
    ap.add_argument("--videos", help="папка с роликами: посчитать sha256 локально для сопоставления с кэшем")
    ap.add_argument("--rules", default=None, help="путь к rules.yaml (по умолчанию ml/rules/rules.yaml)")
    ap.add_argument("--profile", default=None, help="профиль из rules.yaml (baseline | phase2 ...)")
    ap.add_argument("--hypothesis", dest="hyps", action="append", default=[], help="добавить гипотезу (H1_... )")
    ap.add_argument("--hypotheses", default=None, metavar="all|H1,H2",
                    help="абляция: baseline, baseline+каждая гипотеза, профиль phase2 — одна таблица для L2")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                    help="переопределение параметра, напр. rules.V01_outsider_in_cashier_area.on=0.6")
    ap.add_argument("--sweep", action="append", default=[], metavar="KEY=v1,v2",
                    help="перебор значений (декартово произведение, ≤500 комбинаций)")
    ap.add_argument("--propose", action="store_true", help="≤ 2 правки порогов по данным (диагностика, не применяются)")
    ap.add_argument("--catalog", default=None, help="norms/catalog.yaml (иначе norms.load_catalog() или fallback)")
    ap.add_argument("--assume-verified", action="store_true",
                    help="КАЛИБРОВКА: считать нормы всех типов подтверждёнными (для --verdict legal)")
    ap.add_argument("--verdict", choices=("detection", "legal"), default="detection",
                    help="detection: violation_candidate по видео (по умолчанию); legal: + подтверждённая норма")
    ap.add_argument("--count-review", action="store_true", help="диагностика: review_observation = нарушение")
    ap.add_argument("--no-dedupe", action="store_true", help="не схлопывать дубликаты (group/md5)")
    ap.add_argument("--tol", type=float, default=2.0, help="допуск таймкода, с")
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--events", action="store_true", help="печатать все события")
    ap.add_argument("--trace", action="store_true", help="печатать trace предикатов/исключений по событиям")
    ap.add_argument("--json", dest="json_out", help="сохранить результаты в JSON")
    ap.add_argument("--csv", dest="csv_out", help="сохранить таблицу по видео в CSV")
    ap.add_argument("--rules-table", action="store_true", help="напечатать таблицу правил (markdown) и выйти")
    ap.add_argument("--no-rules-table", action="store_true", help="не печатать таблицу правил в отчёте")
    args = ap.parse_args(argv)

    raw, src = read_raw_config(args.rules)
    cfg = build_config(raw, src, args.profile, args.hyps, args.sets)
    if args.rules_table:
        print(rules_table_markdown(cfg))
        return 0
    if not args.cache:
        ap.error("нужен --cache (или --rules-table)")
    cache = load_cache_dir(args.cache)
    if not cache:
        print(f"[replay] в {args.cache} не найдено кэшей состояний", file=sys.stderr)
        return 2
    name_map: dict[str, str] = {}
    if args.names:
        name_map.update(load_name_map(args.names))
    if args.videos:
        name_map.update(hash_videos(args.videos))
    cache = apply_name_map(cache, name_map)
    labels = load_labels(args.labels) if args.labels else {}
    catalog, cat_src = _load_catalog(args.catalog, args.assume_verified)
    dedupe = not args.no_dedupe
    if args.hypotheses:
        rows = run_ablation(cache, labels, raw, src, catalog, args.hypotheses, args.sets, args.tol, args.min_conf,
                            args.verdict)
        f3 = (lambda x: None if x is None else f"{x:.3f}")  # noqa: E731
        print(_table([[r["variant"], r["tp"], r["fp"], r["fn"], r["tn"], f3(r["recall"]), f3(r["det_score"]),
                       f3(r["time_score"]), f3(r["joint"]), f3(r["d_joint"]), r["d_fp"], f3(r["hit1"]), f3(r["hit2"]),
                       f3(r["median_abs_dt"]), json.dumps(r["types"], ensure_ascii=False), r["latency_ms"]]
                      for r in rows],
                     ["вариант", "TP", "FP", "FN", "TN", "recall", "det", "time", "joint", "Δjoint", "ΔFP",
                      "hit≤1", "hit≤2", "med|Δt|", "типы на TP", "мс/видео"]))
        print("VRAM: 0 (правила — CPU). Решение о включении гипотезы: Δjoint > 0 без роста FP; "
              "размеченные ролики уже использованы в разведке — подтверждать на отложенных/LOO.")
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=str),
                                           encoding="utf-8")
        return 0
    if args.propose:
        props = propose_edits(cache, labels, raw, src, catalog, args.profile, args.hyps, args.sets, args.tol,
                              args.verdict)
        if not props:
            print("Правок порогов, повышающих joint без роста FP, не найдено — пороги не трогаем.")
        for i, p in enumerate(props, 1):
            print(f"{i}. --set {p['set']}: joint {p['joint_before']:.3f} → {p['joint_after']:.3f} "
                  f"(TP={p['tp']} FP={p['fp']} FN={p['fn']} TN={p['tn']}, med|Δt|={_fmt(p['median_abs_dt'], 2)})")
        print("Это диагностика по тем же данным: применять только после проверки на отложенных роликах/LOO.")
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(props, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0
    if args.sweep:
        results = _sweep(cache, labels, raw, src, args.profile, args.hyps, args.sets, args.sweep, catalog,
                         args.tol, args.min_conf, args.count_review, args.verdict)
        print(_table([[json.dumps(r["params"], ensure_ascii=False), r["tp"], r["fp"], r["fn"], r["tn"],
                       r["joint"], r["hit"], r["median_abs_dt"]] for r in results],
                     ["параметры", "TP", "FP", "FN", "TN", "joint", f"hit≤{args.tol:g}", "med|Δt|"]))
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0
    res = run_replay(cache, labels, cfg, catalog, args.tol, args.min_conf, args.count_review, args.verdict, dedupe)
    _print_report(res, args.tol, args.events, cat_src, cfg, args.trace, not args.no_rules_table)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if args.csv_out:
        keys = ["name", "gt", "gt_t", "pred", "type_id", "detection", "legal_basis", "n_cand", "n_review", "n_insuff",
                "t_peak", "t_start", "dt", "dt_onset", "hit", "evidence_kind", "conf", "rule", "clauses"]
        with open(args.csv_out, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(keys)
            for v in res["videos"]:
                w.writerow([v.get(k) for k in keys])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
