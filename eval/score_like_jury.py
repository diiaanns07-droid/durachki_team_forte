"""НАШ PROXY-балл (0..100) для сравнения итераций. Формула жюри НЕИЗВЕСТНА — это не прогноз оценки жюри.

Главное — scorecard из eval.metrics (реальные precision/recall/FP/FN/timeouts/latency с n и версиями);
proxy лишь сворачивает его в одно число по весам критериев ТЗ (Детекция 25 · Таймкод+кадр 15 · Тип 15 ·
Нормы 15 · Устойчивость 10 · Продукт/UI 10 · Архитектура 10). Компоненты без настоящих данных — n/a
(тип — только при реальной разметке типов; нормы — полнота ссылок, не корректность трактовки;
устойчивость — синтетические возмущения, не реальный новый ракурс). Формулы — в README.

CLI::

    python -m eval.score_like_jury --metrics runs/eval/metrics.json [--robustness robustness.json] [--jury-sim jury_sim.json]
    python -m eval.score_like_jury --labels data/labels.csv --preds runs/eval/preds
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any, Optional

from .common import load_config, read_json, write_json

def _step(dt_abs: Optional[float], steps: list[list[float]]) -> float:
    if dt_abs is None:
        return 0.0
    for thr, credit in sorted(steps, key=lambda x: x[0]):
        if dt_abs <= float(thr) + 1e-9:
            return float(credit)
    return 0.0


def _avg(xs: list[float]) -> Optional[float]:
    return statistics.fmean(xs) if xs else None


def _norm_score(n: dict) -> float:
    has = 1.0 if n.get("n_rules", 0) > 0 else 0.0
    src = n.get("in_catalog") if n.get("in_catalog") is not None else n.get("url_ok")
    return 0.4 * has + 0.3 * (1.0 if src else 0.0) + 0.3 * float(n.get("verified_share") or 0.0)


def component_detection(summary: dict, cfg: dict) -> tuple[Optional[float], str]:
    v = summary["video"]
    j = cfg.get("jury") or {}
    cfn, cfp = float(j.get("fn_cost", 2.0)), float(j.get("fp_cost", 1.0))
    P, N = v["n_pos"], v["n_neg"]
    if not P + N:
        return None, "нет видео"
    # failed/partial на негативе — тоже ошибка (ответа «чисто» нет); на позитиве уже входит в FN
    fp_eff = v["fp"] + int(v.get("unresolved_neg", len([x for x in v.get("failed_ids", [])
                                                       if x not in set(v.get("fn_ids", []))])))
    cost = cfn * v["fn"] + cfp * fp_eff
    val = max(0.0, 1.0 - cost / (cfn * P + cfp * N))
    return val, f"1 − ({cfn:g}·FN + {cfp:g}·(FP+без ответа⁻)) / ({cfn:g}·P + {cfp:g}·N): FN={v['fn']} FP={v['fp']} " \
                f"без ответа⁻={fp_eff - v['fp']} P={P} N={N}"


def component_timecode(per_video: list[dict], cfg: dict) -> tuple[Optional[float], str]:
    j = cfg.get("jury") or {}
    tp_part = float(j.get("time_part", 0.667))
    steps = j.get("time_steps") or [[1.0, 1.0], [2.0, 0.75], [3.0, 0.5], [5.0, 0.25]]
    vals = []
    for e in per_video:
        if e["gt_label"] != "violation" or not e["gt_times"]:
            continue
        if e["status"] != "done" or e["verdict"] != "violation":
            vals.append(0.0)
            continue
        vals.append(tp_part * _step(e.get("dt_abs"), steps) + (1 - tp_part) * (1.0 if e.get("evidence_ok") else 0.0))
    return _avg(vals), f"среднее по {len(vals)} позитивам с GT-таймкодом: {tp_part:g}·ступень(|Δt|) + " \
                       f"{1 - tp_part:g}·[кадр в ±tol]; FN → 0"


def component_type(summary: dict) -> tuple[Optional[float], str, str]:
    """Только по настоящей разметке типов (data/labels_types.csv). Без неё — n/a, никаких proxy."""
    ty = summary.get("types") or {}
    if ty.get("available"):
        return ty.get("accuracy"), f"точность типа по {ty.get('n_events')} GT-событиям (пропуск → 0)", "measured"
    return None, "нет настоящей разметки типов (labels_types.csv) — не оцениваем", "n/a"


def component_norms(per_video: list[dict]) -> tuple[Optional[float], str]:
    vals = []
    for e in per_video:
        if e["gt_label"] != "violation":
            continue
        if e["status"] != "done" or e["verdict"] != "violation" or not e.get("norms"):
            vals.append(0.0)
            continue
        vals.append(statistics.fmean(_norm_score(n) for n in e["norms"]))
    return _avg(vals), "PROXY полноты ссылок (не корректности трактовки): среднее по позитивам 0.4·[≥1 rule] + " \
                       "0.3·[все rules из каталога (или adilet URL)] + 0.3·доля verified; FN → 0"


def component_robustness(rob: Optional[dict]) -> tuple[Optional[float], str]:
    if not rob:
        return None, "нет robustness.json"
    s = rob.get("summary") or rob
    flip = s.get("flip_rate")
    stab = s.get("timecode_stable_rate")
    if flip is None:
        return None, "в robustness.json нет flip_rate"
    val = 0.7 * (1 - float(flip)) + 0.3 * (float(stab) if stab is not None else 1 - float(flip))
    return val, f"PROXY на СИНТЕТИЧЕСКИХ возмущениях (не реальный новый ракурс): 0.7·(1 − смена вердикта={flip}) + " \
                f"0.3·стабильный таймкод={stab}"


def component_product(summary: dict) -> tuple[Optional[float], str]:
    sch, ev, lng = summary["schema"], summary["evidence"], summary["language"]
    parts = []
    if sch.get("valid_rate") is not None:
        parts.append((0.4, sch["valid_rate"]))
    if ev.get("valid_jpeg_rate") is not None:
        parts.append((0.2, ev["valid_jpeg_rate"]))
    if ev.get("frames_per_violation_ok") is not None:
        parts.append((0.2, ev["frames_per_violation_ok"]))
    if lng.get("n"):
        parts.append((0.2, float(lng["hedged_rate"] or 0) * (1 - float(lng["forbidden_rate"] or 0))))
    if not parts:
        return None, "нет ответов"
    w = sum(p[0] for p in parts)
    return sum(a * b for a, b in parts) / w, "PROXY: 0.4·схема ТЗ + 0.2·валидный JPEG + 0.2·1–3 кадра + " \
                                             "0.2·hedged·(1−forbidden) (UI жюри оценит само)"


def component_architecture(summary: dict, jury_sim: Optional[dict]) -> tuple[Optional[float], str]:
    lat, v = summary["latency"], summary["video"]
    parts = []
    if lat.get("within_sla_rate") is not None:
        parts.append((0.5, lat["within_sla_rate"]))
    if v["n"]:
        parts.append((0.3, 1 - v["failed"] / v["n"]))
    if jury_sim:
        js = jury_sim.get("summary") or jury_sim
        if js.get("pass_rate") is not None:
            parts.append((0.2, float(js["pass_rate"])))
    if not parts:
        return None, "нет данных"
    w = sum(p[0] for p in parts)
    return sum(a * b for a, b in parts) / w, "PROXY: 0.5·доля в SLA + 0.3·(1 − доля failed) + 0.2·jury_sim pass_rate"


def score(metrics: dict, robustness: Optional[dict] = None, jury_sim: Optional[dict] = None,
          cfg: Optional[dict] = None) -> dict:
    cfg = cfg or load_config()
    j = cfg.get("jury") or {}
    w = j.get("weights") or {}
    defaults = j.get("defaults") or {}
    summary, per_video = metrics["summary"], metrics["per_video"]
    comps: dict[str, dict[str, Any]] = {}

    def put(name: str, val: Optional[float], note: str, source: str = "measured") -> None:
        if val is None and defaults.get(name) is not None:
            val, source, note = float(defaults[name]), "default", f"значение по умолчанию из config ({note})"
        elif val is None:
            source = "n/a"
        weight = float(w.get(name, 0))
        comps[name] = {"weight": weight, "value": None if val is None else round(float(val), 4),
                       "points": None if val is None else round(weight * float(val), 2), "source": source,
                       "note": note}

    put("detection", *component_detection(summary, cfg))
    put("timecode", *component_timecode(per_video, cfg))
    tv, tn, ts = component_type(summary)
    put("type", tv, tn, ts)
    nv, nn = component_norms(per_video)
    put("norms", nv, nn, "proxy")
    rv, rn = component_robustness(robustness)
    put("robustness", rv, rn, "proxy")
    pv, pn = component_product(summary)
    put("product", pv, pn, "proxy")
    av, an = component_architecture(summary, jury_sim)
    put("architecture", av, an, "proxy")
    avail = [c for c in comps.values() if c["points"] is not None]
    raw = sum(c["points"] for c in avail)
    wmax = sum(c["weight"] for c in avail)
    val = round(100.0 * raw / wmax, 2) if wmax else 0.0
    return {"proxy_score": val, "jury_score": val,  # jury_score — алиас для совместимости, это НЕ оценка жюри
            "disclaimer": "Наш proxy; формула жюри неизвестна. Решения принимать по scorecard.",
            "points": round(raw, 2), "max_points_available": wmax,
            "missing": [k for k, c in comps.items() if c["points"] is None],
            "components": comps, "scorecard": metrics.get("scorecard")}


def format_score(res: dict) -> str:
    lines = []
    if res.get("scorecard"):
        from .metrics import format_scorecard

        lines.append(format_scorecard(res["scorecard"]))
    lines.append(f"PROXY_SCORE = {res['proxy_score']:.1f} / 100  (НАШ proxy, формула жюри неизвестна; "
                 f"{res['points']:.1f} из {res['max_points_available']:g} доступных баллов"
                 + (f"; n/a: {', '.join(res['missing'])}" if res["missing"] else "") + ")")
    for k, c in res["components"].items():
        pts = "  n/a" if c["points"] is None else f"{c['points']:5.1f}"
        lines.append(f"  {k:<13} {pts} / {c['weight']:>4g}  [{c['source']}]  {c['note']}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="100-балльная аппроксимация оценки жюри")
    ap.add_argument("--metrics", default=None, help="metrics.json от eval.metrics / eval.run")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--preds", default=None)
    ap.add_argument("--types", default="data/labels_types.csv")
    ap.add_argument("--robustness", default=None)
    ap.add_argument("--jury-sim", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.metrics:
        metrics = read_json(args.metrics)
    elif args.preds:
        from .metrics import evaluate_dirs

        metrics = evaluate_dirs(args.labels, args.preds, args.types, cfg)
    else:
        ap.error("нужен --metrics или --preds")
    rob = read_json(args.robustness) if args.robustness else None
    js = read_json(args.jury_sim) if args.jury_sim else None
    res = score(metrics, rob, js, cfg)
    print(format_score(res))
    if args.out:
        write_json(args.out, res)
    elif args.metrics:
        write_json(Path(args.metrics).with_name("jury_score.json"), res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
