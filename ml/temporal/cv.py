"""Nested grouped CV для baseline и temporal head (разведочная оценка: все 42 строки уже просматривались).

    python -m ml.temporal.cv [--kind baseline|temporal|both] [--outer loo|5] [--inner 3] [--repeats 1]

Внешний сплит — оценка; внутренний (только на train внешнего фолда) — калибровка: null-CDF блоковой
статистики чистых роликов (для пулинга topk_null) и порог. Скейлер/PCA/модель подгоняются на train каждого
фолда. Валидация — только оригиналы; дубликаты — одна группа. Все режимы пулинга (max/topk/topk_null)
считаются по одним и тем же OOF-кривым — честное сравнение без переобучения. Отчёт: reports/temporal_cv.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from .config import load_config, reports_dir
from .dataset import Sample, fit_spec, prepare_samples
from .features import TextBank
from .candidates import candidate_recall, change_curve, resample_to, select_candidates
from .metrics import recommend_threshold, roc_auc, spearman, summarize
from .postproc import block_stats, pool_score, risk_curve, timecode
from .predictors import fit_predictor
from .splits import check_split, grouped_folds, holdout_split

log = logging.getLogger("sentinel.temporal")
POOLINGS = ("max", "topk", "topk_null")
EXPLORATORY = ("разведочная оценка: все строки разметки уже просматривались командой — не независимый тест; "
               "сравнивайте конфигурации между собой, абсолютные цифры оптимистичны")


def _parse_folds(v: Union[str, int, None]) -> Union[str, int]:
    if v is None:
        return "loo"
    return "loo" if str(v).lower() == "loo" else int(v)


def duration_of(s: Sample) -> float:
    fs = s.versions[0]
    return float(fs.meta.get("duration") or (fs.times[-1] if len(fs.times) else 0.0))


def fit_and_score(kind: str, train: Sequence[Sample], val: Sequence[Sample], bank: TextBank, cfg: dict,
                  seed: int, epochs: Optional[int] = None) -> list[dict]:
    """Подогнать spec+модель на train (все версии), вернуть риск-кривые val ТОЛЬКО по оригиналу (версия 0)."""
    spec = fit_spec([fs for s in train for fs in s.versions], bank, cfg)
    prepare_samples(list(train), bank, spec)
    val_only = [Sample(name=s.name, group=s.group, label=s.label, times=s.times, versions=s.versions[:1])
                for s in val]
    prepare_samples(val_only, bank, spec)
    model = fit_predictor(kind, train, spec, cfg, seed=seed, epochs=epochs)
    out = []
    for s, orig in zip(val_only, val):
        t_inf = time.perf_counter()
        p, _ = model.predict(s.X[0], spec.fps)
        dur = duration_of(orig)
        t, r = risk_curve(p, s.grid[0], cfg, dur)
        infer_ms = (time.perf_counter() - t_inf) * 1000.0
        fs = orig.versions[0]
        ch = resample_to(fs.times, change_curve(fs.times, fs.emb, fs.dyn, cfg), t)
        out.append({"t": t, "r": r, "duration": dur, "change": ch, "infer_ms": infer_ms})
    return out


def _quantiles(x: np.ndarray, n: int = 256) -> list[float]:
    x = np.sort(np.asarray(x, np.float64))
    if len(x) <= n:
        return [round(float(v), 6) for v in x]
    return [round(float(v), 6) for v in np.quantile(x, np.linspace(0, 1, n))]


def calibrate(kind: str, train: Sequence[Sample], bank: TextBank, cfg: dict, seed: int,
              epochs: Optional[int] = None) -> dict:
    """Внутренний grouped CV ТОЛЬКО на train: OOF-кривые -> null-CDF (чистые ролики) + порог по OOF-скорам.
    Собственные блоки ролика исключаются из null при его оценке (leave-own-blocks-out)."""
    cv = cfg["cv"]
    n_in = min(int(cv.get("inner", 3)), len({s.group for s in train}))
    curves: dict[int, dict] = {}
    if n_in >= 2:
        for itr, iva in grouped_folds(train, n_in, seed=seed + 101):
            check_split(train, itr, iva)
            res = fit_and_score(kind, [train[i] for i in itr], [train[i] for i in iva], bank, cfg, seed, epochs)
            for i, c in zip(iva, res):
                curves[i] = c
    blocks = {i: block_stats(c["r"], cfg) for i, c in curves.items() if not train[i].label}
    null_all = np.concatenate(list(blocks.values())) if blocks else np.zeros(0)
    scores, labels = [], []
    for i, c in curves.items():
        nl = np.concatenate([b for j, b in blocks.items() if j != i]) if (i in blocks and len(blocks) > 1) else null_all
        scores.append(pool_score(c["r"], cfg, nl, c["duration"]))
        labels.append(train[i].label)
    thr = (recommend_threshold(labels, scores, cv.get("threshold_margin", 0.02), cv.get("min_recall_fp0", 0.8))
           if len(set(labels)) == 2 else float(cfg["infer"]["event_threshold"]))
    return {"null": _quantiles(null_all), "n_null_blocks": int(len(null_all)), "threshold": float(thr),
            "n_inner": n_in, "oof_auc": roc_auc(labels, scores) if len(set(labels)) == 2 else None}


def nested_cv(samples: Sequence[Sample], bank: TextBank, cfg: dict, kind: str,
              outer: Union[str, int] = "loo", inner: int = 3, seed: int = 0,
              epochs: Optional[int] = None) -> dict:
    t0 = time.monotonic()
    cv = cfg["cv"]
    c2 = {**cfg, "cv": {**cv, "inner": inner}}
    folds = grouped_folds(samples, outer, seed=seed)
    n = len(samples)
    pool_mode = str(cfg["infer"].get("pooling", "topk_null"))
    scores = {m: np.zeros(n, np.float32) for m in POOLINGS}
    thr_inner = np.full(n, np.nan, np.float32)
    curves: list[Optional[dict]] = [None] * n
    for fi, (tr, va) in enumerate(folds):
        check_split(samples, tr, va)
        train = [samples[i] for i in tr]
        cal = calibrate(kind, train, bank, c2, seed=seed + fi, epochs=epochs)
        res = fit_and_score(kind, train, [samples[i] for i in va], bank, cfg, seed=seed + fi, epochs=epochs)
        null = np.asarray(cal["null"])
        for i, c in zip(va, res):
            curves[i] = c
            for m in POOLINGS:
                scores[m][i] = pool_score(c["r"], {**cfg, "infer": {**cfg["infer"], "pooling": m}}, null, c["duration"])
            thr_inner[i] = cal["threshold"]
        log.info("cv[%s] фолд %d/%d готов (%.1f с)", kind, fi + 1, len(folds), time.monotonic() - t0)

    labels = [s.label for s in samples]
    tol = cv.get("hit_tolerances_s", [1, 2, 3])
    sc = scores[pool_mode]
    policies = list(cv.get("timecode_policies", ["peak", "onset"]))
    peaks = {pol: [timecode(c["t"], c["r"], pol, cfg, c.get("change")) if c is not None else None for c in curves]
             for pol in policies}
    main_pol = str(cfg["infer"].get("timecode_policy", "peak"))
    free = summarize(labels, sc.tolist(), peaks.get(main_pol, peaks[policies[0]]), [s.times for s in samples], tol)
    free["note"] = ("recall_at_fp*/thr_fp* — с «оракульным» порогом по тем же OOF (оптимистично); "
                    "auc и таймкод — независимы от порога")
    pred = sc > thr_inner
    y = np.asarray(labels, bool)
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    nested = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
              "recall": tp / max(tp + fn, 1), "fpr": fp / max(fp + tn, 1),
              "precision": tp / max(tp + fp, 1) if tp + fp else None, "accuracy": (tp + tn) / max(n, 1),
              "note": "порог и null каждого внешнего фолда подобраны внутренним CV только на его train"}
    durs = [duration_of(s) for s in samples]
    pos_idx = [i for i, v in enumerate(labels) if v]
    time_by_policy = {}
    for pol in policies:
        errs = [min(abs(peaks[pol][i] - t) for t in samples[i].times) if peaks[pol][i] is not None and samples[i].times
                else float("inf") for i in pos_idx]
        fin = [e for e in errs if np.isfinite(e)]
        time_by_policy[pol] = {"dt_median": float(np.median(fin)) if fin else None,
                               **{f"hit@{t:g}s": float(np.mean([e <= t for e in errs])) if errs else 0.0 for t in tol}}
    per = []
    for i, s in enumerate(samples):
        dt = None
        if s.label and peaks[main_pol][i] is not None and s.times:
            dt = round(float(min(abs(peaks[main_pol][i] - t) for t in s.times)), 2)
        per.append({"name": s.name, "group": s.group, "label": s.label, "times": s.times,
                    "sources": [{"name": x["name"], "label": x["label"], "times": x["times"]} for x in s.sources],
                    "conflict": s.conflict, "time_conflict": bool(getattr(s, "time_conflict", False)),
                    "score": round(float(sc[i]), 4),
                    "scores_by_pooling": {m: round(float(scores[m][i]), 4) for m in POOLINGS},
                    "t_peak": peaks[main_pol][i], "dt": dt, "thr_inner": round(float(thr_inner[i]), 4),
                    "pred": bool(pred[i]), "duration": round(durs[i], 2)})
    by_pool = {m: {"auc": roc_auc(labels, scores[m].tolist()),
                   "spearman_score_duration_neg": spearman([d for d, l in zip(durs, labels) if not l],
                                                           [float(v) for v, l in zip(scores[m], labels) if not l]),
                   "spearman_score_duration_all": spearman(durs, scores[m].tolist())} for m in POOLINGS}
    pr = cfg.get("propose", {})
    budgets = [int(b) for b in pr.get("budgets", [3, 5, 8])]
    cands = {}
    for mode in ("risk", "change", "hybrid"):
        per_video = [select_candidates(c["t"], c["r"], c.get("change"), cfg, mode=mode, n_max=max(budgets))
                     if c is not None else [] for c in curves]
        cands[mode] = candidate_recall(per_video, [s.times if s.label else [] for s in samples], budgets,
                                       float(pr.get("recall_tol_s", 1.5)))
    watch_lim = float(cv.get("duration_watchdog_rho", 0.3))
    rho_neg = by_pool[pool_mode]["spearman_score_duration_neg"]
    return {"kind": kind, "protocol": "nested_grouped_cv", "evaluation_status": EXPLORATORY,
            "outer": outer, "inner": inner, "seed": seed, "pooling": pool_mode, "timecode_policy": main_pol,
            "n_folds": len(folds), "n_samples": n, "elapsed_s": round(time.monotonic() - t0, 1),
            "threshold_free": free, "nested": nested, "by_pooling": by_pool, "timecode_by_policy": time_by_policy,
            "candidates": cands,
            "latency_model_ms_per_video": round(float(np.mean([c["infer_ms"] for c in curves if c is not None])), 2),
            "duration_watchdog": {"duration_only_auc": roc_auc(labels, durs), "spearman_neg": rho_neg,
                                  "limit": watch_lim, "warn": bool(rho_neg is not None and abs(rho_neg) > watch_lim),
                                  "note": "длительность — не признак; |ρ(скор, длительность)| на чистых роликах должен "
                                          "быть мал, иначе пулинг/признаки протекают длиной"},
            "per_video": per}


def _select(reports: dict) -> str:
    def key(r: dict) -> tuple:
        f = r["threshold_free"]
        return (f.get("auc") or 0.0, r["nested"]["recall"] - r["nested"]["fpr"], -(f.get("dt_median") or 99))
    kinds = [k for k in ("baseline", "temporal") if k in reports]
    best = max(kinds, key=lambda k: key(reports[k]))
    # temporal выбираем, только если он строго лучше baseline по AUC (иначе — проще и стабильнее baseline)
    if best == "temporal" and "baseline" in reports and key(reports["temporal"])[0] <= key(reports["baseline"])[0]:
        best = "baseline"
    return best


def run(cfg: dict, kinds: Sequence[str], outer: Union[str, int, None], inner: int, holdout: float, seed: int,
        index: Optional[str] = None, max_aug: Optional[int] = None, epochs: Optional[int] = None,
        samples: Optional[list[Sample]] = None, bank: Optional[TextBank] = None) -> dict:
    if samples is None or bank is None:
        from .data_index import bank_for_index, load_samples

        samples, idx = load_samples(cfg, index, max_aug=max_aug)
        bank = bank_for_index(cfg, idx)
    rest, hold = holdout_split(samples, holdout, seed=seed)
    if hold:
        check_split(samples, rest, hold)
    rest_s = [samples[i] for i in rest]
    report: dict[str, Any] = {"n_groups": len(samples), "n_pos": sum(s.label for s in samples),
                              "evaluation_status": EXPLORATORY,
                              "conflicts": [s.name for s in samples if s.conflict],
                              "time_conflicts": [s.name for s in samples if getattr(s, "time_conflict", False)]}
    for kind in kinds:
        o = outer if outer is not None else (cfg["cv"]["outer_temporal"] if "temporal" in kinds else cfg["cv"]["outer"])
        report[kind] = nested_cv(rest_s, bank, cfg, kind, _parse_folds(o), inner, seed,
                                 epochs=epochs or (cfg["cv"].get("epochs") if kind == "temporal" else None))
    report["selected"] = _select({k: report[k] for k in kinds})
    if hold:  # отложенные группы: одна оценка выбранной модели, калибровка — только на rest
        kind = report["selected"]
        cal = calibrate(kind, rest_s, bank, {**cfg, "cv": {**cfg["cv"], "inner": inner}}, seed=seed, epochs=epochs)
        res = fit_and_score(kind, rest_s, [samples[i] for i in hold], bank, cfg, seed=seed, epochs=epochs)
        hs = [samples[i] for i in hold]
        sc = [pool_score(c["r"], cfg, np.asarray(cal["null"]), c["duration"]) for c in res]
        pol = str(cfg["infer"].get("timecode_policy", "peak"))
        report["holdout"] = {
            "kind": kind, "threshold": cal["threshold"], "names": [s.name for s in hs],
            **summarize([s.label for s in hs], sc, [timecode(c["t"], c["r"], pol, cfg) for c in res],
                        [s.times for s in hs], cfg["cv"].get("hit_tolerances_s", [1, 2, 3])),
            "pred": [bool(v > cal["threshold"]) for v in sc], "labels": [s.label for s in hs],
            "note": "отложенные группы не участвовали в подгонке и калибровке; данные при этом уже "
                    "просматривались командой — оценка разведочная",
        }
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ml.temporal: nested grouped CV (разведочная оценка)")
    ap.add_argument("--index", default=None, help="cache/features/index.json")
    ap.add_argument("--kind", default="both", choices=["baseline", "temporal", "both"])
    ap.add_argument("--outer", default=None, help="loo | число фолдов")
    ap.add_argument("--inner", type=int, default=None)
    ap.add_argument("--holdout", type=float, default=None)
    ap.add_argument("--max-aug", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    kinds = ["baseline", "temporal"] if a.kind == "both" else [a.kind]
    if "temporal" in kinds:
        try:
            import torch  # noqa: F401
        except ImportError:
            log.warning("torch не установлен — temporal head пропущен, только baseline")
            kinds = ["baseline"]
    rep = run(cfg, kinds, a.outer, a.inner or int(cfg["cv"]["inner"]),
              cfg["cv"]["holdout_frac"] if a.holdout is None else a.holdout,
              cfg["train"]["seed"] if a.seed is None else a.seed, a.index, a.max_aug, a.epochs)
    out = Path(a.out) if a.out else reports_dir(cfg) / "temporal_cv.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    for k in kinds:
        f, nst, dw = rep[k]["threshold_free"], rep[k]["nested"], rep[k]["duration_watchdog"]
        print(f"[{k}] pooling={rep[k]['pooling']} AUC={f['auc']} nested: recall={nst['recall']:.2f} "
              f"FPR={nst['fpr']:.2f} |dt| median={f['dt_median']} hit@2s={f.get('hit@2s')} "
              f"rho(score,dur|neg)={dw['spearman_neg']} warn={dw['warn']} ({rep[k]['elapsed_s']} с)")
        print("      AUC by pooling: " + ", ".join(f"{m}={v['auc']}" for m, v in rep[k]["by_pooling"].items()))
        print("      candidates recall@±1.5s: " + "; ".join(
            f"{m}: " + ", ".join(f"{kk}={vv}" for kk, vv in v.items() if kk.startswith("recall@"))
            for m, v in rep[k]["candidates"].items()))
    print(f"selected={rep['selected']} -> {out}  [{EXPLORATORY}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
