"""Санити-полы и детекторы утечек. Любая модель должна их обгонять; иначе выигрыш — артефакт.

* полы вердикта: «всё violation», «всё clean», baseline по длительности (короче → нарушение; на наших данных
  ролики с нарушением в среднем короче — AUC порядка 0.67): AUC, порог in-sample (оптимистично) и LOGO-порог;
* пол таймкода: t = (медианная относительная позиция GT) × длительность → hit@1/2/3;
* наши скоры: AUC, корреляция Спирмена с длительностью (⚠ при |ρ| ≥ порога), AUC внутри терцилей длительности
  (сигнал сверх длительности), корреляция с номером файла;
* пулинг кривой риска (timeline): max / mean / top3 / p90 / prominence — AUC и ρ с длительностью
  (max по длинному ролику чаще даёт ложный пик).
Всё на 42 видео — разведочная оценка. Длительность, позиция и имя файла — НЕ признаки модели.

CLI::

    python -m eval.sanity --labels data/labels.csv --manifest data/manifest.csv [--run runs/eval/dev1]
        [--scores cache/baseline/siglip2_window_oof.csv:score_oof]
"""
from __future__ import annotations

import argparse
import math
import re
import statistics
from pathlib import Path
from typing import Any, Optional

from .common import load_config, normalize_video_id, read_json, write_json
from .labels import LABEL_VIOLATION, _read_csv_dicts, load_ground_truth


def auc(scores: list[float], labels: list[bool]) -> Optional[float]:
    """ROC-AUC (Манн–Уитни, ничьи = 0.5). None, если нет обоих классов."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a: list[float], b: list[float]) -> Optional[float]:
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return None
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return round(num / den, 4) if den else None


def stratified_auc(scores: list[float], labels: list[bool], strata_key: list[float], n_strata: int = 3) -> Optional[float]:
    """Средний (взвешенный) AUC внутри квантилей длительности: есть ли сигнал сверх длительности."""
    idx = sorted(range(len(scores)), key=lambda i: strata_key[i])
    if len(idx) < n_strata * 2:
        return None
    size = math.ceil(len(idx) / n_strata)
    tot, w = 0.0, 0
    for k in range(n_strata):
        part = idx[k * size:(k + 1) * size]
        a = auc([scores[i] for i in part], [labels[i] for i in part])
        if a is not None:
            n_pairs = sum(labels[i] for i in part) * sum(not labels[i] for i in part)
            tot += a * n_pairs
            w += n_pairs
    return round(tot / w, 4) if w else None


def _confusion(pred: list[bool], labels: list[bool]) -> dict:
    tp = sum(p and y for p, y in zip(pred, labels))
    fp = sum(p and not y for p, y in zip(pred, labels))
    fn = sum((not p) and y for p, y in zip(pred, labels))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": len(pred) - tp - fp - fn,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None}


def _best_threshold(scores: list[float], labels: list[bool], fn_cost: float = 2.0) -> float:
    pts = sorted(set(scores))
    cands = [pts[0] - 1e-9] + [(a + b) / 2 for a, b in zip(pts, pts[1:])] + [pts[-1] + 1e-9]

    def cost(th: float) -> float:
        return sum(fn_cost * (y and s < th) + ((not y) and s >= th) for s, y in zip(scores, labels))

    return min(cands, key=lambda th: (cost(th), th))


def score_floor(name: str, scores: list[float], labels: list[bool], groups: list[str]) -> dict:
    """AUC + порог in-sample (оптимистично) + leave-one-group-out порог."""
    th_in = _best_threshold(scores, labels)
    logo_pred = []
    for i, g in enumerate(groups):
        tr = [k for k in range(len(scores)) if groups[k] != g]
        th = _best_threshold([scores[k] for k in tr], [labels[k] for k in tr])
        logo_pred.append(scores[i] >= th)
    return {"name": name, "auc": auc(scores, labels),
            "in_sample": _confusion([s >= th_in for s in scores], labels) | {"note": "порог подобран на этих же данных"},
            "logo": _confusion(logo_pred, labels)}


# ----------------------------------------------------------------------------- входы
def load_durations(manifest_csv: Optional[str | Path] = None, videos_dir: Optional[str | Path] = None,
                   preds: Optional[dict[str, dict]] = None) -> dict[str, float]:
    out: dict[str, float] = {}
    if manifest_csv and Path(manifest_csv).exists():
        for d in _read_csv_dicts(manifest_csv):
            vid = normalize_video_id(d.get("video_id") or d.get("file"))
            for k in ("duration_s", "duration", "dur"):
                try:
                    if vid and d.get(k):
                        out[vid] = float(d[k])
                        break
                except ValueError:
                    pass
    for vid, api in (preds or {}).items():
        meta = (api or {}).get("meta") or {}
        for k in ("duration", "duration_s"):
            if vid not in out and isinstance(meta.get(k), (int, float)) and meta[k] > 0:
                out[vid] = float(meta[k])
    if videos_dir and Path(videos_dir).exists():
        from .common import ffprobe_info, list_videos, video_id_for_path

        for p in list_videos(videos_dir):
            vid = video_id_for_path(p)
            if vid not in out:
                d = ffprobe_info(p).get("duration")
                if d:
                    out[vid] = float(d)
    return out


def load_score_csv(spec: str) -> tuple[str, dict[str, float]]:
    """'path.csv:column' → (name, {video_id: score})."""
    path, _, col = spec.rpartition(":") if re.search(r":[A-Za-z_][\w]*$", spec) else (spec, "", "score")
    rows = _read_csv_dicts(path)
    out = {}
    for d in rows:
        vid = normalize_video_id(d.get("video_id"))
        try:
            if vid and d.get(col) not in (None, ""):
                out[vid] = float(d[col])
        except ValueError:
            continue
    return f"{Path(path).stem}:{col}", out


def timeline_poolings(preds: dict[str, dict]) -> dict[str, dict[str, float]]:
    pools: dict[str, dict[str, float]] = {k: {} for k in ("tl_max", "tl_mean", "tl_top3", "tl_p90", "tl_prominence")}
    for vid, api in preds.items():
        r = sorted(float(p[1]) for p in (api or {}).get("timeline") or []
                   if isinstance(p, (list, tuple)) and len(p) == 2 and isinstance(p[1], (int, float)))
        if not r:
            continue
        pools["tl_max"][vid] = r[-1]
        pools["tl_mean"][vid] = statistics.fmean(r)
        pools["tl_top3"][vid] = statistics.fmean(r[-3:])
        pools["tl_p90"][vid] = r[min(len(r) - 1, int(0.9 * len(r)))]
        pools["tl_prominence"][vid] = r[-1] - statistics.median(r)
    return {k: v for k, v in pools.items() if v}


def run_scores(run_dir: str | Path) -> dict[str, dict[str, float]]:
    """Скоры из прогона eval.run: scores_raw (max), макс. confidence нарушений, пулинги timeline."""
    from .loo import video_score
    from .metrics import load_predictions

    run = Path(run_dir)
    preds, _ = load_predictions(run / "preds") if (run / "preds").exists() else ({}, {})
    out: dict[str, dict[str, float]] = {}
    sr = run / "scores_raw.jsonl"
    if sr.exists():
        import json

        rows = [json.loads(x) for x in sr.read_text(encoding="utf-8").splitlines() if x.strip()]
        sc = {r["video_id"]: video_score(r) for r in rows}
        out["scores_raw_max"] = {k: v for k, v in sc.items() if v is not None}   # нет скора — не 0
    conf = {}
    for vid, api in preds.items():
        cs = [float(v.get("confidence") or 0) for v in (api or {}).get("violations") or [] if isinstance(v, dict)]
        if (api or {}).get("status") == "done":
            conf[vid] = max(cs) if cs else 0.0
    if conf:
        out["violation_conf_max"] = conf
    out.update(timeline_poolings(preds))
    return out


# ----------------------------------------------------------------------------- главное
def sanity(gt: dict, durations: dict[str, float], scores: dict[str, dict[str, float]],
           cfg: Optional[dict] = None) -> dict:
    cfg = cfg or load_config()
    rho_warn = float((cfg.get("sanity") or {}).get("duration_corr_warn", 0.3))
    windows = [float(k) for k in cfg.get("hit_windows_s", [0.5, 1, 2, 3])]
    items = [(k, g) for k, g in gt.items() if g.label in ("violation", "no_violation")]
    dur_of = {k: durations.get(k, next((durations[m] for m in g.members if m in durations), None)) for k, g in items}
    with_dur = [(k, g) for k, g in items if dur_of[k]]
    labels = [g.label == LABEL_VIOLATION for _, g in with_dur]
    groups = [g.group or k for k, g in with_dur]
    P = sum(g.label == LABEL_VIOLATION for _, g in items)       # полы вердикта — по ВСЕМ размеченным видео
    N = len(items) - P
    out: dict[str, Any] = {"n": len(items), "n_with_duration": len(with_dur), "evaluation_kind": "exploratory",
                           "note": "Полы и проверки утечек — разведочные (все видео уже использованы в разработке)."}
    out["floors"] = {
        "all_violation": {"recall": 1.0 if P else None, "precision": round(P / (P + N), 4) if P + N else None,
                          "fp": N, "fn": 0},
        "all_clean": {"recall": 0.0 if P else None, "precision": None, "fp": 0, "fn": P},
    }
    durs = [dur_of[k] for k, _ in with_dur]
    if with_dur:
        out["floors"]["duration_shorter_is_violation"] = score_floor("−длительность", [-d for d in durs], labels, groups)
        out["duration_stats"] = {"pos_median_s": statistics.median([d for d, y in zip(durs, labels) if y]) if P else None,
                                 "neg_median_s": statistics.median([d for d, y in zip(durs, labels) if not y]) if N else None}
    # пол таймкода: фиксированная относительная позиция
    rel = [t / dur_of[k] for k, g in with_dur if g.label == LABEL_VIOLATION and not g.conflict for t in g.times[:1]]
    if rel:
        frac = statistics.median(rel)
        hits = {f"{w:g}": 0 for w in windows}
        n_ev = 0
        for k, g in with_dur:
            if g.label != LABEL_VIOLATION or not g.times or g.conflict:
                continue
            n_ev += 1
            dt = min(abs(frac * dur_of[k] - t) for t in g.times)
            for w in windows:
                hits[f"{w:g}"] += dt <= w + 1e-9
        out["floors"]["timecode_fixed_position"] = {"fraction": round(frac, 3), "n_events": n_ev,
                                                    "hit": {k: round(v / n_ev, 4) for k, v in hits.items()} if n_ev else {},
                                                    "note": "in-sample медиана позиции — оптимистичный пол"}
    # наши скоры
    res_scores = {}
    for name, sc in scores.items():
        ks = [(k, g) for k, g in with_dur if (k in sc or any(m in sc for m in g.members))]
        if len(ks) < 4:
            continue
        val = [sc.get(k, next((sc[m] for m in g.members if m in sc), 0.0)) for k, g in ks]
        lab = [g.label == LABEL_VIOLATION for _, g in ks]
        dd = [dur_of[k] for k, _ in ks]
        num = [float(re.sub(r"\D", "", k) or 0) for k, _ in ks]
        rho = spearman(val, dd)
        a = auc(val, lab)
        warn = []
        if rho is not None and abs(rho) >= rho_warn:
            warn.append(f"⚠ скор коррелирует с длительностью (ρ={rho:+.2f}) — возможен артефакт пулинга/длины ролика")
        dur_auc = (out["floors"].get("duration_shorter_is_violation") or {}).get("auc")
        if a is not None and dur_auc is not None and a <= dur_auc:
            warn.append(f"⚠ AUC {a:.2f} не выше baseline по длительности {dur_auc:.2f}")
        if a is not None and a < 0.5:
            warn.append("⚠ AUC < 0.5: скор работает в обратную сторону (не годится для отсева видео)")
        rho_id = spearman(val, num)
        if rho_id is not None and abs(rho_id) >= rho_warn:
            warn.append(f"⚠ скор коррелирует с номером файла (ρ={rho_id:+.2f})")
        res_scores[name] = {"n": len(ks), "auc": a, "rho_duration": rho, "auc_within_duration_terciles":
                            stratified_auc(val, lab, dd), "rho_file_number": rho_id, "warnings": warn}
    out["scores"] = res_scores
    return out


def format_sanity(s: dict) -> str:
    f = s["floors"]
    lines = [f"САНИТИ-ПОЛЫ [разведочная оценка] n={s['n']} (с длительностью {s['n_with_duration']})",
             f"  всё violation: recall={f['all_violation']['recall']} precision={f['all_violation']['precision']} "
             f"FP={f['all_violation']['fp']}",
             f"  всё clean:     recall={f['all_clean']['recall']} FN={f['all_clean']['fn']}"]
    d = f.get("duration_shorter_is_violation")
    if d:
        lines.append(f"  длительность (короче → нарушение): AUC={d['auc']}  in-sample FN={d['in_sample']['fn']} "
                     f"FP={d['in_sample']['fp']}  LOGO FN={d['logo']['fn']} FP={d['logo']['fp']}")
    t = f.get("timecode_fixed_position")
    if t:
        lines.append(f"  таймкод = {t['fraction']}×длительность: hit " + " ".join(f"@{k}={v}" for k, v in t["hit"].items()))
    for name, r in s["scores"].items():
        lines.append(f"  скор {name}: AUC={r['auc']} ρ(длительность)={r['rho_duration']} AUC в терцилях длительности="
                     f"{r['auc_within_duration_terciles']} ρ(номер файла)={r['rho_file_number']}")
        lines += [f"    {w}" for w in r["warnings"]]
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Санити-полы и детекторы утечек (длительность/позиция/имя файла)")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--manifest", default="data/manifest.csv", help="L1: duration_s")
    ap.add_argument("--videos", default=None, help="ffprobe, если нет manifest")
    ap.add_argument("--run", action="append", default=[], help="папка eval.run (скоры из preds/scores_raw)")
    ap.add_argument("--scores", action="append", default=[], help="csv:column (напр. OOF baseline L1)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    gt = load_ground_truth(args.labels)
    preds_all: dict[str, dict] = {}
    scores: dict[str, dict[str, float]] = {}
    for r in args.run:
        from .metrics import load_predictions

        if (Path(r) / "preds").exists():
            preds_all.update(load_predictions(Path(r) / "preds")[0])
        scores.update({f"{Path(r).name}/{k}": v for k, v in run_scores(r).items()})
    for spec in args.scores:
        name, sc = load_score_csv(spec)
        scores[name] = sc
    dur = load_durations(args.manifest, args.videos, preds_all)
    res = sanity(gt, dur, scores, cfg)
    out = Path(args.out) if args.out else (Path(args.run[0]) / "sanity.json" if args.run else Path("runs/sanity.json"))
    write_json(out, res)
    print(format_sanity(res))
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
