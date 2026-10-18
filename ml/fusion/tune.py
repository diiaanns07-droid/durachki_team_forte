"""Подбор порогов вердикта по типам на размеченных данных (калибровка ВНУТРИ фолдов).

    python -m ml.fusion.tune --preds preds_raw/ --labels data/labels.csv [--cv 5] [--mode cv|holdout|fit-all]
                             [--target-recall 1.0] [--dt-tol 1.0] [--dt-weight 0.5] [--out thresholds.yaml]

* Единица оценки — уникальное видео (группа): дубликаты с одинаковыми входами (Х-4≡Х-47) — одна группа и
  всегда в одной части сплита. Разные таймкоды в одной группе (57 и 68 с) — конфликт разметки: в штраф по
  таймкоду не идёт, попадает в отчёт («решает человек»).
* Цель подбора: пропусков не больше допустимого (``--target-recall``, по умолчанию 1.0), затем минимум
  ``FP + dt_weight · (TP с |Δt| > dt_tol)``; внутри плато равных решений — точка ближе к нижней границе (``--bias``).
* ``--mode cv`` (по умолчанию): grouped stratified K-fold; пороги подбираются на train-фолдах, предсказания
  на отложенном фолде дают OOF-метрики; итоговые пороги — медиана порогов фолдов (ни один порог не видел
  «свой» тестовый фолд). ``holdout`` — один сплит; ``fit-all`` — на всей выборке (только явно, с пометкой).
* Все 42 строки уже использованы в разведочной оценке: OOF-метрики здесь — НЕ независимый тест.
* Вход ``--preds``: JSON на видео (имя файла = id ролика) — ответ API/UI-отчёт с ``meta.scores_raw``,
  ``AnalysisResult.meta`` или голый ``scores_raw``. Если есть ``inputs`` — слияние пересчитывается текущим
  config.yaml (или вариантом из ``ml.fusion.ab``) без повторного inference.
* Скоры — эвристические (источники коррелированы), не вероятности: пороги — точки отсечения, не «P≥τ».
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from .catalog import resolve_catalog
from .config import load_config, load_thresholds, threshold_for
from .core import fuse_scores

GRID = [round(0.05 + 0.01 * i, 2) for i in range(91)]  # 0.05..0.95
ID_COLS = ("video", "file", "filename", "name", "video_id", "id", "ролик", "видео", "файл", "имя")
LABEL_COLS = ("label", "violation", "verdict", "is_violation", "class", "target", "нарушение", "разметка",
              "метка", "результат")
TIME_COLS = ("timecode", "timecode_sec", "time", "second", "sec", "t", "event_sec", "секунда", "время", "таймкод")
TYPE_COLS = ("type", "type_id", "тип", "violation_type")
NOTE = ("OOF-метрики на разведочных данных (все строки уже использовались в разведке) — оптимистичная оценка, "
        "не независимый тест")


# --------------------------------------------------------------------------- разбор входов


def normalize_id(name: str) -> str:
    """'Х-3.mov' (кириллица) / 'x_3' / 'X–3' → 'X-3'."""
    s = Path(str(name).strip()).name
    s = re.sub(r"\.(mov|mp4|avi|mkv|json)$", "", s, flags=re.IGNORECASE)
    s = s.replace("Х", "X").replace("х", "x").upper()
    s = re.sub(r"[\s_–—]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")


def parse_label(value: Any) -> Optional[bool]:
    s = str(value or "").strip().lower()
    if not s:
        return None
    if "без" in s or s in {"0", "no", "нет", "false", "no_violation", "clean", "чисто", "норма", "ok"}:
        return False
    if s in {"1", "yes", "да", "true", "violation"} or "наруш" in s or "violation" in s:
        return True
    return None


def parse_seconds(value: Any) -> Optional[float]:
    s = str(value or "").strip().lower().replace(",", ".")
    m = re.search(r"(\d+):(\d+(?:\.\d+)?)", s)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    m = re.search(r"\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


@dataclass
class Label:
    video: str
    positive: bool
    second: Optional[float] = None
    type_id: Optional[str] = None


def read_labels(path: str | Path) -> dict[str, list[Label]]:
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8", errors="replace")
    for enc in ("utf-8-sig", "cp1251"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.reader(io.StringIO(text), dialect))
    if not rows:
        return {}
    header = [h.strip().lower() for h in rows[0]]

    def col(names: Sequence[str]) -> Optional[int]:
        for n in names:
            if n in header:
                return header.index(n)
        return None

    has_header = any(h in header for h in ID_COLS + LABEL_COLS + TIME_COLS)
    i_id = col(ID_COLS) if has_header else None
    i_lab = col(LABEL_COLS) if has_header else None
    i_t = col(TIME_COLS) if has_header else None
    i_type = col(TYPE_COLS) if has_header else None
    i_id = 0 if i_id is None else i_id
    i_lab = (1 if len(header) > 1 else 0) if i_lab is None else i_lab
    out: dict[str, list[Label]] = {}
    for row in rows[1:] if has_header else rows:
        if not row or len(row) <= max(i_id, i_lab):
            continue
        pos = parse_label(row[i_lab])
        if pos is None:
            continue
        sec = parse_seconds(row[i_t]) if i_t is not None and i_t < len(row) else None
        if sec is None and pos:
            sec = parse_seconds(re.sub(r"^\D*", "", row[i_lab])) if re.search(r"\d", row[i_lab]) else None
        typ = row[i_type].strip().upper() if i_type is not None and i_type < len(row) and row[i_type].strip() else None
        vid = normalize_id(row[i_id])
        out.setdefault(vid, []).append(Label(vid, pos, sec if pos else None, typ if pos else None))
    return out


def _find_scores_raw(obj: Any) -> Optional[dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    if "candidates" in obj or "inputs" in obj:
        return obj
    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else obj
    raw = meta.get("scores_raw") if isinstance(meta, dict) else None
    return raw if isinstance(raw, dict) else None


@dataclass
class VideoPred:
    video: str
    raw: dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""
    pipeline: dict[str, Any] = field(default_factory=dict)   # timings/VRAM из meta A03 (если есть)


Cand = tuple  # (type_id, score, t_peak, t_start)


def _pipeline_meta(obj: Any) -> dict[str, Any]:
    meta = obj.get("meta") if isinstance(obj, dict) and isinstance(obj.get("meta"), dict) else {}
    out: dict[str, Any] = {}
    timings = meta.get("timings") if isinstance(meta.get("timings"), dict) else {}
    if "total" in timings:
        out["total_s"] = timings["total"]
    vram = [st.get("vram_peak_mb") for st in (meta.get("stages") or {}).values()
            if isinstance(st, dict) and st.get("vram_peak_mb") is not None] if isinstance(meta.get("stages"), dict) else []
    if vram:
        out["vram_peak_mb"] = max(vram)
    return out


def load_preds(preds_dir: str | Path) -> dict[str, VideoPred]:
    out: dict[str, VideoPred] = {}
    for p in sorted(Path(preds_dir).glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        raw = _find_scores_raw(obj)
        if raw is None:
            continue
        inputs = raw.get("inputs")
        fp_src = json.dumps(inputs if inputs else raw.get("candidates"), sort_keys=True, ensure_ascii=False)
        vid = normalize_id(p.stem)
        out[vid] = VideoPred(vid, raw, hashlib.md5(fp_src.encode("utf-8")).hexdigest(), _pipeline_meta(obj))
    return out


def candidates_for(pred: VideoPred, cfg: dict[str, Any], types: dict[str, Any]) -> tuple[list[Cand], float]:
    """Кандидаты, прошедшие гейт (без порогов), и время пересчёта слияния, с."""
    raw = pred.raw
    inputs = raw.get("inputs")
    t0 = time.perf_counter()
    cands: list[Cand] = []
    if isinstance(inputs, dict):
        outcome = fuse_scores(inputs.get("events") or [], inputs.get("risk") or [], float(raw.get("duration") or 0.0),
                              types, cfg, {"default": 0.0, "per_type": {}})
        for c in outcome.candidates:
            if not c.gate and c.suppressed_by is None and c.confidence > 0:
                cands.append((c.type_id, c.confidence, c.t_peak, c.t_start))
    else:
        for c in raw.get("candidates") or []:
            if not c.get("gate") and not c.get("suppressed_by") and float(c.get("confidence", 0)) > 0:
                tp = float(c.get("t_peak", 0.0))
                cands.append((str(c["type_id"]), float(c["confidence"]), tp, float(c.get("t_start", tp))))
    return cands, time.perf_counter() - t0


# --------------------------------------------------------------------------- оценка и подбор


@dataclass
class Sample:
    video: str
    positive: bool
    seconds: list[float]
    cands: list[Cand]
    group: str
    conflict: bool = False          # разные таймкоды у дубликатов — решает человек
    type_id: Optional[str] = None   # тип из разметки (если есть колонка type)
    members: list[str] = field(default_factory=list)


def _tp(c: Cand) -> float:
    return float(c[2])


def predict(s: Sample, thr: dict[str, float], default: float) -> list[Cand]:
    return [c for c in s.cands if c[1] >= thr.get(c[0], default)]


def best_dt(hits: Sequence[Cand], seconds: Sequence[float], policy: str = "peak") -> Optional[float]:
    """min |Δt| между таймкодом нарушения (peak — t_peak, onset — t_start) и размеченной секундой."""
    if not hits or not seconds:
        return None
    idx = 3 if policy == "onset" else 2
    return min(abs(float(h[idx if len(h) > idx else 2]) - sec) for h in hits for sec in seconds)


def evaluate(samples: Sequence[Sample], thr: dict[str, float], default: float, dt_tol: float = 1.0,
             policy: str = "peak") -> dict[str, Any]:
    tp = fn = fp = tn = tc_hit = tc_total = type_hit = type_total = 0
    missed, false_alarms, dt_misses, conflicts = [], [], [], []
    for s in samples:
        hits = predict(s, thr, default)
        if s.positive:
            if hits:
                tp += 1
                if s.conflict:
                    conflicts.append(s.video)
                elif s.seconds:
                    tc_total += 1
                    dt = best_dt(hits, s.seconds, policy)
                    if dt is not None and dt <= dt_tol:
                        tc_hit += 1
                    else:
                        dt_misses.append(s.video)
                if s.type_id:
                    type_total += 1
                    type_hit += int(any(h[0] == s.type_id for h in hits))
            else:
                fn += 1
                missed.append(s.video)
        elif hits:
            fp += 1
            false_alarms.append(s.video)
        else:
            tn += 1
    pos, neg = tp + fn, fp + tn
    return {
        "n": len(samples), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": round(tp / pos, 4) if pos else None,
        "fp_rate": round(fp / neg, 4) if neg else None,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "timecode_hit_rate": round(tc_hit / tc_total, 4) if tc_total else None,
        "timecode_misses": len(dt_misses),
        "type_accuracy": round(type_hit / type_total, 4) if type_total else None,
        "missed": missed, "false_alarms": false_alarms, "dt_misses": dt_misses, "timecode_conflicts": conflicts,
    }


def _cost(samples: Sequence[Sample], thr: dict[str, float], default: float, allowed_fn: int, dt_tol: float,
          dt_weight: float) -> tuple[int, float]:
    m = evaluate(samples, thr, default, dt_tol)
    return (max(0, m["fn"] - allowed_fn), round(m["fp"] + dt_weight * m["timecode_misses"], 6))


def fit(samples: Sequence[Sample], start: dict[str, float], default: float, target_recall: float = 1.0,
        bias: float = 0.35, passes: int = 4, dt_tol: float = 1.0, dt_weight: float = 0.5) -> dict[str, float]:
    """Покоординатный спуск по порогам типов: (пропуски сверх допустимых, FP + штраф Δt) → min; плато — bias."""
    n_pos = sum(1 for s in samples if s.positive)
    allowed_fn = int(math.floor((1.0 - target_recall) * n_pos + 1e-9))
    types = sorted({c[0] for s in samples for c in s.cands})
    thr = dict(start)
    for _ in range(passes):
        changed = False
        for t in types:
            costs = []
            for g in GRID:
                trial = dict(thr)
                trial[t] = g
                costs.append((_cost(samples, trial, default, allowed_fn, dt_tol, dt_weight), g))
            best = min(c for c, _ in costs)
            plateau = [g for c, g in costs if c == best]
            run = [plateau[0]]
            for g in plateau[1:]:
                if abs(g - run[-1] - 0.01) < 1e-6:
                    run.append(g)
                else:
                    break
            choice = run[int(round(bias * (len(run) - 1)))]
            if abs(thr.get(t, default) - choice) > 1e-9:
                thr[t] = choice
                changed = True
        if not changed:
            break
    return thr


def _groups(samples: Sequence[Sample]) -> dict[str, list[Sample]]:
    groups: dict[str, list[Sample]] = {}
    for s in samples:
        groups.setdefault(s.group, []).append(s)
    return groups


def split_groups(samples: Sequence[Sample], holdout: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    """Стратифицированный по классу сплит групп (дубли — в одной части)."""
    rng = random.Random(seed)
    groups = _groups(samples)
    train, hold = [], []
    for positive in (True, False):
        keys = sorted(k for k, g in groups.items() if any(x.positive for x in g) == positive)
        rng.shuffle(keys)
        n_hold = int(round(holdout * len(keys)))
        if holdout > 0 and len(keys) >= 3:
            n_hold = max(1, n_hold)
        for i, k in enumerate(keys):
            (hold if i < n_hold else train).extend(groups[k])
    return train, hold


def kfold(samples: Sequence[Sample], k: int, seed: int) -> list[tuple[list[Sample], list[Sample]]]:
    rng = random.Random(seed)
    groups = _groups(samples)
    folds: list[list[str]] = [[] for _ in range(k)]
    for positive in (True, False):
        keys = sorted(g for g, v in groups.items() if any(x.positive for x in v) == positive)
        rng.shuffle(keys)
        for i, key in enumerate(keys):
            folds[i % k].append(key)
    out = []
    for i in range(k):
        test_keys = set(folds[i])
        train = [s for g, v in groups.items() if g not in test_keys for s in v]
        test = [s for g in folds[i] for s in groups[g]]
        out.append((train, test))
    return out


def cross_validate(samples: Sequence[Sample], start: dict[str, float], default: float, k: int, seed: int,
                   **fit_kw: Any) -> tuple[dict[str, Any], list[dict[str, float]]]:
    """OOF: порог каждого фолда подобран только на его train-части."""
    oof: dict[str, list[Cand]] = {}
    fold_thr: list[dict[str, float]] = []
    tested: list[Sample] = []
    for train, test in kfold(samples, k, seed):
        thr = fit(train, start, default, **fit_kw)
        fold_thr.append(thr)
        for s in test:
            oof[s.video] = predict(s, thr, default)
            tested.append(s)
    # OOF-метрики: подставляем предсказания фолдов как «кандидатов с порогом 0»
    frozen = [Sample(s.video, s.positive, s.seconds, oof[s.video], s.group, s.conflict, s.type_id) for s in tested]
    dt_tol = float(fit_kw.get("dt_tol", 1.0))
    metrics = evaluate(frozen, {}, 0.0, dt_tol)
    metrics["timecode_hit_rate_onset"] = evaluate(frozen, {}, 0.0, dt_tol, policy="onset")["timecode_hit_rate"]
    return metrics, fold_thr


def median_thresholds(fold_thr: Sequence[dict[str, float]], start: dict[str, float]) -> dict[str, float]:
    keys = sorted({k for t in fold_thr for k in t} | set(start))
    return {k: round(statistics.median([t.get(k, start.get(k, 0.5)) for t in fold_thr]), 2) for k in keys}


def build_samples(preds: dict[str, VideoPred], labels: dict[str, list[Label]], cfg: dict[str, Any],
                  types: dict[str, Any]) -> tuple[list[Sample], list[str], list[float]]:
    """Сэмпл = уникальное видео (группа одинаковых входов). → (samples, warnings, время слияния по видео)."""
    warnings: list[str] = []
    by_group: dict[str, list[tuple[str, VideoPred, list[Label]]]] = {}
    for vid, labs in sorted(labels.items()):
        pred = preds.get(vid)
        if pred is None:
            warnings.append(f"нет предсказаний для {vid}")
            continue
        by_group.setdefault(pred.fingerprint or vid, []).append((vid, pred, labs))
    for vid in sorted(set(preds) - set(labels)):
        warnings.append(f"нет разметки для {vid}")
    samples, latencies = [], []
    for group, members in sorted(by_group.items()):
        vid, pred, _ = members[0]
        cands, dt = candidates_for(pred, cfg, types)
        latencies.append(dt)
        labs = [lab for _, _, ls in members for lab in ls]
        positive = any(lab.positive for lab in labs)
        secs = sorted({lab.second for lab in labs if lab.positive and lab.second is not None})
        conflict = len(secs) > 1 and max(secs) - min(secs) > 1.0
        if conflict:
            warnings.append(f"конфликт таймкодов в группе {[m[0] for m in members]}: {secs} — решает человек")
        typ = next((lab.type_id for lab in labs if lab.type_id), None)
        samples.append(Sample(vid, positive, secs, cands, group, conflict, typ, [m[0] for m in members]))
    return samples, warnings, latencies


# --------------------------------------------------------------------------- CLI


def tune(samples: Sequence[Sample], start: dict[str, float], default: float, args: argparse.Namespace
         ) -> tuple[dict[str, float], dict[str, Any]]:
    fit_kw = dict(target_recall=args.target_recall, bias=args.bias, dt_tol=args.dt_tol, dt_weight=args.dt_weight)
    report: dict[str, Any] = {"note": NOTE, "n_groups": len(samples),
                              "baseline_fixed_thresholds": evaluate(samples, start, default, args.dt_tol)}
    mode = "fit-all" if args.fit_all else args.mode
    if mode == "cv":
        k = max(2, min(args.cv or 5, len(samples)))
        oof, fold_thr = cross_validate(samples, start, default, k, args.seed, **fit_kw)
        thr = median_thresholds(fold_thr, start)
        report["cv"] = {"k": k, "oof": oof, "fold_thresholds": fold_thr}
        report["train"] = evaluate(samples, thr, default, args.dt_tol)   # на всех — справочно, оптимистично
    elif mode == "holdout":
        train, hold = split_groups(samples, args.holdout, args.seed)
        thr = fit(train, start, default, **fit_kw)
        report["train"] = evaluate(train, thr, default, args.dt_tol)
        report["holdout"] = evaluate(hold, thr, default, args.dt_tol) if hold else None
        report["train_videos"] = sorted(s.video for s in train)
        report["holdout_videos"] = sorted(s.video for s in hold)
        if args.cv and args.cv >= 2:
            oof, fold_thr = cross_validate(samples, start, default, args.cv, args.seed, **fit_kw)
            report["cv"] = {"k": args.cv, "oof": oof, "fold_thresholds": fold_thr}
    else:
        thr = fit(samples, start, default, **fit_kw)
        report["train"] = evaluate(samples, thr, default, args.dt_tol)
        report["warning"] = "пороги подобраны на ВСЕЙ выборке: метрики оптимистичны"
    report["mode"] = mode
    return thr, report


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--preds", required=True, help="папка с JSON-ответами/scores_raw по видео")
    ap.add_argument("--labels", required=True, help="CSV разметки (id ролика, нарушение/без, секунда[, тип])")
    ap.add_argument("--mode", choices=("cv", "holdout"), default="cv", help="калибровка внутри фолдов (cv) или сплит")
    ap.add_argument("--cv", type=int, default=5, help="число фолдов grouped K-fold")
    ap.add_argument("--holdout", type=float, default=0.3, help="доля групп в отложенной части (mode=holdout)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--target-recall", type=float, default=1.0)
    ap.add_argument("--bias", type=float, default=0.35, help="позиция внутри плато (0 — макс. запас полноты)")
    ap.add_argument("--dt-tol", type=float, default=1.0, help="допуск таймкода |Δt|, с")
    ap.add_argument("--dt-weight", type=float, default=0.5, help="штраф за TP с |Δt| > dt-tol (в единицах FP)")
    ap.add_argument("--fit-all", action="store_true", help="подобрать на ВСЕЙ выборке (без честной оценки)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Подбор порогов ml.fusion (калибровка внутри фолдов)")
    add_args(ap)
    ap.add_argument("--out", default="", help="куда записать YAML (по умолчанию — только stdout)")
    ap.add_argument("--report", default="", help="куда записать JSON-отчёт")
    args = ap.parse_args(argv)

    cfg = load_config()
    current = load_thresholds()
    default = float(current.get("default", 0.45))
    start = {t: threshold_for(current, t) for t in (current.get("per_type") or {})}
    types, _ = resolve_catalog(None)
    samples, warnings, _ = build_samples(load_preds(args.preds), read_labels(args.labels), cfg, types)
    if not samples:
        print("нет пересечения предсказаний и разметки", file=sys.stderr)
        return 2
    thr, report = tune(samples, start, default, args)
    report["warnings"] = warnings
    per_type = {t: round(v, 2) for t, v in sorted({**start, **thr}.items())}
    oof = (report.get("cv") or {}).get("oof") or {}
    doc = {
        "version": 1,
        "tuned": True,
        "default": default,
        "per_type": per_type,
        "tuned_on": {
            "mode": report["mode"], "seed": args.seed, "k": (report.get("cv") or {}).get("k"),
            "target_recall": args.target_recall, "bias": args.bias, "dt_tol": args.dt_tol,
            "dt_weight": args.dt_weight, "n_groups": len(samples),
            "oof_metrics": {k: v for k, v in oof.items() if not isinstance(v, list)},
            "holdout_metrics": {k: v for k, v in (report.get("holdout") or {}).items() if not isinstance(v, list)},
            "note": NOTE,
            "score_note": "эвристические скоры коррелированных источников, не вероятности",
        },
    }
    try:
        import yaml  # lazy

        text = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False)
    except ImportError:
        text = json.dumps(doc, ensure_ascii=False, indent=2)
    print(text)
    print(json.dumps({k: report[k] for k in ("train", "holdout", "cv") if report.get(k)}, ensure_ascii=False,
                     indent=2), file=sys.stderr)
    if args.out:
        Path(args.out).write_text("# Сгенерировано python -m ml.fusion.tune\n" + text, encoding="utf-8")
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
