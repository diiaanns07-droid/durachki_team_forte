"""Замер для L2: сравнение режимов пайплайна на размеченных видео — ошибки, таймкоды, время, память.

    python -m backend.pipeline compare data/videos --labels data/labels.csv \
           --modes baseline,refine --out reports/compare_h1 [--deadline 240] [--cache off]

Порядок намеренно двухфазный: (1) все режимы прогоняются на всех видео, предсказания пишутся на
диск; (2) только после этого читается разметка и считаются метрики — разметка в инференс не попадает.
Режим = YAML-оверрайд конфига из backend/pipeline/profiles/<режим>.yaml (или путь к YAML).
Кэш по умолчанию выключен: иначе второй режим получит чужие CV-стадии даром и время несравнимо.

Метрики (разведочная оценка на тех же роликах — НЕ независимый тест):
* видео: TP/FP/TN/FN, failed отдельно (failed на нарушении = пропуск; failed никогда не no_violation),
  precision / recall / specificity;
* таймкод (только TP с известной секундой; при конфликте дубликатов — ближайшая из t_all):
  |Δ| t_peak и |Δ| onset (t_start), hit@1/2/5 с, медиана;
* тип: только распределение предсказанных type_id (GT-типов нет);
* время: p50/p90/max на видео + средние по стадиям; память: максимум VRAM-пика по стадиям;
* полнота: минимум meta.coverage.ratio (ядро), число неполных/деградированных.
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .config import PIPELINE_DIR, deep_merge
from .media import VIDEO_EXTS, normalize_video_name

PROFILES_DIR = PIPELINE_DIR / "profiles"


def load_profile(mode: str) -> dict:
    import yaml  # noqa: WPS433

    path = Path(mode) if mode.endswith((".yaml", ".yml")) else PROFILES_DIR / f"{mode}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"нет профиля режима {mode!r}: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def list_videos(folder: str | Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def summarize(res: Any, name: str, file: str, wall_s: float) -> dict:
    meta = res.meta or {}
    stages = meta.get("stages", {}) or {}
    vs = sorted(res.violations, key=lambda v: -float(v.confidence or 0.0))
    return {
        "name": name, "file": file, "status": res.status, "verdict": res.verdict,
        "complete": meta.get("complete"), "degraded": bool(meta.get("degraded")),
        "cancelled": bool(meta.get("cancelled")), "error": meta.get("error", ""),
        "violations": [{"type_id": v.type_id, "timecode_sec": v.timecode_sec, "t_start": v.t_start,
                        "t_end": v.t_end, "confidence": v.confidence} for v in vs],
        "wall_s": round(wall_s, 3), "total_s": (meta.get("timings") or {}).get("total"),
        "stage_s": {k: v.get("seconds") for k, v in stages.items()},
        "stage_status": {k: v.get("status") for k, v in stages.items()},
        "vram_mb": {k: v.get("vram_peak_mb") for k, v in stages.items() if v.get("vram_peak_mb") is not None},
        "coverage": meta.get("coverage", {}), "sample_fps": meta.get("sample_fps"),
        "n_sampled": (meta.get("video") or {}).get("n_sampled"), "duration": meta.get("duration_sec"),
        "sha256": (meta.get("video") or {}).get("sha256"), "warnings": len(meta.get("warnings") or []),
        "refine": meta.get("refine"), "modes": meta.get("modes"), "device": meta.get("device"),
    }


def run_mode(mode: str, videos: list[Path], out_dir: Path, deadline_s: float, cache_mode: str,
             extra: Optional[Mapping] = None, log: Any = print) -> list[dict]:
    from . import serde
    from .orchestrator import run

    over = deep_merge(load_profile(mode), {"cache": {"mode": cache_mode}, "partial": {"enabled": False}})
    if extra:
        over = deep_merge(over, extra)
    preds = out_dir / mode / "preds"
    preds.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, p in enumerate(videos, 1):
        name = normalize_video_name(p.name)
        t0 = time.perf_counter()
        res = run(str(p), f"{mode}-{name}", None, deadline_s, overrides=over)
        wall = time.perf_counter() - t0
        serde.write_json(preds / f"{name}.json", res, indent=1)
        row = summarize(res, name, p.name, wall)
        rows.append(row)
        top = row["violations"][0] if row["violations"] else None
        log(f"[{mode} {k}/{len(videos)}] {name}: {row['status']} {row['verdict']} "
            f"{(top['type_id'] + '@' + str(top['timecode_sec'])) if top else ''} {wall:.1f} с")
    with open(out_dir / mode / "rows.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


# ---------------------------------------------------------------- разметка (только после прогонов)

def _times(*cells: Any) -> list[float]:
    out: list[float] = []
    for cell in cells:
        for tok in re.split(r"[;,|\s]+", str(cell or "").strip()):
            try:
                out.append(float(tok))
            except ValueError:
                continue
    return sorted(set(out))


def load_labels(path: str | Path) -> dict[str, dict]:
    """data/labels.csv (схема eval.labels: video_id,file,label,t_sec,…,t_all,conflict)."""
    labels: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            vid = normalize_video_name(str(row.get("video_id") or row.get("file") or ""))
            lab = str(row.get("label") or "").strip().lower()
            labels[vid] = {"label": lab, "times": _times(row.get("t_sec"), row.get("t_all")),
                           "conflict": str(row.get("conflict") or "").strip().lower() in ("1", "true", "yes")}
    return labels


# ---------------------------------------------------------------- метрики

def _pct(xs: list[float], q: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return round(xs[k], 3)


def _ratio(a: int, b: int) -> Optional[float]:
    return round(a / b, 4) if b else None


def metrics(rows: list[dict], labels: Mapping[str, dict], tols: Iterable[float] = (1.0, 2.0, 5.0)) -> dict:
    tols = list(tols)
    c = Counter()
    errors: dict[str, list[str]] = {"fn": [], "fp": [], "failed": []}
    d_peak: list[float] = []
    d_onset: list[float] = []
    types: Counter = Counter()
    for r in rows:
        lab = labels.get(r["name"])
        if lab is None:
            c["unlabeled"] += 1
            continue
        if lab["label"] not in ("violation", "no_violation"):
            c["excluded_conflict"] += 1
            continue
        c["n"] += 1
        pos = lab["label"] == "violation"
        if r["status"] != "done" or r["verdict"] not in ("violation", "no_violation"):
            c["failed_pos" if pos else "failed_neg"] += 1
            errors["failed"].append(r["name"])
            continue
        pred = r["verdict"] == "violation"
        if pred and r["violations"]:
            types[r["violations"][0]["type_id"] or "?"] += 1
        if pos and pred:
            c["tp"] += 1
            if lab["times"] and r["violations"]:
                d_peak.append(min(abs(v["timecode_sec"] - g) for v in r["violations"] for g in lab["times"]))
                d_onset.append(min(abs(v["t_start"] - g) for v in r["violations"] for g in lab["times"]))
        elif pos:
            c["fn"] += 1
            errors["fn"].append(r["name"])
        elif pred:
            c["fp"] += 1
            errors["fp"].append(r["name"])
        else:
            c["tn"] += 1
    wall = [r["wall_s"] for r in rows if r.get("wall_s") is not None]
    stage_s: dict[str, list[float]] = {}
    vram: dict[str, float] = {}
    cov_min: list[float] = []
    for r in rows:
        for k, v in (r.get("stage_s") or {}).items():
            if v is not None:
                stage_s.setdefault(k, []).append(float(v))
        for k, v in (r.get("vram_mb") or {}).items():
            vram[k] = max(vram.get(k, 0.0), float(v))
        cov = r.get("coverage") if isinstance(r.get("coverage"), dict) else {}
        if cov.get("ratio") is not None:
            cov_min.append(float(cov["ratio"]))
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    return {
        "video": {"n": c["n"], "tp": tp, "fp": fp, "tn": tn, "fn": fn, "failed_on_violation": c["failed_pos"],
                  "failed_on_clean": c["failed_neg"], "excluded_conflict": c["excluded_conflict"],
                  "unlabeled": c["unlabeled"],
                  "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn + c["failed_pos"]),
                  "specificity": _ratio(tn, tn + fp + c["failed_neg"])},
        "timecode": {"n": len(d_peak), "median_abs_peak_s": _pct(d_peak, 0.5), "median_abs_onset_s": _pct(d_onset, 0.5),
                     **{f"hit_peak@{t:g}s": _ratio(sum(d <= t for d in d_peak), len(d_peak)) for t in tols},
                     **{f"hit_onset@{t:g}s": _ratio(sum(d <= t for d in d_onset), len(d_onset)) for t in tols}},
        "types_predicted": dict(types.most_common()),
        "latency": {"p50_s": _pct(wall, 0.5), "p90_s": _pct(wall, 0.9), "max_s": _pct(wall, 1.0),
                    "stage_mean_s": {k: round(statistics.fmean(v), 3) for k, v in sorted(stage_s.items())}},
        "vram_peak_mb": {k: round(v, 1) for k, v in sorted(vram.items())},
        "completeness": {"incomplete": sum(1 for r in rows if r.get("complete") is False),
                         "degraded": sum(1 for r in rows if r.get("degraded")),
                         "min_coverage_ratio": round(min(cov_min), 4) if cov_min else None},
        "errors": errors,
    }


def render_md(results: Mapping[str, dict], meta: Mapping[str, Any]) -> str:
    modes = list(results)
    lines = [f"# Сравнение режимов пайплайна: {' vs '.join(modes)}", "",
             f"Видео: {meta.get('n_videos')}, дедлайн {meta.get('deadline_s')} с, кэш: {meta.get('cache')}, "
             f"устройство: {meta.get('device')}. **Разведочная оценка на размеченных роликах — не независимый тест.**",
             "", "| метрика | " + " | ".join(modes) + " |", "|---|" + "---|" * len(modes)]

    def row(label: str, getter) -> None:
        vals = []
        for m in modes:
            try:
                v = getter(results[m])
            except (KeyError, TypeError):
                v = None
            vals.append("—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    for key in ("tp", "fp", "tn", "fn", "failed_on_violation", "failed_on_clean", "precision", "recall", "specificity"):
        row(f"видео: {key}", lambda r, k=key: r["video"][k])
    for key in ("n", "median_abs_peak_s", "median_abs_onset_s", "hit_peak@1s", "hit_peak@2s", "hit_peak@5s",
                "hit_onset@1s", "hit_onset@2s", "hit_onset@5s"):
        row(f"таймкод: {key}", lambda r, k=key: r["timecode"].get(k))
    for key in ("p50_s", "p90_s", "max_s"):
        row(f"время на видео: {key}", lambda r, k=key: r["latency"][k])
    stages = sorted({s for r in results.values() for s in r["latency"]["stage_mean_s"]})
    for st in stages:
        row(f"стадия {st}, с (среднее)", lambda r, s=st: r["latency"]["stage_mean_s"].get(s))
    vstages = sorted({s for r in results.values() for s in r["vram_peak_mb"]})
    for st in vstages:
        row(f"VRAM пик {st}, МБ", lambda r, s=st: r["vram_peak_mb"].get(s))
    row("неполных анализов", lambda r: r["completeness"]["incomplete"])
    row("мин. покрытие ядра (ratio)", lambda r: r["completeness"]["min_coverage_ratio"])
    lines += ["", "Предсказанные типы (GT-типов нет — только распределение):", ""]
    for m in modes:
        lines.append(f"* {m}: {results[m]['types_predicted'] or '—'}")
    lines += ["", "Ошибки (id видео):", ""]
    for m in modes:
        e = results[m]["errors"]
        lines.append(f"* {m}: FN={e['fn'] or '—'} FP={e['fp'] or '—'} failed={e['failed'] or '—'}")
    lines += ["", "Решение: режим включается по умолчанию, только если он не увеличивает FN+failed_on_violation, "
              "не увеличивает FP и укладывается во время/VRAM; иначе остаётся выключенным (откат = профиль baseline)."]
    return "\n".join(lines) + "\n"


def emit_budgets(rows: list[dict], path: str | Path, reference_frames: int = 725, safety: float = 1.5) -> dict:
    """budgets_s из реального прогона: p90 секунд на кадр по стадии × reference_frames × safety."""
    per_frame: dict[str, list[float]] = {}
    for r in rows:
        n = r.get("n_sampled") or 0
        if not n or r.get("status") != "done":
            continue
        for st, sec in (r.get("stage_s") or {}).items():
            if sec is not None and st not in ("total",):
                per_frame.setdefault(st, []).append(float(sec) / n)
    budgets = {st: round((_pct(v, 0.9) or 0.0) * reference_frames * safety, 2) for st, v in sorted(per_frame.items())}
    data = {"budgets_s": budgets, "reference_frames": reference_frames,
            "_source": f"compare: p90 сек/кадр × {reference_frames} × {safety}, видео={len(rows)}"}
    import yaml  # noqa: WPS433

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return data


def compare(videos_dir: str | Path, labels_path: str | Path, modes: list[str], out_dir: str | Path,
            deadline_s: float = 240.0, cache_mode: str = "off", limit: Optional[int] = None,
            tols: Iterable[float] = (1.0, 2.0, 5.0), extra: Optional[Mapping] = None,
            budgets_out: Optional[str | Path] = None, log: Any = print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for m in modes:                         # профили проверяем до долгих прогонов
        load_profile(m)
    videos = list_videos(videos_dir)[: limit or None]
    if not videos:
        raise FileNotFoundError(f"нет видео в {videos_dir}")
    rows_by_mode = {m: run_mode(m, videos, out, deadline_s, cache_mode, extra, log) for m in modes}
    # ---- фаза 2: разметка читается только сейчас
    labels = load_labels(labels_path)
    results = {m: metrics(rows, labels, tols) for m, rows in rows_by_mode.items()}
    devices = sorted({r.get("device") or "?" for rows in rows_by_mode.values() for r in rows})
    meta = {"n_videos": len(videos), "deadline_s": deadline_s, "cache": cache_mode, "device": ",".join(devices),
            "modes": modes, "labels": str(labels_path)}
    (out / "compare.json").write_text(json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    (out / "compare.md").write_text(render_md(results, meta), encoding="utf-8")
    if budgets_out:
        emit_budgets(rows_by_mode[modes[0]], budgets_out)
    return {"meta": meta, "results": results}
