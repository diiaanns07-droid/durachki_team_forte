"""Замер вариантов fusion для L2 (одна команда): что сравниваем → какой выход → какие метрики.

    python -m ml.fusion.ab --preds preds_raw/ --labels data/labels.csv \\
        [--variants ml/fusion/experiments.yaml] [--only baseline,v02_after_window] [--cv 5] \\
        [--out outputs/fusion_ab.json] [--videos data/videos --frames-out outputs/fusion_ab_frames]

* Сравниваем варианты config.yaml (флаги фазы 2) на ОДНИХ И ТЕХ ЖЕ сохранённых входах fusion
  (``meta.scores_raw.inputs`` из прогона пайплайна): inference не повторяется, GPU не нужен.
* Метрики по уникальным видео (дубли — одна группа): FN, FP, полнота, доля FP, попадание таймкода
  |Δt| ≤ dt_tol (политики peak и onset), точность типа (если в разметке есть колонка type);
  при фиксированных порогах thresholds.yaml и OOF с калибровкой порогов внутри фолдов.
  Задержка fusion — замер пересчёта на этой машине; задержка и VRAM пайплайна — из meta A03 в preds_raw
  (fusion сам GPU не использует). Все 42 строки разведочные — OOF не называть независимым тестом.
* ``--videos``: для вариантов с другим evidence.* кадры рендерятся ЛОКАЛЬНО в ``--frames-out`` для просмотра
  человеком (в git и облачные модели не передавать).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

from .catalog import resolve_catalog
from .config import deep_merge, load_config, load_thresholds, threshold_for
from .tune import NOTE, add_args, build_samples, cross_validate, evaluate, load_preds, read_labels

HERE = Path(__file__).resolve().parent
KEY_METRICS = ("fn", "fp", "recall", "fp_rate", "timecode_hit_rate", "type_accuracy")


def load_variants(path: str | Path) -> dict[str, dict[str, Any]]:
    import yaml  # lazy

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {str(k): (v or {}) for k, v in data.items()}


def _strip(m: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in m.items() if not isinstance(v, list)}


def measure(variant: dict[str, Any], preds: dict, labels: dict, args: argparse.Namespace) -> dict[str, Any]:
    cfg = deep_merge(load_config(), variant)
    thresholds = load_thresholds()
    default = float(thresholds.get("default", 0.45))
    start = {t: threshold_for(thresholds, t) for t in (thresholds.get("per_type") or {})}
    types, _ = resolve_catalog(None)
    samples, warnings, lat = build_samples(preds, labels, cfg, types)
    policy = str((cfg.get("timecode") or {}).get("policy", "peak"))
    fixed = evaluate(samples, start, default, args.dt_tol)
    fixed["timecode_hit_rate_onset"] = evaluate(samples, start, default, args.dt_tol, policy="onset")[
        "timecode_hit_rate"]
    out: dict[str, Any] = {"timecode_policy": policy, "fixed_thresholds": fixed, "warnings": warnings,
                           "fusion_latency_ms": {"mean": round(1000 * statistics.mean(lat), 2) if lat else None,
                                                 "max": round(1000 * max(lat), 2) if lat else None}}
    if args.cv and args.cv >= 2 and len(samples) >= args.cv:
        oof, fold_thr = cross_validate(samples, start, default, args.cv, args.seed, target_recall=args.target_recall,
                                       bias=args.bias, dt_tol=args.dt_tol, dt_weight=args.dt_weight)
        out["oof"] = oof
        out["fold_thresholds"] = fold_thr
    return out


def pipeline_stats(preds: dict) -> dict[str, Any]:
    tot = [p.pipeline["total_s"] for p in preds.values() if "total_s" in p.pipeline]
    vram = [p.pipeline["vram_peak_mb"] for p in preds.values() if "vram_peak_mb" in p.pipeline]
    return {"n": len(preds),
            "latency_s": {"mean": round(statistics.mean(tot), 2), "max": round(max(tot), 2)} if tot else None,
            "vram_peak_mb_max": max(vram) if vram else None,
            "note": "из meta A03 в preds_raw (одинаково для всех вариантов: fusion GPU не использует)"}


def render_frames(name: str, variant: dict[str, Any], preds: dict, videos: Path, out_dir: Path) -> list[str]:
    """Кадры-доказательства варианта — локально, для просмотра человеком."""
    import base64

    from .core import fuse
    from .serialize import event_from_dict

    try:
        from ml.ingest import load_video  # type: ignore[import-not-found]  # A04
    except ImportError:
        return ["ml.ingest недоступен — кадры не рендерились"]
    msgs: list[str] = []
    cfg = deep_merge(load_config(), variant)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as fh:
        import yaml  # lazy

        yaml.safe_dump(cfg, fh, allow_unicode=True)
        cfg_path = fh.name
    old = os.environ.get("SENTINEL_FUSION_CONFIG")
    os.environ["SENTINEL_FUSION_CONFIG"] = cfg_path
    try:
        for vid, pred in sorted(preds.items()):
            inputs = pred.raw.get("inputs") or {}
            files = [p for p in videos.glob("*") if p.suffix.lower() in {".mov", ".mp4"}
                     and p.stem.replace("Х", "X").replace("х", "x").upper() == vid]
            if not files or not inputs.get("events"):
                continue
            with load_video(str(files[0])) as clip:
                res = fuse(clip, [event_from_dict(e) for e in inputs["events"]], inputs.get("risk") or [], {},
                           f"ab-{name}-{vid}")
            target = out_dir / name / vid
            target.mkdir(parents=True, exist_ok=True)
            for i, v in enumerate(res.violations):
                for j, f in enumerate(v.evidence_frames):
                    (target / f"{i}_{v.type_id}_{j}_{f.timestamp_sec:.2f}.jpg").write_bytes(
                        base64.b64decode(f.image_base64))
                ov = res.meta["ui"]["evidence_overlays"][i]
                for f in ov.get("frames", []):
                    (target / f"{i}_{v.type_id}_ui_{f.get('role', 'key')}_{f['timestamp_sec']:.2f}.jpg").write_bytes(
                        base64.b64decode(f["image_base64"]))
    finally:
        if old is None:
            os.environ.pop("SENTINEL_FUSION_CONFIG", None)
        else:
            os.environ["SENTINEL_FUSION_CONFIG"] = old
        Path(cfg_path).unlink(missing_ok=True)
    return msgs


def markdown(report: dict[str, Any]) -> str:
    head = "| вариант | FN | FP | полнота | доля FP | таймкод ≤dt | тип | OOF FN | OOF FP | OOF таймкод | fusion мс |"
    lines = [head, "|" + "---|" * 11]
    for name, r in report["variants"].items():
        f, o = r["fixed_thresholds"], r.get("oof") or {}
        lines.append(f"| {name} | {f['fn']} | {f['fp']} | {f['recall']} | {f['fp_rate']} | {f['timecode_hit_rate']} | "
                     f"{f['type_accuracy']} | {o.get('fn', '—')} | {o.get('fp', '—')} | {o.get('timecode_hit_rate', '—')} | "
                     f"{r['fusion_latency_ms']['mean']} |")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Замер вариантов ml.fusion на сохранённых входах (для L2)")
    add_args(ap)
    ap.add_argument("--variants", default=str(HERE / "experiments.yaml"))
    ap.add_argument("--only", default="", help="через запятую: какие варианты мерить")
    ap.add_argument("--out", default="outputs/fusion_ab.json")
    ap.add_argument("--videos", default="", help="папка с видео — отрендерить кадры вариантов с evidence.*")
    ap.add_argument("--frames-out", default="outputs/fusion_ab_frames")
    args = ap.parse_args(argv)

    variants = load_variants(args.variants)
    if args.only:
        wanted = [v.strip() for v in args.only.split(",") if v.strip()]
        variants = {k: v for k, v in variants.items() if k in wanted or k == "baseline"}
    preds, labels = load_preds(args.preds), read_labels(args.labels)
    report: dict[str, Any] = {"note": NOTE, "dt_tol": args.dt_tol, "cv": args.cv, "pipeline": pipeline_stats(preds),
                              "variants": {}}
    for name, over in variants.items():
        report["variants"][name] = measure(over, preds, labels, args)
        if args.videos and "evidence" in over:
            report["variants"][name]["frames"] = render_frames(name, over, preds, Path(args.videos),
                                                               Path(args.frames_out))
    base = report["variants"].get("baseline")
    if base:
        for name, r in report["variants"].items():
            r["delta_vs_baseline"] = {k: (None if r["fixed_thresholds"].get(k) is None
                                          or base["fixed_thresholds"].get(k) is None
                                          else round(r["fixed_thresholds"][k] - base["fixed_thresholds"][k], 4))
                                      for k in KEY_METRICS}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = markdown(report)
    out.with_suffix(".md").write_text(f"{NOTE}\n\n{md}\n", encoding="utf-8")
    print(md)
    print(f"\nотчёт: {out} (+ .md); пайплайн: {json.dumps(report['pipeline'], ensure_ascii=False)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
