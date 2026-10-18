"""Одна команда замера для L2 (реальные видео + GPU): сравнение ≤ 3 заранее описанных конфигураций.

    python -m ml.temporal.bench --experiments E0,E1,E2 [--folds 5 --repeats 3 --inner 3]
                                [--latency-videos data/raw --n-latency 5] [--out reports/temporal_bench.json]

Repeated stratified group k-fold (дубликаты — одна группа), пороги и null — внутри train-фолдов. Для каждой
конфигурации (среднее ± std по повторам): FP/FN при порогах внутренних фолдов, AUC по режимам пулинга
(max — старый, для сравнения), попадание таймкода ±1/2/3 с по политикам, recall кандидатов ±1.5 с при
≤3/5/8 на видео, сторож длительности, latency модели на видео, пик VRAM (CUDA). Опционально — latency
полного score()/propose() на реальных роликах (с извлечением признаков). Оценка разведочная.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .config import _deep_merge, load_config, reports_dir
from .cv import EXPLORATORY, POOLINGS, nested_cv

log = logging.getLogger("sentinel.temporal")


def _ms(vals: Sequence[Optional[float]]) -> dict:
    v = [float(x) for x in vals if x is not None and np.isfinite(x)]
    return {"mean": round(float(np.mean(v)), 4) if v else None, "std": round(float(np.std(v)), 4) if v else None,
            "n": len(v)}


def _vram_reset() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _vram_peak_mb() -> Optional[float]:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    except ImportError:
        pass
    return None


def summarize_repeats(reps: list[dict]) -> dict:
    out: dict[str, Any] = {"n_repeats": len(reps)}
    out["auc_by_pooling"] = {m: _ms([r["by_pooling"][m]["auc"] for r in reps]) for m in POOLINGS}
    out["fp"] = _ms([r["nested"]["fp"] for r in reps])
    out["fn"] = _ms([r["nested"]["fn"] for r in reps])
    out["recall"] = _ms([r["nested"]["recall"] for r in reps])
    out["fpr"] = _ms([r["nested"]["fpr"] for r in reps])
    pols = reps[0]["timecode_by_policy"].keys()
    out["timecode"] = {p: {k: _ms([r["timecode_by_policy"][p][k] for r in reps])
                           for k in reps[0]["timecode_by_policy"][p]} for p in pols}
    out["candidates"] = {m: {k: _ms([r["candidates"][m][k] for r in reps])
                             for k in reps[0]["candidates"][m] if k.startswith("recall@") or k == "mean_n"}
                         for m in reps[0]["candidates"]}
    out["duration_watchdog"] = {"spearman_neg": _ms([r["duration_watchdog"]["spearman_neg"] for r in reps]),
                                "warn_any": any(r["duration_watchdog"]["warn"] for r in reps),
                                "duration_only_auc": reps[0]["duration_watchdog"]["duration_only_auc"]}
    out["latency_model_ms_per_video"] = _ms([r.get("latency_model_ms_per_video") for r in reps])
    out["elapsed_s"] = round(float(sum(r["elapsed_s"] for r in reps)), 1)
    return out


def run_bench(cfg: dict, experiments: Sequence[str], folds: int, repeats: int, inner: int, seed: int,
              samples=None, bank=None, index: Optional[str] = None, epochs: Optional[int] = None) -> dict:
    if samples is None or bank is None:
        from .data_index import bank_for_index, load_samples

        samples, idx = load_samples(cfg, index)
        bank = bank_for_index(cfg, idx)
    exps = cfg.get("experiments", {})
    report: dict[str, Any] = {"evaluation_status": EXPLORATORY, "protocol": "repeated stratified group k-fold, "
                              "пороги/null — внутренним CV на train каждого фолда",
                              "folds": folds, "repeats": repeats, "inner": inner, "seed": seed,
                              "n_groups": len(samples), "n_pos": int(sum(s.label for s in samples)),
                              "type_eval": None if not any(s.type_id for s in samples) else "см. per-experiment",
                              "experiments": {}}
    if report["type_eval"] is None:
        report["type_note"] = "нет подтверждённых меток типа — точность типа не измеряется (голова типа не учится)"
    for name in experiments:
        if name not in exps:
            raise KeyError(f"нет конфигурации {name} в config.yaml: experiments")
        e = exps[name]
        ecfg = _deep_merge(cfg, e.get("overrides") or {})
        kind = e.get("kind", "baseline")
        if kind == "temporal":
            try:
                import torch  # noqa: F401
            except ImportError:
                report["experiments"][name] = {"skipped": "torch не установлен"}
                continue
        _vram_reset()
        reps = []
        for r in range(repeats):
            ep = epochs or (ecfg["cv"].get("epochs") if kind == "temporal" else None)
            reps.append(nested_cv(samples, bank, ecfg, kind, folds, inner, seed + 1000 * r, epochs=ep))
            log.info("bench %s: повтор %d/%d готов", name, r + 1, repeats)
        report["experiments"][name] = {"description": e.get("description", ""), "kind": kind,
                                       "overrides": e.get("overrides") or {}, "summary": summarize_repeats(reps),
                                       "vram_peak_mb_train": _vram_peak_mb(),
                                       "per_video_last_repeat": reps[-1]["per_video"]}
    return report


def latency_probe(videos_dir: str, n: int, cfg: dict) -> dict:
    """Полный путь score()/propose() на реальных роликах (извлечение признаков + модель), с пиком VRAM."""
    from . import TemporalUnavailable, propose, score, status
    from ._video import load_clip
    from .labels import find_videos

    vids = list(find_videos(videos_dir).values())[: max(0, n)]
    rows = []
    for p in vids:
        clip = load_clip(str(p), sample_fps=5.0)
        _vram_reset()
        t0 = time.perf_counter()
        try:
            score(clip)
        except TemporalUnavailable:  # статус честно попадёт в строку (no_checkpoint и т.п.)
            pass
        t1 = time.perf_counter()
        propose(clip)
        t2 = time.perf_counter()
        rows.append({"duration_s": round(float(getattr(clip, "duration", 0.0)), 1),
                     "score_s": round(t1 - t0, 3), "propose_s_after_score": round(t2 - t1, 3),
                     "status": status().get("status"), "vram_peak_mb": _vram_peak_mb()})
    return {"n": len(rows), "rows": rows,
            "note": "первый вызов включает загрузку модели; propose после score использует кэш признаков"}


def _table(rep: dict) -> str:
    lines = ["| конфиг | AUC topk_null | AUC max (старый) | FP | FN | hit@2s peak | hit@2s onset | cand recall@8 hybrid | ρ(скор,длит.)neg | мс/видео |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, e in rep["experiments"].items():
        if "summary" not in e:
            lines.append(f"| {name} | {e.get('skipped')} |")
            continue
        s = e["summary"]
        f = lambda d: "—" if d["mean"] is None else f"{d['mean']:.3f}±{d['std']:.3f}"  # noqa: E731
        tc = s["timecode"]
        lines.append(" | ".join([
            f"| {name}", f(s["auc_by_pooling"]["topk_null"]), f(s["auc_by_pooling"]["max"]), f(s["fp"]), f(s["fn"]),
            f(tc.get("peak", {}).get("hit@2s", {"mean": None})), f(tc.get("onset", {}).get("hit@2s", {"mean": None})),
            f(s["candidates"]["hybrid"]["recall@8"]), f(s["duration_watchdog"]["spearman_neg"]),
            f(s["latency_model_ms_per_video"])]) + " |")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ml.temporal: замер конфигураций для L2 (разведочная оценка)")
    ap.add_argument("--experiments", default="E0,E1,E2")
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--inner", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--index", default=None)
    ap.add_argument("--latency-videos", default=None, help="каталог видео для замера полного score()/propose()")
    ap.add_argument("--n-latency", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    b = cfg.get("bench", {})
    rep = run_bench(cfg, [x.strip() for x in a.experiments.split(",") if x.strip()],
                    int(a.folds or b.get("folds", 5)), int(a.repeats or b.get("repeats", 3)),
                    int(a.inner or b.get("inner", 3)), int(cfg["train"]["seed"] if a.seed is None else a.seed),
                    index=a.index, epochs=a.epochs)
    if a.latency_videos:
        rep["latency_full"] = latency_probe(a.latency_videos, a.n_latency, cfg)
    out = Path(a.out) if a.out else reports_dir(cfg) / "temporal_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(_table(rep))
    print(f"-> {out}  [{EXPLORATORY}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
