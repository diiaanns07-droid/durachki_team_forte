"""Замер ml/ingest для L2: coverage / PTS / latency / память по реальным роликам, несколько вариантов конфига.

    python -m ml.ingest.bench --videos data/videos --labels data/labels.csv --out reports/a04_ingest_bench.json \\
        --variant base --variant skip:decode.skip_nonref=true --variant cv:decode.backend=opencv

Вариант: ``имя[:ключ=значение,ключ=значение]`` — частичный override config.yaml (значения как в YAML).
Только числа и id роликов: никаких кадров, base64 и описаний сцен (результат можно показывать).

Что меряется на ролик и вариант:
  * coverage: complete, coverage, пробелы покрытия (шт., сек), ошибки декодера, битые кадры;
  * PTS: t0, число кадров, шаг (min/median/max/std), VFR, отклонение сэмплов от сетки;
  * точность цепочки времени: |PTS выданного кадра − запрошенное t| на случайных t (max/p95), доля in_gap,
    идемпотентность frame_at(PTS), расхождение сэмпла frame(i) и точного кадра в times[i] (средняя |Δ|, 0..255);
  * latency: загрузка (probe/decode/total), frame() промах/попадание, frames_at(3), coarse_fine.run;
  * память: пик RSS процесса (psutil или resource; на Windows без psutil — null);
  * с --labels: попадает ли размеченная секунда в покрытие, PTS ближайшего кадра и смещение.
VRAM: модуль CPU-only — не меряется (null).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

VIDEO_EXT = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}


def parse_variant(spec: str) -> tuple[str, dict[str, Any]]:
    """'skip:decode.skip_nonref=true,storage.max_side=640' -> ('skip', {'decode': {...}, 'storage': {...}})."""
    import yaml

    name, _, rest = spec.partition(":")
    override: dict[str, Any] = {}
    for item in filter(None, (x.strip() for x in rest.split(","))):
        key, _, val = item.partition("=")
        node = override
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return name.strip() or "base", override


def _peak_rss_mb() -> Optional[float]:
    try:
        import psutil  # type: ignore

        mi = psutil.Process().memory_info()
        return float(getattr(mi, "peak_wset", 0) or mi.rss) / 2**20
    except Exception:
        pass
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(r) / (2**20 if sys.platform == "darwin" else 1024)
    except Exception:
        return None


def _stats(values: Sequence[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"n": 0, "median": None, "p95": None, "max": None}
    a = np.asarray(values, dtype=np.float64)
    return {"n": int(a.size), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def read_labels(path: Optional[str]) -> dict[str, list[float]]:
    """video_id (нормализованный) -> размеченные секунды (схема eval.labels: video_id,file,label,t_sec,...)."""
    from .names import normalize_video_id

    out: dict[str, list[float]] = {}
    if not path:
        return out
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            vid = normalize_video_id(r.get("video_id") or Path(r.get("file", "")).stem)
            for cell in (r.get("t_sec"), r.get("t_all")):
                for tok in str(cell or "").replace(";", " ").replace("|", " ").split():
                    try:
                        out.setdefault(vid, []).append(float(tok.replace(",", ".")))
                    except ValueError:
                        continue
    return {k: sorted(set(v)) for k, v in out.items()}


def bench_video(path: str, override: dict[str, Any], *, sample_fps: float, n_probe: int, seed: int,
                labels: Sequence[float], coarse_fine: bool) -> dict[str, Any]:
    from .errors import IngestError
    from .video import load_video

    rec: dict[str, Any] = {"video": Path(path).name}
    t0 = time.perf_counter()
    try:
        clip = load_video(path, sample_fps, config=override)
    except IngestError as exc:
        rec.update({"ok": False, "error": exc.code, "message": exc.message, "load_s": time.perf_counter() - t0})
        return rec
    load_s = time.perf_counter() - t0
    try:
        m = clip.meta
        st = np.asarray(clip.source_times, dtype=np.float64)
        d = np.diff(st) if st.size > 1 else np.array([0.0])
        grid_dev = [abs(t - round(t * clip.sample_fps) / clip.sample_fps) for t in clip.times] if clip.sample_fps > 0 else []
        cov = clip.coverage
        rec.update({
            "ok": True, "backend": m["backend"], "width": clip.width, "height": clip.height,
            "rotation": m["orientation"]["rotation_cw"], "hflip": m["orientation"]["hflip"],
            "duration": clip.duration, "declared_duration": m.get("declared_duration"),
            "coverage": cov["coverage"], "complete": cov["complete"], "gaps": len(cov["gaps"]),
            "gaps_s": float(sum(b - a for a, b in cov["gaps"])), "decode_errors": m["decode_errors"],
            "corrupt_frames": m["corrupt_frames"], "nonmonotonic_pts": m["nonmonotonic_pts"],
            "skip_nonref": m["skip_nonref"], "t0_pts_sec": m["t0_pts_sec"], "n_source_frames": int(st.size),
            "n_samples": len(clip.times), "measured_fps": m["measured_fps"], "vfr": m["vfr"],
            "dt_min": float(d.min()), "dt_median": float(np.median(d)), "dt_max": float(d.max()), "dt_std": float(d.std()),
            "sample_grid_dev_max": float(max(grid_dev)) if grid_dev else None,
            "load_s": load_s, "probe_s": m["timings"].get("probe_s"), "decode_s": m["timings"].get("decode_s"),
            "ram_samples_mb": m["storage"]["ram_bytes"] / 2**20, "disk_samples_mb": m["storage"]["disk_bytes"] / 2**20,
            "warnings": len(m.get("warnings", [])),
        })
        # frame() промах / попадание
        idx = list(range(0, len(clip), max(1, len(clip) // 10)))[:10]
        miss, hit = [], []
        for i in idx:
            a = time.perf_counter(); clip.frame(i); miss.append((time.perf_counter() - a) * 1e3)
            a = time.perf_counter(); clip.frame(i); hit.append((time.perf_counter() - a) * 1e3)
        rec["frame_miss_ms"], rec["frame_hit_ms"] = float(np.median(miss)), float(np.median(hit))
        # точность цепочки времени на случайных t
        rng = np.random.default_rng(seed)
        req = sorted(rng.uniform(0.0, clip.duration, n_probe).tolist())
        a = time.perf_counter()
        refs = clip.frames_at(req)
        rec["frames_at_ms_per_frame"] = (time.perf_counter() - a) * 1e3 / max(len(req), 1)
        offs = [abs(r.offset_sec) for r in refs]
        rec["exact_offset"] = _stats(offs)
        rec["exact_in_gap_frac"] = float(np.mean([r.in_gap for r in refs])) if refs else 0.0
        again = clip.frames_at([r.timestamp_sec for r in refs[:5]])
        rec["exact_idempotent"] = all(x.timestamp_sec == r.timestamp_sec for x, r in zip(again, refs[:5]))
        diffs = []
        for i in idx[:5]:
            ex = clip.read_exact(clip.times[i])
            fr = clip.frame(i)
            if ex.image.shape != fr.shape:
                import cv2

                fr = cv2.resize(fr, (ex.image.shape[1], ex.image.shape[0]), interpolation=cv2.INTER_LINEAR)
            diffs.append(float(np.abs(ex.image.astype(np.int16) - fr.astype(np.int16)).mean()))
        rec["sample_vs_exact_absdiff"] = _stats(diffs)
        a = time.perf_counter()
        clip.frames_at([clip.duration * 0.25, clip.duration * 0.5, clip.duration * 0.75])
        rec["frames_at3_ms"] = (time.perf_counter() - a) * 1e3
        if coarse_fine:
            from . import coarse_fine as CF

            a = time.perf_counter()
            plan, reads = CF.run(clip, None, config=override)
            rec["coarse_fine"] = {"ms": (time.perf_counter() - a) * 1e3, "coarse": len(plan.coarse_indices),
                                  "windows": len(plan.windows), "dense_frames": sum(len(r.frames) for r in reads),
                                  "incomplete_windows": sum(not r.complete for r in reads)}
        if labels:
            lab = []
            for t in labels:
                ref = clip.read_exact(t)
                in_gap = any(a_ <= t <= b_ for a_, b_ in cov["gaps"])
                lab.append({"t_label": t, "t_frame": ref.timestamp_sec, "offset": ref.offset_sec,
                            "in_coverage": bool(ref.ok and not in_gap), "out_of_range": ref.out_of_range})
            rec["labels"] = lab
        rec["peak_rss_mb"] = _peak_rss_mb()
        rec["vram_mb"] = None  # CPU-only модуль
    finally:
        clip.close()
    return rec


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if r.get("ok")]
    lab = [x for r in ok for x in r.get("labels", [])]
    return {
        "videos": len(records), "failed": len(records) - len(ok),
        "errors": sorted({r.get("error") for r in records if not r.get("ok")} - {None}),
        "incomplete": sum(not r["complete"] for r in ok), "with_gaps": sum(r["gaps"] > 0 for r in ok),
        "coverage_min": min((r["coverage"] for r in ok), default=None),
        "load_s": _stats([r["load_s"] for r in ok]),
        "exact_offset_max": max((r["exact_offset"]["max"] or 0.0 for r in ok), default=None),
        "sample_vs_exact_absdiff_max": max((r["sample_vs_exact_absdiff"]["max"] or 0.0 for r in ok), default=None),
        "idempotent_all": all(r["exact_idempotent"] for r in ok) if ok else None,
        "frame_miss_ms": _stats([r["frame_miss_ms"] for r in ok]),
        "frame_hit_ms": _stats([r["frame_hit_ms"] for r in ok]),
        "frames_at3_ms": _stats([r["frames_at3_ms"] for r in ok]),
        "peak_rss_mb_max": max((r["peak_rss_mb"] or 0.0 for r in ok), default=None),
        "labels": {"n": len(lab), "outside_coverage": sum(not x["in_coverage"] for x in lab),
                   "offset_max": max((abs(x["offset"]) for x in lab), default=None)},
        "vram_mb": None,
    }


def run(videos: str, out: str, *, variants: Sequence[str] = ("base",), labels: Optional[str] = None,
        sample_fps: Optional[float] = None, n_probe: int = 20, seed: int = 0, limit: int = 0,
        coarse_fine: bool = False) -> dict[str, Any]:
    from .config import load_config
    from .names import normalize_video_id

    root = Path(videos)
    files = sorted(p for p in (root.rglob("*") if root.is_dir() else [root]) if p.suffix.lower() in VIDEO_EXT)
    if limit:
        files = files[:limit]
    lab = read_labels(labels)
    rate = float(sample_fps if sample_fps is not None else load_config()["sampling"]["default_fps"])
    result: dict[str, Any] = {"videos_dir": str(root), "sample_fps": rate, "n_probe": n_probe, "variants": {}}
    for spec in variants:
        name, override = parse_variant(spec)
        recs = [bench_video(str(f), override, sample_fps=rate, n_probe=n_probe, seed=seed,
                            labels=lab.get(normalize_video_id(f.name), []), coarse_fine=coarse_fine) for f in files]
        result["variants"][name] = {"override": override, "summary": summarize(recs), "videos": recs}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return result


def _fmt(v: Any) -> str:
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ml.ingest.bench", description="Замер ingest: coverage/PTS/latency")
    ap.add_argument("--videos", required=True, help="папка с роликами (рекурсивно) или один файл")
    ap.add_argument("--out", required=True, help="JSON-отчёт")
    ap.add_argument("--labels", default=None, help="labels.csv (video_id,file,label,t_sec,...)")
    ap.add_argument("--variant", action="append", default=None, help="имя[:ключ=значение,...]; можно несколько")
    ap.add_argument("--sample-fps", type=float, default=None)
    ap.add_argument("--probe-times", type=int, default=20, help="случайных t на ролик для проверки frame_at")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--coarse-fine", action="store_true", help="дополнительно замерить coarse_fine.run")
    args = ap.parse_args(argv)
    res = run(args.videos, args.out, variants=args.variant or ["base"], labels=args.labels, sample_fps=args.sample_fps,
              n_probe=args.probe_times, limit=args.limit, coarse_fine=args.coarse_fine)
    cols = ["videos", "failed", "incomplete", "with_gaps", "coverage_min", "exact_offset_max",
            "sample_vs_exact_absdiff_max", "idempotent_all", "peak_rss_mb_max"]
    print("variant".ljust(10) + "".join(c[:14].rjust(16) for c in cols) + "load_s(med/p95/max)".rjust(24))
    for name, v in res["variants"].items():
        s = v["summary"]
        ls = s["load_s"]
        print(name[:10].ljust(10) + "".join(_fmt(s[c]).rjust(16) for c in cols)
              + f"{_fmt(ls['median'])}/{_fmt(ls['p95'])}/{_fmt(ls['max'])}".rjust(24))
    print(f"отчёт: {args.out}")
    return 0 if all(v["summary"]["failed"] == 0 for v in res["variants"].values()) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
