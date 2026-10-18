"""CLI оркестратора.

    python -m backend.pipeline run video.mov --out out.json [--no-vlm] [--stages detect,track,...] [--mock]
                                             [--dump-intermediate [DIR]] [--from-cache] [--no-cache]
    python -m backend.pipeline batch data/raw --out preds/ [--workers 1] [--no-vlm] [--mock] [--from-cache]
    python -m backend.pipeline profile video.mov [--mock] [--no-vlm] [--json]
    python -m backend.pipeline warmup
    python -m backend.pipeline compare data/videos --labels data/labels.csv --modes baseline,refine --out reports/h1

Код выхода: 0 — status=done; 2 — status=failed (реальный статус ошибки, а не «пустой» успех).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from .config import STAGE_ORDER


def _overrides(args: argparse.Namespace) -> dict:
    over: dict[str, Any] = {}
    if getattr(args, "mock", False):
        over["mock"] = True
        os.environ["SENTINEL_MOCK"] = "1"
    if getattr(args, "stages", None):
        wanted = {s.strip() for s in args.stages.split(",") if s.strip()}
        unknown = wanted - set(STAGE_ORDER)
        if unknown:
            raise SystemExit(f"неизвестные стадии: {', '.join(sorted(unknown))}; доступны: {', '.join(STAGE_ORDER)}")
        over["stages"] = {s: (s in wanted or s in ("ingest", "fusion")) for s in STAGE_ORDER}
    if getattr(args, "no_vlm", False):
        over.setdefault("stages", {}).update({"vlm_verify": False, "vlm_discover": False})
    if getattr(args, "from_cache", False):
        over.setdefault("cache", {})["mode"] = "from_cache"
    if getattr(args, "no_cache", False):
        over.setdefault("cache", {})["mode"] = "off"
    if getattr(args, "inline", False):
        over.setdefault("executor", {})["mode"] = "inline"
    if getattr(args, "device", None):
        over.setdefault("gpu", {})["device"] = args.device
    if getattr(args, "no_partial", False):
        over.setdefault("partial", {})["enabled"] = False
    return over


def _progress_printer(prefix: str = "") -> Any:
    last = {"label": None}

    def cb(label: str, frac: float) -> None:
        if label != last["label"]:
            last["label"] = label
            print(f"{prefix}[{frac * 100:5.1f}%] {label}", file=sys.stderr, flush=True)

    return cb


def _result_dict(result: Any) -> dict:
    from .serde import result_to_dict

    return result_to_dict(result)


def _summary_line(name: str, result: Any) -> str:
    v = result.violations[0] if result.violations else None
    top = f"{v.type_id or v.type}@{v.timecode_sec:.1f}s" if v else "-"
    return (f"{name}: status={result.status} verdict={result.verdict} top={top} "
            f"t={result.meta.get('timings', {}).get('total', 0):.1f}s")


# ---------------------------------------------------------------- команды

def cmd_run(args: argparse.Namespace) -> int:
    from .orchestrator import dump_intermediate, run_with_context

    over = _overrides(args)
    job_id = args.job_id or Path(args.video).stem
    result, ctx = run_with_context(args.video, job_id, None if args.quiet else _progress_printer(),
                                   float(args.deadline), config_path=args.config, overrides=over)
    data = _result_dict(result)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)
    if args.dump_intermediate is not None and ctx is not None:
        target = args.dump_intermediate or (str(Path(args.out).with_suffix("")) + ".intermediate" if args.out
                                            else f"{job_id}.intermediate")
        files = dump_intermediate(ctx, target)
        print(f"промежуточные результаты: {target} ({len(files)} файлов)", file=sys.stderr)
    print(_summary_line(job_id, result), file=sys.stderr)
    for w in result.meta.get("warnings", [])[:20]:
        print(f"  warning: {w}", file=sys.stderr)
    if result.status != "done":
        print(f"  error: {result.meta.get('error')}", file=sys.stderr)
    return 0 if result.status == "done" else 2


def _iter_videos(folder: Path) -> list[Path]:
    from .media import VIDEO_EXTS

    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def cmd_batch(args: argparse.Namespace) -> int:
    from .media import normalize_video_name
    from .orchestrator import run

    over = _overrides(args)
    folder = Path(args.folder)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = _iter_videos(folder)
    if not videos:
        print(f"нет видео в {folder}", file=sys.stderr)
        return 1
    names: dict[Path, str] = {}
    for p in videos:  # X-1 (латиница) и Х-1 (кириллица) → одно имя; коллизии различаем суффиксом
        base = normalize_video_name(p.name)
        name, k = base, 2
        while name in names.values():
            name, k = f"{base}__{k}", k + 1
        names[p] = name
    t0 = time.perf_counter()
    rows: list[dict] = []
    done = 0

    def one(p: Path) -> dict:
        name = names[p]
        res = run(str(p), name, None, float(args.deadline), config_path=args.config, overrides=over)
        (out_dir / f"{name}.json").write_text(json.dumps(_result_dict(res), ensure_ascii=False, indent=2),
                                             encoding="utf-8")
        v = max(res.violations, key=lambda x: x.confidence, default=None)
        return {"file": p.name, "name": name, "status": res.status, "verdict": res.verdict,
                "type_id": v.type_id if v else "", "timecode_sec": v.timecode_sec if v else None,
                "confidence": v.confidence if v else None, "n_violations": len(res.violations),
                "seconds": res.meta.get("timings", {}).get("total"), "sha256": res.meta.get("video", {}).get("sha256"),
                "error": res.meta.get("error", ""), "warnings": len(res.meta.get("warnings", []))}

    workers = max(1, int(args.workers))
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(one, p): p for p in videos}
        for fut in cf.as_completed(futs):
            p = futs[fut]
            try:
                row = fut.result()
            except Exception as exc:  # noqa: BLE001 — run() не бросает, но перестрахуемся
                row = {"file": p.name, "name": names[p], "status": "failed", "error": str(exc)}
            rows.append(row)
            done += 1
            print(f"[{done}/{len(videos)}] {row['name']}: {row['status']} {row.get('verdict')} "
                  f"{row.get('type_id', '')}@{row.get('timecode_sec')} ({row.get('seconds')} с) "
                  f"{('ERROR: ' + row['error']) if row.get('error') else ''}", file=sys.stderr, flush=True)
    rows.sort(key=lambda r: r["name"])
    (out_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    import csv

    with open(out_dir / "summary.csv", "w", encoding="utf-8", newline="") as fh:
        cols = ["name", "file", "status", "verdict", "type_id", "timecode_sec", "confidence", "n_violations",
                "seconds", "warnings", "error", "sha256"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    failed = sum(1 for r in rows if r.get("status") != "done")
    print(f"готово: {len(rows)} видео за {time.perf_counter() - t0:.1f} с, failed={failed} → {out_dir}",
          file=sys.stderr)
    return 0 if failed == 0 else 2


def cmd_profile(args: argparse.Namespace) -> int:
    from .orchestrator import run_with_context

    over = _overrides(args)
    over.setdefault("cache", {})["mode"] = "off" if not args.with_cache else "read_write"
    over.setdefault("partial", {})["enabled"] = False
    reports = []
    rc = 0
    for k in range(max(1, int(args.repeat))):
        rep, status = _profile_once(args, over, f"{args.job_id or 'profile'}-{k + 1}", k)
        reports.append(rep)
        rc = rc or (0 if status == "done" else 2)
    if args.json:
        print(json.dumps(reports if len(reports) > 1 else reports[0], ensure_ascii=False, indent=2))
    return rc


def _profile_once(args: argparse.Namespace, over: dict, job_id: str, k: int) -> tuple[dict, str]:
    from .orchestrator import run_with_context

    result, ctx = run_with_context(args.video, job_id, None, float(args.deadline),
                                   config_path=args.config, overrides=over)
    rows = []
    for name in STAGE_ORDER:
        st = result.meta.get("stages", {}).get(name)
        if st is None:
            continue
        rows.append({"stage": name, "status": st.get("status"), "seconds": st.get("seconds", 0.0),
                     "vram_peak_mb": st.get("vram_peak_mb"), "origin": st.get("origin", ""),
                     "n_out": st.get("n_out"), "cold": bool(st.get("cold"))})
    report = {"video": result.meta.get("video", {}), "device": result.meta.get("device"),
              "sample_fps": result.meta.get("sample_fps"), "total_s": result.meta.get("timings", {}).get("total"),
              "status": result.status, "verdict": result.verdict, "stages": rows,
              "peak_rss_mb": _peak_rss_mb(), "executor": result.meta.get("executor"),
              "vram_events": result.meta.get("vram_events", []), "run": k + 1,
              "queue_wait_s": result.meta.get("queue_wait_s"), "vlm": result.meta.get("vlm"),
              "coverage": result.meta.get("coverage"), "modes": result.meta.get("modes")}
    if not args.json:
        print(f"--- прогон {k + 1} ({'холодный' if k == 0 else 'тёплый'}; cold=первый вызов в воркере)")
        print(f"{'стадия':<14}{'статус':<10}{'сек':>9}{'VRAM пик, МБ':>14}  cold  происхождение / выход")
        for r in rows:
            vram = "—" if r["vram_peak_mb"] is None else f"{r['vram_peak_mb']:.0f}"
            print(f"{r['stage']:<14}{str(r['status']):<10}{r['seconds']:>9.2f}{vram:>14}  "
                  f"{'да ' if r['cold'] else '   '}  {r['origin']} / {r['n_out'] if r['n_out'] is not None else '—'}")
        print(f"{'ИТОГО':<24}{report['total_s'] or 0:>9.2f}   device={report['device']} "
              f"sample_fps={report['sample_fps']} кадров={report['video'].get('n_sampled')} "
              f"peak RSS (оркестратор)={report['peak_rss_mb']} МБ")
        print("VRAM меряется в процессе-воркере (torch.cuda.max_memory_allocated по стадии); «—» = CPU/нет torch.")
        print(f"очередь: {report['queue_wait_s']} с; VLM: {report['vlm'] or '—'}")
        for w in result.meta.get("warnings", [])[:10]:
            print(f"  warning: {w}")
    return report, result.status


def _peak_rss_mb() -> Optional[float]:
    try:
        import resource  # нет на Windows

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(rss / 1024 / (1024 if sys.platform == "darwin" else 1), 1)
    except Exception:  # noqa: BLE001
        try:
            import psutil  # type: ignore

            return round(psutil.Process().memory_info().peak_wset / 1048576, 1)
        except Exception:  # noqa: BLE001
            return None


def cmd_compare(args: argparse.Namespace) -> int:
    from .compare import compare

    over = _overrides(args)
    over.pop("cache", None)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    tols = [float(x) for x in args.tol.split(",") if x.strip()]
    if args.cache != "off":
        print("ВНИМАНИЕ: кэш включён — время стадий между режимами несравнимо", file=sys.stderr)
    rep = compare(args.videos, args.labels, modes, args.out, float(args.deadline), args.cache, args.limit, tols,
                  extra=over or None, budgets_out=args.emit_budgets,
                  log=lambda msg: print(msg, file=sys.stderr, flush=True))
    print((Path(args.out) / "compare.md").read_text(encoding="utf-8"))
    return 0 if rep["results"] else 1


def cmd_warmup(args: argparse.Namespace) -> int:
    from .orchestrator import WarmupError, warmup
    from .registry import ModelRegistry

    t0 = time.perf_counter()
    rc = 0
    try:
        warmup(args.deadline)
    except WarmupError as exc:
        print(f"ПРОГРЕВ НЕ ПРОЙДЕН: {exc}", file=sys.stderr)
        rc = 2
    reg = ModelRegistry.instance()
    print(json.dumps({"ok": rc == 0, "warm": reg.warm, "seconds": round(time.perf_counter() - t0, 2),
                      "report": reg.warmup_report}, ensure_ascii=False, indent=2, default=str))
    return rc


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m backend.pipeline", description="Kassa Sentinel — оркестратор")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", help="доп. YAML поверх backend/pipeline/config.yaml")
        p.add_argument("--deadline", type=float, default=240.0, help="дедлайн задачи, с (по умолчанию 240)")
        p.add_argument("--mock", action="store_true", help="mock-режим: без весов (SENTINEL_MOCK=1)")
        p.add_argument("--no-vlm", action="store_true", help="выключить vlm_verify и vlm_discover")
        p.add_argument("--stages", help="включённые стадии через запятую (ingest и fusion — всегда)")
        p.add_argument("--from-cache", action="store_true", help="CV-стадии из кэша, rules+fusion заново")
        p.add_argument("--no-cache", action="store_true", help="не читать и не писать кэш")
        p.add_argument("--inline", action="store_true", help="без процесса-воркера (отладка; таймаут не останавливает CUDA)")
        p.add_argument("--device", choices=["auto", "cuda", "cpu"])
        p.add_argument("--no-partial", action="store_true", help="не писать data/jobs/<id>/partial.json")

    p = sub.add_parser("run", help="анализ одного видео")
    p.add_argument("video")
    p.add_argument("--out", help="куда записать JSON результата (по умолчанию stdout)")
    p.add_argument("--job-id")
    p.add_argument("--dump-intermediate", nargs="?", const="", default=None, metavar="DIR",
                   help="сохранить выходы стадий (dets.jsonl, tracks.json, …) в DIR")
    p.add_argument("--quiet", action="store_true")
    common(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("batch", help="анализ папки с видео")
    p.add_argument("folder")
    p.add_argument("--out", required=True, help="папка для <name>.json + summary.csv/json")
    p.add_argument("--workers", type=int, default=1,
                   help="параллельных задач (GPU всё равно один владелец: перекрывается только CPU-часть)")
    common(p)
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("profile", help="таблица времени и пикового VRAM по стадиям")
    p.add_argument("video")
    p.add_argument("--job-id")
    p.add_argument("--json", action="store_true")
    p.add_argument("--with-cache", action="store_true", help="не отключать кэш (по умолчанию — честный холодный прогон)")
    p.add_argument("--repeat", type=int, default=1, help="прогонов подряд: 1-й холодный (загрузка моделей), далее тёплые")
    common(p)
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("compare", help="замер L2: режимы пайплайна на размеченных видео (ошибки, таймкоды, время, VRAM)")
    p.add_argument("videos", help="папка с видео")
    p.add_argument("--labels", required=True, help="data/labels.csv (схема eval.labels)")
    p.add_argument("--modes", default="baseline,refine", help="профили из backend/pipeline/profiles/ или пути к YAML")
    p.add_argument("--out", required=True, help="папка отчёта: compare.md, compare.json, <режим>/preds/*.json")
    p.add_argument("--cache", choices=["off", "read_write"], default="off",
                   help="off (по умолч.) — честное время; read_write — быстрее, но время несравнимо")
    p.add_argument("--limit", type=int, help="первые N видео (смоук)")
    p.add_argument("--tol", default="1,2,5", help="допуски таймкода, с")
    p.add_argument("--emit-budgets", metavar="YAML", help="записать budgets_s из первого режима (p90 × запас)")
    common(p)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("warmup", help="прогрев моделей (как при старте сервера); код 2 — не готов")
    p.add_argument("--deadline", type=float, default=None, help="общий дедлайн прогрева, с (по умолч. warmup.deadline_s)")
    p.set_defaults(func=cmd_warmup)
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    import logging

    logging.basicConfig(level=os.environ.get("SENTINEL_LOG_LEVEL", "WARNING"),
                        format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    finally:
        from .worker import reset_executor

        reset_executor()
