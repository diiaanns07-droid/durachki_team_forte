#!/usr/bin/env python3
"""ЗАМЕР SLA по HTTP (как жюри): время от приёма задания (с очередью) до итога у КАЖДОГО задания.

Требование «результат ≤ 300 с» сравнивается с полным временем задания (загрузка → очередь → обработка →
итог), а не со временем одной стадии. Ничего не объявляем — только меряем.

  python deploy/sla_bench.py --mode sequential --n 5 --synth --seconds 145      # 5 подряд
  python deploy/sla_bench.py --mode concurrent --n 2 --synth --seconds 145      # очередь из 2
  python deploy/sla_bench.py --mode concurrent --n 1 --synth --seconds 145 --expect failed   # тест таймаута
  python deploy/sla_bench.py --mode concurrent --videos D:\\data\\a.mov D:\\data\\b.mov     # реальные — только у L2

Синтетика: каждому заданию — СВОЙ файл (seed), иначе A02 вернёт ту же задачу (dedup по sha256), а A03
возьмёт стадии из кэша. Попадания в кэш (meta.cache) и дедуп помечаются: такие задания для SLA не годятся
(для замера: SENTINEL_CACHE=off у A03 или новые ролики). Synthetic-задания не говорят о качестве CV.
Пока идёт замер, ноутбук не должен засыпать (супервизор держит систему; гаснущий дисплей не мешает).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

try:
    from deploy import healthcheck, httpkit
except ImportError:
    import healthcheck  # type: ignore
    import httpkit  # type: ignore

NOTE = ("время задания = от начала загрузки / от приёма (job_id) до итогового статуса, включая очередь; "
        "не время одной стадии. Synthetic — проверка тракта и SLA-логики, не качества CV.")


def _cache_hits(doc: Optional[dict]) -> int:
    meta = (doc or {}).get("meta")
    cache = meta.get("cache") if isinstance(meta, dict) else None
    hits = cache.get("hits") if isinstance(cache, dict) else None
    if isinstance(hits, dict):
        return sum(int(v or 0) for v in hits.values() if isinstance(v, (int, float)))
    if isinstance(hits, (list, tuple)):
        return len(hits)
    return int(hits or 0) if isinstance(hits, (int, float)) else 0


def run_job(url: str, api_key: str, video: Path, t0: float, limit_s: float, poll_s: float, expect: str,
            allow_stub: bool = False) -> dict[str, Any]:
    j: dict[str, Any] = {"video": video.name, "size_mb": round(video.stat().st_size / 2**20, 1)}
    j["submit_at"] = round(time.perf_counter() - t0, 2)
    sub = httpkit.submit(url, video, api_key, timeout=limit_s)
    j["upload_s"] = round(sub.elapsed_s, 2)
    job_id = (sub.json() or {}).get("job_id") if sub.status == 200 else None
    if not job_id:
        j.update(status=f"submit {sub.status or sub.error}", final=False)
        return j
    t_acc = time.perf_counter()
    j["job_id"], j["accepted_at"] = job_id, round(t_acc - t0, 2)
    doc, _ = httpkit.poll(url, job_id, api_key, limit_s=limit_s, every_s=poll_s)
    t_done = time.perf_counter()
    status = (doc or {}).get("status")
    j.update(status=status, final=status in ("done", "failed"), verdict=(doc or {}).get("verdict"),
             n_violations=len((doc or {}).get("violations") or []),
             error=(doc or {}).get("error") or (doc or {}).get("detail"),
             done_at=round(t_done - t0, 2), from_accept_s=round(t_done - t_acc, 2),
             from_submit_s=round(t_done - t0 - j["submit_at"], 2),
             server=healthcheck.server_timings(doc), cache_hits=_cache_hits(doc))
    if status == "done" and expect == "done":
        j["problems"] = healthcheck.result_problems(doc, allow_stub=allow_stub)
    if expect == "failed" and status == "done" and (doc or {}).get("verdict") == "no_violation":
        j["problems"] = ["таймаут превратился в no_violation — так нельзя (неполный анализ ≠ нет нарушений)"]
    return j


def run_bench(url: str, api_key: str, videos: list[Path], limit_s: float = 300.0, mode: str = "concurrent",
              poll_s: float = 1.0, expect: str = "done", allow_stub: bool = False) -> dict[str, Any]:
    t0 = time.perf_counter()
    jobs: list[dict[str, Any]] = [{} for _ in videos]
    if mode == "sequential":
        for i, v in enumerate(videos):
            jobs[i] = run_job(url, api_key, v, t0, limit_s, poll_s, expect, allow_stub)
    else:
        def one(i: int, v: Path) -> None:
            jobs[i] = run_job(url, api_key, v, t0, limit_s, poll_s, expect, allow_stub)
        threads = [threading.Thread(target=one, args=(i, v), daemon=True) for i, v in enumerate(videos)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(limit_s + 120)
    ids = [j.get("job_id") for j in jobs if j.get("job_id")]
    dedup = len(ids) != len(set(ids))
    finals = [j for j in jobs if j.get("final")]
    timed = [j["from_submit_s"] for j in finals if j.get("from_submit_s") is not None]
    rep: dict[str, Any] = {
        "url": url, "mode": mode, "n": len(videos), "expect": expect, "limit_s": limit_s, "poll_s": poll_s,
        "jobs": jobs, "note": NOTE, "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dedup_detected": dedup, "cache_hit_jobs": sum(1 for j in jobs if j.get("cache_hits")),
        "max_from_submit_s": max(timed) if timed else None,
        "max_from_accept_s": max((j["from_accept_s"] for j in finals), default=None),
        "median_from_submit_s": round(statistics.median(timed), 2) if timed else None,
        "max_server_total_s": max((j["server"]["total_s"] for j in finals
                                   if isinstance((j.get("server") or {}).get("total_s"), (int, float))), default=None),
    }
    problems = []
    if len(finals) != len(videos):
        problems.append(f"итог получили {len(finals)} из {len(videos)} за {limit_s:.0f} с")
    want = [j for j in finals if j.get("status") == expect]
    if len(want) != len(finals):
        problems.append(f"ожидали status={expect} у всех: {[j.get('status') for j in finals]}")
    problems += [f"{j['video']}: {p}" for j in jobs for p in j.get("problems") or []]
    if dedup:
        problems.append("dedup: задания получили одинаковый job_id — одинаковые файлы, замер недействителен")
    if rep["cache_hit_jobs"]:
        problems.append(f"{rep['cache_hit_jobs']} заданий с попаданием в кэш стадий A03 — замер недействителен "
                        "(SENTINEL_CACHE=off или новые ролики)")
    if expect == "done" and rep["max_from_submit_s"] is not None and rep["max_from_submit_s"] > limit_s:
        problems.append(f"max {rep['max_from_submit_s']} с > {limit_s:.0f} с")
    rep["problems"], rep["ok"] = problems, not problems
    return rep


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=f"http://127.0.0.1:{os.environ.get('SENTINEL_PORT', '8000')}")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--mode", choices=("sequential", "concurrent"), default="concurrent")
    ap.add_argument("--n", type=int, default=0, help="число синтетических заданий (с --synth)")
    ap.add_argument("--videos", nargs="*", type=Path, default=[])
    ap.add_argument("--synth", action="store_true", help="синтетические видео, у каждого свой seed")
    ap.add_argument("--seconds", type=float, default=145.0)
    ap.add_argument("--seed", type=int, default=1000, help="база seed (разные запуски — разные файлы)")
    ap.add_argument("--limit", type=float, default=300.0)
    ap.add_argument("--poll", type=float, default=1.0, help="шаг опроса = точность замера")
    ap.add_argument("--expect", choices=("done", "failed"), default="done")
    ap.add_argument("--allow-stub", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    videos = list(a.videos)
    if len({p.resolve() for p in videos}) != len(videos):
        print("[sla] одинаковые файлы в --videos: A02 сдедуплицирует их — замер будет недействителен", file=sys.stderr)
    if a.synth:
        try:
            from deploy.synth_video import make_video
        except ImportError:
            from synth_video import make_video  # type: ignore
        cache = Path(os.environ.get("SENTINEL_CACHE_DIR") or Path(__file__).resolve().parents[1] / "cache") / "sla"
        videos += [make_video(cache / f"sla_{int(a.seconds)}s_seed{a.seed + i}.mov", seconds=a.seconds, seed=a.seed + i)
                   for i in range(max(1, a.n))]
    if not videos:
        ap.error("нужны --videos или --synth [--n N]")
    rep = run_bench(a.url, a.api_key or httpkit.api_key_from_env(), videos, a.limit, a.mode, a.poll, a.expect,
                    a.allow_stub)
    text = json.dumps(rep, ensure_ascii=False, indent=2)
    logs = Path(os.environ.get("SENTINEL_LOG_DIR") or Path(__file__).resolve().parents[1] / "runs" / "deploy")
    out = a.out or logs / f"sla_{a.mode}_{len(videos)}_{a.expect}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
