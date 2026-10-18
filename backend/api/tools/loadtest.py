"""Замер для L2: 5+ видео сначала последовательно, потом ограниченной пачкой; время каждого.

    SENTINEL_API_KEY=... python -m backend.api.tools.loadtest --url http://127.0.0.1:8000 \
        --videos D:/data/videos --n 6 --mode both --batch 5 --out runs/load_$(date +%s).json

* sequential: загрузка -> ждём done|failed -> следующее. Чистое время одного видео без очереди.
* batch: --batch видео одновременно (как жюри); лимит 300 с считается ОТ ПРИЁМА каждого job,
  ВКЛЮЧАЯ ОЧЕРЕДЬ. Второй проход идёт с ?force=1, чтобы идемпотентность (sha256) не вернула
  готовый результат первого прохода.
* Во время прогона раз в секунду опрашивается /readyz: длина очереди и VRAM процесса
  (если пайплайн загрузил torch) -> пик VRAM в отчёте.
* В отчёт пишутся ТОЛЬКО имена файлов, статусы, вердикты, число нарушений, type_id/таймкоды и
  тайминги. Никаких кадров и описаний: отчёт можно показывать кому угодно.
Метрики: latency от приёма (p50/p90/max), сколько > лимита, failed/timeout, upload, очередь,
обработка, deadline_s, пик VRAM. Код выхода 0 — все done и ≤ лимита."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from ._http import Client, api_key_from_env

VIDEO_EXT = {".mov", ".mp4", ".m4v"}


def list_videos(items: list[str]) -> list[Path]:
    out: list[Path] = []
    for it in items:
        p = Path(it)
        if p.is_dir():
            out += sorted(q for q in p.iterdir() if q.suffix.lower() in VIDEO_EXT)
        elif p.is_file():
            out.append(p)
    return out


class ReadyzSampler(threading.Thread):
    def __init__(self, url: str, period: float = 1.0):
        super().__init__(daemon=True)
        self.c = Client(url, timeout=10)
        self.period = period
        self.stop = threading.Event()
        self.max_queue = 0
        self.max_vram_mb: Optional[float] = None
        self.states: set[str] = set()

    def run(self) -> None:
        while not self.stop.wait(self.period):
            try:
                d = self.c.get("/readyz").json() or {}
            except Exception:
                continue
            q = d.get("queue") or {}
            self.max_queue = max(self.max_queue, int(q.get("queued") or 0) + int(q.get("running") or 0))
            gm = q.get("gpu_memory") or {}
            for k in ("max_allocated_mb", "reserved_mb"):
                if gm.get(k) is not None:
                    self.max_vram_mb = max(self.max_vram_mb or 0.0, float(gm[k]))
            self.states.add(str(d.get("state")))


def run_one(c: Client, key: str, path: Path, poll: float, timeout: float, force: bool) -> dict[str, Any]:
    rec: dict[str, Any] = {"video": path.name, "size_mb": round(path.stat().st_size / 1e6, 2)}
    data = path.read_bytes()
    t0 = time.perf_counter()
    try:
        r = c.post_file("/api/v1/analyze" + ("?force=1" if force else ""), data, path.name,
                        {"X-API-Key": key})
    except Exception as e:  # noqa: BLE001
        rec.update(status="upload_error", error=f"{type(e).__name__}: {e}"[:200])
        return rec
    t_acc = time.perf_counter()
    rec.update(upload_http=r.status, upload_s=round(t_acc - t0, 2))
    d = r.json() or {}
    if r.status != 200 or "job_id" not in d:
        rec.update(status="rejected", error=str(d.get("detail"))[:200])
        return rec
    rec["job_id"] = d["job_id"]
    rec["deduplicated"] = bool(d.get("deduplicated"))
    last: dict[str, Any] = {}
    while time.perf_counter() - t_acc < timeout:
        try:
            last = c.get(f"/api/v1/jobs/{d['job_id']}", {"X-API-Key": key}).json() or {}
        except Exception:
            pass
        if last.get("status") in ("done", "failed"):
            break
        time.sleep(poll)
    rec["accept_to_result_s"] = round(time.perf_counter() - t_acc, 2)
    rec["status"] = last.get("status") or "timeout"
    rec["verdict"] = last.get("verdict")
    vios = last.get("violations") or []
    rec["n_violations"] = len(vios)
    rec["violations"] = [{"type_id": v.get("type_id"), "timecode_sec": v.get("timecode_sec")} for v in vios]
    if last.get("status") == "failed":
        rec["error"] = str(last.get("error"))[:200]
    api = ((last.get("meta") or {}).get("api") or {})
    for k in ("queue_wait_s", "processing_s", "total_s", "deadline_s", "gpu_peak_mb", "response_bytes",
              "pipeline_version", "partial", "coverage"):
        if k in api:
            rec[k] = api[k]
    meta = last.get("meta") or {}
    if isinstance(meta.get("timings"), dict):  # A03: тайминги стадий
        rec["stage_timings"] = {k: v for k, v in meta["timings"].items() if isinstance(v, (int, float))}
    return rec


def summarize(rows: list[dict[str, Any]], limit_s: float, wall_s: float) -> dict[str, Any]:
    lat = [r["accept_to_result_s"] for r in rows if isinstance(r.get("accept_to_result_s"), (int, float))]
    lat_sorted = sorted(lat)

    def pct(p: float) -> Optional[float]:
        if not lat_sorted:
            return None
        return lat_sorted[min(len(lat_sorted) - 1, int(round(p * (len(lat_sorted) - 1))))]
    return {
        "n": len(rows), "done": sum(r.get("status") == "done" for r in rows),
        "failed": sum(r.get("status") == "failed" for r in rows),
        "not_finished": sum(r.get("status") not in ("done", "failed") for r in rows),
        "over_limit": sum(r.get("accept_to_result_s") is None or r["accept_to_result_s"] > limit_s
                          for r in rows),
        "deduplicated": sum(bool(r.get("deduplicated")) for r in rows),
        "latency_p50_s": pct(0.5), "latency_p90_s": pct(0.9), "latency_max_s": max(lat) if lat else None,
        "latency_mean_s": round(statistics.mean(lat), 2) if lat else None,
        "upload_max_s": max((r.get("upload_s") or 0 for r in rows), default=None),
        "processing_mean_s": round(statistics.mean([r["processing_s"] for r in rows
                                                    if isinstance(r.get("processing_s"), (int, float))]), 2)
        if any(isinstance(r.get("processing_s"), (int, float)) for r in rows) else None,
        "gpu_peak_mb_max": max((r.get("gpu_peak_mb") or 0 for r in rows), default=None) or None,
        "wall_s": round(wall_s, 1), "limit_s": limit_s,
    }


def run_pass(name: str, url: str, key: str, videos: list[Path], concurrency: int, poll: float,
             timeout: float, force: bool, limit_s: float) -> dict[str, Any]:
    sampler = ReadyzSampler(url)
    sampler.start()
    t0 = time.perf_counter()
    try:
        if concurrency <= 1:
            rows = [run_one(Client(url), key, p, poll, timeout, force) for p in videos]
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                rows = list(ex.map(lambda p: run_one(Client(url), key, p, poll, timeout, force), videos))
    finally:
        sampler.stop.set()
    s = summarize(rows, limit_s, time.perf_counter() - t0)
    s.update(max_active_jobs=sampler.max_queue, readyz_vram_peak_mb=sampler.max_vram_mb,
             readyz_states=sorted(sampler.states))
    return {"pass": name, "concurrency": concurrency, "summary": s, "jobs": rows}


def print_pass(p: dict[str, Any]) -> None:
    s = p["summary"]
    print(f"\n== {p['pass']} (параллельно {p['concurrency']}) ==")
    print(f"{'video':28} {'status':8} {'verdict':12} {'nv':>3} {'upload':>7} {'accept→res':>10} "
          f"{'queue':>6} {'proc':>6} {'dl':>5} {'vram':>6}")
    for r in p["jobs"]:
        def f(k: str) -> str:
            v = r.get(k)
            return "—" if v is None else (f"{v:.1f}" if isinstance(v, float) else str(v))
        name = r["video"][:26] + (" *" if r.get("deduplicated") else "")
        print(f"{name:28} {str(r.get('status'))[:8]:8} {str(r.get('verdict'))[:12]:12} "
              f"{r.get('n_violations', 0):>3} {f('upload_s'):>7} {f('accept_to_result_s'):>10} "
              f"{f('queue_wait_s'):>6} {f('processing_s'):>6} {f('deadline_s'):>5} {f('gpu_peak_mb'):>6}")
    print(f"done {s['done']}/{s['n']}, failed {s['failed']}, не завершено {s['not_finished']}, "
          f"> {s['limit_s']:g} с: {s['over_limit']}; latency p50 {s['latency_p50_s']} p90 {s['latency_p90_s']} "
          f"max {s['latency_max_s']} c; wall {s['wall_s']} c; пик активных {s['max_active_jobs']}; "
          f"VRAM пик {s['readyz_vram_peak_mb'] or s['gpu_peak_mb_max'] or 'n/a'} МБ"
          + (f"; * дубликат по sha256 (готовый результат): {s['deduplicated']}" if s["deduplicated"] else ""))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backend.api.tools.loadtest",
                                 description="Замер времени API на реальных видео (ключ — из env)")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--videos", nargs="+", required=True, help="папка(и) или файлы .mov/.mp4")
    ap.add_argument("--n", type=int, default=5, help="сколько видео взять (по порядку имён)")
    ap.add_argument("--mode", choices=["sequential", "batch", "both"], default="both")
    ap.add_argument("--batch", type=int, default=5, help="параллельность пачки (как жюри)")
    ap.add_argument("--poll", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=420.0, help="сколько ждать одну задачу, с")
    ap.add_argument("--limit", type=float, default=300.0, help="лимит от приёма до результата, с")
    ap.add_argument("--force-first", action="store_true", help="?force=1 и в первом проходе")
    ap.add_argument("--out", type=Path, required=True, help="JSON-отчёт (+ .csv рядом)")
    a = ap.parse_args(argv)
    key = api_key_from_env()
    if not key:
        print("нужен ключ: export SENTINEL_API_KEY=... (в argv ключ не передаём)", file=sys.stderr)
        return 2
    videos = list_videos(a.videos)[: a.n]
    if len(videos) < 1:
        print("нет видео .mov/.mp4", file=sys.stderr)
        return 2
    info = Client(a.url).get("/readyz").json() or {}
    report: dict[str, Any] = {"url_host": a.url.split("//")[-1].split("/")[0], "n_videos": len(videos),
                              "server": {k: info.get(k) for k in ("state", "pipeline", "pipeline_version",
                                                                  "version", "jury_profile")},
                              "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "passes": []}
    if info.get("pipeline") == "fake":
        print("ВНИМАНИЕ: сервер с SENTINEL_FAKE_PIPELINE=1 — цифры не про реальный пайплайн", file=sys.stderr)
    if a.mode in ("sequential", "both"):
        p = run_pass("sequential", a.url, key, videos, 1, a.poll, a.timeout, a.force_first, a.limit)
        report["passes"].append(p)
        print_pass(p)
    if a.mode in ("batch", "both"):
        p = run_pass(f"batch{a.batch}", a.url, key, videos, a.batch, a.poll, a.timeout,
                     a.force_first or a.mode == "both", a.limit)
        report["passes"].append(p)
        print_pass(p)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(a.out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        cols = ["pass", "video", "status", "verdict", "n_violations", "upload_s", "accept_to_result_s",
                "queue_wait_s", "processing_s", "total_s", "deadline_s", "gpu_peak_mb", "response_bytes",
                "deduplicated", "error"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for p in report["passes"]:
            for r in p["jobs"]:
                w.writerow({"pass": p["pass"], **r})
    print(f"\nотчёт: {a.out} (+ {a.out.with_suffix('.csv').name})")
    ok = all(p["summary"]["done"] == p["summary"]["n"] and p["summary"]["over_limit"] == 0
             for p in report["passes"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
