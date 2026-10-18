"""Приёмка локального API (L2): N последовательных + M параллельных задач через POST /api/v1/analyze.

Замеры: загрузка (POST), ожидание в очереди, обработка, всего (meta.api сервера) и время клиента; лимит 300 с —
по фактическому времени (лимиты сервера не меняются). Фоном: пик VRAM (nvidia-smi), пик RAM серверных процессов
(psutil), соединения серверных процессов с НЕлокальными адресами (офлайн). Ключ читается из deploy/.env и не
печатается. Печатаются только тайминги и статусы — без описаний и кадров.

    python tools/l2/api_acceptance.py --seq data/raw/X-1.mov ... --par data/raw/X-2.mov data/raw/X-3.mov
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import threading
import time
import uuid
from pathlib import Path
from urllib import request as urlreq

REPO = Path(__file__).resolve().parents[2]
SERVER_MARKERS = ("backend.api", "uvicorn", "backend.pipeline", "supervisor", "multiprocessing")


def api_key(env_file: Path) -> str:
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("SENTINEL_API_KEYS="):
            return line.split("=", 1)[1].split(",")[0].strip().strip('"').strip("'")
    raise SystemExit("SENTINEL_API_KEYS не найден в deploy/.env")


def _multipart(path: Path) -> tuple[bytes, str]:
    b = uuid.uuid4().hex
    head = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
            "Content-Type: video/quicktime\r\n\r\n").encode("utf-8")
    return head + path.read_bytes() + f"\r\n--{b}--\r\n".encode(), f"multipart/form-data; boundary={b}"


def _call(url: str, key: str, data: bytes | None = None, ctype: str | None = None, timeout: float = 600) -> dict:
    req = urlreq.Request(url, data=data, method="POST" if data is not None else "GET")
    req.add_header("X-API-Key", key)
    if ctype:
        req.add_header("Content-Type", ctype)
    with urlreq.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class Sampler(threading.Thread):
    def __init__(self, every: float = 0.5):
        super().__init__(daemon=True)
        self.every, self.stop_ev = every, threading.Event()
        self.vram_peak_mb = 0
        self.vram_base_mb = None
        self.ram_peak_mb = 0.0
        self.remote: set[str] = set()
        self.samples = 0

    def run(self) -> None:
        import psutil

        while not self.stop_ev.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
                used = int(out[0]) if out else 0
                self.vram_base_mb = used if self.vram_base_mb is None else self.vram_base_mb
                self.vram_peak_mb = max(self.vram_peak_mb, used)
            except Exception:  # noqa: BLE001
                pass
            rss = 0.0
            for p in psutil.process_iter(["name", "cmdline", "memory_info"]):
                try:
                    if "python" not in (p.info["name"] or "").lower():
                        continue
                    cmd = " ".join(p.info["cmdline"] or [])
                    if not any(m in cmd for m in SERVER_MARKERS) or "api_acceptance" in cmd:
                        continue
                    rss += p.info["memory_info"].rss / 2**20
                    for c in p.net_connections(kind="inet"):
                        if c.raddr and c.raddr.ip not in ("127.0.0.1", "::1"):
                            self.remote.add(f"{c.raddr.ip}:{c.raddr.port}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            self.ram_peak_mb = max(self.ram_peak_mb, rss)
            self.samples += 1
            self.stop_ev.wait(self.every)


def run_job(base: str, key: str, path: Path, poll: float = 1.0, limit_s: float = 300.0) -> dict:
    data, ctype = _multipart(path)
    t0 = time.time()
    sub = _call(f"{base}/api/v1/analyze?force=1", key, data, ctype)
    t_up = time.time() - t0
    jid = sub["job_id"]
    while True:
        d = _call(f"{base}/api/v1/jobs/{jid}", key)
        if d.get("status") not in ("processing", "queued"):
            break
        if time.time() - t0 > limit_s + 120:
            d = {"status": "client_timeout"}
            break
        time.sleep(poll)
    t_all = time.time() - t0
    api = (d.get("meta") or {}).get("api") or {}
    return {"file": path.name, "job_id": jid, "status": d.get("status"), "upload_s": round(t_up, 2),
            "queue_wait_s": api.get("queue_wait_s"), "processing_s": api.get("processing_s"),
            "server_total_s": api.get("total_s"), "client_total_s": round(t_all, 2),
            "within_300s": t_all <= limit_s and (api.get("total_s") or 0) <= limit_s,
            "complete": (d.get("meta") or {}).get("complete"), "gpu_peak_mb": api.get("gpu_peak_mb"),
            "duration_s": ((api.get("video") or {}).get("duration"))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--env", default=str(REPO / "deploy" / ".env"))
    ap.add_argument("--seq", nargs="+", required=True)
    ap.add_argument("--par", nargs="+", default=[])
    ap.add_argument("--out", default=str(REPO / "runs" / "l2" / "acceptance"))
    a = ap.parse_args()
    key = api_key(Path(a.env))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s = Sampler()
    s.start()
    rows = []
    for v in a.seq:
        r = run_job(a.base, key, Path(v))
        r["mode"] = "sequential"
        rows.append(r)
        print(f"seq {r['file']}: {r['status']} upload={r['upload_s']}s queue={r['queue_wait_s']} proc={r['processing_s']} "
              f"total={r['server_total_s']} client={r['client_total_s']} ≤300={r['within_300s']}", flush=True)
    if a.par:
        res: list[dict] = []
        th = [threading.Thread(target=lambda p=p: res.append(run_job(a.base, key, Path(p)))) for p in a.par]
        for t in th:
            t.start()
        for t in th:
            t.join()
        for r in res:
            r["mode"] = "concurrent"
            rows.append(r)
            print(f"par {r['file']}: {r['status']} upload={r['upload_s']}s queue={r['queue_wait_s']} proc={r['processing_s']} "
                  f"total={r['server_total_s']} client={r['client_total_s']} ≤300={r['within_300s']}", flush=True)
    s.stop_ev.set()
    s.join(timeout=5)
    tot = [r["client_total_s"] for r in rows]
    summary = {"n": len(rows), "done": sum(r["status"] == "done" for r in rows),
               "within_300s": sum(bool(r["within_300s"]) for r in rows), "client_total_s": {
                   "median": round(st.median(tot), 1), "max": max(tot)},
               "vram_peak_mb": s.vram_peak_mb, "vram_baseline_mb": s.vram_base_mb, "ram_peak_mb": round(s.ram_peak_mb),
               "remote_connections": sorted(s.remote), "samples": s.samples, "jobs": rows}
    (out / "acceptance.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "jobs"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
