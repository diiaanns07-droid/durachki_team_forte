#!/usr/bin/env python3
"""ЗАМЕР публичного URL (Cloudflare Tunnel или любой другой) — ничего не обещаем заранее, только меряем.

  real  — против реального backend: HTTPS, задержка /healthz, 401 без ключа, загрузка MOV ~N МБ через
          /api/v1/analyze (413/524/обрыв?), время до результата, Set-Cookie (API должен работать без cookies),
          Range на / (для UI).
  tunnel — против deploy/stub_api.py за тем же туннелем: лестница размеров тела (echo), до какой паузы
          ответа туннель держит соединение (slow), проходит ли Range (206).
  python deploy/tunnel_probe.py real   --url https://xxx.trycloudflare.com --api-key KEY --mb 30
  python deploy/tunnel_probe.py tunnel --url https://xxx.trycloudflare.com --sizes 1,30,100 --slow 30,90,130
Отчёт JSON (--out), в нём только измеренное.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.parse
from pathlib import Path

try:
    from deploy import httpkit
except ImportError:
    import httpkit  # type: ignore


def _latency(url: str, n: int = 5) -> dict:
    vals, codes = [], []
    for _ in range(n):
        r = httpkit.request("GET", url.rstrip("/") + "/healthz", timeout=15)
        codes.append(r.status)
        if r.status:
            vals.append(r.elapsed_s)
    return {"codes": codes, "median_s": round(statistics.median(vals), 3) if vals else None}


def probe_real(url: str, api_key: str, mb: float, seconds: float, limit_s: float) -> dict:
    rep: dict = {"mode": "real", "url": url, "https": urllib.parse.urlsplit(url).scheme == "https"}
    rep["healthz"] = _latency(url)
    root = httpkit.request("GET", url.rstrip("/") + "/", {"Range": "bytes=0-99"}, timeout=15)
    rep["range_on_root"] = {"status": root.status, "accept_ranges": root.headers.get("accept-ranges")}
    rep["set_cookie_on_root"] = root.set_cookies
    try:
        from deploy.synth_video import make_video
    except ImportError:
        from synth_video import make_video  # type: ignore
    cache = Path(os.environ.get("SENTINEL_CACHE_DIR") or Path(__file__).resolve().parents[1] / "cache")
    video = make_video(cache / f"probe_{int(mb)}mb.mov", seconds=seconds, target_mb=mb)
    rep["upload_file_mb"] = round(video.stat().st_size / 2**20, 2)
    body, ctype = httpkit.multipart("file", video.name, video.read_bytes())
    rep["no_key_status"] = httpkit.request("POST", url.rstrip("/") + "/api/v1/analyze", {"Content-Type": ctype},
                                           body, timeout=300).status
    sub = httpkit.submit(url, video, api_key, timeout=600)
    rep["upload"] = {"status": sub.status, "elapsed_s": round(sub.elapsed_s, 2), "error": sub.error,
                     "mbit_s": round(rep["upload_file_mb"] * 8 / sub.elapsed_s, 1) if sub.status == 200 else None,
                     "set_cookie": sub.set_cookies, "stub": bool(sub.headers.get("x-sentinel-stub"))}
    job = (sub.json() or {}).get("job_id") if sub.status == 200 else None
    if job:
        doc, waited = httpkit.poll(url, job, api_key, limit_s=limit_s)
        rep["result"] = {"status": (doc or {}).get("status"), "waited_s": round(waited, 1),
                         "schema_problems": httpkit.validate_job(doc) if doc else ["нет ответа"]}
    rep["ok"] = bool(job) and rep["no_key_status"] == 401 and (rep.get("result") or {}).get("status") == "done"
    return rep


def probe_tunnel(url: str, sizes: list[float], slow: list[float]) -> dict:
    rep: dict = {"mode": "tunnel", "url": url, "healthz": _latency(url), "upload": [], "slow": []}
    for mb in sizes:
        data = os.urandom(int(mb * 2**20))
        r = httpkit.request("POST", url.rstrip("/") + "/__probe/echo", {"Content-Type": "application/octet-stream"},
                            data, timeout=900)
        echoed = (r.json() or {}) if r.status == 200 else {}
        rep["upload"].append({"mb": mb, "status": r.status, "error": r.error, "elapsed_s": round(r.elapsed_s, 2),
                              "intact": echoed.get("sha256") == hashlib.sha256(data).hexdigest()})
    for s in slow:
        r = httpkit.request("GET", url.rstrip("/") + f"/__probe/slow?seconds={s:g}", timeout=s + 60)
        rep["slow"].append({"seconds": s, "status": r.status, "elapsed_s": round(r.elapsed_s, 1), "error": r.error})
    rr = httpkit.request("GET", url.rstrip("/") + "/__probe/range", {"Range": "bytes=100-199"}, timeout=30)
    rep["range"] = {"status": rr.status, "bytes": len(rr.body), "content_range": rr.headers.get("content-range")}
    ok_up = [u["mb"] for u in rep["upload"] if u["status"] == 200 and u["intact"]]
    ok_slow = [x["seconds"] for x in rep["slow"] if x["status"] == 200]
    rep["measured"] = {"max_upload_ok_mb": max(ok_up) if ok_up else None,
                       "max_response_delay_ok_s": max(ok_slow) if ok_slow else None,
                       "range_206": rr.status == 206 and len(rr.body) == 100}
    rep["ok"] = bool(ok_up)
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("real", "tunnel"))
    ap.add_argument("--url", required=True)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--mb", type=float, default=30.0, help="real: размер синтетического MOV")
    ap.add_argument("--seconds", type=float, default=60.0, help="real: длина синтетического MOV")
    ap.add_argument("--limit", type=float, default=300.0)
    ap.add_argument("--sizes", default="1,30,100", help="tunnel: МБ через запятую")
    ap.add_argument("--slow", default="30,90,130", help="tunnel: паузы ответа, с")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.mode == "real":
        rep = probe_real(a.url, a.api_key or httpkit.api_key_from_env(), a.mb, a.seconds, a.limit)
    else:
        rep = probe_tunnel(a.url, [float(x) for x in a.sizes.split(",") if x],
                           [float(x) for x in a.slow.split(",") if x])
    rep["measured_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    text = json.dumps(rep, ensure_ascii=False, indent=2)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
