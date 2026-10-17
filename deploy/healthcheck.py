#!/usr/bin/env python3
"""Проверки сервиса по HTTP (stdlib, без зависимостей).

  live  — процесс жив: GET /healthz отвечает кодом < 500 (для Docker HEALTHCHECK / супервизора).
  ready — production gate: (1) /readyz A02 = 200, ready=true, state=ready, pipeline=real (fake / mock_not_allowed /
          unavailable / warmup_failed / gpu_hung / 404 — НЕ проходят, пробную задачу тогда не шлём: у A02 успешная
          задача переводит warmup_failed в ready); (2) без ключа → 401; (3) полноценная synthetic-задача
          POST → GET → done с валидным JSON ТЗ, не mock/fake, no_violation только с подтверждённой полнотой.
          Synthetic-задача проверяет тракт и контракт, а НЕ качество CV. Туннель — только после ready.

  python deploy/healthcheck.py live --url http://127.0.0.1:8000
  python deploy/healthcheck.py ready --url http://127.0.0.1:8000 --api-key KEY [--video synth.mov] [--json]
Ключ: --api-key или SENTINEL_API_KEY / первый из SENTINEL_API_KEYS (не храните ключи в репо).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from deploy import httpkit
except ImportError:  # запуск как `python deploy/healthcheck.py`
    import httpkit  # type: ignore

DEFAULT_URL = os.environ.get("SENTINEL_HEALTH_URL") or f"http://127.0.0.1:{os.environ.get('SENTINEL_PORT', '8000')}"


WAITING_STATES = {"loading", "warming", "starting"}
TERMINAL_STATES = {"unavailable", "warmup_failed", "mock_not_allowed", "gpu_hung"}
SYNTH_NOTE = "synthetic job: проверка тракта POST/GET и контракта ответа, НЕ качества CV"


def readiness(url: str, allow_stub: bool = False) -> dict:
    """Классификация /readyz A02: ready | waiting | terminal | bad (терминальное/плохое — не ждать)."""
    r = httpkit.request("GET", url.rstrip("/") + "/readyz", timeout=10)
    body = r.json() if r.status else None
    body = body if isinstance(body, dict) else {}
    out = {"status": r.status, "ready": body.get("ready"), "state": body.get("state"),
           "pipeline": body.get("pipeline"), "pipeline_version": body.get("pipeline_version"),
           "jury_profile": body.get("jury_profile"), "detail": body.get("detail")}
    ok_pipelines = {"real", "stub"} if allow_stub else {"real"}
    if r.status == 0:
        out["verdict"], out["why"] = "waiting", f"нет ответа: {r.error}"
    elif r.status == 200 and body.get("ready") is True and body.get("state") == "ready" \
            and body.get("pipeline") in ok_pipelines:
        out["verdict"], out["why"] = "ready", "ok"
    elif r.status == 200:
        out["verdict"] = "bad"
        out["why"] = f"/readyz 200, но ready={body.get('ready')} state={body.get('state')} pipeline={body.get('pipeline')}"
    elif r.status == 503 and body.get("state") in TERMINAL_STATES:
        out["verdict"], out["why"] = "terminal", f"/readyz 503 state={body.get('state')}: {body.get('detail') or ''}"[:300]
    elif r.status == 503:
        out["verdict"], out["why"] = "waiting", f"/readyz 503 state={body.get('state')} (прогрев)"
    elif r.status == 404 and allow_stub:
        out["verdict"], out["why"] = "ready", "нет /readyz (разрешено только в тестах deploy)"
    else:
        out["verdict"], out["why"] = "bad", f"/readyz {r.status}: production gate требует /readyz A02"
    return out


def _mock_reasons(meta: dict) -> list[str]:
    """Как A02 result_is_mock / A01 contracts.extensions.is_mock_result (канон, если доступен)."""
    try:
        from contracts.extensions import is_mock_result  # type: ignore
        if is_mock_result(meta):
            return ["результат помечен как mock (contracts.extensions.is_mock_result)"]
    except ImportError:
        pass
    out = []
    if meta.get("mock") is True or meta.get("fake_pipeline") is True:
        out.append("meta.mock/fake_pipeline=true")
    prov = meta.get("provenance")
    if isinstance(prov, dict) and prov.get("mock") is True:
        out.append("meta.provenance.mock=true")
    for key in ("stages", "models"):
        block = meta.get(key)
        items = block.items() if isinstance(block, dict) else []
        for name, it in items:
            v = it if isinstance(it, dict) else {"status": it}
            if v.get("status") == "mock" or "mock" in str(v.get("origin", "")).lower():
                out.append(f"meta.{key}.{name}: mock")
    api = meta.get("api")
    if isinstance(api, dict) and api.get("pipeline") not in (None, "real"):
        out.append(f"meta.api.pipeline={api.get('pipeline')}")
    return out


def _coverage_reason(meta: dict, min_coverage: float) -> str | None:
    try:
        from contracts.extensions import coverage_problem  # type: ignore
        return coverage_problem(meta, min_coverage, allow_mock=False)
    except ImportError:
        pass
    if meta.get("complete") is False:
        return "meta.complete=false"
    cov = meta.get("coverage")
    ratio = cov.get("ratio") if isinstance(cov, dict) else cov
    if not isinstance(ratio, (int, float)):
        return "полнота анализа не подтверждена: нет meta.coverage"
    if ratio + 1e-9 < min_coverage:
        return f"coverage {ratio:.2f} < {min_coverage}"
    if isinstance(cov, dict) and cov.get("complete") is False:
        return "meta.coverage.complete=false"
    return None


def result_problems(doc: dict, allow_stub: bool = False, min_coverage: float = 0.95,
                    jury_profile: str | None = None) -> list[str]:
    """Итоговый JSON задачи: схема ТЗ + не mock/fake + no_violation только с подтверждённой полнотой."""
    p = httpkit.validate_job(doc)
    meta = doc.get("meta")
    if not isinstance(meta, dict):
        if jury_profile != "strict":
            p.append("в ответе нет meta — нельзя подтвердить реальный пайплайн и полноту")
        return p
    if not allow_stub:
        p += _mock_reasons(meta)
    if doc.get("verdict") == "no_violation":
        why = _coverage_reason(meta, min_coverage)
        if why:
            p.append(f"no_violation без подтверждённой полноты: {why} — неполный анализ ≠ «нарушений нет»")
    return p


def server_timings(doc: dict | None) -> dict:
    api = ((doc or {}).get("meta") or {}).get("api") if isinstance((doc or {}).get("meta"), dict) else None
    api = api if isinstance(api, dict) else {}
    return {k: api.get(k) for k in ("queue_wait_s", "processing_s", "total_s", "deadline_s", "pipeline")}


def live(url: str) -> dict:
    r = httpkit.request("GET", url.rstrip("/") + "/healthz", timeout=5)
    return {"check": "live", "ok": 0 < r.status < 500, "status": r.status, "elapsed_s": round(r.elapsed_s, 3),
            "error": r.error}


def ready(url: str, api_key: str, video: Path, limit_s: float = 300.0, allow_stub: bool = False,
          min_coverage: float = 0.95) -> dict:
    rep: dict = {"check": "ready", "url": url, "problems": [], "note": SYNTH_NOTE}
    p = rep["problems"]
    if not api_key:
        p.append("нет API-ключа (--api-key / SENTINEL_API_KEYS)")
        rep["ok"] = False
        return rep
    lv = live(url)
    rep["live"] = lv
    if not lv["ok"]:
        p.append(f"сервер не отвечает: {lv['error'] or lv['status']}")
        rep["ok"] = False
        return rep
    rz = readiness(url, allow_stub)
    rep["readyz"] = rz
    if rz["verdict"] != "ready":
        p.append(f"/readyz не готов ({rz['verdict']}): {rz['why']}")
        rep["ok"] = False
        return rep  # пробную задачу НЕ шлём: она может «вылечить» warmup_failed у A02
    body, ctype = httpkit.multipart("file", video.name, video.read_bytes())
    noauth = httpkit.request("POST", url.rstrip("/") + "/api/v1/analyze", {"Content-Type": ctype}, body, timeout=120)
    rep["no_key_status"] = noauth.status
    if noauth.status != 401:
        p.append(f"без X-API-Key ожидали 401, получили {noauth.status or noauth.error}")
    t0 = time.perf_counter()
    sub = httpkit.submit(url, video, api_key, timeout=300)
    rep["upload_s"] = round(sub.elapsed_s, 2)
    if sub.headers.get("x-sentinel-stub") and not allow_stub:
        p.append("отвечает ЗАГЛУШКА deploy/stub_api.py, а не реальный backend")
    job_id = (sub.json() or {}).get("job_id") if sub.status == 200 else None
    if not job_id:
        p.append(f"POST /api/v1/analyze: {sub.status or sub.error} {sub.body[:200]!r}")
        rep["ok"] = False
        return rep
    t_acc = time.perf_counter()
    doc, waited = httpkit.poll(url, job_id, api_key, limit_s=limit_s)
    t_done = time.perf_counter()
    rep["job_id"] = job_id
    rep["time_to_result_s"] = round(t_done - t0, 2)          # от начала загрузки
    rep["from_accept_s"] = round(t_done - t_acc, 2)          # от приёма (job_id) до итога, с очередью
    rep["server"] = server_timings(doc)                      # meta.api A02: queue_wait/processing/total
    rep["status"] = (doc or {}).get("status")
    rep["verdict"] = (doc or {}).get("verdict")
    rep["n_violations"] = len((doc or {}).get("violations") or [])
    if doc is None or doc.get("status") != "done":
        err = (doc or {}).get("error") or (doc or {}).get("detail")
        p.append(f"задача не завершилась успешно за {limit_s:.0f} с: status={rep['status']!r} {err or ''}".strip())
    else:
        p += result_problems(doc, allow_stub, min_coverage, rz.get("jury_profile"))
    rep["ok"] = not p
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("live", "ready"))
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--video", type=Path, default=None, help="по умолчанию — синтетическое 10 с")
    ap.add_argument("--limit", type=float, default=300.0, help="сек на пробный inference")
    ap.add_argument("--allow-stub", action="store_true", help="только для тестов deploy/")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.mode == "live":
        rep = live(a.url)
    else:
        video = a.video
        if video is None:
            try:
                from deploy.synth_video import make_video
            except ImportError:
                from synth_video import make_video  # type: ignore
            cache = Path(os.environ.get("SENTINEL_CACHE_DIR") or Path(__file__).resolve().parents[1] / "cache")
            video = cache / "healthcheck_synth.mov"
            if not video.is_file():
                make_video(video, seconds=10)
        rep = ready(a.url, a.api_key or httpkit.api_key_from_env(), video, a.limit, a.allow_stub)
    if not a.quiet:
        print(json.dumps(rep, ensure_ascii=False, indent=2) if a.json else
              f"[{rep['check']}] {'OK' if rep['ok'] else 'FAIL'} " + json.dumps(
                  {k: v for k, v in rep.items() if k not in ("check", "ok")}, ensure_ascii=False))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
