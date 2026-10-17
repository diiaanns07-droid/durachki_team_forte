#!/usr/bin/env python3
"""Доказательство локальности: РЕАЛЬНЫЙ worker (backend.pipeline.run) на локальных весах без сети.

Это проверка worker-а, а не публичного API: входящий HTTP проверяют отдельно (healthcheck.py,
tunnel_probe.py). Порядок:
  1) жёсткий офлайн HF/transformers + netguard=block в этом же процессе (до импорта моделей);
  2) weights.py verify: каждый файл из deploy/weights.lock.json на месте и совпадает по sha256;
  3) backend.pipeline.warmup() + run() на синтетическом видео (SENTINEL_MOCK=1 → отказ);
  4) FAIL, если была хоть одна попытка выхода в сеть (даже если пайплайн её «проглотил») — считаем по
     общему JSONL-логу netguard ВСЕХ процессов, включая spawn-воркер A03 (env и PYTHONPATH наследуются);
  5) FAIL на mock / fake-пайплайне и на no_violation при неполном анализе (meta.complete=false / coverage);
  6) --egress-selftest: отдельный python.exe БЕЗ netguard пытается выйти в интернет — под правилом
     фаервола (windows/offline_smoke.ps1 -Firewall) это должно быть заблокировано ОС.
Пишет отчёт JSON и человекочитаемый лог для питча (runs/deploy/offline_proof_<время>.md).
Отчёт — только числа/типы/таймкоды; ни кадров, ни описаний (в т.ч. чтобы не утекли в чат-ассистента).

  python deploy/offline_worker_smoke.py                        # Windows/Linux нативно
  docker run --rm --network none -v ./weights:/app/weights:ro IMAGE python deploy/offline_worker_smoke.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EGRESS_PROBE = """
import json, socket, sys
out = {}
for name, fn in (("tcp_1.1.1.1:443", lambda: socket.create_connection(("1.1.1.1", 443), timeout=4).close()),
                 ("dns_huggingface.co", lambda: socket.getaddrinfo("huggingface.co", 443))):
    try:
        fn()
        out[name] = "OPEN"
    except Exception as exc:
        out[name] = "blocked: " + type(exc).__name__ + ": " + str(exc)[:120]
print(json.dumps(out))
"""
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
               "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1"}


def _install_guard(log_path: Path) -> object:
    try:
        from deploy.netguard import sentinel_netguard as ng
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "netguard"))
        import sentinel_netguard as ng  # type: ignore
    ng.install("block", str(log_path))
    return ng


def _get(obj: object, key: str, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def check_result(result: object) -> list[str]:
    problems = []
    meta = _get(result, "meta", {}) or {}
    meta = meta if isinstance(meta, dict) else {}
    if _get(result, "status") != "done":
        problems.append(f"status={_get(result, 'status')!r}")
    if _get(result, "verdict") not in ("violation", "no_violation"):
        problems.append(f"verdict={_get(result, 'verdict')!r}")
    if meta.get("mock") or meta.get("fake_pipeline"):
        problems.append("meta.mock/fake_pipeline=true — это не реальные модели")
    stages = meta.get("stages") if isinstance(meta.get("stages"), dict) else {}
    mocked = sorted(k for k, v in stages.items() if isinstance(v, dict) and "mock" in str(v.get("origin", "")).lower())
    if mocked:
        problems.append(f"стадии из mock: {mocked}")
    cov = meta.get("coverage")
    incomplete = meta.get("complete") is False or (isinstance(cov, dict) and cov.get("complete") is False)
    if _get(result, "verdict") == "no_violation" and incomplete:
        problems.append("no_violation при неполном анализе (meta.complete/coverage) — неполный анализ ≠ «нарушений нет»")
    for i, v in enumerate(_get(result, "violations", []) or []):
        for k in ("type", "timecode_sec", "description", "evidence_frames"):
            if _get(v, k) in (None, ""):
                problems.append(f"violations[{i}].{k} пусто")
    return problems


def summarize(result: object) -> dict:
    vs = _get(result, "violations", []) or []
    meta = _get(result, "meta", {}) or {}
    meta = meta if isinstance(meta, dict) else {}
    stages = meta.get("stages") if isinstance(meta.get("stages"), dict) else {}
    return {"status": _get(result, "status"), "verdict": _get(result, "verdict"), "n_violations": len(vs),
            "violations": [{"type_id": _get(v, "type_id", ""), "timecode_sec": _get(v, "timecode_sec"),
                            "n_frames": len(_get(v, "evidence_frames", []) or [])} for v in vs],
            "complete": meta.get("complete"), "degraded": meta.get("degraded"), "coverage": meta.get("coverage"),
            "executor": meta.get("executor"), "device": meta.get("device"),
            "stages": {k: {x: v.get(x) for x in ("status", "origin", "seconds", "vram_peak_mb")}
                       for k, v in stages.items() if isinstance(v, dict)},
            "models": meta.get("models"), "vram_events": meta.get("vram_events"),
            "n_warnings": len(meta.get("warnings") or []), "meta_keys": sorted(meta)}


def count_attempts(log: Path) -> tuple[int, list[dict]]:
    recs = []
    try:
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                recs.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        pass
    return len(recs), recs[:20]


def egress_selftest() -> dict:
    """Отдельный процесс того же python.exe БЕЗ netguard: проверяем, что ОС (фаервол/--network none) режет выход."""
    import subprocess

    env = {k: v for k, v in os.environ.items() if not k.startswith("SENTINEL_NETGUARD")}
    env["PYTHONPATH"] = str(REPO_ROOT)
    for k in [k for k in env if k.lower() in ("http_proxy", "https_proxy", "all_proxy")]:
        env.pop(k)
    try:
        r = subprocess.run([sys.executable, "-I", "-c", EGRESS_PROBE], capture_output=True, text=True, timeout=30,
                           env=env)
        res = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    res["all_blocked"] = all(str(v).startswith("blocked") for v in res.values())
    return res


def write_pitch_log(path: Path, report: dict) -> None:
    r, w, res = report, report.get("weights") or {}, report.get("result") or {}
    lines = [
        "# Kassa Sentinel — доказательство локальной обработки",
        "",
        f"- Время: {r.get('started_at')}; коммит: `{r.get('git_sha') or '—'}`",
        f"- Итог: **{'OK' if r.get('ok') else 'FAIL'}**" + ("" if r.get("ok") else f" — {'; '.join(r.get('problems', []))}"),
        f"- Сеть в процессе анализа: netguard=block (аудит-хук Python, наследуется spawn-воркером); "
        f"попыток выхода: **{r.get('network_attempts')}**",
        f"- Офлайн-режим HF: {r.get('env')}",
        f"- Веса: {len(w.get('models') or [])} моделей проверены по weights.lock.json (точный commit + sha256/размер): "
        f"{', '.join(w.get('models') or []) or '—'}",
        f"- Видео: синтетическое={r.get('video', {}).get('synthetic')} (реальные кадры в отчёт не попадают)",
        f"- Результат: status={res.get('status')}, verdict={res.get('verdict')}, complete={res.get('complete')}, "
        f"executor={res.get('executor')}",
        f"- Время: {r.get('timings_s')}; GPU: {r.get('gpu')}",
    ]
    eg = r.get("egress_selftest")
    if eg:
        lines.append(f"- Проверка ОС (python.exe без netguard → интернет): {eg}")
    fw = os.environ.get("SENTINEL_FIREWALL_RULE")
    if fw:
        lines.append(f"- Правило фаервола (только на python.exe, временное): {fw}")
    lines += ["", "Публичный HTTP (cloudflared — отдельный процесс) проверяется отдельно: deploy/tunnel_probe.py.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def gpu_evidence() -> dict:
    """VRAM ЭТОГО процесса. При executor=process (прод A03) модели живут в spawn-воркере — тогда
    смотрите result.stages[*].vram_peak_mb и result.vram_events (пишет воркер), а не это поле."""
    torch = sys.modules.get("torch")
    if torch is None:
        return {"torch_imported": False, "note": "модели в процессе-воркере A03: см. result.stages/vram_events"}
    out = {"torch_imported": True, "cuda_available": bool(torch.cuda.is_available())}
    if out["cuda_available"]:
        out["device"] = torch.cuda.get_device_name(0)
        out["max_memory_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    return out


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # консоль Windows (cp1252/cp866) без PYTHONUTF8 падает на «≠» и кириллице
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, default=None, help="по умолчанию — синтетическое")
    ap.add_argument("--seconds", type=float, default=20.0, help="длина синтетического видео")
    ap.add_argument("--deadline", type=float, default=240.0)
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--fast-verify", action="store_true", help="размеры без sha256")
    ap.add_argument("--allow-empty-lock", action="store_true", help="lock пуст — всё равно гонять")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--egress-selftest", action="store_true", help="проверить, что ОС режет выход python.exe")
    ap.add_argument("--require-egress-blocked", action="store_true", help="FAIL, если self-test нашёл выход")
    a = ap.parse_args(argv)

    os.environ.update(OFFLINE_ENV)
    log_dir = Path(os.environ.get("SENTINEL_LOG_DIR") or REPO_ROOT / "runs" / "deploy")
    log_dir.mkdir(parents=True, exist_ok=True)
    guard_log = log_dir / "netguard_offline_smoke.jsonl"
    guard_log.unlink(missing_ok=True)
    egress = egress_selftest() if a.egress_selftest else None  # до установки guard: проверяем именно ОС
    # spawn-воркер A03 наследует env → netguard и в нём (sitecustomize из deploy/netguard)
    os.environ["SENTINEL_NETGUARD"] = "block"
    os.environ["SENTINEL_NETGUARD_LOG"] = str(guard_log)
    pp = [str(REPO_ROOT / "deploy" / "netguard"), str(REPO_ROOT)]
    if os.environ.get("PYTHONPATH"):
        pp.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(pp)
    ng = _install_guard(guard_log)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    report: dict = {"check": "offline_worker_smoke", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "netguard": "block (process + spawn workers)", "env": {k: os.environ.get(k) for k in OFFLINE_ENV}}
    try:
        import subprocess
        report["git_sha"] = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True,
                                           text=True, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        report["git_sha"] = ""
    problems: list[str] = []
    if egress is not None:
        report["egress_selftest"] = egress
        if a.require_egress_blocked and not egress.get("all_blocked"):
            problems.append(f"ОС не блокирует выход python.exe: {egress}")

    if (os.environ.get("SENTINEL_MOCK") or "0") not in ("0", "", "false", "False"):
        print("[smoke] SENTINEL_MOCK включён: mock не доказывает локальность моделей — отказ", file=sys.stderr)
        return 2

    from deploy import weights as W  # noqa: E402 — после установки guard
    weights_dir = a.weights or Path(os.environ.get("SENTINEL_WEIGHTS") or REPO_ROOT / "weights")
    lock = W.load_lock(REPO_ROOT / "deploy" / "weights.lock.json")
    code, results = W.verify(weights_dir, lock, fast=a.fast_verify)
    report["weights"] = {"dir": str(weights_dir), "models": [r["name"] for r in results],
                         "ok": code == 0, "problems": [p for r in results for p in r["problems"]]}
    if code:
        problems.append("weights verify не прошёл")
    if not results and not a.allow_empty_lock:
        problems.append("weights.lock.json пуст — нечего доказывать (weights.py prepare)")

    video = a.video
    if video is None:
        from deploy.synth_video import make_video
        video = make_video(log_dir / "offline_smoke_synth.mov", seconds=a.seconds)
    report["video"] = {"synthetic": a.video is None, "seconds": a.seconds if a.video is None else None}

    if not problems:
        try:
            import backend.pipeline as pipeline  # noqa: E402
            t0 = time.perf_counter()
            pipeline.warmup()
            t1 = time.perf_counter()
            stages: list[tuple[str, float]] = []
            result = pipeline.run(str(video), "offline-smoke", lambda s, p: stages.append((s, round(p, 3))),
                                  deadline_s=a.deadline)
            t2 = time.perf_counter()
            report["timings_s"] = {"warmup": round(t1 - t0, 2), "run": round(t2 - t1, 2)}
            if a.video is None and a.seconds:
                report["timings_s"]["run_per_video_second"] = round((t2 - t1) / a.seconds, 3)
            report["stages"] = [s for s, _ in stages]
            report["result"] = summarize(result)
            problems += check_result(result)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"pipeline: {type(exc).__name__}: {exc}")
            report["traceback"] = traceback.format_exc(limit=8)
    report["gpu"] = gpu_evidence()
    attempts, sample = count_attempts(guard_log)  # все процессы, включая воркер
    attempts = max(attempts, ng.status()["attempts"])
    report["network_attempts"] = attempts
    if attempts:
        report["network_attempts_sample"] = sample
        problems.append(f"{attempts} попыток выхода в сеть (см. {guard_log})")
    report["ok"] = not problems
    report["problems"] = problems
    out = a.report or log_dir / "offline_worker_smoke.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pitch = log_dir / f"offline_proof_{time.strftime('%Y%m%d_%H%M%S')}.md"
    write_pitch_log(pitch, report)
    print(f"[smoke] лог для питча: {pitch}", file=sys.stderr)
    print(json.dumps({k: report.get(k) for k in ("ok", "problems", "timings_s", "network_attempts", "gpu")},
                     ensure_ascii=False, indent=2))
    print(f"[smoke] отчёт: {out}", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
