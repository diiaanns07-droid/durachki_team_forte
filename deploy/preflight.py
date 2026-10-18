#!/usr/bin/env python3
"""Preflight перед запуском сервиса: если хоть одна блокирующая проверка FAIL — сервис не стартует.

  python deploy/preflight.py            # таблица + runs/deploy/preflight.json; код 1 при FAIL
  python deploy/preflight.py --json

Проверки: Python, venv против lock L1 + требования модулей, CUDA-сборка torch (без создания
CUDA-контекста — не отнимаем VRAM у работающего сервиса), свободная VRAM (nvidia-smi), обязательные
веса (снапшот полный, лицензия карточки), конфиги модулей парсятся, диск, права на каталоги,
наличие API-ключа (значение НЕ печатается), порт свободен, нет SENTINEL_MOCK/FAKE_PIPELINE,
ffmpeg/ffprobe/cloudflared. Модели не грузит: настоящий прогрев — warmup сервиса + пробная задача.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deploy import deploy_config as dc  # noqa: E402
from deploy import envcheck as ec  # noqa: E402
from deploy import weights as W  # noqa: E402

PLACEHOLDER_KEYS = {"change-me", "changeme", "test", "secret", "key", ""}


@dataclass
class Check:
    name: str
    status: str          # ok | warn | fail
    detail: str


class Probes:
    """Всё, что трогает систему, — здесь (тесты подменяют)."""

    def installed_versions(self) -> dict[str, str]:
        return ec.installed_versions()

    def torch_info(self) -> dict[str, Any]:
        try:
            import torch
        except ImportError as exc:
            return {"installed": False, "error": str(exc)}
        return {"installed": True, "version": torch.__version__, "cuda_build": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()), "device_count": torch.cuda.device_count()}

    def nvidia_smi(self) -> Optional[list[dict[str, str]]]:
        exe = shutil.which("nvidia-smi")
        if not exe:
            return None
        try:
            out = subprocess.run([exe, "--query-gpu=name,driver_version,memory.total,memory.used,memory.free",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15,
                                 check=True).stdout
        except (subprocess.SubprocessError, OSError):
            return None
        keys = ["name", "driver", "total_mib", "used_mib", "free_mib"]
        return [dict(zip(keys, [x.strip() for x in line.split(",")])) for line in out.strip().splitlines()]

    def port_in_use(self, host: str, port: int) -> bool:
        with socket.socket() as s:
            s.settimeout(0.5)
            return s.connect_ex((host, port)) == 0

    def which(self, name: str) -> Optional[str]:
        return shutil.which(name)


def _writable(d: Path) -> Optional[str]:
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / f".preflight_{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return None
    except OSError as exc:
        return str(exc)


def _keys_state(environ: dict, data_dir: Path) -> tuple[str, str]:
    raw = environ.get("SENTINEL_API_KEYS", "") + "," + environ.get("SENTINEL_API_KEY", "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if any(k.lower() in PLACEHOLDER_KEYS for k in keys):
        return "fail", "SENTINEL_API_KEYS содержит заглушку (change-me) — сгенерируйте ключ"
    if keys:
        short = sum(1 for k in keys if len(k) < 16)
        return ("warn" if short else "ok"), f"задано ключей: {len(keys)}" + (f", коротких (<16): {short}" if short else "")
    kf = environ.get("SENTINEL_API_KEYS_FILE")
    if kf and Path(kf).is_file():
        return "ok", "ключи из SENTINEL_API_KEYS_FILE"
    if (data_dir / "api_keys.txt").is_file():
        return "ok", f"ключи из {data_dir / 'api_keys.txt'}"
    return "warn", "ключ не задан: A02 сгенерирует data/api_keys.txt — передайте его жюри"


_TRUE, _FALSE = {"1", "true", "yes", "on"}, {"0", "false", "no", "off"}
A02_KEYS = ("workers", "gpu_slots", "warmup", "allow_mock", "fake_pipeline", "pipeline", "budget_s", "watchdog_s",
            "job_cap_s", "warmup_wait_s", "hung_gpu_s", "min_coverage", "jury_profile", "partial_accept")


def _yaml(path: Path) -> dict:
    import yaml
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return {}


def _envval(raw: str, current: Any) -> Any:
    v = raw.strip()
    if isinstance(current, bool) or v.lower() in _TRUE | _FALSE:
        return v.lower() in _TRUE
    try:
        return int(v) if v.lstrip("-").isdigit() else float(v)
    except ValueError:
        return v


def effective_a02(root: Path, environ: dict) -> Optional[dict]:
    """Как backend/api/config.py: config.yaml (или SENTINEL_API_CONFIG) < env SENTINEL_<KEY> (+ алиас DEADLINE_S)."""
    path = Path(environ.get("SENTINEL_API_CONFIG") or root / "backend" / "api" / "config.yaml")
    if not path.is_file():
        return None
    cfg = _yaml(path)
    out = {k: cfg.get(k) for k in A02_KEYS}
    if environ.get("SENTINEL_DEADLINE_S") and not environ.get("SENTINEL_BUDGET_S"):
        out["budget_s"] = _envval(environ["SENTINEL_DEADLINE_S"], out.get("budget_s"))
    for k in A02_KEYS:
        if environ.get("SENTINEL_" + k.upper()):
            out[k] = _envval(environ["SENTINEL_" + k.upper()], out.get(k))
    return out


def effective_a03(root: Path, environ: dict) -> Optional[dict]:
    """backend/pipeline/config.yaml + оверлей SENTINEL_PIPELINE_CONFIG + SENTINEL_EXECUTOR / SENTINEL_MOCK."""
    path = root / "backend" / "pipeline" / "config.yaml"
    if not path.is_file():
        return None
    cfg = _yaml(path)
    mode = (cfg.get("executor") or {}).get("mode")
    mock = cfg.get("mock")
    wdl = (cfg.get("warmup") or {}).get("deadline_s")
    extra = environ.get("SENTINEL_PIPELINE_CONFIG")
    if extra and Path(extra).is_file():
        over = _yaml(Path(extra))
        mode = (over.get("executor") or {}).get("mode", mode)
        mock = over.get("mock", mock)
        wdl = (over.get("warmup") or {}).get("deadline_s", wdl)
    if environ.get("SENTINEL_EXECUTOR"):
        mode = environ["SENTINEL_EXECUTOR"].strip()
    if environ.get("SENTINEL_MOCK"):
        mock = environ["SENTINEL_MOCK"].strip().lower() in _TRUE
    return {"executor_mode": mode, "mock": bool(mock), "warmup_deadline_s": wdl}


def run_checks(cfg: dict[str, Any], root: Path = REPO_ROOT, environ: Optional[dict] = None,
               probes: Optional[Probes] = None) -> list[Check]:
    environ = dict(os.environ if environ is None else environ)
    probes = probes or Probes()
    out: list[Check] = []
    add = lambda n, s, d: out.append(Check(n, s, d))  # noqa: E731
    env_cfg, paths, pf = cfg.get("env") or {}, cfg.get("paths") or {}, cfg.get("preflight") or {}
    svc, tun = cfg.get("service") or {}, cfg.get("tunnel") or {}

    want_py = str(env_cfg.get("python", "3.11"))
    have_py = f"{sys.version_info.major}.{sys.version_info.minor}"
    add("python", "ok" if have_py == want_py else "fail", f"{sys.version.split()[0]} (нужен {want_py})")

    lock = dc.repo_path(env_cfg.get("lock", ""), root)
    if lock.is_file():
        reqs, conflicts = ec.module_requirements(root, lock)
        rep = ec.check(ec.parse_lock(lock.read_text(encoding="utf-8-sig")), probes.installed_versions(),
                       env_cfg.get("critical") or [], reqs, str(lock))
        rep.blocking += [f"конфликт модулей с lock: {c}" for c in conflicts]
        detail = f"пинов {rep.checked}; блок: {len(rep.blocking)}; предупр.: {len(rep.warnings)}"
        if rep.blocking:
            detail += " — " + "; ".join(rep.blocking[:6])
        add("env-lock", "fail" if rep.blocking else ("warn" if rep.warnings else "ok"), detail)
    else:
        add("env-lock", "fail", f"нет lock L1: {lock}")

    device = (environ.get("SENTINEL_DEVICE") or "auto").lower()
    t = probes.torch_info()
    want_cuda = {"cu128": "12.8", "cu126": "12.6", "cu124": "12.4"}.get(str(env_cfg.get("torch_backend")), "")
    if not t.get("installed"):
        add("cuda", "fail", f"torch не импортируется: {t.get('error')}")
    elif device == "cpu":
        add("cuda", "warn", f"SENTINEL_DEVICE=cpu — GPU не используется (torch {t.get('version')})")
    elif not t.get("cuda_available"):
        add("cuda", "fail", f"torch {t.get('version')} не видит CUDA (сборка cuda={t.get('cuda_build')})")
    elif want_cuda and t.get("cuda_build") != want_cuda:
        add("cuda", "fail", f"torch собран под CUDA {t.get('cuda_build')}, lock требует {want_cuda}")
    else:
        add("cuda", "ok", f"torch {t.get('version')} cuda {t.get('cuda_build')}, устройств {t.get('device_count')}")

    host, port = str(svc.get("host", "127.0.0.1")), int(environ.get("SENTINEL_PORT") or svc.get("port", 8000))
    busy = probes.port_in_use(host, port)
    gpus = probes.nvidia_smi()
    need = float(pf.get("min_free_vram_gb", 0))
    if device == "cpu":
        add("vram", "warn", "CPU-режим")
    elif gpus is None:
        add("vram", "fail", "nvidia-smi недоступен")
    else:
        free_gb = float(gpus[0].get("free_mib") or 0) / 1024
        status = "ok" if free_gb >= need else ("warn" if busy else "fail")
        note = " (порт занят — возможно, VRAM держит уже запущенный сервис)" if busy and free_gb < need else ""
        add("vram", status, f"{gpus[0].get('name')}: свободно {free_gb:.1f} ГБ, нужно ≥ {need:.1f}{note}")

    weights_dir = Path(environ.get("SENTINEL_WEIGHTS") or dc.repo_path(paths.get("weights", "weights"), root))
    hub = Path(environ.get("HF_HUB_CACHE") or dc.repo_path(paths.get("hf_hub_cache", "weights/hf/hub"), root))
    req, opt = dc.required_models(cfg, environ)
    for m in req + opt:
        info = W.inspect_snapshot(hub, m["repo"], hash_files=bool(pf.get("hash_weights")))
        lic = info.get("license_card", "")
        problems = list(info["problems"])
        if info["commit"] and not W.policy.is_allowed(lic):
            problems.append(f"лицензия карточки '{lic or '—'}' не Apache/MIT/BSD")
        optional = m in opt
        status = "ok" if not problems else ("warn" if optional else "fail")
        size = sum(f["size"] for f in info["files"]) / 2**30
        detail = (f"@{info['commit'][:12]} {size:.2f} ГБ license={lic or '—'}" if info["commit"] else "")
        if problems or info["warnings"]:
            detail += " " + "; ".join(problems + info["warnings"])
        add(f"weights:{m['repo']}" + (" (опц.)" if optional else ""), status, detail.strip())

    for rel in pf.get("configs") or []:
        p = dc.repo_path(rel, root)
        if not p.is_file():
            add(f"config:{rel}", "fail", "нет файла")
            continue
        try:
            import yaml
            yaml.safe_load(p.read_text(encoding="utf-8"))
            add(f"config:{rel}", "ok", "парсится")
        except Exception as exc:  # noqa: BLE001
            add(f"config:{rel}", "fail", f"не парсится: {exc}")

    need_disk = float(pf.get("min_free_disk_gb", 0))
    for key in ("data", "cache", "logs"):
        d = Path(environ.get({"data": "SENTINEL_DATA_DIR", "cache": "SENTINEL_CACHE_DIR",
                              "logs": "SENTINEL_LOG_DIR"}[key]) or dc.repo_path(paths.get(key, key), root))
        err = _writable(d)
        if err:
            add(f"path:{key}", "fail", f"{d}: нет записи ({err})")
            continue
        free = shutil.disk_usage(d).free / 2**30
        add(f"path:{key}", "ok" if free >= need_disk else "fail", f"{d}: свободно {free:.1f} ГБ (нужно ≥ {need_disk:.0f})")
    add("path:weights", "ok" if hub.is_dir() else "fail", f"{hub}" + ("" if hub.is_dir() else ": нет каталога"))

    data_dir = Path(environ.get("SENTINEL_DATA_DIR") or dc.repo_path(paths.get("data", "data"), root))
    add("api-key", *_keys_state(environ, data_dir))
    mode = str(environ.get("SENTINEL_TUNNEL_MODE") or tun.get("mode", "quick"))
    if mode == "named":
        add("tunnel-token", "ok" if environ.get("TUNNEL_TOKEN") else "fail",
            "TUNNEL_TOKEN задан" if environ.get("TUNNEL_TOKEN") else "named tunnel без TUNNEL_TOKEN")

    a02 = effective_a02(root, environ)
    if a02 is None:
        add("a02:config", "fail", "нет backend/api/config.yaml — production gate не проверить")
    else:
        single = a02.get("workers") == 1 and a02.get("gpu_slots") == 1
        add("a02:single-worker", "ok" if single else "fail",
            f"workers={a02.get('workers')} gpu_slots={a02.get('gpu_slots')} (нужно 1/1: один GPU-воркер)")
        add("a02:warmup", "ok" if a02.get("warmup") is True else "fail",
            f"warmup={a02.get('warmup')} — без прогрева /readyz=ready ничего не доказывает")
        real = (a02.get("fake_pipeline") is not True and a02.get("allow_mock") is not True
                and str(a02.get("pipeline") or "").startswith("backend.pipeline"))
        add("a02:real-pipeline", "ok" if real else "fail",
            f"pipeline={a02.get('pipeline')} fake_pipeline={a02.get('fake_pipeline')} allow_mock={a02.get('allow_mock')}")
    a03 = effective_a03(root, environ)
    if a03 is None:
        add("a03:config", "fail", "нет backend/pipeline/config.yaml")
    else:
        add("a03:executor", "ok" if a03["executor_mode"] == "process" else "fail",
            f"executor.mode={a03['executor_mode']} (process: один процесс-владелец GPU, kill по таймауту "
            f"освобождает VRAM; inline — не для прода)")
        add("a03:mock", "fail" if a03["mock"] else "ok", f"mock={a03['mock']}")
    measured = bool(svc.get("budgets_measured_on_gpu"))
    budgets = {k: (a02 or {}).get(k) for k in ("budget_s", "watchdog_s", "job_cap_s", "warmup_wait_s", "hung_gpu_s")}
    budgets.update({"a03.warmup.deadline_s": (a03 or {}).get("warmup_deadline_s"),
                    "deploy.ready_timeout_s": svc.get("ready_timeout_s"),
                    "deploy.trial_job_limit_s": svc.get("trial_job_limit_s")})
    add("budgets", "ok" if measured else "warn",
        ("замерено на GPU L2: " if measured else "UNMEASURED на GPU L2 (цифры из конфигов, не замеры): ")
        + ", ".join(f"{k}={v}" for k, v in budgets.items()))

    for var, bad in (pf.get("forbid_env") or {}).items():
        if (environ.get(var) or "").strip() == str(bad):
            add(f"env:{var}", "fail", f"{var}={bad} — фиктивный режим запрещён в проде")
    add("port", "fail" if busy else "ok", f"{host}:{port} " + ("занят (сервис уже запущен? deploy\\windows\\status.ps1)"
                                                            if busy else "свободен"))
    for tool, var in (("ffmpeg", "SENTINEL_FFMPEG"), ("ffprobe", "SENTINEL_FFPROBE")):
        path = environ.get(var) or probes.which(tool)
        add(f"tool:{tool}", "ok" if path else "warn", path or "не найден (декодер A04 уйдёт на запасной путь)")
    cf = probes.which("cloudflared")
    add("tool:cloudflared", "ok" if cf else ("warn" if mode == "off" else "fail"), cf or "не найден")
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    cfg = dc.load(a.config)
    checks = run_checks(cfg)
    fails = [c for c in checks if c.status == "fail"]
    rep = {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ok": not fails, "config": cfg["_path"],
           "checks": [asdict(c) for c in checks]}
    logs = Path(os.environ.get("SENTINEL_LOG_DIR") or dc.repo_path((cfg.get("paths") or {}).get("logs", "runs/deploy")))
    out = a.out or logs / "preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if a.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    else:
        for c in checks:
            print(f"{c.status.upper():<5} {c.name:<48} {c.detail}")
        print(f"[preflight] {'OK' if not fails else f'FAIL ({len(fails)})'} → {out}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
