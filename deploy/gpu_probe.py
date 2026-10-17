#!/usr/bin/env python3
"""Проба GPU на КОНКРЕТНОЙ машине до запуска: драйвер, VRAM, torch CUDA, fp16-matmul, ORT providers.

  python deploy/gpu_probe.py                    # ожидаем CUDA 12.8 (lock L1) и >= 7.5 GiB VRAM
  python deploy/gpu_probe.py --allow-cpu        # CPU-режим (План Б без GPU)
Отчёт: logs/gpu_probe.json. Никаких «должно работать» — только то, что вернула машина.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def nvidia_smi() -> dict:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"present": False}
    q = "name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,power.draw"
    try:
        out = subprocess.run([exe, f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=15, check=True).stdout.strip().splitlines()
    except (subprocess.SubprocessError, OSError) as exc:
        return {"present": True, "error": str(exc)}
    gpus = []
    for line in out:
        v = [x.strip() for x in line.split(",")]
        gpus.append(dict(zip(["name", "driver", "mem_total_mib", "mem_used_mib", "mem_free_mib", "temp_c", "power_w"], v)))
    return {"present": True, "gpus": gpus}


def torch_probe() -> dict:
    try:
        import torch
    except ImportError as exc:
        return {"installed": False, "error": str(exc)}
    out = {"installed": True, "version": torch.__version__, "cuda_build": torch.version.cuda,
           "cuda_available": bool(torch.cuda.is_available())}
    if out["cuda_available"]:
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        out.update(device=p.name, capability=f"{p.major}.{p.minor}", vram_total_gib=round(total / 2**30, 2),
                   vram_free_gib=round(free / 2**30, 2), cudnn=torch.backends.cudnn.version(),
                   bf16=bool(torch.cuda.is_bf16_supported()))
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            a = a @ a.T
            a = a / a.norm()
        torch.cuda.synchronize()
        out["fp16_matmul_ms"] = round((time.perf_counter() - t0) * 100, 2)
        out["fp16_ok"] = bool(torch.isfinite(a).all().item())
        del a
        torch.cuda.empty_cache()
    return out


def ort_probe() -> dict:
    try:
        import onnxruntime as ort
    except ImportError:
        return {"installed": False}
    return {"installed": True, "version": ort.__version__, "providers": ort.get_available_providers()}


def evaluate(rep: dict, min_vram_gib: float, expect_cuda: str, allow_cpu: bool) -> list[str]:
    p, t = [], rep["torch"]
    if not t.get("installed"):
        return ["torch не установлен"]
    if allow_cpu:
        return p
    if not t.get("cuda_available"):
        p.append(f"torch {t.get('version')} не видит CUDA (сборка cuda={t.get('cuda_build')})")
        return p
    if expect_cuda and t.get("cuda_build") != expect_cuda:
        p.append(f"torch собран под CUDA {t.get('cuda_build')}, ожидали {expect_cuda}")
    if t.get("vram_total_gib", 0) < min_vram_gib:
        p.append(f"VRAM {t.get('vram_total_gib')} GiB < {min_vram_gib} GiB")
    if not t.get("fp16_ok", False):
        p.append("fp16 matmul дал NaN/Inf")
    ort = rep["onnxruntime"]
    if ort.get("installed") and "CUDAExecutionProvider" not in ort.get("providers", []):
        p.append("onnxruntime без CUDAExecutionProvider (стоит CPU-пакет или нет cuDNN 9)")
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-vram-gib", type=float, default=7.5)
    ap.add_argument("--expect-cuda", default="12.8")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    rep = {"probed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "python": sys.version.split()[0],
           "nvidia_smi": nvidia_smi(), "torch": torch_probe(), "onnxruntime": ort_probe()}
    rep["problems"] = evaluate(rep, a.min_vram_gib, a.expect_cuda, a.allow_cpu)
    rep["ok"] = not rep["problems"]
    out = a.out or Path(os.environ.get("SENTINEL_LOG_DIR") or Path(__file__).resolve().parents[1] / "logs") / "gpu_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
