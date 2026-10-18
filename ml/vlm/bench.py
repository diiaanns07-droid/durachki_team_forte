"""python -m ml.vlm.bench — замер PRODUCTION-режима VLM (не smoke на 2 токенах) для выбора конфигурации.

Режимы (--mode, через запятую; по умолчанию json,qa,chat):
  json — промпт verify как в проде (кадры verify_n_frames с подсветкой, max side, max_new_tokens_verify, схема):
         входные кадры и токены, фактические выходные токены, finish_reason (eos|length|time), доля валидного
         JSON (как в проде: обрезанный ответ не принимается) и обрезанных, время: подготовка кадров + prep
         (шаблон/процессор) + generation + decode + parse;
  discover — то же для промпта discover (frames_per_window кадров, max_new_tokens_discover) — только замер;
  qa   — вопросы qa_spec для --qa-type (forward без генерации): latency на вопрос, mass, words/letters;
  chat — ml.vlm.service.chat на нейтральном тексте (как у копилота): latency, токены, busy/ошибки.
Холодный старт (загрузка + первый вызов) и тёплые вызовы считаются отдельно. VRAM: после загрузки, пик на
вызов (torch), nvidia-smi до/после. Кадры: --frames DIR (jpg/png) или --synthetic N (без данных).
Печатаются ТОЛЬКО агрегаты; тексты ответов — в локальный дамп (privacy.local_dump_dir). Поля, которые на этой
машине не измерить (нет CUDA/nvidia-smi), остаются null — ничего не выдумываем.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional


def _nvidia_smi_used_mb() -> Optional[float]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
        vals = [float(x) for x in out.stdout.split() if x.strip()]
        return max(vals) if vals else None
    except Exception:  # noqa: BLE001
        return None


def _torch_peak_mb() -> Optional[float]:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except ImportError:
        pass
    return None


def _load_frames(frames_dir: Optional[str], synthetic: int) -> list[Any]:
    """BGR-кадры (как отдаёт VideoClip): из папки или синтетика 464x848."""
    import numpy as np

    if frames_dir:
        from PIL import Image

        paths = sorted(p for p in Path(frames_dir).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
        return [np.asarray(Image.open(p).convert("RGB"))[:, :, ::-1].copy() for p in paths]
    rng = np.random.default_rng(0)
    out = []
    for i in range(max(1, synthetic)):
        f = (rng.random((848, 464, 3)) * 40 + 150).astype("uint8")
        f[380:520, 60:400] = (70, 110, 150)        # «стол» (BGR)
        f[250:360, 330:420] = (130, 120, 120)      # «ящик»
        f[200:300, 180 + 10 * i:280 + 10 * i] = (30, 30, 220)
        out.append(f)
    return out


def _pct(xs: list[float], q: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))], 4)


def _stats(xs: list[float]) -> dict[str, Optional[float]]:
    return {"mean": round(statistics.mean(xs), 4) if xs else None, "p50": _pct(xs, 0.5), "p95": _pct(xs, 0.95),
            "max": round(max(xs), 4) if xs else None}


def _bench_json(be: Any, frames: list[Any], cfg: dict[str, Any], n: int, k: int, dump: Any) -> dict[str, Any]:
    """Полный verify-JSON как в проде: k кадров (verify_n_frames) с подсветкой, max_new_tokens_verify, схема."""
    return _bench_gen("verify", be, frames, cfg, n, k, dump)


def _bench_discover(be: Any, frames: list[Any], cfg: dict[str, Any], n: int, dump: Any) -> dict[str, Any]:
    """Полный discover-JSON как в проде (замер; сам discover в пайплайне по-прежнему выключен)."""
    k = int((cfg.get("discover") or {}).get("frames_per_window", 4))
    return _bench_gen("discover", be, frames, cfg, n, k, dump)


def _bench_gen(kind: str, be: Any, frames: list[Any], cfg: dict[str, Any], n: int, k: int,
               dump: Any) -> dict[str, Any]:
    from .catalog import catalog_types, load_catalog
    from .frames import prepare_images
    from .parsing import (discover_schema, extract_json, normalize_discover, normalize_verify,
                          verify_schema)
    from .prompting import render_discover, render_verify

    fcfg = cfg.get("frames") or {}
    gcfg = cfg.get("generation") or {}
    mnt = int(gcfg.get("max_new_tokens_verify" if kind == "verify" else "max_new_tokens_discover",
                       180 if kind == "verify" else 220))
    timeout = float(gcfg.get("call_timeout_s", 12.0))
    lang = str(cfg.get("lang", "ru"))
    allowed = sorted(catalog_types(load_catalog()))
    types = [t for t in allowed if t != "V99"] or ["V99"]
    checklist = {kk: v for kk, v in ((cfg.get("discover") or {}).get("checklist") or {}).items() if v in allowed}
    items = list(checklist)
    rows = []
    for i in range(max(1, n)):
        batch = [frames[(i * k + j) % len(frames)] for j in range(k)]
        times = [round(2.0 * j, 1) for j in range(k)]
        t0 = time.perf_counter()
        if kind == "verify":
            imgs = prepare_images(batch, times, {"person": (0.3, 0.2, 0.7, 0.9)}, times[k // 2], fcfg)
        else:
            imgs = prepare_images(batch, times, None, None, fcfg)
        prep_frames = time.perf_counter() - t0
        if kind == "verify":
            system, user = render_verify(types[i % len(types)], ["(синтетический кандидат для замера)"], times,
                                         None, lang)
            schema = verify_schema(allowed, len(times))
        else:
            system, user = render_discover(checklist, times, None, lang)
            schema = discover_schema(items, len(times))
        res = be.generate(imgs, system, user, max_new_tokens=mnt, timeout_s=timeout, json_schema=schema)
        t1 = time.perf_counter()
        pr = extract_json(res.text)
        if kind == "verify":
            parsed = pr.ok and normalize_verify(pr.data, set(allowed), len(times))[0] is not None
        else:
            parsed = pr.ok and normalize_discover(pr.data, items, len(times))[0] is not None
        parse_s = time.perf_counter() - t1
        tm = (res.meta or {}).get("timings") or {}
        rows.append({"total_s": prep_frames + res.latency_s + parse_s, "prep_frames_s": prep_frames,
                     "prep_s": tm.get("prep_s"), "gen_s": tm.get("gen_s"), "decode_s": tm.get("decode_s"),
                     "parse_s": parse_s, "out_tokens": res.completion_tokens, "in_tokens": res.prompt_tokens,
                     "finish": res.finish_reason or ("time" if res.timed_out else ""),
                     # как в проде: обрезанный ответ не принимается, даже если JSON «чинится»
                     "json_ok": bool(parsed and not res.truncated), "parse_ok": bool(parsed),
                     "repaired": bool(pr.ok and pr.repaired), "truncated": res.truncated,
                     "vram_peak_mb": (res.meta or {}).get("vram_peak_mb"), "error": res.error})
        dump({"kind": f"bench_{kind}", "call": i, "images": imgs, "prompt": user, "raw": [res.text],
              "finish": res.finish_reason, "error": res.error})
    out = _summ(rows)
    out.update({"input_frames_per_call": k, "max_new_tokens": mnt, "call_timeout_s": timeout,
                "parse_ok_rate": round(sum(r["parse_ok"] for r in rows) / len(rows), 3) if rows else None})
    return out


def _summ(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cold, warm = rows[:1], rows[1:] or rows[:1]

    def col(rs: list[dict[str, Any]], key: str) -> list[float]:
        return [float(r[key]) for r in rs if r.get(key) is not None]

    fin: dict[str, int] = {}
    for r in rows:
        fin[r.get("finish") or "unknown"] = fin.get(r.get("finish") or "unknown", 0) + 1
    out = {
        "calls": len(rows),
        "cold_first_call_s": round(rows[0]["total_s"], 4) if rows else None,
        "warm": {k: _stats(col(warm, k)) for k in ("total_s", "prep_frames_s", "prep_s", "gen_s", "decode_s",
                                                    "parse_s")},
        "out_tokens": _stats(col(rows, "out_tokens")),
        "in_tokens": _stats(col(rows, "in_tokens")),
        "finish_reasons": fin,
        "json_valid_rate": round(sum(r["json_ok"] for r in rows) / len(rows), 3) if rows else None,
        "json_repaired_rate": round(sum(r["repaired"] for r in rows) / len(rows), 3) if rows else None,
        "truncated_rate": round(sum(bool(r["truncated"]) for r in rows) / len(rows), 3) if rows else None,
        "errors": sum(bool(r.get("error")) for r in rows),
        "vram_peak_mb_max": max(col(rows, "vram_peak_mb")) if col(rows, "vram_peak_mb") else None,
    }
    if cold:
        out["cold_first_call_s"] = round(cold[0]["total_s"], 4)
    return out


def _bench_qa(be: Any, frames: list[Any], cfg: dict[str, Any], n: int, qa_type: str, dump: Any) -> dict[str, Any]:
    from .frames import resize_max_side, to_pil_rgb
    from .prompting import get_prompt
    from .qa import _prompt, crop_zone, load_spec

    if not getattr(be, "supports_option_probs", False):
        return {"status": "backend_without_logits"}
    spec = load_spec((cfg.get("qa") or {}).get("spec_path"))
    plan = (spec.get("types") or {}).get(qa_type) or []
    system = get_prompt("system_qa", str(cfg.get("prompt_version", "v1")), str(cfg.get("lang", "ru"))).render()
    max_side = int((cfg.get("frames") or {}).get("max_side", 448))
    lat, mass, modes = [], [], {}
    for i in range(max(1, n)):
        for j, (q, _at) in enumerate(plan):
            f = frames[(i + j) % len(frames)]
            imgs = [resize_max_side(to_pil_rgb(f), max_side)]
            if (spec["questions"][q].get("crop")):
                c = crop_zone(f, (0.12, 0.44, 0.86, 0.62), 0.1, max_side)  # синтетический «стол»
                if c is not None:
                    imgs.append(c)
            prompt = _prompt(spec, q, 2.0 * j, len(imgs) > 1)
            opts = (spec.get("options") or {})[spec["questions"][q].get("options", "yn")]
            sc = be.option_probs(imgs, system, prompt, opts, timeout_s=10.0)
            lat.append(sc.latency_s)
            mass.append(sc.mass)
            modes[sc.mode] = modes.get(sc.mode, 0) + 1
            dump({"kind": "bench_qa", "question": q, "images": imgs, "prompt": prompt, "probs": sc.probs,
                  "mass": sc.mass})
    return {"type": qa_type, "questions_per_candidate": len(plan), "forwards": len(lat),
            "cold_first_forward_s": round(lat[0], 4) if lat else None,
            "forward_s": _stats(lat[1:] or lat), "candidate_s_est": round(statistics.mean(lat[1:] or lat) * len(plan), 3)
            if lat else None, "mass": _stats(mass), "option_mode": modes}


def _bench_chat(n: int, dump: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    from .service import VLMService

    svc = VLMService(cfg)
    rows = []
    for i in range(max(1, n)):
        r = svc.chat([{"role": "system", "content": "Ты помощник проверяющего. Отвечай кратко по-русски."},
                      {"role": "user", "content": "Перечисли три общих правила хранения наличных в кассе."}])
        rows.append(r)
        dump({"kind": "bench_chat", "call": i, "ok": r.ok, "raw": [r.text], "error": r.error})
    lat = [r.latency_s for r in rows if r.ok]
    return {"calls": len(rows), "ok": sum(r.ok for r in rows), "errors": sorted({r.error for r in rows if r.error}),
            "latency_s": _stats(lat), "out_tokens": _stats([float(r.completion_tokens) for r in rows
                                                            if r.completion_tokens is not None]),
            "truncated": sum(r.truncated for r in rows)}


def run(argv: Optional[list[str]] = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(prog="python -m ml.vlm.bench", description=__doc__.split("\n\n")[0])
    ap.add_argument("--frames", default=None, help="папка с кадрами (jpg/png)")
    ap.add_argument("--synthetic", type=int, default=0, help="N синтетических кадров вместо --frames")
    ap.add_argument("--mode", default="json,qa,chat")
    ap.add_argument("--n", type=int, default=5, help="вызовов на режим (первый — холодный)")
    ap.add_argument("--images-per-call", type=int, default=None, help="по умолчанию frames.verify_n_frames")
    ap.add_argument("--qa-type", default="V02")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--no-dump", action="store_true")
    ap.add_argument("--out", default=None, help="записать агрегаты в JSON-файл")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    from .budget import visual_tokens
    from .config import load_config, preset_cfg
    from .frames import resize_max_side, to_pil_rgb
    from .localdump import LocalDump
    from .runtime import ensure_loaded, get_backend

    over: dict[str, Any] = {}
    if args.backend:
        over["backend"] = args.backend
    if args.preset:
        over["preset"] = args.preset
    cfg = load_config(args.config, overrides=over or None)
    modes = [m.strip() for m in args.mode.split(",") if m.strip()]
    k = int(args.images_per_call or (cfg.get("frames") or {}).get("verify_n_frames", 5))
    px = int(preset_cfg(cfg, "transformers").get("px_per_token", 28))
    rep: dict[str, Any] = {"backend_cfg": cfg.get("backend"), "preset": cfg.get("preset"), "modes": modes,
                           "images_per_call": k, "calls_per_mode": args.n,
                           "max_side": int((cfg.get("frames") or {}).get("max_side", 448)),
                           "max_new_tokens_verify": int((cfg.get("generation") or {}).get("max_new_tokens_verify", 180))}
    if not args.frames and not args.synthetic:
        rep["status"] = "no_input (use --frames DIR or --synthetic N)"
        return rep
    frames = _load_frames(args.frames, args.synthetic or k)
    rep["n_images_loaded"] = len(frames)
    rep["visual_tokens_est_per_call"] = k * visual_tokens(
        resize_max_side(to_pil_rgb(frames[0]), rep["max_side"]).size, px)
    rep["vram_before_mb_nvidia_smi"] = _nvidia_smi_used_mb()
    be, reason = get_backend(cfg)
    if be is None:
        rep.update(status="backend_unavailable", reason=reason)
        return rep
    t0 = time.perf_counter()
    ok, why = ensure_loaded(be)
    rep["load_s"] = round(time.perf_counter() - t0, 2)
    rep["backend"] = be.info()
    if not ok:
        rep.update(status="backend_load_failed", reason=why)
        return rep
    rep["vram_after_load_mb_nvidia_smi"] = _nvidia_smi_used_mb()
    dump = LocalDump(None if args.no_dump else (args.dump_dir or (cfg.get("privacy") or {}).get("local_dump_dir")),
                     tag="bench")
    if "json" in modes:
        rep["json"] = _bench_json(be, frames, cfg, args.n, k, dump)
        # совместимость с отчётом фазы 1
        rep["calls"] = rep["json"]["calls"]
        rep["json_valid_rate"] = rep["json"]["json_valid_rate"]
    if "discover" in modes:
        rep["discover"] = _bench_discover(be, frames, cfg, args.n, dump)
    if "qa" in modes:
        rep["qa"] = _bench_qa(be, frames, cfg, args.n, args.qa_type, dump)
    if "chat" in modes:
        rep["chat"] = _bench_chat(args.n, dump, cfg)
    rep.update({"status": "ok", "vram_peak_mb_torch": _torch_peak_mb(),
                "vram_after_mb_nvidia_smi": _nvidia_smi_used_mb(), "dump_dir": dump.path})
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return rep


def main(argv: Optional[list[str]] = None) -> int:
    rep = run(argv)
    json.dump(rep, sys.stdout, ensure_ascii=False, indent=2, default=str)
    sys.stdout.write("\n")
    return 0 if rep.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
