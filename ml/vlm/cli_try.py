"""python -m ml.vlm.try video.mov --t 22 [--type V05] [--mode verify|discover|both]

Ручная проверка VLM на одном моменте видео. ПРИВАТНОСТЬ: кадры, промпты и ответы модели сохраняются только
локально (--dump-dir, по умолчанию outputs/vlm); в stdout — технические агрегаты (статус, задержки,
согласие/уверенность, число фактов, флаги валидации). Тексты фактов печатаются только с --show-text —
не используйте этот флаг, если вывод читает облачный ассистент.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Optional


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ml.vlm.try", description=__doc__.split("\n\n")[0])
    p.add_argument("video")
    p.add_argument("--t", type=float, required=True, help="секунда события (таймкод разметки)")
    p.add_argument("--type", default="V99", help="тип-кандидат V01..V08/V99 для verify")
    p.add_argument("--conf", type=float, default=0.5, help="исходная уверенность кандидата")
    p.add_argument("--span", type=float, default=2.0, help="полуширина окна события, с")
    p.add_argument("--mode", choices=["verify", "discover", "both"], default="verify")
    p.add_argument("--verify-mode", choices=["json", "qa"], default=None, help="режим verify (по умолчанию из config)")
    p.add_argument("--budget", type=float, default=60.0, help="бюджет времени стадии, с")
    p.add_argument("--backend", default=None, help="mock|transformers|llamacpp_server|llamacpp_python|auto")
    p.add_argument("--preset", default=None)
    p.add_argument("--lang", default=None, choices=["ru", "en"])
    p.add_argument("--config", default=None, help="альтернативный yaml (поверх config.yaml)")
    p.add_argument("--dump-dir", default=None, help="локальная папка для кадров/ответов (по умолчанию из config)")
    p.add_argument("--no-dump", action="store_true")
    p.add_argument("--show-text", action="store_true", help="печатать тексты фактов (только для человека!)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def run(argv: Optional[list[str]] = None) -> dict[str, Any]:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    from ._contract import Event
    from ._video import open_clip
    from .config import load_config
    from .discoverer import discover_events
    from .localdump import LocalDump
    from .runtime import ensure_loaded, get_backend
    from .verifier import verify_events

    over: dict[str, Any] = {}
    if args.backend:
        over["backend"] = args.backend
    if args.preset:
        over["preset"] = args.preset
    if args.lang:
        over["lang"] = args.lang
    cfg = load_config(args.config, overrides=over or None)
    dump_root = None if args.no_dump else (args.dump_dir or (cfg.get("privacy") or {}).get("local_dump_dir"))
    dump = LocalDump(dump_root, tag="try")

    report: dict[str, Any] = {"mode": args.mode, "t": args.t, "candidate_type": args.type}
    be, reason = get_backend(cfg)
    if be is None:
        report.update(status="backend_unavailable", reason=reason)
        return report
    ok, why = ensure_loaded(be)
    report["backend"] = be.info()
    if not ok:
        report.update(status="backend_load_failed", reason=why)
        return report

    clip = open_clip(args.video)
    report["clip"] = {"duration": round(float(clip.duration), 2), "fps": clip.fps, "size": [clip.width, clip.height]}
    texts: dict[str, Any] = {}
    if args.mode in ("verify", "both"):
        t0, t1 = max(0.0, args.t - args.span), args.t + args.span
        ev = Event(type_id=args.type, t_start=t0, t_end=t1, t_peak=args.t, confidence=args.conf, source="manual")
        out = verify_events(clip, [ev], args.budget, cfg=cfg, backend=be, dump=dump, mode=args.verify_mode)[0]
        v = out.extra.get("vlm") or {}
        report["verify"] = {k: v.get(k) for k in ("status", "mode", "agreement", "verdict", "score", "contradiction",
                                                  "vlm_confidence", "p_prior", "p_post", "p_support", "p_contra",
                                                  "evidence", "suggested_type_id", "t_best", "t_visible",
                                                  "t_transition", "asked", "low_mass", "crops_used", "flags",
                                                  "dropped_facts", "attempts", "latency_s", "repaired_json",
                                                  "finish_reason", "complete")}
        report["verify"]["n_facts"] = len(v.get("facts") or [])
        report["verify"]["budget"] = v.get("budget")
        texts["verify_facts"] = v.get("facts") or []
        texts["verify_uncertain_because"] = v.get("uncertain_because")
    if args.mode in ("discover", "both"):
        rng = (max(0.0, args.t - 6.0), args.t + 6.0)
        found = discover_events(clip, args.budget, cfg=cfg, backend=be, force=True, dump=dump, t_range=rng)
        report["discover"] = {"n_events": len(found),
                              "events": [{"type_id": e.type_id, "t_peak": e.t_peak, "confidence": e.confidence,
                                          "agreement": (e.extra.get("vlm") or {}).get("agreement"),
                                          "n_facts": len(e.facts)} for e in found],
                              "budget": (found[0].extra["vlm"].get("budget") if found else None)}
        texts["discover_facts"] = [e.facts for e in found]
    dump.write_json("report.json", {**report, "texts": texts})
    report["dump_dir"] = dump.path
    if args.show_text:
        report["texts"] = texts
    return report


def main(argv: Optional[list[str]] = None) -> int:
    rep = run(argv)
    json.dump(rep, sys.stdout, ensure_ascii=False, indent=2, default=str)
    sys.stdout.write("\n")
    return 0 if str(rep.get("status", "ok")) not in ("backend_unavailable", "backend_load_failed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
