"""python -m ml.vlm.compare — сколько ошибок VLM исправляет и создаёт и сколько это стоит (для L2).

Сравнивает режимы на ОДНИХ И ТЕХ ЖЕ кандидатах пайплайна A03 (окна — от A03/A04, не от разметки):
  none     — без VLM (rules + temporal -> fusion);
  json     — verify.mode=json (фаза 1);
  qa       — verify.mode=qa (фаза 2: нейтральные вопросы + логиты);
  discover — json + ограниченный discover (только если нужен; по умолчанию не запускается).
Вердикт/таймкод/тип считает настоящий ml.fusion.fuse (A12), если он есть; иначе — порог по max confidence.

Кандидаты (без VLM) готовит A03, по одному каталогу на ролик:
    python -m backend.pipeline run <video> --no-vlm --out preds/<stem>.json --dump-intermediate cands/<stem>
Каталог содержит events_rules.json, events_temporal.json (или events_all.json), zones.json, risk.json.
Либо JSONL: {"video": "<имя>", "events": [...], "zones": [...], "risk": [[t, r], ...]} на строку.

В stdout и --out — ТОЛЬКО агрегаты; построчные результаты по роликам — в локальный дамп.
Все 42 строки разметки уже использовались в разведочной оценке: это НЕ независимый тест.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional

from .calibrate import _md5, normalize_name, read_labels

MODES = ("none", "json", "qa", "discover")
TOLERANCES = (1.0, 2.0, 3.0, 5.0)


def _load_events(items: Any) -> list[Any]:
    from ._contract import Event

    out = []
    for d in items or []:
        if isinstance(d, Event):
            out.append(d)
            continue
        out.append(Event(type_id=str(d["type_id"]), t_start=float(d["t_start"]), t_end=float(d["t_end"]),
                         t_peak=float(d["t_peak"]), confidence=float(d["confidence"]), source=str(d.get("source", "")),
                         facts=[str(f) for f in d.get("facts") or []],
                         bboxes={str(k): tuple(float(x) for x in v) for k, v in (d.get("bboxes") or {}).items()},
                         track_ids=[int(x) for x in d.get("track_ids") or []], extra=dict(d.get("extra") or {})))
    return out


def _load_zones(items: Any) -> list[Any]:
    from ._contract import Zone

    out = []
    for z in items or []:
        if isinstance(z, dict) and z.get("polygon"):
            out.append(Zone(name=str(z.get("name", "")), polygon=[tuple(map(float, p)) for p in z["polygon"]],
                            conf=float(z.get("conf", 1.0))))
    return out


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_candidates(path: str) -> dict[str, dict[str, Any]]:
    """-> {нормализованное имя: {"events", "zones", "risk"}} из каталога дампов A03 или JSONL."""
    src = Path(path)
    out: dict[str, dict[str, Any]] = {}
    if src.is_file():
        for line in src.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                out[normalize_name(d["video"])] = {"events": _load_events(d.get("events")),
                                                   "zones": _load_zones(d.get("zones")), "risk": d.get("risk") or []}
        return out
    for sub in sorted(p for p in src.iterdir() if p.is_dir()):
        ev_all = _read_json(sub / "events_all.json")
        if ev_all is None:
            ev_all = (_read_json(sub / "events_rules.json") or []) + (_read_json(sub / "events_temporal.json") or [])
        # кандидаты для verify — только rules/temporal (на случай, если дамп сделан с включённой VLM)
        events = [e for e in _load_events(ev_all) if not str(e.source).startswith("vlm")]
        risk = _read_json(sub / "risk.json") or []
        out[normalize_name(sub.name)] = {"events": events, "zones": _load_zones(_read_json(sub / "zones.json")),
                                         "risk": [(float(t), float(r)) for t, r in risk]}
    return out


def _verdict(clip: Any, events_fusion: list[Any], events_plain: list[Any], risk: list[Any], threshold: float,
             catalog: Optional[dict], how: str = "auto") -> dict[str, Any]:
    """Вердикт как в продукте: ml.fusion.fuse (A12) на всех событиях, как их отдаёт A03 (кандидаты + проверенные
    копии). Без fusion — max confidence >= threshold по кандидатам, заменённым проверенными копиями.
    verdict=None (анализ неполный/упал) — это «неизвестно», а не no_violation."""
    try:
        if how == "threshold":
            raise ImportError("verdict=threshold")
        from ml.fusion import fuse  # type: ignore

        try:
            res = fuse(clip, events_fusion, risk, catalog or {}, "compare")
            top = max(res.violations, key=lambda v: float(getattr(v, "confidence", 0.0)), default=None)
            return {"verdict": res.verdict, "t": getattr(top, "timecode_sec", None),
                    "type_id": getattr(top, "type_id", None), "n_violations": len(res.violations), "by": "fusion",
                    "complete": (res.meta or {}).get("complete", None)}
        except Exception as e:  # noqa: BLE001
            logging.getLogger(__name__).warning("fusion failed: %s", type(e).__name__)
            by = "threshold_after_fusion_error"
    except ImportError:
        if how == "fusion":
            raise
        by = "threshold"
    top = max(events_plain, key=lambda e: float(e.confidence), default=None)
    is_v = top is not None and float(top.confidence) >= threshold
    return {"verdict": "violation" if is_v else "no_violation", "t": top.t_peak if is_v else None,
            "type_id": top.type_id if is_v else None, "n_violations": int(is_v), "by": by}


def _run_mode(mode: str, clip: Any, cand: dict[str, Any], budget_s: float, cfg: dict[str, Any], backend: Any,
              catalog: Optional[dict], threshold: float, dump: Any, how: str = "auto") -> dict[str, Any]:
    from .discoverer import discover_events
    from .verifier import coverage, verify_events

    base = list(cand["events"])
    t0 = time.perf_counter()
    added: list[Any] = []
    plain = list(base)  # для порогового вердикта: кандидаты, заменённые проверенными копиями
    info: dict[str, Any] = {"vlm_calls": 0, "coverage_complete": True, "agreements": {}}
    if mode in ("json", "qa", "discover") and base:
        vmode = "qa" if mode == "qa" else "json"
        checked = verify_events(clip, base, budget_s, cfg=cfg, backend=backend, catalog=catalog,
                                zones=cand.get("zones"), mode=vmode, dump=dump)
        added += checked
        plain = list(checked)
        cov = coverage(checked)
        info["coverage_complete"] = cov["complete"]
        info["vlm_calls"] += int(((checked[0].extra.get("vlm") or {}).get("budget") or {}).get("calls") or 0)
        for e in checked:
            a = str((e.extra.get("vlm") or {}).get("agreement", "not_checked"))
            info["agreements"][a] = info["agreements"].get(a, 0) + 1
    if mode == "discover":
        found = discover_events(clip, budget_s, cfg=cfg, backend=backend, catalog=catalog, force=True, dump=dump)
        added += found
        plain += found
        info["discover_events"] = len(found)
        if found:
            info["vlm_calls"] += int((found[0].extra["vlm"].get("budget") or {}).get("calls") or 0)
    vlm_s = time.perf_counter() - t0
    out = _verdict(clip, base + added, plain, cand.get("risk") or [], threshold, catalog, how)
    out.update(info)
    out["vlm_s"] = round(vlm_s, 3)
    return out


def _vram_peak_mb() -> Optional[float]:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except ImportError:
        pass
    return None


_DEFINITE = ("violation", "no_violation")


def summarize(rows: list[dict[str, Any]], modes: list[str]) -> dict[str, Any]:
    """Агрегаты по режимам + исправленные/созданные ошибки относительно режима none.

    Режимы сравниваются на ОДНОМ наборе роликов: тех, где вердикт определён во всех запрошенных режимах
    (common). Ролики с неопределённым вердиктом хотя бы в одном режиме считаются отдельно (unknown_verdicts)
    и не превращаются в «нарушений нет». Самопроверка: tp_m - tp_none == fn_fixed - fn_created и
    fp_m - fp_none == fp_created - fp_fixed (на common) -> vs_none.consistent.
    """
    rows_m = [r for r in rows if all(m in r["modes"] for m in modes)]
    common = [r for r in rows_m if all(r["modes"][m]["verdict"] in _DEFINITE for m in modes)]
    summary: dict[str, Any] = {"_common": {"videos_total": len(rows), "videos_compared": len(common),
                                           "excluded_unknown": len(rows_m) - len(common)}}
    for m in modes:
        unknown = sum(r["modes"][m]["verdict"] not in _DEFINITE for r in rows_m)
        rs = common
        tp = sum(r["gt"] and r["modes"][m]["verdict"] == "violation" for r in rs)
        fn = sum(r["gt"] and r["modes"][m]["verdict"] != "violation" for r in rs)
        fp = sum((not r["gt"]) and r["modes"][m]["verdict"] == "violation" for r in rs)
        tn = sum((not r["gt"]) and r["modes"][m]["verdict"] != "violation" for r in rs)
        dts = [abs(r["modes"][m]["t"] - r["t_gt"]) for r in rs
               if r["gt"] and r["t_gt"] is not None and not r.get("t_conflict")
               and r["modes"][m]["verdict"] == "violation" and r["modes"][m]["t"] is not None]
        lat = [r["modes"][m]["vlm_s"] for r in rows_m]  # стоимость — по всем роликам режима
        types: dict[str, int] = {}
        for r in rs:
            if r["modes"][m]["verdict"] == "violation":
                tid = str(r["modes"][m].get("type_id"))
                types[tid] = types.get(tid, 0) + 1
        s = {"n": len(rs), "unknown_verdicts": unknown, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
             "accuracy": round((tp + tn) / len(rs), 3) if rs else None,
             "recall": round(tp / (tp + fn), 3) if tp + fn else None,
             "specificity": round(tn / (tn + fp), 3) if tn + fp else None,
             "timecode_hit": {f"<={tol:g}s": round(sum(d <= tol for d in dts) / len(dts), 3) if dts else None
                              for tol in TOLERANCES},
             "timecode_abs_err_median_s": round(statistics.median(dts), 2) if dts else None,
             "types_predicted": types,
             "vlm_s_mean": round(statistics.mean(lat), 3) if lat else None,
             "vlm_s_p95": round(sorted(lat)[max(0, int(round(0.95 * (len(lat) - 1))))], 3) if lat else None,
             "vlm_s_max": round(max(lat), 3) if lat else None,
             "vlm_calls_total": sum(r["modes"][m].get("vlm_calls", 0) for r in rows_m),
             "coverage_incomplete_videos": sum(not r["modes"][m].get("coverage_complete", True) for r in rows_m),
             "verdict_by": sorted({str(r["modes"][m].get("by")) for r in rows_m})}
        summary[m] = s
    if "none" in modes:
        base = summary["none"]
        for m in modes:
            if m == "none":
                continue
            pairs = [(r["gt"], r["modes"]["none"]["verdict"] == "violation", r["modes"][m]["verdict"] == "violation")
                     for r in common]
            vs = {"fp_fixed": sum((not g) and b and not v for g, b, v in pairs),
                  "fn_fixed": sum(g and (not b) and v for g, b, v in pairs),
                  "fp_created": sum((not g) and (not b) and v for g, b, v in pairs),
                  "fn_created": sum(g and b and not v for g, b, v in pairs)}
            s = summary[m]
            vs["consistent"] = (s["tp"] - base["tp"] == vs["fn_fixed"] - vs["fn_created"]
                                and s["fp"] - base["fp"] == vs["fp_created"] - vs["fp_fixed"]
                                and s["n"] == base["n"])
            s["vs_none"] = vs
    return summary


def run(argv: Optional[list[str]] = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(prog="python -m ml.vlm.compare", description=__doc__.split("\n\n")[0])
    ap.add_argument("--videos", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--candidates", required=True, help="каталог дампов A03 (--dump-intermediate) или JSONL")
    ap.add_argument("--modes", default="none,json,qa")
    ap.add_argument("--budget", type=float, default=30.0, help="бюджет VLM на ролик, с (как даёт A03)")
    ap.add_argument("--threshold", type=float, default=0.5, help="порог вердикта, если ml.fusion недоступен")
    ap.add_argument("--verdict", choices=["auto", "fusion", "threshold"], default="auto",
                    help="auto: ml.fusion, если есть (как в продукте); threshold: чистый эффект VLM без гейтов fusion")
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--no-dump", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    from ._video import open_clip
    from .catalog import load_catalog
    from .config import load_config
    from .localdump import LocalDump
    from .runtime import ensure_loaded, get_backend

    modes = [m.strip() for m in args.modes.split(",") if m.strip() in MODES]
    cfg = load_config(args.config, overrides={"backend": args.backend} if args.backend else None)
    rep: dict[str, Any] = {"modes": modes, "budget_s": args.budget, "verdict_how": args.verdict,
                           "note": "разведочная оценка на уже использованных роликах — не независимый тест"}
    be = None
    if any(m != "none" for m in modes):
        be, reason = get_backend(cfg)
        if be is None:
            rep.update(status="backend_unavailable", reason=reason)
            return rep
        t0 = time.perf_counter()
        ok, why = ensure_loaded(be)
        rep["load_s"] = round(time.perf_counter() - t0, 2)
        if not ok:
            rep.update(status="backend_load_failed", reason=why)
            return rep
        rep["backend"] = be.info()
    catalog = load_catalog()
    labels = read_labels(args.labels)
    by_key: dict[str, list[dict[str, Any]]] = {}
    for r in labels:
        by_key.setdefault(r["key"], []).append(r)
    cands = load_candidates(args.candidates)
    dump = LocalDump(None if args.no_dump else (args.dump_dir or (cfg.get("privacy") or {}).get("local_dump_dir")),
                     tag="compare")
    vids = sorted(p for p in Path(args.videos).iterdir() if p.suffix.lower() in (".mov", ".mp4", ".avi", ".mkv"))
    md5_t: dict[str, set] = {}
    rows: list[dict[str, Any]] = []
    skipped = {"unlabeled": 0, "no_candidates_file": 0, "duplicate": 0}
    seen: set[str] = set()
    for p in vids:
        key = normalize_name(p.name)
        labs = by_key.get(key)
        if not labs:
            skipped["unlabeled"] += 1
            continue
        h = _md5(p)
        md5_t.setdefault(h, set()).update(r["t"] for r in labs if r["t"] is not None)
        if h in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(h)
        if key not in cands:
            skipped["no_candidates_file"] += 1
            continue
        if args.max_videos and len(rows) >= args.max_videos:
            break
        clip = open_clip(str(p))
        row: dict[str, Any] = {"video": p.name, "gt": bool(labs[0]["violation"]), "t_gt": labs[0]["t"],
                               "n_candidates": len(cands[key]["events"]), "modes": {}, "_md5": h}
        for m in modes:
            row["modes"][m] = _run_mode(m, clip, cands[key], args.budget, cfg, be, catalog, args.threshold,
                                        dump if m != "none" else None, args.verdict)
        rows.append(row)
        close = getattr(clip, "close", None)
        if callable(close):
            close()
    for r in rows:  # дубликаты с разными таймкодами — конфликт разметки, таймкод не оцениваем
        r["t_conflict"] = len(md5_t.get(r.pop("_md5"), set())) > 1
    rep.update({"status": "ok", "videos": len(rows), "skipped": skipped,
                "timecode_conflicts_excluded": sum(r["t_conflict"] for r in rows),
                "summary": summarize(rows, modes), "vram_peak_mb_torch": _vram_peak_mb(), "dump_dir": dump.path})
    dump.write_json("rows.json", rows)  # построчно — только локально
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
