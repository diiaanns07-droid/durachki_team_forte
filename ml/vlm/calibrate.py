"""python -m ml.vlm.calibrate --videos DIR --labels labels.csv — калибровка VLM на локальной разметке.

Зачем: решить, включать ли discover, и подобрать verify.vlm_weight/max_raise по ИЗМЕРЕНИЯМ, а не на глаз.
Для каждого ролика с нарушением берётся окно вокруг размеченной секунды (позитив), для чистых роликов —
случайные моменты (негатив). На каждом окне:
  * discover-вызов (чек-лист) -> «нашёл что-то» / тип;
  * verify-вызов кандидата V99 («иное видимое нарушение») -> agreement.
В stdout — ТОЛЬКО агрегаты (доли срабатываний на позитивах/негативах, задержки, валидность JSON).
Поролевые результаты, кадры и ответы — только в локальный дамп (privacy.local_dump_dir).

Формат labels.csv (разделитель , или ;, заголовок обязателен; регистр/язык значений не важен):
    file,label,t
    X-1.mov,нарушение,22
    Х-3.mov,без нарушения,
Имена нормализуются (кириллическая «Х» -> латинская «X», регистр, расширение); дубликаты по md5 пропускаются.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Optional

# кириллические «двойники» латиницы в именах файлов (Х-3 vs X-1)
_CYR2LAT = str.maketrans({"Х": "X", "х": "x", "А": "A", "а": "a", "В": "B", "Е": "E", "е": "e", "К": "K", "М": "M",
                          "Н": "H", "О": "O", "о": "o", "Р": "P", "р": "p", "С": "C", "с": "c", "Т": "T", "У": "Y",
                          "у": "y"})


def normalize_name(name: str) -> str:
    stem = Path(str(name).strip()).stem
    return re.sub(r"\s+", "", stem.translate(_CYR2LAT)).lower()


def parse_label(v: str) -> Optional[bool]:
    s = str(v or "").strip().lower()
    if not s:
        return None
    if s.startswith(("без", "no", "clean", "0", "нет", "false")):
        return False
    if s.startswith(("наруш", "viol", "1", "да", "true", "yes")):
        return True
    return None


def parse_t(v: str) -> Optional[float]:
    m = re.search(r"\d+(?:[.,]\d+)?", str(v or ""))
    return float(m.group(0).replace(",", ".")) if m else None


def read_labels(path: str) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8-sig")
    dialect = csv.Sniffer().sniff(text.splitlines()[0], delimiters=",;\t") if text else csv.excel
    rows = []
    for r in csv.DictReader(text.splitlines(), dialect=dialect):
        r = {str(k).strip().lower(): v for k, v in r.items() if k}
        f = r.get("file") or r.get("video") or r.get("name") or r.get("файл")
        lab = parse_label(r.get("label") or r.get("verdict") or r.get("метка") or "")
        if not f or lab is None:
            continue
        rows.append({"key": normalize_name(f), "violation": lab, "t": parse_t(r.get("t") or r.get("time") or r.get("sec") or "")})
    return rows


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rate(xs: list[bool]) -> Optional[float]:
    return round(sum(xs) / len(xs), 3) if xs else None


def run(argv: Optional[list[str]] = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(prog="python -m ml.vlm.calibrate", description=__doc__.split("\n\n")[0])
    ap.add_argument("--videos", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--neg-per-video", type=int, default=2)
    ap.add_argument("--half-window", type=float, default=2.0, help="окно позитива: t ± half-window")
    ap.add_argument("--budget", type=float, default=60.0, help="бюджет на один вызов-окно, с")
    ap.add_argument("--modes", default="discover,verify")
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--no-dump", action="store_true")
    ap.add_argument("--out", default=None, help="агрегаты в JSON-файл")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

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
    cfg = load_config(args.config, overrides=over or None)
    modes = {m.strip() for m in args.modes.split(",") if m.strip()}
    rep: dict[str, Any] = {"modes": sorted(modes), "backend_cfg": cfg.get("backend"), "preset": cfg.get("preset")}
    be, reason = get_backend(cfg)
    if be is None:
        rep.update(status="backend_unavailable", reason=reason)
        return rep
    ok, why = ensure_loaded(be)
    if not ok:
        rep.update(status="backend_load_failed", reason=why)
        return rep
    rep["backend"] = be.info()

    labels = {r["key"]: r for r in read_labels(args.labels)}
    vids = sorted(p for p in Path(args.videos).iterdir() if p.suffix.lower() in (".mov", ".mp4", ".avi", ".mkv"))
    rng = random.Random(args.seed)
    dump = LocalDump(None if args.no_dump else (args.dump_dir or (cfg.get("privacy") or {}).get("local_dump_dir")),
                     tag="calibrate")
    seen_md5: set[str] = set()
    pos_disc, neg_disc, pos_ver, neg_ver = [], [], [], []
    per_item: dict[str, dict[str, int]] = {}
    lat: list[float] = []
    json_ok: list[bool] = []
    n_dupes = n_unlabeled = n_used = 0
    for p in vids:
        lab = labels.get(normalize_name(p.name))
        if lab is None:
            n_unlabeled += 1
            continue
        h = _md5(p)
        if h in seen_md5:
            n_dupes += 1
            continue
        seen_md5.add(h)
        if args.max_videos and n_used >= args.max_videos:
            break
        n_used += 1
        clip = open_clip(str(p))
        dur = float(clip.duration)
        moments: list[tuple[float, bool]] = []
        if lab["violation"] and lab["t"] is not None:
            moments.append((min(max(lab["t"], 0.0), dur), True))
        elif not lab["violation"]:
            moments += [(rng.uniform(args.half_window, max(args.half_window, dur - args.half_window)), False)
                        for _ in range(max(0, args.neg_per_video))]
        for (t, is_pos) in moments:
            rec: dict[str, Any] = {"video": p.name, "t": round(t, 2), "positive": is_pos}
            if "discover" in modes:
                found = discover_events(clip, args.budget, cfg=cfg, backend=be, force=True,
                                        t_range=(max(0.0, t - args.half_window), t + args.half_window),
                                        limits={"max_calls": 1, "max_retries_total": 1}, dump=dump)
                hit = bool(found)
                (pos_disc if is_pos else neg_disc).append(hit)
                for e in found:
                    item = (e.extra.get("vlm_discover") or {}).get("item", "?")
                    d = per_item.setdefault(item, {"pos": 0, "neg": 0})
                    d["pos" if is_pos else "neg"] += 1
                rec["discover"] = [{"type_id": e.type_id, "conf": e.confidence} for e in found]
            if "verify" in modes:
                ev = Event("V99", max(0.0, t - args.half_window), t + args.half_window, t, 0.5, "calibration")
                out = verify_events(clip, [ev], args.budget, cfg=cfg, backend=be, dump=dump)[0]
                v = out.extra.get("vlm") or {}
                json_ok.append(v.get("status") == "ok")
                if v.get("latency_s") is not None:
                    lat.append(float(v["latency_s"]))
                (pos_ver if is_pos else neg_ver).append(v.get("agreement") in ("supports", "type_mismatch"))
                rec["verify"] = {k: v.get(k) for k in ("status", "agreement", "vlm_confidence", "suggested_type_id")}
            dump(dict(rec, kind="calib"))  # поролевые результаты — только локально
        close = getattr(clip, "close", None)
        if callable(close):
            close()
    rep.update({
        "status": "ok",
        "videos_used": n_used, "videos_unlabeled": n_unlabeled, "duplicates_skipped": n_dupes,
        "discover_hit_rate_pos": _rate(pos_disc), "discover_hit_rate_neg": _rate(neg_disc),
        "discover_n_pos": len(pos_disc), "discover_n_neg": len(neg_disc),
        "discover_items": per_item,
        "verify_support_rate_pos": _rate(pos_ver), "verify_support_rate_neg": _rate(neg_ver),
        "verify_n_pos": len(pos_ver), "verify_n_neg": len(neg_ver),
        "verify_json_ok_rate": _rate(json_ok),
        "verify_latency_p50_s": round(statistics.median(lat), 3) if lat else None,
        "verify_latency_max_s": round(max(lat), 3) if lat else None,
        "dump_dir": dump.path,
    })
    if args.out:
        Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return rep


def main(argv: Optional[list[str]] = None) -> int:
    rep = run(argv)
    json.dump(rep, sys.stdout, ensure_ascii=False, indent=2, default=str)
    sys.stdout.write("\n")
    return 0 if rep.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
