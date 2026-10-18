"""Параметризованный replay: ТЕКУЩИЙ режим (baseline) против ОДНОГО изменения (variant).

Синтетика (облако, без данных):
    python -m ml.track.replay --variant smooth [--out ml/track/_out/replay] [--seeds 3]
Реальные кэши детекций (только L2, локально; кэш стадий A03 = cache/<sha256>/dets.jsonl + manifest.json):
    python -m ml.track.replay --real --dets-glob "cache/*/dets.jsonl" --videos-dir <папка с роликами> \
        --labels labels.csv --variant smooth --out ml/track/_out/replay_real

Метрики:
  ID: fragmentation (лишние ID на GT-человека), switches (трек собран из 2+ людей), spurious (трек из шума);
  события (±tol с): TP/FP/FN и |Δt| по person_entered, cashier_left_workplace, cashier_left_room,
  at_cashier_side (outsider|unknown); latency стадии (мс). VRAM: стадия CPU-only (0).
Real-режим (нет GT по ID): FP-прокси на роликах без нарушения (ролики с любым onset «тревожных» событий),
  hit-прокси на роликах с нарушением (onset в ±tol от размеченной секунды), число треков, latency.
Цифры синтетики — не точность на реальных данных; решение о включении флага — только по --real у L2.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from . import synth as S
from .api import track_full
from .config import load_config
from .queries import TrackQueries

VARIANTS: dict[str, dict] = {
    "baseline": {},
    "smooth": {"queries": {"smoothing": "hysteresis"}},                 # H-SMOOTH — сравниваемое изменение
    "vis_bridge": {"visibility": {"bridge_tracks": True}},              # H-VIS — следующий кандидат
    "appearance": {"appearance": {"enabled": True}},                    # H-APP
    "naive_entries": {"queries": {"entry_requires_boundary": False}},   # эталон «до исправления» входов
    "no_lost_penalty": {"tracker": {"lost_cost_penalty": 0.0}},         # прежнее сопоставление (подмена ID при проходе)
    "speed_gate": {"tracker": {"speed_gate": True}},                    # H-SPEED: запрет «прыжка» трека на соседа
}
ALARM_KINDS = ("cashier_left_workplace", "cashier_left_room", "outsider_at_cashier_side")
REVIEW_KINDS = ("unknown_at_cashier_side",)   # честное «неизвестно»: на проверку, не тревога
KINDS = ("person_entered",) + ALARM_KINDS + REVIEW_KINDS

# Синтетические сценарии и их GT-события (секунды заданы самой сценой, а не алгоритмом)
SCENARIOS: dict[str, tuple[Callable[[], S.Scene], dict[str, list[float]]]] = {
    "clean": (S.scene_clean, {"person_entered": [8.0]}),
    "v01": (S.scene_v01, {"person_entered": [8.0], "unknown_at_cashier_side": [11.0]}),
    "v02": (S.scene_v02, {"cashier_left_workplace": [16.3], "cashier_left_room": [19.0], "person_entered": [40.0]}),
    "v02_bounce": (S.scene_v02_bounce, {"cashier_left_workplace": [16.0], "cashier_left_room": [27.0]}),
    "v07": (S.scene_v07, {"person_entered": [10.0]}),
    "two_staff": (S.scene_two_staff, {"unknown_at_cashier_side": [0.0]}),  # второй у кассы: unknown, не «посторонний»
    "fidget": (S.scene_fidget, {}),
    "cashier_enters": (S.scene_cashier_enters_client_seated, {"person_entered": [1.0]}),
    "crossing_front": (S.scene_crossing_front, {"person_entered": [8.0]}),
}
PERTURBATIONS: dict[str, dict] = {
    "none": {},
    "dropout": {"drop": 0.15},
    "jitter": {"jitter": 0.012},
    "occlusion": {"_miss": [(20.0, 23.0)]},
    "blackout": {"blackout": [(20.0, 25.0)]},
    "mirror": {"A": S.affine(0.0, 1.0, flip_x=True)},
    "perspective": {"H": S.homography(0.25)},
    "noise": {"noise_dets": 0.08},
}


def _cfg(variant: str) -> dict:
    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant}; known: {sorted(VARIANTS)}")
    return load_config(overrides=VARIANTS[variant])


def _kind(k: str) -> str:
    return k


def _onsets(q: TrackQueries) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for e in q.events():
        k = _kind(e.kind)
        if k in KINDS:
            out[k].append(float(e.t))
    return out


def match_events(pred: Sequence[float], gt: Sequence[float], tol: float) -> tuple[int, int, int, list[float]]:
    used: set[int] = set()
    dts = []
    tp = 0
    for g in sorted(gt):
        best, bi = None, -1
        for i, p in enumerate(pred):
            if i in used or abs(p - g) > tol:
                continue
            if best is None or abs(p - g) < best:
                best, bi = abs(p - g), i
        if bi >= 0:
            used.add(bi)
            tp += 1
            dts.append(best)
    return tp, len(pred) - len(used), len(gt) - tp, dts


def id_metrics(dets: Sequence[Any], gt: dict, tracks: Sequence[Any]) -> dict[str, int]:
    by_track: dict[int, list] = defaultdict(list)
    by_gt: dict[str, set] = defaultdict(set)
    for d in dets:
        if d.label != "person":
            continue
        key = gt.get(id(d), {}).get("key")
        if d.track_id is not None:
            by_track[d.track_id].append(key)
            if key is not None:
                by_gt[key].add(d.track_id)
    switches = spurious = 0
    for keys in by_track.values():
        real = [k for k in keys if k is not None]
        if len(real) < 0.5 * len(keys):
            spurious += 1
            continue
        counts = sorted((real.count(k) for k in set(real)), reverse=True)
        if len(counts) > 1 and counts[1] >= 0.2 * len(real):
            switches += 1
    frag = sum(max(0, len(v) - 1) for v in by_gt.values())
    return {"gt_persons": len(by_gt), "person_tracks": len(by_track), "fragmentation": frag,
            "switches": switches, "spurious": spurious}


def run_synthetic(variants: Sequence[str], seeds: int, tol: float) -> dict:
    rows = []
    for variant in variants:
        cfg = _cfg(variant)
        for sname, (sfn, gt_ev) in SCENARIOS.items():
            for pname, pk in PERTURBATIONS.items():
                for seed in range(seeds):
                    scene = sfn()
                    kw = dict(pk)
                    miss = kw.pop("_miss", None)
                    if miss and scene.persons:
                        scene.persons[0].miss = list(miss)
                    clip, dets, gt = S.make(scene, seed=seed, **kw)
                    t0 = time.perf_counter()
                    res = track_full(clip, dets, cfg=cfg)
                    q = TrackQueries(res.tracks, res.zones, res.info, cfg)
                    pred = _onsets(q)
                    ms = (time.perf_counter() - t0) * 1000
                    row = {"variant": variant, "scenario": sname, "perturbation": pname, "seed": seed,
                           "latency_ms": round(ms, 1), **id_metrics(dets, gt, res.tracks)}
                    for k in KINDS:
                        tp, fp, fn, dts = match_events(pred.get(k, []), gt_ev.get(k, []), tol)
                        row[f"{k}.tp"], row[f"{k}.fp"], row[f"{k}.fn"] = tp, fp, fn
                        row[f"{k}.dt"] = dts
                    rows.append(row)
    return {"mode": "synthetic", "tol_s": tol, "seeds": seeds, "rows": rows, "summary": summarize(rows)}


def summarize(rows: Sequence[dict]) -> dict:
    out: dict[str, dict] = {}
    for variant in sorted({r["variant"] for r in rows}):
        rs = [r for r in rows if r["variant"] == variant]
        s: dict[str, Any] = {"runs": len(rs)}
        for m in ("fragmentation", "switches", "spurious"):
            s[m] = int(sum(r[m] for r in rs))
        for k in KINDS:
            tp, fp, fn = (int(sum(r[f"{k}.{x}"] for r in rs)) for x in ("tp", "fp", "fn"))
            dts = [d for r in rs for d in r[f"{k}.dt"]]
            s[k] = {"tp": tp, "fp": fp, "fn": fn, "mean_abs_dt_s": round(float(np.mean(dts)), 3) if dts else None}
        s["alarm_fp"] = int(sum(s[k]["fp"] for k in ALARM_KINDS))
        s["alarm_fn"] = int(sum(s[k]["fn"] for k in ALARM_KINDS))
        lat = [r["latency_ms"] for r in rs]
        s["latency_ms_p50"] = round(float(np.median(lat)), 1)
        s["latency_ms_max"] = round(float(np.max(lat)), 1)
        by_p: dict[str, dict] = {}
        for p in sorted({r["perturbation"] for r in rs}):
            rp = [r for r in rs if r["perturbation"] == p]
            by_p[p] = {"alarm_fp": int(sum(r[f"{k}.fp"] for r in rp for k in ALARM_KINDS)),
                       "alarm_fn": int(sum(r[f"{k}.fn"] for r in rp for k in ALARM_KINDS)),
                       "entry_fp": int(sum(r["person_entered.fp"] for r in rp)),
                       "fragmentation": int(sum(r["fragmentation"] for r in rp)),
                       "switches": int(sum(r["switches"] for r in rp))}
        s["by_perturbation"] = by_p
        out[variant] = s
    return out


# ---------------- real (L2) ----------------
def norm_name(name: str) -> str:
    """X-1 (латиница) == Х-1 (кириллица); регистр и расширение не важны."""
    n = Path(str(name)).stem.strip().lower()
    return n.replace("х", "x").replace("Х", "x")


def load_labels(path: Optional[str]) -> dict[str, dict]:
    """Ключи — нормализованные имя файла и video_id, а также sha256 (если есть такая колонка).

    Понимает схему eval.labels (tools/dataprep/build_labels.py -> data/labels.csv):
    video_id, file, label, t_sec, md5, group, t_all, conflict, raw. t_all — все таймкоды группы
    дубликатов; conflict=1 — разные таймкоды или метки у дубликатов (решает человек).
    """
    if not path:
        return {}
    out: dict[str, dict] = {}
    num = re.compile(r"\d+(?:[.,]\d+)?")
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            low = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            name = low.get("video") or low.get("file") or low.get("filename") or low.get("name") or low.get("video_id") or ""
            lab = (low.get("label") or low.get("violation") or low.get("verdict") or "").lower()
            viol = lab in ("1", "true", "yes", "violation", "нарушение", "да")
            raw_t = low.get("second") or low.get("t_sec") or low.get("t") or low.get("timecode") or ""
            secs = [float(x.replace(",", ".")) for x in num.findall(raw_t)[:1]]
            secs += [float(x.replace(",", ".")) for x in num.findall(low.get("t_all") or "")]
            entry = out.setdefault(norm_name(name), {"violation": viol, "seconds": [], "conflict": False})
            for s in secs:
                if s not in entry["seconds"]:
                    entry["seconds"].append(s)
            entry["conflict"] = entry["conflict"] or low.get("conflict", "") in ("1", "true", "yes")
            if low.get("video_id"):
                out.setdefault(norm_name(low["video_id"]), entry)
            if low.get("sha256"):
                out[low["sha256"].lower()] = entry
    return out


VIDEO_EXT = (".mov", ".mp4", ".avi", ".mkv", ".m4v")


def sha_names(videos_dir: Optional[str]) -> dict[str, str]:
    """sha256 файла -> имя ролика: каталоги кэша A03 называются по sha256, а метки — по имени файла.
    Считается ЛОКАЛЬНО у L2 по своим роликам; видео никуда не передаются."""
    if not videos_dir:
        return {}
    import hashlib

    out = {}
    for p in sorted(Path(videos_dir).rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXT:
            h = hashlib.sha256()
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            out[h.hexdigest()] = p.name
    return out


class _CacheClip:
    """Мета ролика из кэша (без кадров): width/height/duration из meta.json | manifest.json["video"] (A03);
    times — из meta, иначе обработанные кадры = моменты детекций (статичные предметы видны почти всегда)."""

    def __init__(self, d: Path, dets: Sequence[Any], aspect: float) -> None:
        from .serialize import load_clip_meta

        meta = load_clip_meta(str(d))
        self.width = meta.get("width") or int(round(aspect * 1000))
        self.height = meta.get("height") or 1000
        self.times = meta.get("times") or sorted({round(float(x.t), 4) for x in dets})
        self.duration = meta.get("duration") or (self.times[-1] if self.times else 0.0)


def run_real(variants: Sequence[str], dets_glob: str, labels: dict, tol: float, aspect: float,
             names: Optional[dict[str, str]] = None) -> dict:
    from .serialize import load_detections

    rows = []
    names = names or {}
    paths = sorted(glob.glob(dets_glob))
    if not paths:  # раньше: молча 0 роликов
        raise SystemExit(f"replay --real: по шаблону {dets_glob!r} не найдено ни одного файла детекций "
                         "(кэш A03: cache/<sha256>/dets.jsonl; корень — $SENTINEL_CACHE_DIR или <repo>/cache)")
    for variant in variants:
        cfg = _cfg(variant)
        for p in paths:
            d = Path(p).parent
            dets = load_detections(str(d)) or []
            clip = _CacheClip(d, dets, aspect)
            t0 = time.perf_counter()
            res = track_full(clip, dets, cfg=cfg)
            q = TrackQueries(res.tracks, res.zones, res.info, cfg)
            pred = _onsets(q)
            ms = (time.perf_counter() - t0) * 1000
            vid = names.get(d.name, d.name)
            lab = (labels.get(norm_name(vid)) or labels.get(d.name.lower()) or labels.get(norm_name(Path(p).stem)) or {})
            alarm = sorted(t for k in ALARM_KINDS for t in pred.get(k, []))
            near = [min(abs(a - s) for a in alarm) for s in lab.get("seconds", []) if alarm]
            rows.append({"variant": variant, "video": vid, "cache_dir": d.name, "violation": lab.get("violation"),
                         "label_conflict": bool(lab.get("conflict")),
                         "label_seconds": lab.get("seconds", []), "latency_ms": round(ms, 1),
                         "person_tracks": sum(1 for t in res.tracks if t.label == "person"),
                         "n_events": {k: len(pred.get(k, [])) for k in KINDS},
                         "first_alarm_s": alarm[0] if alarm else None,
                         "hit": bool(near and min(near) <= tol), "nearest_dt_s": round(min(near), 2) if near else None,
                         "visibility_loss": res.info["coverage"].get("visibility_loss"),
                         "roles": {t.track_id: [t.role, t.role_conf] for t in res.tracks if t.label == "person"}})
    summ = {}
    for variant in variants:
        rs = [r for r in rows if r["variant"] == variant]
        clean = [r for r in rs if r["violation"] is False]
        viol = [r for r in rs if r["violation"] is True]
        summ[variant] = {
            "videos": len(rs), "labeled_clean": len(clean), "labeled_violation": len(viol),
            "clean_with_alarm (FP-proxy)": sum(1 for r in clean if r["first_alarm_s"] is not None),
            "violation_without_alarm (FN-proxy)": sum(1 for r in viol if r["first_alarm_s"] is None),
            f"violation_hit_within_{tol}s": sum(1 for r in viol if r["hit"]),
            "mean_person_tracks": round(float(np.mean([r["person_tracks"] for r in rs])), 2) if rs else None,
            "latency_ms_p50": round(float(np.median([r["latency_ms"] for r in rs])), 1) if rs else None,
            "vram": "0 (stage is CPU-only)"}
    for variant in variants:
        summ[variant]["unlabeled_videos"] = sum(1 for r in rows if r["variant"] == variant and r["violation"] is None)
        summ[variant]["label_conflicts"] = sum(1 for r in rows if r["variant"] == variant and r["label_conflict"])
    return {"mode": "real", "tol_s": tol, "rows": rows, "summary": summ,
            "note": "Все 42 строки уже использованы в разведке — это не независимый тест."}


def to_markdown(rep: dict) -> str:
    lines = [f"# ml.track replay ({rep['mode']}), tol={rep['tol_s']} s", ""]
    summ = rep["summary"]
    vs = list(summ)
    if rep["mode"] == "synthetic":
        lines += ["| metric | " + " | ".join(vs) + " |", "|---|" + "---|" * len(vs)]
        keys = ["fragmentation", "switches", "spurious", "alarm_fp", "alarm_fn", "latency_ms_p50", "latency_ms_max"]
        for k in keys:
            lines.append(f"| {k} | " + " | ".join(str(summ[v][k]) for v in vs) + " |")
        for k in KINDS:
            lines.append(f"| {k} tp/fp/fn (|Δt|) | " + " | ".join(
                f"{summ[v][k]['tp']}/{summ[v][k]['fp']}/{summ[v][k]['fn']} ({summ[v][k]['mean_abs_dt_s']})" for v in vs) + " |")
        lines += ["", "## by perturbation (alarm_fp / alarm_fn / entry_fp / fragmentation / switches)", "",
                  "| perturbation | " + " | ".join(vs) + " |", "|---|" + "---|" * len(vs)]
        for p in summ[vs[0]]["by_perturbation"]:
            lines.append(f"| {p} | " + " | ".join(
                "{alarm_fp}/{alarm_fn}/{entry_fp}/{fragmentation}/{switches}".format(**summ[v]["by_perturbation"][p]) for v in vs) + " |")
        lines += ["", "_Синтетика: цифры показывают поведение кода, а не точность на реальных роликах._"]
    else:
        keys = list(next(iter(summ.values())).keys())
        lines += ["| metric | " + " | ".join(vs) + " |", "|---|" + "---|" * len(vs)]
        for k in keys:
            lines.append(f"| {k} | " + " | ".join(str(summ[v][k]) for v in vs) + " |")
        lines += ["", f"_{rep['note']}_"]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ml.track.replay")
    ap.add_argument("--variant", default="smooth", help=f"одно изменение против baseline: {sorted(VARIANTS)}")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tol", type=float, default=2.0, help="допуск таймкода, с")
    ap.add_argument("--out", default="ml/track/_out/replay")
    ap.add_argument("--real", action="store_true", help="реальные кэши детекций (только L2)")
    ap.add_argument("--dets-glob", default="cache/*/dets.jsonl", help="кэш A03: cache/<sha256>/dets.jsonl")
    ap.add_argument("--labels", default=None, help="CSV: video|file, label|violation, second|t[, sha256]")
    ap.add_argument("--videos-dir", default=None, help="локальная папка роликов: sha256 -> имя для меток (только L2)")
    ap.add_argument("--aspect", type=float, default=464 / 848, help="width/height, если в кэше нет meta.json")
    args = ap.parse_args(argv)
    variants = [args.baseline, args.variant] if args.variant != args.baseline else [args.baseline]
    rep = (run_real(variants, args.dets_glob, load_labels(args.labels), args.tol, args.aspect,
                    sha_names(args.videos_dir)) if args.real
           else run_synthetic(variants, args.seeds, args.tol))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "replay.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    md = to_markdown(rep)
    (out / "replay.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"written {out / 'replay.json'} and {out / 'replay.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
