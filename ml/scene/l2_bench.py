"""Одна команда замера для L2 (реальные видео и GPU есть только у L2): baseline против каждой гипотезы.

    python -X utf8 -m ml.scene.l2_bench --videos data/videos --labels data/labels.csv \\
        --out outputs/scene_bench --variants baseline,H1,H2,H3,H4,H5

1. Апстрим (ml.ingest → ml.detect → ml.track → ml.pose) считается ОДИН раз на ролик и кэшируется в
   <out>/_upstream/<id>.json; варианты A08 переигрываются на тех же детекциях (честное сравнение).
2. Каждый вариант пишет <out>/<variant>/<id>/scene_states.json (смотреть: python -m ml.scene.report) и
   <out>/<variant>/a09/<id>.states.json — раскладку кэша A09, чтобы прогнать правила на тех же состояниях:
   python -X utf8 -m ml.rules.replay --cache <out>/<variant>/a09 --labels data/labels.csv
3. Метрики (на видео; РАЗВЕДОЧНЫЕ — все 42 ролика уже смотрели, это не независимый тест):
   FP/FN по «кандидат = есть true-интервал нарушающего предиката»; попадание таймкода (|onset − GT| ≤ 2/5 с,
   дубли с конфликтом таймкодов исключены); распределение типов (подсказка, разметки типов нет);
   покрытие (доля известного); латентность A08 на ролик; пик VRAM (только H2); FP по терцилям длительности
   (ловушка «длинные ролики чаще ложно срабатывают»). Длительность/имя файла признаками не являются.

Варианты (все — флаги config.yaml, по умолчанию выкл):
  baseline — текущие умолчания;  H1 — safe/door.cluster_mode=kmeans2;  H2 — safe.namer=siglip2 (GPU);
  H3 — lights.freeze;  H4 — predicates.phone_det_only;  H5 — predicates.leave_onset=left_desk;
  H6 — presence.outsider_anchor_only (legacy outsider: клиент напротив не повышает).
Политики t_peak (onset/mid/max) считаются для каждого варианта без перезапуска.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .io import SimpleClip, dump_states, normalize_video_name
from .predicates import TYPE_HINT

VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "H1": {"safe": {"cluster_mode": "kmeans2"}, "door": {"cluster_mode": "kmeans2"}},
    "H2": {"safe": {"namer": {"backend": "siglip2"}}},
    "H3": {"lights": {"freeze": True}},
    "H4": {"predicates": {"phone_det_only": True}},
    "H5": {"predicates": {"leave_onset": "left_desk"}},
    "H6": {"presence": {"outsider_anchor_only": True}},
}


# ------------------------------------------------------------------------------------------ апстрим
def _upstream(path: Path, cache: Path, log: list[str]) -> tuple[Any, list[Any], list[Any], list[Any], list[Any], dict[str, bool]]:
    from ._contract import Detection, SceneState, Track, Zone

    used = {"ingest": False, "detect": False, "track": False, "pose": False}
    try:
        from ml.ingest import load_video  # type: ignore

        clip = load_video(str(path))
        used["ingest"] = True
    except Exception as e:  # noqa: BLE001
        log.append(f"{path.name}: ml.ingest unavailable ({e}); SimpleClip (PTS)")
        clip = SimpleClip(str(path))
    if cache.is_file():
        d = json.loads(cache.read_text(encoding="utf-8"))
        dets = [Detection(**x) for x in d["dets"]]
        tracks = [Track(**{**x, "boxes": [(float(t), tuple(b)) for t, b in x["boxes"]]}) for x in d["tracks"]]
        zones = [Zone(z["name"], [tuple(p) for p in z["polygon"]], z.get("conf", 1.0)) for z in d["zones"]]
        pose = [SceneState(t=x["t"], signals=x["signals"]) for x in d["pose"]]
        return clip, dets, tracks, zones, pose, d.get("used", used)
    dets, tracks, zones, pose = [], [], [], []
    try:
        from ml.detect import detect  # type: ignore

        dets = list(detect(clip))
        used["detect"] = True
    except Exception as e:  # noqa: BLE001
        log.append(f"{path.name}: ml.detect unavailable ({e})")
    try:
        from ml.track import track  # type: ignore

        tracks, zones = track(clip, dets)
        tracks, zones = list(tracks), list(zones)
        used["track"] = True
    except Exception as e:  # noqa: BLE001
        log.append(f"{path.name}: ml.track unavailable ({e})")
    try:
        from ml.pose import estimate, pose_signals  # type: ignore

        pose = list(pose_signals(estimate(clip, tracks), tracks, zones))
        used["pose"] = True
    except Exception as e:  # noqa: BLE001
        log.append(f"{path.name}: ml.pose unavailable ({e})")
    cache.parent.mkdir(parents=True, exist_ok=True)
    doc = {"dets": [asdict(d) for d in dets], "tracks": [asdict(t) for t in tracks], "zones": [asdict(z) for z in zones],
           "pose": [{"t": float(s.t), "signals": {k: float(v) for k, v in s.signals.items()}} for s in pose], "used": used}
    cache.write_text(json.dumps(doc), encoding="utf-8")
    return clip, dets, tracks, zones, pose, used


def _dump_for_a09(path: Path, states: list[Any], tracks: list[Any], zones: list[Any]) -> None:
    """Раскладка кэша A09 (ml/rules/cache_io.py): <cache>/<video_id>.states.json со states/tracks/zones.
    Поля SceneStateEx (known/conf/observed/status) сохраняются — A09 build_series их понимает."""
    from .types import state_to_dict

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"states": [state_to_dict(s) for s in states], "tracks": [asdict(t) for t in tracks], "zones": [asdict(z) for z in zones]}
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------------------------------ метрики
def _first_onsets(states: list[Any], policy: str) -> list[tuple[float, str]]:
    from .intervals import intervals

    iv = intervals(states, list(TYPE_HINT), kind="pred", policy=policy)
    return sorted((x["t_peak"], k) for k, v in iv.items() for x in v)


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pos = [r for r in rows if r["label"] == "violation"]
    neg = [r for r in rows if r["label"] == "no_violation"]
    tp = sum(r["candidate"] for r in pos)
    fp = sum(r["candidate"] for r in neg)
    out: dict[str, Any] = {"n": len(rows), "n_pos": len(pos), "n_neg": len(neg), "TP": tp, "FN": len(pos) - tp, "FP": fp, "TN": len(neg) - fp}
    for pol in ("onset", "mid", "max"):
        errs = [r["tc_err"][pol] for r in pos if r["tc_err"].get(pol) is not None and not r["conflict"]]
        out[f"tc_{pol}"] = {"n": len(errs), "hit2s": round(float(np.mean([e <= 2 for e in errs])), 3) if errs else None,
                            "hit5s": round(float(np.mean([e <= 5 for e in errs])), 3) if errs else None,
                            "median_err_s": round(float(np.median(errs)), 2) if errs else None}
    lat = [r["latency_s"] for r in rows]
    out["latency_s"] = {"mean": round(float(np.mean(lat)), 2), "max": round(float(np.max(lat)), 2)} if lat else None
    vr = [r["vram_mb"] for r in rows if r.get("vram_mb") is not None]
    out["vram_peak_mb"] = max(vr) if vr else None
    types: dict[str, int] = {}
    for r in rows:
        if r["type_hint"]:
            types[r["type_hint"]] = types.get(r["type_hint"], 0) + 1
    out["types"] = types
    if neg:
        durs = np.array([r["duration_s"] for r in neg])
        qs = np.quantile(durs, [1 / 3, 2 / 3]) if len(durs) >= 3 else [np.inf, np.inf]
        terc = {"short": [], "mid": [], "long": []}
        for r in neg:
            k = "short" if r["duration_s"] <= qs[0] else ("mid" if r["duration_s"] <= qs[1] else "long")
            terc[k].append(r["candidate"])
        out["fp_rate_by_duration"] = {k: (round(float(np.mean(v)), 3) if v else None) for k, v in terc.items()}
    out["known_frac_mean"] = round(float(np.mean([r["known_frac"] for r in rows])), 3) if rows else None
    return out


def read_labels(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            t_all = [float(x) for x in (r.get("t_all") or r.get("t_sec") or "").replace("|", ",").split(",") if x.strip()]
            rows.append({"id": r.get("video_id") or r.get("file"), "file": r.get("file") or "", "label": r.get("label", ""),
                         "t": t_all, "group": r.get("group") or r.get("video_id"), "conflict": str(r.get("conflict", "0")) in ("1", "true")})
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--videos", required=True)
    ap.add_argument("--labels", required=True, help="data/labels.csv (eval.labels)")
    ap.add_argument("--out", default="outputs/scene_bench")
    ap.add_argument("--variants", default="baseline,H1,H3,H4,H5,H6", help="через запятую; H2 требует GPU/весов SigLIP2")
    ap.add_argument("--limit", type=int, default=0, help="только первые N роликов (smoke)")
    ap.add_argument("--dedup", action="store_true", help="по одному ролику на md5-группу")
    args = ap.parse_args(argv)
    from .core import scene_states_with_trace
    from .intervals import coverage
    from .namer import vram_peak_mb

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = read_labels(args.labels)
    if args.dedup:
        seen: set[str] = set()
        labels = [r for r in labels if not (r["group"] in seen or seen.add(r["group"]))]
    if args.limit:
        labels = labels[: args.limit]
    vids = {normalize_video_name(p.name): p for p in Path(args.videos).rglob("*") if p.suffix.lower() in (".mov", ".mp4", ".avi")}
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    log: list[str] = []
    per: dict[str, list[dict[str, Any]]] = {v: [] for v in variants}
    used_any: dict[str, bool] = {}
    for lab in labels:
        key = normalize_video_name(lab["file"] or lab["id"])
        path = vids.get(key) or vids.get(normalize_video_name(lab["id"]))
        if path is None:
            log.append(f"{lab['id']}: video not found")
            continue
        clip, dets, tracks, zones, pose, used = _upstream(path, out / "_upstream" / f"{key}.json", log)
        used_any = {k: used_any.get(k, False) or bool(v) for k, v in used.items()}
        dur = float(getattr(clip, "duration", 0.0) or (clip.times[-1] if len(clip.times) else 0.0))
        for v in variants:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except ImportError:
                pass
            t0 = time.perf_counter()
            states, trace = scene_states_with_trace(clip, dets, tracks, zones, pose, config=VARIANTS.get(v, {}))
            lat = time.perf_counter() - t0
            dump_states(states, out / v / key / "scene_states.json", trace=trace, meta={"video_id": lab["id"], "variant": v, "gt_sec": lab["t"]})
            _dump_for_a09(out / v / "a09" / f"{lab['id']}.states.json", states, tracks, zones)
            row = {"id": lab["id"], "label": lab["label"], "conflict": lab["conflict"], "duration_s": round(dur, 2), "latency_s": round(lat, 3),
                   "vram_mb": vram_peak_mb() if v == "H2" else None, "tc_err": {}}
            for pol in ("onset", "mid", "max"):
                ons = _first_onsets(states, pol)
                if pol == "onset":
                    row["candidate"] = bool(ons)
                    row["first"] = ons[0] if ons else None
                    row["type_hint"] = TYPE_HINT.get(ons[0][1]) if ons else None
                if ons and lab["t"]:
                    row["tc_err"][pol] = round(min(abs(o - g) for o, _ in ons for g in lab["t"]), 2)
            cov = coverage(states)
            row["known_frac"] = float(np.mean([x["known_frac"] for x in cov["per_name"].values()])) if cov["per_name"] else 0.0
            row["warnings"] = trace.get("warnings", [])[:5]
            per[v].append(row)
        print(f"{lab['id']}: done ({', '.join(variants)})", flush=True)
    summary = {"exploratory": True, "note": "все ролики уже использованы в разведке — не независимый тест", "upstream_used": used_any,
               "variants": {v: {"overrides": VARIANTS.get(v, {}), **_metrics(rows)} for v, rows in per.items()}, "log": log}
    (out / "bench.json").write_text(json.dumps({"summary": summary, "per_video": per}, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# ml.scene — замер L2 (разведочный, не независимый тест)", "", f"Апстрим: {used_any}", "",
             "| вариант | TP | FN | FP | TN | onset hit@2s | onset hit@5s | med err, с | mid hit@2s | лат. сред/макс, с | VRAM, МБ | FP по длит. (кор/сред/длин) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v, m in summary["variants"].items():
        fr = m.get("fp_rate_by_duration") or {}
        lat = m.get("latency_s") or {}
        lines.append(f"| {v} | {m['TP']} | {m['FN']} | {m['FP']} | {m['TN']} | {m['tc_onset']['hit2s']} | {m['tc_onset']['hit5s']} | "
                     f"{m['tc_onset']['median_err_s']} | {m['tc_mid']['hit2s']} | {lat.get('mean')}/{lat.get('max')} | {m['vram_peak_mb']} | "
                     f"{fr.get('short')}/{fr.get('mid')}/{fr.get('long')} |")
    (out / "bench.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not used_any.get("detect") or not used_any.get("track"):
        print("ВНИМАНИЕ: ml.detect/ml.track недоступны — метрики A08 без детекций не показательны (только латентность).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
