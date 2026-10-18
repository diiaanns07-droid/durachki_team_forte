"""Опциональное дообучение сигналов safe_open / door_open / cash_visible по ПРОВЕРЕННОЙ ручной разметке.

    python -m ml.scene.train_signals --labels data/scene_labels.csv --videos data/videos \\
        --out weights/scene/scene_signals.json [--model lr|mlp] [--zones-json data/scene_zones.json]

Формат ``data/scene_labels.csv`` (UTF-8, заголовок обязателен):

    video,signal,t_start,t_end,value,verified
    X-1.mov,safe_open,0,12.5,0,1
    Х-3.mov,safe_open,12.5,30,1,1        # кириллическая Х нормализуется
    X-7,cash_visible,4.2,,1,1            # пустой t_end — точечная метка (ближайший сэмпл)

* ``signal`` ∈ {safe_open, door_open, cash_visible}; ``value`` ∈ {0, 1};
* ``verified`` — 1/true/yes/да: строку проверил человек. Непроверенные строки игнорируются
  (флаг ``--allow-unverified`` — только для экспериментов);
* участки без разметки не используются (неизвестно ≠ 0).

Входы для признаков берутся из модулей команды, если они есть (ml.ingest / ml.detect / ml.track), иначе —
SimpleClip + зоны из ``--zones-json`` ({"x-1": [{"name": "safe", "polygon": [[x, y], ...]}], "default": [...]}).
Валидация — group k-fold по видео (одно видео целиком в одном фолде). Модель — JSON, в git не коммитится.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .io import SimpleClip, normalize_video_name
from .learned import FEATURES, TRAINABLE, auc, feature_matrix, fit_lr, fit_mlp, predict

log = logging.getLogger("ml.scene.train")
_TRUE = {"1", "true", "yes", "y", "да", "+"}
VIDEO_EXT = (".mov", ".mp4", ".avi", ".mkv", ".m4v")


def read_labels(path: str | Path, allow_unverified: bool = False) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = 0
    with open(path, encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        has_verified = "verified" in (rd.fieldnames or [])
        for row in rd:
            sig = (row.get("signal") or "").strip()
            if sig not in TRAINABLE:
                skipped += 1
                continue
            if has_verified and not allow_unverified and (row.get("verified") or "").strip().lower() not in _TRUE:
                skipped += 1
                continue
            if not has_verified and not allow_unverified:
                raise ValueError("labels CSV has no 'verified' column: add it or pass --allow-unverified")
            try:
                t0 = float(row["t_start"])
                t1 = float(row["t_end"]) if (row.get("t_end") or "").strip() else t0
                val = int(float(row["value"]))
            except (KeyError, ValueError):
                skipped += 1
                continue
            if val not in (0, 1):
                skipped += 1
                continue
            out[normalize_video_name(row["video"])].append({"signal": sig, "t0": min(t0, t1), "t1": max(t0, t1), "value": val})
    if skipped:
        log.info("labels: skipped %d rows (unknown signal / unverified / malformed)", skipped)
    return dict(out)


def find_videos(root: str | Path) -> dict[str, Path]:
    res: dict[str, Path] = {}
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() in VIDEO_EXT and p.is_file():
            res.setdefault(normalize_video_name(p.name), p)
    return res


def _zones_from_json(doc: Optional[dict[str, Any]], key: str) -> list[Any]:
    from ._contract import Zone

    if not doc:
        return []
    items = doc.get(key) or doc.get("default") or []
    return [Zone(str(z["name"]), [(float(x), float(y)) for x, y in z["polygon"]], float(z.get("conf", 1.0))) for z in items]


def load_inputs(path: Path, zones_doc: Optional[dict[str, Any]], use_upstream: bool = True) -> tuple[Any, list[Any], list[Any], list[Any]]:
    """clip, dets, tracks, zones — из модулей команды, если доступны, иначе простой фолбэк."""
    clip = dets = tracks = zones = None
    if use_upstream:
        try:
            from ml.ingest import load_video  # type: ignore

            clip = load_video(str(path))
        except Exception as e:  # noqa: BLE001
            log.info("ml.ingest unavailable (%s): SimpleClip", e)
        if clip is not None:
            try:
                from ml.detect import detect  # type: ignore

                dets = detect(clip)
            except Exception as e:  # noqa: BLE001
                log.info("ml.detect unavailable (%s): no detections", e)
            try:
                from ml.track import track  # type: ignore

                tracks, zones = track(clip, dets or [])
            except Exception as e:  # noqa: BLE001
                log.info("ml.track unavailable (%s): zones from json", e)
    if clip is None:
        clip = SimpleClip(str(path))
    key = normalize_video_name(path.name)
    jz = _zones_from_json(zones_doc, key)
    return clip, list(dets or []), list(tracks or []), jz or list(zones or [])


def collect(labels: dict[str, list[dict[str, Any]]], videos: dict[str, Path], zones_doc: Optional[dict[str, Any]], use_upstream: bool = True) -> dict[str, dict[str, list[Any]]]:
    """Признаки и метки по сигналам: {signal: {"x": [...], "y": [...], "g": [...]}}."""
    from .core import scene_states_with_trace

    data: dict[str, dict[str, list[Any]]] = {s: {"x": [], "y": [], "g": []} for s in TRAINABLE}
    for key, rows in sorted(labels.items()):
        if key not in videos:
            log.warning("video for labels '%s' not found", key)
            continue
        clip, dets, tracks, zones = load_inputs(videos[key], zones_doc, use_upstream)
        _, trace = scene_states_with_trace(clip, dets, tracks, zones, [], config={"learned": {"enabled": False}})
        times = np.asarray(trace.get("times", []), dtype=np.float64)
        if len(times) == 0:
            continue
        for sig in TRAINABLE:
            feats = trace.get("features", {}).get(sig)
            if not feats:
                continue
            x = feature_matrix(feats, FEATURES[sig], len(times))
            y = np.full(len(times), -1)
            for r in rows:
                if r["signal"] != sig:
                    continue
                if r["t1"] - r["t0"] < 1e-6:
                    sel = [int(np.argmin(np.abs(times - r["t0"])))]
                else:
                    sel = np.flatnonzero((times >= r["t0"]) & (times <= r["t1"]))
                y[sel] = r["value"]
            for i in np.flatnonzero(y >= 0):
                data[sig]["x"].append(x[i])
                data[sig]["y"].append(int(y[i]))
                data[sig]["g"].append(key)
    return data


def group_cv(x: np.ndarray, y: np.ndarray, g: np.ndarray, kind: str, k: int = 5) -> dict[str, Any]:
    groups = sorted(set(g.tolist()))
    k = min(k, len(groups))
    if k < 2 or len(set(y.tolist())) < 2:
        return {"folds": 0, "auc": None, "acc": None}
    scores = np.full(len(y), np.nan)
    for f in range(k):
        test_g = set(groups[f::k])
        te = np.array([gg in test_g for gg in g])
        if te.all() or not te.any() or len(set(y[~te].tolist())) < 2:
            continue
        m = fit_mlp(x[~te], y[~te]) if kind == "mlp" else fit_lr(x[~te], y[~te])
        scores[te] = predict(m, x[te])
    ok = np.isfinite(scores)
    if not ok.any():
        return {"folds": k, "auc": None, "acc": None}
    a = auc(y[ok], scores[ok])
    return {"folds": k, "auc": None if not np.isfinite(a) else round(a, 4), "acc": round(float(np.mean((scores[ok] >= 0.5) == (y[ok] == 1))), 4), "n_scored": int(ok.sum())}


def train(data: dict[str, dict[str, list[Any]]], kind: str = "lr", min_samples: int = 20) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for sig, d in data.items():
        if len(d["y"]) < min_samples or len(set(d["y"])) < 2:
            log.warning("%s: not enough verified labels (%d) — skipped", sig, len(d["y"]))
            continue
        x, y, g = np.asarray(d["x"], dtype=np.float64), np.asarray(d["y"], dtype=np.float64), np.asarray(d["g"])
        cv = group_cv(x, y, g, kind)
        m = fit_mlp(x, y) if kind == "mlp" else fit_lr(x, y)
        rel = 0.5 if cv["auc"] is None else float(np.clip(2.0 * (cv["auc"] - 0.5), 0.0, 1.0))
        m.update({"features": FEATURES[sig], "cv": cv, "n": int(len(y)), "n_pos": int(y.sum()), "n_videos": int(len(set(g.tolist()))), "reliability": round(rel, 3)})
        models[sig] = m
        log.info("%s: n=%d videos=%d cv=%s", sig, len(y), m["n_videos"], cv)
    return {"version": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": kind, "signals": models}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="data/scene_labels.csv")
    ap.add_argument("--videos", required=True, help="папка с видео (рекурсивно)")
    ap.add_argument("--out", default="weights/scene/scene_signals.json")
    ap.add_argument("--model", choices=["lr", "mlp"], default="lr")
    ap.add_argument("--zones-json", default=None, help="зоны для обучения, если ml.track недоступен")
    ap.add_argument("--no-upstream", action="store_true", help="не вызывать ml.ingest/ml.detect/ml.track")
    ap.add_argument("--allow-unverified", action="store_true")
    ap.add_argument("--min-samples", type=int, default=20)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    labels = read_labels(args.labels, args.allow_unverified)
    if not labels:
        print("no usable (verified) labels", file=sys.stderr)
        return 2
    zones_doc = json.loads(Path(args.zones_json).read_text(encoding="utf-8")) if args.zones_json else None
    data = collect(labels, find_videos(args.videos), zones_doc, use_upstream=not args.no_upstream)
    doc = train(data, args.model, args.min_samples)
    if not doc["signals"]:
        print("nothing trained (too few verified labels)", file=sys.stderr)
        return 3
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({s: m["cv"] for s, m in doc["signals"].items()}, ensure_ascii=False))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
