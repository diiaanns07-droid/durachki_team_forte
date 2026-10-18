"""Быстрый офлайн-прогон ml.seqrules по кэшу стадий A03 (cache/<sha>/: dets, tracks, zones, poses) без кадров.

Проверки по кадрам (OWLv2) здесь не выполняются — правило Б получает not_checked, а не «сейф закрыт».
Печатает только агрегаты; подробности по видео — JSON в --out (локально, вне git).

    python tools/l2/seqrules_replay.py --cache cache --labels data/labels.csv --manifest data/manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


class _Clip:
    def __init__(self, times, width, height):
        self.times, self.width, self.height, self.path = times, width, height, ""


def _load(d: Path):
    from backend.pipeline import serde

    def rd(name, kind):
        p = d / name
        if not p.exists():
            return []
        raw = serde.read_jsonl(p) if p.suffix == ".jsonl" else serde.read_json(p)
        return [serde.LOADERS[kind](x) for x in raw]

    return rd("dets.jsonl", "detection"), rd("tracks.json", "track"), rd("zones.json", "zone"), rd("poses.jsonl", "pose")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="runs/l2/seqrules_replay")
    ap.add_argument("--config", default=None, help="YAML поверх ml/seqrules/config.yaml")
    a = ap.parse_args()

    import yaml

    from ml.seqrules import run
    from ml.seqrules.checks import Checker

    over = yaml.safe_load(open(a.config, encoding="utf-8")) if a.config else None
    man = {r["sha256"]: r for r in csv.DictReader(open(a.manifest, encoding="utf-8"))}
    lab = {r["video_id"]: r for r in csv.DictReader(open(a.labels, encoding="utf-8"))}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    agg = Counter()
    by_label = {"violation": Counter(), "no_violation": Counter()}
    conf_m = Counter()
    for d in sorted(Path(a.cache).iterdir()):
        if not (d / "dets.jsonl").exists():
            continue
        m = man.get(d.name)
        if m is None:
            agg["no_manifest"] += 1
            continue
        vids = [r["video_id"] for r in man.values() if r["sha256"] == d.name]
        labels = {lab[v]["label"] for v in vids if v in lab}
        conflict = any(lab.get(v, {}).get("conflict") == "1" for v in vids) or len(labels) > 1
        dets, tracks, zones, poses = _load(d)
        times = sorted({round(float(x.t), 4) for x in dets})
        rot = int(float(m.get("rotation") or 0)) % 180
        w, h = float(m["width"]), float(m["height"])
        clip = _Clip(times, h if rot else w, w if rot else h)
        events, obs = run(clip, dets, tracks, zones, poses, None, config=over, checker=Checker(None, {"enabled": False}))
        rules = Counter(e.extra.get("rule") for e in events)
        cand = [e for e in events if e.extra.get("assessment") == "candidate_violation"]
        verdict = "violation" if cand else "no_violation"
        rec = {"videos": vids, "labels": sorted(labels), "conflict": conflict, "verdict": verdict,
               "rules": dict(rules), "meta": {k: obs[0][k] for k in ("frames", "rules", "warnings")},
               "events": [{"type_id": e.type_id, "t_peak": e.t_peak, "conf": e.confidence, "rule": e.extra.get("rule"),
                           "assessment": e.extra.get("assessment")} for e in events]}
        (out / f"{d.name[:12]}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        agg["videos"] += 1
        if conflict or len(labels) != 1:
            agg["conflict_or_unlabeled"] += 1
            continue
        y = labels.pop()
        for r, n in rules.items():
            by_label[y][r] += 1 if n else 0
        by_label[y]["videos"] += 1
        conf_m[("TP" if y == "violation" else "FP") if verdict == "violation" else ("FN" if y == "violation" else "TN")] += 1
    tp, fp, fn, tn = conf_m["TP"], conf_m["FP"], conf_m["FN"], conf_m["TN"]
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-9, p + r)
    print("videos:", dict(agg))
    print("per label (videos with ≥1 event of rule):", {k: dict(v) for k, v in by_label.items()})
    print(f"verdict: TP={tp} FP={fp} TN={tn} FN={fn} precision={p:.3f} recall={r:.3f} F1={f1:.3f} "
          f"accuracy={(tp + tn) / max(1, tp + fp + tn + fn):.3f}")


if __name__ == "__main__":
    main()
