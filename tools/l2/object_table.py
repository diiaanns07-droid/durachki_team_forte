"""Таблица классов объектов (L2): класс → модель → что реально наблюдаемо на этих видео → что проверено.

Источник — кэш стадий A03 (cache/<sha>/dets.jsonl) и наблюдения seqrules прогона (data/jobs/<id>/observations.json).
Печатает только агрегаты. Кандидаты для ЛОКАЛЬНОЙ ручной разметки (видео, t, bbox, score) пишет в CSV вне git —
это псевдоразметка для проверки человеком, а не эталон.

    python tools/l2/object_table.py --cache cache --jobs data/jobs --out runs/l2/objects
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = {   # класс таблицы → метки детектора
    "person": ["person"],
    "phone": ["cell_phone"],
    "cash": ["cash"],
    "bag": ["bag"],
    "cash_bag": ["cash_bag"],
    "safe": ["safe", "safe_open"],
    "money_counter": ["money_counter"],
}
MODEL = {
    "person": "RT-DETRv2-r18 (COCO person)",
    "phone": "RT-DETRv2 (COCO cell phone) + OWLv2 «mobile phone in hand»; кроп-контраст OWLv2 в seqrules",
    "cash": "OWLv2 «stack of banknotes» / «cash money» (в COCO класса нет)",
    "bag": "RT-DETRv2 (COCO handbag/backpack/suitcase → bag)",
    "cash_bag": "OWLv2 «green canvas bag»",
    "safe": "OWLv2 «metal safe» / «open safe door»; состояние — контраст open/closed (A05)",
    "money_counter": "OWLv2 «money counting machine»",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--jobs", default="data/jobs")
    ap.add_argument("--out", default="runs/l2/objects")
    ap.add_argument("--top", type=int, default=5, help="кандидатов на класс и видео")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    label2cls = {lab: c for c, labs in CLASSES.items() for lab in labs}
    vids, dets, scores, sources = Counter(), Counter(), defaultdict(list), defaultdict(Counter)
    n_videos = 0
    cands: list[dict] = []
    for d in sorted(Path(a.cache).iterdir()):
        f = d / "dets.jsonl"
        if not f.exists():
            continue
        n_videos += 1
        seen: set[str] = set()
        per: dict[str, list[dict]] = defaultdict(list)
        for line in f.open(encoding="utf-8"):
            x = json.loads(line)
            c = label2cls.get(x["label"])
            if c is None:
                continue
            seen.add(c)
            dets[c] += 1
            scores[c].append(float(x["score"]))
            sources[c][(x.get("source") or "").split(":")[0] or "?"] += 1
            if c in ("cash", "cash_bag", "safe", "phone", "money_counter"):
                per[c].append(x)
        for c in seen:
            vids[c] += 1
        for c, xs in per.items():
            for x in sorted(xs, key=lambda y: -float(y["score"]))[: a.top]:
                cands.append({"sha256": d.name, "class": c, "label": x["label"], "t": x["t"], "score": round(float(x["score"]), 3),
                              "bbox": json.dumps([round(float(v), 4) for v in x["bbox"]]),
                              "source": x.get("source", ""), "status": "кандидат — не эталон, проверить человеку"})
    # наблюдения seqrules (последний прогон профиля): исходы правил и проверок
    phone_status, contrast, safe_status, safe_checks, entry_status = Counter(), Counter(), Counter(), Counter(), Counter()
    n_obs = 0
    for p in sorted(Path(a.jobs).glob("*/observations.json")):
        items = json.loads(p.read_text(encoding="utf-8")).get("items", [])
        if not items:
            continue
        n_obs += 1
        for e in items[1:]:
            if e.get("rule") == "phone":
                phone_status[e.get("status")] += 1
                for c in e.get("contrast", []) or []:
                    contrast[c.get("state")] += 1
            elif e.get("rule") == "safe":
                safe_status[e.get("status")] += 1
                for c in e.get("safe_checks", []) or []:
                    safe_checks[c.get("state")] += 1
            elif e.get("rule") == "entry":
                entry_status[e.get("status")] += 1
    rows = []
    for c in CLASSES:
        sc = scores.get(c) or []
        rows.append({"class": c, "model": MODEL[c], "videos_with_dets": f"{vids[c]}/{n_videos}", "dets": dets[c],
                     "median_score": round(st.median(sc), 3) if sc else None, "sources": dict(sources[c])})
    summary = {"n_videos_cache": n_videos, "classes": rows, "n_jobs_with_observations": n_obs,
               "phone_episodes": dict(phone_status), "phone_contrast_checks": dict(contrast),
               "safe_episodes": dict(safe_status), "safe_checks": dict(safe_checks), "entry_episodes": dict(entry_status),
               "labeling_candidates": len(cands)}
    (out / "object_table.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out / "labeling_candidates.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sha256", "class", "label", "t", "score", "bbox", "source", "status"])
        w.writeheader()
        w.writerows(cands)
    for r in rows:
        print(f"{r['class']:13s} videos={r['videos_with_dets']:6s} dets={r['dets']:6d} med={r['median_score']} src={r['sources']}")
    print("observations jobs:", n_obs, "phone:", dict(phone_status), "contrast:", dict(contrast))
    print("safe:", dict(safe_status), "safe checks:", dict(safe_checks), "entry:", dict(entry_status))
    print("labeling candidates (local CSV, not ground truth):", len(cands))


if __name__ == "__main__":
    main()
