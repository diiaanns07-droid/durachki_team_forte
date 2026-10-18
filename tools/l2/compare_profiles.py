"""Попарное сравнение профилей по SHA-группам (диагностика на знакомых видео, не независимый тест).

Вход — metrics.json от `python -m eval.metrics` (per_video — по группам дубликатов). Выход:
- локальная таблица <out>.csv: группа → эталон → вердикт каждого профиля → категория (TP/FP/TN/FN/failed/partial);
- агрегаты в stdout: TP/FP/TN/FN, accuracy, balanced accuracy, precision, recall, сбои, время;
  исправленные и испорченные группы для каждой пары соседних профилей (только числа).

    python tools/l2/compare_profiles.py old=runs/l2/compare/exp4_metrics.json new=runs/l2/pose_rules/metrics.json \\
        --out runs/l2/compare/pairwise.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path


def load(path: str) -> dict[str, dict]:
    pv = json.loads(Path(path).read_text(encoding="utf-8"))["per_video"]
    return {e["video_id"]: e for e in pv}


def correct(e: dict) -> bool:
    return e.get("category") in ("TP", "TN", "early", "late", "wrong_type")


def summary(rows: dict[str, dict]) -> dict:
    ev = [e for e in rows.values() if e.get("gt_label") not in ("conflict", "unknown")]
    pos = [e for e in ev if e["gt_label"] == "violation"]
    neg = [e for e in ev if e["gt_label"] != "violation"]
    tp = sum(1 for e in pos if e.get("verdict") == "violation")
    fp = sum(1 for e in neg if e.get("verdict") == "violation")
    tn = sum(1 for e in neg if e.get("verdict") == "no_violation" and e.get("status") == "done" and not e.get("partial"))
    fn = len(pos) - tp
    failed = sum(1 for e in ev if e.get("status") != "done" or e.get("invalid_verdict") or e.get("partial"))
    tpr, tnr = tp / max(1, len(pos)), tn / max(1, len(neg))
    lat = [e["latency_s"] for e in ev if isinstance(e.get("latency_s"), (int, float))]
    return {"n": len(ev), "TP": tp, "FP": fp, "TN": tn, "FN": fn, "failed_or_partial": failed,
            "accuracy": round((tp + tn) / max(1, len(ev)), 3), "balanced_accuracy": round((tpr + tnr) / 2, 3),
            "precision": round(tp / max(1, tp + fp), 3), "recall": round(tpr, 3),
            "latency_median_s": round(st.median(lat), 1) if lat else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("profiles", nargs="+", help="name=metrics.json (по порядку: контроль → эксперименты)")
    ap.add_argument("--out", default="runs/l2/compare/pairwise.csv")
    a = ap.parse_args()
    profs = [(p.split("=", 1)[0], load(p.split("=", 1)[1])) for p in a.profiles]
    groups = sorted(set().union(*[set(r) for _, r in profs]), key=lambda g: (len(g), g))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "members", "gt_label", "conflict"] + [f"{n}:{k}" for n, _ in profs for k in ("verdict", "category", "pred_type")])
        for g in groups:
            ref = next(r[g] for _, r in profs if g in r)
            w.writerow([g, " ".join(ref.get("members") or []), ref.get("gt_label"), ref.get("conflict")]
                       + [x for _, r in profs for x in ((r.get(g) or {}).get("verdict"), (r.get(g) or {}).get("category"),
                                                       (r.get(g) or {}).get("pred_type"))])
    for n, r in profs:
        print(f"{n:22s}", summary(r))
    for (na, ra), (nb, rb) in zip(profs, profs[1:]):
        common = [g for g in groups if g in ra and g in rb and ra[g].get("gt_label") not in ("conflict", "unknown")]
        fixed = [g for g in common if not correct(ra[g]) and correct(rb[g])]
        broken = [g for g in common if correct(ra[g]) and not correct(rb[g])]
        print(f"{na} -> {nb}: исправлено {len(fixed)}, испорчено {len(broken)}, без изменений {len(common) - len(fixed) - len(broken)}")
    print(f"таблица по группам (локально): {out}")


if __name__ == "__main__":
    main()
