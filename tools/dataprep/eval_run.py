"""Метрики прогона сервиса (runs/<…>/summary.json + <video>.json) против data/labels.csv — локально, CPU.

Печатает только агрегаты и id видео (без описаний сцен). Подробности по видео — в cache/eval_runs/<run>.csv.

usage:
  python -I -X utf8 tools/dataprep/eval_run.py <run_dir|summary.json> [--root .] [--compare <другой run>]
Сопоставление видео: по имени (X-<n>), иначе по sha256. Таймкод: |Δt| до ближайшей метки md5-группы
(для дубликата с двумя таймкодами засчитывается любой — как альтернативы в eval.labels).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def load_labels(root: Path) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    with open(root / "data/manifest.csv", encoding="utf-8") as f:
        man = list(csv.DictReader(f))
    by_id = {m["video_id"]: m for m in man}
    by_sha: dict[str, list[dict]] = {}
    for m in man:
        by_sha.setdefault(m["sha256"], []).append(m)
    return by_id, by_sha


def run_paths(p: Path) -> tuple[Path, Path]:
    p = Path(p)
    return (p.parent, p) if p.suffix == ".json" else (p, p / "summary.json")


def per_video(run: Path, root: Path) -> list[dict]:
    """Строка на видео прогона: метка, вердикт, исход, таймкод, оценки из полного ответа (если есть)."""
    run_dir, summary = run_paths(run)
    by_id, by_sha = load_labels(root)
    out = []
    for r in json.loads(summary.read_text(encoding="utf-8")):
        m = by_id.get(str(r.get("name", "")).strip()) or (by_sha.get(r.get("sha256", "")) or [None])[0]
        if m is None:
            continue
        group = by_sha[m["sha256"]]
        t_alt = sorted({float(g["t_sec"]) for g in group if g["t_sec"]})
        y = m["label"] == "violation"
        pred = r.get("verdict") == "violation"
        tc = r.get("timecode_sec")
        full = run_dir / f"{r.get('name', m['video_id'])}.json"
        risk_max = max_conf = None
        stages: dict = {}
        if full.exists():
            j = json.loads(full.read_text(encoding="utf-8"))
            sr = (j.get("meta") or {}).get("scores_raw") or {}
            risk_max, max_conf = sr.get("risk_max"), sr.get("max_conf")
            stages = ((j.get("coverage") or {}).get("stages")) or {}
        out.append(dict(
            video_id=m["video_id"], label=m["label"], t_label=";".join(f"{t:g}" for t in t_alt),
            status=r.get("status", ""), verdict=r.get("verdict") or "", type_id=r.get("type_id") or "",
            timecode=tc, dt=min(abs(tc - t) for t in t_alt) if (tc is not None and t_alt and y) else None,
            confidence=r.get("confidence"), risk_max=risk_max, max_conf=max_conf,
            n_violations=r.get("n_violations", 0), seconds=r.get("seconds"),
            outcome=("TP" if pred else "FN") if y else ("FP" if pred else "TN"), stages=stages))
    out.sort(key=lambda x: int(x["video_id"].split("-")[1]))
    return out


def summarize(rows: list[dict], name: str) -> None:
    n = len(rows)
    c = Counter(x["outcome"] for x in rows)
    ok = c["TP"] + c["TN"]
    lo, hi = wilson(ok, n)
    rec = c["TP"] / max(1, c["TP"] + c["FN"])
    spec = c["TN"] / max(1, c["TN"] + c["FP"])
    failed = sum(x["status"] != "done" for x in rows)
    print(f"== {name}: n={n}, не done: {failed}")
    print(f"   accuracy {ok / n:.0%} (95% ДИ {lo:.0%}–{hi:.0%}) | balanced {(rec + spec) / 2:.0%} | "
          f"recall {c['TP']}/{c['TP'] + c['FN']} | specificity {c['TN']}/{c['TN'] + c['FP']}")
    ids = lambda o: [x["video_id"] for x in rows if x["outcome"] == o]  # noqa: E731
    print(f"   FN (пропуски): {ids('FN')}")
    print(f"   FP (ложные тревоги): {ids('FP')}")
    tp_t = Counter(x["type_id"] or "?" for x in rows if x["outcome"] == "TP")
    fp_t = Counter(x["type_id"] or "?" for x in rows if x["outcome"] == "FP")
    print(f"   тип в вердикте — TP {dict(tp_t)} | FP {dict(fp_t)}")
    y = lambda x: x["label"] == "violation"  # noqa: E731
    for key in ("confidence", "risk_max", "max_conf"):
        vals = [(x[key] if x[key] is not None else 0.0, y(x)) for x in rows]
        if any(x[key] is not None for x in rows):
            a = auc([v for v, t in vals if t], [v for v, t in vals if not t])
            print(f"   AUC({key}) = {a:.2f}  (0.5 — не различает; ≥0.8 — порогом можно поднять точность)")
    errs = [x["dt"] for x in rows if x["outcome"] == "TP" and x["dt"] is not None]
    if errs:
        print(f"   таймкод на TP: медиана |Δt| {statistics.median(errs):.1f} с; ≤1 с {sum(e <= 1 for e in errs)}"
              f"/{len(errs)}, ≤3 с {sum(e <= 3 for e in errs)}/{len(errs)}, ≤5 с {sum(e <= 5 for e in errs)}/{len(errs)}")
    secs = [x["seconds"] for x in rows if x["seconds"] is not None]
    if secs:
        print(f"   время на видео: медиана {statistics.median(secs):.0f} с, макс {max(secs):.0f} с (> 300 с: "
              f"{sum(s > 300 for s in secs)})")
    st = Counter()
    for x in rows:
        for k, v in (x["stages"] or {}).items():
            st[(k, str(v))] += 1
    if st:
        by_stage: dict[str, dict] = {}
        for (k, v), cnt in st.items():
            by_stage.setdefault(k, {})[v] = cnt
        print("   стадии: " + "; ".join(f"{k}: {by_stage[k]}" for k in by_stage))


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [k for k in rows[0] if k != "stages"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--compare", type=Path, default=None, help="другой прогон: что починилось / сломалось")
    a = ap.parse_args()
    root = a.root.resolve()
    rows = per_video(a.run, root)
    name = run_paths(a.run)[0].name
    summarize(rows, name)
    out = root / "cache/eval_runs" / f"{name}.csv"
    write_csv(rows, out)
    print(f"   по видео (локально): {out}")
    if a.compare:
        base = {x["video_id"]: x for x in per_video(a.compare, root)}
        right = lambda o: o in ("TP", "TN")  # noqa: E731
        fixed = [x["video_id"] for x in rows if x["video_id"] in base and right(x["outcome"])
                 and not right(base[x["video_id"]]["outcome"])]
        broken = [x["video_id"] for x in rows if x["video_id"] in base and not right(x["outcome"])
                  and right(base[x["video_id"]]["outcome"])]
        print(f"== {name} против {run_paths(a.compare)[0].name}: починилось {len(fixed)} {fixed} | "
              f"сломалось {len(broken)} {broken}")


if __name__ == "__main__":
    main()
