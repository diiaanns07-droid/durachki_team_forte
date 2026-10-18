"""Групповой split ДО любых аугментаций: train / calib / holdout (+ фолды для CV).

* группа = дубликаты по содержимому (md5/sha256: Х-4≡Х-47, Х-5≡Х-48) ∪ одна сессия съёмки
  (необязательный ``--sessions sessions.csv``: video_id,session_id) — члены группы всегда в одном split;
* стратификация по метке группы; детерминированно (seed);
* train/calib — подбор моделей и порогов; holdout — НЕ трогать до финальных цифр (каждое обращение
  через ``eval.run --subset holdout --final`` пишется в data/holdout_access.jsonl);
* аугментации (eval.robustness, обучение A10) наследуют split исходного видео → утечки нет.

CLI::

    python -m eval.split --labels data/labels.csv --out data/splits.csv [--holdout 0.25 --calib 0.2 --folds 5]
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import random
from pathlib import Path
from typing import Optional

from .common import normalize_video_id, video_sort_key, write_json
from .labels import LABEL_VIOLATION, _read_csv_dicts, read_labels_csv

SPLITS = ("train", "calib", "holdout")
COLUMNS = ["video_id", "group", "split_group", "split", "fold", "label"]


class _DSU:
    def __init__(self) -> None:
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            lo, hi = sorted((ra, rb), key=video_sort_key)
            self.p[hi] = lo


def make_splits(labels_csv: str | Path, sessions_csv: Optional[str | Path] = None, holdout: float = 0.25,
                calib: float = 0.2, folds: int = 5, seed: int = 2026) -> list[dict]:
    rows = read_labels_csv(labels_csv)
    dsu = _DSU()
    for r in rows:
        dsu.union(r.video_id, r.group or r.video_id)
    if sessions_csv and Path(sessions_csv).exists():
        by_session: dict[str, str] = {}
        for d in _read_csv_dicts(sessions_csv):
            vid = normalize_video_id(d.get("video_id")) or d.get("video_id", "")
            sid = (d.get("session_id") or "").strip()
            if vid and sid:
                if sid in by_session:
                    dsu.union(by_session[sid], vid)
                else:
                    by_session[sid] = vid
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(dsu.find(r.video_id), []).append(r)
    strata: dict[str, list[str]] = {"pos": [], "neg": []}
    for g, members in groups.items():
        strata["pos" if any(m.label == LABEL_VIOLATION for m in members) else "neg"].append(g)
    rng = random.Random(seed)
    assign: dict[str, tuple[str, int]] = {}
    for name in ("pos", "neg"):
        gs = sorted(strata[name], key=video_sort_key)
        rng.shuffle(gs)
        n = len(gs)
        n_hold = int(round(n * holdout))
        n_cal = int(round(n * calib))
        for i, g in enumerate(gs):
            split = "holdout" if i < n_hold else ("calib" if i < n_hold + n_cal else "train")
            fold = -1 if split == "holdout" else (i - n_hold) % max(1, folds)
            assign[g] = (split, fold)
    out = []
    for g, members in groups.items():
        split, fold = assign[g]
        for m in members:
            out.append({"video_id": m.video_id, "group": m.group or m.video_id, "split_group": g, "split": split,
                        "fold": fold, "label": m.label})
    return sorted(out, key=lambda d: video_sort_key(d["video_id"]))


def write_splits(rows: list[dict], path: str | Path, manifest: Optional[dict] = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in COLUMNS})
    hold = sorted((r["video_id"] for r in rows if r["split"] == "holdout"), key=video_sort_key)
    counts = {s: {"videos": sum(r["split"] == s for r in rows),
                  "groups": len({r["split_group"] for r in rows if r["split"] == s}),
                  "pos_videos": sum(r["split"] == s and r["label"] == LABEL_VIOLATION for r in rows)} for s in SPLITS}
    man = {"created_at": _dt.datetime.now().isoformat(timespec="seconds"), "counts": counts,
           "holdout_sha256": hashlib.sha256(",".join(hold).encode()).hexdigest()[:16],
           "rule": "split по группам (дубликаты/сессии) ДО аугментаций; holdout не трогать до финальных цифр"}
    man.update(manifest or {})
    write_json(p.with_name(p.stem + "_manifest.json"), man)


def load_splits(path: str | Path) -> dict[str, dict]:
    out = {}
    for d in _read_csv_dicts(path):
        vid = normalize_video_id(d.get("video_id")) or d.get("video_id", "")
        if vid:
            out[vid] = {"split": d.get("split", ""), "fold": int(d.get("fold") or -1), "group": d.get("group", vid),
                        "split_group": d.get("split_group", vid)}
    return out


def ids_for(splits: dict[str, dict], name: str) -> set[str]:
    """ID видео и их групп для split'а (eval.metrics.evaluate(subset=...))."""
    names = {x.strip() for x in name.split(",")}
    out: set[str] = set()
    for vid, d in splits.items():
        if d["split"] in names:
            out |= {vid, d["group"], d["split_group"]}
    return out


def log_holdout_access(log_path: str | Path, info: dict) -> int:
    """Пишет обращение к holdout; возвращает число ПРЕДЫДУЩИХ обращений."""
    p = Path(log_path)
    prev = 0
    if p.exists():
        prev = sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": _dt.datetime.now().isoformat(timespec="seconds"), **info}, ensure_ascii=False) + "\n")
    return prev


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Групповой split train/calib/holdout")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--sessions", default=None, help="video_id,session_id — видео одной сессии в одну группу")
    ap.add_argument("--out", default="data/splits.csv")
    ap.add_argument("--holdout", type=float, default=0.25)
    ap.add_argument("--calib", type=float, default=0.2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--force", action="store_true", help="перезаписать существующий split (holdout сменится!)")
    args = ap.parse_args(argv)
    if Path(args.out).exists() and not args.force:
        print(f"{args.out} уже есть — split фиксирован. Перезапись только с --force (holdout перестанет быть нетронутым).")
        return 1
    rows = make_splits(args.labels, args.sessions, args.holdout, args.calib, args.folds, args.seed)
    write_splits(rows, args.out, {"seed": args.seed, "holdout": args.holdout, "calib": args.calib,
                                  "folds": args.folds, "labels": str(args.labels)})
    for s in SPLITS:
        sub = [r for r in rows if r["split"] == s]
        print(f"{s:8s}: {len(sub)} видео, {len({r['split_group'] for r in sub})} групп, "
              f"{sum(r['label'] == LABEL_VIOLATION for r in sub)} с нарушением")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
