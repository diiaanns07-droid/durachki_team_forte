"""Cross-fit (out-of-fold) чекпоинты temporal — для честного сравнения temporal off/on на уже исследованных роликах.

    python -m ml.temporal.oof [--splits data/splits.csv] [--folds 5] [--kind baseline]

Для каждого фолда k модель И порог (внутренний grouped CV) обучаются ТОЛЬКО на группах других фолдов.
Группы дубликатов (md5) и split_group из eval.split всегда целиком в одном фолде; противоречие — ошибка.
Holdout-группы eval.split (fold = -1) не входят в обучение фолдов и оцениваются моделью, обученной на всех
не-holdout группах. Индекс: weights/temporal/oof/index.json (md5 видео -> каталог фолда).
В инференсе включается только явно: SENTINEL_TEMPORAL_OOF=1 (режим оценки; прод использует обычный чекпоинт).
Это cross-fitting на 42 разведочных роликах — не независимый holdout и не измерение пользы.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from .config import load_config, weights_dir
from .dataset import Sample
from .labels import normalize_name
from .splits import grouped_folds

log = logging.getLogger("sentinel.temporal")
HOLDOUT = -1
EXPLORATORY = ("cross-fit на роликах, которые команда уже исследовала: оценка разведочная, не независимый holdout; "
               "польза temporal считается только по сравнению off/on (eval.ablate)")


def read_splits(path: str | Path) -> dict[str, dict]:
    """eval.split: video_id,group,split_group,split,fold,label -> {нормализованный id: {fold, split, split_group}}."""
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            vid = normalize_name(row.get("video_id") or row.get("file") or "")
            if not vid:
                continue
            raw = str(row.get("fold", "")).strip()
            out[vid] = {"fold": int(raw) if raw.lstrip("-").isdigit() else None,
                        "split": str(row.get("split", "")).strip(),
                        "split_group": str(row.get("split_group") or row.get("group") or vid)}
    return out


def assign_folds(samples: Sequence[Sample], splits: Optional[dict[str, dict]], n_folds: int,
                 seed: int) -> dict[str, Optional[int]]:
    """group -> fold. Из splits.csv: все члены группы дубликатов обязаны иметь один фолд (иначе утечка)."""
    if splits is None:
        out: dict[str, Optional[int]] = {}
        for k, (_, va) in enumerate(grouped_folds(samples, n_folds, seed=seed)):
            for i in va:
                out[samples[i].group] = k
        return out
    out = {}
    for s in samples:
        members = {normalize_name(x["name"]) for x in s.sources} | {normalize_name(s.name)}
        folds = {splits[m]["fold"] for m in members if m in splits and splits[m]["fold"] is not None}
        if len(folds) > 1:
            raise ValueError(f"группа {s.group}: члены {sorted(members)} в разных фолдах splits.csv {sorted(folds)} — "
                             "утечка дубликатов; пересоберите split (python -m eval.split)")
        out[s.group] = folds.pop() if folds else None
    return out


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_oof(samples: list[Sample], idx: dict, bank, cfg: dict, kinds: Sequence[str], out_root: Path,
              splits_path: Optional[str] = None, n_folds: int = 5, seed: int = 13,
              epochs: Optional[int] = None) -> dict:
    from .train import train_all

    splits = read_splits(splits_path) if splits_path else None
    fold_of = assign_folds(samples, splits, n_folds, seed)
    folds = sorted({f for f in fold_of.values() if f is not None and f != HOLDOUT})
    plan: list[tuple[str, list[Sample], list[str]]] = []
    for k in folds:
        train = [s for s in samples if fold_of.get(s.group) not in (k, HOLDOUT, None)]
        val = [s.group for s in samples if fold_of.get(s.group) == k]
        plan.append((f"fold_{k}", train, val))
    hold = [s.group for s in samples if fold_of.get(s.group) == HOLDOUT]
    if hold:
        plan.append(("fold_holdout", [s for s in samples if fold_of.get(s.group) not in (HOLDOUT, None)], hold))
    out_root.mkdir(parents=True, exist_ok=True)
    folds_info: dict[str, dict] = {}
    for name, train, val in plan:
        tr_groups = {s.group for s in train}
        assert tr_groups.isdisjoint(val), f"{name}: группа валидации в train"
        if len({s.label for s in train}) < 2:
            log.warning("oof %s: в train один класс — модель не обучается, видео фолда будут «не проанализированы»", name)
            folds_info[name] = {"trained": False, "val_groups": sorted(val), "reason": "один класс в train"}
            continue
        info = train_all(train, bank, cfg, list(kinds), out_root / name, select="auto", threshold="auto",
                         epochs=epochs)
        meta = json.loads((out_root / name / info["selected"] / "meta.json").read_text(encoding="utf-8"))
        assert set(meta["train_groups"]).isdisjoint(val), f"{name}: утечка группы валидации в чекпоинт"
        folds_info[name] = {"trained": True, "selected": info["selected"], "train_groups": sorted(tr_groups),
                            "val_groups": sorted(val), "threshold": meta.get("threshold"),
                            "threshold_source": meta.get("threshold_source")}
        log.info("oof %s: train=%d групп, val=%d групп, порог %.3f", name, len(tr_groups), len(val),
                 meta.get("threshold", float("nan")))
    by_group = {g: name for name, _, val in plan for g in val if folds_info.get(name, {}).get("trained")}
    videos = {}
    for e in idx.get("videos", []):
        g = next((s.group for s in samples if e["name"] in {x["name"] for x in s.sources} or e["name"] == s.name), None)
        if g in by_group and e.get("md5"):
            videos[e["md5"]] = {"dir": by_group[g], "group": g, "name": e["name"]}
    index = {"version": 1, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "kinds": list(kinds), "splits": str(splits_path) if splits_path else None,
             "splits_sha256": _sha256(splits_path) if splits_path else None,
             "n_folds": len(folds) if splits else n_folds, "folds": folds_info, "videos": videos,
             "not_covered_groups": sorted(s.group for s in samples if s.group not in by_group),
             "evaluation_status": EXPLORATORY,
             "holdout_policy": "holdout eval.split (fold=-1) — модель на всех не-holdout группах" if hold else None}
    (out_root / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ml.temporal: cross-fit чекпоинты для сравнения temporal off/on")
    ap.add_argument("--index", default=None, help="cache/features/index.json (ml.temporal.extract)")
    ap.add_argument("--splits", default=None, help="data/splits.csv из eval.split (рекомендуется)")
    ap.add_argument("--folds", type=int, default=5, help="без --splits: свои сгруппированные фолды")
    ap.add_argument("--kind", default="baseline", choices=["baseline", "temporal", "both"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default=None, help="по умолчанию weights/temporal/oof")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    from .data_index import bank_for_index, load_samples

    samples, idx = load_samples(cfg, a.index)
    bank = bank_for_index(cfg, idx)
    kinds = ["baseline", "temporal"] if a.kind == "both" else [a.kind]
    out = Path(a.out) if a.out else weights_dir(cfg) / "oof"
    index = train_oof(samples, idx, bank, cfg, kinds, out, a.splits, a.folds, int(cfg["train"]["seed"]), a.epochs)
    trained = [k for k, v in index["folds"].items() if v.get("trained")]
    print(f"OOF: обучено фолдов {len(trained)}/{len(index['folds'])}, видео в индексе {len(index['videos'])}, "
          f"без модели групп {len(index['not_covered_groups'])} -> {out / 'index.json'}  [{EXPLORATORY}]")
    return 0 if trained else 1


if __name__ == "__main__":
    raise SystemExit(main())
