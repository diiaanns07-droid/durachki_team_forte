"""Проверка сплитов на реальном индексе: группы дубликатов, конфликты меток, отсутствие утечек.

    python -m ml.temporal.check_split [--index cache/features/index.json] [--folds loo 5 3]

Печатает только имена видео/групп и метки (без кадров и описаний сцен). Код возврата 1 при утечке.
"""
from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from .config import load_config
from .splits import check_split, grouped_folds, holdout_split


def verify(samples: Sequence, folds: Sequence[str], holdout: float = 0.2, seed: int = 0) -> dict:
    rep: dict = {"n_groups": len(samples), "n_pos": sum(s.label for s in samples),
                 "duplicate_groups": {s.name: [x["name"] for x in s.sources] for s in samples if len(s.sources) > 1},
                 "conflicts": {s.name: [(x["name"], x["label"], x["times"]) for x in s.sources]
                               for s in samples if s.conflict},
                 "versions_per_group": sorted({len(s.versions) for s in samples}), "folds": {}, "ok": True}
    for f in folds:
        fv = "loo" if f == "loo" else int(f)
        try:
            fl = grouped_folds(samples, fv, seed=seed)
            for tr, va in fl:
                check_split(samples, tr, va)
            seen = sorted(i for _, va in fl for i in va)
            assert seen == list(range(len(samples))), "не каждая группа ровно один раз в валидации"
            rep["folds"][f] = {"n_folds": len(fl), "ok": True,
                               "val_pos_per_fold": [sum(samples[i].label for i in va) for _, va in fl]}
        except AssertionError as e:
            rep["folds"][f] = {"ok": False, "error": str(e)}
            rep["ok"] = False
    rest, hold = holdout_split(samples, holdout, seed=seed)
    if hold:
        check_split(samples, rest, hold)
        rep["holdout"] = {"frac": holdout, "names": [samples[i].name for i in hold]}
    return rep


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ml.temporal: проверка сплитов")
    ap.add_argument("--index", default=None)
    ap.add_argument("--folds", nargs="+", default=["loo", "5", "3"])
    ap.add_argument("--holdout", type=float, default=0.2)
    a = ap.parse_args(argv)
    cfg = load_config()
    from .data_index import load_samples

    samples, _ = load_samples(cfg, a.index)
    rep = verify(samples, a.folds, a.holdout, int(cfg["train"]["seed"]))
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
