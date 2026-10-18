"""Одна команда для L2: сравнение temporal off/on в полном пайплайне по FN/FP/локализации/времени.

    python -m ml.temporal.offon --splits data/splits.csv [--videos data/raw --labels data/labels.csv
                                --manifest data/manifest.csv --out runs/ablate/A10-temporal-offon]
                                [--reuse-oof] [--dry-run]

Шаги: (1) кэш признаков (ml.temporal.extract), если его нет; (2) cross-fit чекпоинты по фолдам splits.csv
(ml.temporal.oof: модель и порог — только на train фолда, дубликаты — в одной группе); (3) eval.ablate (A16):
baseline = пайплайн с stages.temporal=false, candidate = тот же пайплайн + SENTINEL_TEMPORAL_OOF=1. Метрики,
вердикт PASS/FAIL/INCONCLUSIVE и группировку md5-дубликатов считает eval (dedupe_groups). Оценка разведочная.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

from .config import PKG_DIR, REPO_ROOT, cache_dir, load_config, weights_dir

OFF_CONFIG = PKG_DIR / "ablation" / "temporal_off.yaml"


def ablate_cmd(a: argparse.Namespace) -> list[str]:
    # SENTINEL_TEMPORAL_OOF=1 — обеим рукам (при выключенной стадии ни на что не влияет), чтобы руки различались
    # только stages.temporal. eval.ablate всё равно покажет 2 ключа (путь SENTINEL_PIPELINE_CONFIG и sha конфига) —
    # это одно логическое изменение: оверлей temporal_off.yaml.
    cmd = [sys.executable, "-m", "eval.ablate", "--name", a.name,
           "--baseline-config", str(OFF_CONFIG),
           "--baseline-env", "SENTINEL_TEMPORAL_OOF=1", "--candidate-env", "SENTINEL_TEMPORAL_OOF=1",
           "--videos", a.videos, "--labels", a.labels, "--manifest", a.manifest, "--out", a.out]
    if a.splits:
        cmd += ["--splits", a.splits]
    return cmd


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ml.temporal: temporal off/on в полном пайплайне (eval.ablate)")
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--videos", default="data/raw")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="runs/ablate/A10-temporal-offon")
    ap.add_argument("--name", default="A10-temporal-offon")
    ap.add_argument("--kind", default="baseline", choices=["baseline", "temporal", "both"])
    ap.add_argument("--reuse-oof", action="store_true", help="не переобучать, если oof/index.json уже есть")
    ap.add_argument("--dry-run", action="store_true", help="только напечатать команды")
    a = ap.parse_args(argv)
    cfg = load_config()
    steps: list[list[str]] = []
    if not (cache_dir(cfg) / "index.json").exists():
        steps.append([sys.executable, "-m", "ml.temporal.extract", "--videos", a.videos, "--labels", a.labels])
    if not (a.reuse_oof and (weights_dir(cfg) / "oof" / "index.json").exists()):
        oof = [sys.executable, "-m", "ml.temporal.oof", "--kind", a.kind]
        steps.append(oof + (["--splits", a.splits] if a.splits else []))
    steps.append(ablate_cmd(a))
    for cmd in steps:
        print("$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    if a.dry_run:
        return 0
    if a.splits and not Path(a.splits).exists():
        print(f"нет {a.splits}: сначала python -m eval.split --labels {a.labels} --out {a.splits}", file=sys.stderr)
        return 2
    try:
        import eval.ablate  # noqa: F401  (A16; есть в интеграции)
    except ImportError as e:
        print(f"NOT RUN: eval.ablate недоступен ({e}) — запускать из интеграционной ветки", file=sys.stderr)
        return 3
    for cmd in steps:
        rc = subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode
        if rc != 0:
            print(f"шаг завершился с кодом {rc}: {' '.join(cmd[:4])} …", file=sys.stderr)
            return rc
    print(f"готово: {a.out}/compare.json (локально, с ID видео), {a.out}/compare_public.json (только агрегаты)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
