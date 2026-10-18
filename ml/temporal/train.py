"""Обучение на всех размеченных видео и сохранение чекпоинтов в weights/temporal/<kind>/.

    python -m ml.temporal.train [--kind baseline|temporal|both] [--select auto|baseline|temporal]
                                [--threshold auto|0.6] [--cv-report reports/temporal_cv.json]

Порог берётся из отчёта CV (deploy_threshold выбранной модели) или из config.yaml (infer.event_threshold).
Голова типа учится только при наличии проверенного type_id в labels.csv.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional, Sequence

from .config import feature_signature, load_config, reports_dir, stable_hash, weights_dir
from .dataset import Sample, fit_spec, prepare_samples
from .features import TextBank
from .predictors import fit_predictor, has_type_labels, save_checkpoint, type_head_source

log = logging.getLogger("sentinel.temporal")


def train_all(samples: Sequence[Sample], bank: TextBank, cfg: dict, kinds: Sequence[str], out_dir: Path,
              select: str = "auto", threshold: str = "auto", cv_report: Optional[dict] = None,
              epochs: Optional[int] = None) -> dict:
    from .cv import calibrate

    t0 = time.monotonic()
    deadline = t0 + 60.0 * float(cfg["train"].get("max_minutes", 8.0))
    seed = int(cfg["train"]["seed"])
    info: dict = {"kinds": {}, "n_samples": len(samples), "n_pos": sum(s.label for s in samples),
                  "type_labels": has_type_labels(samples)}
    inf = cfg["infer"]
    pooling = {k: inf.get(k) for k in ("pooling", "pool_k", "null_tau_s", "risk_step_s", "smooth_s")}
    for kind in kinds:
        # калибровка (null-CDF чистых роликов + порог) — внутренним grouped CV по обучающим данным;
        # затем финальная модель на всех данных (calibrate перезаписывает s.X, поэтому spec — после неё)
        cal = calibrate(kind, samples, bank, cfg, seed=seed, epochs=epochs)
        spec = fit_spec([fs for s in samples for fs in s.versions], bank, cfg)
        prepare_samples(list(samples), bank, spec)
        model = fit_predictor(kind, samples, spec, cfg, seed=seed, epochs=epochs, deadline=deadline)
        thr, src = cal["threshold"], f"inner grouped CV ({cal['n_inner']} фолдов) по обучающим данным, OOF"
        if threshold not in ("auto", "", None):
            thr, src = float(threshold), "cli"
        meta = {"threshold": thr, "threshold_source": src, "n_train": len(samples),
                "reproducibility": _repro(cfg),
                "train_groups": sorted(s.group for s in samples),
                "pooling": pooling, "null": cal["null"],
                "feature_signature": stable_hash(feature_signature(cfg, spec.backend), 16),
                "type_head_source": type_head_source(samples, cfg) if model.type_trained else None,
                "calibration": {k: cal[k] for k in ("n_inner", "n_null_blocks", "oof_auc")},
                "prompts": {"en": bank.texts_en, "ru": bank.texts_ru, "owner": bank.owner},
                "cv": ({k: cv_report[kind][k] for k in ("threshold_free", "nested", "protocol", "outer", "inner")}
                       if cv_report and kind in cv_report else None)}
        d = save_checkpoint(out_dir, model, spec, meta)
        info["kinds"][kind] = {"dir": str(d), "threshold": thr, "threshold_source": src,
                               "type_head_trained": bool(model.type_trained)}
        log.info("train: %s сохранён в %s (порог %.3f, %s)", kind, d, thr, src)
    if select == "auto":
        select = (cv_report or {}).get("selected") or ("baseline" if "baseline" in kinds else kinds[0])
        if select not in kinds:
            select = kinds[0]
    (out_dir / "selected.json").write_text(json.dumps({"kind": select}), encoding="utf-8")
    info["selected"] = select
    info["elapsed_s"] = round(time.monotonic() - t0, 1)
    return info


def _repro(cfg: dict) -> dict:
    """Что нужно, чтобы воспроизвести чекпоинт: версия признаков, хэш конфига, коммит, сид."""
    import subprocess

    from .config import REPO_ROOT, stable_hash
    from .features import FEATURE_VERSION

    commit = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                                timeout=5).stdout.strip() or None
    except Exception:
        pass
    return {"feature_version": FEATURE_VERSION, "config_hash": stable_hash(cfg, 16), "git_commit": commit,
            "seed": int(cfg["train"]["seed"]), "model_id": cfg["features"]["model_id"]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ml.temporal: обучение")
    ap.add_argument("--index", default=None)
    ap.add_argument("--kind", default="both", choices=["baseline", "temporal", "both"])
    ap.add_argument("--select", default="auto", choices=["auto", "baseline", "temporal"])
    ap.add_argument("--threshold", default="auto")
    ap.add_argument("--cv-report", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-aug", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    from .data_index import bank_for_index, load_samples

    samples, idx = load_samples(cfg, a.index, max_aug=a.max_aug)
    bank = bank_for_index(cfg, idx)
    kinds = ["baseline", "temporal"] if a.kind == "both" else [a.kind]
    if "temporal" in kinds:
        try:
            import torch  # noqa: F401
        except ImportError:
            log.warning("torch не установлен — обучаю только baseline")
            kinds = [k for k in kinds if k != "temporal"] or ["baseline"]
    rep_path = Path(a.cv_report) if a.cv_report else reports_dir(cfg) / "temporal_cv.json"
    cv_report = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.exists() else None
    out = Path(a.out) if a.out else weights_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    info = train_all(samples, bank, cfg, kinds, out, a.select, a.threshold, cv_report, a.epochs)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
