from __future__ import annotations

from pathlib import Path

from eval.common import load_config, write_json
from eval.labels import LabelRow, load_ground_truth, write_labels_csv
from eval.metrics import evaluate
from eval.report import build_report, main

from .conftest import make_pred, make_violation


def _res(tmp_path: Path):
    rows = [LabelRow("X-1", "violation", [22.0]), LabelRow("X-2", "no_violation", []), LabelRow("X-3", "violation", [5.0])]
    p = tmp_path / "labels.csv"
    write_labels_csv(rows, p)
    preds = {"X-1": make_pred("a", [make_violation(22.3)]), "X-2": make_pred("b", [make_violation(9.0)]),
             "X-3": make_pred("c", [])}
    return evaluate(load_ground_truth(p), preds, {"X-1": {"latency_s": 4.2}}, load_config()), preds


def test_report_self_contained(tmp_path: Path):
    res, preds = _res(tmp_path)
    h = build_report(res, preds, robustness={"disclaimer": "синтетика", "per_variant": {
        "mirror": {"n": 1, "flip_rate": 0.0, "timecode_stable_rate": 1.0, "mean_shift_s": 0.1, "accuracy_vs_gt": 1.0}}})
    assert h.startswith("<!doctype html>") and "ТОЛЬКО ЛОКАЛЬНО" in h
    assert "<script src" not in h and "http://" not in h and "https://" not in h.replace("https://adilet", "")
    assert "ПРОПУСК" in h and "ложное срабатывание" in h and "data:image/jpeg;base64," in h and "4.2" in h
    assert "data:image" not in build_report(res, preds, with_frames=False)


def test_report_cli(tmp_path: Path):
    res, preds = _res(tmp_path)
    run = tmp_path / "run"
    write_json(run / "metrics.json", res)
    for k, v in preds.items():
        write_json(run / "preds" / f"{k}.json", v)
    assert main(["--run", str(run)]) == 0 and (run / "report.html").exists()
