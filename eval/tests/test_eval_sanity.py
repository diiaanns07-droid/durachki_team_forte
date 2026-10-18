"""Санити-полы на синтетике: короткие ролики «с нарушением» — длительность сама по себе даёт AUC > 0.5."""
from __future__ import annotations

from pathlib import Path

import pytest

from eval.labels import LabelRow, load_ground_truth, write_labels_csv
from eval.sanity import auc, format_sanity, load_score_csv, sanity, spearman, stratified_auc, timeline_poolings

from .conftest import make_pred


def _gt(tmp_path: Path):
    rows = []
    for i in range(1, 13):
        pos = i <= 6
        rows.append(LabelRow(f"X-{i}", "violation" if pos else "no_violation", [10.0 + i] if pos else []))
    p = tmp_path / "labels.csv"
    write_labels_csv(rows, p)
    dur = {f"X-{i}": (30.0 + i if i <= 6 else 60.0 + 5 * i) for i in range(1, 13)}  # позитивы короче
    return load_ground_truth(p), dur


def test_auc_spearman_basics():
    assert auc([0.9, 0.8, 0.1, 0.2], [True, True, False, False]) == 1.0
    assert auc([0.5, 0.5], [True, False]) == 0.5
    assert auc([1.0], [True]) is None
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert stratified_auc([0.1, 0.9, 0.2, 0.8, 0.3, 0.7], [False, True, False, True, False, True],
                          [1, 1, 2, 2, 3, 3]) == 1.0


def test_floors_and_duration_leak_warning(tmp_path: Path):
    gt, dur = _gt(tmp_path)
    length_score = {k: d / 100 for k, d in dur.items()}           # «скор» = длина ролика (max-пулинг-артефакт)
    good = {f"X-{i}": (0.9 if i <= 6 else 0.1) for i in range(1, 13)}
    s = sanity(gt, dur, {"length_like": length_score, "good": good})
    f = s["floors"]
    assert f["all_violation"]["fp"] == 6 and f["all_violation"]["recall"] == 1.0 and f["all_clean"]["fn"] == 6
    assert f["duration_shorter_is_violation"]["auc"] == 1.0         # синтетика: длина идеально разделяет
    assert "timecode_fixed_position" in f
    bad = s["scores"]["length_like"]
    assert bad["auc"] == 0.0 and any("длительност" in w for w in bad["warnings"])
    assert any("AUC < 0.5" in w for w in bad["warnings"])
    assert s["scores"]["good"]["auc"] == 1.0
    assert "разведочная" in format_sanity(s)


def test_timeline_poolings_and_csv(tmp_path: Path):
    preds = {"X-1": make_pred("a", [], timeline=[[0, 0.1], [1, 0.9], [2, 0.2]]),
             "X-2": make_pred("b", [], timeline=[[0, 0.3], [1, 0.3]])}
    pools = timeline_poolings(preds)
    assert pools["tl_max"]["X-1"] == 0.9 and pools["tl_prominence"]["X-2"] == 0.0
    assert pools["tl_mean"]["X-1"] == pytest.approx(0.4)
    csvp = tmp_path / "oof.csv"
    csvp.write_text("video_id,label,score_oof\nХ-1,violation,0.7\nX-2,no_violation,0.2\n", encoding="utf-8")
    name, sc = load_score_csv(f"{csvp}:score_oof")
    assert name == "oof:score_oof" and sc == {"X-1": 0.7, "X-2": 0.2}
