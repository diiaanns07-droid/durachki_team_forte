from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from eval.common import load_config
from eval.extra import format_strata, freeze, make_extra_splits, read_extra, stratified_report, to_eval_labels
from eval.labels import load_ground_truth
from eval.metrics import evaluate

from .conftest import make_pred, make_violation

EXTRA = """video_id,file,label,t_all,type_id,camera_id,scene_id,layout,lighting
E-1,C1/e1.mov,violation,12;40,V02;V03,C1,S1,L1,normal
E-2,C3/e2.mov,violation,12;40,V02;V03,C3,S1,L1,normal
E-3,C1/e3.mov,no_violation,,,C1,S2,L2,off
E-4,C3/e4.mov,no_violation,,,C3,S3,L2,dim
E-5,C2/e5.mov,violation,7,V05,C2,S4,L1,normal
E-6,C2/e6.mov,???,,,C2,S5,L1,normal
"""


@pytest.fixture
def extra(tmp_path: Path) -> Path:
    p = tmp_path / "extra" / "labels.csv"
    p.parent.mkdir()
    p.write_text(EXTRA, encoding="utf-8")
    return p


def test_read_and_convert(extra: Path):
    rows, problems = read_extra(extra)
    assert len(rows) == 6 and any("неизвестная метка" in x for x in problems)     # unknown ≠ false
    lab, types = to_eval_labels(rows, extra.parent)
    gt = load_ground_truth(lab, types, dedupe=False)
    assert gt["E-1"].times == [12.0, 40.0] and gt["E-1"].types == [(12.0, "V02"), (40.0, "V03")]
    assert gt["E-6"].label == "unknown"


def test_freeze_c3_before_tuning(extra: Path):
    rows, _ = read_extra(extra)
    man = freeze(rows, extra.parent / "holdout_manifest.json", "C3")
    assert man["video_ids"] == ["E-2", "E-4"] and man["scene_overlap_not_frozen"] == ["E-1"]
    with pytest.raises(FileExistsError):
        freeze(rows, extra.parent / "holdout_manifest.json", "C3")               # заморозка необратима без --force
    strict = freeze(rows, extra.parent / "m2.json", "C3", strict_scenes=True)
    assert strict["video_ids"] == ["E-1", "E-2", "E-4"]
    sp = make_extra_splits(rows, man, extra.parent / "splits.csv")
    with open(sp, encoding="utf-8") as f:
        split = {r["video_id"]: r["split"] for r in csv.DictReader(f)}
    assert split["E-2"] == split["E-4"] == "holdout" and split["E-6"] == "excluded"
    assert {split["E-1"], split["E-3"], split["E-5"]} <= {"train", "calib"}
    assert json.loads((extra.parent / "splits_manifest.json").read_text(encoding="utf-8"))["frozen"] is True


def test_strata_report(extra: Path):
    rows, _ = read_extra(extra)
    lab, types = to_eval_labels(rows, extra.parent)
    gt = load_ground_truth(lab, types, dedupe=False)
    preds = {"E-1": make_pred("a", [make_violation(12.0, type_id="V02"), make_violation(40.0, type_id="V03")]),
             "E-2": make_pred("b", []),                                   # C3 «другой угол» — пропуск
             "E-3": make_pred("c", [make_violation(5.0)]),               # свет выключен — ложная тревога
             "E-4": make_pred("d", []), "E-5": make_pred("e", [make_violation(7.5, type_id="V05")])}
    res = evaluate(gt, preds, {}, load_config(overrides={"dedupe_groups": False}))
    assert res["summary"]["types"]["available"] and res["summary"]["types"]["accuracy"] == pytest.approx(3 / 5)
    rep = stratified_report(res, rows)
    assert rep["camera_id"]["C3"]["fn"] == 1 and rep["camera_id"]["C1"]["fp"] == 1
    assert rep["lighting"]["off"]["fp"] == 1 and rep["layout"]["L1"]["recall"] == pytest.approx(2 / 3, abs=1e-3)
    assert "НЕ признаки" in format_strata(rep)


def test_kind_guard(tmp_path: Path, capsys):
    from eval import run as runmod
    from eval.labels import LabelRow, write_labels_csv

    lab = tmp_path / "l.csv"
    write_labels_csv([LabelRow("X-1", "violation", [1.0])], lab)
    (tmp_path / "preds").mkdir()
    runmod.main(["--labels", str(lab), "--preds", str(tmp_path / "preds"), "--kind", "holdout_frozen", "--no-log",
                 "--types", str(tmp_path / "none.csv")])
    assert "помечаю как exploratory" in capsys.readouterr().out
    m = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert m["scorecard"]["evaluation_kind"] == "exploratory"
