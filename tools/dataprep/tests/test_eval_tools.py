"""eval_run / error_review — на синтетическом прогоне (без реальных видео и ответов сервиса)."""
from __future__ import annotations

import base64
import csv
import json
import math


def _root(tmp_path):
    (tmp_path / "data").mkdir()
    cols = ["video_id", "file", "label", "t_sec", "sha256", "duration_s"]
    rows = [("X-1", "violation", "10", "a"), ("X-2", "violation", "57", "d"), ("X-3", "violation", "68", "d"),
            ("X-4", "no_violation", "", "b"), ("X-5", "no_violation", "", "c")]
    with open(tmp_path / "data/manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for vid, lab, t, sha in rows:
            w.writerow([vid, f"{vid}.mov", lab, t, sha, "80"])
    run = tmp_path / "run"
    run.mkdir()
    summ = [dict(name="X-1", sha256="a", status="done", verdict="violation", type_id="V05", timecode_sec=12.0,
                 confidence=0.9, n_violations=1, seconds=3.0),
            dict(name="X-2", sha256="d", status="done", verdict="violation", type_id="V01", timecode_sec=67.5,
                 confidence=0.6, n_violations=1, seconds=4.0),
            dict(name="X-3", sha256="d", status="done", verdict="no_violation", type_id="", timecode_sec=None,
                 confidence=None, n_violations=0, seconds=4.0),
            dict(name="X-4", sha256="b", status="done", verdict="violation", type_id="V05", timecode_sec=5.0,
                 confidence=0.7, n_violations=2, seconds=2.0),
            dict(name="X-5", sha256="c", status="done", verdict="no_violation", type_id="", timecode_sec=None,
                 confidence=None, n_violations=0, seconds=2.0)]
    (run / "summary.json").write_text(json.dumps(summ), encoding="utf-8")
    (run / "X-1.json").write_text(json.dumps({"meta": {"scores_raw": {"risk_max": 0.8, "max_conf": 0.9}},
                                              "coverage": {"stages": {"temporal": "error", "vlm_verify": "skipped"}}}),
                                  encoding="utf-8")
    return tmp_path, run


def test_per_video_outcomes_and_timecode_alternatives(dp, tmp_path):
    er = dp("eval_run")
    root, run = _root(tmp_path)
    rows = {x["video_id"]: x for x in er.per_video(run, root)}
    assert [rows[v]["outcome"] for v in ("X-1", "X-2", "X-3", "X-4", "X-5")] == ["TP", "TP", "FN", "FP", "TN"]
    assert rows["X-1"]["dt"] == 2.0
    assert rows["X-2"]["dt"] == 0.5  # дубликат: ближайшая из меток группы (57/68)
    assert rows["X-1"]["risk_max"] == 0.8 and rows["X-1"]["stages"]["temporal"] == "error"


def test_summarize_prints_aggregates_only(dp, tmp_path, capsys):
    er = dp("eval_run")
    root, run = _root(tmp_path)
    er.summarize(er.per_video(run, root), "run")
    out = capsys.readouterr().out
    assert "accuracy 60%" in out and "recall 2/3" in out and "specificity 1/2" in out
    assert "temporal: {'error': 1}" in out


def test_auc_and_wilson(dp):
    er = dp("eval_run")
    assert er.auc([0.9, 0.8], [0.1, 0.2]) == 1.0 and er.auc([0.5], [0.5]) == 0.5 and math.isnan(er.auc([], [1]))
    lo, hi = er.wilson(21, 42)
    assert 0.35 < lo < 0.5 < hi < 0.65


def test_error_review_helpers(dp, tmp_path):
    rv = dp("error_review")
    jpg = b"\xff\xd8\xff\xe0synthetic"
    names = rv.save_evidence([{"image_base64": base64.b64encode(jpg).decode()}, {"image_base64": ""}, "x"],
                             tmp_path, "X-9_v0")
    assert names == ["X-9_v0_e0.jpg"] and (tmp_path / names[0]).read_bytes() == jpg
    svg = rv.sparkline([[0, 0.1], [5, 0.9], [10, None]], 10.0, [5.0], [6.0])
    assert svg.count("<line") == 2 and "polyline" in svg
