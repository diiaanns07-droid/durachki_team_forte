"""Замер L2 (compare): метрики, двухфазность (разметка — только после прогонов), CLI на синтетике."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys

from backend.pipeline import compare as cmp
from backend.pipeline.tests.conftest import ROOT


def _row(name, status="done", verdict="violation", tc=None, ts=None, wall=1.0):
    v = [{"type_id": "V02", "timecode_sec": tc, "t_start": ts if ts is not None else tc, "t_end": tc,
          "confidence": 0.8}] if tc is not None else []
    return {"name": name, "status": status, "verdict": verdict, "violations": v, "wall_s": wall,
            "stage_s": {"detect": 1.0}, "vram_mb": {"detect": 900.0}, "complete": status == "done",
            "coverage": {"ratio": 0.95 if status == "done" else 0.4}}


def test_metrics_counts_and_timecodes():
    labels = {"X-1": {"label": "violation", "times": [22.0], "conflict": False},
              "X-2": {"label": "no_violation", "times": [], "conflict": False},
              "X-3": {"label": "violation", "times": [57.0, 68.0], "conflict": True},   # конфликт таймкодов
              "X-4": {"label": "violation", "times": [10.0], "conflict": False},
              "X-5": {"label": "no_violation", "times": [], "conflict": False},
              "X-6": {"label": "conflict", "times": [], "conflict": True}}
    rows = [_row("X-1", tc=23.5, ts=21.8), _row("X-2", verdict="no_violation"), _row("X-3", tc=67.0),
            _row("X-4", status="failed", verdict=None), _row("X-5", tc=5.0), _row("X-6", tc=1.0), _row("X-9", tc=1.0)]
    m = cmp.metrics(rows, labels)
    v = m["video"]
    assert (v["tp"], v["fp"], v["tn"], v["fn"]) == (2, 1, 1, 0)
    assert v["failed_on_violation"] == 1 and v["recall"] == round(2 / 3, 4)   # failed на нарушении = пропуск
    assert v["excluded_conflict"] == 1 and v["unlabeled"] == 1
    t = m["timecode"]
    assert t["n"] == 2 and t["hit_peak@1s"] == 0.5 and t["hit_peak@2s"] == 1.0  # 1.5 с и 1.0 с (ближайшая из t_all)
    assert t["hit_onset@1s"] == 1.0                                              # 0.2 с и 1.0 с
    assert m["vram_peak_mb"]["detect"] == 900.0 and m["completeness"]["incomplete"] == 1
    assert m["completeness"]["min_coverage_ratio"] == 0.4
    assert "X-4" in m["errors"]["failed"] and "X-5" in m["errors"]["fp"]


def test_labels_read_only_after_all_runs(tmp_path, monkeypatch):
    order = []
    (tmp_path / "v").mkdir()
    (tmp_path / "v" / "X-1.mp4").write_bytes(b"0")
    monkeypatch.setattr(cmp, "run_mode", lambda m, *a, **k: order.append(f"run:{m}") or [])
    monkeypatch.setattr(cmp, "load_labels", lambda p: order.append("labels") or {})
    cmp.compare(tmp_path / "v", tmp_path / "labels.csv", ["baseline", "refine"], tmp_path / "out")
    assert order == ["run:baseline", "run:refine", "labels"]


def test_load_labels_schema(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("video_id,file,label,t_sec,md5,group,t_all,conflict,raw\n"
                 "X-1,X-1.mov,violation,22,a,g1,22,0,x\n"
                 "X-47,Х-47.mov,violation,68,b,g4,57;68,1,x\n"
                 "X-2,X-2.mov,no_violation,,c,g2,,0,x\n", encoding="utf-8")
    lab = cmp.load_labels(p)
    assert lab["X-1"]["times"] == [22.0] and lab["X-47"]["times"] == [57.0, 68.0] and lab["X-47"]["conflict"]
    assert lab["X-2"]["label"] == "no_violation" and lab["X-2"]["times"] == []


def test_compare_cli_end_to_end(synth_video, tmp_path):
    vids = tmp_path / "videos"
    vids.mkdir()
    shutil.copy(synth_video, vids / "X-1.mp4")
    shutil.copy(synth_video, vids / "Х-2.mp4")       # кириллица
    labels = tmp_path / "labels.csv"
    labels.write_text("video_id,file,label,t_sec,md5,group,t_all,conflict,raw\n"
                      "X-1,X-1.mp4,violation,2,a,g1,2,0,\nX-2,Х-2.mp4,no_violation,,a,g1,,0,\n", encoding="utf-8")
    out = tmp_path / "rep"
    p = subprocess.run([sys.executable, "-m", "backend.pipeline", "compare", str(vids), "--labels", str(labels),
                        "--modes", "baseline,refine", "--out", str(out), "--mock", "--inline",
                        "--emit-budgets", str(tmp_path / "budgets.yaml")],
                       cwd=ROOT, capture_output=True, text=True, timeout=240)
    assert p.returncode == 0, p.stderr[-2000:]
    rep = json.loads((out / "compare.json").read_text(encoding="utf-8"))
    assert set(rep["results"]) == {"baseline", "refine"}
    for m in rep["results"].values():
        assert m["video"]["n"] == 2 and "p90_s" in m["latency"]
    assert (out / "baseline" / "preds" / "X-2.json").exists() and (out / "compare.md").exists()
    assert "не независимый тест" in (out / "compare.md").read_text(encoding="utf-8")
    assert "budgets_s" in (tmp_path / "budgets.yaml").read_text(encoding="utf-8")
