"""Абляции: правила вердикта на синтетике + сквозной прогон двух рук с фейковым пайплайном (mock — проверка кода)."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from eval.ablate import compare, mcnemar_exact
from eval.common import load_config, normalize_video_id
from eval.labels import LabelRow, load_ground_truth, write_labels_csv
from eval.metrics import evaluate

from .conftest import canonical_coverage, jpeg, make_pred, make_violation, write_synthetic_video


def _gt(tmp_path: Path):
    rows = [LabelRow(f"X-{i}", "violation" if i <= 5 else "no_violation", [2.0] if i <= 5 else []) for i in range(1, 11)]
    p = tmp_path / "labels.csv"
    write_labels_csv(rows, p)
    return load_ground_truth(p)


def _run(gt, detect: set[int], lat: float = 10.0, proc_mb=4000.0, dev_mb=None, dev_base_mb=None):
    """proc_mb — память воркера пайплайна (A03 meta.stages); dev_mb/dev_base_mb — занятость всей карты (nvidia-smi)."""
    stages = {"detect": {"status": "ok", "vram_peak_mb": proc_mb}} if proc_mb is not None else {}
    preds = {f"X-{i}": make_pred(f"j{i}", [make_violation(2.0)] if i in detect else [], meta={"stages": stages})
             for i in range(1, 11)}
    info = {k: {"latency_s": lat, "vram_device_peak_mb": dev_mb, "vram_device_baseline_mb": dev_base_mb}
            for k in preds}
    return evaluate(gt, preds, info, load_config())


def test_mcnemar():
    assert mcnemar_exact(0, 0) is None
    assert mcnemar_exact(5, 0) == pytest.approx(0.0625)
    assert mcnemar_exact(3, 3) == 1.0


def test_verdict_rules(tmp_path: Path):
    gt = _gt(tmp_path)
    base = _run(gt, {1, 2, 3})                                     # FN=2, FP=0
    better = compare(base, _run(gt, {1, 2, 3, 4, 5}))              # FN 2→0
    assert better["verdict"] == "PASS" and better["paired"]["fixed"] == 2 and better["deltas"]["fn"] == -2
    noisy = compare(base, _run(gt, {1, 2, 3, 4, 5, 6, 7, 8}))      # recall↑, но FP +3 > допуска
    assert noisy["verdict"] == "FAIL" and any("рост recall" in r for r in noisy["reasons_fail"])
    same = compare(base, _run(gt, {1, 2, 3}))
    assert same["verdict"] == "INCONCLUSIVE"
    slow = compare(base, _run(gt, {1, 2, 3}, lat=70.0))
    assert slow["verdict"] == "FAIL" and any("SLA" in r for r in slow["reasons_fail"])
    fat = compare(base, _run(gt, {1, 2, 3}, proc_mb=7900.0))      # воркер пайплайна сам превысил бюджет
    assert fat["verdict"] == "FAIL" and any("VRAM процесса/воркера" in r for r in fat["reasons_fail"])
    # аудит: карта хоста занята чужими процессами (7675 МБ), сам пайплайн — 4000 МБ → не FAIL, но предупреждение
    busy = compare(base, _run(gt, {1, 2, 3}, proc_mb=4000.0, dev_mb=7675.0, dev_base_mb=3800.0))
    assert busy["verdict"] == "INCONCLUSIVE" and busy["vram_check"]["status"] == "OK"
    assert any("все процессы" in w for w in busy["warnings"])
    # память воркера не сообщена → бюджет по приросту устройства над фоном
    delta_ok = compare(base, _run(gt, {1, 2, 3}, proc_mb=None, dev_mb=7675.0, dev_base_mb=3800.0))
    assert delta_ok["vram_check"]["measure"].startswith("прирост") and delta_ok["verdict"] == "INCONCLUSIVE"
    delta_bad = compare(base, _run(gt, {1, 2, 3}, proc_mb=None, dev_mb=7900.0, dev_base_mb=200.0))
    assert delta_bad["verdict"] == "FAIL"
    # нет ни одного измерения → NOT MEASURED (не 0 и не PASS), бюджет не подгоняется
    none = compare(base, _run(gt, {1, 2, 3}, proc_mb=None))
    assert none["vram_check"]["status"] == "NOT MEASURED" and none["not_measured"] and none["verdict"] == "INCONCLUSIVE"
    assert none["vram_check"]["budget_mb"] == 7500
    worse = compare(base, _run(gt, {1, 2}))
    assert worse["verdict"] == "FAIL" and worse["paired"]["broken"] == 1
    prot = {"labels_sha256": "l", "mode": "inproc", "subset": "all", "eval_version": "e", "inputs": {"inputs_sha": "a"},
            "eval_config": {}, "pipeline_env": {"SENTINEL_X": "0"}}
    inv = compare(base, base, prot_base=prot, prot_cand=dict(prot, inputs={"inputs_sha": "b"}))
    assert inv["verdict"] == "INVALID"
    two = compare(base, base, prot_base=prot, prot_cand=dict(prot, pipeline_env={"SENTINEL_X": "1", "SENTINEL_Y": "1"}))
    assert any("больше одного" in w for w in two["warnings"])


def test_ablate_end_to_end_inproc(tmp_path: Path, monkeypatch, mock_mode):
    pytest.importorskip("cv2")
    from eval import ablate
    from eval._contracts import AnalysisResult, EvidenceFrame, Violation

    gtp = tmp_path / "labels.csv"
    write_labels_csv([LabelRow(f"X-{i}", "violation" if i <= 3 else "no_violation", [0.4] if i <= 3 else [])
                      for i in range(1, 7)], gtp)
    vdir = tmp_path / "v"
    vdir.mkdir()
    for i in range(1, 7):
        write_synthetic_video(vdir / f"X-{i}.mov", seconds=1.0, fps=5)

    def run(video_path, job_id, progress, deadline_s=240):
        import os

        n = int(normalize_video_id(Path(video_path).name).split("-")[1])
        hit = n <= (3 if os.environ.get("SENTINEL_FAKE_SENS") == "high" else 2)   # кандидат находит X-3
        vios = [Violation(type="t", timecode_sec=0.4, description="Признаки возможного нарушения: …",
                          evidence_frames=[EvidenceFrame(0.4, jpeg())], type_id="V01", confidence=0.9)] if hit else []
        return AnalysisResult(job_id=job_id, status="done", verdict="violation" if hit else "no_violation",
                              violations=vios, timeline=[(0.0, 0.1), (0.4, 0.9 if hit else 0.2)],
                              meta={"version": "fake", "mock": True, "coverage": canonical_coverage(1.0),
                                    "scores_raw": {"max": 0.9 if hit else 0.2}})

    pipe = types.ModuleType("backend.pipeline")
    pipe.run, pipe.warmup = run, (lambda: None)
    pkg = types.ModuleType("backend")
    pkg.pipeline = pipe
    monkeypatch.setitem(sys.modules, "backend", pkg)
    monkeypatch.setitem(sys.modules, "backend.pipeline", pipe)
    monkeypatch.delenv("SENTINEL_FAKE_SENS", raising=False)
    log = tmp_path / "reports" / "runs.jsonl"
    rc = ablate.main(["--name", "H01-sens", "--videos", str(vdir), "--labels", str(gtp), "--types", str(tmp_path / "n"),
                      "--manifest", str(tmp_path / "none.csv"), "--out", str(tmp_path / "abl"), "--log", str(log),
                      "--baseline-env", "SENTINEL_FAKE_SENS=low", "--candidate-env", "SENTINEL_FAKE_SENS=high",
                      "--inproc-arms"])
    assert rc == 0
    cmp = json.loads((tmp_path / "abl" / "compare.json").read_text(encoding="utf-8"))
    assert cmp["verdict"] == "PASS" and cmp["deltas"]["fn"] == -1 and cmp["changed_params"] == ["SENTINEL_FAKE_SENS"]
    pub = json.loads((tmp_path / "abl" / "compare_public.json").read_text(encoding="utf-8"))
    assert "fixed_ids" not in json.dumps(pub) and pub["paired"]["fixed"] == 1
    rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert [r["role"] for r in rows] == ["baseline", "candidate", "ablation"]
    assert (tmp_path / "reports" / "runs.html").exists()
    ch = (tmp_path / "abl" / "compare.html").read_text(encoding="utf-8")
    assert "ТОЛЬКО ЛОКАЛЬНО" in ch and "X-3" in ch and "baseline/report.html" in ch
    assert (tmp_path / "abl" / "candidate" / "report.html").exists()
