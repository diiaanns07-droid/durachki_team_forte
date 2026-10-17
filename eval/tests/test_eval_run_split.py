"""run.py с фейковым backend.pipeline (mock: проверяет код, не качество CV) + групповой split."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from eval.common import load_config, normalize_video_id
from eval.labels import LabelRow, assign_groups, write_labels_csv
from eval.split import ids_for, load_splits, log_holdout_access, make_splits, write_splits

from .conftest import write_synthetic_video


def _labels(tmp_path: Path, n: int = 12) -> Path:
    rows = [LabelRow(f"X-{i}", "violation" if i % 2 else "no_violation", [1.0] if i % 2 else [], md5=f"m{i}")
            for i in range(1, n + 1)]
    rows += [LabelRow("X-47", "violation", [1.5], md5="m1"), LabelRow("X-48", "no_violation", [], md5="m2")]
    assign_groups(rows)
    p = tmp_path / "labels.csv"
    write_labels_csv(rows, p)
    return p


def test_split_groups_never_cross(tmp_path: Path):
    lab = _labels(tmp_path)
    sess = tmp_path / "sessions.csv"
    sess.write_text("video_id,session_id\nX-3,s1\nX-5,s1\n", encoding="utf-8")
    rows = make_splits(lab, sess, holdout=0.25, calib=0.25, folds=3, seed=1)
    by = {r["video_id"]: r for r in rows}
    assert by["X-1"]["split"] == by["X-47"]["split"]          # дубликаты по md5 — вместе
    assert by["X-2"]["split"] == by["X-48"]["split"]
    assert by["X-3"]["split"] == by["X-5"]["split"]           # одна сессия — вместе
    assert {r["split"] for r in rows} == {"train", "calib", "holdout"}
    assert rows == make_splits(lab, sess, holdout=0.25, calib=0.25, folds=3, seed=1)   # детерминизм
    out = tmp_path / "splits.csv"
    write_splits(rows, out)
    sp = load_splits(out)
    hold = ids_for(sp, "holdout")
    assert hold and all(sp[v]["split"] == "holdout" for v in hold if v in sp)
    assert json.loads((tmp_path / "splits_manifest.json").read_text(encoding="utf-8"))["holdout_sha256"]
    log = tmp_path / "holdout_access.jsonl"
    assert log_holdout_access(log, {"x": 1}) == 0 and log_holdout_access(log, {"x": 2}) == 1


def _fake_pipeline(calls: list) -> types.ModuleType:
    """Фейковый backend.pipeline (MOCK, явно помечен): нарушение у нечётных X-n на 1.0 с (по имени файла).

    Ответы — по контракту A01: done/no_violation только с подтверждённой полнотой (канонический meta.coverage),
    meta.mock=True → сериализатор A01 принимает их лишь при явном SENTINEL_MOCK=1 (фикстура mock_mode)."""
    from eval._contracts import AnalysisResult, EvidenceFrame, Violation

    from .conftest import canonical_coverage, jpeg

    mod = types.ModuleType("backend.pipeline")

    def run(video_path, job_id, progress, deadline_s=240):
        calls.append(video_path)
        vid = normalize_video_id(Path(video_path).name)
        n = int(vid.split("-")[1])
        if n == 7:
            raise RuntimeError("deadline exceeded")
        vios = [Violation(type="Посторонний", timecode_sec=1.0, evidence_frames=[EvidenceFrame(1.0, jpeg())],
                          description="Признаки возможного нарушения: в зоне кассира второй человек.", type_id="V01",
                          confidence=0.9)] if n % 2 else []
        return AnalysisResult(job_id=job_id, status="done", verdict="violation" if vios else "no_violation",
                              violations=vios, meta={"version": "fake-1", "duration": 3.0, "mock": True,
                                                     "coverage": canonical_coverage(3.0),
                                                     "scores_raw": {"max": 0.9 if vios else 0.1}})

    mod.run = run
    mod.warmup = lambda: None
    mod.release_gpu = lambda: calls.append("release_gpu") or {}      # A03: остановить воркер после прогона
    return mod


def test_run_inproc_end_to_end(tmp_path: Path, monkeypatch, capsys, mock_mode):
    pytest.importorskip("cv2")
    from eval import run as runmod

    lab = _labels(tmp_path, n=8)
    vdir = tmp_path / "videos"
    vdir.mkdir()
    for i in list(range(1, 9)) + [47]:
        name = f"Х-{i}.mov" if i % 3 == 0 else f"X-{i}.mov"       # кириллица и латиница
        write_synthetic_video(vdir / name, seconds=1.0, fps=5)
    calls: list = []
    pkg = types.ModuleType("backend")
    pipe = _fake_pipeline(calls)
    pkg.pipeline = pipe
    monkeypatch.setitem(sys.modules, "backend", pkg)
    monkeypatch.setitem(sys.modules, "backend.pipeline", pipe)
    out = tmp_path / "run"
    rc = runmod.main(["--videos", str(vdir), "--labels", str(lab), "--out", str(out), "--raw",
                      "--types", str(tmp_path / "none.csv"), "--log", str(tmp_path / "reports" / "runs.jsonl"),
                      "--label", "baseline-mock", "--env", "SENTINEL_FLAG_TEST=1"])
    assert rc == 0
    assert not any("47" in Path(c).name for c in calls)                # дубликат X-47 не гоняем
    assert calls[-1] == "release_gpu"                                   # воркер пайплайна остановлен (иначе висим на выходе)
    preds = sorted(p.name for p in (out / "preds").glob("X-*.json"))
    assert len(preds) == 8
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    v = m["summary"]["video"]
    assert v["fn"] == 1 and v["failed"] == 1 and v["timeouts"] == 1    # X-7 упал по deadline
    assert m["scorecard"]["versions"]["pipeline_versions"] == ["fake-1"]
    pub = json.loads((out / "summary_public.json").read_text(encoding="utf-8"))
    txt = json.dumps(pub, ensure_ascii=False)
    assert "image_base64" not in txt and "Признаки" not in txt and "X-7" not in txt   # только агрегаты
    raw = [json.loads(x) for x in (out / "scores_raw.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["video_id"] for r in raw} >= {"X-1", "X-2"} and raw[0]["scores_raw"] is not None
    assert "SCORECARD" in capsys.readouterr().out
    prot = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    assert prot["inputs"]["n"] == 8 and prot["inputs"]["inputs_sha"] and prot["labels_sha256"]
    assert prot["pipeline_env"]["SENTINEL_FLAG_TEST"] == "1" and prot["evaluation_kind"] == "exploratory"
    log = [json.loads(x) for x in (tmp_path / "reports" / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert log[-1]["label"] == "baseline-mock" and log[-1]["metrics"]["fn"] == 1
    assert "image_base64" not in json.dumps(log) and "Признаки" not in json.dumps(log)
    info = list(json.loads((out / "preds" / "_runinfo.json").read_text()).values())
    # измерители хоста в юниттесте выключены (фикстура): состояние чужой GPU не попадает в метрики; None, не 0
    assert all(r["vram_device_peak_mb"] is None and r["vram_eval_process_peak_mb"] is None for r in info)
    assert m["scorecard"]["mock_results"] == 7 and m["scorecard"]["vram_process_peak_mb_max"] is None


def test_holdout_requires_final(tmp_path: Path):
    from eval import run as runmod

    lab = _labels(tmp_path)
    write_splits(make_splits(lab), tmp_path / "splits.csv")
    with pytest.raises(SystemExit):
        runmod.main(["--labels", str(lab), "--splits", str(tmp_path / "splits.csv"), "--subset", "holdout",
                     "--preds", str(tmp_path)])


def test_select_videos_dedupes(tmp_path: Path):
    from eval.run import select_videos

    lab = _labels(tmp_path, n=3)
    vdir = tmp_path / "v"
    vdir.mkdir()
    for n in ("X-1.mov", "Х-47.mov", "X-2.MOV"):
        (vdir / n).write_bytes(b"x")
    ids = [v for v, _ in select_videos(vdir, lab, load_config())]
    assert ids == ["X-1", "X-2"]
    ids = [v for v, _ in select_videos(vdir, lab, load_config(), all_members=True)]
    assert "X-47" in ids


def test_run_main_restores_environment(tmp_path: Path):
    """eval.run.main не оставляет --env/raw-флагов в окружении процесса после себя."""
    import os

    from eval import run as runmod

    lab = _labels(tmp_path, n=2)
    (tmp_path / "preds").mkdir()
    before = dict(os.environ)
    runmod.main(["--labels", str(lab), "--preds", str(tmp_path / "preds"), "--no-log", "--raw",
                 "--env", "SENTINEL_LEAK_PROBE=1", "--types", str(tmp_path / "none.csv")])
    assert dict(os.environ) == before
