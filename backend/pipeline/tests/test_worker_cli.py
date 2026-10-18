"""Процесс-воркер (один владелец GPU, kill по таймауту) и CLI — реальный spawn, без весов."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from backend.pipeline import run
from backend.pipeline.tests.conftest import FAST, ROOT

pytestmark = pytest.mark.slow


def test_process_worker_e2e_mock(synth_video, monkeypatch):
    monkeypatch.setenv("SENTINEL_EXECUTOR", "process")
    res = run(synth_video, "job-proc", None, 120, overrides={"mock": True})
    assert res.status == "done", res.meta.get("error")
    assert res.meta["executor"]["mode"] == "process" and res.meta["executor"]["pid"]
    # без модуля ml.detect в ветке — встроенный фейк; в интеграции — mock-бэкенд самого модуля
    assert res.meta["stages"]["detect"]["origin"] in ("builtin-mock", "module")


def test_process_worker_killed_on_timeout_and_restarted(synth_video, monkeypatch, fake_ml_files):
    base = fake_ml_files()
    monkeypatch.setenv("SENTINEL_EXECUTOR", "process")
    monkeypatch.setenv("SENTINEL_FAKE_SLEEP", "30")
    over = {**FAST, "executor": {"extra_sys_path": [str(base)]}, "budgets_s": {"detect": 0.5}}
    t0 = time.perf_counter()
    res = run(synth_video, "job-kill", None, 60, overrides=over)
    assert time.perf_counter() - t0 < 25, "зависшая стадия не держит задачу"
    det = res.meta["stages"]["detect"]
    assert det["status"] == "timeout" and "убит" in det["error"]
    assert res.meta["executor"]["kills"] >= 1
    # после убийства воркер поднят заново: последующие стадии отработали на перезагруженном клипе
    assert res.meta["stages"]["track"]["status"] == "ok"
    assert res.meta["stages"]["track"]["origin"] == "module"
    assert res.status in ("failed", "done") and res.verdict != "no_violation"


def test_worker_crash_is_reported(synth_video, monkeypatch, fake_ml_files, tmp_path):
    base = fake_ml_files()
    (base / "ml" / "track" / "__init__.py").write_text(
        "import os\n__version__='x'\ndef track(clip, dets):\n    os._exit(3)\n", encoding="utf-8")
    monkeypatch.setenv("SENTINEL_EXECUTOR", "process")
    res = run(synth_video, "job-crash", None, 60, overrides={"executor": {"extra_sys_path": [str(base)]}})
    assert res.meta["stages"]["track"]["status"] == "crashed"
    assert res.verdict != "no_violation"


def _cli(*args: str, env_extra: dict | None = None, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    import os

    env = {**os.environ, **(env_extra or {})}
    return subprocess.run([sys.executable, "-m", "backend.pipeline", *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=180)


def test_cli_run_and_dump(synth_video, tmp_path):
    out = tmp_path / "out.json"
    p = _cli("run", synth_video, "--out", str(out), "--mock", "--dump-intermediate", str(tmp_path / "dump"),
             env_extra={"SENTINEL_EXECUTOR": "process"})
    assert p.returncode == 0, p.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["status"] == "done" and data["verdict"] in ("violation", "no_violation")
    for v in data["violations"]:
        assert {"type", "timecode_sec", "description", "evidence_frames"} <= set(v)
    assert (tmp_path / "dump" / "dets.jsonl").exists()
    assert "Декодирование" in p.stderr


def test_cli_run_failed_exit_code(tmp_path):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"not a video at all")                    # не декодируется ни ml.ingest, ни OpenCV
    p = _cli("run", str(bad), "--out", str(tmp_path / "o.json"), "--no-vlm", "--mock")
    assert p.returncode == 2
    assert json.loads((tmp_path / "o.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_cli_batch_normalizes_names(synth_video, tmp_path):
    import shutil

    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy(synth_video, raw / "Х-4.mp4")   # кириллица
    shutil.copy(synth_video, raw / "Х-47.mp4")  # дубликат по содержимому → кэш
    p = _cli("batch", str(raw), "--out", str(tmp_path / "preds"), "--mock", "--inline")
    assert p.returncode == 0, p.stderr
    assert (tmp_path / "preds" / "X-4.json").exists() and (tmp_path / "preds" / "X-47.json").exists()
    rows = json.loads((tmp_path / "preds" / "summary.json").read_text(encoding="utf-8"))
    assert len(rows) == 2 and rows[0]["sha256"] == rows[1]["sha256"]
    assert (tmp_path / "preds" / "summary.csv").exists()


def test_cli_profile(synth_video):
    p = _cli("profile", synth_video, "--mock", "--inline")
    assert p.returncode == 0, p.stderr
    assert "detect" in p.stdout and "ИТОГО" in p.stdout
    p = _cli("profile", synth_video, "--mock", "--inline", "--json")
    rep = json.loads(p.stdout)
    assert rep["status"] == "done" and any(r["stage"] == "detect" for r in rep["stages"])


def test_deadline_respected_when_worker_keeps_dying(synth_video, monkeypatch, fake_ml_files):
    """Регрессия пробной интеграции A01: 6-с ролик, deadline 40 с, ответ через 110 с.
    Служебные RPC (identity/device/prepare_vram) и перезапуски воркера тоже обязаны укладываться в дедлайн."""
    base = fake_ml_files()
    for name in ("track", "scene", "rules", "temporal", "vlm"):   # медленный импорт каждого модуля в воркере
        p = base / "ml" / name / "__init__.py"
        p.write_text("import time\ntime.sleep(2.0)\n" + p.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("SENTINEL_EXECUTOR", "process")
    monkeypatch.setenv("SENTINEL_FAKE_SLEEP", "60")               # detect зависает → воркер убивается
    deadline = 8.0
    t0 = time.perf_counter()
    res = run(synth_video, "job-deadline", None, deadline,
              overrides={"executor": {"extra_sys_path": [str(base)]}, "timeouts": {"min_s": 2.0}})
    elapsed = time.perf_counter() - t0
    assert elapsed < deadline + 4.0, f"run() занял {elapsed:.1f} с при дедлайне {deadline} с"
    assert res.status == "failed" and res.verdict is None
    assert res.meta["stages"]["detect"]["status"] == "timeout"


def test_process_worker_refine_windows(synth_video, monkeypatch):
    monkeypatch.setenv("SENTINEL_EXECUTOR", "process")
    res = run(synth_video, "job-proc-refine", None, 120, overrides={"mock": True, "refine": {"enabled": True}})
    assert res.status == "done", res.meta.get("error")
    st = res.meta["stages"]["refine"]["status"]
    assert st in ("ok", "skipped"), res.meta["stages"]["refine"]
    if st == "ok":
        assert res.meta["refine"]["processed"] >= 1


def test_cli_profile_repeat_cold_warm(synth_video):
    p = _cli("profile", synth_video, "--mock", "--inline", "--json", "--repeat", "2")
    assert p.returncode == 0, p.stderr
    reps = json.loads(p.stdout)
    assert [r["run"] for r in reps] == [1, 2]
    det = [next(s for s in r["stages"] if s["stage"] == "detect") for r in reps]
    assert det[0]["cold"] is True and det[1]["cold"] is False
