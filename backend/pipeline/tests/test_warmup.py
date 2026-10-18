"""Прогрев = готовность: fault-injection без весов и реальных видео (фейковые модули ml.*)."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time

import pytest

from backend.pipeline import mocks
from backend.pipeline.contract import SceneState
from backend.pipeline.orchestrator import WarmupError, warmup
from backend.pipeline.registry import ModelRegistry
from backend.pipeline.tests.conftest import ROOT
from backend.pipeline.worker import get_executor

ALL_ML = ("ml.ingest", "ml.detect", "ml.track", "ml.pose", "ml.scene", "ml.rules", "ml.temporal", "ml.vlm",
          "ml.fusion")


def _no_weights():
    raise FileNotFoundError("weights/hf/hub/models--PekingU--rtdetr_v2_r18vd: нет весов (офлайн)")


def _expect_failure(needle: str, deadline: float = 60.0) -> dict:
    with pytest.raises(WarmupError) as ei:
        warmup(deadline)
    reg = ModelRegistry.instance()
    assert reg.warm is False and reg.warmup_error and needle in str(ei.value), str(ei.value)
    assert ei.value.report["ok"] is False and reg.warmup_report is not None
    return ei.value.report


def test_warmup_success_reports_policy_explicitly(fake_ml):
    fake_ml(ingest=True)
    warmup(60.0)
    reg = ModelRegistry.instance()
    rep = reg.warmup_report
    assert reg.warm is True and reg.warmup_error is None and rep["ok"] is True and not rep["failures"]
    for st in rep["required_stages"]:
        assert rep["stages"][st]["status"] in ("ok", "cached"), st
        assert rep["stages"][st]["origin"] in ("module", "ml.ingest"), st   # не builtin-mock и не fallback
    assert rep["stages"]["vlm_verify"]["policy"] == "disabled"     # warmup.vlm=false — явно, не «успех»
    assert rep["stages"]["refine"]["policy"] == "disabled"
    assert rep["hooks"]["ml.detect"]["status"] == "no_hook"
    assert rep["seconds"] <= rep["deadline_s"]


def test_warmup_missing_required_weights(fake_ml):
    mods = fake_ml(ingest=True)
    mods["ml.detect"].warmup = _no_weights
    rep = _expect_failure("нет весов")
    assert "ml.detect" in rep["failures"][0] and "synthetic" not in rep     # синтетику не жгём


def test_warmup_missing_required_module(fake_ml):
    fake_ml(ingest=True, drop=("ml.scene",))
    rep = _expect_failure("ml.scene: missing")
    assert rep["hooks"]["ml.scene"]["status"] == "missing"


def test_warmup_optional_hook_failure_policy(fake_ml, tmp_path, monkeypatch):
    mods = fake_ml(ingest=True)
    mods["ml.temporal"].warmup = lambda: (_ for _ in ()).throw(RuntimeError("temporal: нет чекпойнта"))
    warmup(60.0)                                                   # политика warn: готов, но с деградацией
    rep = ModelRegistry.instance().warmup_report
    assert ModelRegistry.instance().warm is True and any("ml.temporal" in w for w in rep["warnings"])
    cfg = tmp_path / "strict.yaml"
    cfg.write_text("warmup:\n  optional_policy: fail\n", encoding="utf-8")
    monkeypatch.setenv("SENTINEL_PIPELINE_CONFIG", str(cfg))
    _expect_failure("ml.temporal")                                 # политика fail: не готов


def test_warmup_synthetic_failed_without_exception(fake_ml):
    def half_scene(clip, dets, tracks, zones, pose_states):    # покрыта половина видео → run() = failed
        return [SceneState(t, {"n_persons": 1.0}) for t in clip.times[: len(clip.times) // 2]]

    fake_ml({"ml.scene.scene_states": half_scene, "ml.temporal.score": lambda clip: ([], [])}, ingest=True)
    rep = _expect_failure("status=failed")
    assert rep["synthetic"]["status"] == "failed"


def test_warmup_degraded_synthetic_is_failure(fake_ml):
    """Сценарий аудита: обязательная стадия падает, но правила (мок) находят «нарушение» →
    прогон done/violation degraded. Это НЕ готовность."""
    def detect_fails(clip, every_n=1):
        raise RuntimeError("CUDA error: detector not loaded")

    fake_ml({"ml.detect.detect": detect_fails}, ingest=True)
    rep = _expect_failure("detect: error")
    assert rep["synthetic"]["degraded"] is True


def test_warmup_lease_not_acquired(fake_ml):
    fake_ml(ingest=True)
    ex = get_executor({"executor": {"mode": "inline"}})
    ex._lease.acquire()                                            # воркер занят другой задачей
    try:
        t0 = time.perf_counter()
        _expect_failure("аренда не получена", deadline=1.0)
        assert time.perf_counter() - t0 < 3.0
    finally:
        ex._lease.release()


def test_warmup_deadline_expired_in_hook(fake_ml):
    mods = fake_ml(ingest=True)
    gate = threading.Event()
    mods["ml.detect"].warmup = lambda: gate.wait(30)              # «загрузка весов» висит
    t0 = time.perf_counter()
    try:
        _expect_failure("таймаут прогрева", deadline=1.5)
        assert time.perf_counter() - t0 < 4.0, "дедлайн прогрева распространяется на RPC"
    finally:
        gate.set()


def test_warmup_explicit_mock_mode(fake_ml, monkeypatch):
    fake_ml(drop=ALL_ML)                                           # модулей нет вовсе
    monkeypatch.setenv("SENTINEL_MOCK", "1")                       # mock включён ЯВНО
    warmup(60.0)
    rep = ModelRegistry.instance().warmup_report
    assert ModelRegistry.instance().warm is True and rep["mock"] is True
    assert rep["stages"]["detect"]["origin"] == "builtin-mock"


def test_warmup_without_mock_and_without_modules_fails(fake_ml):
    fake_ml(drop=ALL_ML)
    rep = _expect_failure("missing")
    assert all(rep["hooks"][m]["status"] == "missing" for m in ("ml.detect", "ml.fusion"))


@pytest.mark.slow
def test_cli_warmup_exit_code_on_deadline(tmp_path):
    import os

    env = {**os.environ, "SENTINEL_EXECUTOR": "process", "SENTINEL_CACHE_DIR": str(tmp_path / "c"),
           "SENTINEL_DATA_DIR": str(tmp_path / "d")}
    env.pop("SENTINEL_MOCK", None)
    p = subprocess.run([sys.executable, "-m", "backend.pipeline", "warmup", "--deadline", "0.05"], cwd=ROOT,
                       env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 2, p.stderr[-1500:]
    out = json.loads(p.stdout)
    assert out["ok"] is False and out["warm"] is False and out["report"]["failures"]
