"""_finalize: сбой fusion, неполный ingest и неизвестное покрытие никогда не дают done/no_violation;
чистый успех несёт канонический meta.coverage, который принимает сериализатор A01."""
from __future__ import annotations

import pytest

from backend.pipeline import run
from backend.pipeline.contract import AnalysisResult
from backend.pipeline.media import SimpleClip

QUIET = {"ml.rules.run_rules": lambda *a: [], "ml.temporal.score": lambda clip: ([], [])}  # без находок


def _boom(*_a, **_k):
    raise RuntimeError("fusion crashed")


def _assert_not_clean(res):
    assert res.status == "failed" and res.verdict is None, (res.status, res.verdict)
    assert res.meta["complete"] is False and res.meta["coverage"]["complete"] is False


def test_fusion_exception_without_findings_is_failed(fake_ml, synth_video):
    fake_ml({**QUIET, "ml.fusion.fuse": _boom}, ingest=True)
    res = run(synth_video, "fin-fusion-exc", None, 60)
    _assert_not_clean(res)
    assert res.meta["stages"]["fusion"]["status"] == "error" and "fusion" in res.meta["error"]


def test_fusion_reporting_failed_is_failed(fake_ml, synth_video):
    def fuse_failed(clip, events, risk, catalog, job_id):
        return AnalysisResult(job_id=job_id, status="failed", verdict=None, meta={"error": "evidence encoder down"})

    fake_ml({**QUIET, "ml.fusion.fuse": fuse_failed}, ingest=True)
    res = run(synth_video, "fin-fusion-failed", None, 60)
    _assert_not_clean(res)
    assert "status=failed" in res.meta["stages"]["fusion"]["error"]


def test_fusion_missing_module_is_failed(fake_ml, synth_video):
    fake_ml(QUIET, ingest=True, drop=("ml.fusion",))
    res = run(synth_video, "fin-fusion-missing", None, 60)
    _assert_not_clean(res)
    assert res.meta["stages"]["fusion"]["status"] == "missing"


def test_incomplete_ingest_is_failed(fake_ml, synth_video):
    def half_clip(path, sample_fps=5.0):            # декодирована только первая половина, длительность — полная
        c = SimpleClip(path, sample_fps)
        n = len(c.times) // 2
        c.times, c._jpegs = c.times[:n], c._jpegs[:n]
        return c

    fake_ml({**QUIET, "ml.ingest.load_video": half_clip}, ingest=True)
    res = run(synth_video, "fin-ingest-half", None, 60)
    _assert_not_clean(res)
    assert res.meta["coverage"]["ratio"] < 0.9 and res.meta["coverage"]["gaps"]
    assert "покрытие" in res.meta["error"]


def test_unknown_coverage_is_failed(fake_ml, synth_video, monkeypatch):
    import backend.pipeline.orchestrator as orch

    def no_duration_clip(path, sample_fps=5.0):     # длительность неизвестна ни клипу, ни контейнеру
        c = SimpleClip(path, sample_fps)
        c.duration = 0.0
        return c

    monkeypatch.setattr(orch, "probe_video", lambda path: {})
    fake_ml({**QUIET, "ml.ingest.load_video": no_duration_clip}, ingest=True)
    res = run(synth_video, "fin-unknown-cov", None, 60)
    _assert_not_clean(res)
    assert res.meta["coverage"]["ratio"] == 0.0


def test_clean_success_has_canonical_coverage_accepted_by_a01(fake_ml, synth_video):
    fake_ml(QUIET, ingest=True)
    res = run(synth_video, "fin-clean", None, 60)
    assert res.status == "done" and res.verdict == "no_violation" and res.meta["complete"] is True
    cov = res.meta["coverage"]
    assert {"ratio", "analyzed_sec", "duration_sec", "unknown_spans", "stages", "required", "complete"} <= set(cov)
    assert cov["ratio"] >= 0.98 and cov["complete"] is True and cov["unknown_spans"] == [] and cov["gaps"] == []
    assert set(cov["required"]) == {"ingest", "detect", "track", "scene", "rules", "fusion"}
    assert all(cov["stages"][s] in ("ok", "cached") for s in cov["required"])
    ext = pytest.importorskip("contracts.extensions", reason="канон A01 (contracts/) не в этой ветке — проверяет L2")
    assert ext.coverage_problem(res.meta, 0.98, allow_mock=False) is None


def test_jury_serializer_a01_accepts_clean_and_rejects_fusion_failure(fake_ml, synth_video):
    ser = pytest.importorskip("contracts.serialize", reason="канон A01 (contracts/) не в этой ветке — проверяет L2")
    fake_ml(QUIET, ingest=True)
    clean = ser.result_to_jury_api(run(synth_video, "fin-jury-clean", None, 60))
    assert clean["status"] == "done" and clean["verdict"] == "no_violation"
    fake_ml({**QUIET, "ml.fusion.fuse": _boom}, ingest=True)
    broken = ser.result_to_jury_api(run(synth_video, "fin-jury-broken", None, 60))
    assert broken["status"] == "failed" and broken["verdict"] is None
