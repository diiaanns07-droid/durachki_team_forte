"""Оркестратор: обязательные/дополнительные стадии, вердикты, partial, прогресс, кэш (inline-исполнитель)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from backend.pipeline import mocks, run, run_with_context
from backend.pipeline.contract import AnalysisResult, Event
from backend.pipeline.tests.conftest import FAST


def _boom(*_a, **_k):
    raise RuntimeError("CUDA out of memory (тест)")


def _temporal_with_event(clip):
    risk, _ = mocks.score(clip)
    return risk, [Event("V03", 1.0, 2.0, 1.5, 0.9, "temporal", ["сейф открыт (тест)"])]


def test_happy_path_done(fake_ml, synth_video):
    fake_ml()
    res = run(synth_video, "job-ok", None, 60)
    assert isinstance(res, AnalysisResult)
    assert res.status == "done" and res.verdict == "violation"
    assert res.meta["complete"] is True
    v = res.violations[0]
    assert v.evidence_frames and v.evidence_frames[0].image_base64
    assert set(res.meta["timings"]) >= {"ingest", "detect", "track", "scene", "rules", "fusion", "total"}
    assert res.meta["pipeline_version"] and res.meta["sample_fps"] > 0 and res.meta["device"] == "cpu"
    assert res.timeline, "риск-кривая для UI"


def test_required_stage_failure_never_no_violation(fake_ml, synth_video):
    fake_ml({"ml.detect.detect": _boom, "ml.temporal.score": lambda clip: ([], []),
             "ml.rules.run_rules": lambda *a: []})
    res = run(synth_video, "job-fail", None, 60)
    assert res.status == "failed" and res.verdict is None
    assert "detect" in res.meta["error"]
    assert res.meta["stages"]["detect"]["status"] == "error"
    assert res.meta["complete"] is False


def test_required_failure_but_violation_found_is_degraded(fake_ml, synth_video):
    fake_ml({"ml.detect.detect": _boom, "ml.temporal.score": _temporal_with_event})
    res = run(synth_video, "job-degraded", None, 60)
    assert res.status == "done" and res.verdict == "violation"
    assert res.meta["degraded"] is True and res.meta["complete"] is False
    assert any(v.type_id == "V03" for v in res.violations)


def test_on_required_failure_fail_policy(fake_ml, synth_video):
    fake_ml({"ml.detect.detect": _boom, "ml.temporal.score": _temporal_with_event})
    res = run(synth_video, "job-strict", None, 60, overrides={"on_required_failure": "fail"})
    assert res.status == "failed" and res.verdict is None


def test_missing_required_module_is_failure_not_empty_success(fake_ml, synth_video):
    fake_ml({"ml.rules.run_rules": lambda *a: [], "ml.temporal.score": lambda clip: ([], [])}, drop=("ml.scene",))
    res = run(synth_video, "job-missing", None, 60)
    assert res.meta["stages"]["scene"]["status"] == "missing"
    assert res.status == "failed" and res.verdict is None


def test_optional_failure_is_warning_only(fake_ml, synth_video):
    fake_ml({"ml.vlm.verify": _boom, "ml.vlm.discover": _boom, "ml.pose.estimate": _boom})
    res = run(synth_video, "job-opt", None, 60)
    assert res.status == "done" and res.meta["complete"] is True
    joined = " ".join(res.meta["warnings"])
    assert "vlm_verify" in joined and "pose" in joined


def test_clean_video_no_violation(fake_ml, synth_video):
    fake_ml({"ml.rules.run_rules": lambda *a: [], "ml.temporal.score": lambda clip: ([(0.0, 0.1)], []),
             "ml.vlm.verify": lambda clip, ev, b: []})
    res = run(synth_video, "job-clean", None, 60)
    assert res.status == "done" and res.verdict == "no_violation" and not res.violations


def test_disabled_optional_stages_warn(fake_ml, synth_video):
    fake_ml()
    res = run(synth_video, "job-novlm", None, 60,
              overrides={"stages": {"vlm_verify": False, "vlm_discover": False, "temporal": False}})
    assert res.status == "done"
    assert res.meta["stages"]["vlm_verify"]["status"] == "disabled"
    assert any("выключена" in w for w in res.meta["warnings"])


def test_disabled_required_stage_blocks_no_violation(fake_ml, synth_video):
    fake_ml({"ml.temporal.score": lambda clip: ([], []), "ml.rules.run_rules": lambda *a: []})
    res = run(synth_video, "job-nodet", None, 60, overrides={"stages": {"detect": False}})
    assert res.status == "failed" and res.verdict is None


def test_stage_timeout_inline(fake_ml, synth_video):
    def slow(clip, every_n=1):
        time.sleep(3.0)
        return []

    fake_ml({"ml.detect.detect": slow, "ml.temporal.score": lambda clip: ([], []),
             "ml.rules.run_rules": lambda *a: []})
    t0 = time.perf_counter()
    res = run(synth_video, "job-timeout", None, 30, overrides=FAST)
    assert time.perf_counter() - t0 < 3.0, "оркестратор не ждёт зависшую стадию"
    assert res.meta["stages"]["detect"]["status"] == "timeout"
    assert res.status == "failed"
    assert res.meta["executor"]["poisoned"] is True


def test_deadline_skips_optional_and_respects_deadline(fake_ml, synth_video):
    fake_ml()
    res = run(synth_video, "job-tight", None, 1.0,
              overrides={"reserve_s": {"fusion": 0.5}, "min_start_fraction": 0.0, "budgets_s": {"vlm_verify": 50}})
    # VLM не влезает в остаток — пропускается с warning; базовые стадии успевают
    assert res.meta["stages"]["vlm_discover"]["status"] in ("skipped", "ok")
    assert res.meta["timings"]["total"] < 5


def test_deadline_at_counts_queue_time(fake_ml, synth_video):
    fake_ml({"ml.rules.run_rules": lambda *a: [], "ml.temporal.score": lambda clip: ([], [])})
    res = run(synth_video, "job-late", None, 240, deadline_at=time.time() - 1)  # SLA уже истёк в очереди
    assert res.status == "failed" and res.verdict is None
    assert res.meta["budget"]["deadline_s"] < 0


def test_partial_never_done_or_no_violation(fake_ml, synth_video, tmp_path):
    fake_ml({"ml.rules.run_rules": lambda *a: [], "ml.temporal.score": lambda clip: ([], [])})
    res = run(synth_video, "job/partial 1", None, 60)
    assert res.meta["partial_written"] == ["after_core"]
    p = Path(tmp_path / "data" / "jobs" / "job_partial_1" / "partial.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["status"] == "processing" and data["verdict"] is None and data["partial_verdict"] is None
    assert data["meta"]["partial"] is True and data["meta"]["complete"] is False


def test_partial_with_violation(fake_ml, synth_video, tmp_path):
    fake_ml({"ml.temporal.score": _temporal_with_event})
    run(synth_video, "jp2", None, 60)
    data = json.loads((tmp_path / "data" / "jobs" / "jp2" / "partial.json").read_text(encoding="utf-8"))
    assert data["status"] == "processing" and data["verdict"] is None      # промежуточный — без вердикта
    assert data["partial_verdict"] == "violation" and data["partial_violations"]


def test_progress_monotonic_russian(fake_ml, synth_video):
    fake_ml()
    seen: list[tuple[str, float]] = []
    run(synth_video, "job-progress", lambda s, f: seen.append((s, f)), 60)
    fracs = [f for _, f in seen]
    assert fracs == sorted(fracs) and fracs[-1] == 1.0
    labels = {s for s, _ in seen}
    assert {"Декодирование", "Поиск людей и объектов", "Трекинг", "Состояние сцены", "Правила",
            "Временная модель", "Проверка VLM", "Формирование отчёта"} <= labels


def test_progress_callback_errors_ignored(fake_ml, synth_video):
    fake_ml()
    res = run(synth_video, "job-cb", lambda s, f: 1 / 0, 60)
    assert res.status == "done"


def test_contract_violation_by_module_is_stage_error(fake_ml, synth_video):
    fake_ml({"ml.track.track": lambda clip, dets: "не кортеж"})
    res = run(synth_video, "job-badtype", None, 60)
    assert res.meta["stages"]["track"]["status"] == "error"
    assert "контракт" in res.meta["stages"]["track"]["error"]


def test_missing_file_failed():
    res = run("/nonexistent/video.mov", "job-nofile", None, 10)
    assert res.status == "failed" and "не найден" in res.meta["error"]


def test_fusion_module_used_and_fallback_on_error(fake_ml, synth_video):
    calls = {}

    def fuse(clip, events, risk, catalog, job_id):
        calls["n"] = len(events)
        return AnalysisResult(job_id=job_id, status="done", verdict="no_violation", meta={"warnings": ["from-fusion"]})

    fake_ml({"ml.fusion.fuse": fuse, "ml.rules.run_rules": lambda *a: []})
    res = run(synth_video, "job-fuse", None, 60)
    assert "n" in calls and res.meta["stages"]["fusion"]["origin"] == "module"
    assert "from-fusion" in res.meta["warnings"]

    fake_ml({"ml.fusion.fuse": _boom})          # сбой обязательной стадии fusion; правила нашли нарушение
    res = run(synth_video, "job-fuse2", None, 60)
    assert res.meta["stages"]["fusion"]["status"] == "error" and res.meta["stages"]["fusion"]["origin"] == "fallback"
    assert res.status == "done" and res.verdict == "violation" and res.meta["degraded"] is True
    assert res.meta["complete"] is False


def test_run_with_context_outputs(fake_ml, synth_video, tmp_path):
    from backend.pipeline.orchestrator import dump_intermediate

    fake_ml()
    res, ctx = run_with_context(synth_video, "job-ctx", None, 60)
    files = dump_intermediate(ctx, tmp_path / "dump")
    names = {Path(f).name for f in files}
    assert {"dets.jsonl", "tracks.json", "zones.json", "states.jsonl", "events_rules.json", "risk.json",
            "events_all.json", "stages.json"} <= names


def test_cold_flag_queue_wait_and_vlm_usage(fake_ml, synth_video):
    def verify(clip, events, budget_s):
        out = []
        for e in events:
            out.append(Event(e.type_id, e.t_start, e.t_end, e.t_peak, e.confidence, "vlm", [], {}, [],
                             {"vlm": {"status": "checked", "budget": {"calls": 1, "output_tokens": 37}}}))
        return out

    fake_ml({"ml.vlm.verify": verify})
    r1 = run(synth_video, "job-cold-1", None, 60, overrides={"cache": {"mode": "off"}})
    r2 = run(synth_video, "job-cold-2", None, 60, overrides={"cache": {"mode": "off"}})
    assert r1.meta["stages"]["detect"].get("cold") is True          # первый вызов — с загрузкой модели
    assert not r2.meta["stages"]["detect"].get("cold")              # модель уже тёплая
    assert r1.meta["queue_wait_s"] >= 0
    assert r1.meta["vlm"]["vlm_verify"]["budget"]["output_tokens"] == 37
    assert r1.meta["vlm"]["vlm_verify"]["statuses"] == {"checked": 1}


def test_mock_config_does_not_leak_into_process_env(fake_ml, synth_video):
    """mock одной задачи не должен переключать процесс API (и следующие задачи) в mock."""
    import os

    fake_ml()
    run(synth_video, "job-mock-cfg", None, 60, overrides={"mock": True})
    assert "SENTINEL_MOCK" not in os.environ
