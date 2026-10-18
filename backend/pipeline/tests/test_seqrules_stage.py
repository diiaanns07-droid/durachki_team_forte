"""Стадия seqrules в оркестраторе: выключена по умолчанию, наблюдения для UI, неполный анализ ≠ «нет нарушения»."""
from __future__ import annotations

import json
import os
from pathlib import Path

from backend.pipeline import serde
from backend.pipeline.cache import STAGE_FILES, StageCache
from backend.pipeline.contract import Event
from backend.pipeline.orchestrator import _normalize, run

REQ = ["ingest", "detect", "track", "pose", "seqrules", "fusion"]
ON = {"stages": {"seqrules": True}, "required_stages": REQ}


def _obs_path(job: str) -> Path:
    return Path(os.environ["SENTINEL_DATA_DIR"]) / "jobs" / job / "observations.json"


def test_seqrules_is_disabled_by_default(fake_ml, synth_video):
    fake_ml()
    res = run(synth_video, "job-default", None, 60)
    assert res.meta["stages"]["seqrules"]["status"] == "disabled"
    assert not _obs_path("job-default").exists()


def test_real_seqrules_over_mock_cv_writes_observations(fake_ml, synth_video):
    from ml.seqrules import run as seq_run     # настоящий модуль поверх фейковых detect/track/pose

    fake_ml({"ml.seqrules.run": seq_run})
    res = run(synth_video, "job-seq", None, 60, overrides=ON)
    assert res.meta["stages"]["seqrules"]["status"] == "ok", res.meta["stages"]["seqrules"]
    assert res.status == "done" and res.meta["complete"] is True
    data = json.loads(_obs_path("job-seq").read_text(encoding="utf-8"))
    assert data["job_id"] == "job-seq" and data["items"][0]["kind"] == "meta"
    assert set(data["items"][0]["rules"]) == {"phone", "safe", "storage", "entry", "hand_cash"}
    assert data["items"][0]["rules"]["storage"]["status"] == "disabled"     # экспериментальное — только по профилю


def test_rule_error_makes_analysis_incomplete_not_no_violation(fake_ml, synth_video):
    def broken(clip, dets, tracks, zones, poses, catalog):
        meta = {"kind": "meta", "rules": {"phone": {"status": "error", "error": "сбой"}},
                "warnings": ["seqrules/phone: ошибка сбой"]}
        return [], [meta]

    fake_ml({"ml.seqrules.run": broken, "ml.rules.run_rules": lambda *a: [],
             "ml.temporal.score": lambda clip: ([], [])})
    res = run(synth_video, "job-seq-err", None, 60, overrides=ON)
    assert res.meta["complete"] is False
    assert res.verdict != "no_violation"
    assert any("seqrules/phone" in w for w in res.meta["warnings"])


def test_stage_crash_when_required_is_failure(fake_ml, synth_video):
    def crash(*a, **k):
        raise RuntimeError("seqrules упал")

    fake_ml({"ml.seqrules.run": crash, "ml.rules.run_rules": lambda *a: [], "ml.temporal.score": lambda clip: ([], [])})
    res = run(synth_video, "job-seq-crash", None, 60, overrides=ON)
    assert res.meta["stages"]["seqrules"]["status"] == "error"
    assert res.status == "failed" and res.verdict is None


def test_seqrules_outputs_roundtrip_through_cache(tmp_path):
    assert [k for _f, k in STAGE_FILES["seqrules"]] == ["event", "obs"]
    ev = Event(type_id="V05", t_start=1.0, t_end=4.0, t_peak=1.0, confidence=0.6, source="rules_seq",
               facts=["факт"], bboxes={"phone": (0.1, 0.1, 0.2, 0.2)}, track_ids=[1],
               extra={"rule": "A_phone", "assessment": "candidate_violation"})
    obs = [{"kind": "meta", "rules": {"phone": {"status": "ok"}}},
           {"kind": "episode", "id": "A-1-1", "frames": [{"t": 1.0, "state": "observed", "kp": [[0.1, 0.2, 0.9]]}]}]
    values = _normalize("seqrules", [(([ev], obs))], True)
    cache = StageCache(tmp_path / "c", "sha", "read_write")
    cache.save("seqrules", "k1", values, {})
    hit = cache.load("seqrules", "k1")
    assert hit is not None
    events2, obs2 = hit
    assert events2[0].type_id == "V05" and events2[0].extra["rule"] == "A_phone"
    assert obs2 == json.loads(json.dumps(serde.to_jsonable(obs)))


def test_vlm_discover_type_allowlist_filters_before_fusion(fake_ml, synth_video):
    """vlm_discover_types: из пути VLM в слияние доходят только разрешённые типы; стадия и её ключ не меняются."""
    def discover(clip, budget=None):
        mk = lambda t, typ: Event(type_id=typ, t_start=t, t_end=t + 1, t_peak=t, confidence=0.9, source="vlm",
                                  facts=["синтетика"], extra={"assessment": "candidate_violation"})
        return [mk(1.0, "V01"), mk(2.0, "V05")]

    fake_ml({"ml.vlm.discover": discover, "ml.rules.run_rules": lambda *a: [], "ml.temporal.score": lambda clip: ([], [])})
    res = run(synth_video, "job-vlm-types", None, 60,
              overrides={"stages": {"vlm_discover": True, "temporal": False}, "vlm_discover_types": ["V01"]})
    assert res.meta["stages"]["vlm_discover"]["status"] == "ok"
    assert {v.type_id for v in res.violations} <= {"V01"}
    assert any("допущены только типы V01 (1 из 2" in w for w in res.meta["warnings"])
