"""Совместимость с соседями: оценки A09 (extra.assessment), каталог A13 без норм, покрытие A04 (meta.coverage)."""
from __future__ import annotations

import numpy as np
import pytest

from ml.fusion import fuse
from ml.fusion.catalog import resolve_catalog
from ml.fusion.completeness import coverage_info
from ml.fusion.config import load_config, load_thresholds
from ml.fusion.core import fuse_scores, upstream_assessment

from .conftest import FakeClip, ev

CFG = load_config()
A13_LIKE = {"types": {
    "V01": {"name": "Посторонний в зоне кассира", "rules": [],
            "description_template": "{t}: наблюдение для проверки человеком — {facts}. Нормативное основание не "
                                    "подтверждено; по видео нельзя установить допуск и личность человека."},
    "V02": {"name": "Кассир покинул рабочее место, оставив наличные без присмотра", "rules": []},
    "V05": {"name": "Использование мобильного телефона при работе с ценностями", "rules": []},
    "V08": {"name": "Пересчёт наличных не в присутствии клиента",
            "rules": [{"document": "Д", "clause": "п. 1", "url": "https://adilet.zan.kz/rus/docs/X", "quote": "q",
                       "verified": False}]},
    "V99": {"name": "Иное видимое подозрительное действие", "rules": []},
}}
NORM_REASON = ["нет проверенной применимой нормы для V02 в каталоге A13 (статус: absent)"]
OBS_REASON = ["правило-наблюдение: само по себе не доказывает нарушение"]
TRACE = {"confidence_terms": {"base": 0.3, "strength": 0.35, "duration": 0.2, "factor": 1.0, "cap": 0.95}}


def a09(type_id="V02", t=10.0, conf=0.45, assessment="review_observation", reasons=None, **extra):
    x = {"assessment": assessment, "assessment_reasons": reasons if reasons is not None else NORM_REASON,
         "legal_claim": False, "trace": TRACE}
    x.update(extra)
    return ev(type_id, t=t, conf=conf, facts=["кассир вышел из зоны рабочего места; наличные остаются на столе"],
              extra=x)


def run(events, catalog=A13_LIKE):
    return fuse_scores(events, [], 60.0, resolve_catalog(catalog)[0], CFG, load_thresholds())


def test_parse_assessment_variants():
    assert upstream_assessment(0.6, {"assessment": "candidate_violation"}, CFG) == ("candidate", 0.6)
    assert upstream_assessment(0.3, {"assessment": "insufficient_context"}, CFG) == ("insufficient", 0.3)
    a, c = upstream_assessment(0.45, {"assessment": "review_observation", "assessment_reasons": NORM_REASON,
                                      "trace": TRACE}, CFG)
    assert a == "norm_review" and c == pytest.approx(0.85)          # восстановлено из trace A09
    a, c = upstream_assessment(0.45, {"assessment": "review_observation", "assessment_reasons": NORM_REASON,
                                      "confidence_raw": 0.7}, CFG)
    assert (a, c) == ("norm_review", 0.7)
    assert upstream_assessment(0.45, {"assessment": "review_observation", "assessment_reasons": OBS_REASON},
                               CFG) == ("observation", 0.45)
    assert upstream_assessment(0.5, {}, CFG) == ("candidate", 0.5)


def test_norm_review_reported_without_clause_on_a13_catalog():
    out = run([a09()])
    c = out.candidates[0]
    assert c.reported and c.assessment == "norm_review" and c.confidence == pytest.approx(0.85)


def test_observation_rule_is_hint():
    out = run([a09("V05", reasons=OBS_REASON)])
    assert not out.reported and out.candidates[0].gate == ["observation_only"]


def test_insufficient_context_is_hint_not_clean_no():
    out = run([a09("V02", conf=0.3, assessment="insufficient_context",
                   reasons=["начало ситуации не попало в ролик"])])
    assert not out.reported and out.candidates[0].gate == ["insufficient_context"]


def test_vlm_confirm_lifts_observation():
    vlm = ev("V05", t=10.2, conf=0.8, source="vlm", facts=["кассир держит у уха мобильный телефон"],
             extra={"vlm": {"verdict": "confirm", "score": 0.8}})
    out = run([a09("V05", reasons=OBS_REASON), vlm])
    assert out.candidates[0].assessment == "candidate" and out.reported


def test_flags_off_restore_plain_behaviour():
    cfg = load_config()
    cfg["report_gate"]["upstream_assessment"] = False
    out = fuse_scores([a09("V05", reasons=OBS_REASON)], [], 60.0, resolve_catalog(A13_LIKE)[0], cfg,
                      {"default": 0.45, "per_type": {}})
    assert out.reported and out.candidates[0].confidence == pytest.approx(0.45)


def test_coverage_complete_and_partial():
    full = FakeClip(duration=20.0)
    cov = coverage_info(full, 20.0, CFG)
    assert cov["complete"] is True and cov["sampled"] > 0.95

    class Partial(FakeClip):
        meta = {"coverage": 0.7, "complete": False}

    cov = coverage_info(Partial(duration=20.0), 20.0, CFG)
    assert cov["complete"] is False and any("декодировано" in r for r in cov["reasons"])

    gappy = FakeClip(duration=20.0)
    gappy.times = [t for t in gappy.times if not 5.0 < t < 12.0]
    cov = coverage_info(gappy, 20.0, CFG)
    assert cov["complete"] is False and cov["max_gap_s"] > 2.0


def test_partial_coverage_blocks_clean_verdict(no_frames):
    from ml.fusion import result_to_jury

    class Partial(FakeClip):
        meta = {"coverage": 0.6, "complete": False}

    res = fuse(Partial(), [], [], A13_LIKE, "j")
    # fusion не утверждает «нарушений нет»: verdict=None; status=done — failed A03 счёл бы падением fusion
    assert res.status == "done" and res.verdict is None and res.violations == []
    assert res.meta["fusion"]["coverage"]["complete"] is False and res.meta["ui"]["incomplete"]["message_ru"]
    assert "coverage" not in res.meta            # канонический meta.coverage — у A03/A01, fusion его не пишет
    jury = result_to_jury(res)
    assert jury["status"] == "failed" and jury["verdict"] is None and jury["violations"] == []

    res2 = fuse(Partial(), [a09(t=5.0)], [], A13_LIKE, "j")
    assert res2.status == "done" and res2.verdict == "violation" and res2.meta["analysis_complete"] is False


def test_unknown_coverage_is_not_clean(no_frames):
    from ml.fusion import result_to_jury

    class NoTimes:
        duration, fps, path = 20.0, 30.0, "x"
        times: list = []

        def frame(self, i):
            return np.zeros((8, 8, 3), np.uint8)

    res = fuse(NoTimes(), [], [], A13_LIKE, "j")
    assert res.status == "done" and res.verdict is None
    assert result_to_jury(res)["status"] == "failed"


def test_fusion_never_returns_failed_for_coverage(no_frames):
    # A03 (orchestrator._fusion) считает status=="failed" падением fusion и подменяет результат fallback_fuse
    class Partial(FakeClip):
        meta = {"coverage": 0.3, "complete": False}

    for events in ([], [a09(t=5.0)]):
        assert fuse(Partial(), events, [], A13_LIKE, "j").status == "done"


def test_a04_frame_at_ts_timestamp_used():
    class A04Clip(FakeClip):
        def frame_at_ts(self, t):
            idx = int(round(t * self.fps)) + 1           # «фактический» кадр на 1 позже запрошенного
            return self.coded_frame(idx), idx / self.fps

    res = fuse(A04Clip(duration=20.0), [a09(t=5.0)], [], A13_LIKE, "j")
    ts = [f.timestamp_sec for f in res.violations[0].evidence_frames]
    assert ts[0] == pytest.approx(151 / 30, abs=1e-3)


def test_zone_and_hash_box_keys():
    from ml.fusion.evidence import parse_box_key, ru_label

    assert parse_box_key("visitor#7") == ("visitor", 7)
    assert ru_label("safe_open#3") == "сейф открыт #3"
    assert ru_label("zone:safe") == "зона сейфа"
    assert ru_label("zone:cashier_area") == "зона кассира"
    assert parse_box_key("zone:safe")[0] == "zone:safe"


def test_temporal_does_not_lift_observation_rule():
    temporal = ev("V05", t=10.2, conf=0.9, source="temporal", facts=[])
    out = run([a09("V05", reasons=OBS_REASON), temporal])
    c = out.candidates[0]
    assert c.assessment == "observation" and not out.reported and "observation_only" in c.gate
