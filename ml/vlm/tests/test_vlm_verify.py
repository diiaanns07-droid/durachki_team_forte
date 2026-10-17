import json

import pytest

from ml.vlm._contract import Event
from ml.vlm.backends.mock import MockBackend
from ml.vlm.verifier import verify_events


def _ev(tid="V05", t=5.0, conf=0.7, source="rules", facts=None):
    return Event(tid, t - 1.0, t + 1.0, t, conf, source, list(facts or []), {"cell_phone": (0.3, 0.2, 0.4, 0.3)})


def _ans(visible=True, tid="V05", conf=0.9, facts=None, best=3, vis_frames=(2, 3)):
    return json.dumps({"visible": visible, "type_id": tid, "observed_facts": list(facts or []), "confidence": conf,
                       "best_frame": best, "visible_frames": list(vis_frames), "uncertain_because": ""},
                      ensure_ascii=False)


def test_one_to_one_order_source_and_inputs_untouched(clip, cfg, clock):
    evs = [_ev("V05", 3.0, 0.7), _ev("V01", 8.0, 0.4, "temporal")]
    be = MockBackend(mode="confirm", latency_s=0.5, clock=clock)
    out = verify_events(clip, evs, 30.0, cfg=cfg, backend=be, clock=clock)
    assert [e.type_id for e in out] == ["V05", "V01"] and [e.source for e in out] == ["rules", "temporal"]
    assert all(e.extra["vlm"]["status"] == "ok" for e in out)
    assert evs[0].confidence == 0.7 and "vlm" not in evs[0].extra  # входные события не мутируются
    # кадры: 4–6 штук, длинная сторона <= 448
    assert all(4 <= c["n_images"] <= 6 for c in be.calls)
    assert all(max(s) <= 448 for c in be.calls for s in c["sizes"])


def test_supports_raises_confidence_and_adds_filtered_facts(clip, cfg, clock):
    be = MockBackend(responses=[_ans(True, "V05", 0.9, ["Кассир держит у уха мобильный телефон",
                                                         "Кассир звонит, чтобы скрыть деньги",
                                                         "Сумма 5000 тенге"])], clock=clock)
    out = verify_events(clip, [_ev(conf=0.5, facts=["Телефон у уха"])], 30.0, cfg=cfg, backend=be, clock=clock)[0]
    v = out.extra["vlm"]
    assert v["agreement"] == "supports" and not v["contradiction"]
    assert 0.5 < out.confidence <= 0.75 + 1e-9  # max_raise = 0.25
    assert out.facts == ["Телефон у уха", "Кассир держит у уха мобильный телефон"]
    assert sorted(v["dropped_facts"]) == ["amount", "intent"]
    assert v["t_best"] == v["frame_times"][2] and len(v["t_visible"]) == 2


def test_strong_rule_contradicted_is_lowered_and_flagged_not_floored(clip, cfg, clock):
    be = MockBackend(responses=[_ans(False, "V05", 0.95, [])], clock=clock)
    out = verify_events(clip, [_ev(conf=0.9)], 30.0, cfg=cfg, backend=be, clock=clock)[0]
    v = out.extra["vlm"]
    assert v["agreement"] == "contradicts" and v["contradiction"] is True
    assert out.confidence < 0.9 * 0.7  # никакого «пола» 0.7*p0 для сильных правил
    assert out.source == "rules" and out.type_id == "V05"  # событие не удалено, тип/источник сохранены


def test_uncertain_answer_keeps_confidence(clip, cfg, clock):
    be = MockBackend(mode="neutral", clock=clock)
    out = verify_events(clip, [_ev(conf=0.66)], 30.0, cfg=cfg, backend=be, clock=clock)[0]
    assert out.extra["vlm"]["agreement"] == "uncertain" and out.confidence == pytest.approx(0.66, abs=1e-3)


def test_type_mismatch_retype_only_for_weak_sources(clip, cfg, clock):
    ans = _ans(True, "V06", 0.9, ["Кассир пьёт из чашки у стола с наличными"])
    rules = verify_events(clip, [_ev("V05", conf=0.6)], 30.0, cfg=cfg, backend=MockBackend(responses=[ans], clock=clock),
                          clock=clock)[0]
    assert rules.type_id == "V05" and rules.extra["vlm"]["agreement"] == "type_mismatch"
    assert rules.extra["vlm"]["suggested_type_id"] == "V06" and rules.confidence == 0.6
    temporal = verify_events(clip, [_ev("V05", conf=0.6, source="temporal")], 30.0, cfg=cfg,
                             backend=MockBackend(responses=[ans], clock=clock), clock=clock)[0]
    assert temporal.type_id == "V06" and temporal.extra["vlm_original_type_id"] == "V05"
    assert temporal.extra["vlm"]["retyped"] and temporal.confidence > 0.6


def test_invalid_fields_flagged_and_template_fact_used(clip, cfg, clock):
    raw = json.dumps({"visible": True, "type_id": "V77", "observed_facts": ["Nice image"], "confidence": 0.9,
                      "best_frame": 42, "visible_frames": [0, 1, 99], "uncertain_because": ""})
    out = verify_events(clip, [_ev(conf=0.5)], 30.0, cfg=cfg, backend=MockBackend(responses=[raw], clock=clock),
                        clock=clock)[0]
    v = out.extra["vlm"]
    assert {"type_id_invalid", "best_frame_out_of_range", "visible_frames_out_of_range"} <= set(v["flags"])
    assert v["t_best"] is None and len(v["t_visible"]) == 1
    assert v["agreement"] == "supports"  # неизвестный тип -> считается ответом про кандидата
    assert any(f.startswith("Локальная VLM-модель отметила") for f in out.facts)


def test_budget_stops_calls_and_marks_rest(clip, cfg, clock):
    evs = [_ev("V05", t, 0.5) for t in (2.0, 5.0, 8.0, 11.0)]
    be = MockBackend(mode="confirm", latency_s=2.0, clock=clock)
    out = verify_events(clip, evs, 5.0, cfg=cfg, backend=be, clock=clock, limits={"est_call_s": 2.0, "safety": 1.0})
    statuses = [e.extra["vlm"]["status"] for e in out]
    assert statuses.count("ok") == 2 and statuses.count("skipped_budget") == 2
    assert clock() <= 5.0 + 1e-9
    assert all(e.extra["vlm"]["budget"]["stop_reason"] == "time" for e in out)


def test_max_calls_limit(clip, cfg, clock):
    evs = [_ev("V05", t, 0.5) for t in (2.0, 5.0, 8.0)]
    be = MockBackend(mode="confirm", clock=clock)
    out = verify_events(clip, evs, 100.0, cfg=cfg, backend=be, clock=clock, limits={"max_calls": 1})
    assert len(be.calls) == 1
    assert sum(e.extra["vlm"]["status"] == "ok" for e in out) == 1


def test_duplicates_verified_once(clip, cfg, clock):
    evs = [_ev("V05", 5.0, 0.6), _ev("V05", 5.2, 0.5, "temporal")]
    be = MockBackend(mode="confirm", clock=clock)
    out = verify_events(clip, evs, 30.0, cfg=cfg, backend=be, clock=clock)
    assert len(be.calls) == 1
    assert sorted(e.extra["vlm"]["shared"] for e in out) == [False, True]  # второй берёт результат первого


def test_garbage_twice_leaves_event_unchanged(clip, cfg, clock):
    be = MockBackend(mode="garbage", clock=clock)
    out = verify_events(clip, [_ev(conf=0.7)], 30.0, cfg=cfg, backend=be, clock=clock)[0]
    assert out.extra["vlm"]["status"] == "parse_failed" and out.confidence == 0.7
    assert len(be.calls) == 2  # 1 ретрай


def test_backend_unavailable_is_graceful(clip, cfg):
    cfg["backend"] = "off"
    out = verify_events(clip, [_ev(conf=0.7)], 30.0, cfg=cfg)
    assert out[0].extra["vlm"]["status"] == "unavailable" and out[0].confidence == 0.7


def test_no_raw_text_in_extra_by_default(clip, cfg, clock):
    be = MockBackend(responses=[_ans(True, "V05", 0.9, ["Кассир держит телефон"])], clock=clock)
    out = verify_events(clip, [_ev()], 30.0, cfg=cfg, backend=be, clock=clock)[0]
    assert "raw" not in out.extra["vlm"]
    assert all(isinstance(c, str) for c in out.extra["vlm"]["dropped_facts"])


def test_empty_events(clip, cfg):
    assert verify_events(clip, [], 10.0, cfg=cfg) == []


def test_retyped_event_does_not_share_result_with_other_type(clip, cfg, clock):
    ans_v06 = _ans(True, "V06", 0.9, ["Кассир пьёт из чашки у стола"])
    evs = [_ev("V05", 5.0, 0.5, "temporal"), _ev("V06", 5.1, 0.45, "rules")]
    be = MockBackend(responses=[ans_v06, ans_v06], clock=clock)
    out = verify_events(clip, evs, 30.0, cfg=cfg, backend=be, clock=clock)
    assert out[0].type_id == "V06" and out[0].extra["vlm"]["retyped"]
    assert len(be.calls) == 2  # V06-событие проверено своим вызовом, а не результатом про V05
    assert out[1].extra["vlm"]["shared"] is False and out[1].extra["vlm"]["agreement"] == "supports"


def test_nan_confidence_is_neutral_prior(clip, cfg, clock):
    be = MockBackend(mode="neutral", clock=clock)
    out = verify_events(clip, [_ev(conf=float("nan"))], 30.0, cfg=cfg, backend=be, clock=clock)[0]
    assert out.confidence == 0.5


def test_fusion_compatible_opinion(clip, cfg, clock):
    """ml.fusion (A12) читает extra.vlm.verdict/score; 'status' не должен выглядеть как вердикт."""
    sup = verify_events(clip, [_ev(conf=0.5)], 30.0, cfg=cfg,
                        backend=MockBackend(responses=[_ans(True, "V05", 0.9, [])], clock=clock), clock=clock)[0]
    assert sup.extra["vlm"]["verdict"] == "confirm" and sup.extra["vlm"]["score"] == pytest.approx(0.9)
    rej = verify_events(clip, [_ev(conf=0.9)], 30.0, cfg=cfg,
                        backend=MockBackend(responses=[_ans(False, "V05", 0.8, [])], clock=clock), clock=clock)[0]
    assert rej.extra["vlm"]["verdict"] == "reject" and rej.extra["vlm"]["score"] == pytest.approx(0.8)
    assert "vlm_verdict" not in rej.extra  # A03-fallback не выбрасывает событие без явного флага
    unk = verify_events(clip, [_ev(conf=0.6)], 30.0, cfg=cfg, backend=MockBackend(mode="neutral", clock=clock),
                        clock=clock)[0]
    assert unk.extra["vlm"]["verdict"] == "unsure"
    cfg["backend"] = "off"
    off = verify_events(clip, [_ev(conf=0.6)], 30.0, cfg=cfg)[0]
    assert "verdict" not in off.extra["vlm"] and off.extra["vlm"]["coverage"]["complete"] is False


def test_coverage_summary(clip, cfg, clock):
    evs = [_ev("V05", t, 0.5) for t in (2.0, 5.0, 8.0)]
    out = verify_events(clip, evs, 100.0, cfg=cfg, backend=MockBackend(mode="confirm", clock=clock), clock=clock,
                        limits={"max_calls": 2})
    cov = out[0].extra["vlm"]["coverage"]
    assert cov == {"checked": 2, "total": 3, "complete": False, "statuses": {"ok": 2, "skipped_budget": 1}}


def test_emit_rejected_flag(clip, cfg, clock):
    cfg["verify"]["emit_vlm_verdict_rejected"] = True
    rej = verify_events(clip, [_ev(conf=0.9)], 30.0, cfg=cfg,
                        backend=MockBackend(responses=[_ans(False, "V05", 0.8, [])], clock=clock), clock=clock)[0]
    assert rej.extra["vlm_verdict"] == "rejected"


def test_truncated_answer_is_never_a_confirmation(clip, cfg, clock):
    """Обрезанный по max_new_tokens ответ (finish=length) — не подтверждение, даже если JSON «чинится»."""
    from ml.vlm.backends.base import GenResult

    class Trunc(MockBackend):
        def generate(self, images, system, prompt, **kw):
            self.calls.append({"prompt": prompt})
            return GenResult(text='{"visible": true, "type_id": "V05", "confidence": 0.9, "observed_facts": ["Кас',
                             latency_s=1.0, completion_tokens=180, finish_reason="length")

    be = Trunc(clock=clock)
    out = verify_events(clip, [_ev(conf=0.5)], 30.0, cfg=cfg, backend=be, clock=clock)[0]
    v = out.extra["vlm"]
    assert v["status"] == "truncated" and v["verdict"] == "unsure" and v["score"] == 0.0
    assert v["agreement"] == "uncertain" and out.confidence == 0.5 and out.facts == []
    assert len(be.calls) == 2 and be.calls[1]["prompt"] != be.calls[0]["prompt"]  # 1 ретрай со строгим суффиксом
    assert v["coverage"]["complete"] is False


def test_truncated_then_complete_answer_is_used(clip, cfg, clock):
    from ml.vlm.backends.base import GenResult
    answers = [GenResult(text='{"visible": true, "type_id": "V05", "confid', latency_s=1.0, finish_reason="length"),
               GenResult(text=_ans(True, "V05", 0.9, []), latency_s=1.0, finish_reason="eos")]

    class Seq(MockBackend):
        def generate(self, images, system, prompt, **kw):
            self.calls.append({"prompt": prompt})
            return answers.pop(0)

    out = verify_events(clip, [_ev(conf=0.5)], 30.0, cfg=cfg, backend=Seq(clock=clock), clock=clock)[0]
    assert out.extra["vlm"]["status"] == "ok" and out.extra["vlm"]["verdict"] == "confirm"


def test_regex_fallback_never_confirms_or_rejects(clip, cfg, clock):
    be = MockBackend(responses=["visible: yes, type_id: V05, confidence: 0.95 (сломанный ответ",
                                "visible: yes, confidence 0.95 и снова не JSON"], clock=clock)
    out = verify_events(clip, [_ev(conf=0.5)], 30.0, cfg=cfg, backend=be, clock=clock)[0]
    v = out.extra["vlm"]
    assert "regex_fallback" in v["flags"] and v["agreement"] == "uncertain" and v["verdict"] == "unsure"
    assert out.confidence == 0.5 and out.facts == []
