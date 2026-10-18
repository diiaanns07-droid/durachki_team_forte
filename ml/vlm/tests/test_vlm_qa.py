"""verify.mode=qa на mock-бэкенде: ответы задаются по тексту вопроса и времени кадра."""
import re

import numpy as np
import pytest

from ml.vlm._contract import Event, Zone
from ml.vlm.backends.mock import MockBackend
from ml.vlm.qa import decide, load_spec, transition_time
from ml.vlm.verifier import verify_events

YES, NO, UNK = {"yes": 9, "no": 1, "unknown": 0.5}, {"yes": 1, "no": 9, "unknown": 0.5}, {"yes": 0.2, "no": 0.2, "unknown": 9}


def scripted(rules, default=UNK):
    """rules: [(подстрока вопроса, функция t -> ответ)]; ответ — веса вариантов."""
    def responder(images, system, prompt, options):
        t = float(re.search(r"время ([0-9.]+) с", prompt).group(1))
        for sub, fn in rules:
            if sub in prompt:
                ans = fn(t)
                return {k: ans.get(k, 0.0) for k in options}
        return {k: default.get(k, 0.0) for k in options}
    return responder


def _ev(tid, t=6.0, conf=0.6, source="rules", bboxes=None):
    return Event(tid, t - 0.5, t + 0.5, t, conf, source, [], dict(bboxes or {}))


@pytest.fixture
def qcfg(cfg):
    cfg["verify"]["mode"] = "qa"
    return cfg


def _run(clip, qcfg, clock, ev, responder, **kw):
    be = MockBackend(option_responder=responder, clock=clock)
    return verify_events(clip, [ev], 60.0, cfg=qcfg, backend=be, clock=clock, **kw)[0], be


def test_v02_left_with_cash_supports(clip, qcfg, clock):
    resp = scripted([("рабочем месте", lambda t: YES if t < 6 else NO),
                     ("наличные", lambda t: YES), ("инкассаторская сумка", lambda t: NO), ("сейфа", lambda t: NO)])
    ev, be = _run(clip, qcfg, clock, _ev("V02"), resp)
    v = ev.extra["vlm"]
    assert v["mode"] == "qa" and v["agreement"] == "supports" and v["verdict"] == "confirm"
    assert ev.confidence > 0.6 and any("наличные" in f for f in ev.facts)
    assert v["t_transition"] == pytest.approx(7.0)  # смена ответа между 4.0 и 10.0 с
    assert all(c["kind"] == "option_probs" for c in be.calls) and len(be.calls) == 5


def test_v02_left_tidy_is_normal(clip, qcfg, clock):
    resp = scripted([("рабочем месте", lambda t: YES if t < 6 else NO),
                     ("наличные", lambda t: NO), ("инкассаторская сумка", lambda t: NO), ("сейфа", lambda t: NO)])
    ev, _ = _run(clip, qcfg, clock, _ev("V02", conf=0.8), resp)
    v = ev.extra["vlm"]
    assert v["agreement"] == "contradicts" and v["verdict"] == "reject" and ev.confidence < 0.8
    assert ev.source == "rules" and ev.type_id == "V02"  # не удаляем и не меняем источник — решает fusion
    assert not any("наличные" in f for f in ev.facts) and v["facts_con"]


def test_unknown_is_not_false(clip, qcfg, clock):
    ev, _ = _run(clip, qcfg, clock, _ev("V02", conf=0.7), scripted([]))  # везде «Не видно»
    v = ev.extra["vlm"]
    assert v["agreement"] == "uncertain" and v["verdict"] == "unsure" and ev.confidence == pytest.approx(0.7)


def test_v05_phone_and_v06_drink(clip, qcfg, clock):
    ev, _ = _run(clip, qcfg, clock, _ev("V05"), scripted([("мобильный телефон", lambda t: YES)]))
    assert ev.extra["vlm"]["agreement"] == "supports" and any("телефон" in f for f in ev.facts)
    ev, _ = _run(clip, qcfg, clock, _ev("V06", conf=0.7), scripted([("ест или пьёт", lambda t: NO)]))
    assert ev.extra["vlm"]["agreement"] == "contradicts"


def test_v01_counts(clip, qcfg, clock):
    two = {"0": 0.1, "1": 0.5, "2": 9, "3": 0.2, "unknown": 0.2}
    one = {"0": 0.1, "1": 9, "2": 0.3, "3": 0.1, "unknown": 0.2}
    ev, _ = _run(clip, qcfg, clock, _ev("V01"), scripted([("стороне кассира", lambda t: two),
                                                          ("стороны клиентов", lambda t: one)]))
    assert ev.extra["vlm"]["agreement"] == "supports"
    ev, _ = _run(clip, qcfg, clock, _ev("V01", conf=0.7), scripted([("стороне кассира", lambda t: one),
                                                                    ("стороны клиентов", lambda t: one)]))
    assert ev.extra["vlm"]["agreement"] == "contradicts"


def test_zone_crop_added_and_no_boxes_drawn(clip, qcfg, clock):
    zones = [Zone("desk", [(0.1, 0.3), (0.9, 0.3), (0.9, 0.6), (0.1, 0.6)])]
    ev = _ev("V09", bboxes={"person": (0.2, 0.2, 0.5, 0.8)})
    _, be = _run(clip, qcfg, clock, ev, scripted([("личные вещи", lambda t: YES)]), zones=zones)
    assert [c["n_images"] for c in be.calls] == [2, 2]  # кадр + кроп стола
    _, be2 = _run(clip, qcfg, clock, _ev("V05", bboxes={"person": (0.2, 0.2, 0.5, 0.8)}),
                  scripted([("мобильный", lambda t: YES)]))
    assert [c["n_images"] for c in be2.calls] == [1, 1]  # у вопроса про телефон кропа нет


def test_qa_images_are_neutral(clip, qcfg, clock):
    seen = []

    def resp(images, system, prompt, options):
        seen.extend(images)
        return {k: 1.0 for k in options}

    _run(clip, qcfg, clock, _ev("V05", bboxes={"cell_phone": (0.3, 0.3, 0.5, 0.5)}), resp)
    arr = np.asarray(seen[0])
    assert not ((arr[:, :, 0] > 200) & (arr[:, :, 1] < 80)).any()  # красных рамок нет


def test_prompts_do_not_hint_violation():
    spec = load_spec()
    for q in spec["questions"].values():
        assert "нарушен" not in q["text"].lower() and "подозр" not in q["text"].lower()


def test_v99_falls_back_to_json(clip, qcfg, clock):
    be = MockBackend(mode="confirm", clock=clock)
    ev = verify_events(clip, [_ev("V99")], 60.0, cfg=qcfg, backend=be, clock=clock)[0]
    assert ev.extra["vlm"].get("mode") != "qa" and ev.extra["vlm"]["status"] == "ok"
    assert all(c.get("kind") != "option_probs" for c in be.calls)


def test_backend_without_logits_falls_back_to_json(clip, qcfg, clock):
    be = MockBackend(mode="confirm", clock=clock)
    be.supports_option_probs = False
    ev = verify_events(clip, [_ev("V05")], 60.0, cfg=qcfg, backend=be, clock=clock)[0]
    assert ev.extra["vlm"].get("mode") != "qa" and ev.extra["vlm"]["agreement"] == "supports"


def test_budget_cuts_questions(clip, qcfg, clock):
    ev, be = _run(clip, qcfg, clock, _ev("V02"), scripted([]), limits={"max_calls": 2})
    assert len(be.calls) == 2 and ev.extra["vlm"]["skipped"] == 3


def test_low_mass_answers_ignored(clip, qcfg, clock):
    be = MockBackend(option_responder=scripted([("мобильный", lambda t: YES)]), clock=clock)
    orig = be.option_probs

    def low(*a, **k):
        sc = orig(*a, **k)
        sc.mass = 0.05
        return sc

    be.option_probs = low
    ev = verify_events(clip, [_ev("V05", conf=0.6)], 60.0, cfg=qcfg, backend=be, clock=clock)[0]
    assert ev.extra["vlm"]["low_mass"] == 2 and ev.extra["vlm"]["agreement"] == "uncertain"


def test_refine_t_peak_flag(clip, qcfg, clock):
    qcfg["qa"]["refine_t_peak"] = True
    resp = scripted([("рабочем месте", lambda t: YES if t < 6 else NO), ("наличные", lambda t: YES)])
    ev, _ = _run(clip, qcfg, clock, _ev("V02"), resp)
    assert ev.t_peak == pytest.approx(7.0) and ev.extra["vlm_original_t_peak"] == 6.0


def test_decide_and_transition_units():
    spec = load_spec()
    a = {("safe_open", "at"): {"yes": 0.9, "no": 0.05, "unknown": 0.05}}
    d = decide("V03", a, spec)
    assert d["p_support"] == pytest.approx(0.9) and d["p_vlm"] > 0.9
    assert decide("V42", a, spec)["agreement"] == "uncertain"
    tt = transition_time({("cashier_present", "before"): {"yes": 0.9, "no": 0.1},
                          ("cashier_present", "after2"): {"yes": 0.1, "no": 0.9}}, {"before": 4.0, "after2": 10.0})
    assert tt == pytest.approx(7.0)
