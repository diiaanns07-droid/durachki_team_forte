import json
import re

from ml.vlm.backends.mock import MockBackend
from ml.vlm.discoverer import discover_events, plan_windows
from ml.vlm.tests.conftest import SyntheticClip


def _finding(item="phone", conf=0.85, facts=("Кассир держит в руке мобильный телефон",), frame=2):
    return json.dumps({"findings": [{"item": item, "observed_facts": list(facts), "confidence": conf,
                                     "frame": frame}], "uncertain_because": ""}, ensure_ascii=False)


EMPTY = '{"findings": [], "uncertain_because": ""}'
CONFIRM = json.dumps({"visible": True, "type_id": "V05", "observed_facts": [], "confidence": 0.8, "best_frame": 1,
                      "visible_frames": [1], "uncertain_because": ""})
REJECT = json.dumps({"visible": False, "type_id": "V05", "observed_facts": [], "confidence": 0.9, "best_frame": 1,
                     "visible_frames": [], "uncertain_because": "нет телефона"})


def scripted(discover_answers, verify_answer=CONFIRM):
    """Ответы по типу промпта: discover-окна по очереди (потом пусто), verify — фиксированный ответ."""
    it = iter(discover_answers)

    def responder(images, system, prompt):
        return next(it, EMPTY) if '"findings"' in prompt else verify_answer
    return responder


def test_disabled_by_default(clip, cfg):
    assert cfg["discover"]["enabled"] is False
    assert discover_events(clip, 60.0, cfg=cfg, backend=MockBackend(mode="phone")) == []


def test_plan_windows_coverage_and_motion_order():
    motion = [(t / 2, 1.0 if 20 <= t / 2 <= 23 else 0.0) for t in range(0, 120)]
    wins = plan_windows(60.0, 3.0, motion, min_coverage=3)
    assert len(wins) == 20
    assert wins[0][0] <= 21.0 <= wins[0][1] or wins[0][0] <= 22.0 <= wins[0][1]  # самое активное — первым
    assert all(abs((w[1] - w[0]) - 3.0) < 1e-6 for w in wins)
    assert plan_windows(0.0, 3.0) == []


def test_finding_becomes_low_prior_event_and_is_reverified(clip, cfg, clock):
    be = MockBackend(responder=scripted([_finding()]), clock=clock, latency_s=0.5)
    out = discover_events(clip, 60.0, cfg=cfg, backend=be, clock=clock, force=True)
    assert len(out) == 1
    e = out[0]
    assert e.type_id == "V05" and e.source == "vlm" and e.confidence <= cfg["discover"]["max_conf"]
    assert e.facts and e.extra["vlm_discover"]["item"] == "phone"
    assert e.extra["vlm"]["stage"] == "discover+verify" and e.extra["vlm"]["agreement"] == "supports"
    assert len(be.calls) <= cfg["budget"]["discover"]["max_calls"]  # reverify — в том же общем лимите
    b = e.extra["vlm"]["budget"]
    assert b["calls"] == len(be.calls) and b["reverify_calls"] == 1  # отчёт включает вызов reverify


def test_finding_without_concrete_fact_is_dropped(clip, cfg, clock):
    be = MockBackend(responder=scripted([_finding(facts=["Что-то подозрительное"])]), clock=clock)
    cfg["discover"]["reverify"] = False
    assert discover_events(clip, 60.0, cfg=cfg, backend=be, clock=clock, force=True) == []


def test_low_confidence_finding_dropped(clip, cfg, clock):
    be = MockBackend(responder=scripted([_finding(conf=0.3)]), clock=clock)
    cfg["discover"]["reverify"] = False
    assert discover_events(clip, 60.0, cfg=cfg, backend=be, clock=clock, force=True) == []


def test_other_item_gives_v99(clip, cfg, clock):
    be = MockBackend(responder=scripted([_finding("other", 0.9, ["Человек передаёт пакет через перегородку"])]),
                     clock=clock)
    cfg["discover"]["reverify"] = False
    out = discover_events(clip, 60.0, cfg=cfg, backend=be, clock=clock, force=True)
    assert [e.type_id for e in out] == ["V99"]


def test_reverify_contradiction_drops_finding(clip, cfg, clock):
    be = MockBackend(responder=scripted([_finding()], REJECT), clock=clock)
    cfg["budget"]["discover"]["max_calls"] = 6
    out = discover_events(clip, 60.0, cfg=cfg, backend=be, clock=clock, force=True)
    assert out == []


def test_reserve_call_for_reverify(clip, cfg, clock):
    be = MockBackend(responder=scripted([_finding()], REJECT), clock=clock)
    cfg["budget"]["discover"]["max_calls"] = 4  # 12 с = 4 окна; одно окно уступает вызов под reverify
    out = discover_events(clip, 60.0, cfg=cfg, backend=be, clock=clock, force=True)
    assert len(be.calls) == 4 and out == []
    assert sum('"findings"' in c["prompt"] for c in be.calls) == 3


def test_no_reverify_without_reserve(clip, cfg, clock):
    be = MockBackend(responder=scripted([_finding()], REJECT), clock=clock)
    cfg["budget"]["discover"]["max_calls"] = 4
    cfg["discover"]["reverify_reserve_calls"] = 0  # окна съели все вызовы -> находка остаётся непроверенной
    out = discover_events(clip, 60.0, cfg=cfg, backend=be, clock=clock, force=True)
    assert len(be.calls) == 4 and len(out) == 1
    assert out[0].extra["vlm"]["stage"] == "discover" and out[0].extra["vlm"]["agreement"] == "not_checked"


def test_budget_limits_windows(cfg, clock):
    long_clip = SyntheticClip(duration=60.0)
    be = MockBackend(mode="neutral", latency_s=3.0, clock=clock)
    out = discover_events(long_clip, 10.0, cfg=cfg, backend=be, clock=clock, force=True,
                          limits={"est_call_s": 3.0, "safety": 1.0})
    assert out == [] and len(be.calls) <= 3 and clock() <= 10.0 + 1e-9


def test_merge_adjacent_windows(clip, cfg, clock):
    be = MockBackend(responder=scripted([_finding(), _finding()]), clock=clock)
    cfg["discover"]["reverify"] = False
    cfg["discover"]["min_coverage_windows"] = 0
    cfg["discover"]["motion_weighting"] = False
    out = discover_events(clip, 60.0, cfg=cfg, backend=be, clock=clock, force=True)
    assert len(out) == 1 and out[0].extra["vlm_discover"].get("windows", 1) == 2


def test_external_motion_curve_is_used(cfg, clock):
    long_clip = SyntheticClip(duration=30.0)
    seen = []

    def responder(images, system, prompt):
        seen.append(prompt)
        return EMPTY

    be = MockBackend(responder=responder, clock=clock)
    cfg["discover"]["min_coverage_windows"] = 0
    motion = [(t * 0.5, 1.0 if 24.0 <= t * 0.5 <= 27.0 else 0.0) for t in range(60)]
    discover_events(long_clip, 60.0, cfg=cfg, backend=be, clock=clock, force=True, motion=motion,
                    limits={"max_calls": 1})
    assert len(seen) == 1
    line = seen[0].split("время кадров (с): ")[1].split("\n")[0]
    times = [float(x) for x in re.findall(r"\d+: (\d+\.\d+)", line)]
    assert times
    assert all(24.0 <= t <= 27.0 for t in times)  # первое окно — самое активное по внешней кривой
