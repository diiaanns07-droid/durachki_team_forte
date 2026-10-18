"""coarse→fine (H1) и политика таймкода (H2): по умолчанию выключены; кандидаты не теряются."""
from __future__ import annotations

from backend.pipeline import mocks, run
from backend.pipeline.contract import Event
from backend.pipeline.refine import Window, apply_refinement, apply_timecode_policy, build_windows

CFG = {"refine": {"pre_s": 1.0, "post_s": 1.0, "merge_gap_s": 1.0, "max_windows": 2}}


def _ev(tid, t0, t1, tp, conf, src="rules"):
    return Event(tid, t0, t1, tp, conf, src, [f"{tid}@{tp}"])


def test_build_windows_merge_and_limit():
    c = [_ev("V01", 2, 3, 2.5, 0.9), _ev("V02", 3.5, 4, 3.8, 0.5), _ev("V03", 20, 21, 20.5, 0.7),
         _ev("V04", 40, 41, 40.5, 0.2)]
    ws = build_windows(c, CFG, duration=45.0)
    assert len(ws) == 2                                    # max_windows: самые уверенные первыми
    assert sorted(ws[0].candidates) == [0, 1]              # близкие кандидаты — одно окно
    assert ws[0].t0 == 1.0 and ws[0].t1 == 5.0
    assert ws[1].candidates == [2]


def test_apply_refinement_never_drops():
    c = [_ev("V01", 1.5, 3.0, 2.0, 0.6), _ev("V03", 10, 11, 10.5, 0.8), _ev("V05", 30, 31, 30.5, 0.4)]
    w1 = Window(0.5, 4.0, [0], [_ev("V01", 2.125, 3.0, 2.25, 0.7), _ev("V06", 3.0, 3.5, 3.2, 0.6)], True)
    w2 = Window(9.0, 12.0, [1], [], True)                  # окно не подтвердило V03
    w3 = Window(29.0, 32.0, [2], [], False, "таймаут")      # окно не обработано
    out, st = apply_refinement(c, [w1, w2, w3])
    by = {e.type_id: e for e in out}
    assert by["V01"].t_start == 2.125 and by["V01"].t_peak == 2.25 and by["V01"].extra["refine"] == "confirmed"
    assert by["V01"].extra["overview"]["t_peak"] == 2.0 and by["V01"].confidence == 0.7
    assert by["V03"].extra["refine"] == "not_confirmed" and by["V03"].t_peak == 10.5   # не отброшен
    assert by["V05"].t_peak == 30.5 and "refine" not in by["V05"].extra                  # не уточнялся
    assert by["V06"].extra["refine"] == "window_new"
    assert st == {"windows": 3, "processed": 2, "confirmed": 1, "not_confirmed": 1, "unrefined": 1,
                  "window_new": 1}


def test_timecode_policy():
    e = _ev("V01", 2.0, 6.0, 4.5, 0.9)
    assert apply_timecode_policy([e], {})[0].t_peak == 4.5                          # по умолчанию — как есть
    on = apply_timecode_policy([e], {"timecode": {"policy": "onset"}})[0]
    assert on.t_peak == 2.0 and on.extra["t_peak_module"] == 4.5
    off = apply_timecode_policy([e], {"timecode": {"policy": "onset_offset", "offset_s": 0.5}})[0]
    assert off.t_peak == 2.5


def _rules_coarse_fine(states, tracks, zones, catalog):
    ts = [s.t for s in states]
    dense = len(ts) > 2 and min(b - a for a, b in zip(ts, ts[1:])) < 0.3
    if dense:   # плотное окно: точное начало перехода
        return [Event("V02", 2.125, 3.0, 2.25, 0.8, "rules", ["кассир отошёл (окно)"])]
    return [Event("V02", 1.5, 3.0, 2.0, 0.6, "rules", ["кассир отошёл (обзор)"])]


def test_refine_end_to_end(fake_ml, synth_video):
    fake_ml({"ml.rules.run_rules": _rules_coarse_fine, "ml.temporal.score": lambda clip: ([], []),
             "ml.vlm.verify": lambda clip, ev, b: []})
    base = run(synth_video, "job-base", None, 120)
    assert base.meta["modes"]["refine"] is False and base.meta["stages"]["refine"]["status"] == "disabled"
    assert base.violations[0].timecode_sec == 2.0
    res = run(synth_video, "job-refine", None, 120, overrides={"refine": {"enabled": True, "fps": 8.0}})
    assert res.status == "done" and res.meta["sample_fps"] <= 2.0                     # обзор реже
    assert res.meta["stages"]["refine"]["status"] == "ok"
    assert res.meta["refine"]["confirmed"] == 1
    v = res.violations[0]
    assert v.timecode_sec == 2.25 and v.t_start == 2.12


def test_refine_failure_keeps_candidates(fake_ml, synth_video):
    calls = {"n": 0}

    def detect(clip, every_n=1):
        calls["n"] += 1
        if calls["n"] > 1:           # окно (второй вызов) падает
            raise RuntimeError("окно: ошибка детектора")
        return mocks.detect(clip, every_n)

    fake_ml({"ml.detect.detect": detect, "ml.rules.run_rules": _rules_coarse_fine,
             "ml.temporal.score": lambda clip: ([], []), "ml.vlm.verify": lambda clip, ev, b: []})
    res = run(synth_video, "job-refine-fail", None, 120, overrides={"refine": {"enabled": True}})
    assert res.status == "done" and res.verdict == "violation"           # обзорный кандидат остался
    assert res.meta["stages"]["refine"]["status"] == "error"
    assert any("refine" in w for w in res.meta["warnings"])
    assert res.violations[0].timecode_sec == 2.0


def test_refine_is_optional_for_completeness(fake_ml, synth_video):
    fake_ml({"ml.rules.run_rules": lambda *a: [], "ml.temporal.score": lambda clip: ([], [])})
    res = run(synth_video, "job-refine-none", None, 120, overrides={"refine": {"enabled": True}})
    assert res.status == "done" and res.verdict == "no_violation"
    assert res.meta["stages"]["refine"]["status"] == "skipped"            # нет кандидатов
