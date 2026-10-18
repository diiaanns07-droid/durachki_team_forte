from __future__ import annotations

from pathlib import Path

import pytest

from eval.common import load_config, write_json
from eval.labels import LabelRow, load_ground_truth, write_labels_csv
from eval.metrics import evaluate, evaluate_dirs, format_scorecard, format_summary, match_one_to_one, wilson
from eval.schema import check_api_response
from eval.score_like_jury import format_score, score

from .conftest import make_pred, make_violation


def _labels(tmp_path: Path) -> Path:
    rows = [
        LabelRow("X-1", "violation", [22.0], md5="a"),
        LabelRow("X-2", "violation", [30.0], md5="b"),
        LabelRow("X-3", "no_violation", [], md5="c"),
        LabelRow("X-6", "no_violation", [], md5="d"),
        LabelRow("X-4", "violation", [57.0], md5="dup", raw="57 секунда"),
        LabelRow("X-47", "violation", [68.0], md5="dup", raw="1минута 8 секунда"),
        LabelRow("X-7", "violation", [40.0], md5="e"),
        LabelRow("X-8", "violation", [15.0], md5="f"),
        LabelRow("X-9", "violation", [50.0], md5="g"),
        LabelRow("X-10", "violation", [5.0], md5="lab"),
        LabelRow("X-11", "no_violation", [], md5="lab"),
    ]
    from eval.labels import assign_groups

    assign_groups(rows)
    p = tmp_path / "labels.csv"
    write_labels_csv(rows, p)
    return p


def _preds() -> dict:
    return {
        "X-1": make_pred("j1", [make_violation(22.4)]),
        "X-2": make_pred("j2", []),
        "X-3": make_pred("j3", [make_violation(10.0)]),
        "X-6": make_pred("j6", []),
        "X-4": make_pred("j4", [make_violation(67.5, frames=[67.5])]),
        "X-7": make_pred("j7", [make_violation(48.0)]),
        "X-8": make_pred("j8", [make_violation(15.5), make_violation(16.0), make_violation(90.0)]),
        "X-9": {"job_id": "j9", "status": "failed", "verdict": None, "violations": [],
                "meta": {"error": "deadline exceeded"}},
        "X-10": make_pred("j10", [make_violation(5.0)]),
    }


@pytest.fixture
def result(tmp_path: Path) -> dict:
    gt = load_ground_truth(_labels(tmp_path))
    return evaluate(gt, _preds(), {"X-1": {"latency_s": 12.0}, "X-2": {"latency_s": 75.0}}, load_config())


def test_wilson():
    assert wilson(0, 0) is None
    lo, hi = wilson(19, 20)
    assert 0.7 < lo < 0.95 and hi <= 1.0


def test_binary(result):
    v = result["summary"]["video"]
    assert (v["tp"], v["fn"], v["fp"], v["tn"]) == (4, 2, 1, 1)
    assert v["failed"] == 1 and v["timeouts"] == 1 and v["timeout_ids"] == ["X-9"]
    assert v["label_conflict_ids"] == ["X-10"]          # конфликт меток — не выбираем, исключаем
    assert v["n"] == 8 and v["recall"] == pytest.approx(4 / 6, abs=1e-3)
    assert v["recall_ci95"][0] < v["recall"] < v["recall_ci95"][1]
    assert set(v["fn_ids"]) == {"X-2", "X-9"} and v["fp_ids"] == ["X-3"]


def test_one_to_one_and_conflict_alternatives(result):
    by = {e["video_id"]: e for e in result["per_video"]}
    x4 = by["X-4"]                       # одно событие с альтернативами [57, 68]
    assert x4["events_total"] == 1 and x4["events_matched"] == 1
    assert x4["dt_abs"] == pytest.approx(0.5) and x4["matched_gt_t"] == 68.0 and x4["evidence_ok"]
    x8 = by["X-8"]
    assert x8["events_matched"] == 1 and x8["extra_candidates"] == 2 and x8["dt_abs"] == pytest.approx(0.5)
    x7 = by["X-7"]                       # вне окна сопоставления: не hit, но «поздно» и справочный Δt
    assert x7["category"] == "late" and x7["events_matched"] == 0 and x7["nearest_dt_signed"] == 8.0
    assert by["X-1"]["category"] == "TP" and by["X-1"]["hits"]["0.5"]
    loc = result["summary"]["localization"]
    assert loc["extra_candidates_total"] == 3 and "неполная" in loc["caveat"]   # X-8: 2, X-7: 1 (вне окна)
    # политика separate: конфликт 57/68 — отдельным блоком, в основные метрики локализации не входит
    assert loc["conflict_policy"] == "separate"
    assert result["summary"]["localization_alt"]["events_total"] == loc["events_total"] + 1
    cg = result["summary"]["conflict_groups"]
    assert len(cg) == 1 and cg[0]["group"] == "X-4" and cg[0]["dt_to_alternatives"] == {"57": 10.5, "68": -0.5}


def test_match_one_to_one_unique():
    pairs = match_one_to_one([[10.0], [11.0]], [(0, 10.4)], 5.0)
    assert len(pairs) == 1 and pairs[0]["event"] == 0
    pairs = match_one_to_one([[10.0]], [(0, 10.4), (1, 9.9)], 5.0)
    assert len(pairs) == 1 and pairs[0]["pred"] == 1


def test_types_only_with_real_labels(result, tmp_path: Path):
    assert result["summary"]["types"]["available"] is False and result["summary"]["types"]["accuracy"] is None
    s = score(result)
    assert s["components"]["type"]["points"] is None and "type" in s["missing"]
    types = tmp_path / "labels_types.csv"
    types.write_text("video_id,t_sec,type_id\nX-1,22,V01\nX-8,15,V05\n", encoding="utf-8")
    gt = load_ground_truth(tmp_path / "labels.csv", types)
    res = evaluate(gt, _preds(), {}, load_config())
    ty = res["summary"]["types"]
    assert ty["available"] and ty["n_events"] == 2 and ty["accuracy"] == 0.5
    assert {e["video_id"]: e["category"] for e in res["per_video"]}["X-8"] == "wrong_type"
    assert score(res)["components"]["type"]["source"] == "measured"


def test_norms_and_language(result):
    nr = result["summary"]["norms"]
    assert nr["with_rule_rate"] == 1.0 and nr["url_ok_rate"] == 1.0 and "≠" in nr["note"]
    assert result["summary"]["language"]["hedged_rate"] == 1.0


def test_latency_and_scorecard(result):
    lat = result["summary"]["latency"]
    assert lat["over_sla_ids"] == ["X-2"] and lat["max"] == 75.0
    sc = result["scorecard"]
    assert sc["n_videos"] == 8 and sc["timeouts"] == 1 and sc["versions"]["eval_version"]
    txt = format_scorecard(sc) + format_summary(result["summary"])
    assert "SCORECARD" in txt and "recall" in txt


def test_proxy_score_perfect_and_disclaimer(tmp_path: Path):
    rows = [LabelRow("X-1", "violation", [22.0]), LabelRow("X-2", "no_violation", [])]
    p = tmp_path / "l.csv"
    write_labels_csv(rows, p)
    gt = load_ground_truth(p)
    res = evaluate(gt, {"X-1": make_pred("a", [make_violation(22.0)]), "X-2": make_pred("b", [])},
                   {"X-1": {"latency_s": 5}, "X-2": {"latency_s": 5}}, load_config())
    s = score(res)
    assert s["proxy_score"] == pytest.approx(100.0) and "неизвестна" in s["disclaimer"]
    assert set(s["missing"]) == {"type", "robustness"}
    assert "PROXY_SCORE" in format_score(s)
    bad = evaluate(gt, {"X-1": make_pred("a", []), "X-2": make_pred("b", [make_violation(3.0)])}, {}, load_config())
    assert score(bad)["components"]["detection"]["value"] == 0.0


def test_evaluate_dirs_and_wrapper(tmp_path: Path):
    lab = _labels(tmp_path)
    pdir = tmp_path / "preds"
    for vid, d in _preds().items():
        write_json(pdir / f"{vid}.json", {"video_id": vid, "latency_s": 3.0, "response": d} if vid == "X-1" else d)
    write_json(pdir / "_runinfo.json", {"X-2": {"latency_s": 4.0}})
    res = evaluate_dirs(lab, pdir, None, load_config(), subset={"X-1", "X-2"})
    assert res["summary"]["video"]["n"] == 2
    lat = {e["video_id"]: e["latency_s"] for e in res["per_video"]}
    assert lat == {"X-1": 3.0, "X-2": 4.0}
    assert res["scorecard"]["versions"]["labels_sha256"]


def test_schema_errors():
    assert check_api_response(make_pred("a", [make_violation(1.0)])).valid
    bad = make_pred("a", [make_violation(1.0)])
    del bad["violations"][0]["description"]
    bad["violations"][0]["evidence_frames"][0]["image_base64"] = "not-base64!!"
    rep = check_api_response(bad)
    assert not rep.valid and any("description" in e for e in rep.errors)
    assert any("JPEG" in w for w in rep.warnings)
    rep = check_api_response({"job_id": "x", "status": "done", "verdict": "maybe", "violations": []})
    assert any("verdict" in e for e in rep.errors)


def test_failed_partial_invalid_never_clean(tmp_path: Path):
    """unknown ≠ false: failed / partial / невалидный вердикт на негативе — не TN; неизвестная метка исключена."""
    rows = [LabelRow("X-1", "no_violation", []), LabelRow("X-2", "no_violation", []), LabelRow("X-3", "no_violation", []),
            LabelRow("X-4", "violation", [5.0]), LabelRow("X-5", "no_violation", []), LabelRow("X-6", "violation", [3.0])]
    p = tmp_path / "l.csv"
    write_labels_csv(rows, p)
    txt = p.read_text(encoding="utf-8").replace("X-6,,violation", "X-6,,???")
    p.write_text(txt, encoding="utf-8")
    preds = {
        "X-1": {"job_id": "a", "status": "failed", "verdict": None, "violations": [], "error": "CUDA OOM"},
        "X-2": make_pred("b", [], meta={"coverage": 0.4}),                       # partial → не «чисто»
        "X-3": {"job_id": "c", "status": "done", "verdict": None, "violations": []},   # невалидный вердикт
        "X-4": make_pred("d", [], meta={"coverage": {"frames": 0.5, "vlm": 0.0}}),  # partial на позитиве → FN
        "X-5": make_pred("e", [], meta={"coverage": 1.0, "vram_peak_mb": 4700}),   # legacy число coverage
        "X-6": make_pred("f", [make_violation(3.0)]),
    }
    res = evaluate(load_ground_truth(p), preds, {}, load_config())
    v = res["summary"]["video"]
    assert v["label_unknown_ids"] == ["X-6"] and v["n"] == 5
    assert v["tn"] == 1 and v["unresolved_neg"] == 3 and v["specificity"] == 0.25
    assert v["fn"] == 1 and v["fn_failed_or_partial"] == 1 and v["partial"] == 2
    cats = {e["video_id"]: e["category"] for e in res["per_video"]}
    assert cats["X-2"] == "partial" and cats["X-3"] == "failed" and cats["X-1"] == "failed" and cats["X-5"] == "TN"
    sc = res["scorecard"]
    assert sc["evaluation_kind"] == "exploratory" and "не независимый" in sc["kind_note"]
    assert sc["vram_process_peak_mb_max"] == 4700 and sc["vram_process_sources"] == ["pipeline_meta"]
    assert sc["vram_device_peak_mb_max"] is None                                      # не измерено ≠ 0
    assert sc["coverage_unknown"] == 1          # X-4: coverage без ratio (не канон A01) — доля неизвестна
    assert "РАЗВЕДОЧНАЯ" in format_scorecard(sc)
    # канон A01 (require по умолчанию): no_violation без meta.coverage не подтверждён → partial, не «чисто»
    res2 = evaluate(load_ground_truth(p), {"X-5": make_pred("e", [], meta={"coverage": None})}, {}, load_config())
    assert {e["video_id"]: e["category"] for e in res2["per_video"]}["X-5"] == "partial"
    # канонический объект A01 с ratio ниже порога / complete=false — partial
    for cov in ({"ratio": 0.5, "complete": True}, {"ratio": 1.0, "complete": False}):
        r3 = evaluate(load_ground_truth(p), {"X-5": make_pred("e", [], meta={"coverage": cov})}, {}, load_config())
        assert {e["video_id"]: e["category"] for e in r3["per_video"]}["X-5"] == "partial"
    # старые прогоны без coverage — только явно coverage.require=false
    r4 = evaluate(load_ground_truth(p), {"X-5": make_pred("e", [], meta={"coverage": None})}, {},
                  load_config(overrides={"coverage": {"require": False}}))
    assert {e["video_id"]: e["category"] for e in r4["per_video"]}["X-5"] == "TN"


def test_l1_labels_format_and_policy_whatif(tmp_path: Path):
    """Формат L1 (tools/dataprep/build_labels.py): без sha256, t_sec целым, conflict=1 у дубликатов 57/68."""
    p = tmp_path / "labels.csv"
    p.write_text("video_id,file,label,t_sec,md5,group,t_all,conflict,raw\n"
                 "X-4,Х-4.mov,violation,57,aa,X-4,57,1,Х-4 | 57 секунда\n"
                 "X-47,Х-47.mov,violation,68,aa,X-4,68,1,Х-47 | 1минута 8 секунда\n"
                 "X-1,X-1.mov,violation,22,bb,X-1,22,0,X-1 | 22 секунда\n"
                 "X-2,X-2.mov,no_violation,,cc,X-2,,0,X-2 | без нарушения\n", encoding="utf-8")
    gt = load_ground_truth(p)
    assert set(gt) == {"X-4", "X-1", "X-2"} and gt["X-4"].conflict and gt["X-4"].events == [[57.0, 68.0]]
    assert gt["X-4"].raw["X-47"].endswith("1минута 8 секунда")                   # исходные строки сохранены
    v = make_violation(23.5)
    v["t_start"], v["t_end"] = 22.2, 26.0
    res = evaluate(gt, {"X-1": make_pred("a", [v]), "X-2": make_pred("b", []), "X-4": make_pred("c", [])},
                   {}, load_config())
    tp = res["summary"]["timecode_policy"]
    assert tp["timecode_sec"]["median_abs_dt"] == 1.5 and tp["t_start"]["median_abs_dt"] == pytest.approx(0.2)
    assert tp["t_start"]["hit"]["0.5"] == 1.0 and tp["timecode_sec"]["hit"]["0.5"] == 0.0
    assert res["summary"]["conflict_groups"][0]["gt_alternatives"] == [57.0, 68.0]


def test_degraded_uses_a03_structured_status(tmp_path: Path):
    """«vlm_verify: skipped — нет кандидатов» — штатно; timeout/дедлайн/fallback/meta.degraded — деградация."""
    rows = [LabelRow(f"X-{i}", "no_violation", []) for i in range(1, 6)]
    p = tmp_path / "l.csv"
    write_labels_csv(rows, p)
    ok_skip = {"vlm_verify": {"status": "skipped", "error": "нет кандидатов"}, "detect": {"status": "ok"}}
    preds = {
        "X-1": make_pred("a", [], meta={"stages": ok_skip,
                                        "warnings": ["дополнительная стадия vlm_verify: skipped — нет кандидатов"]}),
        "X-2": make_pred("b", [], meta={"stages": {"detect": {"status": "timeout"}}}),
        "X-3": make_pred("c", [], meta={"stages": {"vlm_discover": {"status": "skipped",
                                                                    "error": "до дедлайна осталось 3.0 с"}}}),
        "X-4": make_pred("d", [], meta={"degraded": True}),
        "X-5": make_pred("e", [], meta={"stages": {"detect": {"status": "ok", "origin": "fallback"}}}),
    }
    res = evaluate(load_ground_truth(p), preds, {}, load_config())
    deg = {e["video_id"]: e["degraded"] for e in res["per_video"]}
    assert deg == {"X-1": False, "X-2": True, "X-3": True, "X-4": True, "X-5": True}
