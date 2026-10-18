import json
from pathlib import Path

import pytest

from ml.vlm.compare import load_candidates, summarize
from ml.vlm.tests.test_vlm_cli_api import _write_video


def _ev(tid, t, conf, source="rules"):
    return {"type_id": tid, "t_start": t - 1, "t_end": t + 1, "t_peak": t, "confidence": conf, "source": source,
            "facts": [], "bboxes": {}, "track_ids": [], "extra": {}}


@pytest.fixture
def setup(tmp_path):
    vids = tmp_path / "videos"
    vids.mkdir()
    _write_video(vids / "X-1.avi", seconds=6.0)   # нарушение, кандидат слабый (FN без VLM)
    _write_video(vids / "X-2.avi", seconds=5.0)   # чисто, кандидат сильный (FP без VLM)
    _write_video(vids / "X-3.avi", seconds=4.0)   # чисто, кандидатов нет (TN)
    (tmp_path / "labels.csv").write_text("file,label,t\nX-1.mov,нарушение,3\nX-2.mov,без нарушения,\n"
                                         "X-3.mov,без нарушения,\n", encoding="utf-8")
    cands = tmp_path / "cands"
    for name, events in (("X-1", [_ev("V05", 3.5, 0.4)]), ("X-2", [_ev("V05", 2.0, 0.6)]), ("X-3", [])):
        d = cands / name
        d.mkdir(parents=True)
        (d / "events_rules.json").write_text(json.dumps(events), encoding="utf-8")
        (d / "events_temporal.json").write_text("[]", encoding="utf-8")
        (d / "zones.json").write_text(json.dumps([{"name": "desk", "polygon": [[0.1, 0.3], [0.9, 0.3], [0.9, 0.6]],
                                                   "conf": 0.9}]), encoding="utf-8")
    return tmp_path


def test_load_candidates_dir_and_jsonl(setup, tmp_path):
    c = load_candidates(str(setup / "cands"))
    assert set(c) == {"x-1", "x-2", "x-3"} and c["x-1"]["events"][0].confidence == 0.4
    assert c["x-1"]["zones"][0].name == "desk"
    jl = tmp_path / "c.jsonl"
    jl.write_text(json.dumps({"video": "Х-1.MOV", "events": [_ev("V02", 1, 0.5)], "zones": []}) + "\n",
                  encoding="utf-8")
    assert load_candidates(str(jl))["x-1"]["events"][0].type_id == "V02"


SYNTH_CATALOG = {"version": 1, "types": {
    "V05": {"name": "Телефон (синтетический каталог)", "observable": ["держит телефон"], "not_observable": []},
    "V99": {"name": "Иное (синтетический каталог)", "observable": ["иное"], "not_observable": []}}}


@pytest.fixture(autouse=True)
def _synthetic_catalog():
    """Изоляция: тесты compare не зависят от norms A13 в сборке (имена/нормы/гейты fusion)."""
    from ml.vlm.catalog import set_catalog
    set_catalog(SYNTH_CATALOG)
    yield
    set_catalog(None)


def test_compare_counts_fixes_and_breaks(setup, monkeypatch, capsys):
    from ml.vlm.compare import main
    monkeypatch.setenv("SENTINEL_VLM_MOCK_MODE", "confirm")
    rc = main(["--videos", str(setup / "videos"), "--labels", str(setup / "labels.csv"), "--candidates",
               str(setup / "cands"), "--modes", "none,json,qa", "--verdict", "threshold", "--threshold", "0.5",
               "--dump-dir", str(setup / "dump"),
               "--out", str(setup / "rep.json")])
    out = capsys.readouterr().out
    rep = json.loads(out)
    assert rc == 0 and rep["videos"] == 3
    none, js = rep["summary"]["none"], rep["summary"]["json"]
    assert (none["tp"], none["fn"], none["fp"], none["tn"]) == (0, 1, 1, 1)
    assert js["vs_none"]["fn_fixed"] == 1 and js["vs_none"]["fp_created"] == 0  # mock «confirm» поднял слабый
    assert js["vs_none"]["consistent"] is True and rep["summary"]["_common"]["videos_compared"] == 3
    assert none["n"] == js["n"] == rep["summary"]["qa"]["n"] == 3  # все режимы — на одних и тех же роликах
    assert all(m["verdict_by"] == ["threshold"] for k, m in rep["summary"].items() if k != "_common")
    assert js["timecode_hit"]["<=1s"] == 1.0 and js["vlm_calls_total"] >= 2
    assert "qa" in rep["summary"] and "note" in rep
    assert "X-1" not in out  # построчно — только в локальном дампе
    assert (Path(rep["dump_dir"]) / "rows.json").exists()


def test_compare_reject_fixes_fp(setup, monkeypatch, capsys):
    from ml.vlm.compare import main
    monkeypatch.setenv("SENTINEL_VLM_MOCK_MODE", "reject")
    main(["--videos", str(setup / "videos"), "--labels", str(setup / "labels.csv"), "--candidates",
          str(setup / "cands"), "--modes", "none,json", "--verdict", "threshold", "--threshold", "0.5",
          "--no-dump"])
    js = json.loads(capsys.readouterr().out)["summary"]["json"]
    assert js["vs_none"]["fp_fixed"] == 1 and js["fp"] == 0 and js["vs_none"]["consistent"] is True


def test_summarize_unknown_not_negative():
    rows = [{"gt": True, "t_gt": 3.0, "modes": {"none": {"verdict": None, "t": None, "vlm_s": 0.0}}}]
    s = summarize(rows, ["none"])["none"]
    assert s["unknown_verdicts"] == 1 and s["fn"] == 0 and s["n"] == 0


def test_verdict_fusion_required_but_missing(setup, monkeypatch, capsys):
    import sys
    from ml.vlm.compare import main
    monkeypatch.setitem(sys.modules, "ml.fusion", None)  # fusion недоступен
    with pytest.raises(ImportError):
        main(["--videos", str(setup / "videos"), "--labels", str(setup / "labels.csv"), "--candidates",
              str(setup / "cands"), "--modes", "none", "--verdict", "fusion", "--no-dump"])


def _row(gt, verdicts, t_gt=3.0):
    return {"gt": gt, "t_gt": t_gt if gt else None,
            "modes": {m: {"verdict": v, "t": t_gt if v == "violation" else None, "vlm_s": 0.1, "vlm_calls": 1}
                      for m, v in verdicts.items()}}


def test_fixes_and_breaks_on_same_examples():
    """Ролик с неопределённым вердиктом в одном из режимов исключается из сравнения ВСЕХ режимов."""
    rows = [_row(True, {"none": "no_violation", "json": "violation"}),       # fn_fixed
            _row(False, {"none": "violation", "json": "no_violation"}),     # fp_fixed
            _row(False, {"none": "no_violation", "json": "violation"}),     # fp_created
            _row(True, {"none": "violation", "json": "no_violation"}),      # fn_created
            _row(True, {"none": "violation", "json": None})]                # json не досчитал -> вне сравнения
    s = summarize(rows, ["none", "json"])
    assert s["_common"] == {"videos_total": 5, "videos_compared": 4, "excluded_unknown": 1}
    assert s["none"]["n"] == s["json"]["n"] == 4 and s["json"]["unknown_verdicts"] == 1
    vs = s["json"]["vs_none"]
    assert (vs["fn_fixed"], vs["fp_fixed"], vs["fp_created"], vs["fn_created"]) == (1, 1, 1, 1)
    assert vs["consistent"] is True
    assert (s["none"]["tp"], s["json"]["tp"]) == (1, 1)  # раньше none считал и 5-й ролик -> tp 2 против 1
