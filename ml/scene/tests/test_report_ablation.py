"""Отчёт (OpenCV, без matplotlib) и абляция защит на синтетике (демо-данные)."""
from __future__ import annotations

import json

from ml.scene import ablation, report
from ml.scene.io import dump_states
from ml.scene.tests import test_phase2 as S


def test_report_renders_png_and_index(tmp_path):
    tr = S.leaving_cashier()
    st, trace = S.run(S.person_dets([tr]) + S.cash(), [tr])
    dump_states(st, tmp_path / "cache" / "abc" / "scene_states.json", trace=trace, meta={"video_id": "Х-7"})
    labels = tmp_path / "labels.csv"
    labels.write_text("video_id,file,label,t_sec,md5,group,t_all,conflict,raw\nX-7,X-7.mov,violation,4,0,X-7,4,0,x\n", encoding="utf-8")
    out = tmp_path / "rep"
    assert report.main(["--cache-root", str(tmp_path / "cache"), "--labels", str(labels), "--out", str(out)]) == 0
    assert (out / "x-7.png").stat().st_size > 5000
    summ = json.loads((out / "summary.json").read_text(encoding="utf-8"))["x-7"]
    assert summ["label"] == "violation" and summ["first_predicate"] == "cash_left_after_leave"
    assert summ["onset_within_tol"] is True and summ["type_hint"] == "V02"
    assert "x-7.png" in (out / "index.md").read_text(encoding="utf-8")


def test_ablation_safeguards_remove_false_positives():
    res = ablation.run({k: v for k, v in ablation.VARIANTS.items() if k == "hmm" or k.startswith("naive (")})
    full = res["hmm"]
    assert full["fp_cases"] == 0 and full["fn_cases"] == 0
    naive = next(v for k, v in res.items() if k.startswith("naive"))
    assert naive["fp_cases"] >= 2, "без защит unknown/ролей/эталона появляются ложные срабатывания"
