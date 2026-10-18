"""Команда замера для L2: варианты флагов на сохранённых входах fusion (синтетика)."""
from __future__ import annotations

import json

from ml.fusion import fuse
from ml.fusion.ab import load_variants, main

from .conftest import FakeClip, ev


def test_variants_file_parses_and_has_baseline():
    v = load_variants("ml/fusion/experiments.yaml")
    assert "baseline" in v and v["v02_after_window"]["policies"]["v02_after_window"]["enabled"] is True


def test_ab_end_to_end(tmp_path, catalog, monkeypatch):
    monkeypatch.setenv("SENTINEL_FUSION_NO_FRAMES", "1")
    preds = tmp_path / "preds"
    preds.mkdir()
    rows = ["video,label,second"]
    for i in range(8):
        positive = i % 2 == 0
        extra = {"state_after": {"cash_visible": 0.0, "cash_bag_visible": 0.0, "safe_open": 0.0}}
        kw = {} if positive else {"extra": extra}
        events = [ev("V02", t=10.0 + i, conf=0.7, facts=["кассир вышел из зоны рабочего места"], **kw)]
        res = fuse(FakeClip(duration=30.0), events, [], catalog, f"j{i}")
        res.meta["timings"] = {"total": 12.5 + i}
        (preds / f"X-{i}.json").write_text(json.dumps({"meta": res.meta}, ensure_ascii=False), encoding="utf-8")
        rows.append(f"X-{i},{'нарушение' if positive else 'без нарушения'},{10 + i if positive else ''}")
    labels = tmp_path / "labels.csv"
    labels.write_text("\n".join(rows) + "\n", encoding="utf-8")
    out = tmp_path / "ab.json"
    assert main(["--preds", str(preds), "--labels", str(labels), "--out", str(out), "--cv", "2",
                 "--only", "v02_after_window"]) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    base, v02 = rep["variants"]["baseline"], rep["variants"]["v02_after_window"]
    assert base["fixed_thresholds"]["fp"] == 4                 # у чистых «ушла, но всё убрано» → ложные
    assert v02["fixed_thresholds"]["fp"] == 0 and v02["fixed_thresholds"]["fn"] == 0
    assert v02["delta_vs_baseline"]["fp"] == -4
    assert rep["pipeline"]["latency_s"]["max"] == 19.5 and "oof" in v02
    assert (tmp_path / "ab.md").exists()
