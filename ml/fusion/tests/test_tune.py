"""Тюнер порогов: разбор разметки, сплит с дублями в одной части, подбор только на train."""
from __future__ import annotations

import json

import pytest

import yaml

from ml.fusion import fuse
from ml.fusion.tune import (
    Sample,
    evaluate,
    fit,
    main,
    predict,
    normalize_id,
    parse_label,
    parse_seconds,
    read_labels,
    split_groups,
)

from .conftest import FakeClip, ev


def test_parsers():
    assert normalize_id("Х-3.mov") == "X-3" == normalize_id("x_3") == normalize_id("X–3")
    assert parse_label("нарушение") is True and parse_label("без нарушения") is False
    assert parse_label("1") is True and parse_label("no_violation") is False and parse_label("?") is None
    assert parse_seconds("22 секунда") == 22.0 and parse_seconds("1:05") == 65.0 and parse_seconds("") is None


def test_read_labels_semicolon_cyrillic(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("Видео;Разметка;Секунда\nХ-4;нарушение;57\nХ-47;нарушение;68\nX-1;без нарушения;\n",
                 encoding="utf-8")
    labels = read_labels(p)
    assert labels["X-4"][0].positive and labels["X-4"][0].second == 57.0
    assert labels["X-1"][0].positive is False


def test_split_keeps_duplicates_together():
    samples = [Sample(f"V{i}", i % 2 == 0, [], [], f"g{i}") for i in range(20)]
    samples += [Sample("X-4", True, [57.0], [], "dup"), Sample("X-47", True, [68.0], [], "dup")]
    train, hold = split_groups(samples, 0.3, seed=1)
    names_t, names_h = {s.video for s in train}, {s.video for s in hold}
    assert not names_t & names_h
    assert ({"X-4", "X-47"} <= names_t) or ({"X-4", "X-47"} <= names_h)
    assert any(s.positive for s in hold) and any(not s.positive for s in hold)


def test_fit_reaches_target_recall_on_train():
    pos = [Sample(f"P{i}", True, [10.0], [("V01", 0.5 + i * 0.03, 10.0)], f"p{i}") for i in range(6)]
    neg = [Sample(f"N{i}", False, [], [("V01", 0.2 + i * 0.03, 5.0)], f"n{i}") for i in range(6)]
    thr = fit(pos + neg, {"V01": 0.9}, 0.5)
    m = evaluate(pos + neg, thr, 0.5)
    assert m["recall"] == 1.0 and m["fp"] == 0
    assert 0.35 < thr["V01"] <= 0.5


def _write_preds(tmp_path, catalog, n=12, dup=False):
    preds = tmp_path / "preds_raw"
    preds.mkdir()
    rows = ["video,label,second"]
    for i in range(n):
        positive = i % 2 == 0
        events = [ev(t=10.0, conf=0.55 + i * 0.02)] if positive else [ev(t=5.0, conf=0.2 + i * 0.01)]
        res = fuse(FakeClip(duration=30.0), events, [], catalog, f"job{i}")
        name = f"Х-{i}" if i % 3 == 0 else f"X-{i}"
        (preds / f"{name}.json").write_text(json.dumps({"meta": res.meta}, ensure_ascii=False), encoding="utf-8")
        rows.append(f"{name}.mov,{'нарушение' if positive else 'без нарушения'},{'10' if positive else ''}")
    if dup:  # дубль X-0 с другим таймкодом (как Х-4≡Х-47: 57 и 68 с)
        (preds / "X-100.json").write_text((preds / "Х-0.json").read_text(encoding="utf-8"), encoding="utf-8")
        rows.append("X-100.mov,нарушение,25")
    labels = tmp_path / "labels.csv"
    labels.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return preds, labels


def test_cli_end_to_end_cv(tmp_path, catalog, monkeypatch):
    monkeypatch.setenv("SENTINEL_FUSION_NO_FRAMES", "1")
    preds, labels = _write_preds(tmp_path, catalog, dup=True)
    out, report = tmp_path / "thr.yaml", tmp_path / "report.json"
    assert main(["--preds", str(preds), "--labels", str(labels), "--out", str(out), "--report", str(report),
                 "--cv", "3"]) == 0
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert doc["tuned"] is True and doc["tuned_on"]["mode"] == "cv" and doc["tuned_on"]["k"] == 3
    assert "не независимый тест" in doc["tuned_on"]["note"]
    rep = json.loads(report.read_text(encoding="utf-8"))
    assert rep["n_groups"] == 12                       # дубль X-100 ≡ Х-0 — одна группа
    assert rep["cv"]["oof"]["recall"] == 1.0 and len(rep["cv"]["fold_thresholds"]) == 3
    assert any("конфликт таймкодов" in w for w in rep["warnings"])
    assert rep["cv"]["oof"]["timecode_conflicts"] == ["X-0"]


def test_cli_holdout_mode(tmp_path, catalog, monkeypatch):
    monkeypatch.setenv("SENTINEL_FUSION_NO_FRAMES", "1")
    preds, labels = _write_preds(tmp_path, catalog)
    report = tmp_path / "report.json"
    assert main(["--preds", str(preds), "--labels", str(labels), "--mode", "holdout", "--report", str(report)]) == 0
    rep = json.loads(report.read_text(encoding="utf-8"))
    assert rep["mode"] == "holdout" and rep["holdout"]["n"] > 0
    assert not set(rep["train_videos"]) & set(rep["holdout_videos"])


@pytest.mark.parametrize("dt_weight,expect_fp,expect_hit", [(0.5, 0, 0.0), (2.0, 1, 1.0)])
def test_dt_penalty_tradeoff(dt_weight, expect_fp, expect_hit):
    # P ловится либо V01 с верным таймкодом (0.5), либо V03 с неверным (0.8); у чистого N есть V01 0.55.
    # Штраф Δt дешевле FP → отсекаем V01 (промах таймкода); дороже FP → берём верный таймкод ценой FP.
    pos = [Sample("P", True, [10.0], [("V01", 0.5, 10.0, 9.5), ("V03", 0.8, 30.0, 29.0)], "p")]
    neg = [Sample("N", False, [], [("V01", 0.55, 5.0, 5.0)], "n")]
    thr = fit(pos + neg, {"V01": 0.45, "V03": 0.45}, 0.45, dt_weight=dt_weight)
    m = evaluate(pos + neg, thr, 0.45)
    assert m["fn"] == 0 and m["fp"] == expect_fp and m["timecode_hit_rate"] == expect_hit


def test_onset_vs_peak_timecode_metric():
    s = [Sample("P", True, [9.0], [("V01", 0.9, 10.5, 9.2)], "p")]
    assert evaluate(s, {}, 0.0, 1.0)["timecode_hit_rate"] == 0.0
    assert evaluate(s, {}, 0.0, 1.0, policy="onset")["timecode_hit_rate"] == 1.0


def test_type_column_accuracy(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("video,label,second,type\nX-1,нарушение,10,V02\nX-2,без нарушения,,\n", encoding="utf-8")
    labels = read_labels(p)
    assert labels["X-1"][0].type_id == "V02" and labels["X-2"][0].type_id is None
    s = [Sample("X-1", True, [10.0], [("V02", 0.9, 10.0, 10.0)], "g", type_id="V02")]
    assert evaluate(s, {}, 0.0)["type_accuracy"] == 1.0


def test_documented_adapter_recipe_on_a16_rows(catalog, monkeypatch):
    """Рецепт из README («API подбора порогов»): строки eval.run --raw (A16) → Sample → fit/predict."""
    monkeypatch.setenv("SENTINEL_FUSION_NO_FRAMES", "1")
    from ml.fusion.catalog import resolve_catalog
    from ml.fusion.config import load_config, load_thresholds, threshold_for
    from ml.fusion.tune import VideoPred, candidates_for

    rows = []
    for i in range(6):
        positive = i % 2 == 0
        events = [ev(t=10.0 + i, conf=0.6 + 0.03 * i)] if positive else [ev(t=5.0 + i, conf=0.2)]
        res = fuse(FakeClip(duration=30.0), events, [], catalog, f"job{i}")
        rows.append({"video_id": f"X-{i}", "group": f"X-{i}", "label": "violation" if positive else "no_violation",
                     "gt_times": [10.0 + i] if positive else [], "scores_raw": res.meta["scores_raw"]})

    cfg, thr0 = load_config(), load_thresholds()
    default = float(thr0["default"])
    start = {t: threshold_for(thr0, t) for t in thr0["per_type"]}
    types, _ = resolve_catalog(catalog)

    def to_sample(row):
        pred = VideoPred(row["video_id"], row["scores_raw"], fingerprint=row.get("group") or row["video_id"])
        cands, _ = candidates_for(pred, cfg, types)
        return Sample(video=row["video_id"], positive=row["label"] == "violation",
                      seconds=[float(t) for t in row.get("gt_times") or []], cands=cands,
                      group=row.get("group") or row["video_id"])

    train = [to_sample(r) for r in rows[:4]]
    thr = fit(train, start, default, target_recall=1.0, bias=0.35, dt_tol=1.0, dt_weight=0.5)
    assert isinstance(thr, dict) and all(isinstance(v, float) for v in thr.values())
    hits = predict(to_sample(rows[4]), thr, default)          # list[Cand] над порогом
    assert bool(hits) is True and hits[0][0] == "V01" and abs(hits[0][2] - 14.0) <= 0.11
    assert bool(predict(to_sample(rows[5]), thr, default)) is False
