from __future__ import annotations

from ml.temporal import cv as cvmod

from .conftest import make_bank, make_samples


def test_nested_cv_baseline_report(cfg):
    samples, bank = make_samples(n_pos=6, n_neg=6, T=80), make_bank()
    samples[1].group = samples[0].group                    # дубликат в одной группе
    rep = cvmod.nested_cv(samples, bank, cfg, "baseline", outer=4, inner=2, seed=0)
    assert rep["n_samples"] == 12 and rep["n_folds"] == 4
    f, n = rep["threshold_free"], rep["nested"]
    assert 0.0 <= f["auc"] <= 1.0 and f["auc"] >= 0.8
    assert n["tp"] + n["fp"] + n["fn"] + n["tn"] == 12
    assert {"hit@1s", "hit@2s", "hit@3s", "dt_median"} <= set(f)
    assert all(0.05 <= v["thr_inner"] <= 0.95 for v in rep["per_video"])
    assert set(rep["by_pooling"]) == {"max", "topk", "topk_null"}
    assert set(rep["candidates"]) == {"risk", "change", "hybrid"}
    assert {"recall@3", "recall@5", "recall@8", "mean_n"} <= set(rep["candidates"]["hybrid"])
    assert rep["candidates"]["hybrid"]["mean_n"] <= 8
    assert {"duration_only_auc", "spearman_neg", "warn"} <= set(rep["duration_watchdog"])
    assert {"peak", "onset", "change_onset"} <= set(rep["timecode_by_policy"])
    assert "разведочная" in rep["evaluation_status"]
    assert all(set(v["scores_by_pooling"]) == {"max", "topk", "topk_null"} for v in rep["per_video"])


def test_run_with_holdout_and_selection(cfg):
    samples, bank = make_samples(n_pos=6, n_neg=6, T=60), make_bank()
    rep = cvmod.run(cfg, ["baseline"], outer=3, inner=2, holdout=0.25, seed=0, samples=samples, bank=bank)
    assert rep["selected"] == "baseline"
    h = rep["holdout"]
    assert set(h["names"]).isdisjoint({v["name"] for v in rep["baseline"]["per_video"]})
    assert len(h["pred"]) == len(h["names"]) >= 2


def test_nested_cv_temporal_smoke(cfg):
    import pytest

    pytest.importorskip("torch")
    samples, bank = make_samples(n_pos=4, n_neg=4, T=60), make_bank()
    c = {**cfg, "train": {**cfg["train"], "steps_per_epoch": 3}}
    rep = cvmod.run(c, ["baseline", "temporal"], outer=2, inner=2, holdout=0.0, seed=0, epochs=2,
                    samples=samples, bank=bank)
    assert rep["selected"] in ("baseline", "temporal")
    assert rep["temporal"]["n_folds"] == rep["baseline"]["n_folds"] == 2   # одинаковые внешние фолды
    assert [v["name"] for v in rep["temporal"]["per_video"]] == [v["name"] for v in rep["baseline"]["per_video"]]


def test_bench_repeated_group_kfold_two_configs(cfg):
    from ml.temporal import bench

    samples, bank = make_samples(n_pos=6, n_neg=6, T=60), make_bank()
    samples[1].group = samples[0].group
    rep = bench.run_bench(cfg, ["E0", "E1"], folds=3, repeats=2, inner=2, seed=0, samples=samples, bank=bank)
    assert set(rep["experiments"]) == {"E0", "E1"} and "разведочная" in rep["evaluation_status"]
    s = rep["experiments"]["E1"]["summary"]
    assert s["n_repeats"] == 2 and s["auc_by_pooling"]["topk_null"]["n"] == 2
    assert {"peak", "onset"} <= set(s["timecode"]) and "recall@8" in s["candidates"]["hybrid"]
    assert rep["type_eval"] is None and "type_note" in rep
    assert "| E0 |" in bench._table(rep)


def test_cv_invariant_to_row_order(cfg):
    """Порядок строк/групп не влияет на результат: фолды строятся по отсортированным id групп."""
    samples, bank = make_samples(n_pos=5, n_neg=5, T=60), make_bank()
    a = cvmod.nested_cv(samples, bank, cfg, "baseline", outer=3, inner=2, seed=0)
    b = cvmod.nested_cv(list(reversed(samples)), bank, cfg, "baseline", outer=3, inner=2, seed=0)
    sa = {v["name"]: v["score"] for v in a["per_video"]}
    sb = {v["name"]: v["score"] for v in b["per_video"]}
    assert sa.keys() == sb.keys() and all(abs(sa[k] - sb[k]) < 1e-3 for k in sa)


def test_bench_temporal_config_smoke(cfg):
    import pytest

    pytest.importorskip("torch")
    from ml.temporal import bench

    samples, bank = make_samples(n_pos=4, n_neg=4, T=60), make_bank()
    c = {**cfg, "train": {**cfg["train"], "steps_per_epoch": 2}}
    rep = bench.run_bench(c, ["E2"], folds=2, repeats=1, inner=2, seed=0, samples=samples, bank=bank, epochs=1)
    s = rep["experiments"]["E2"]["summary"]
    assert rep["experiments"]["E2"]["kind"] == "temporal" and s["n_repeats"] == 1
    assert s["latency_model_ms_per_video"]["mean"] is not None
