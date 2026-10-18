from __future__ import annotations

import numpy as np
import pytest

from ml.temporal.dataset import collate, fit_spec, prepare_samples, sample_window
from ml.temporal.metrics import roc_auc
from ml.temporal.postproc import risk_curve, video_score
from ml.temporal.predictors import BaselineLR, fit_predictor, load_checkpoint, save_checkpoint


def _prep(toy, cfg):
    samples, bank = toy
    spec = fit_spec([fs for s in samples for fs in s.versions], bank, cfg)
    prepare_samples(samples, bank, spec)
    return samples, bank, spec


def test_input_spec_shapes_and_view_columns(toy, cfg):
    samples, bank, spec = _prep(toy, cfg)
    X = samples[0].X[0]
    assert X.shape[1] == spec.in_dim and np.isfinite(X).all()
    cols = spec.view_columns()
    assert len(cols) == 2 and set(cols[0]).isdisjoint(cols[1]) and max(map(max, cols)) < spec.in_dim
    it = sample_window(samples[0], cfg, np.random.default_rng(0), use_aug=True, view_cols=cols)
    b = collate([it, sample_window(samples[-1], cfg, np.random.default_rng(1), use_aug=False)])
    assert b["X"].ndim == 3 and b["mask"].shape == b["y"].shape


def test_missing_view_is_zero_filled(toy, cfg):
    samples, bank, spec = _prep(toy, cfg)
    fs = samples[0].versions[0]
    from ml.temporal.dataset import build_inputs
    from ml.temporal.features import FeatureSet

    only_full = FeatureSet(fs.times, fs.emb[:, :1], np.concatenate([fs.dyn[:, :3]], 1), fs.boxes, ["full"], fs.meta)
    X, _ = build_inputs(only_full, bank, spec)
    assert np.isfinite(X).all() and np.all(X[:, spec.view_columns()[1]] == 0.0)


def _video_auc(model, samples, spec, cfg):
    sc = []
    for s in samples:
        p, _ = model.predict(s.X[0], spec.fps)
        _, r = risk_curve(p, s.grid[0], cfg)
        sc.append(video_score(r))
    return roc_auc([s.label for s in samples], sc), sc


def test_baseline_converges_on_toy(toy, cfg):
    samples, bank, spec = _prep(toy, cfg)
    m = BaselineLR(cfg).fit(samples, cfg)
    auc, _ = _video_auc(m, samples, spec, cfg)
    assert auc is not None and auc >= 0.9
    p, tp = m.predict(samples[0].X[0], spec.fps)
    assert p.shape == (len(samples[0].X[0]),) and tp is None and not m.type_trained   # нет type_id -> нет головы


def test_baseline_type_head_only_with_verified_types(toy, cfg):
    samples, bank, spec = _prep(toy, cfg)
    for i, s in enumerate(samples):
        if s.label:
            s.type_id = "V05" if i % 2 else "V01"
    m = BaselineLR(cfg).fit(samples, cfg)
    _, tp = m.predict(samples[0].X[0], spec.fps)
    assert m.type_trained and tp.shape[1] == 9 and np.allclose(tp.sum(1), 1, atol=1e-4)


def test_checkpoint_roundtrip(toy, cfg, tmp_path):
    samples, bank, spec = _prep(toy, cfg)
    m = fit_predictor("baseline", samples, spec, cfg)
    save_checkpoint(tmp_path / "w", m, spec, {"threshold": 0.4})
    m2, spec2, meta = load_checkpoint(tmp_path / "w", cfg)
    assert meta["kind"] == "baseline" and meta["threshold"] == 0.4 and meta["type_head_trained"] is False
    a, _ = m.predict(samples[1].X[0], spec.fps)
    b, _ = m2.predict(samples[1].X[0], spec2.fps)
    assert np.allclose(a, b, atol=1e-6)


@pytest.mark.slow
def test_temporal_head_shapes_and_learning(toy, cfg, tmp_path):
    torch = pytest.importorskip("torch")
    samples, bank, spec = _prep(toy, cfg)
    from ml.temporal.model import build_model

    gru_cfg = {**cfg, "model": {**cfg["model"], "arch": "tcn_gru"}}
    net = build_model(spec.in_dim, gru_cfg)               # вариант с BiGRU и упакованными последовательностями
    x = torch.randn(3, 50, spec.in_dim)
    mask = torch.ones(3, 50)
    mask[1, 30:] = 0
    lg, tl = net(x, mask)
    assert lg.shape == (3, 50) and tl.shape == (3, 50, 9)
    cfg2 = {**cfg, "train": {**cfg["train"], "epochs": 20, "steps_per_epoch": 8, "mixup_prob": 0.0}}
    m = fit_predictor("temporal", samples, spec, cfg2, seed=0)
    auc, _ = _video_auc(m, samples, spec, cfg2)
    assert auc is not None and auc >= 0.8
    save_checkpoint(tmp_path / "w", m, spec, {"threshold": 0.5})
    m2, spec2, meta = load_checkpoint(tmp_path / "w", cfg2, kind="temporal")
    a, _ = m.predict(samples[0].X[0], spec.fps)
    b, _ = m2.predict(samples[0].X[0], spec2.fps)
    assert np.allclose(a, b, atol=1e-5) and meta["type_head_trained"] is False


def test_pseudo_types_ignored_unless_flag(toy, cfg):
    from ml.temporal.predictors import type_head_source

    samples, bank, spec = _prep(toy, cfg)
    for s in samples:
        if s.label:
            s.pseudo_type_id = "V07"
    assert not BaselineLR(cfg).fit(samples, cfg).type_trained and type_head_source(samples, cfg) is None
    cfg2 = {**cfg, "train": {**cfg["train"], "use_pseudo_types": True}}
    m = BaselineLR(cfg2).fit(samples, cfg2)
    assert m.type_trained and type_head_source(samples, cfg2) == "pseudo"
