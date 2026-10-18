"""Фаза 2, п.2: признаки по зонам (неизвестная зона = NaN, флаг avail) и признаки изменения («до/после»)."""
from __future__ import annotations

import numpy as np
import pytest

from ml.temporal.config import load_config
from ml.temporal.dataset import fit_spec, prepare_samples
from ml.temporal.features import FeatureSet, MockBackend, extract_clip
from ml.temporal.predictors import BaselineLR

from .conftest import make_bank, make_samples, write_video


def test_zone_views_known_and_unknown(tmp_path):
    pytest.importorskip("cv2")
    from contracts.sentinel_contracts import Zone
    from ml.temporal._video import SimpleClip

    cfg = load_config()
    clip = SimpleClip(str(write_video(tmp_path / "v.avi", seconds=4.0)), sample_fps=4.0)
    zones = [Zone("desk", [(0.1, 0.55), (0.9, 0.55), (0.9, 0.9), (0.1, 0.9)])]
    fs = extract_clip(clip, MockBackend(16), cfg, zones=zones, fps=4.0, views=["full", "zone:desk", "zone:door"])
    assert np.isfinite(fs.emb[:, 0]).all() and np.isfinite(fs.emb[:, 1]).all()
    assert np.isnan(fs.emb[:, 2]).all() and fs.meta["zones_found"] == ["zone:desk"]   # двери нет -> неизвестно


def _with_zone(fs: FeatureSet, known: bool, rng) -> FeatureSet:
    z = rng.standard_normal((len(fs), 1, fs.emb.shape[-1])).astype(np.float32)
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    if not known:
        z[:] = np.nan
    dyn = np.concatenate([fs.dyn, np.full((len(fs), 1), 0.1 if known else np.nan, np.float32)], axis=1)
    return FeatureSet(fs.times, np.concatenate([fs.emb, z], axis=1), dyn, fs.boxes, fs.views + ["zone:desk"], fs.meta)


def test_spec_with_zone_views_and_change_group():
    cfg = load_config({"features": {"zone_views": ["desk"]},
                       "model": {"input_groups": ["sims", "emb", "dyn", "change"]}})
    rng = np.random.default_rng(0)
    samples, bank = make_samples(n_pos=4, n_neg=4, T=80, n_aug=0), make_bank()
    for i, s in enumerate(samples):                     # у половины роликов зоны нет
        s.versions = [_with_zone(s.versions[0], known=i % 2 == 0, rng=rng)]
    spec = fit_spec([fs for s in samples for fs in s.versions], bank, cfg)
    assert spec.views == ["full", "focus", "zone:desk"] and "avail" in spec.groups and "change" in spec.groups
    prepare_samples(samples, bank, spec)
    X = samples[1].X[0]
    assert X.shape[1] == spec.in_dim and np.isfinite(X).all()
    cols = spec.view_columns()
    assert len(cols) == 3 and all(len(c) for c in cols)
    m = BaselineLR(cfg).fit(samples, cfg)
    p, _ = m.predict(X, spec.fps)
    assert p.shape == (len(X),) and np.isfinite(p).all()


def test_change_block_peaks_at_switch():
    cfg = load_config({"model": {"input_groups": ["change"], "emb_pca": 0}})
    T, fps, dim = 120, 4.0, 16
    rng = np.random.default_rng(1)
    times = (np.arange(T) / fps).astype(np.float32)
    a, b = rng.standard_normal(dim), rng.standard_normal(dim)
    emb = np.where((times < 15.0)[:, None, None], a, b) + 0.02 * rng.standard_normal((T, 2, dim))
    emb = (emb / np.linalg.norm(emb, axis=-1, keepdims=True)).astype(np.float32)
    fs = FeatureSet(times, emb, np.zeros((T, 4), np.float32), np.zeros((T, 4), np.float32), ["full", "focus"],
                    {"fps": fps, "backend": "mock"})
    from ml.temporal.dataset import change_block

    spec = fit_spec([fs], make_bank(dim), cfg)
    ch = change_block(fs.times, fs.emb, spec)
    assert ch.shape == (T, 2 * spec.n_change)
    contrast_1s = ch[:, 2]                                # вид 0: [lag2, lag8, contrast1, contrast3]
    assert abs(float(times[int(np.nanargmax(contrast_1s))]) - 15.0) <= 0.5
    assert np.isnan(ch[0, 0])                             # нет истории для лага -> неизвестно
