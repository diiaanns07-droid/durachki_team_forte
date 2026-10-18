"""Пункт 1 фазы 2: длинные (чаще чистые) ролики не должны получать преимущество ни в обучении, ни в пулинге."""
from __future__ import annotations

import numpy as np

from ml.temporal.config import load_config
from ml.temporal.dataset import build_inputs, fit_spec, prepare_samples
from ml.temporal.features import FeatureSet
from ml.temporal.metrics import spearman
from ml.temporal.postproc import block_stats, pool_score
from ml.temporal.predictors import BaselineLR

from .conftest import make_bank, make_fs
from ml.temporal.dataset import Sample


def _samples_long_neg():
    out = []
    for i in range(4):
        out.append(Sample(name=f"P{i}", group=f"P{i}", label=1, times=[6.0],
                          versions=[make_fs(60, 1, 6.0, seed=i), make_fs(60, 1, 6.0, seed=100 + i)]))
    for i, T in enumerate([80, 200, 400, 800]):          # чистые ролики разной (большой) длины
        out.append(Sample(name=f"N{i}", group=f"N{i}", label=0, times=[], versions=[make_fs(T, 0, None, seed=50 + i)]))
    return out


def test_group_weighting_each_group_weighs_the_same():
    cfg = load_config()
    samples, bank = _samples_long_neg(), make_bank()
    spec = fit_spec([fs for s in samples for fs in s.versions], bank, cfg)
    prepare_samples(samples, bank, spec)
    _, _, wt, _, gg = BaselineLR(cfg).gather(samples, cfg)
    per_group = np.bincount(gg, weights=wt, minlength=len(samples))
    assert np.allclose(per_group[:4], per_group[0]) and np.allclose(per_group[4:], per_group[4])  # длина не важна
    assert np.isclose(per_group[:4].sum(), per_group[4:].sum())                                   # классы поровну
    cfg_old = load_config({"train": {"weighting": "frame"}})
    _, _, wt2, _, gg2 = BaselineLR(cfg_old).gather(samples, cfg_old)
    pg2 = np.bincount(gg2, weights=wt2, minlength=len(samples))
    assert pg2[7] > 5 * pg2[4]                                     # старый режим: длинный ролик доминирует


def test_pooling_watchdog_null_removes_length_bias():
    """Чистые ролики = шум одинаковой природы, разной длины: max растёт с длиной, topk_null — нет."""
    rng = np.random.default_rng(0)
    cfg = load_config()

    def curve(n):  # сглаженный шум ~ риск модели на чистом ролике
        x = rng.beta(2, 8, size=n)
        return np.convolve(x, np.ones(3) / 3, mode="same")

    lens = rng.integers(20, 600, size=120)
    train_null = np.concatenate([block_stats(curve(int(n)), cfg) for n in rng.integers(20, 600, size=40)])
    test = [curve(int(n)) for n in lens]
    s_max = [pool_score(c, {**cfg, "infer": {**cfg["infer"], "pooling": "max"}}) for c in test]
    s_topk = [pool_score(c, {**cfg, "infer": {**cfg["infer"], "pooling": "topk"}}) for c in test]
    s_null = [pool_score(c, cfg, train_null) for c in test]
    assert spearman(lens, s_max) > 0.3 and spearman(lens, s_topk) > 0.3   # без нормировки длина протекает
    assert abs(spearman(lens, s_null)) < 0.2                              # topk_null — сторож проходит


def test_risk_is_local_not_dependent_on_total_length():
    cfg = load_config()
    samples, bank = _samples_long_neg(), make_bank()
    spec = fit_spec([fs for s in samples for fs in s.versions], bank, cfg)
    prepare_samples(samples, bank, spec)
    m = BaselineLR(cfg).fit(samples, cfg)
    fs = samples[-1].versions[0]
    cut = 300
    pref = FeatureSet(fs.times[:cut], fs.emb[:cut], fs.dyn[:cut], fs.boxes[:cut], fs.views, fs.meta)
    Xa, _ = build_inputs(fs, bank, spec)
    Xb, _ = build_inputs(pref, bank, spec)
    pa, _ = m.predict(Xa, spec.fps)
    pb, _ = m.predict(Xb, spec.fps)
    k = cut - int(3 * spec.fps)                                     # вдали от конца префикса
    assert np.allclose(pa[:k], pb[:k], atol=1e-6)
