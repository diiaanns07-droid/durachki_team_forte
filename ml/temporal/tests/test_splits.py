from __future__ import annotations

import pytest

from ml.temporal import cv as cvmod
from ml.temporal.splits import check_split, grouped_folds, holdout_split

from .conftest import make_samples


def _dup_samples():
    s = make_samples(n_pos=4, n_neg=4, T=60, n_aug=1)
    s[1].group = s[0].group      # дубликат: X-2 в группе X-1
    s[5].group = s[4].group
    return s


@pytest.mark.parametrize("folds", ["loo", 3])
def test_grouped_folds_keep_duplicates_together(folds):
    s = _dup_samples()
    fl = grouped_folds(s, folds, seed=1)
    seen = []
    for tr, va in fl:
        check_split(s, tr, va)
        assert {s[i].group for i in tr}.isdisjoint({s[i].group for i in va})
        seen += va
    assert sorted(seen) == list(range(len(s)))              # каждая запись в валидации ровно один раз
    if folds == "loo":
        assert len(fl) == len({x.group for x in s})


def test_check_split_detects_leak():
    s = _dup_samples()
    with pytest.raises(AssertionError):
        check_split(s, [0, 2, 3], [1])                     # X-2 — дубликат X-1
    s2 = make_samples(n_pos=2, n_neg=2, T=40)
    s2[3].md5 = s2[0].md5
    with pytest.raises(AssertionError):
        check_split(s2, [0, 1], [3])


def test_holdout_grouped_stratified():
    s = make_samples(n_pos=6, n_neg=6, T=40)
    rest, hold = holdout_split(s, 0.25, seed=0)
    check_split(s, rest, hold)
    assert {s[i].label for i in hold} == {0, 1}
    assert sorted(rest + hold) == list(range(len(s)))


def test_validation_uses_only_original_version(monkeypatch, cfg):
    s = make_samples(n_pos=3, n_neg=3, T=60, n_aug=2)
    bank = __import__("ml.temporal.tests.conftest", fromlist=["make_bank"]).make_bank()
    seen = {}
    real = cvmod.prepare_samples

    def spy(samples, b, spec):
        for x in samples:
            seen.setdefault(x.name, []).append(len(x.versions))
        return real(samples, b, spec)

    monkeypatch.setattr(cvmod, "prepare_samples", spy)
    cvmod.fit_and_score("baseline", s[:2] + s[3:5], [s[2], s[5]], bank, cfg, seed=0)
    assert seen["X-3"] == [1] and seen["X-6"] == [1]       # валидация — только оригинал
    assert seen["X-1"] == [3]                               # train — оригинал + аугментации


def test_check_split_verify_report():
    from ml.temporal.check_split import verify

    s = _dup_samples()
    s[0].sources = [{"name": "X-1", "label": 1, "times": [5.0]}, {"name": "X-2", "label": 1, "times": [9.0]}]
    rep = verify(s, ["loo", "3"], holdout=0.25)
    assert rep["ok"] and rep["folds"]["loo"]["n_folds"] == len({x.group for x in s})
    assert rep["duplicate_groups"] == {"X-1": ["X-1", "X-2"]}
