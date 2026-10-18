"""Сгруппированные сплиты (дубликаты по md5 / списку — одна группа) и проверка отсутствия утечек."""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np

from .dataset import Sample


def grouped_folds(samples: Sequence[Sample], folds: Union[str, int] = "loo",
                  seed: int = 0) -> list[tuple[list[int], list[int]]]:
    """[(train_idx, val_idx)] по группам; стратификация по метке группы. 'loo' — одна группа на фолд."""
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        groups.setdefault(s.group or s.name, []).append(i)
    keys = sorted(groups)
    if folds == "loo" or (isinstance(folds, int) and folds >= len(keys)):
        assign = {g: k for k, g in enumerate(keys)}
        n = len(keys)
    else:
        n = max(2, int(folds))
        rng = np.random.default_rng(seed)
        assign = {}
        for lab in (1, 0):  # раскладываем позитивные и негативные группы по фолдам по очереди
            gs = [g for g in keys if max(samples[i].label for i in groups[g]) == lab]
            rng.shuffle(gs)
            off = int(rng.integers(0, n))
            for j, g in enumerate(gs):
                assign[g] = (j + off) % n
    out = []
    for k in range(n):
        val = [i for g in keys if assign[g] == k for i in groups[g]]
        if not val:
            continue
        tr = [i for g in keys if assign[g] != k for i in groups[g]]
        out.append((tr, val))
    return out


def holdout_split(samples: Sequence[Sample], frac: float, seed: int = 0) -> tuple[list[int], list[int]]:
    """Отложенные группы (сгруппированно, стратифицированно): (rest_idx, holdout_idx). На уже исследованных
    роликах это не независимый holdout — только разведочная проверка процедуры подбора."""
    if frac <= 0:
        return list(range(len(samples))), []
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        groups.setdefault(s.group or s.name, []).append(i)
    rng = np.random.default_rng(seed + 7919)
    hold: list[int] = []
    for lab in (1, 0):
        gs = sorted(g for g in groups if max(samples[i].label for i in groups[g]) == lab)
        rng.shuffle(gs)
        k = int(round(frac * len(gs)))
        k = min(max(k, 1 if len(gs) >= 3 else 0), max(len(gs) - 1, 0))
        for g in gs[:k]:
            hold += groups[g]
    rest = [i for i in range(len(samples)) if i not in set(hold)]
    return rest, sorted(hold)


def check_split(samples: Sequence[Sample], train_idx: Sequence[int], val_idx: Sequence[int]) -> None:
    """AssertionError при утечке: общая группа / общий md5 / пересечение индексов между train и val."""
    tr, va = set(train_idx), set(val_idx)
    assert not (tr & va), "индекс в train и val одновременно"
    g_tr = {samples[i].group or samples[i].name for i in tr}
    g_va = {samples[i].group or samples[i].name for i in va}
    assert not (g_tr & g_va), f"группа дубликатов в train и val: {sorted(g_tr & g_va)}"
    m_tr = {samples[i].name for i in tr} | {getattr(samples[i], 'md5', '') for i in tr}
    m_va = {getattr(samples[i], 'md5', '') for i in va} - {""}
    assert not (m_tr & m_va), "одинаковый md5 в train и val"
