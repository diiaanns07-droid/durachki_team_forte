"""Метрики уровня видео и таймкода (без sklearn)."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    """AUC через ранги (Манна–Уитни), ничьи — половина. None, если один класс."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def recall_at_fp(labels: Sequence[int], scores: Sequence[float], max_fp: int) -> tuple[float, float]:
    """Максимальный recall при ≤ max_fp ложных срабатываниях; возвращает (recall, порог «строго больше»)."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    neg = np.sort(s[y == 0])[::-1]
    pos = s[y == 1]
    if len(pos) == 0:
        return 0.0, float("nan")
    thr = float(neg[max_fp]) if len(neg) > max_fp else -np.inf
    return float((pos > thr).mean()), thr


def timecode_errors(pred_t: Sequence[Optional[float]], true_times: Sequence[Sequence[float]]) -> list[float]:
    """|Δt| до ближайшей размеченной секунды (для дубликатов — несколько секунд)."""
    out = []
    for p, ts in zip(pred_t, true_times):
        if p is None or not ts:
            out.append(float("inf"))
        else:
            out.append(float(min(abs(p - t) for t in ts)))
    return out


def hit_rates(errors: Iterable[float], tolerances: Sequence[float]) -> dict[str, float]:
    e = np.asarray(list(errors), dtype=float)
    if len(e) == 0:
        return {f"hit@{t:g}s": 0.0 for t in tolerances}
    return {f"hit@{t:g}s": float((e <= t).mean()) for t in tolerances}


def summarize(labels: Sequence[int], scores: Sequence[float], pred_t: Sequence[Optional[float]],
              true_times: Sequence[Sequence[float]], tolerances: Sequence[float] = (1, 2, 3)) -> dict:
    y = np.asarray(labels, dtype=int)
    r0, thr0 = recall_at_fp(y, scores, 0)
    r1, thr1 = recall_at_fp(y, scores, 1)
    pos_idx = [i for i, v in enumerate(y) if v == 1]
    errs = timecode_errors([pred_t[i] for i in pos_idx], [true_times[i] for i in pos_idx])
    finite = [e for e in errs if np.isfinite(e)]
    return {
        "n_videos": int(len(y)), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
        "auc": roc_auc(y, scores),
        "recall_at_fp0": r0, "thr_fp0": thr0,
        "recall_at_fp1": r1, "thr_fp1": thr1,
        "dt_mean": float(np.mean(finite)) if finite else None,
        "dt_median": float(np.median(finite)) if finite else None,
        **hit_rates(errs, tolerances),
    }


def recommend_threshold(labels: Sequence[int], scores: Sequence[float], margin: float = 0.02,
                        min_recall_fp0: float = 0.8) -> float:
    """Порог для meta.json: при FP=0, если recall там ≥ min_recall_fp0, иначе при FP≤1. В [0.05, 0.95]."""
    r0, thr0 = recall_at_fp(labels, scores, 0)
    r1, thr1 = recall_at_fp(labels, scores, 1)
    thr = thr0 if r0 >= min_recall_fp0 or not np.isfinite(thr1) else thr1
    if not np.isfinite(thr):
        thr = 0.5
    return float(np.clip(thr + margin, 0.05, 0.95))


def spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Ранговая корреляция Спирмена (ничьи — средний ранг). None, если < 3 точек или константа."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if len(x) < 3 or len(x) != len(y):
        return None

    def rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v), dtype=float)
        for val in np.unique(v):  # средний ранг для ничьих
            m = v == val
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r

    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])
