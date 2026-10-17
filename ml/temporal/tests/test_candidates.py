"""Фаза 2, п.4: генератор кандидатов (3–8 моментов) и метрика recall в ±1.5 с."""
from __future__ import annotations

import numpy as np
import pytest

from ml.temporal.candidates import candidate_recall, change_curve, resample_to, select_candidates
from ml.temporal.config import load_config

from .conftest import write_video


def _switch_emb(T=160, fps=4.0, t_switch=20.0, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    times = (np.arange(T) / fps).astype(np.float32)
    a, b = rng.standard_normal(dim), rng.standard_normal(dim)
    emb = np.where((times < t_switch)[:, None, None], a, b) + 0.05 * rng.standard_normal((T, 2, dim))
    emb /= np.linalg.norm(emb, axis=-1, keepdims=True)
    return times, emb.astype(np.float32)


def test_change_curve_peaks_at_transition():
    cfg = load_config()
    times, emb = _switch_emb()
    c = change_curve(times, emb, None, cfg)
    assert c.shape == times.shape and 0.0 <= c.min() and c.max() <= 1.0
    assert abs(float(times[int(np.argmax(c))]) - 20.0) <= 1.0


def test_select_candidates_modes_spacing_and_budget():
    cfg = load_config()
    t = np.arange(0, 60, 0.5, dtype=np.float32)
    rng = np.random.default_rng(1)
    risk = rng.random(len(t)) * 0.2
    risk[np.abs(t - 30) < 1] = 0.9
    change = rng.random(len(t))
    for mode in ("risk", "change", "hybrid"):
        cands = select_candidates(t, risk, change, cfg, mode=mode)
        assert 3 <= len(cands) <= 8
        ts = sorted(c["t"] for c in cands)
        assert all(b - a >= cfg["propose"]["nms_s"] - 1e-6 for a, b in zip(ts, ts[1:]))
    hyb = select_candidates(t, risk, change, cfg, mode="hybrid")
    assert hyb[0]["source"] == "risk" and abs(hyb[0]["t"] - 30) <= 1.0
    assert all(c["salience"] <= cfg["propose"]["change_salience_cap"] + 1e-9 for c in hyb if c["source"] == "change")
    only_change = select_candidates(t, None, change, cfg, mode="hybrid")       # без чекпоинта
    assert only_change and all(c["source"] == "change" for c in only_change)
    assert select_candidates(t, None, None, cfg) == []


def test_candidate_recall_metric():
    c = [[{"t": 10.0}, {"t": 30.0}, {"t": 50.0}], [{"t": 5.0}], [{"t": 1.0}]]
    r = candidate_recall(c, [[31.2], [9.0], []], budgets=(1, 3), tol=1.5)
    assert r["recall@1"] == 0.0 and r["recall@3"] == 0.5 and r["n_pos"] == 2


def test_resample_to_grid():
    t_src = np.arange(0, 10, 0.25)
    x = np.zeros(len(t_src)); x[20] = 1.0
    out = resample_to(t_src, x, np.arange(0, 10.01, 0.5))
    assert out.shape == (21,) and abs(float(np.arange(0, 10.01, 0.5)[int(np.argmax(out))]) - 5.0) < 1e-6


def test_propose_without_checkpoint_mock(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    monkeypatch.setenv("SENTINEL_MOCK", "1")              # только тест: явный mock-режим
    import ml.temporal as temporal
    from ml.temporal._video import SimpleClip
    from ml.temporal.infer import propose_status

    clip = SimpleClip(str(write_video(tmp_path / "v.avi", seconds=12.0, event=(5.0, 7.0))), sample_fps=5.0)
    cands = temporal.propose(clip)
    assert 3 <= len(cands) <= 8 and propose_status()["checkpoint"] is False
    for e in cands:
        assert e.extra["role"] == "candidate" and e.extra["candidate_source"] == "change"
        assert e.type_id == "V99" and e.confidence <= 0.3 + 1e-9 and e.t_start <= e.t_peak <= e.t_end
    with pytest.raises(temporal.TemporalUnavailable):     # а вердиктного скора без чекпоинта нет — и это не «чисто»
        temporal.score(clip)
    assert propose_status()["risk_used"] is False
