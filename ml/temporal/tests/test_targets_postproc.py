from __future__ import annotations

import numpy as np

from ml.temporal.dataset import frame_targets, model_grid
from ml.temporal.postproc import find_peaks, risk_curve


def test_positive_far_region_is_masked_not_negative():
    g = model_grid(30.0, 4.0)
    y, w = frame_targets(g, 1, [10.0], sigma=1.5, far_radius=3.0, far_weight=0.0)
    assert abs(g[np.argmax(y)] - 10.0) < 1e-6 and y.max() == 1.0
    assert np.all(w[np.abs(g - 10.0) > 3.0] == 0.0)        # вне окрестности метки — неизвестно (маска)
    assert np.all(w[np.abs(g - 10.0) <= 3.0] == 1.0)


def test_negative_video_full_supervision_and_multi_anchor():
    g = model_grid(20.0, 4.0)
    y, w = frame_targets(g, 0, [], 1.5, 3.0, 0.0)
    assert y.sum() == 0 and np.all(w == 1)
    y2, _ = frame_targets(g, 1, [5.0, 15.0], 1.5, 3.0, 0.0)
    assert y2[int(5 * 4)] == 1.0 and y2[int(15 * 4)] == 1.0


def test_risk_curve_step_and_peaks(cfg):
    grid = model_grid(20.0, 4.0)
    p = np.zeros(len(grid), np.float32)
    p[np.abs(grid - 6.0) < 1.0] = 0.9
    p[np.abs(grid - 15.0) < 1.0] = 0.7
    t, r = risk_curve(p, grid, cfg, duration=20.0)
    assert np.allclose(np.diff(t), 0.5) and t[-1] == 20.0 and r.min() >= 0 and r.max() <= 1
    pk = find_peaks(t, r, 0.5, 6.0, 3)
    assert [round(x["t_peak"]) for x in pk] == [6, 15]
    assert all(x["t_start"] <= x["t_peak"] <= x["t_end"] for x in pk)
    assert find_peaks(t, r, 0.95, 6.0, 3) == []
