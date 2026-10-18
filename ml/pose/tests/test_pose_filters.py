"""One-Euro, гистерезис, конфиг (без весов и GPU)."""
from __future__ import annotations

import numpy as np

from ml.pose.config import load_config
from ml.pose.filters import OneEuro, gated_values, hysteresis_mask, ramp, smooth_keypoint_track


def test_one_euro_reduces_jitter_and_follows_step():
    rng = np.random.default_rng(0)
    ts = np.arange(0, 20, 0.2)
    x = np.where(ts < 10, 0.3, 0.6) + rng.normal(0, 0.01, len(ts))
    f = OneEuro(1.0, 4.0, 1.0)
    y = np.array([f(t, np.array([v]))[0] for t, v in zip(ts, x)])
    assert np.std(y[10:45] - 0.3) < np.std(x[10:45] - 0.3)
    assert abs(y[-1] - 0.6) < 0.02


def test_smoothing_holds_low_confidence_points():
    ts = np.arange(0, 2.0, 0.2)
    kps = np.zeros((len(ts), 17, 3), np.float32)
    kps[:, :, 0] = 0.5
    kps[:, :, 1] = 0.5
    kps[:, :, 2] = 0.9
    kps[5, 9] = (0.95, 0.05, 0.05)  # выброс с низкой уверенностью
    out = smooth_keypoint_track(ts, kps, {"min_score": 0.3, "hold_s": 0.6, "one_euro": {}})
    assert abs(out[5, 9, 0] - 0.5) < 1e-6 and abs(out[5, 9, 1] - 0.5) < 1e-6
    assert out[5, 9, 2] == np.float32(0.05)  # score остаётся низким — потребители видят unknown


def test_hysteresis_min_duration_and_gap():
    ts = np.arange(0, 10, 0.2)
    ev = np.zeros(len(ts))
    ev[(ts >= 1) & (ts < 1.6)] = 0.9            # 0.6 с — короче min_duration
    ev[(ts >= 4) & (ts < 5)] = 0.9
    ev[(ts >= 5.4) & (ts < 6.4)] = 0.9         # разрыв 0.4 с → склейка
    m = hysteresis_mask(ts, ev, 0.6, 0.4, 1.5, 0.6)
    assert not m[(ts >= 1) & (ts < 1.6)].any()
    assert m[(ts >= 4) & (ts < 6.4)].all()


def test_hysteresis_grows_by_off_threshold():
    ts = np.arange(0, 4, 0.2)
    ev = np.full(len(ts), 0.45)
    ev[10] = 0.7
    m = hysteresis_mask(ts, ev, 0.6, 0.4, 0.0, 0.0)
    assert m.all()


def test_gated_values_threshold_semantics():
    ts = np.arange(0, 6, 0.2)
    ev = np.where((ts > 2) & (ts < 4.5), 0.8, 0.3)
    vals, mask = gated_values(ts, ev, {"thr_on": 0.6, "thr_off": 0.4, "min_duration_s": 1.0, "max_gap_s": 0.2})
    assert (vals[mask] >= 0.5).all() and (vals[~mask] < 0.5).all()
    assert mask.any() and (~mask).any()


def test_ramp():
    assert ramp(0.3, 0.3, 0.7) == 1.0 and ramp(0.7, 0.3, 0.7) == 0.0
    assert abs(ramp(0.5, 0.3, 0.7) - 0.5) < 1e-9
    assert ramp(-1.0, -0.3, 0.2) == 1.0


def _walk(d, path=""):
    for k, v in d.items():
        yield path + str(k), k
        if isinstance(v, dict):
            yield from _walk(v, path + str(k) + ".")


def test_config_has_no_yaml_boolean_keys():
    cfg = load_config()
    bad = [p for p, k in _walk(cfg) if not isinstance(k, str)]
    assert not bad, f"YAML-ключи стали не-строками (on/off/yes/no?): {bad}"
    for name in ("phone_to_ear", "phone_in_hand", "eating_drinking", "cashier_facing_away", "handover_motion"):
        p = cfg["signals"][name]
        assert p["thr_on"] > p["thr_off"]


def test_config_overrides_merge():
    cfg = load_config({"signals": {"phone_to_ear": {"min_duration_s": 3.0}}})
    assert cfg["signals"]["phone_to_ear"]["min_duration_s"] == 3.0
    assert cfg["signals"]["phone_to_ear"]["near"] == load_config()["signals"]["phone_to_ear"]["near"]
