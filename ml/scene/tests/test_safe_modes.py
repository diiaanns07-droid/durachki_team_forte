"""Режимы сейфа фазы 2 (флаги, по умолчанию выкл): kmeans2 по всему ролику и open-vocab именователь.

Именователь здесь — МОК (только в тестах): «открыто», если в центре кропа тёмный проём."""
from __future__ import annotations

import numpy as np

from ml.scene import namer as namer_mod
from ml.scene.core import scene_states_with_trace
from ml.scene.tests.synth import make_clip, zones_default

KM = {"safe": {"cluster_mode": "kmeans2"}}


def series(st):
    t = np.array([s.t for s in st])
    return t, np.array([s.signals["safe_open"] for s in st]), np.array([s.status["safe_open"] for s in st])


def test_kmeans2_opening_and_open_from_start():
    st, tr = scene_states_with_trace(make_clip(10.0, safe_open=lambda t: t >= 5.0), [], [], zones_default(), [], config=KM)
    t, p, s = series(st)
    assert tr["info"]["safe_open"]["cluster_mode"] == "kmeans2" and tr["info"]["safe_open"]["two_states"]
    assert (p[t < 4.8] < 0.5).all() and (p[t >= 5.4] >= 0.5).all()
    st2, _ = scene_states_with_trace(make_clip(10.0, safe_open=lambda t: t < 5.0), [], [], zones_default(), [], config=KM)
    t2, p2, _ = series(st2)
    assert (p2[t2 < 4.6] >= 0.5).all() and (p2[t2 > 5.6] < 0.5).all()


def test_kmeans2_static_safe_stays_unknown():
    st, tr = scene_states_with_trace(make_clip(8.0), [], [], zones_default(), [], config=KM)
    _, p, s = series(st)
    assert not tr["info"]["safe_open"]["two_states"]
    assert set(s) == {"unknown"} and (p < 0.5).all()


def _mock_namer(crops, obj):  # МОК для теста: тёмный центр кропа = «открыто»
    out = []
    for c in crops:
        h, w = c.shape[:2]
        center = c[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean()
        out.append(0.9 if center < 70 else 0.1)
    return out


def test_namer_names_static_safe(monkeypatch):
    monkeypatch.setattr(namer_mod, "get_namer", lambda cfg, backend: _mock_namer)
    cfg = {"safe": {"namer": {"backend": "siglip2"}}}
    st, tr = scene_states_with_trace(make_clip(8.0), [], [], zones_default(), [], config=cfg)
    _, p, s = series(st)
    assert tr["info"]["safe_open"]["namer_p_a"] < 0.5
    assert (s[5:] == "false").all(), "статичный закрытый сейф назван именователем → известно «закрыт»"
    st2, tr2 = scene_states_with_trace(make_clip(8.0, safe_open=lambda t: True), [], [], zones_default(), [], config=cfg)
    _, p2, s2 = series(st2)
    assert (s2[5:] == "true").all()


def test_namer_unavailable_falls_back_without_mock(monkeypatch):
    monkeypatch.setattr(namer_mod, "_STATE", {})
    monkeypatch.delenv("SENTINEL_MOCK", raising=False)
    cfg = {"safe": {"namer": {"backend": "siglip2"}}, "namer": {"cache_dir": "/nonexistent/hf"}}
    st, tr = scene_states_with_trace(make_clip(6.0, safe_open=lambda t: t >= 3.0), [], [], zones_default(), [], config=cfg)
    assert any("namer" in w for w in tr["warnings"])
    t, p, _ = series(st)
    assert (p[t >= 3.6] >= 0.5).all(), "без именователя работает прежняя логика"
    monkeypatch.setenv("SENTINEL_MOCK", "1")
    assert namer_mod.get_namer({}, "siglip2") is None, "в mock-режиме CI именователь выключен, а не подменён"
