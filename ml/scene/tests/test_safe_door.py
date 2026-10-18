"""Сейф/дверь: открытие, «открыт с начала», окклюзия, освещение, низкое качество, ракурс/дрожание."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from ml.scene.tests.synth import SAFE, H, W, affine_zones, make_clip, person_track, person_dets, zones_default
from ml.scene._contract import Detection
from ml.scene.core import scene_states_with_trace


def run(clip, dets=(), tracks=(), zones=None, pose=()):
    return scene_states_with_trace(clip, list(dets), list(tracks), zones if zones is not None else zones_default(), list(pose))


def series(states, name):
    t = np.array([s.t for s in states])
    p = np.array([s.signals[name] for s in states])
    c = np.array([s.conf[name] for s in states])
    k = np.array([s.known[name] for s in states])
    return t, p, c, k


def test_safe_opens_at_5s_delay_le_half_second():
    clip = make_clip(10.0, safe_open=lambda t: t >= 5.0)
    st, tr = run(clip)
    t, p, c, k = series(st, "safe_open")
    assert (p[t < 4.8] < 0.5).all(), "до открытия сейф не должен считаться открытым"
    assert (p[(t >= 5.5)] >= 0.5).all(), "задержка роста safe_open > 0.5 с"
    assert k[t >= 5.5].all(), "открытие подтверждено наблюдением"
    assert tr["info"]["safe_open"]["two_states"] is True


def test_safe_open_from_start_is_not_taken_as_closed_reference():
    clip = make_clip(10.0, safe_open=lambda t: t < 5.0)
    st, _ = run(clip)
    t, p, c, k = series(st, "safe_open")
    # главное требование: открытый с начала сейф не объявляется «известно закрыт»
    assert not ((p[t < 4.8] < 0.5) & k[t < 4.8]).any()
    assert (p[t < 4.6] >= 0.5).mean() > 0.9
    assert (p[t > 5.6] < 0.5).all()


def test_safe_open_from_start_with_detector_confirmation():
    clip = make_clip(10.0, safe_open=lambda t: t < 5.0)
    dets = [Detection(t=t, label="safe_open", bbox=SAFE, score=0.7) for t in clip.times if t < 5.0]
    dets += [Detection(t=t, label="safe", bbox=SAFE, score=0.7) for t in clip.times if t >= 5.0]
    st, tr = run(clip, dets)
    t, p, c, k = series(st, "safe_open")
    assert (p[t < 4.6] >= 0.5).all() and k[t < 4.6].all()
    assert (p[t > 5.6] < 0.5).all() and k[t > 5.6].all()
    assert "det" in st[0].provenance["safe_open"]


def test_static_closed_safe_without_evidence_is_unknown_not_false():
    clip = make_clip(8.0)
    st, tr = run(clip)
    t, p, c, k = series(st, "safe_open")
    assert (p < 0.5).all()
    assert not k.any(), "без детекций и без смены вида состояние сейфа неизвестно"
    assert tr["info"]["safe_open"]["label_conf"] == pytest.approx(0.0)


def test_person_occluding_safe_is_not_opening():
    box = (SAFE[0] - 0.05, SAFE[1] - 0.04, SAFE[2] + 0.04, SAFE[3] + 0.06)
    occl = lambda t: [box] if 4.0 <= t < 6.5 else []
    clip = make_clip(10.0, persons=occl)
    tr_ = person_track(1, {t: box for t in clip.times if 4.0 <= t < 6.5}, role="cashier", conf=0.9)
    st, _ = run(clip, person_dets([tr_]), [tr_])
    t, p, c, k = series(st, "safe_open")
    assert (p < 0.5).all(), "окклюзия человеком не равна открытию"
    vis = np.array([s.visibility["safe_open"] for s in st])
    assert vis[(t >= 4.2) & (t < 6.3)].max() < 0.5


def test_person_occludes_open_safe_state_is_held():
    box = (SAFE[0] - 0.05, SAFE[1] - 0.04, SAFE[2] + 0.04, SAFE[3] + 0.06)
    clip = make_clip(12.0, safe_open=lambda t: t >= 3.0, persons=lambda t: [box] if 6.0 <= t < 7.5 else [])
    tr_ = person_track(1, {t: box for t in clip.times if 6.0 <= t < 7.5}, role="cashier", conf=0.9)
    st, _ = run(clip, person_dets([tr_]), [tr_])
    t, p, c, k = series(st, "safe_open")
    assert (p[(t >= 3.6)] >= 0.5).all(), "во время кратковременной окклюзии открытый сейф остаётся открытым"


def test_global_lighting_change_is_not_opening():
    dim = lambda f, t, i: np.clip(f.astype(np.float32) * (0.55 if t >= 5.0 else 1.0), 0, 255).astype(np.uint8)
    clip = make_clip(10.0, post=dim)
    st, _ = run(clip)
    t, p, c, k = series(st, "safe_open")
    assert (p < 0.5).all()


def test_soft_shadow_on_safe_is_not_opening():
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = (SAFE[0] + SAFE[2]) / 2 * W, (SAFE[1] + SAFE[3]) / 2 * H
    shade = 1.0 - 0.45 * np.exp(-(((xx - cx) / 40.0) ** 2 + ((yy - cy) / 60.0) ** 2))

    def shadow(f, t, i):
        if t < 5.0:
            return f
        return np.clip(f.astype(np.float32) * shade[..., None], 0, 255).astype(np.uint8)

    st, _ = run(make_clip(10.0, post=shadow))
    t, p, c, k = series(st, "safe_open")
    assert not ((p >= 0.5) & k).any(), "мягкая тень не должна давать подтверждённое открытие"


def test_low_quality_frames_lower_confidence_and_no_false_open():
    def degrade(f, t, i):
        if 4.0 <= t < 6.0:
            g = cv2.GaussianBlur(f, (31, 31), 0)
            return np.clip(g.astype(np.float32) * 0.12, 0, 255).astype(np.uint8)
        return f

    clip = make_clip(10.0, post=degrade)
    dets = [Detection(t=t, label="safe", bbox=SAFE, score=0.6) for t in clip.times if not (4.0 <= t < 6.0)]
    st, _ = run(clip, dets)
    t, p, c, k = series(st, "safe_open")
    q = np.array([s.quality for s in st])
    bad = (t >= 4.2) & (t < 5.8)
    assert q[bad].max() < 0.4 < q[~((t >= 4.0) & (t < 6.0))].min()
    assert (p < 0.5).all()
    cash = series(st, "cash_visible")
    assert not (cash[3][bad] & (cash[1][bad] < 0.5)).any(), "в плохих кадрах «нет наличных» не утверждается"


@pytest.mark.parametrize("angle,scale,tx,ty", [(4.0, 0.92, 10.0, -14.0), (-6.0, 1.05, -8.0, 12.0)])
def test_affine_view_change_robust(angle, scale, tx, ty):
    m = cv2.getRotationMatrix2D((W / 2, H / 2), angle, scale)
    m[0, 2] += tx
    m[1, 2] += ty
    warp = lambda f, t, i: cv2.warpAffine(f, m, (W, H), borderMode=cv2.BORDER_REFLECT)
    clip = make_clip(10.0, safe_open=lambda t: t >= 5.0, post=warp)
    st, _ = run(clip, zones=affine_zones(m, zones_default()))
    t, p, c, k = series(st, "safe_open")
    assert (p[t < 4.8] < 0.5).all()
    assert (p[t >= 5.5] >= 0.5).all()


def test_camera_jitter_is_compensated():
    rng = np.random.default_rng(3)
    offs = rng.integers(-3, 4, size=(60, 2))

    def jitter(f, t, i):
        m = np.float32([[1, 0, offs[i, 0]], [0, 1, offs[i, 1]]])
        return cv2.warpAffine(f, m, (W, H), borderMode=cv2.BORDER_REFLECT)

    st, _ = run(make_clip(10.0, post=jitter))
    t, p, c, k = series(st, "safe_open")
    assert not ((p >= 0.5) & k).any()
    st2, _ = run(make_clip(10.0, safe_open=lambda t: t >= 5.0, post=jitter))
    t2, p2, _, _ = series(st2, "safe_open")
    assert (p2[t2 >= 5.5] >= 0.5).all() and (p2[t2 < 4.8] < 0.5).all()


def test_door_crossing_gives_open_evidence():
    # человек проходит из клиентской зоны через дверь в зону кассира
    path = {}
    for k in range(25):
        t = round(k / 5.0, 4)
        y = 0.10 + 0.02 * k
        path[t] = (0.06, y, 0.18, y + 0.10)
    clip = make_clip(8.0, persons=lambda t: [path[t]] if t in path else [])
    tr_ = person_track(5, path, role="visitor", conf=0.8)
    st, tr = run(clip, person_dets([tr_]), [tr_])
    t, p, c, k = series(st, "door_open")
    assert tr["info"]["door_crossings"], "пересечение границы через дверь найдено"
    tc = tr["info"]["door_crossings"][0]
    near = np.abs(t - tc) <= 0.4
    assert (p[near] >= 0.5).all() and k[near].all()
    assert "traffic" in st[0].provenance["door_open"]


def test_door_without_any_evidence_is_unknown():
    st, _ = run(make_clip(6.0))
    t, p, c, k = series(st, "door_open")
    assert (p < 0.5).all() and not k.any()
