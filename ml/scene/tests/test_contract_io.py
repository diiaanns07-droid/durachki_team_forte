"""Контракт, сглаживание, pose-слияние, видео через cv2.VideoWriter, дамп/загрузка."""
from __future__ import annotations

import numpy as np

from ml.scene.tests.synth import make_clip, zones_default
from ml.scene import scene_states
from ml.scene._contract import SIGNALS, SceneState
from ml.scene.core import scene_states_with_trace
from ml.scene.io import SimpleClip, dump_states, find_states_file, load_states, normalize_video_name
from ml.scene.smoothing import hmm2_smooth, hold_conf, runs_above
from ml.scene.types import SceneStateEx, signal_value


def test_contract_shape_and_ranges():
    clip = make_clip(4.0, safe_open=lambda t: t > 2.0)
    st = scene_states(clip, [], [], zones_default(), [])
    assert len(st) == len(clip.times)
    for s in st:
        assert isinstance(s, SceneState) and isinstance(s, SceneStateEx)
        assert set(s.signals) == set(SIGNALS)
        for k, v in s.signals.items():
            if k == "n_persons":
                assert float(v).is_integer()
            else:
                assert 0.0 <= v <= 1.0
        assert set(s.conf) == set(SIGNALS) and set(s.known) == set(SIGNALS)


def test_empty_inputs_and_no_zones():
    clip = make_clip(2.0)
    st = scene_states(clip, [], [], [], [])
    assert len(st) == len(clip.times)
    assert not any(s.known["outsider_in_cashier_area"] for s in st)
    clip.times = []
    assert scene_states(clip, [], [], [], []) == []


def test_frames_unavailable_falls_back_to_detections():
    class NoFrames:
        times = [k / 5.0 for k in range(10)]
        width, height, fps, duration, path = 100, 200, 30.0, 2.0, "x"

        def frame(self, i):
            raise IOError("decoder failed")

    st, tr = scene_states_with_trace(NoFrames(), [], [], zones_default(), [])
    assert len(st) == 10 and any("frames unavailable" in w for w in tr["warnings"])


def test_signal_value_helper_returns_none_for_unknown():
    st = scene_states(make_clip(3.0), [], [], zones_default(), [])
    assert signal_value(st[0], "safe_open") is None
    plain = SceneState(t=0.0, signals={"safe_open": 0.7})
    assert signal_value(plain, "safe_open") == 0.7


def test_pose_signals_are_merged():
    clip = make_clip(4.0)
    pose = [SceneState(t=t, signals={"phone_in_hand": 1.0 if 1.0 <= t < 3.0 else 0.0}) for t in clip.times]
    st = scene_states(clip, [], [], zones_default(), pose)
    t = np.array([s.t for s in st])
    p = np.array([s.signals["phone_in_hand"] for s in st])
    assert p[(t >= 1.4) & (t < 2.6)].min() > 0.5 and p[t < 0.6].max() < 0.5
    assert st[10].provenance["phone_in_hand"] == "pose"


def test_hmm_prior_reversion_and_hold():
    t = np.arange(0, 20, 0.2)
    p = np.where(t < 2, 0.95, 0.5)
    c = np.where(t < 2, 1.0, 0.0)
    post = hmm2_smooth(p, t, c, switch_per_s=0.2, prior=0.1)
    assert post[5] > 0.9 and post[-1] < 0.2, "без наблюдений апостериорная возвращается к априорной"
    h = hold_conf(c, t, 1.0)
    assert h[9] == 1.0 and h[20] < 0.2
    runs = runs_above(np.where((t > 5) & (t < 9), 1.0, 0.0), t, 0.5, 3.0)
    assert len(runs) == 1


def test_video_file_roundtrip_with_videowriter(tmp_path):
    import cv2

    clip = make_clip(4.0, fps=10.0, safe_open=lambda t: t >= 2.0)
    path = tmp_path / "synthetic.avi"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (clip.width, clip.height))
    for f in clip.frames:
        vw.write(f)
    vw.release()
    sc = SimpleClip(str(path), sample_fps=5.0)
    assert len(sc.times) == 20
    st, tr = scene_states_with_trace(sc, [], [], zones_default(), [])
    t = np.array([s.t for s in st])
    p = np.array([s.signals["safe_open"] for s in st])
    assert (p[t >= 2.5] >= 0.5).all() and (p[t < 1.8] < 0.5).all()
    out = dump_states(st, tmp_path / "cache" / "scene_states.json", trace=tr, meta={"gt_sec": 2.0})
    assert find_states_file(tmp_path / "cache") == out
    st2, extra = load_states(out)
    assert len(st2) == len(st) and st2[5].signals == st[5].signals and st2[5].known == st[5].known
    assert extra["meta"]["gt_sec"] == 2.0


def test_video_name_normalization():
    assert normalize_video_name("Х-3.MOV") == normalize_video_name("X-3.mov") == "x-3"
    assert normalize_video_name("X-47") == "x-47"


def test_numpy_times_list_boxes_and_none_inputs():
    from ml.scene._contract import Detection, Track, Zone

    clip = make_clip(3.0)
    clip.times = np.asarray(clip.times)
    tr = Track(track_id=1, label="person", role="cashier", role_conf=0.9, boxes=[[float(t), [0.3, 0.78, 0.46, 0.92]] for t in clip.times])
    dets = [Detection(t=np.float64(t), label="person", bbox=[0.3, 0.78, 0.46, 0.92], score=0.9, track_id=1) for t in clip.times]
    zones = [Zone(z.name, [list(p) for p in z.polygon]) for z in zones_default()]
    st = scene_states(clip, dets, [tr], zones, None)
    assert len(st) == len(clip.times)
    assert all(s.signals["cashier_in_workplace"] > 0.5 and s.known["cashier_in_workplace"] for s in st[2:])
    assert len(scene_states(clip, None, None, None, None)) == len(clip.times)
