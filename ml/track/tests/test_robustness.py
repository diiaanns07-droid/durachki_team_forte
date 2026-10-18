"""Устойчивость: двое рядом, перекрытия, пропуски детекций, «свет выключили», шумовые детекции."""
import numpy as np
import pytest

from ml.track import track_full
from ml.track.config import load_config
from ml.track.tests import synth as S


def _purity(dets, gt, res):
    """Для каждого трека людей: доля детекций самого частого GT-человека; треки без GT — ложные."""
    by_track: dict = {}
    for d in dets:
        if d.label == "person" and d.track_id is not None:
            by_track.setdefault(d.track_id, []).append(gt.get(id(d), {}).get("key"))
    out = {}
    for tid, keys in by_track.items():
        real = [k for k in keys if k is not None]
        out[tid] = (max(real.count(k) for k in set(real)) / len(keys)) if real else 0.0
    return out


def test_two_people_side_by_side_overlapping_boxes_keep_two_ids():
    a = S.Person("a", [(0, 0.46, 0.73), (30, 0.46, 0.73)], [(0, 30)])
    b = S.Person("b", [(0, 0.60, 0.74), (30, 0.60, 0.74)], [(0, 30)], color=(0, 160, 0))  # боксы перекрываются
    clip, dets, gt = S.make(S.Scene(30.0, [a, b], S.base_objects(30.0)), drop=0.1, seed=3)
    res = track_full(clip, dets)
    persons = [t for t in res.tracks if t.label == "person"]
    assert len(persons) == 2
    assert all(p >= 0.95 for p in _purity(dets, gt, res).values())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_noise_detections_do_not_create_person_tracks(seed):
    clip, dets, gt = S.make(S.scene_clean(), noise_dets=0.1, seed=seed)
    res = track_full(clip, dets)
    pur = _purity(dets, gt, res)
    assert all(p > 0.5 for p in pur.values()), pur  # нет треков, собранных из шума


def test_blackout_no_false_tracks_and_bridge_flag():
    clip, dets, gt = S.make(S.scene_clean(), blackout=[(20.0, 25.0)])
    base = track_full(clip, dets)
    assert all(p >= 0.95 for p in _purity(dets, gt, base).values())
    n_base = sum(1 for t in base.tracks if t.label == "person")
    cfg = load_config(overrides={"visibility": {"bridge_tracks": True}})
    clip2, dets2, gt2 = S.make(S.scene_clean(), blackout=[(20.0, 25.0)])
    br = track_full(clip2, dets2, cfg=cfg)
    n_br = sum(1 for t in br.tracks if t.label == "person")
    assert n_br <= n_base and n_br == 2  # H-VIS: кассир и клиент — те же ID после включения света
    assert any(e["kind"] == "visibility_bridge" for e in br.info["stitch"])
    roles = sorted(t.role for t in br.tracks if t.label == "person")
    assert roles == ["cashier", "visitor"]


def test_long_dropout_in_view_fragments_but_keeps_roles():
    sc = S.scene_v02()
    sc.persons[0].miss = [(5.0, 9.0)]  # кассир сидит, детектор 4 с его не видит
    clip, dets, gt = S.make(sc)
    res = track_full(clip, dets)
    roles = {t.track_id: t.role for t in res.tracks if t.label == "person"}
    assert set(roles.values()) <= {"cashier", "unknown"} and "cashier" in roles.values()


def test_empty_and_single_frame_inputs():
    clip = S.SynthClip(S.scene_clean())
    from ml.track.contract import Detection
    assert track_full(clip, []).tracks == []
    one = track_full(clip, [Detection(1.0, "person", (0.4, 0.4, 0.6, 0.6), 0.9)])
    assert len(one.tracks) <= 1 and one.info["coverage"]["frames_processed"] == 1
    assert np.isfinite(one.info["timings"]["total_s"])
