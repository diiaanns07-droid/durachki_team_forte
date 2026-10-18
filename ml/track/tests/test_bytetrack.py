import numpy as np

from ml.track import track, track_full
from ml.track.bytetrack import ByteTracker, iou_matrix, linear_assignment
from ml.track.config import load_config
from ml.track.contract import Detection
from ml.track.tests import synth as S
from ml.track.tests._util import run


def test_empty_detections():
    clip = S.SynthClip(S.scene_clean())
    assert track(clip, []) == ([], [])
    res = track_full(clip, None)
    assert res.tracks == [] and res.zones == [] and "no_detections" in res.info["warnings"]


def test_invalid_detections_are_skipped():
    clip = S.SynthClip(S.scene_clean())
    dets = [Detection(0.0, "person", (0.5, 0.5, 0.4, 0.6), 0.9), {"t": "x"},
            Detection(0.0, "person", (0.1, 0.1, 0.3, 0.3), 0.9), Detection(0.2, "person", (0.1, 0.1, 0.3, 0.3), 0.9)]
    res = track_full(clip, dets)
    assert len(res.tracks) == 1 and any("skipped_invalid" in w for w in res.info["warnings"])


def test_single_person_stable_id_with_dropouts_and_jitter():
    _, ids, _, _, _ = run(S.scene_clean(), drop=0.2, seed=5)
    assert len(ids["cashier"]) == 1
    assert len(ids["visitor"]) == 1


def test_every_n_detection_keeps_ids():
    _, ids, _, _, _ = run(S.scene_clean(), every_n=3)
    assert len(ids["cashier"]) == 1 and len(ids["visitor"]) == 1


def test_real_timestamps_preserved_no_fabricated_boxes():
    clip, dets, gt = S.make(S.scene_clean(), drop=0.3, seed=1)
    res = track_full(clip, dets)
    det_times = {round(d.t, 4) for d in dets if d.label == "person"}
    for t in res.tracks:
        if t.label == "person":
            assert {round(tt, 4) for tt, _ in t.boxes} <= det_times


def test_low_score_detections_extend_track():
    cfg = load_config()["tracker"]
    tr = ByteTracker(cfg, 1.0, "person", iter(range(1, 100)))

    class D:
        def __init__(self, box, score):
            self.box, self.score, self.label, self.ref = np.array(box), score, "person", None

    for k in range(10):
        s = 0.9 if k < 3 else 0.2  # дальше — только «слабые» детекции (окклюзия)
        tr.update(k * 0.2, [D((0.1 + 0.002 * k, 0.1, 0.3 + 0.002 * k, 0.3), s)])
    out = tr.finish()
    assert len(out) == 1 and len(out[0].times) == 10


def test_two_crossing_people_no_swap():
    a = S.Person("a", [(0, 0.15, 0.35), (14, 0.85, 0.35)], [(0, 14)], color=(200, 0, 0))
    b = S.Person("b", [(0, 0.85, 0.38), (14, 0.15, 0.38)], [(0, 6.4), (7.6, 14)], color=(0, 0, 200))
    sc = S.Scene(14.0, [a, b], [])
    clip, dets, gt = S.make(sc, seed=2)
    track_full(clip, dets)
    for key, start_x, end_x in (("a", 0.15, 0.85), ("b", 0.85, 0.15)):
        mine = [d for d in dets if id(d) in gt and gt[id(d)]["key"] == key]
        first, last = mine[0].track_id, mine[-1].track_id
        assert first == last, f"{key}: id switched {first}->{last}"


def test_iou_and_assignment_helpers():
    m = iou_matrix(np.array([[0, 0, 1, 1]]), np.array([[0, 0, 1, 1], [2, 2, 3, 3]]))
    assert np.allclose(m, [[1.0, 0.0]])
    matches, ua, ub = linear_assignment(np.array([[0.1, 0.9], [0.8, 0.2]]), 0.5)
    assert sorted(matches) == [(0, 0), (1, 1)] and not ua and not ub


def test_object_tracks_and_label_groups():
    res, _, by_id, _, _ = run(S.scene_v07())
    labels = sorted(t.label for t in res.tracks if t.label != "person")
    assert "cash_bag" in labels and "safe" in labels and "laptop" in labels
    for t in res.tracks:
        assert t.label != "person" or t.role in ("cashier", "visitor", "collector", "unknown")
        if t.label != "person":
            assert t.role == "unknown"
