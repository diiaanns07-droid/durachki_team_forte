import numpy as np
import pytest

from ml.detect.boxes import (RawDets, Tile, classwise_nms, cxcywh_to_xyxy, filter_by_area, hflip_boxes,
                             iou_matrix, make_tiles, nms, suppress_contained, tile_to_frame, touches_inner_edge,
                             weighted_boxes_fusion)


def rd(boxes, scores, labels, src="a"):
    return RawDets(np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(scores, np.float32), list(labels),
                   [src] * len(labels))


def test_iou_basic():
    a = np.array([[0, 0, 0.5, 0.5]], np.float32)
    b = np.array([[0, 0, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75], [0.6, 0.6, 1, 1]], np.float32)
    iou = iou_matrix(a, b)[0]
    assert iou[0] == pytest.approx(1.0)
    assert iou[1] == pytest.approx(0.0625 / (0.25 + 0.25 - 0.0625))
    assert iou[2] == 0.0
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


def test_nms_keeps_best_and_distinct():
    boxes = [[0, 0, 0.5, 0.5], [0.01, 0.01, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]]
    keep = nms(np.asarray(boxes), np.asarray([0.5, 0.9, 0.3]), 0.5)
    assert keep.tolist() == [1, 2]


def test_classwise_nms_does_not_cross_labels():
    d = rd([[0, 0, 0.5, 0.5], [0, 0, 0.5, 0.5]], [0.9, 0.8], ["cup", "bottle"])
    out = classwise_nms(d, 0.5)
    assert sorted(out.labels) == ["bottle", "cup"]


def test_hflip_roundtrip():
    b = np.array([[0.1, 0.2, 0.3, 0.6]], np.float32)
    f = hflip_boxes(b)
    assert f[0].tolist() == pytest.approx([0.7, 0.2, 0.9, 0.6])
    assert hflip_boxes(f)[0].tolist() == pytest.approx(b[0].tolist())


def test_wbf_fuses_views_and_keeps_single_view():
    orig = rd([[0.10, 0.10, 0.30, 0.30], [0.6, 0.6, 0.7, 0.7]], [0.8, 0.5], ["person", "cell_phone"], "o")
    flip = rd([[0.12, 0.10, 0.32, 0.30]], [0.6], ["person"], "f")
    out = weighted_boxes_fusion([orig, flip], iou_thr=0.5, single_view_penalty=0.85)
    assert len(out) == 2
    i = out.labels.index("person")
    assert out.boxes[i][0] == pytest.approx((0.10 * 0.8 + 0.12 * 0.6) / 1.4, abs=1e-5)
    assert out.scores[i] == pytest.approx(0.7, abs=1e-5)       # среднее по двум видам
    assert out.sources[i] == "f+o"
    j = out.labels.index("cell_phone")
    assert out.scores[j] == pytest.approx(0.5 * 0.85, abs=1e-5)  # только в одном виде → штраф, но не исчез


def test_tiles_cover_frame_portrait_and_landscape():
    for w, h in [(464, 848), (1920, 1080), (640, 640), (100, 37)]:
        for frac, ov in [(1.0, 0.15), (0.6, 0.25)]:
            tiles = make_tiles(w, h, frac, ov)
            cov = np.zeros((h, w), bool)
            for t in tiles:
                assert 0 <= t.x0 < t.x1 <= w and 0 <= t.y0 < t.y1 <= h
                cov[t.y0:t.y1, t.x0:t.x1] = True
            assert cov.all(), (w, h, frac)
    assert len(make_tiles(464, 848, 1.0, 0.15)) == 2


def test_tile_to_frame_and_edges():
    t = Tile(0, 384, 464, 848)
    b = tile_to_frame(np.array([[0.0, 0.0, 1.0, 1.0]]), t, 464, 848)[0]
    assert b.tolist() == pytest.approx([0.0, 384 / 848, 1.0, 1.0])
    m = touches_inner_edge(np.array([[0.2, 0.0, 0.4, 0.3], [0.2, 0.3, 0.4, 0.6]]), t, 464, 848, 0.01)
    assert m.tolist() == [True, False]  # верх тайла — внутренняя граница кадра


def test_suppress_contained_and_area_filter():
    base = rd([[0.1, 0.1, 0.5, 0.5]], [0.9], ["cash"])
    extra = rd([[0.15, 0.15, 0.3, 0.3], [0.6, 0.6, 0.7, 0.7], [0.15, 0.15, 0.3, 0.3]], [0.5, 0.5, 0.5],
               ["cash", "cash", "keys"])
    out = suppress_contained(extra, base, 0.8)
    assert out.labels == ["cash", "keys"]
    big = rd([[0, 0, 1, 1], [0.1, 0.1, 0.2, 0.2]], [0.5, 0.5], ["safe", "safe"])
    assert len(filter_by_area(big, 0.0, {"default": 0.6})) == 1


def test_cxcywh():
    assert cxcywh_to_xyxy(np.array([[0.5, 0.5, 0.2, 0.4]]))[0].tolist() == pytest.approx([0.4, 0.3, 0.6, 0.7])
