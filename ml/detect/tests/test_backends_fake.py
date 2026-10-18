"""Постобработка реальных бэкендов на фейковых выходах модели (нужен только torch, весов нет)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")


class FakeDetr(torch.nn.Module):
    """Возвращает фиксированные logits/pred_boxes (как RT-DETR: sigmoid-логиты, cxcywh в долях входа)."""

    def __init__(self, logits, boxes, id2label):
        super().__init__()
        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.boxes = torch.tensor(boxes, dtype=torch.float32)
        self.config = SimpleNamespace(id2label=id2label)
        self.seen = []

    def forward(self, pixel_values):
        self.seen.append(tuple(pixel_values.shape))
        b = pixel_values.shape[0]
        return SimpleNamespace(logits=self.logits.expand(b, -1, -1), pred_boxes=self.boxes.expand(b, -1, -1))


def test_hf_detr_postprocess_maps_labels_and_sources():
    from ml.detect.backends.hf_detr import HFDetrBackend

    id2label = {0: "person", 1: "car", 2: "book"}
    big, small = 5.0, -5.0
    logits = [[[big, small, small], [small, big, small], [small, small, big], [small, small, small]]]
    boxes = [[[0.5, 0.5, 0.2, 0.4], [0.1, 0.1, 0.1, 0.1], [0.3, 0.7, 0.2, 0.2], [0.9, 0.9, 0.1, 0.1]]]
    m = FakeDetr(logits, boxes, id2label)
    be = HFDetrBackend("fake/rtdetr", device="cpu", model=m, processor=SimpleNamespace(size={"height": 64, "width": 64}),
                       min_score=0.5, max_dets=10)
    frames = [np.zeros((848, 464, 3), np.uint8), np.zeros((848, 464, 3), np.uint8)]
    res = be.predict(frames)
    assert m.seen == [(2, 3, 64, 64)]
    assert len(res) == 2
    d = res[0]
    assert sorted(d.labels) == ["document", "person"]          # car отброшен, book → document
    i = d.labels.index("person")
    assert d.boxes[i].tolist() == pytest.approx([0.4, 0.3, 0.6, 0.7])
    j = d.labels.index("document")
    assert d.sources[j] == "rtdetr:book~proxy" and d.sources[i] == "rtdetr:person"
    be2 = HFDetrBackend("fake/rtdetr", device="cpu", model=m, processor=None, min_score=0.5, allow_proxy=False)
    assert be2.predict(frames[:1])[0].labels == ["person"]


class FakeOwl(torch.nn.Module):
    def __init__(self, logits, boxes, image_size=96):
        super().__init__()
        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.boxes = torch.tensor(boxes, dtype=torch.float32)
        self.config = SimpleNamespace(vision_config=SimpleNamespace(image_size=image_size))
        self.calls = []

    def forward(self, input_ids, pixel_values, attention_mask, **kw):
        self.calls.append((tuple(input_ids.shape), tuple(pixel_values.shape), kw))
        b = pixel_values.shape[0]
        return SimpleNamespace(logits=self.logits.expand(b, -1, -1), pred_boxes=self.boxes.expand(b, -1, -1))


class FakeTok:
    def __call__(self, texts, **kw):
        n = len(texts)
        return {"input_ids": torch.ones((n, 16), dtype=torch.long), "attention_mask": torch.ones((n, 16), dtype=torch.long)}


def test_owlv2_postprocess_unpads_portrait_frame():
    from ml.detect.backends.owlv2 import OwlV2Backend

    queries = [{"text": "stack of banknotes", "label": "cash", "thr": 0.3},
               {"text": "metal safe", "label": "safe", "thr": 0.3}]
    lo, hi = -4.0, 4.0
    # 3 патча: 1 — cash, 2 — safe ниже порога, 3 — safe
    logits = [[[hi, lo], [lo, -1.0], [lo, hi]]]
    # cxcywh в долях КВАДРАТА 848×848 (кадр 464×848 занимает левые 464/848 по x)
    boxes = [[[0.25, 0.5, 0.1, 0.2], [0.1, 0.1, 0.05, 0.05], [0.4, 0.8, 0.1, 0.1]]]
    m = FakeOwl(logits, boxes)
    be = OwlV2Backend("fake/owl", queries, device="cpu", input_size=96, model=m,
                      processor=SimpleNamespace(tokenizer=FakeTok(), image_processor=None), max_area={"default": 0.9})
    res = be.predict([np.zeros((848, 464, 3), np.uint8)])
    assert m.calls[0][0] == (2, 16) and m.calls[0][1] == (1, 3, 96, 96) and m.calls[0][2] == {}
    d = res[0]
    assert sorted(d.labels) == ["cash", "safe"]
    i = d.labels.index("cash")
    sx = 848 / 464
    assert d.boxes[i].tolist() == pytest.approx([0.2 * sx, 0.4, 0.3 * sx, 0.6], abs=1e-5)
    assert d.sources[i] == "owlv2:stack of banknotes"
    j = d.labels.index("safe")
    assert d.boxes[j].tolist() == pytest.approx([0.35 * sx, 0.75, min(1.0, 0.45 * sx), 0.85], abs=1e-5)


def test_owlv2_non_native_size_interpolates():
    from ml.detect.backends.owlv2 import OwlV2Backend

    m = FakeOwl([[[0.0]]], [[[0.5, 0.5, 0.1, 0.1]]], image_size=96)
    be = OwlV2Backend("fake/owl", [{"text": "keys", "label": "keys", "thr": 0.9}], device="cpu", input_size=64,
                      model=m, processor=SimpleNamespace(tokenizer=FakeTok(), image_processor=None))
    assert be.predict([np.zeros((100, 200, 3), np.uint8)])[0].labels == []
    assert m.calls[0][2] == {"interpolate_pos_encoding": True}


def test_preprocess_pad_square_and_rgb():
    from ml.detect.backends.base import torch_preprocess

    f = np.zeros((4, 2, 3), np.uint8)
    f[..., 0] = 255  # синий канал BGR
    x = torch_preprocess([f], (4, 4), "cpu", torch.float32, pad_square=True, pad_value=0.5)
    assert tuple(x.shape) == (1, 3, 4, 4)
    assert float(x[0, 2, 0, 0]) == pytest.approx(1.0)   # B → канал 2 (RGB)
    assert float(x[0, 0, 0, 0]) == pytest.approx(0.0)
    assert float(x[0, 0, 0, 3]) == pytest.approx(0.5)   # паддинг справа


def test_owlv2_raw_score_all_queries_unpadded():
    from ml.detect.backends.owlv2 import OwlV2Backend

    logits = [[[4.0, -4.0], [-4.0, 4.0]]]
    boxes = [[[0.25, 0.5, 0.1, 0.2], [0.4, 0.8, 0.1, 0.1]]]
    m = FakeOwl(logits, boxes)
    be = OwlV2Backend("fake/owl", [{"text": "open safe"}, {"text": "closed metal safe"}], device="cpu", input_size=96,
                      model=m, processor=SimpleNamespace(tokenizer=FakeTok(), image_processor=None))
    (b, p), = be.score([np.zeros((848, 464, 3), np.uint8)])
    assert b.shape == (2, 4) and p.shape == (2, 2)
    sx = 848 / 464
    assert b[0].tolist() == pytest.approx([0.2 * sx, 0.4, 0.3 * sx, 0.6], abs=1e-5)
    assert p[0, 0] > 0.9 and p[0, 1] < 0.1 and p[1, 1] > 0.9
    assert be.predict([np.zeros((848, 464, 3), np.uint8)])[0].labels == []   # без label запросы не дают Detection


@pytest.mark.parametrize("hw", [(848, 464), (464, 848), (100, 100)])
def test_hf_detr_box_is_independent_of_frame_size_and_orientation(hw):
    """Модель видит кадр, растянутый в квадрат: нормированный бокс не зависит от размера/ориентации кадра,
    пиксели = норм. × (W, H)."""
    from ml.detect.backends.hf_detr import HFDetrBackend

    m = FakeDetr([[[5.0]]], [[[0.25, 0.75, 0.1, 0.2]]], {0: "person"})
    be = HFDetrBackend("fake/rtdetr", device="cpu", model=m, processor=SimpleNamespace(size={"height": 32, "width": 32}),
                       min_score=0.5)
    h, w = hw
    d = be.predict([np.zeros((h, w, 3), np.uint8)])[0]
    assert d.labels == ["person"]
    assert d.boxes[0].tolist() == pytest.approx([0.2, 0.65, 0.3, 0.85])
    px = d.boxes[0] * np.array([w, h, w, h])
    assert px.tolist() == pytest.approx([0.2 * w, 0.65 * h, 0.3 * w, 0.85 * h], rel=1e-5)


def test_hf_detr_mixed_frame_sizes_in_one_call():
    from ml.detect.backends.hf_detr import HFDetrBackend

    m = FakeDetr([[[5.0]]], [[[0.5, 0.5, 0.2, 0.2]]], {0: "cup"})
    be = HFDetrBackend("fake/rtdetr", device="cpu", model=m, processor=SimpleNamespace(size={"height": 32, "width": 32}),
                       min_score=0.5)
    res = be.predict([np.zeros((848, 464, 3), np.uint8), np.zeros((464, 848, 3), np.uint8),
                      np.zeros((848, 464, 3), np.uint8)])
    assert len(res) == 3 and all(r.labels == ["cup"] for r in res)
    assert sorted(m.seen) == [(1, 3, 32, 32), (2, 3, 32, 32)]       # группы по размеру кадра


def test_owlv2_landscape_frame_unpadding():
    """Горизонтальный кадр (другой угол): OWLv2 паддит снизу; y пересчитывается ×S/H, x не меняется."""
    from ml.detect.backends.owlv2 import OwlV2Backend

    m = FakeOwl([[[4.0]]], [[[0.5, 0.25, 0.2, 0.1]]])
    be = OwlV2Backend("fake/owl", [{"text": "open safe", "label": "safe_open", "thr": 0.3}], device="cpu",
                      input_size=96, model=m, processor=SimpleNamespace(tokenizer=FakeTok(), image_processor=None),
                      max_area={"default": 0.9})
    h, w = 464, 848
    d = be.predict([np.zeros((h, w, 3), np.uint8)])[0]
    sy = w / h
    assert d.labels == ["safe_open"]
    assert d.boxes[0].tolist() == pytest.approx([0.4, 0.2 * sy, 0.6, 0.3 * sy], abs=1e-5)


def test_preprocess_pads_landscape_at_bottom():
    from ml.detect.backends.base import torch_preprocess

    f = np.full((2, 4, 3), 255, np.uint8)
    x = torch_preprocess([f], (4, 4), "cpu", torch.float32, pad_square=True, pad_value=0.5)
    assert float(x[0, 0, 0, 0]) == pytest.approx(1.0) and float(x[0, 0, 3, 0]) == pytest.approx(0.5)
