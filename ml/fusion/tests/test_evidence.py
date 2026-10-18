"""Кадры-доказательства: PTS, валидный JPEG ≤ 200 КБ в base64, дубли, резкость, приватность, оверлеи UI."""
from __future__ import annotations

import base64

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from ml.fusion import fuse  # noqa: E402
from ml.fusion.config import load_config  # noqa: E402
from ml.fusion.core import Candidate  # noqa: E402
from ml.fusion.evidence import (  # noqa: E402
    EvidenceRenderer,
    blur_regions,
    encode_jpeg,
    head_boxes_from_keypoints,
    parse_box_key,
    ru_label,
)

from .conftest import FakeClip, ev  # noqa: E402


def decode(b64: str) -> np.ndarray:
    data = base64.b64decode(b64, validate=True)
    assert data[:2] == b"\xff\xd8", "не JPEG"
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    return img


def textured(idx: int, h: int = 160, w: int = 96) -> np.ndarray:
    rng = np.random.default_rng(idx)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    img[: h // 4, : w // 2] = idx % 256
    return img


def cand(t: float = 5.0, **kw) -> Candidate:
    base = dict(type_id="V01", t_start=t - 2.0, t_end=t + 0.5, t_peak=t, confidence=0.8)
    base.update(kw)
    return Candidate(**base)


def test_jury_frames_valid_jpeg_base64_and_size(catalog):
    clip = FakeClip(maker=textured)
    res = fuse(clip, [ev(t=5.0, bboxes={"visitor:1": (0.1, 0.1, 0.6, 0.9)})], [], catalog, "job")
    frames = res.violations[0].evidence_frames
    assert 1 <= len(frames) <= 3
    for f in frames:
        img = decode(f.image_base64)
        assert img.shape[:2] == (clip.height, clip.width)
        assert len(base64.b64decode(f.image_base64)) <= 200_000


def test_timestamp_equals_real_pts_of_encoded_frame(catalog):
    clip = FakeClip(maker=textured)
    res = fuse(clip, [ev(t=5.1)], [], catalog, "job")
    for f in res.violations[0].evidence_frames:
        img = decode(f.image_base64)
        coded = int(round(float(np.median(img[5:30, 5:40]))))   # номер кадра, «зашитый» в пиксели
        assert coded == int(round(f.timestamp_sec * clip.fps)) % 256
        assert any(abs(f.timestamp_sec - t) < 1e-6 for t in clip.times), "timestamp не PTS кадра"
    assert res.violations[0].evidence_frames[0].timestamp_sec == pytest.approx(res.violations[0].timecode_sec)


def test_no_duplicate_evidence_on_static_scene(catalog):
    static = np.full((160, 96, 3), 77, np.uint8)
    clip = FakeClip(maker=lambda idx: static.copy())
    res = fuse(clip, [ev(t=5.0, dur=4.0)], [], catalog, "job")
    frames = res.violations[0].evidence_frames
    assert len(frames) == 1


def test_distinct_moments_and_sharpest_chosen():
    def maker(idx: int) -> np.ndarray:
        img = textured(idx)
        return img if idx == 168 else cv2.GaussianBlur(img, (9, 9), 0)  # резкий только кадр 5.6 с

    clip = FakeClip(maker=maker)
    r = EvidenceRenderer(clip, load_config())
    frames, _ = r.render(cand(5.0), {"name": "x"})
    ts = [f.timestamp_sec for f in frames]
    assert ts[0] == 5.0 and 5.6 in ts
    assert len(set(ts)) == len(ts)
    assert all(abs(a - b) >= 0.4 - 1e-9 for i, a in enumerate(ts) for b in ts[i + 1:])


def test_frame_at_failure_falls_back_to_sampled():
    class Broken(FakeClip):
        def frame_at(self, t):
            raise OSError("seek failed")

    clip = Broken(maker=textured)
    r = EvidenceRenderer(clip, load_config())
    frames, _ = r.render(cand(5.0), {"name": "x"})
    assert frames and frames[0].timestamp_sec == 5.0
    assert any("frame_at failed" in w for w in r.warnings)


def test_no_frames_available_no_crash(catalog):
    class Empty(FakeClip):
        def frame(self, i):
            return None

        def frame_at(self, t):
            return None

    res = fuse(Empty(), [ev(t=5.0)], [], catalog, "job")
    assert res.verdict == "violation" and res.violations[0].evidence_frames == []
    assert any("no frame" in w for w in res.meta["warnings"])


def test_large_frame_compressed_under_limit():
    big = np.random.default_rng(0).integers(0, 255, (2160, 3840, 3), dtype=np.uint8)
    data = encode_jpeg(big, 200_000, [90, 70, 50], max_side=1280)
    assert len(data) <= 200_000
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert max(img.shape[:2]) <= 1280


def smooth(idx: int, h: int = 160, w: int = 96) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([(xx * 2 + idx) % 256, (yy * 1.5) % 256, ((xx + yy) + idx * 3) % 256], -1)
    return cv2.GaussianBlur(img.astype(np.uint8), (5, 5), 0)


def test_head_blur_from_keypoints_only_touches_head():
    img = np.random.default_rng(3).integers(0, 255, (400, 300, 3), dtype=np.uint8)
    kps = [(0.5, 0.2, 0.9)] * 5 + [(0.4, 0.3, 0.9), (0.6, 0.3, 0.9)] + [(0.0, 0.0, 0.0)] * 10
    box = head_boxes_from_keypoints(kps, 300, 400, 0.3)
    assert box is not None
    x1, y1, x2, y2 = box
    assert x1 <= 150 <= x2 and y1 <= 80 <= y2
    out = blur_regions(img, [box])
    roi_before = img[y1:y2, x1:x2].astype(float)
    roi_after = out[y1:y2, x1:x2].astype(float)
    assert roi_after.var() < 0.3 * roi_before.var()
    mask = np.ones(img.shape[:2], bool)
    mask[y1:y2, x1:x2] = False
    assert np.array_equal(out[mask], img[mask])  # руки/стол не тронуты


def test_no_keypoints_no_heuristic_blur_by_default():
    clip = FakeClip(maker=smooth)
    r = EvidenceRenderer(clip, load_config())
    c = cand(5.0, bboxes={"cashier": (0.1, 0.1, 0.9, 0.9)})
    assert r.head_regions(c, (160, 96, 3), 5.0) == []
    frames, _ = r.render(c, {"name": "x"})
    jury = decode(frames[0].image_base64).astype(float)
    orig = smooth(150).astype(float)
    assert np.abs(jury - orig).mean() < 6  # только JPEG-шум, без замазанных областей
    top = (slice(16, 40), slice(10, 86))   # «верх bbox» кассира не замазан
    assert jury[top].var() > 0.8 * orig[top].var()
    assert any("privacy" in w for w in r.warnings)


def test_top_frac_fallback_only_when_enabled():
    cfg = load_config()
    cfg["evidence"]["head_fallback"] = "top_frac"
    r = EvidenceRenderer(FakeClip(), cfg)
    regions = r.head_regions(cand(5.0, bboxes={"cashier": (0.0, 0.0, 1.0, 1.0)}), (100, 100, 3), 5.0)
    assert len(regions) == 1 and regions[0][3] <= 20


def test_ui_overlay_separate_from_clean_jury_frame(catalog):
    clip = FakeClip(maker=textured)
    res = fuse(clip, [ev(t=5.0, bboxes={"visitor:2": (0.1, 0.1, 0.6, 0.9), "cash_bag": (0.5, 0.6, 0.9, 0.8)})],
               [], catalog, "job")
    jury = decode(res.violations[0].evidence_frames[0].image_base64)
    ov = res.meta["ui"]["evidence_overlays"][0]
    ui = decode(ov["frames"][0]["image_base64"])
    assert jury.shape[:2] == (clip.height, clip.width)
    assert ui.shape[0] > jury.shape[0]  # поля под штамп времени и подпись типа
    labels = {s["label"] for s in ov["shapes"] if s["kind"] == "box"}
    assert labels == {"посетитель #2", "инкассаторская сумка"}
    assert ov["frames"][0]["timestamp_sec"] == res.violations[0].evidence_frames[0].timestamp_sec


def test_ru_labels():
    assert parse_box_key("visitor:3") == ("visitor", 3)
    assert parse_box_key("cash_bag") == ("cash_bag", None)
    assert ru_label("cashier") == "кассир"
    assert ru_label("person_12") == "человек #12"
    assert ru_label("cell_phone") == "телефон"


def test_budget_degrades_to_single_frame():
    cfg = load_config()
    cfg["evidence"]["budget_s"] = 0.0
    r = EvidenceRenderer(FakeClip(maker=textured), cfg)
    r.elapsed = 1.0
    frames, ui = r.render(cand(5.0), {"name": "x"})
    assert len(frames) == 1 and ui["frames"] == []
