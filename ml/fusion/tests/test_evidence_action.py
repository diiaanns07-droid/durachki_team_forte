"""Кадры-доказательства, политика action (фаза 2, по умолчанию выключена): ключевой момент + до/после."""
from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")

from ml.fusion.config import load_config  # noqa: E402
from ml.fusion.core import Candidate  # noqa: E402
from ml.fusion.evidence import EvidenceRenderer  # noqa: E402

from .test_evidence import decode, textured  # noqa: E402
from .conftest import FakeClip  # noqa: E402


def action_cfg():
    cfg = load_config()
    cfg["evidence"]["policy"] = "action"
    return cfg


def cand(**kw):
    base = dict(type_id="V02", t_start=8.0, t_end=12.0, t_peak=9.0, confidence=0.8)
    base.update(kw)
    return Candidate(**base)


def test_default_policy_is_phase1():
    assert load_config()["evidence"]["policy"] == "sharpest"


def test_key_moment_not_replaced_by_sharper_frame():
    def maker(idx):
        img = textured(idx)
        return img if idx == 9 * 30 + 6 else cv2.GaussianBlur(img, (15, 15), 0)  # резкий только 9.2 с

    r = EvidenceRenderer(FakeClip(maker=maker), action_cfg())
    frames, ui = r.render(cand(), {"name": "x"})
    ts = [f.timestamp_sec for f in frames]
    assert ts[0] == 9.0                                   # ключевой момент = таймкод, даже если он размыт
    assert len(set(ts)) == len(ts) and all(abs(a - b) >= 0.4 for i, a in enumerate(ts) for b in ts[i + 1:])
    roles = [f["role"] for f in ui["frames"]]
    assert roles[0] == "key"


def test_before_and_after_moments():
    r = EvidenceRenderer(FakeClip(maker=textured), action_cfg())
    moments = r.select_moments(cand(), degraded=False)
    assert moments[0] == (9.0, "key")
    roles = dict((role, t) for t, role in moments[1:])
    assert roles["before"] == pytest.approx(7.0, abs=0.21) and roles["after"] == pytest.approx(14.0, abs=0.21)


def test_a09_evidence_t_used():
    r = EvidenceRenderer(FakeClip(maker=textured), action_cfg())
    c = cand(extra={"evidence_t": [8.4, 9.0, 10.6]})
    moments = r.select_moments(c, degraded=False)
    assert moments[0] == (9.0, "key")
    assert {round(t, 1) for t, _ in moments[1:]} <= {8.4, 10.6, 7.0, 14.0}
    assert any(abs(t - 8.4) < 0.21 or abs(t - 10.6) < 0.21 for t, _ in moments[1:])


def test_v02_after_frame_highlights_left_object():
    cfg = action_cfg()
    cfg["evidence"]["ui_overlay_frames"] = 3
    c = cand(t_peak=8.0, extra={"state_after": {"t_after": 11.0, "valuables_bboxes": {"cash#2": (0.2, 0.5, 0.6, 0.7)}}},
             bboxes={"cashier": (0.1, 0.1, 0.5, 0.9)})
    r = EvidenceRenderer(FakeClip(maker=textured), cfg)
    frames, ui = r.render(c, {"name": "x"})
    roles = {f["role"]: f["timestamp_sec"] for f in ui["frames"]}
    assert roles["key"] == 8.0 and roles["after"] == pytest.approx(11.0, abs=0.21)
    after_boxes = [s for s in ui["shapes"] if s.get("frame_role") == "after" and s["kind"] == "box"]
    assert [s["label"] for s in after_boxes] == ["наличные #2"]
    for f in frames:                                       # кадры жюри — чистые, исходного размера
        assert decode(f.image_base64).shape[:2] == (160, 96)


@pytest.mark.parametrize("mode,clean_above", [("legend", True), ("tags", False)])
def test_legend_mode_keeps_labels_off_the_scene(mode, clean_above):
    import numpy as np

    cfg = load_config()
    cfg["evidence"]["ui_label_mode"] = mode
    r = EvidenceRenderer(FakeClip(w=192, h=320, maker=lambda i: np.full((320, 192, 3), 90, np.uint8)), cfg)
    c = cand(bboxes={"visitor#2": (0.3, 0.3, 0.7, 0.8)})
    frames, ui = r.render(c, {"name": "x"})
    ov = decode(ui["frames"][0]["image_base64"]).astype(float)
    size = max(11, int(round(192 * 0.032)))
    header = size + 12
    y_top, x1, x2 = int(0.3 * 320), int(0.3 * 192) + 8, int(0.7 * 192) - 8
    above = ov[header + y_top - size: header + y_top - 2, x1:x2]
    assert (np.abs(above - 90).mean() < 6) == clean_above   # в legend над рамкой нет плашки с текстом
    assert ui["shapes"][0]["label"] == "посетитель #2"
