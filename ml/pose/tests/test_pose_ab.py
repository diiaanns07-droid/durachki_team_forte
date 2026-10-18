"""ml.pose.ab: A/B на синтетических кадрах (mock-бэкенды явно помечены; реальных данных нет)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from ml.pose import ab
from ml.pose.backends.base import WeightsNotFound
from ml.pose.backends.mock import MockPose

cv2 = pytest.importorskip("cv2")


@pytest.fixture()
def frames_dir(tmp_path):
    d = tmp_path / "frames"
    d.mkdir()
    for i in range(6):
        img = np.full((424, 232, 3), 60, np.uint8)
        cv2.rectangle(img, (70 + i, 120), (150 + i, 320), (180, 180, 180), -1)
        cv2.imwrite(str(d / f"f{i:04d}.jpg"), img)
    boxes = {f"f{i:04d}.jpg": [{"bbox": [0.28, 0.27, 0.66, 0.77], "track_id": 1, "role": "cashier"},
                               [0.70, 0.30, 0.95, 0.70]] for i in range(6)}
    (tmp_path / "boxes.json").write_text(json.dumps(boxes), encoding="utf-8")
    return d


def _factory(name, cfg):
    if name in ("mock", "rtmpose_onnx"):  # подмена: оба — mock (только для теста)
        return MockPose(cfg)
    raise WeightsNotFound(f"{name}: нет весов (тест)")


def test_ab_end_to_end(frames_dir, tmp_path):
    out = tmp_path / "ab_out"
    rc = ab.main(["--frames", str(frames_dir), "--out", str(out), "--boxes", str(tmp_path / "boxes.json"),
                  "--backends", "rtmpose_onnx,mock,mediapipe"], make_backend=_factory)
    assert rc == 0
    rep = json.loads((out / "ab_report.json").read_text(encoding="utf-8"))
    assert rep["n_frames"] == 6
    assert rep["backends"]["mediapipe"]["status"] == "unavailable"          # не роняет A/B
    m = rep["backends"]["mock"]
    assert m["status"] == "ok" and m["mock"] is True and m["n_crops"] == 12
    assert m["groups"]["wrists"]["visible_frac"] == 1.0
    assert m["latency_ms"]["p50"] is not None and m["jitter"]["n_pairs"] > 0
    agr = rep["agreement"]["rtmpose_onnx~mock"]
    assert agr["n_common_crops"] == 12 and agr["wrists"]["pck_0.5"] == 1.0  # одинаковые mock-и согласны
    assert (out / "ab_report.md").read_text(encoding="utf-8").count("|") > 10
    assert (out / "per_crop.csv").exists()
    assert len(list((out / "overlays" / "mock").glob("*.jpg"))) == 6
    assert (out / ".gitignore").read_text(encoding="utf-8").strip().endswith("*")


def test_ab_roles_filter_and_no_overlays(frames_dir, tmp_path):
    out = tmp_path / "ab2"
    rc = ab.main(["--frames", str(frames_dir), "--out", str(out), "--boxes", str(tmp_path / "boxes.json"),
                  "--backends", "mock", "--roles", "cashier", "--no-overlays"], make_backend=_factory)
    assert rc == 0
    rep = json.loads((out / "ab_report.json").read_text(encoding="utf-8"))
    assert rep["backends"]["mock"]["n_crops"] == 6
    assert not (out / "overlays").exists()


def test_ab_rejects_unknown_variant(frames_dir, tmp_path):
    with pytest.raises(SystemExit):
        ab.main(["--frames", str(frames_dir), "--out", str(tmp_path / "x"), "--backends", "yolo"])


def test_overlay_blurs_head_region():
    img = np.zeros((200, 120, 3), np.uint8)
    img[::2, ::2] = 255                                   # высокочастотная текстура
    kp = np.tile(np.array([[60.0, 140.0]], np.float32), (17, 1))
    kp[:5] = [60.0, 40.0]                                 # голова
    kp[5], kp[6] = [40.0, 70.0], [80.0, 70.0]
    sc = np.full(17, 0.9, np.float32)
    box = np.array([20, 20, 100, 190], np.float32)
    outimg = ab.draw_overlay(img, [(kp, sc, box)], 0.3, blur_heads=True)
    head = outimg[30:50, 50:70].astype(float)
    assert head.std() < img[30:50, 50:70].astype(float).std()  # текстура у головы размыта


def test_jitter_and_groups_math():
    kp = np.zeros((17, 2), np.float32)
    sc = np.full(17, 0.9, np.float32)
    seqs = {1: [(0, kp, sc, 10.0), (1, kp + 1.0, sc, 10.0), (2, kp + 2.0, sc, 10.0)]}
    j = ab.jitter_stats(seqs, 0.3)
    assert j["velocity_p50"] == pytest.approx(np.sqrt(2) / 10.0, rel=1e-3)
    assert j["accel_p50"] == pytest.approx(0.0, abs=1e-6)      # равномерное движение — не дрожание
    g = ab.group_stats(np.vstack([sc, np.zeros(17)]), 0.3)
    assert g["wrists"]["visible_frac"] == 0.5


def test_ab_video_mode_uses_pipeline_modules(monkeypatch, tmp_path):
    """--video: кадры и bbox из ingest/detect/track (здесь — фейковые модули, без реальных данных)."""
    import sys
    import types

    from ml.pose._contract import Track

    class Clip:
        times = [0.0, 0.2, 0.4, 0.6, 0.8]  # человек виден до 0.2 с (+допуск 0.25 с → кадр 0.4 с тоже)

        def frame(self, i):
            return np.full((424, 232, 3), 50, np.uint8)

    fake = {
        "ml.ingest": types.SimpleNamespace(load_video=lambda path, fps=5.0: Clip()),
        "ml.detect": types.SimpleNamespace(detect=lambda clip, every_n=1: []),
        "ml.track": types.SimpleNamespace(track=lambda clip, dets: (
            [Track(3, "person", "cashier", 0.9, [(t, (0.3, 0.3, 0.6, 0.7)) for t in Clip.times[:2]])], [])),
    }
    for k, v in fake.items():
        monkeypatch.setitem(sys.modules, k, v)
    out = tmp_path / "abv"
    rc = ab.main(["--video", str(tmp_path / "x.mov"), "--out", str(out), "--backends", "mock", "--no-overlays"],
                 make_backend=_factory)
    assert rc == 0
    rep = json.loads((out / "ab_report.json").read_text(encoding="utf-8"))
    assert rep["n_frames"] == 3 and rep["backends"]["mock"]["n_crops"] == 3   # кадры без людей пропущены


def test_overlay_without_head_keypoints_blurs_whole_person_not_guess():
    """Нет точек головы → место головы не угадывается: размыт весь bbox человека (и это посчитано)."""
    img = np.zeros((200, 120, 3), np.uint8)
    img[::2, ::2] = 255
    kp = np.tile(np.array([[60.0, 120.0]], np.float32), (17, 1))
    sc = np.full(17, 0.9, np.float32)
    sc[:5] = 0.0                                          # голова не видна
    box = np.array([20, 20, 100, 190], np.float32)
    counts: dict = {}
    out = ab.draw_overlay(img, [(kp, sc, box)], 0.3, blur_heads=True, counts=counts)
    assert counts == {"whole_bbox": 1}
    low = out[150:180, 30:50].astype(float)               # низ bbox (не «верх-голова») тоже размыт
    assert low.std() < img[150:180, 30:50].astype(float).std()
    counts2: dict = {}
    sc2 = np.full(17, 0.9, np.float32)
    ab.draw_overlay(img, [(kp, sc2, box)], 0.3, blur_heads=True, counts=counts2)
    assert counts2 == {"head_keypoints": 1}
