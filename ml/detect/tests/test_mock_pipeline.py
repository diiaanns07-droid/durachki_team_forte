"""SENTINEL_MOCK=1: детерминированные детекции на синтетике, формат Detection, гейтинг, every_n."""
from __future__ import annotations

import numpy as np
import pytest

from ml.detect import detect, last_stats
from contracts.sentinel_contracts import LABELS, Detection
from ml.detect.boxes import iou_matrix
from ml.detect.synth import ArrayClip, make_synth_video, synth_clip


def _check_format(dets):
    for d in dets:
        assert isinstance(d, Detection)
        assert d.label in LABELS
        assert len(d.bbox) == 4 and all(isinstance(v, float) for v in d.bbox)
        x1, y1, x2, y2 = d.bbox
        assert 0.0 <= x1 <= x2 <= 1.0 and 0.0 <= y1 <= y2 <= 1.0
        assert 0.0 <= d.score <= 1.0
        assert d.track_id is None
        assert isinstance(d.t, float) and d.source


def test_mock_detects_synth_objects_in_memory(mock_mode):
    clip, gts = synth_clip(duration_s=4.0, sample_fps=5.0)
    dets = detect(clip)
    _check_format(dets)
    st = last_stats()
    assert st["status"] == "ok" and st["mock"] is True
    by_t: dict[float, list[Detection]] = {}
    for d in dets:
        by_t.setdefault(d.t, []).append(d)
    assert sorted(by_t) == pytest.approx(clip.times)
    for t, gt in zip(clip.times, gts):
        got = by_t[t]
        for label, box in gt:
            cand = [d for d in got if d.label == label]
            assert cand, (t, label)
            best = max(iou_matrix(np.array([box]), np.array([d.bbox for d in cand]))[0])
            assert best > 0.85, (t, label, best)
    # телефон виден только на 1..3 с
    phone_t = sorted({d.t for d in dets if d.label == "cell_phone"})
    assert phone_t and min(phone_t) >= 1.0 and max(phone_t) < 3.0


def test_mock_is_deterministic(mock_mode):
    clip, _ = synth_clip(duration_s=2.0)
    a = [(d.t, d.label, d.bbox, d.score) for d in detect(clip)]
    b = [(d.t, d.label, d.bbox, d.score) for d in detect(clip)]
    assert a == b


def test_every_n_subsamples(mock_mode):
    clip, _ = synth_clip(duration_s=4.0)
    ts = sorted({d.t for d in detect(clip, every_n=3)})
    assert ts == pytest.approx(clip.times[::3])


def test_mock_on_written_video_file(mock_mode, tmp_path):
    p = tmp_path / "synth.mp4"
    make_synth_video(p, duration_s=2.0, fps=10.0)
    clip = ArrayClip.from_file(p, sample_fps=5.0)
    assert len(clip.times) == 10 and clip.width == 464 and clip.height == 848
    dets = detect(clip)
    _check_format(dets)
    labels = {d.label for d in dets}
    assert {"person", "safe", "laptop", "door", "money_counter"} <= labels  # пережили mp4-сжатие


def test_open_vocab_extra_runs_on_keyframes_only(mock_mode, monkeypatch):
    monkeypatch.setenv("SENTINEL_DETECT_EXTRAS", "open_vocab")
    clip, _ = synth_clip(duration_s=4.0, sample_fps=5.0)
    dets = detect(clip)
    st = last_stats()
    assert st["n_open"] == st["n_keyframes"]
    assert 4 <= st["n_keyframes"] < len(clip.times)          # ~1 fps + всплески движения
    assert st["n_open"] > 0 and dets


def test_extras_tiles_do_not_duplicate(mock_mode):
    clip, gts = synth_clip(duration_s=2.0)
    dets = detect(clip, config={"tiling": {"enabled": True, "on": "all"}})
    st = last_stats()
    assert st["n_tiles"] > 0
    for t in clip.times:
        safes = [d for d in dets if d.t == t and d.label == "safe"]
        assert len(safes) == 1


def test_motion_quality_from_clip_is_used(mock_mode):
    clip, _ = synth_clip(duration_s=4.0)
    clip.quality = {"motion_energy": [0.0] * len(clip.times)}
    detect(clip, config={"open_vocab": {"enabled": True}})
    st = last_stats()
    assert st["n_keyframes"] == 4  # только регулярная сетка 1 fps, всплесков нет


def test_empty_and_broken_frames(mock_mode):
    assert detect(ArrayClip(frames=[], times=[])) == []
    clip, _ = synth_clip(duration_s=1.0)
    clip.frames[2] = None  # битый кадр не должен ронять детекцию
    dets = detect(clip)
    assert clip.times[2] not in {d.t for d in dets}
    st = last_stats()
    assert any("frame 2" in w for w in st["warnings"])
    # битый кадр = пробел в покрытии, а не «объектов нет»
    assert st["status"] == "partial"
    assert st["coverage"]["frames_processed"] == len(clip.times) - 1
    assert st["coverage"]["gaps_s"] == [[clip.times[2], clip.times[2]]]


def test_baseline_over_soft_budget_is_not_degraded(mock_mode):
    clip, _ = synth_clip(duration_s=1.0)
    detect(clip, config={"runtime": {"budget_s": 0.0, "hard_budget_s": 1e9}})
    st = last_stats()
    assert st["extras_cut"] is False and st["status"] == "ok"  # опций нет — нечего «срезать»


def test_budget_forces_stride(mock_mode, monkeypatch):
    clip, _ = synth_clip(duration_s=8.0, sample_fps=5.0)
    dets = detect(clip, config={"runtime": {"budget_s": 0.0, "hard_budget_s": 1e-9, "batch_size": 4}})
    st = last_stats()
    assert st["stride_final"] > 1 and st["status"] == "partial"
    assert len({d.t for d in dets}) < len(clip.times)
    assert st["coverage"]["fraction"] < 1.0 and st["coverage"]["gaps_s"]


def test_last_stats_are_per_thread(mock_mode):
    import threading

    clip_a, _ = synth_clip(duration_s=1.0)
    clip_b, _ = synth_clip(duration_s=2.0)
    seen = {}

    def run(name, clip):
        detect(clip)
        seen[name] = last_stats()["n_sampled"]

    ts = [threading.Thread(target=run, args=("a", clip_a)), threading.Thread(target=run, args=("b", clip_b))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert seen == {"a": len(clip_a.times), "b": len(clip_b.times)}


def test_full_coverage_and_every_n_not_counted_as_gap(mock_mode):
    clip, _ = synth_clip(duration_s=2.0)
    detect(clip, every_n=2)
    cov = last_stats()["coverage"]
    assert cov == {"frames_planned": 5, "frames_processed": 5, "fraction": 1.0, "gaps_s": []}
    assert last_stats()["status"] == "ok"


def test_backend_failure_is_partial_not_empty(mock_mode, monkeypatch):
    from ml.detect.backends.mock import MockColorDetector

    calls = {"n": 0}
    orig = MockColorDetector.predict

    def flaky(self, frames):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("device lost")
        return orig(self, frames)

    monkeypatch.setattr(MockColorDetector, "predict", flaky)
    clip, _ = synth_clip(duration_s=4.0)
    detect(clip, config={"runtime": {"batch_size": 5}})
    st = last_stats()
    assert st["status"] == "partial"
    assert st["coverage"]["frames_processed"] == 5 and st["coverage"]["gaps_s"][0][0] == clip.times[5]
    assert any("all closed-set backends failed" in w for w in st["warnings"])


def test_landscape_video_other_angle(mock_mode):
    """Тест «под другим углом» может быть горизонтальным: 848×464, нормированные боксы, без координат-констант."""
    clip, gts = synth_clip(duration_s=1.0, width=848, height=464)
    dets = detect(clip)
    _check_format(dets)
    assert {lab for lab, _ in gts[0]} <= {d.label for d in dets if d.t == 0.0}


def test_bbox_normalized_to_actual_frame_not_clip_metadata(mock_mode):
    """Если A04 отдаёт кадры другого размера, чем clip.width/height (например, уменьшенные), bbox нормируется
    к фактическому кадру — координаты не «съезжают»."""
    import cv2

    clip, gts = synth_clip(duration_s=0.4)
    small = [cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2), interpolation=cv2.INTER_NEAREST) for f in clip.frames]
    clip2 = ArrayClip(frames=small, times=clip.times, width=464, height=848, fps=clip.fps, duration=clip.duration)
    dets = detect(clip2)
    for label, box in gts[0]:
        cand = [d for d in dets if d.t == 0.0 and d.label == label]
        assert cand and max(iou_matrix(np.array([box]), np.array([d.bbox for d in cand]))[0]) > 0.8, label


def test_missing_detection_under_poor_visibility_is_not_absence(mock_mode):
    """Тёмные кадры (свет выключили): детекций нет, но это НЕ «объектов нет». detect() ничего не утверждает об
    отсутствии; inventory/safe_door_state отвечают unknown; coverage считает кадры обработанными, а не «чистыми»."""
    from ml.detect import desk_inventory, safe_door_state
    from ml.detect.synth import SynthObject

    clip, _ = synth_clip(duration_s=3.0, objects=[SynthObject("cash", "rect", (0.25, 0.35, 0.33, 0.40)),
                                                  SynthObject("safe_open", "rect", (0.70, 0.65, 0.95, 0.85))])
    for k, t in enumerate(clip.times):
        if t >= 1.0:
            clip.frames[k] = np.zeros_like(clip.frames[k])
    dets = detect(clip)
    assert {d.label for d in dets if d.t < 1.0} >= {"cash", "safe_open"}
    assert not [d for d in dets if d.t >= 1.0]
    st = last_stats()
    assert st["coverage"]["fraction"] == 1.0                      # кадры проанализированы ...
    assert not any(k in st for k in ("absent", "absence", "objects_absent"))   # ... но отсутствие не заявлено
    zones = [{"name": "desk", "polygon": [(0.1, 0.3), (0.9, 0.3), (0.9, 0.6), (0.1, 0.6)]},
             {"name": "safe", "polygon": [(0.65, 0.62), (0.98, 0.62), (0.98, 0.88), (0.65, 0.88)]}]
    inv = desk_inventory(clip, 2.0, zones, config={"inventory": {"mode": "rare_open_vocab"}})
    assert all(it.state == "unknown" for it in inv.items.values())
    assert safe_door_state(clip, 2.0, zones, config={"inventory": {"mode": "rare_open_vocab"}}).state == "unknown"
    assert desk_inventory(clip, 0.4, zones, config={"inventory": {"mode": "rare_open_vocab"}}).present("cash")
