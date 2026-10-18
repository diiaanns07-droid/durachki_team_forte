"""Время в отчётах VLM = фактический PTS показанного кадра (A04 frame_at_ts), а не запрошенное время."""
import json

import numpy as np
import pytest

from ml.vlm._contract import Event
from ml.vlm.backends.mock import MockBackend
from ml.vlm.frames import frames_at, get_frame_ts
from ml.vlm.verifier import verify_events


class PtsClip:
    """Как ml.ingest: кадры только на PTS-сетке (шаг 0.4 с, «VFR»); frame_at_ts отдаёт ближайший кадр и его PTS."""
    path, fps, width, height = "synthetic", 2.5, 464, 848

    def __init__(self, duration=12.0, step=0.4):
        self.duration = duration
        self.pts = [round(k * step, 4) for k in range(int(duration / step) + 1)]
        self.times = list(self.pts)
        self.calls = []

    def _img(self, k):
        f = np.full((self.height, self.width, 3), 40, np.uint8)
        f[:20, :20] = k % 255
        return f

    def frame_at_ts(self, t):
        self.calls.append(t)
        k = min(range(len(self.pts)), key=lambda j: abs(self.pts[j] - t))
        return self._img(k), self.pts[k]

    def frame_at(self, t):
        return self.frame_at_ts(t)[0]

    def frame(self, i):
        return self._img(i)


class NoTsClip(PtsClip):
    """Клип без frame_at_ts: frame_at отдаёт кадр, но не сообщает его PTS."""
    frame_at_ts = None

    def frame_at(self, t):
        k = min(range(len(self.pts)), key=lambda j: abs(self.pts[j] - t))
        return self._img(k)


class SampledOnlyClip(PtsClip):
    frame_at_ts = None

    def frame_at(self, t):
        raise RuntimeError("no exact seek")


def test_get_frame_ts_prefers_actual_pts():
    c = PtsClip()
    f, ts = get_frame_ts(c, 1.1)
    assert ts == pytest.approx(1.2) and f is not None
    _, ts2 = get_frame_ts(NoTsClip(), 1.1)
    assert ts2 == pytest.approx(1.1)  # PTS неизвестен — запрошенное время
    _, ts3 = get_frame_ts(SampledOnlyClip(), 1.1)
    assert ts3 == pytest.approx(1.2)  # ближайший сэмплированный кадр — его время


def test_frames_at_dedupes_same_frame():
    frames, ts = frames_at(PtsClip(), [1.0, 1.05, 1.1, 2.0])
    assert ts == [pytest.approx(1.2), pytest.approx(2.0)] or ts == [pytest.approx(0.8), pytest.approx(1.2),
                                                                     pytest.approx(2.0)]
    assert len(frames) == len(ts) == len(set(ts))


def test_verify_reports_actual_pts(cfg):
    clip = PtsClip()
    ans = json.dumps({"visible": True, "type_id": "V05", "observed_facts": [], "confidence": 0.9,
                      "best_frame": 2, "visible_frames": [2, 3], "uncertain_because": ""})
    be = MockBackend(responses=[ans])
    ev = Event("V05", 4.1, 4.3, 4.2, 0.5, "rules")  # окно 3.1..5.3 -> запросы не на сетке PTS
    out = verify_events(clip, [ev], 30.0, cfg=cfg, backend=be)[0]
    v = out.extra["vlm"]
    assert all(any(abs(t - p) < 1e-6 for p in clip.pts) for t in v["frame_times"])  # только реальные PTS
    assert len(v["frame_times"]) == len(set(v["frame_times"])) == be.calls[0]["n_images"]
    assert v["t_best"] == v["frame_times"][1] and set(v["t_visible"]) <= set(v["frame_times"])
    assert "1: " + format(v["frame_times"][0], ".1f") in be.calls[0]["prompt"]  # в промпте — фактическое время


def test_qa_and_discover_use_actual_pts(cfg):
    clip = PtsClip()
    cfg["verify"]["mode"] = "qa"
    be = MockBackend(mode="confirm")
    out = verify_events(clip, [Event("V05", 5.9, 6.1, 6.1, 0.5, "rules")], 30.0, cfg=cfg, backend=be)[0]
    ft = out.extra["vlm"]["frame_times"]
    assert all(any(abs(t - p) < 1e-6 for p in clip.pts) for t in ft.values())
    from ml.vlm.discoverer import discover_events
    cfg["discover"]["reverify"] = False
    found = discover_events(clip, 60.0, cfg=cfg, backend=MockBackend(mode="phone"), force=True)
    assert found and all(any(abs(e.t_peak - p) < 1e-6 for p in clip.pts) for e in found)
