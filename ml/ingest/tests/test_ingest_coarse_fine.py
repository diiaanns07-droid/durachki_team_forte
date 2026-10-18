from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.ingest import coarse_fine as CF
from ml.ingest import load_video
from ml.ingest.synthetic import read_index_code

from .conftest import FPS, SIZE


@pytest.fixture(scope="module")
def clip(cfr_video: str):  # type: ignore[no-untyped-def]
    return load_video(cfr_video)


def test_disabled_by_default() -> None:
    assert CF.enabled() is False
    assert CF.enabled({"coarse_fine": {"enabled": True}}) is True


@pytest.mark.parametrize("kw", [{"step_s": 1.0}, {"step_s": 0.4}, {"n_frames": 4}, {"n_frames": 1}])
def test_coarse_covers_whole_duration(clip, kw) -> None:  # type: ignore[no-untyped-def]
    idx = CF.coarse_indices(clip, **kw)
    assert idx[0] == 0 and idx[-1] == len(clip) - 1 and idx == sorted(set(idx))
    t = np.asarray([clip.times[i] for i in idx])
    if "step_s" in kw and len(t) > 2:
        assert np.max(np.diff(t)) <= kw["step_s"] + 0.2 + 1e-6


def test_merge_windows_debounce_and_clamp() -> None:
    evs = [(0.2, 0.5, 0.9), (1.0, 1.2, 0.3), (5.0, 5.1, 0.5), {"t_start": 9.5, "t_end": 9.9, "confidence": 0.7}]
    wins = CF.merge_windows(evs, duration=10.0, pad_s=0.5, merge_gap_s=0.6, min_window_s=1.0)
    assert [(round(w.t0, 2), round(w.t1, 2)) for w in wins] == [(0.0, 1.7), (4.5, 5.6), (9.0, 10.0)]
    assert wins[0].score == 0.9 and len(wins[0].meta["members"]) == 2
    top = CF.merge_windows(evs, duration=10.0, pad_s=0.5, merge_gap_s=0.6, max_windows=2)
    assert [w.score for w in top] == [0.9, 0.7] and top[0].t0 < top[1].t0


def test_event_objects_are_accepted() -> None:
    try:
        from contracts.sentinel_contracts import Event  # type: ignore
    except ImportError:
        from contracts.sentinel_contracts import Event
    ev = Event(type_id="V02", t_start=2.0, t_end=3.0, t_peak=2.4, confidence=0.6, source="rules")
    (w,) = CF.merge_windows([ev], duration=10.0, pad_s=0.0, min_window_s=0.5)
    assert (w.t0, w.t1, w.score) == (2.0, 3.0, 0.6) and w.meta["type_id"] == "V02"


def test_dense_times_snapped_unique_and_bounded(clip) -> None:  # type: ignore[no-untyped-def]
    w = CF.Window(1.0, 2.0)
    times = CF.dense_times(clip, w, fps=10.0, max_frames=8)
    assert len(times) <= 8 and times == sorted(set(times))
    assert all(t in clip.source_times for t in times)
    assert all(1.0 - 1 / FPS <= t <= 2.0 + 1 / FPS for t in times)
    many = CF.dense_times(clip, w, fps=200.0, max_frames=0)  # плотнее fps источника -> без дублей
    assert len(many) == len(set(many)) <= 32


def test_read_window_exact_frames(clip) -> None:  # type: ignore[no-untyped-def]
    r = CF.read_window(clip, CF.Window(0.5, 1.5), fps=4.0, max_frames=10)
    assert r.complete and r.timestamps == sorted(set(r.timestamps))
    for f in r.frames:
        assert f.image.shape == (SIZE[1], SIZE[0], 3)
        assert read_index_code(f.image) == round(f.timestamp_sec * FPS)


def test_plan_with_budget(clip) -> None:  # type: ignore[no-untyped-def]
    cands = [(0.6, 0.9, 0.8), (2.2, 2.4, 0.5)]
    pl = CF.plan(clip, cands, budget_frames=12, coarse_share=0.4, window_pad_s=0.3, min_window_s=0.5, merge_gap_s=0.2)
    assert pl.n_frames <= 12 and len(pl.coarse_indices) >= 2 and len(pl.windows) == 2
    assert all(len(d) >= 2 for d in pl.dense_times)
    merged = CF.plan(clip, cands, budget_frames=12, window_pad_s=0.3, merge_gap_s=1.0)  # дребезг -> одно окно
    assert len(merged.windows) == 1 and merged.windows[0].score == 0.8
    tiny = CF.plan(clip, cands, budget_frames=6, coarse_share=0.6, window_pad_s=0.3, merge_gap_s=0.2)
    assert tiny.n_frames <= 6 and (not tiny.windows or tiny.windows[0].score == 0.8)


def test_run_without_candidates_uses_activity(tmp_path: Path) -> None:
    import cv2

    path = tmp_path / "motion.mov"
    w, h = SIZE
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for i in range(120):
        img = np.full((h, w, 3), 90, np.uint8)
        if 60 <= i < 80:
            cv2.rectangle(img, (20 + (i - 60) * 6, 150), (80 + (i - 60) * 6, 260), (255, 255, 255), -1)
        vw.write(img)
    vw.release()
    clip = load_video(str(path))
    pl, reads = CF.run(clip, None, activity_top_k=1, activity_window_s=1.0, window_pad_s=0.2)
    assert len(pl.windows) == 1 and pl.windows[0].source == "activity"
    w = pl.windows[0]
    assert w.t0 <= 2.0 and w.t1 >= 2.6 and w.length <= 2.0 and reads[0].frames  # движение: кадры 60..79


def test_window_reports_gap(media_dir: Path, cfr_video: str) -> None:
    from .conftest import FFMPEG, ffmpeg

    if FFMPEG is None:
        pytest.skip("нужен ffmpeg")
    out = media_dir / "cf_holed.mov"
    if not out.exists():
        ffmpeg("-i", cfr_video, "-vf", r"select='not(between(n\,30\,60))'", "-fps_mode", "vfr",
               "-c:v", "mpeg4", "-q:v", "2", str(out))
    clip = load_video(str(out))
    r = CF.read_window(clip, CF.Window(1.2, 1.8))
    assert r.gaps and not r.complete  # «кадров нет» = неизвестно, а не «нарушений нет»
