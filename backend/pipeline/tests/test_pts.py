"""Переменный fps: время берётся из PTS, а не из индекс/fps (запасной декодер и кадры-доказательства)."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from backend.pipeline.media import SimpleClip, read_frame_at_pts


def _vfr_video(path) -> list[float]:
    av = pytest.importorskip("av")
    pts_s = [i / 30 for i in range(60)] + [2.0 + i / 10 for i in range(20)]  # 2 с по 30 fps, 2 с по 10 fps
    tb = Fraction(1, 1000)
    with av.open(str(path), "w") as out:
        st = out.add_stream("mpeg4", rate=30)
        st.width, st.height, st.pix_fmt = 64, 112, "yuv420p"
        st.codec_context.time_base = tb
        for t in pts_s:
            img = np.full((112, 64, 3), int(t * 50) % 256, np.uint8)  # яркость кодирует время
            fr = av.VideoFrame.from_ndarray(img, format="bgr24")
            fr.pts, fr.time_base = int(round(t * 1000)), tb
            for pkt in st.encode(fr):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)
    return pts_s


def test_simpleclip_uses_pts_on_vfr(tmp_path):
    path = tmp_path / "vfr.mp4"
    _vfr_video(path)
    clip = SimpleClip(str(path), sample_fps=5.0)
    assert clip.timebase == "pts"
    assert 3.8 <= clip.duration <= 4.3, clip.duration           # индекс/fps дал бы 80/30 ≈ 2.7 с
    late = [t for t in clip.times if t > 2.0]
    assert late and max(clip.times) > 3.5
    for t in clip.times:                                         # каждый таймстемп — реальный PTS кадра
        assert min(abs(t - k / 30) for k in range(60)) < 1e-3 or min(abs(t - (2 + k / 10)) for k in range(20)) < 1e-3


def test_frame_at_by_pts(tmp_path):
    path = tmp_path / "vfr.mp4"
    _vfr_video(path)
    frame, pts = read_frame_at_pts(str(path), 3.02)
    assert frame is not None and abs(pts - 3.0) < 1e-3
    assert abs(float(frame.mean()) - (3.0 * 50) % 256) < 12      # содержимое — кадр 3.0 с, а не 3.02×fps
    frame, pts = read_frame_at_pts(str(path), 0.5)
    assert abs(pts - 0.5) < 0.02
