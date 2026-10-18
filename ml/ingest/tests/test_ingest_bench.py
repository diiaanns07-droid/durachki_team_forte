from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.ingest import bench
from ml.ingest.synthetic import write_test_video

from .conftest import FFMPEG, ffmpeg


def test_parse_variant() -> None:
    assert bench.parse_variant("base") == ("base", {})
    name, ov = bench.parse_variant("skip:decode.skip_nonref=true,storage.max_side=640")
    assert name == "skip" and ov == {"decode": {"skip_nonref": True}, "storage": {"max_side": 640}}


def test_bench_end_to_end(tmp_path: Path, cfr_video: str, capsys: pytest.CaptureFixture[str]) -> None:
    vids = tmp_path / "videos"
    vids.mkdir()
    write_test_video(str(vids / "X-1.mov"), duration_s=2.0, fps=30.0, size=(240, 432))
    write_test_video(str(vids / "Х-2.mov"), duration_s=2.0, fps=30.0, size=(432, 240))  # кириллица + ландшафт
    if FFMPEG:
        ffmpeg("-i", cfr_video, "-vf", r"select='not(between(n\,30\,60))'", "-fps_mode", "vfr", "-c:v", "mpeg4",
               "-q:v", "2", str(vids / "X-3.mov"))
    (vids / "X-4.mov").write_bytes(b"broken" * 100)
    labels = tmp_path / "labels.csv"
    labels.write_text("video_id,file,label,t_sec\nX-1,X-1.mov,violation,1\nX-2,Х-2.mov,violation,1.5\n"
                      "X-3,X-3.mov,violation,1.5\n", encoding="utf-8")
    out = tmp_path / "bench.json"
    code = bench.main(["--videos", str(vids), "--out", str(out), "--labels", str(labels), "--variant", "base",
                       "--variant", "cv:decode.backend=opencv", "--probe-times", "8", "--coarse-fine"])
    printed = capsys.readouterr().out
    assert code == 1  # битый файл -> failed > 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    for name in ("base", "cv"):
        s = rep["variants"][name]["summary"]
        assert s["failed"] == 1 and s["errors"] and s["idempotent_all"] is True
        assert s["exact_offset_max"] is not None and s["sample_vs_exact_absdiff_max"] < 8.0
        recs = {r["video"]: r for r in rep["variants"][name]["videos"]}
        assert recs["X-1.mov"]["labels"][0]["in_coverage"] is True
        assert recs["X-1.mov"]["labels"][0]["t_frame"] == pytest.approx(1.0)
        assert recs["Х-2.mov"]["width"] == 432 and recs["Х-2.mov"]["labels"][0]["t_label"] == 1.5
        assert recs["X-1.mov"]["vram_mb"] is None and "coarse_fine" in recs["X-1.mov"]
        if FFMPEG:
            assert recs["X-3.mov"]["gaps"] == 1 and recs["X-3.mov"]["labels"][0]["in_coverage"] is False
    assert "base" in printed and "base64" not in out.read_text(encoding="utf-8")
