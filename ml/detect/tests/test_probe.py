"""probe для L2: отчёт с latency/VRAM/классами/абляцией inventory; картинки только в локальной папке с .gitignore."""
from __future__ import annotations

import json

import pytest

from ml.detect.probe import main
from ml.detect.synth import SynthObject, synth_clip


@pytest.fixture
def frames_dir(tmp_path):
    import cv2

    clip, _ = synth_clip(duration_s=1.0, sample_fps=4.0, objects=[
        SynthObject("person", "ellipse", (0.30, 0.60, 0.60, 0.85)),
        SynthObject("cup", "ellipse", (0.20, 0.35, 0.26, 0.40)),
        SynthObject("cash", "rect", (0.60, 0.35, 0.68, 0.40)),
        SynthObject("safe_open", "rect", (0.70, 0.65, 0.95, 0.85)),
    ])
    d = tmp_path / "frames" / "X-1"
    d.mkdir(parents=True)
    for k, f in enumerate(clip.frames):
        cv2.imwrite(str(d / f"{k:04d}.png"), f)
    zones = tmp_path / "zones.json"
    zones.write_text(json.dumps([{"name": "desk", "polygon": [[0.1, 0.3], [0.9, 0.3], [0.9, 0.6], [0.1, 0.6]]},
                                 {"name": "safe", "polygon": [[0.65, 0.62], [0.98, 0.62], [0.98, 0.88], [0.65, 0.88]]}]))
    return tmp_path / "frames", zones


def test_probe_report_and_local_vis(mock_mode, frames_dir, tmp_path, capsys):
    frames, zones = frames_dir
    out = tmp_path / "probe_out"
    assert main(["--frames", str(frames), "--out", str(out), "--zones", str(zones), "--batch", "2"]) == 0
    rep = json.loads((out / "probe_report.json").read_text())
    assert (out / ".gitignore").read_text().strip() == "*"
    assert rep["mock"] is True and any("SENTINEL_MOCK" in w for w in rep["warnings"])
    assert rep["n_frames"] == 4 and rep["frame_shapes"] == {"464x848": 4}
    assert rep["detect"]["latency_ms_per_frame"]["n"] >= 1
    assert rep["detect"]["classes"]["person"]["n_frames"] == 4
    base, rare = rep["inventory"]["baseline"], rep["inventory"]["rare_open_vocab"]
    assert base["items"]["cup"] == {"present": 4}
    assert base["items"]["cash"] == {"unknown": 4} and rare["items"]["cash"] == {"present": 4}
    assert base["safe"] == {"unknown": 4} and rare["safe"] == {"open": 4}
    assert rare["effective_mode"] == {"rare_open_vocab": 4}
    assert len(list((out / "vis" / "detect").glob("*.jpg"))) == 4
    assert len(list((out / "vis" / "rare_open_vocab").glob("*.jpg"))) == 4
    assert "inventory[rare_open_vocab]" in capsys.readouterr().out


def test_probe_no_vis_and_bad_mode(mock_mode, frames_dir, tmp_path):
    frames, _ = frames_dir
    out = tmp_path / "o2"
    assert main(["--frames", str(frames), "--out", str(out), "--modes", "baseline", "--no-vis"]) == 0
    assert not (out / "vis").exists()
    with pytest.raises(SystemExit):
        main(["--frames", str(frames), "--out", str(out), "--modes", "everything"])


def test_probe_from_video(mock_mode, tmp_path):
    from ml.detect.synth import make_synth_video

    v = tmp_path / "v.mp4"
    make_synth_video(v, duration_s=2.0, fps=10.0)
    out = tmp_path / "o3"
    assert main(["--video", str(v), "--fps", "2", "--out", str(out), "--modes", "baseline", "--no-vis"]) == 0
    assert json.loads((out / "probe_report.json").read_text())["n_frames"] == 4
