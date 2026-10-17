from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from eval.labels import GroundTruth
from eval.robustness import DISCLAIMER, compare, make_variant

from .conftest import make_pred, make_violation, write_synthetic_video


def test_compare_flip_and_shift():
    orig = {"X-1": make_pred("a", [make_violation(10.0)]), "X-2": make_pred("b", [])}
    var = {"mirror": {"X-1": make_pred("c", [make_violation(11.0)]), "X-2": make_pred("d", [])},
           "rot_p20": {"X-1": make_pred("e", []), "X-2": make_pred("f", [make_violation(4.0)])}}
    gt = {"X-1": GroundTruth("X-1", "violation", [10.0]), "X-2": GroundTruth("X-2", "no_violation", [])}
    res = compare(orig, var, gt, shift_tol_s=2.0)
    assert res["per_variant"]["mirror"]["flip_rate"] == 0.0
    assert res["per_variant"]["mirror"]["timecode_stable_rate"] == 1.0
    assert res["per_variant"]["rot_p20"]["flip_rate"] == 1.0
    assert res["per_variant"]["rot_p20"]["accuracy_vs_gt"] == 0.0
    assert res["summary"]["flip_rate"] == 0.5 and res["summary"]["synthetic"] is True
    assert res["summary"]["disclaimer"] == DISCLAIMER


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="нет ffmpeg")
@pytest.mark.parametrize("name", ["mirror", "rot_p10", "persp_left", "zoom80", "fps25"])
def test_make_variant_ffmpeg(tmp_path: Path, name: str):
    pytest.importorskip("cv2")
    from eval.common import ffprobe_info

    src = write_synthetic_video(tmp_path / "X-1.mp4", seconds=1.0, fps=10, size=(64, 96))
    dst = make_variant(src, tmp_path / name / "X-1.mov", name)
    info = ffprobe_info(dst)
    assert (info["width"], info["height"]) == (64, 96)
    if name == "fps25":
        assert abs(info["fps"] - 25) < 0.5


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="нет ffmpeg")
def test_environment_variants(tmp_path: Path):
    """Горизонтальное видео, поворот в метаданных, переменный fps (время по PTS должно сохраниться)."""
    pytest.importorskip("cv2")
    from eval.common import ffprobe_info

    src = write_synthetic_video(tmp_path / "X-1.mp4", seconds=3.0, fps=10, size=(64, 96))
    land = ffprobe_info(make_variant(src, tmp_path / "a" / "X-1.mov", "landscape_rot90"))
    assert land["width"] > land["height"]
    pad = ffprobe_info(make_variant(src, tmp_path / "b" / "X-1.mov", "landscape_pad"))
    assert pad["width"] > pad["height"]
    meta = ffprobe_info(make_variant(src, tmp_path / "c" / "X-1.mov", "rotmeta90"))
    assert abs(meta["rotation"]) == 90 and (meta["width"], meta["height"]) == (96, 64)   # с учётом rotation
    vfr_path = make_variant(src, tmp_path / "d" / "X-1.mov", "vfr_drop")
    assert abs(ffprobe_info(vfr_path)["duration"] - 3.0) < 0.4                            # PTS сохранены
    import subprocess

    pts = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time",
                          "-of", "csv=p=0", str(vfr_path)], capture_output=True, text=True).stdout.split()
    t = sorted(float(x.split(",")[0]) for x in pts if x.split(",")[0].strip())   # csv может дать «0.000,»
    gaps = {round(b - a, 2) for a, b in zip(t, t[1:])}
    assert len(t) < 30 and len(gaps) > 1                                  # кадры выброшены неравномерно = VFR
