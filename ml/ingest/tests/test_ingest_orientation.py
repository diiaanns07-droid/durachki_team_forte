"""Портрет и ландшафт во всех функциях модуля (тестовое видео «под другим углом» может быть горизонтальным)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ml.ingest import load_video
from ml.ingest.augment import PRESETS, apply, augment_file, transform_points
from ml.ingest.contact_sheet import make_contact_sheet
from ml.ingest.quality import frame_metrics
from ml.ingest.synthetic import read_index_code, render_scene, write_test_video

from .conftest import FFMPEG, FPS, ffmpeg, needs_ffmpeg

ORIENTATIONS = {"portrait": (240, 432), "landscape": (432, 240)}


@pytest.fixture(scope="module", params=sorted(ORIENTATIONS))
def oriented(request: pytest.FixtureRequest, media_dir: Path) -> tuple[str, tuple[int, int]]:
    size = ORIENTATIONS[request.param]
    path = media_dir / f"orient_{request.param}.mov"
    if not path.exists():
        write_test_video(str(path), duration_s=2.0, fps=FPS, size=size)
    return str(path), size


@pytest.mark.parametrize("backend", ["pyav", "opencv"])
def test_load_and_exact_frames(oriented: tuple[str, tuple[int, int]], backend: str) -> None:
    path, (w, h) = oriented
    clip = load_video(path, backend=backend)
    assert (clip.width, clip.height) == (w, h)
    assert clip.frame(0).shape == (h, w, 3)
    assert np.allclose(clip.times, [read_index_code(clip.frame(i)) / FPS for i in range(len(clip))], atol=1e-6)
    img, ts = clip.frame_at_ts(1.2)
    assert img.shape == (h, w, 3) and read_index_code(img) == 36 and ts == pytest.approx(1.2)


def test_max_side_keeps_orientation(oriented: tuple[str, tuple[int, int]]) -> None:
    path, (w, h) = oriented
    clip = load_video(path, config={"storage": {"max_side": 216}})
    sh, sw = clip.frame(0).shape[:2]
    assert max(sw, sh) == 216 and (sw > sh) == (w > h)
    assert clip.frame_at(0.0).shape[:2] == (h, w)


@needs_ffmpeg
@pytest.mark.parametrize("rotation", ["-90", "90", "180"])
def test_rotation_both_orientations(tmp_path: Path, oriented: tuple[str, tuple[int, int]], rotation: str) -> None:
    path, (w, h) = oriented
    out = tmp_path / "rot.mov"
    ffmpeg("-display_rotation", rotation, "-i", path, "-c", "copy", str(out))
    clip = load_video(str(out))
    exp = (w, h) if rotation == "180" else (h, w)
    assert (clip.width, clip.height) == exp
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", str(out), "-frames:v", "1", "-f", "rawvideo",
                          "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
    ref = np.frombuffer(raw, np.uint8).reshape(exp[1], exp[0], 3)
    assert np.abs(clip.frame_at(0.0).astype(int) - ref.astype(int)).mean() < 4.0


def test_quality_metrics_orientation_invariant() -> None:
    portrait = render_scene(3, 0.1, ORIENTATIONS["portrait"], code=False)
    landscape = np.ascontiguousarray(np.rot90(portrait))  # тот же контент, повёрнутый
    cfg = {"analysis_short_side": 120}
    mp, gp = frame_metrics(portrait, cfg)
    ml_, gl = frame_metrics(landscape, cfg)
    assert min(gp.shape) == min(gl.shape) == 120
    assert ml_["brightness"] == pytest.approx(mp["brightness"], abs=0.01)
    assert ml_["sharpness"] == pytest.approx(mp["sharpness"], rel=0.2)


@pytest.mark.parametrize("name", ["angle_left", "high", "rot_cw", "zoom", "mirror", "dark"])
def test_augment_both_orientations(oriented: tuple[str, tuple[int, int]], name: str) -> None:
    _, (w, h) = oriented
    img = render_scene(5, 0.2, (w, h))
    out = apply(img, PRESETS[name])
    assert out.shape == img.shape and out.dtype == np.uint8
    (x, y), = transform_points([(0.5, 0.5)], PRESETS[name], w, h)
    assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


@needs_ffmpeg
def test_augment_ffmpeg_both_orientations(tmp_path: Path, oriented: tuple[str, tuple[int, int]]) -> None:
    path, (w, h) = oriented
    out = augment_file(path, str(tmp_path / "a.mov"), PRESETS["angle_right"], engine="ffmpeg")
    base = load_video(path, config={"storage": {"mode": "raw"}}).frame_at(0.0)
    ff = load_video(out).frame_at(0.0)
    assert ff.shape == (h, w, 3)
    assert np.abs(ff.astype(int) - apply(base, PRESETS["angle_right"]).astype(int)).mean() < 2.0


def test_contact_sheet_both_orientations(tmp_path: Path, oriented: tuple[str, tuple[int, int]]) -> None:
    import cv2

    path, (w, h) = oriented
    meta = make_contact_sheet(path, str(tmp_path / "s.jpg"), t=1.0, window=0.5, step=0.5, cols=3, tile_width=120)
    sheet = cv2.imread(str(tmp_path / "s.jpg"))
    tile_h = round(h * 120 / w)
    assert sheet is not None and sheet.shape[0] > tile_h and meta["times"] == pytest.approx([0.5, 1.0, 1.5])
    json.dumps(meta)  # только метаданные, сериализуемо
