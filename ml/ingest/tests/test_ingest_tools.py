from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml.ingest import load_video
from ml.ingest.augment import (PERTURBATION_NOTE, PRESETS, Aug, ClipAugmenter, apply, augment_file, homography,
                               main as augment_main, output_size, resample_indices, transform_bbox, transform_points)
from ml.ingest.contact_sheet import main as sheet_main
from ml.ingest.contact_sheet import make_contact_sheet
from ml.ingest.names import find_duplicates, normalize_video_id
from ml.ingest.synthetic import read_index_code, render_scene

from .conftest import FPS, SIZE, needs_ffmpeg


@pytest.fixture()
def frame() -> np.ndarray:
    return render_scene(7, 0.3, SIZE)


# ---------------------------------------------------------------- augment (numpy)


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_presets_keep_shape_and_dtype(frame: np.ndarray, name: str) -> None:
    out = apply(frame, PRESETS[name])
    h, w = frame.shape[:2]
    exp_w, exp_h = output_size(PRESETS[name], w, h)
    assert out.shape == (exp_h, exp_w, 3) and out.dtype == np.uint8 and out.flags["C_CONTIGUOUS"]
    assert PRESETS[name].kind == "perturbation"
    if PRESETS[name].is_geometric or name in ("dark", "bright", "blur", "noise", "jpeg"):
        assert not np.array_equal(out, frame)


def test_identity_is_noop(frame: np.ndarray) -> None:
    assert np.array_equal(apply(frame, Aug()), frame)
    assert np.allclose(homography(Aug(), 100, 200), np.eye(3))


@pytest.mark.parametrize("name", ["angle_left", "angle_right", "high", "low", "rot_cw", "zoom", "mirror", "angle_jury"])
def test_points_follow_image(name: str) -> None:
    """Геометрия кадра и разметки согласована: точка-метка переезжает туда, куда говорит transform_points."""
    w, h = SIZE
    img = np.zeros((h, w, 3), np.uint8)
    px, py = 0.4, 0.6
    img[int(py * h) - 3 : int(py * h) + 4, int(px * w) - 3 : int(px * w) + 4] = 255
    out = apply(img, Aug(**{**PRESETS[name].to_dict(), "border": "constant"}))
    oh, ow = out.shape[:2]
    ys, xs = np.nonzero(out[..., 0] > 128)
    (ex, ey), = transform_points([(px + 0.5 / w, py + 0.5 / h)], PRESETS[name], w, h)
    assert abs(xs.mean() / ow - ex) < 0.02 and abs(ys.mean() / oh - ey) < 0.02


def test_bbox_transform_and_visibility() -> None:
    w, h = SIZE
    box = transform_bbox((0.4, 0.4, 0.6, 0.6), PRESETS["mirror"], w, h)
    assert box is not None and box[0] == pytest.approx(0.4, abs=0.01) and box[2] == pytest.approx(0.6, abs=0.01)
    assert transform_bbox((0.0, 0.0, 0.05, 0.05), Aug(zoom=0.5, center=(0.75, 0.75)), w, h) is None
    z = transform_bbox((0.4, 0.4, 0.6, 0.6), Aug(zoom=0.8), w, h)
    assert z is not None and (z[2] - z[0]) > 0.2


def test_clip_augmenter_is_consistent_per_clip(frame: np.ndarray) -> None:
    aug1 = ClipAugmenter(seed=3).sample()
    aug2 = ClipAugmenter(seed=3).sample()
    assert aug1 == aug2
    frames, aug = ClipAugmenter(seed=5).apply_frames([frame, frame])
    assert all(f.shape == frame.shape for f in frames) and isinstance(aug, Aug)


def test_resample_indices_30_to_25() -> None:
    times = [i / 30 for i in range(90)]
    idx = resample_indices(times, 25.0)
    assert len(idx) == 75 and idx == sorted(idx) and idx[0] == 0
    assert max(abs(times[k] - j / 25) for j, k in enumerate(idx)) <= 1 / 60 + 1e-9


# ---------------------------------------------------------------- augment (файлы)


@needs_ffmpeg
@pytest.mark.parametrize("name", ["angle_left", "rot_cw", "mirror", "high"])
def test_ffmpeg_engine_matches_numpy(tmp_path: Path, cfr_video: str, name: str) -> None:
    out = augment_file(cfr_video, str(tmp_path / f"{name}.mov"), PRESETS[name], engine="ffmpeg")
    base = load_video(cfr_video, config={"storage": {"mode": "raw"}}).frame_at(0.0)
    ff = load_video(out).frame_at(0.0)
    assert ff.shape == base.shape
    diff_np = np.abs(ff.astype(int) - apply(base, PRESETS[name]).astype(int)).mean()
    diff_orig = np.abs(ff.astype(int) - base.astype(int)).mean()
    assert diff_np < 2.0 and diff_np < diff_orig / 3


def test_python_engine_and_fps_change(tmp_path: Path, cfr_video: str) -> None:
    out = augment_file(cfr_video, str(tmp_path / "py25.mov"), Aug(name="x", fps=25.0, mirror=True), engine="python")
    clip = load_video(out, sample_fps=0.0)
    assert clip.fps == pytest.approx(25.0, rel=1e-3)
    assert len(clip) == 75 and (clip.width, clip.height) == SIZE


@needs_ffmpeg
def test_augment_cli_all(tmp_path: Path, cfr_video: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert augment_main([cfr_video, str(tmp_path / "aug.mov"), "--preset", "all"]) == 0
    outputs = json.loads(capsys.readouterr().out)
    assert len(outputs) >= 8
    for item in outputs:
        assert item["kind"] == "perturbation" and item["note"] == PERTURBATION_NOTE
        clip = load_video(item["output"], sample_fps=1.0)
        assert (clip.width, clip.height) == output_size(PRESETS[item["preset"]], *SIZE)
        expected_fps = 25.0 if item["preset"] == "fps25" else FPS
        assert clip.fps == pytest.approx(expected_fps, rel=0.02)


# ---------------------------------------------------------------- angle_jury (perturbation)


def test_angle_jury_is_landscape_perturbation(frame: np.ndarray) -> None:
    aug = PRESETS["angle_jury"]
    assert aug.kind == "perturbation" and aug.out_orientation == "landscape"
    h, w = frame.shape[:2]  # портрет
    out = apply(frame, aug)
    assert out.shape[1] > out.shape[0] and out.shape[:2] == (w, h)
    land = np.ascontiguousarray(np.rot90(frame))  # ландшафтный вход остаётся ландшафтом
    assert apply(land, aug).shape == land.shape
    box = transform_bbox((0.3, 0.5, 0.6, 0.65), aug, w, h)
    assert box is not None and all(0.0 <= v <= 1.0 for v in box)


@needs_ffmpeg
def test_angle_jury_ffmpeg_matches_numpy(tmp_path: Path, cfr_video: str) -> None:
    aug = PRESETS["angle_jury"]
    out = augment_file(cfr_video, str(tmp_path / "jury.mov"), aug, engine="ffmpeg")
    base = load_video(cfr_video, config={"storage": {"mode": "raw"}}).frame_at(0.0)
    ff = load_video(out).frame_at(0.0)
    npy = apply(base, aug)
    assert ff.shape == npy.shape and ff.shape[1] > ff.shape[0]
    assert np.abs(ff.astype(int) - npy.astype(int)).mean() < 3.0


def test_angle_jury_python_engine(tmp_path: Path, cfr_video: str) -> None:
    out = augment_file(cfr_video, str(tmp_path / "jury_py.mov"), PRESETS["angle_jury"], engine="python")
    clip = load_video(out)
    assert (clip.width, clip.height) == (SIZE[1], SIZE[0])


def test_random_aug_landscape_opt_in() -> None:
    rng = np.random.default_rng(0)
    assert all(ClipAugmenter(seed=k).sample().out_orientation is None for k in range(20))
    from ml.ingest.augment import random_aug

    assert any(random_aug(rng, 1.0, p_landscape=0.5).out_orientation == "landscape" for _ in range(20))


# ---------------------------------------------------------------- contact sheet / names


def test_contact_sheet_is_written_locally_only(tmp_path: Path, cfr_video: str, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "sheet.jpg"
    assert sheet_main([cfr_video, "--t", "1.5", "--window", "1", "--step", "0.5", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    meta = json.loads(printed)
    assert out.is_file() and out.stat().st_size > 1000
    assert meta["times"] == pytest.approx([0.5, 1.0, 1.5, 2.0, 2.5]) and meta["highlight_index"] == 2
    assert "base64" not in printed and len(printed) < 500  # в stdout только метаданные
    import cv2

    img = cv2.imread(str(out))
    assert img is not None and img.shape[1] == meta["size"][0]


def test_contact_sheet_overview(tmp_path: Path, cfr_video: str) -> None:
    meta = make_contact_sheet(cfr_video, str(tmp_path / "all.jpg"), every=1.0)
    assert meta["times"] == pytest.approx([0.0, 1.0, 2.0]) and meta["highlight_index"] is None


@pytest.mark.parametrize("raw,expected", [("Х-3.MOV", "X-3"), ("X-1.mov", "X-1"), ("х_47.mp4", "X-47"),
                                          ("/data/Х – 5.mov", "X-5"), ("X-10", "X-10")])
def test_normalize_video_id(raw: str, expected: str) -> None:
    assert normalize_video_id(raw) == expected


def test_find_duplicates(tmp_path: Path) -> None:
    a, b, c = tmp_path / "a.mov", tmp_path / "b.mov", tmp_path / "c.mov"
    a.write_bytes(b"same-bytes")
    b.write_bytes(b"same-bytes")
    c.write_bytes(b"other-byte")
    groups = find_duplicates([str(a), str(b), str(c)])
    assert list(groups.values()) == [[str(a), str(b)]]


def test_synthetic_code_roundtrip() -> None:
    for n in (0, 1, 777, 16383):
        assert read_index_code(render_scene(n, 0.0, SIZE)) == n
