from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.ingest.errors import InvalidDurationError, NoVideoStreamError, VideoNotFoundError, VideoProbeError
from ml.ingest.orientation import Orientation, from_display_matrix, matrix_from_ffprobe
from ml.ingest.probe import main as probe_main
from ml.ingest.probe import probe, validate

from .conftest import FFMPEG, FFPROBE, FPS, SIZE, ffmpeg, needs_ffmpeg


@pytest.mark.parametrize("backends", [("ffprobe",), ("pyav",), ("opencv",)])
def test_probe_backends_agree(cfr_video: str, backends: tuple[str, ...]) -> None:
    if backends == ("ffprobe",) and FFPROBE is None:
        pytest.skip("нет ffprobe")
    info = validate(probe(cfr_video, backends=backends))
    assert info["has_video"] and info["source"] == backends[0]
    assert (info["display_width"], info["display_height"]) == SIZE
    assert info["rotation"] == 0
    assert info["fps"] == pytest.approx(FPS, rel=1e-3)
    assert info["duration"] == pytest.approx(3.0, abs=0.1)


def test_missing_and_empty_files(tmp_path: Path) -> None:
    with pytest.raises(VideoNotFoundError) as e:
        probe(str(tmp_path / "nope.mov"))
    assert e.value.code == "file_not_found"
    empty = tmp_path / "empty.mov"
    empty.write_bytes(b"")
    with pytest.raises(VideoProbeError):
        probe(str(empty))


def test_garbage_file(tmp_path: Path) -> None:
    junk = tmp_path / "junk.mov"
    junk.write_bytes(b"not a video at all " * 500)
    with pytest.raises(VideoProbeError) as e:
        probe(str(junk))
    assert "видео" in e.value.message


def test_duration_validation(cfr_video: str) -> None:
    info = probe(cfr_video)
    with pytest.raises(InvalidDurationError) as e:
        validate(dict(info), min_duration=5.0, max_duration=900.0)
    assert e.value.code == "duration_out_of_range" and "3.0" in e.value.message
    with pytest.raises(InvalidDurationError):
        validate(dict(info), min_duration=0.5, max_duration=2.0)


@needs_ffmpeg
def test_audio_only_is_rejected(tmp_path: Path) -> None:
    out = tmp_path / "audio.mov"
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "aac", str(out))
    with pytest.raises(NoVideoStreamError):
        validate(probe(str(out)))


@needs_ffmpeg
@pytest.mark.parametrize("backend", ["ffprobe", "pyav", "opencv"])
def test_probe_rotation(tmp_path: Path, cfr_video: str, backend: str) -> None:
    out = tmp_path / "rot.mov"
    ffmpeg("-display_rotation", "-90", "-i", cfr_video, "-c", "copy", str(out))  # как у iPhone-портрета
    if backend == "ffprobe" and FFPROBE is None:
        pytest.skip("нет ffprobe")
    info = probe(str(out), backends=(backend,))
    assert info["rotation"] == 90
    assert (info["width"], info["height"]) == SIZE
    assert (info["display_width"], info["display_height"]) == (SIZE[1], SIZE[0])


def test_display_matrix_parsing() -> None:
    text = "\n00000000:            0       65536           0\n00000001:       -65536           0           0\n00000002:            0           0  1073741824\n"
    m = matrix_from_ffprobe(text)
    assert m == [0, 65536, 0, -65536, 0, 0, 0, 0, 1 << 30]
    assert from_display_matrix(m) == Orientation(90, False, "displaymatrix")
    assert from_display_matrix([-65536, 0, 0, 0, -65536, 0, 0, 0, 1 << 30]).rotation_cw == 180
    assert from_display_matrix([-65536, 0, 0, 0, 65536, 0, 0, 0, 1 << 30]) == Orientation(0, True, "displaymatrix")
    assert from_display_matrix([65536, 0, 0, 0, 65536, 0, 0, 0, 1 << 30]).is_identity


def test_probe_cli_reports_error_as_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert probe_main([str(tmp_path / "missing.mov")]) == 2
    out = json.loads(capsys.readouterr().out)
    assert out["error"]["code"] == "file_not_found"
