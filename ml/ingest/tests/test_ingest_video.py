from __future__ import annotations

import pickle
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ml.ingest import load_video
from ml.ingest.errors import DecodeError, IngestError, VideoProbeError
from ml.ingest.synthetic import SyntheticClip, read_index_code, write_test_video
from ml.ingest.video import Clip, _NearestSampler

from .conftest import FFMPEG, FPS, SIZE, ffmpeg, needs_ffmpeg

BACKENDS = ["pyav", "opencv"]


def _code_times(clip: Clip) -> list[float]:
    return [read_index_code(clip.frame(i)) / FPS for i in range(len(clip))]


# ---------------------------------------------------------------- время = фактические PTS


@pytest.mark.parametrize("backend", BACKENDS)
def test_times_are_actual_pts(cfr_video: str, backend: str) -> None:
    clip = load_video(cfr_video, backend=backend)
    assert isinstance(clip, Clip) and clip.meta["backend"] == backend
    assert (clip.width, clip.height) == SIZE  # размер из декодированного кадра, не зашитый
    assert clip.times[0] == 0.0
    assert len(clip) == 15  # 3 с * 5 fps, узел 3.0 уже за последним кадром
    assert np.allclose(clip.times, _code_times(clip), atol=1e-6)
    assert np.allclose(np.diff(clip.times), 0.2, atol=1e-6)
    assert clip.meta["complete"] and not clip.meta["vfr"]
    assert len(clip.source_times) == 90


@pytest.mark.parametrize("rate,expected", [(2.0, 6), (10.0, 30), (0.0, 90)])
def test_sample_fps_configurable(cfr_video: str, rate: float, expected: int) -> None:
    clip = load_video(cfr_video, sample_fps=rate)
    assert len(clip) == expected
    assert np.allclose(clip.times, _code_times(clip), atol=1e-6)


def test_sample_fps_above_source_fps_has_no_duplicates(cfr_video: str) -> None:
    clip = load_video(cfr_video, sample_fps=60.0)
    assert len(clip) == 90 and len(set(clip.times)) == 90


def test_max_samples_guard(cfr_video: str) -> None:
    clip = load_video(cfr_video, config={"sampling": {"max_samples": 6}})
    assert len(clip) <= 7 and any("sample_fps снижен" in w for w in clip.warnings)


@needs_ffmpeg
@pytest.mark.parametrize("backend", BACKENDS)
def test_vfr_timestamps(vfr_video: str, backend: str) -> None:
    clip = load_video(vfr_video, backend=backend)
    src = np.asarray(clip.source_times)
    assert np.diff(src).std() > 0.005, "фикстура должна быть VFR"
    assert clip.meta["vfr"]
    # каждый сэмпл: его время == реальное время кадра по штрих-коду (а не index/fps)
    assert np.allclose(clip.times, _code_times(clip), atol=2e-3)
    # и это ближайший доступный кадр к узлу сетки
    for t in clip.times:
        k = round(t * 5) / 5
        assert abs(t - k) <= np.min(np.abs(src - k)) + 1e-6 or backend == "opencv"


def test_nearest_sampler_unit() -> None:
    s = _NearestSampler(5.0)
    picked = []
    for t in [0.0, 0.09, 0.19, 0.33, 0.41, 0.62, 0.66]:
        picked += [x[0] for x in s.push(t, None)]
    picked += [x[0] for x in s.finish(0.02)]
    assert picked == [0.0, 0.19, 0.41, 0.62]


# ---------------------------------------------------------------- seek / frame_at


@pytest.mark.parametrize("backend", BACKENDS)
def test_frame_at_returns_actual_timestamp(cfr_video: str, backend: str) -> None:
    clip = load_video(cfr_video, backend=backend)
    for t in [1.01, 0.5, 2.98, 0.0, 1.517, 0.05]:  # вперёд и назад (seek)
        img, ts = clip.frame_at_ts(t)
        n = read_index_code(img)
        assert n == round(t * FPS), (t, n)
        assert ts == pytest.approx(n / FPS, abs=1e-6)
        assert img.shape == (SIZE[1], SIZE[0], 3) and img.dtype == np.uint8
    ref = clip.read_exact(1.01)
    assert ref.requested_sec == 1.01 and ref.timestamp_sec == pytest.approx(1.0) and ref.offset_sec < 0
    assert clip.frame_at(1.01).shape == ref.image.shape  # контрактный метод — только картинка


@pytest.mark.parametrize("backend", BACKENDS)
def test_frame_at_out_of_range(cfr_video: str, backend: str) -> None:
    clip = load_video(cfr_video, backend=backend)
    img, ts = clip.frame_at_ts(99.0)
    assert read_index_code(img) == 89 and ts == pytest.approx(89 / FPS)
    img, ts = clip.frame_at_ts(-5.0)
    assert read_index_code(img) == 0 and ts == 0.0


def test_frames_at_keeps_request_order_and_snap_time(cfr_video: str) -> None:
    clip = load_video(cfr_video)
    req = [2.5, 0.3, 1.2, 0.3]
    refs = clip.frames_at(req)
    assert [r.requested_sec for r in refs] == req
    assert [read_index_code(r.image) for r in refs] == [75, 9, 36, 9]
    assert clip.snap_time(1.21) == pytest.approx(36 / FPS)
    assert clip.index_at(1.29) == 6


# ---------------------------------------------------------------- кэш / память


def test_lru_hits_and_bounds(cfr_video: str) -> None:
    clip = load_video(cfr_video, config={"storage": {"lru_size": 4}})
    for i in range(10):
        clip.frame(i)
    clip.frame(9)
    info = clip.cache_info()["samples"]
    assert info["items"] <= 4 and info["hits"] >= 1
    a = clip.frame(9)
    a[:] = 0  # copy_on_access: кэш нельзя испортить
    assert clip.frame(9).any()


def test_ram_cap_spills_to_disk(cfr_video: str) -> None:
    mem = load_video(cfr_video)
    disk = load_video(cfr_video, config={"storage": {"max_ram_mb": 0.05}})
    assert disk.meta["storage"]["disk_bytes"] > 0
    for i in (0, 7, len(mem) - 1):
        assert np.array_equal(mem.frame(i), disk.frame(i))
    disk.close()


def test_raw_storage_is_lossless_vs_frame_at(cfr_video: str) -> None:
    clip = load_video(cfr_video, config={"storage": {"mode": "raw"}})
    img, ts = clip.frame_at_ts(clip.times[3])
    assert ts == clip.times[3]
    assert np.array_equal(clip.frame(3), img)


def test_max_side_downscales_samples_only(cfr_video: str) -> None:
    clip = load_video(cfr_video, config={"storage": {"max_side": 216}})
    assert clip.frame(0).shape[:2] == (216, 120)
    assert (clip.width, clip.height) == SIZE  # размер исходника (после поворота)
    assert clip.frame_at(0.0).shape[:2] == (SIZE[1], SIZE[0])  # доказательства — полный размер


def test_pickle_roundtrip(cfr_video: str) -> None:
    clip = load_video(cfr_video)
    clone = pickle.loads(pickle.dumps(clip))
    assert clone.times == clip.times and np.array_equal(clone.frame(5), clip.frame(5))
    assert read_index_code(clone.frame_at(1.0)) == 30


# ---------------------------------------------------------------- поворот


def _ffmpeg_first_frame(path: str, w: int, h: int) -> np.ndarray:
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)


@needs_ffmpeg
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("opts,swap", [(["-display_rotation", "-90"], True), (["-display_rotation", "90"], True),
                                       (["-display_rotation", "180"], False),
                                       (["-display_rotation", "-90", "-display_hflip"], True)])
def test_rotation_matches_ffmpeg_autorotate(tmp_path: Path, cfr_video: str, backend: str, opts: list[str], swap: bool) -> None:
    out = tmp_path / "rot.mov"
    ffmpeg(*opts, "-i", cfr_video, "-c", "copy", str(out))
    clip = load_video(str(out), backend=backend)
    exp_w, exp_h = (SIZE[1], SIZE[0]) if swap else SIZE
    assert (clip.width, clip.height) == (exp_w, exp_h)
    ours = clip.frame_at(0.0)
    ref = _ffmpeg_first_frame(str(out), exp_w, exp_h)  # ffmpeg CLI поворачивает сам (autorotate)
    assert np.abs(ours.astype(int) - ref.astype(int)).mean() < 4.0
    assert clip.frame(0).shape == ours.shape
    if backend == "pyav" and "-display_hflip" not in opts:
        assert clip.meta["orientation"]["source"] == "displaymatrix"


# ---------------------------------------------------------------- битые файлы


def test_truncated_mov_without_moov(tmp_path: Path, cfr_video: str) -> None:
    data = Path(cfr_video).read_bytes()
    bad = tmp_path / "cut.mov"
    bad.write_bytes(data[: len(data) // 2])  # cv2 пишет moov в конце -> индекса нет
    with pytest.raises(IngestError) as e:
        load_video(str(bad))
    assert isinstance(e.value, (VideoProbeError, DecodeError))


@needs_ffmpeg
def test_truncated_faststart_is_not_a_silent_success(tmp_path: Path, faststart_video: str) -> None:
    data = Path(faststart_video).read_bytes()
    short = tmp_path / "short.mov"
    short.write_bytes(data[: int(len(data) * 0.3)])
    with pytest.raises(DecodeError) as e:
        load_video(str(short))
    assert e.value.code in ("incomplete_video", "decode_failed")
    almost = tmp_path / "almost.mov"
    almost.write_bytes(data[: int(len(data) * 0.85)])
    clip = load_video(str(almost))
    assert not clip.meta["complete"] and clip.meta["coverage"] < 0.98
    assert any("неполное" in w for w in clip.warnings)
    # duration — номинальная (3 с), недекодированный хвост виден как пробел, а не «укорачивает» ролик
    assert clip.duration == pytest.approx(3.0, abs=1e-3) and len(clip) > 0
    assert clip.coverage["decoded_span"] < 2.9 and clip.coverage["unknown_spans"][-1][1] == pytest.approx(3.0, abs=1e-3)


@needs_ffmpeg
def test_corrupted_payload_never_returns_empty_clip(tmp_path: Path, faststart_video: str) -> None:
    data = bytearray(Path(faststart_video).read_bytes())
    rng = np.random.default_rng(0)
    start = int(len(data) * 0.45)
    data[start : start + 20000] = rng.integers(0, 256, 20000, dtype=np.uint8).tobytes()
    bad = tmp_path / "corrupt.mov"
    bad.write_bytes(bytes(data))
    try:
        clip = load_video(str(bad))
    except DecodeError:
        return
    assert len(clip) > 0 and all(clip.frame(i) is not None for i in range(len(clip)))


def test_min_coverage_threshold_is_configurable(cfr_video: str) -> None:
    clip = load_video(cfr_video, config={"decode": {"min_coverage": 0.5}})
    assert clip.meta["coverage"] == pytest.approx(1.0)


# ---------------------------------------------------------------- качество / активность


def test_quality_and_motion(tmp_path: Path) -> None:
    import cv2

    path = tmp_path / "static_then_motion.mov"
    w, h = SIZE
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for i in range(90):
        img = np.full((h, w, 3), 90, np.uint8)
        if i >= 45:  # вторая половина — движущийся объект
            x = int(20 + (i - 45) * 4)
            cv2.rectangle(img, (x, 150), (x + 60, 260), (255, 255, 255), -1)
        vw.write(img)
    vw.release()
    clip = load_video(str(path))
    assert len(clip.quality) == len(clip)
    q0 = clip.quality[0]
    for key in ("t", "sharpness", "brightness", "contrast", "dark_frac", "bright_frac", "motion_energy", "is_blurry"):
        assert key in q0
    energy = [q["motion_energy"] for q in clip.quality]
    assert max(energy[:6]) < 0.05 and max(energy[9:]) > 0.3
    segs = clip.activity_segments()
    assert segs and 0.3 <= segs[0][0] <= 1.5  # движение с t=1.5 с, pad_s=1.0 + сглаживание
    gated = clip.gated_indices()
    assert gated[0] == 0 and gated[-1] == len(clip) - 1
    assert len(gated) < len(clip)
    assert clip.motion_curve()[0][0] == 0.0


def test_dark_video_warning(tmp_path: Path) -> None:
    import cv2

    path = tmp_path / "dark.mov"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    for _ in range(45):
        vw.write(np.full((SIZE[1], SIZE[0], 3), 8, np.uint8))
    vw.release()
    clip = load_video(str(path))
    assert clip.quality_summary["dark_frac"] > 0.9
    assert any("тёмное" in w for w in clip.warnings)


def test_synthetic_uri_and_progress(cfr_video: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SENTINEL_MOCK", raising=False)
    with pytest.raises(IngestError):  # production сам на мок не переключается
        load_video("synthetic://4")
    monkeypatch.setenv("SENTINEL_MOCK", "1")  # явно помеченный мок для тестов/демо
    syn = load_video("synthetic://4")
    assert isinstance(syn, SyntheticClip) and len(syn) == 20 and syn.frame(0).shape == (848, 464, 3)
    assert syn.meta["mock"] is True
    calls: list[float] = []
    load_video(cfr_video, progress=calls.append)
    assert calls and calls[-1] == 1.0


@pytest.mark.slow
def test_full_size_portrait(tmp_path: Path) -> None:
    path = write_test_video(str(tmp_path / "full.mov"), duration_s=12.0, fps=30.0, size=(464, 848))
    clip = load_video(path)
    assert (clip.width, clip.height) == (464, 848) and len(clip) == 60
    assert np.allclose(clip.times, _code_times(clip), atol=1e-6)
