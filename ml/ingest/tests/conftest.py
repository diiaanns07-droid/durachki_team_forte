"""Фикстуры ml/ingest: только синтетика (cv2.VideoWriter / ffmpeg lavfi), без реальных данных."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: долгие тесты (полноразмерное видео)")
    config.addinivalue_line("markers", "gpu: тесты, требующие GPU")


FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="нужен ffmpeg в PATH")

FPS = 30.0
SIZE = (240, 432)  # (w, h) — уменьшенный «портрет» для скорости тестов


def ffmpeg(*args: str) -> None:
    proc = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args], capture_output=True)
    if proc.returncode != 0:
        pytest.skip(f"ffmpeg не смог: {proc.stderr.decode(errors='replace')[-300:]}")


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("ingest_media")


@pytest.fixture(scope="session")
def cfr_video(media_dir: Path) -> str:
    """3 с, 30 fps, штрих-код номера кадра в каждом кадре (.mov, mp4v)."""
    from ml.ingest.synthetic import write_test_video

    return write_test_video(str(media_dir / "cfr.mov"), duration_s=3.0, fps=FPS, size=SIZE)


@pytest.fixture(scope="session")
def faststart_video(media_dir: Path, cfr_video: str) -> str:
    """Тот же ролик, moov в начале (нужен для «обрезанного, но открываемого» файла)."""
    if FFMPEG is None:
        pytest.skip("нужен ffmpeg")
    out = media_dir / "faststart.mov"
    ffmpeg("-i", cfr_video, "-c", "copy", "-movflags", "+faststart", str(out))
    return str(out)


@pytest.fixture(scope="session")
def vfr_video(media_dir: Path, cfr_video: str) -> str:
    """VFR: из CFR выкинуты кадры по нерегулярному шаблону, исходные PTS сохранены."""
    if FFMPEG is None:
        pytest.skip("нужен ffmpeg")
    out = media_dir / "vfr.mov"
    keep = r"select='not(mod(n\,3))+eq(mod(n\,7)\,1)+between(n\,40\,47)'"
    ffmpeg("-i", cfr_video, "-vf", keep, "-fps_mode", "vfr", "-c:v", "mpeg4", "-q:v", "2", str(out))
    return str(out)
