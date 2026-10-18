from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.temporal.config import load_config  # noqa: E402
from ml.temporal.dataset import Sample  # noqa: E402
from ml.temporal.features import FeatureSet, TextBank  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: долгие тесты (обучение torch-головы)")
    config.addinivalue_line("markers", "gpu: тесты, требующие CUDA")


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SENTINEL_TEMPORAL_WEIGHTS", str(tmp_path / "weights" / "temporal"))
    monkeypatch.delenv("SENTINEL_MOCK", raising=False)
    yield


@pytest.fixture
def cfg():
    return load_config({"train": {"epochs": 6, "steps_per_epoch": 6, "batch_size": 6, "window_s": 20.0},
                        "baseline": {"iters": 250}, "cv": {"epochs": 6}})


def make_bank(dim: int = 16, seed: int = 0) -> TextBank:
    rng = np.random.default_rng(seed)
    owner = ["V01", "V01", "V05", "V07", "V99", "normal", "normal"]
    emb = rng.standard_normal((len(owner), dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return TextBank([f"p{i}" for i in range(len(owner))], [f"п{i}" for i in range(len(owner))], owner, emb)


def make_fs(T: int, label: int, t_event: float | None, seed: int, dim: int = 16, fps: float = 4.0,
            signal: float = 2.5) -> FeatureSet:
    """Синтетика: шум + для позитива «сигнатура» в эмбеддинге и движении вокруг события (±2 с)."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(dim)
    sig_dir = np.random.default_rng(999).standard_normal(dim)
    times = (np.arange(T) / fps).astype(np.float32)
    emb = base[None, None, :] + 0.6 * rng.standard_normal((T, 2, dim))
    dyn = np.abs(rng.normal(0.1, 0.05, (T, 4))).astype(np.float32)
    if label and t_event is not None:
        m = np.abs(times - t_event) <= 2.0
        emb[m] += signal * sig_dir
        dyn[m, 0] += 1.0
    emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
    boxes = np.tile(np.array([0.1, 0.1, 0.6, 0.6], np.float32), (T, 1))
    return FeatureSet(times=times, emb=emb.astype(np.float32), dyn=dyn, boxes=boxes, views=["full", "focus"],
                      meta={"fps": fps, "backend": "mock"})


def make_samples(n_pos: int = 8, n_neg: int = 8, T: int = 120, n_aug: int = 1, seed: int = 0) -> list[Sample]:
    out = []
    rng = np.random.default_rng(seed)
    for i in range(n_pos + n_neg):
        lab = int(i < n_pos)
        te = float(rng.uniform(5, T / 4.0 - 5)) if lab else None
        versions = [make_fs(T, lab, te, seed=1000 * seed + 10 * i + k) for k in range(1 + n_aug)]
        out.append(Sample(name=f"X-{i + 1}", group=f"X-{i + 1}", label=lab, times=[te] if lab else [],
                          versions=versions, md5=f"md5_{i}"))
    return out


@pytest.fixture
def toy():
    return make_samples(), make_bank()


def write_video(path: Path, seconds: float = 8.0, fps: int = 10, w: int = 116, h: int = 212,
                event: tuple[float, float] | None = (3.0, 5.0), shade: int = 90) -> Path:
    """Синтетическое портретное видео: статичный «стол» + движущийся прямоугольник в интервале event."""
    cv2 = pytest.importorskip("cv2")
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    n = int(seconds * fps)
    for i in range(n):
        t = i / fps
        fr = np.full((h, w, 3), shade, np.uint8)
        cv2.rectangle(fr, (10, 120), (100, 190), (60, 60, 60), -1)
        if event and event[0] <= t <= event[1]:
            x = int(10 + (t - event[0]) * 30) % (w - 30)
            cv2.rectangle(fr, (x, 40), (x + 30, 80), (20, 200, 30), -1)
        vw.write(fr)
    vw.release()
    return path
