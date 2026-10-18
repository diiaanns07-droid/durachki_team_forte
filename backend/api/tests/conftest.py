"""Фикстуры: приложение на tmp_path, синтетические видео, тестовые пайплайны. Без GPU и весов."""
from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from backend.api.config import Settings  # noqa: E402
from backend.api.fake_pipeline import _PLACEHOLDER_JPEG  # noqa: E402
from backend.api.worker import PipelineSpec  # noqa: E402

try:
    from contracts.sentinel_contracts import (  # noqa: E402
        AnalysisResult, EvidenceFrame, RuleRef, Violation)
except ImportError:
    from contracts.sentinel_contracts import (  # noqa: E402
        AnalysisResult, EvidenceFrame, RuleRef, Violation)

KEY = "test-key-123"
H = {"X-API-Key": KEY}
# минимальный ISO-BMFF: проходит сигнатуру, но НЕ декодируется
FTYP_ONLY = (b"\x00\x00\x00\x14ftypqt  \x00\x00\x02\x00qt  " + b"\x00\x00\x00\x08free"
             + b"\x00" * 64)


def pytest_configure(config: Any) -> None:
    config.addinivalue_line("markers", "gpu: требует GPU")
    config.addinivalue_line("markers", "slow: медленный тест")


def make_result(job_id: str, t: float = 4.2) -> Any:
    v = Violation(type="Посторонний в кассовом помещении", timecode_sec=t,
                  description="Признаки возможного нарушения: тест.",
                  evidence_frames=[EvidenceFrame(timestamp_sec=t, image_base64=_PLACEHOLDER_JPEG)],
                  type_id="V01", confidence=0.8, t_start=t - 1, t_end=t + 1)
    return AnalysisResult(job_id=job_id, status="done", verdict="violation", violations=[v],
                          timeline=[(0.0, 0.1), (t, 0.9)], meta={"version": "test"})


def spec(run: Callable[..., Any], warmup: Callable[[], None] | None = None) -> PipelineSpec:
    return PipelineSpec(run=run, warmup=warmup, mode="real")


def ok_run(video_path: str, job_id: str, progress: Callable[[str, float], None],
           deadline_s: float = 240) -> Any:
    progress("detect", 0.5)
    return make_result(job_id)


@pytest.fixture
def settings_factory(tmp_path: Path) -> Callable[..., Settings]:
    def make(**kw: Any) -> Settings:
        base: dict[str, Any] = dict(data_dir=tmp_path / "data", api_keys=[KEY],
                                    frontend_dir=tmp_path / "no-frontend", video_validation="off",
                                    log_json=True, job_ttl_h=0, rl_auth_fail="10000/60",
                                    dedup=False)  # тесты грузят одни и те же байты; dedup — свой тест
        base.update(kw)
        return Settings.from_env(env={}, **base)
    return make


@pytest.fixture
def client_factory(settings_factory: Callable[..., Settings]) -> Iterator[Callable[..., Any]]:
    from fastapi.testclient import TestClient
    from backend.api.main import create_app
    stack: list[Any] = []

    @contextmanager
    def make(pipe: Any = None, **kw: Any) -> Iterator[Any]:
        app = create_app(settings_factory(**kw), pipeline=pipe)
        with TestClient(app) as c:
            stack.append(c)
            yield c

    yield make


def wait_job(c: Any, job_id: str, headers: dict[str, str] | None = H, prefix: str = "/api/v1",
             timeout: float = 15.0) -> dict[str, Any]:
    end = time.time() + timeout
    while time.time() < end:
        r = c.get(f"{prefix}/jobs/{job_id}", headers=headers)
        assert r.status_code == 200, r.text
        d = r.json()
        if d["status"] != "processing":
            return d
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} still processing after {timeout}s")


def wait_ready(c: Any, timeout: float = 5.0) -> Any:
    end = time.time() + timeout
    r = c.get("/readyz")
    while time.time() < end and r.json().get("state") in ("loading", "warming", "starting"):
        time.sleep(0.02)
        r = c.get("/readyz")
    return r


def upload(c: Any, data: bytes, name: str = "X-1.mov", headers: dict[str, str] | None = H,
           prefix: str = "/api/v1") -> Any:
    return c.post(f"{prefix}/analyze", headers=headers,
                  files={"file": (name, data, "video/quicktime")})


def write_mov(path: Path, seconds: float = 2.0, fps: int = 30, size: tuple[int, int] = (232, 424)) -> Path:
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not w.isOpened():
        pytest.skip("cv2 VideoWriter cannot write mp4v/.mov here")
    for i in range(int(seconds * fps)):
        f = np.full((size[1], size[0], 3), (i * 5) % 255, np.uint8)
        cv2.rectangle(f, (10 + i, 40), (80 + i, 140), (0, 200, 0), -1)
        w.write(f)
    w.release()
    return path


@pytest.fixture
def mov_bytes(tmp_path: Path) -> bytes:
    return write_mov(tmp_path / "syn.mov").read_bytes()


class Gate:
    """Пайплайн, который ждёт сигнала (для проверки очереди/рестарта)."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.started = threading.Event()
        self.calls: list[tuple[str, float]] = []

    def run(self, video_path: str, job_id: str, progress: Callable[[str, float], None],
            deadline_s: float = 240) -> Any:
        self.calls.append((job_id, deadline_s))
        self.started.set()
        self.release.wait(10)
        return make_result(job_id)


# ------------------------------------------------------------------ синтетический VFR
_MP4_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf", b"udta"}


def _atom_chain(buf: bytes, path: list[bytes], start: int = 0, end: int | None = None,
                chain: tuple = ()) -> tuple | None:
    import struct
    end = len(buf) if end is None else end
    pos = start
    while pos + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[pos:pos + 8])
        hl = 8
        if size == 1:
            size, hl = struct.unpack(">Q", buf[pos + 8:pos + 16])[0], 16
        elif size == 0:
            size = end - pos
        if typ == path[0]:
            ch = chain + ((pos, size, hl),)
            if len(path) == 1:
                return ch
            r = _atom_chain(buf, path[1:], pos + hl, pos + size, ch)
            if r:
                return r
        pos += size
    return None


def write_vfr_mov(path: Path, fast: int = 30, slow: int = 30, slow_dt: float = 0.25,
                  size: tuple[int, int] = (232, 424)) -> Path:
    """Настоящий VFR-файл без внешних утилит: пишем CFR 30 fps через cv2 и переписываем stts
    (длительности сэмплов) + длительности mdhd/mvhd/tkhd/elst. Кадр k залит яркостью (k*4)%255.
    PTS: k/30 для k < fast, затем шаг slow_dt (1.0, 1.25, 1.5, ...)."""
    import struct
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    src = path.with_name(path.stem + "_cfr.mov")
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
    if not w.isOpened():
        pytest.skip("cv2 VideoWriter cannot write mp4v/.mov here")
    for k in range(fast + slow):
        w.write(np.full((size[1], size[0], 3), (k * 4) % 255, np.uint8))
    w.release()
    b = bytearray(src.read_bytes())
    if _atom_chain(b, [b"moov"])[0][0] < _atom_chain(b, [b"mdat"])[0][0]:
        pytest.skip("moov before mdat: stco would need rewriting")
    mdhd = _atom_chain(b, [b"moov", b"trak", b"mdia", b"mdhd"])[-1]
    p = mdhd[0] + mdhd[2]
    ts = struct.unpack(">I", b[p + 12:p + 16])[0]
    deltas = [ts // 30] * fast + [int(ts * slow_dt)] * slow
    chain = _atom_chain(b, [b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stts"])
    pos, old, _ = chain[-1]
    runs: list[list[int]] = []
    for d in deltas:
        if runs and runs[-1][1] == d:
            runs[-1][0] += 1
        else:
            runs.append([1, d])
    body = struct.pack(">II", 0, len(runs)) + b"".join(struct.pack(">II", c, d) for c, d in runs)
    new = struct.pack(">I4s", 8 + len(body), b"stts") + body
    b[pos:pos + old] = new
    for apos, asize, _ in chain[:-1]:
        struct.pack_into(">I", b, apos, asize + len(new) - old)
    total = sum(deltas)
    mdhd = _atom_chain(b, [b"moov", b"trak", b"mdia", b"mdhd"])[-1]
    struct.pack_into(">I", b, mdhd[0] + mdhd[2] + 16, total)
    mvhd = _atom_chain(b, [b"moov", b"mvhd"])[-1]
    q = mvhd[0] + mvhd[2]
    mdur = int(round(total / ts * struct.unpack(">I", b[q + 12:q + 16])[0]))
    struct.pack_into(">I", b, q + 16, mdur)
    tkhd = _atom_chain(b, [b"moov", b"trak", b"tkhd"])[-1]
    struct.pack_into(">I", b, tkhd[0] + tkhd[2] + 20, mdur)
    elst = _atom_chain(b, [b"moov", b"trak", b"edts", b"elst"])
    if elst:
        struct.pack_into(">I", b, elst[-1][0] + elst[-1][2] + 8, mdur)
    path.write_bytes(bytes(b))
    src.unlink()
    return path
