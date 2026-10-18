"""Фикстуры eval: мини-xlsx, синтетические видео, синтетические предсказания (без GPU и весов)."""
from __future__ import annotations

import base64
import os
import sys
from pathlib import Path
from typing import Any, Optional

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Окружение процесса при импорте НЕ меняем (раньше тут ставился SENTINEL_MOCK=1 на всю сессию — это включало
# mock у чужих модулей в общем прогоне). Mock — только явно, через фикстуру mock_mode (monkeypatch, с откатом).


@pytest.fixture(autouse=True)
def _isolate_eval_env(monkeypatch: pytest.MonkeyPatch):
    """Каждый тест eval стартует без SENTINEL_* хоста и без GPU-измерителей хоста (nvidia-smi/torch):
    иначе состояние чужой видеокарты (например, 7.6 ГБ занято) попадает в юниттест. После теста убираются и
    SENTINEL_*, поставленные кодом мимо monkeypatch; затем monkeypatch возвращает значения хоста."""
    for k in [k for k in os.environ if k.startswith("SENTINEL_")]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SENTINEL_EVAL_DEVICE_METERS", "off")
    yield
    for k in [k for k in os.environ if k.startswith("SENTINEL_")]:
        del os.environ[k]


@pytest.fixture
def mock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Явный mock-режим (A01: mock-результаты допустимы только при SENTINEL_MOCK=1) — только в этом тесте."""
    monkeypatch.setenv("SENTINEL_MOCK", "1")


def canonical_coverage(duration: float, stages: Optional[dict] = None) -> dict:
    """meta.coverage в каноне A01 (contracts.extensions.compute_coverage), полный анализ."""
    stages = stages or {"ingest": "ok", "detect": "ok", "fusion": "ok"}
    try:
        from contracts.extensions import compute_coverage
    except ImportError:  # автономная ветка без A01: та же форма вручную
        return {"ratio": 1.0, "analyzed_sec": duration, "duration_sec": duration, "unknown_spans": [],
                "stages": stages, "required": [], "complete": True, "source": "test"}
    return compute_coverage(duration, duration, stages=stages, source="test")


def pytest_configure(config: Any) -> None:
    config.addinivalue_line("markers", "slow: долгие тесты")
    config.addinivalue_line("markers", "gpu: нужен GPU")


def tiny_jpeg_b64() -> str:
    """Минимальный валидный JPEG (8×8 серый) — без cv2/PIL, если их нет."""
    try:
        import cv2
        import numpy as np

        ok, buf = cv2.imencode(".jpg", np.full((8, 8, 3), 128, np.uint8))
        assert ok
        return base64.b64encode(buf.tobytes()).decode()
    except Exception:  # noqa: BLE001
        raw = bytes.fromhex(
            "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707070909080a0c140d0c0b0b0c1912130f141d1a1f"
            "1e1d1a1c1c20242e2720222c231c1c2837292c30313434341f27393d38323c2e333432ffc0000b080008000801011100ffc4001f0000"
            "010501010101010100000000000000000102030405060708090a0bffc400b5100002010303020403050504040000017d0102030004"
            "1105122131410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a43"
            "4445464748494a535455565758595a636465666768696a737475767778797a838485868788898a92939495969798999aa2a3a4a5a6"
            "a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffda"
            "0008010100003f00fbd3ffd9")
        return base64.b64encode(raw).decode()


JPEG_B64 = None


def jpeg() -> str:
    global JPEG_B64
    if JPEG_B64 is None:
        JPEG_B64 = tiny_jpeg_b64()
    return JPEG_B64


def make_violation(t: float, type_id: str = "V01", frames: Optional[list[float]] = None, rules: Any = "default",
                   description: str = "Признаки возможного нарушения: в зоне кассира находится второй человек.",
                   confidence: float = 0.8) -> dict:
    if rules == "default":
        rules = [{"document": "Правила НБРК", "clause": "п. 1", "url": "https://adilet.zan.kz/rus/docs/V000",
                  "quote": "...", "verified": True}]
    frames = [t] if frames is None else frames
    return {"type": "Посторонний в кассовом помещении", "timecode_sec": t, "description": description,
            "evidence_frames": [{"timestamp_sec": f, "image_base64": jpeg()} for f in frames],
            "type_id": type_id, "confidence": confidence, "t_start": max(0.0, t - 1), "t_end": t + 1,
            "rules": rules or []}


def make_pred(job_id: str, violations: Optional[list[dict]] = None, verdict: Optional[str] = None,
              status: str = "done", duration: float = 60.0, timeline: Optional[list] = None, meta: Optional[dict] = None
              ) -> dict:
    """Синтетический ответ API. По умолчанию — полный анализ (канон A01 meta.coverage); meta перекрывает."""
    violations = violations or []
    if verdict is None:
        verdict = "violation" if violations else "no_violation"
    m = {"duration": duration, "coverage": canonical_coverage(duration)}
    m.update(meta or {})
    return {"job_id": job_id, "status": status, "verdict": verdict if status == "done" else None,
            "violations": violations, "timeline": timeline or [[0.0, 0.1], [duration, 0.1]], "meta": m}


@pytest.fixture
def mini_xlsx(tmp_path: Path) -> Path:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Лист1"
    ws.append(["№", "Название видео", "Нарушение"])
    ws.append([1, "X-1", "22 секунда"])            # латинская X
    ws.append([2, "Х-3", "без нарушения"])         # кириллическая Х
    ws.append([3, "Х-4", "57 секунда"])
    ws.append([4, "Х-47", "1минута 8 секунда"])     # дубль Х-4, другой таймкод
    ws.append([5, "Х-5", "Без нарушений"])
    ws.append([6, "Х-48", "без нарушения"])        # дубль Х-5, без конфликта
    ws.append([7, "х-12.mov", "на 1:05"])
    ws.append([None, None, None])
    p = tmp_path / "open_labels.xlsx"
    wb.save(p)
    return p


@pytest.fixture
def mini_videos(tmp_path: Path) -> Path:
    """Фейковые «видео» (байты) — для md5-дедупа видео не нужны настоящие."""
    d = tmp_path / "videos"
    d.mkdir()
    (d / "X-1.mov").write_bytes(b"video-1")
    (d / "Х-3.mov").write_bytes(b"video-3")
    (d / "Х-4.mov").write_bytes(b"video-4-same")
    (d / "Х-47.MOV").write_bytes(b"video-4-same")
    (d / "Х-5.mov").write_bytes(b"video-5-same")
    (d / "Х-48.mov").write_bytes(b"video-5-same")
    (d / "х-12.mov").write_bytes(b"video-12")
    (d / "X-99.mov").write_bytes(b"unlabeled")
    return d


def write_synthetic_video(path: Path, seconds: float = 3.0, fps: int = 10, size: tuple[int, int] = (64, 96),
                          event_at: Optional[float] = None) -> Path:
    """Синтетическое видео через cv2.VideoWriter (портрет w×h); белый квадрат «событие» после event_at."""
    import cv2
    import numpy as np

    w, h = size
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(int(seconds * fps)):
        img = np.full((h, w, 3), 40, np.uint8)
        x = int((i * 3) % max(1, w - 10))
        img[10:20, x:x + 10] = (0, 0, 255)
        if event_at is not None and i / fps >= event_at:
            img[h // 2:h // 2 + 16, w // 2 - 8:w // 2 + 8] = 255
        vw.write(img)
    vw.release()
    return path
