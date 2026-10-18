"""Общие фикстуры тестов copilot: без GPU, без весов, без сети."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from copilot.config import load_config  # noqa: E402
from copilot.corpus import Corpus, load_corpus  # noqa: E402
from copilot.embed import HashingEmbedder  # noqa: E402

FIXTURE_DIR = ROOT / "copilot" / "fixtures" / "norms"


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ("COPILOT_ENABLED", "COPILOT_LLM_BACKEND", "COPILOT_MODEL_PATH", "COPILOT_SERVER_URL",
                 "COPILOT_CONFIG", "COPILOT_NORMS_DIR", "COPILOT_ALLOW_FIXTURE", "SENTINEL_MOCK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("COPILOT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    # герметичность: настоящий ml.vlm.service (A11, после мерджа) не должен влиять на тесты «нет модели»;
    # тесты VLM подставляют свой фейковый модуль через monkeypatch.setitem
    monkeypatch.setitem(sys.modules, "ml.vlm.service", None)
    monkeypatch.delenv("COPILOT_VLM_MODULE", raising=False)


@pytest.fixture()
def cfg() -> dict[str, Any]:
    """Конфиг для тестов: фикстура норм, hashing-эмбеддинги, mock-LLM, copilot включён."""
    return load_config({
        "enabled": True,
        "corpus": {"catalog_path": str(FIXTURE_DIR / "catalog.yaml"),
                   "sources_glob": str(FIXTURE_DIR / "sources" / "*.md"),
                   "fixture_fallback": False},
        "embed": {"backend": "hashing"},
        "llm": {"backend": "mock", "timeout_s": 5, "model_candidates": []},
    })


@pytest.fixture()
def corpus(cfg: dict[str, Any]) -> Corpus:
    return load_corpus(cfg)


@pytest.fixture()
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=2048)


@pytest.fixture()
def job_violation() -> dict[str, Any]:
    return {
        "job_id": "job-test-1",
        "status": "done",
        "verdict": "violation",
        "violations": [
            {"type": "Использование мобильного телефона при работе с ценностями", "type_id": "V05",
             "timecode_sec": 22.0, "t_start": 20.0, "t_end": 25.0, "confidence": 0.81,
             "description": "Кассир держит телефон у уха, на столе видны наличные",
             "evidence_frames": [{"timestamp_sec": 21.0, "image_base64": "QUFBQQ=="},
                                 {"timestamp_sec": 22.0, "image_base64": "QkJCQg=="}],
             "rules": [{"document": "ФИКСТУРА: Тестовый регламент кассы", "clause": "п. 9",
                        "url": "https://example.invalid/fixture/reglament#p9", "quote": "…", "verified": False}]},
            {"type": "Сейф открыт без присмотра или при посторонних", "type_id": "V03",
             "timecode_sec": 47.5, "t_start": 45.0, "t_end": 52.0, "confidence": 0.64,
             "description": "Дверца сейфа открыта, кассир вне рабочей зоны",
             "evidence_frames": [{"timestamp_sec": 47.5, "image_base64": "Q0NDQw=="}],
             "rules": [{"document": "Выдуманный закон о кассах", "clause": "п. 999",
                        "url": "https://adilet.zan.kz/rus/docs/FAKE", "quote": "нет такого"}]},
        ],
        "meta": {"duration": 62.0, "version": "test"},
    }


@pytest.fixture()
def job_clean() -> dict[str, Any]:
    return {"job_id": "job-test-2", "status": "done", "verdict": "no_violation", "violations": [], "meta": {}}
