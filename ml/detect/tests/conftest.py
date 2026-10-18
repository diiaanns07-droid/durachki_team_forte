from __future__ import annotations

import importlib
import socket
import sys
import types

import pytest

# Тесты импортируют канонический контракт (contracts.sentinel_contracts). В интеграции он есть и
# fallback-копия удалена. Только в отдельной ветке A05 (без contracts/) подставляем побайтно идентичную
# копию ml/detect/_contract_fallback.py под каноническим именем — тестовая прокладка, не прод-код.
try:
    importlib.import_module("contracts.sentinel_contracts")
except ModuleNotFoundError:  # pragma: no cover — ветка без A01
    from ml.detect import _contract_fallback as _fb

    _pkg = sys.modules.setdefault("contracts", types.ModuleType("contracts"))
    _pkg.sentinel_contracts = _fb  # type: ignore[attr-defined]
    sys.modules["contracts.sentinel_contracts"] = _fb


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Каждый тест: пустой SENTINEL_WEIGHTS, без mock/extras по умолчанию, чистый кэш бэкендов."""
    from ml.detect.backends import clear_cache
    from ml.detect.inventory import clear_cache as clear_inv_cache

    monkeypatch.setenv("SENTINEL_WEIGHTS", str(tmp_path / "weights"))
    for k in ("SENTINEL_MOCK", "SENTINEL_DETECT_EXTRAS", "SENTINEL_DETECT_CONFIG", "SENTINEL_DETECT_BUDGET_S"):
        monkeypatch.delenv(k, raising=False)
    clear_cache()
    clear_inv_cache()
    yield
    clear_cache()
    clear_inv_cache()


@pytest.fixture
def mock_mode(monkeypatch):
    monkeypatch.setenv("SENTINEL_MOCK", "1")
    yield


@pytest.fixture
def no_network(monkeypatch):
    """Любая попытка сетевого соединения в тесте — ошибка."""
    def guard(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
    yield


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: тяжёлый тест (крошечные модели transformers на CPU)")
    config.addinivalue_line("markers", "gpu: требует CUDA")
