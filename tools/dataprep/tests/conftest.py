"""Тесты tools/dataprep: только синтетика (никаких реальных видео, кадров и меток)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

DATAPREP = Path(__file__).resolve().parents[1]


def load(name: str):
    """Скрипты tools/dataprep — не пакет: грузим по пути (как делают сами скрипты под python -I)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, DATAPREP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def dp():
    return load
