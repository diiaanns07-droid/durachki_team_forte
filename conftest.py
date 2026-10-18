"""Интеграционный glue (A03, ветка claude/integration-cloud): изоляция окружения между тестами модулей.

Часть тестов меняет os.environ процесса: eval/tests/conftest.py (A16) при импорте ставит
SENTINEL_MOCK=1 (действует на ВСЮ сессию — conftest грузится при сборке), testkit/e2e_stub.py (A01) —
во время теста. В общем прогоне это переключало чужие модули в mock (A02: mock_not_allowed, A14 и др.).
Здесь: снимок SENTINEL_* до сборки тестов; перед и после каждого теста вне eval/ окружение
возвращается к снимку. Тесты eval/ получают то, на что рассчитан их conftest. Код модулей не меняется.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

_PREFIX = "SENTINEL_"
_OWN_ENV_DIRS = ("eval",)   # каталоги, чей conftest намеренно задаёт SENTINEL_* для своих тестов
# Снимок при импорте корневого conftest: он грузится раньше conftest-ов модулей (в т.ч. eval/tests,
# который pytest подгружает как «initial conftest» ещё до pytest_configure).
_SNAPSHOT: dict[str, str] = {k: v for k, v in os.environ.items() if k.startswith(_PREFIX)}


def _reset_to_snapshot() -> None:
    for k in [k for k in os.environ if k.startswith(_PREFIX) and k not in _SNAPSHOT]:
        del os.environ[k]
    os.environ.update(_SNAPSHOT)


@pytest.fixture(autouse=True)
def _isolate_sentinel_env(request):
    try:
        rel = Path(str(request.node.fspath)).resolve().relative_to(Path(__file__).resolve().parent)
        own = rel.parts[0] in _OWN_ENV_DIRS
    except ValueError:
        own = False
    if own:
        yield
        return
    _reset_to_snapshot()
    yield
    _reset_to_snapshot()
