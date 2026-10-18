"""Наблюдаемость: статус последнего запуска estimate() (сбой позы не скрывается, но и не роняет пайплайн).

A03 может положить ``ml.pose.status()`` в ``AnalysisResult.meta`` (warnings/timings).
"""
from __future__ import annotations

import copy
import threading
from typing import Any

_LOCK = threading.Lock()
_STATUS: dict[str, Any] = {"state": "not_run", "stage_status": None, "backend": None, "error": None, "warnings": []}


def status() -> dict[str, Any]:
    """Копия статуса последнего estimate().

    state ∈ {not_run, ok, partial, failed, mock, no_persons, disabled};
    stage_status ∈ contracts.extensions.STAGE_STATUSES (ok | skipped | failed | timeout | mock) — для
    meta.completeness.stages["pose"]; coverage — сколько кадров/людей реально покрыто позой.
    """
    with _LOCK:
        return copy.deepcopy(_STATUS)


def set_status(**kwargs: Any) -> None:
    with _LOCK:
        _STATUS.clear()
        _STATUS.update({"state": "not_run", "stage_status": None, "backend": None, "error": None, "warnings": []})
        _STATUS.update(kwargs)
