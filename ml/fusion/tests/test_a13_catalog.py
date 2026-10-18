"""fusion на НАСТОЯЩЕМ каталоге A13 (norms.load_catalog). Пропускается, если пакета norms нет в окружении
(ветка A12 без norms/); в интеграции L2 выполняется. Нормы не выдумываются, неподтверждённые не цитируются."""
from __future__ import annotations

import pytest

norms = pytest.importorskip("norms")

from ml.fusion import fuse  # noqa: E402
from ml.fusion.catalog import is_verified, resolve_catalog  # noqa: E402
from ml.fusion.config import load_config  # noqa: E402

from .conftest import FakeClip, ev  # noqa: E402

CAT = norms.load_catalog()
TYPES = CAT.get("types") or {}
FACT = "наблюдаемый факт синтетического теста"


def _expected_rules(type_id):
    out = []
    for r in TYPES[type_id].get("rules") or []:
        if isinstance(r, dict) and r.get("document") and r.get("url"):
            out.append((r["document"], r.get("clause"), r["url"], is_verified(r.get("verified", False))))
    return out


def test_real_catalog_is_used_as_is():
    types, warnings = resolve_catalog(CAT)
    assert set(types) == set(TYPES) and warnings == []


def test_empty_catalog_resolves_to_real_a13(monkeypatch):
    types, warnings = resolve_catalog({})
    assert set(types) == set(TYPES) and not any("unavailable" in w for w in warnings)


@pytest.mark.parametrize("type_id", sorted(TYPES))
def test_violation_names_and_norms_come_only_from_a13(type_id, no_frames):
    res = fuse(FakeClip(duration=20.0), [ev(type_id, t=5.0, conf=0.9, facts=[FACT])], [], CAT, "job")
    need_norm = type_id in (load_config()["report_gate"].get("require_norm_types") or [])
    rules = _expected_rules(type_id)
    if need_norm and not rules:
        # V99 без нормы в каталоге — только подсказка, без юридического обвинения
        assert res.violations == [] and res.verdict == "no_violation"
        hint = [h for h in res.meta["ui"]["review_hints"] if h["type_id"] == type_id]
        assert hint and "no_norm" in hint[0]["reasons"]
        return
    assert res.verdict == "violation" and len(res.violations) == 1
    v = res.violations[0]
    assert v.type == TYPES[type_id]["name"]
    # нормы — ровно из каталога, verified как в каталоге (строго), ничего не добавлено
    assert [(r.document, r.clause, r.url, r.verified) for r in v.rules] == rules
    cited = "(норма:" in v.description
    assert cited == any(verified for *_, verified in rules)


@pytest.mark.parametrize("type_id", sorted(TYPES))
def test_temporal_only_never_a_violation(type_id, no_frames):
    res = fuse(FakeClip(duration=20.0), [ev(type_id, t=5.0, conf=0.99, source="temporal", facts=[])],
               [(t / 5, 0.99) for t in range(100)], CAT, "job")
    assert res.violations == [] and res.verdict == "no_violation"
