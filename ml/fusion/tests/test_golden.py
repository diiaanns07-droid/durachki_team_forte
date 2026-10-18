"""Золотые случаи (tests/golden/cases.json) и детерминизм отчёта."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from ml.fusion import fuse
from ml.fusion.serialize import result_to_api, result_to_jury

from .conftest import FakeClip
from .test_upstream import A13_LIKE

CASES = json.loads((Path(__file__).parent / "golden" / "cases.json").read_text(encoding="utf-8"))


def make_clip(spec: dict) -> FakeClip:
    maker = spec.get("maker", "coded")
    clip = FakeClip(duration=float(spec.get("duration", 20.0)))
    if maker == "static":
        static = np.full((clip.height, clip.width, 3), 77, np.uint8)
        clip._maker = lambda idx: static.copy()
    elif maker == "none":
        clip._maker = lambda idx: None
    if "meta" in spec:
        clip.meta = dict(spec["meta"])
    return clip


def run_case(case: dict, catalog: dict):
    cat = A13_LIKE if case.get("catalog") == "a13_like" else catalog
    return fuse(make_clip(case["clip"]), copy.deepcopy(case["events"]), case.get("risk", []), cat, case["id"])


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_golden(case, catalog):
    res = run_case(case, catalog)
    exp = case["expect"]
    assert res.status == exp["status"]
    assert res.verdict == exp["verdict"]
    assert [v.type_id for v in res.violations] == exp["types"]
    if "timecodes" in exp:
        assert [v.timecode_sec for v in res.violations] == pytest.approx(exp["timecodes"], abs=0.11)
    for v in res.violations:
        assert 0.0 <= v.t_start <= v.timecode_sec <= v.t_end <= case["clip"]["duration"]
        ts = [f.timestamp_sec for f in v.evidence_frames]
        assert len(ts) == len(set(ts)) <= 3
        if exp.get("pts_on_grid"):
            assert all(any(abs(t - g) < 1e-6 for g in make_clip(case["clip"]).times) for t in ts)
    if exp.get("rules_empty"):
        assert all(v.rules == [] for v in res.violations)
        assert all("норма:" not in v.description for v in res.violations)
    if "n_frames" in exp:
        assert [len(v.evidence_frames) for v in res.violations] == exp["n_frames"]
    if "warning" in exp:
        assert any(exp["warning"] in w for w in res.meta["warnings"])
    if "failure" in exp:
        assert res.meta["failure"]["reason"] == exp["failure"] and res.violations == []
    if exp.get("incomplete"):
        assert res.meta["ui"]["incomplete"]["message_ru"] and res.meta["fusion"]["coverage"]["complete"] is False
    if "jury_status" in exp:
        assert result_to_jury(res)["status"] == exp["jury_status"]
    hints = {h["type_id"]: h["reasons"] for h in res.meta.get("ui", {}).get("review_hints", [])}
    for type_id, reasons in (exp.get("hint_reasons") or {}).items():
        assert hints.get(type_id) == reasons
    json.dumps(result_to_api(res), ensure_ascii=False)


def _strip_volatile(obj):
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items() if k not in ("timings",)}
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_report_is_deterministic(case, catalog):
    a = json.dumps(_strip_volatile(result_to_api(run_case(case, catalog))), ensure_ascii=False, sort_keys=True)
    b = json.dumps(_strip_volatile(result_to_api(run_case(case, catalog))), ensure_ascii=False, sort_keys=True)
    assert a == b
