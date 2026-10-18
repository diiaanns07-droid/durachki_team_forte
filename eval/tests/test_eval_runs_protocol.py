from __future__ import annotations

import json
from pathlib import Path

from eval.common import load_config
from eval.protocol import env_flags, inputs_fingerprint, same_procedure, sha256_json
from eval.runs import append, load, render_html


def _rec(i: int, fn: int, fp: int) -> dict:
    return {"run_id": f"r{i}", "label": f"iter {i}", "role": "single", "protocol": {"git_sha": "abc", "evaluation_kind":
            "exploratory"}, "metrics": {"fn": fn, "fp": fp, "hit_2": 0.5 + i / 10, "latency_p90": 40 + i,
                                        "vram_peak_mb": 4700}}


def test_runs_log_and_html(tmp_path: Path):
    log = tmp_path / "reports" / "runs.jsonl"
    for i, (fn, fp) in enumerate([(5, 3), (3, 4), (2, 2)]):
        append(log, _rec(i, fn, fp))
    runs = load(log)
    assert len(runs) == 3 and runs[-1]["metrics"]["fn"] == 2
    h = render_html(runs)
    assert h.startswith("<!doctype html>") and "FN (пропуски)" in h and "<title>" in h and "разведочные" in h
    assert "<script" not in h


def test_protocol_helpers(tmp_path: Path):
    v = tmp_path / "X-1.mov"
    v.write_bytes(b"abc")
    fp1 = inputs_fingerprint([("X-1", v)])
    v.write_bytes(b"abcd")
    assert inputs_fingerprint([("X-1", v)])["inputs_sha"] != fp1["inputs_sha"]
    assert env_flags({"SENTINEL_API_KEY": "secret", "SENTINEL_X": 1})["SENTINEL_API_KEY"] == "***"
    cfg = load_config()
    a = {"labels_sha256": "l", "mode": "inproc", "subset": "all", "eval_version": "e", "inputs": {"inputs_sha": "i"},
         "eval_config": cfg}
    assert same_procedure(a, json.loads(json.dumps(a))) == []
    b = dict(a, inputs={"inputs_sha": "other"})
    assert any("inputs_sha" in p for p in same_procedure(a, b))
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})


def test_vram_sampler_device_baseline_and_none(monkeypatch):
    from eval.protocol import VramSampler, torch_peak_mb

    monkeypatch.setenv("SENTINEL_EVAL_DEVICE_METERS", "on")
    vals = iter([3000.0, 5200.0, 4100.0, None, 3900.0])
    s = VramSampler(reader=lambda: next(vals, None))
    s.sample(); s.sample(); s.sample(); s.sample(); s.sample()
    info = s.as_info()
    assert info["vram_device_baseline_mb"] == 3000.0 and info["vram_device_peak_mb"] == 5200.0
    assert info["vram_device_samples"] == 4                     # None пропущен, а не записан нулём
    empty = VramSampler(reader=lambda: None)
    empty.sample()
    assert empty.as_info()["vram_device_peak_mb"] is None
    monkeypatch.setenv("SENTINEL_EVAL_DEVICE_METERS", "off")
    off = VramSampler(reader=lambda: 9999.0)
    with off:
        pass
    assert off.available is False and off.peak_mb is None and torch_peak_mb() is None


def test_conftest_import_has_no_env_side_effects(tmp_path):
    """eval/tests/conftest.py при импорте не ставит SENTINEL_MOCK (раньше — на всю pytest-сессию)."""
    import os
    import subprocess
    import sys

    from eval.common import REPO_ROOT

    env = {k: v for k, v in os.environ.items() if not k.startswith("SENTINEL_")}
    code = "import os, eval.tests.conftest as c; print(repr(os.environ.get('SENTINEL_MOCK')))"
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "None"
