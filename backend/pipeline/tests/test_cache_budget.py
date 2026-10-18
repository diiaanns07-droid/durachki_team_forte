"""Кэш (второй прогон быстрее и идентичен, инвалидация ключей), бюджет времени, утилиты."""
from __future__ import annotations

import json
import time

import pytest

from backend.pipeline import mocks, run
from backend.pipeline.budget import TimeBudget, choose_sample_fps
from backend.pipeline.config import load_config
from backend.pipeline.media import normalize_video_name
from backend.pipeline.serde import result_to_dict


def _strip(res) -> str:
    """Содержательный результат (без метаданных исполнения: статусы ok/cached, тайминги, кэш-хиты)."""
    d = result_to_dict(res)
    cov = res.meta.get("coverage") or {}
    keep = {k: d.get(k) for k in ("status", "verdict", "violations", "timeline")}
    keep["coverage"] = {k: cov.get(k) for k in ("ratio", "complete", "unknown_spans", "duration_sec")}
    keep["stages_status"] = {k: v["status"] for k, v in res.meta["stages"].items()}
    return json.dumps(keep, sort_keys=True, ensure_ascii=False)


def _slow_detect(clip, every_n=1):
    time.sleep(0.4)
    return mocks.detect(clip, every_n)


def test_second_run_from_cache_faster_and_identical(fake_ml, synth_video):
    fake_ml({"ml.detect.detect": _slow_detect})
    t0 = time.perf_counter()
    r1 = run(synth_video, "job-c", None, 60)
    t1 = time.perf_counter() - t0
    t0 = time.perf_counter()
    r2 = run(synth_video, "job-c", None, 60)
    t2 = time.perf_counter() - t0
    assert r2.meta["stages"]["detect"]["status"] == "cached"
    assert "detect" in r2.meta["cache"]["hits"]
    assert t2 < t1 - 0.3
    a, b = json.loads(_strip(r1)), json.loads(_strip(r2))
    a.pop("stages_status"), b.pop("stages_status")
    assert a == b


def test_cache_invalidated_by_module_version_fold_and_fps(fake_ml, synth_video, monkeypatch):
    fake_ml(versions={"ml.detect": "1"})
    run(synth_video, "j", None, 60)
    fake_ml(versions={"ml.detect": "2"})
    r = run(synth_video, "j", None, 60)
    assert r.meta["stages"]["detect"]["status"] == "ok"           # новая версия детектора
    assert r.meta["stages"]["track"]["status"] == "ok"            # и всё, что от него зависит
    r = run(synth_video, "j", None, 60)
    assert r.meta["stages"]["detect"]["status"] == "cached"
    monkeypatch.setenv("SENTINEL_FOLD", "3")                       # другой fold обучаемой модели
    r = run(synth_video, "j", None, 60)
    assert r.meta["stages"]["temporal"]["status"] == "ok"
    base_fps = r.meta["sample_fps"]
    r = run(synth_video, "j", None, 60, overrides={"sampling": {"base_fps": 1.0, "min_fps": 1.0}})
    assert r.meta["sample_fps"] == 1.0 != base_fps and r.meta["stages"]["detect"]["status"] == "ok"


def test_catalog_change_invalidates_rules(fake_ml, synth_video, monkeypatch):
    fake_ml()
    run(synth_video, "j", None, 60)
    import backend.pipeline.orchestrator as orch

    monkeypatch.setattr(orch, "load_catalog_safe", lambda: ({"version": 1, "types": {"V99": {"name": "x"}}}, "test", None))
    r = run(synth_video, "j", None, 60)
    assert r.meta["stages"]["rules"]["status"] == "ok" and r.meta["stages"]["scene"]["status"] == "cached"


def test_from_cache_mode_reruns_rules_only(fake_ml, synth_video):
    fake_ml(versions={"ml.detect": "1"})
    run(synth_video, "j", None, 60)
    calls = []
    fake_ml({"ml.rules.run_rules": lambda *a: calls.append(1) or []}, versions={"ml.detect": "999"})
    r = run(synth_video, "j", None, 60, overrides={"cache": {"mode": "from_cache"}})
    assert r.meta["stages"]["detect"]["status"] == "cached"       # устаревший ключ допустим в from_cache
    assert "detect(stale)" in r.meta["cache"]["hits"]
    assert r.meta["stages"]["rules"]["status"] == "ok" and calls


def test_cache_off(fake_ml, synth_video, tmp_path):
    fake_ml()
    run(synth_video, "j", None, 60, overrides={"cache": {"mode": "off"}})
    assert not (tmp_path / "cache").exists()


def test_choose_sample_fps():
    cfg = load_config()
    assert choose_sample_fps(60, 240, "cuda", cfg) == 5.0
    assert choose_sample_fps(145, 240, "cuda", cfg) == 5.0          # 725 кадров ≤ 900
    assert choose_sample_fps(400, 240, "cuda", cfg) == 2.0          # длинное видео → вниз, но ≥ min_fps
    assert choose_sample_fps(145, 240, "cpu", cfg) == 2.0           # CPU: ≤ 300 кадров
    assert choose_sample_fps(60, 120, "cuda", cfg) == 2.5           # жёсткий дедлайн → меньше fps


def test_budget_timeouts_and_skip():
    cfg = load_config()
    now = [0.0]
    b = TimeBudget(60, cfg, n_frames=725, device="cuda", clock=lambda: now[0])
    assert 0 < b.timeout("detect") <= 60 - cfg["reserve_s"]["fusion_min"]
    assert b.timeout("vlm_verify") <= 60 - cfg["reserve_s"]["fusion"]
    assert b.can_start("vlm_verify")[0]
    now[0] = 50.0                                                   # осталось 10 с
    ok, why = b.can_start("vlm_discover")
    assert not ok and "бюджет" in why
    assert b.can_start("rules")[0]                                  # обязательная стадия ещё идёт
    assert b.timeout("detect") <= 10 - cfg["reserve_s"]["fusion_min"] + 1e-9
    now[0] = 70.0
    assert not b.can_start("detect")[0] and b.timeout("fusion") >= 3.0


@pytest.mark.parametrize("raw,norm", [("Х-3.mov", "X-3"), ("X-1.MOV", "X-1"), ("х-47.mov", "x-47"),
                                      ("ХР-5.mov", "XP-5")])
def test_normalize_video_name(raw, norm):
    assert normalize_video_name(raw) == norm


def test_normalize_video_name_always_ascii():
    out = normalize_video_name("видео 1.mov")
    assert out.isascii() and out.endswith("1")
