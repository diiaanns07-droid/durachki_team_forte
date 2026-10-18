"""Отмена задачи и освобождение GPU: воркер останавливается (модели выгружены), вердикта нет."""
from __future__ import annotations

import threading
import time

import pytest

from backend.pipeline import cancel, mocks, release_gpu, run
from backend.pipeline.worker import get_executor


def _run_in_thread(*args, **kwargs):
    box = {}
    th = threading.Thread(target=lambda: box.setdefault("res", run(*args, **kwargs)), daemon=True)
    th.start()
    return th, box


@pytest.mark.slow
def test_cancel_kills_worker_during_hung_stage(synth_video, monkeypatch, fake_ml_files):
    base = fake_ml_files()
    monkeypatch.setenv("SENTINEL_EXECUTOR", "process")
    monkeypatch.setenv("SENTINEL_FAKE_SLEEP", "60")
    th, box = _run_in_thread(synth_video, "job-cancel", None, 120,
                             overrides={"executor": {"extra_sys_path": [str(base)]}})
    time.sleep(3.0)
    t0 = time.perf_counter()
    assert cancel("job-cancel") is True
    th.join(timeout=20)
    assert not th.is_alive() and time.perf_counter() - t0 < 15
    res = box["res"]
    assert res.status == "failed" and res.verdict is None and res.meta["cancelled"] is True
    assert res.meta["stages"]["detect"]["status"] == "cancelled"
    assert res.meta["executor"]["kills"] >= 1
    assert not get_executor({"executor": {"mode": "process"}}).alive()   # процесс с моделями остановлен
    assert cancel("job-cancel") is False                                 # задача уже завершена


def test_cancel_while_waiting_for_worker(fake_ml, synth_video):
    fake_ml()
    ex = get_executor({"executor": {"mode": "inline"}})
    ex._lease.acquire()  # другая задача держит воркер
    try:
        th, box = _run_in_thread(synth_video, "job-queued", None, 60)
        time.sleep(0.5)
        assert cancel("job-queued") is True
        th.join(timeout=5)
        res = box["res"]
        assert res.status == "failed" and res.verdict is None and "очеред" in res.meta["error"]
    finally:
        ex._lease.release()


def test_cancel_between_stages_inline(fake_ml, synth_video):
    def detect_then_cancel(clip, every_n=1):
        cancel("job-mid")
        return mocks.detect(clip, every_n)

    fake_ml({"ml.detect.detect": detect_then_cancel})
    res = run(synth_video, "job-mid", None, 60)
    assert res.status == "failed" and res.verdict is None and res.meta["cancelled"] is True
    assert res.meta["stages"]["track"]["status"] == "cancelled"
    assert res.meta["completeness"]["complete"] is False


def test_release_gpu_inline_calls_unload(fake_ml, synth_video):
    calls = []
    mods = fake_ml()
    mods["ml.detect"].unload = lambda: calls.append("detect")
    run(synth_video, "job-rel", None, 60)
    rep = release_gpu()
    assert rep["mode"] == "inline" and "ml.detect" in rep["hooks"] and calls == ["detect"]


@pytest.mark.slow
def test_release_gpu_process_stops_worker(synth_video, monkeypatch):
    monkeypatch.setenv("SENTINEL_EXECUTOR", "process")
    run(synth_video, "job-rel2", None, 60, overrides={"mock": True})
    ex = get_executor({"executor": {"mode": "process"}, "mock": True})
    assert ex.alive()
    assert release_gpu()["action"] == "shutdown" and not ex.alive()
