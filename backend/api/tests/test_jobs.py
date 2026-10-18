"""Жизненный цикл задач: схема ТЗ, нет модели, исключение, таймаут, очередь, рестарт."""
from __future__ import annotations

import base64
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from backend.api.results import schema_errors
from backend.api.store import JobStore
from backend.api.worker import JobManager, PipelineSpec

from .conftest import FTYP_ONLY, H, KEY, Gate, make_result, ok_run, spec, upload, wait_job, wait_ready


def test_full_cycle_matches_tz_schema(client_factory):
    with client_factory(spec(ok_run)) as c:
        r = upload(c, FTYP_ONLY, name="Х-3.MOV")  # кириллическая Х, верхний регистр
        assert r.status_code == 200
        body = r.json()
        assert set(body) >= {"job_id"} and body["status"] == "processing"
        p = c.get(f"/api/v1/jobs/{body['job_id']}", headers=H).json()
        assert p["status"] in ("processing", "done")
        if p["status"] == "processing":
            assert {"job_id", "status", "stage", "progress"} <= set(p)
        d = wait_job(c, body["job_id"])
        assert d["status"] == "done" and d["verdict"] == "violation"
        assert schema_errors(d) == []
        v = d["violations"][0]
        assert {"type", "timecode_sec", "description", "evidence_frames"} <= set(v)
        f = v["evidence_frames"][0]
        assert isinstance(f["timestamp_sec"], float)
        assert base64.b64decode(f["image_base64"])[:2] == b"\xff\xd8"
        assert d["meta"]["api"]["video"]["filename_normalized"] == "X-3.MOV"
        assert d["meta"].get("fake_pipeline") is None


def test_explicit_fake_pipeline_only(client_factory):
    with client_factory(None, fake_pipeline=True, fake_pipeline_s=0.05) as c:
        assert wait_ready(c).json()["pipeline"] == "fake"
        d = wait_job(c, upload(c, FTYP_ONLY).json()["job_id"])
        assert d["status"] == "done" and d["meta"]["fake_pipeline"] is True
        assert "ТЕСТОВЫЙ" in d["violations"][0]["description"]
        assert schema_errors(d) == []


def test_missing_pipeline_is_not_ready_and_fails(client_factory):
    """Нет backend.pipeline / весов -> НЕ фейк: readyz=503, задача failed."""
    with client_factory(None, pipeline="no_such_module_for_sentinel_tests") as c:
        r = wait_ready(c)
        assert r.status_code == 503 and r.json()["ready"] is False
        assert r.json()["state"] == "unavailable" and r.json()["pipeline"] == "unavailable"
        d = wait_job(c, upload(c, FTYP_ONLY).json()["job_id"])
        assert d["status"] == "failed" and d["verdict"] is None and d["violations"] == []
        assert "pipeline_unavailable" in d["error"]


def test_warmup_failure_reports_not_ready(client_factory):
    def bad_warmup():
        raise RuntimeError("weights not found")
    with client_factory(spec(ok_run, warmup=bad_warmup)) as c:
        r = wait_ready(c)
        assert r.status_code == 503 and r.json()["state"] == "warmup_failed"
        assert "weights not found" in r.json()["detail"]


def test_inference_exception_is_failed(client_factory):
    def boom(video_path, job_id, progress, deadline_s=240):
        progress("detect", 0.1)
        raise RuntimeError("CUDA out of memory")
    with client_factory(spec(boom)) as c:
        d = wait_job(c, upload(c, FTYP_ONLY).json()["job_id"])
        assert d["status"] == "failed" and d["verdict"] is None
        assert "CUDA out of memory" in d["error"]


def test_pipeline_reporting_failed_or_garbage_is_failed(client_factory):
    def failed(video_path, job_id, progress, deadline_s=240):
        r = make_result(job_id)
        r.status = "failed"
        r.meta = {"error": "decoder error"}
        return r

    def garbage(video_path, job_id, progress, deadline_s=240):
        return 42
    for fn, needle in ((failed, "decoder error"), (garbage, "unusable result")):
        with client_factory(spec(fn)) as c:
            d = wait_job(c, upload(c, FTYP_ONLY).json()["job_id"])
            assert d["status"] == "failed" and needle in d["error"]


def test_timeout_is_failed_partial_only_for_ui(client_factory):
    """Watchdog: не успели -> честный failed. partial.json -> только UI-подробности."""
    cancelled = threading.Event()

    def slow(video_path, job_id, progress, deadline_s=240):
        (Path(video_path).parent / "partial.json").write_text(json.dumps(
            {"verdict": "violation", "violations": [{"type": "x", "timecode_sec": 1.0}]}))
        try:
            for _ in range(200):
                time.sleep(0.02)
                progress("vlm", 0.5)
        except BaseException:
            cancelled.set()
            raise
        return make_result(job_id)

    kw = dict(watchdog_s=0.6, budget_s=0.5, job_cap_s=0.3, min_deadline_s=0.1, deadline_grace_s=0.1)
    with client_factory(spec(slow), **kw) as c:
        jid = upload(c, FTYP_ONLY).json()["job_id"]
        d = wait_job(c, jid)
        assert d["status"] == "failed" and d["verdict"] is None and d["violations"] == []
        assert "timeout" in d["error"] and "partial" not in d
        c.post("/ui/v1/session", json={"api_key": KEY})
        u = c.get(f"/ui/v1/jobs/{jid}").json()
        assert u["status"] == "failed" and u["partial"]["partial"]["verdict"] == "violation"
        assert u["partial"]["complete"] is False
        assert cancelled.wait(3), "progress() должен оборвать run() после таймаута"


def test_hung_pipeline_single_gpu_owner(client_factory):
    """Зависший run() (без progress): watchdog -> failed; GPU НЕ отдаётся второй задаче (один владелец
    вместе с A03), та честно получает failed по таймауту, /readyz = 503 gpu_hung. Как только run()
    вернулся — очередь работает дальше."""
    hang = threading.Event()
    calls = []

    def run(video_path, job_id, progress, deadline_s=240):
        calls.append(job_id)
        if len(calls) == 1:
            hang.wait(10)
        return make_result(job_id)

    kw = dict(watchdog_s=0.8, budget_s=0.7, job_cap_s=0.3, min_deadline_s=0.1, deadline_grace_s=0.1,
              hung_gpu_s=0.2)
    try:
        with client_factory(spec(run), **kw) as c:
            j1 = upload(c, FTYP_ONLY).json()["job_id"]
            time.sleep(0.1)
            j2 = upload(c, FTYP_ONLY).json()["job_id"]
            assert wait_job(c, j1)["status"] == "failed"
            d2 = wait_job(c, j2)
            assert d2["status"] == "failed" and "timeout" in d2["error"]
            assert len(calls) == 1  # второй run() не стартовал параллельно зависшему
            end = time.time() + 3
            while time.time() < end and c.get("/readyz").json()["state"] != "gpu_hung":
                time.sleep(0.05)
            r = c.get("/readyz")
            assert r.status_code == 503 and r.json()["state"] == "gpu_hung"
            hang.set()
            end = time.time() + 3
            while time.time() < end and c.get("/readyz").status_code != 200:
                time.sleep(0.05)
            assert c.get("/readyz").status_code == 200
            assert wait_job(c, upload(c, FTYP_ONLY).json()["job_id"])["status"] == "done"
    finally:
        hang.set()


def test_hung_pipeline_release_gpu_opt_in(client_factory):
    hang = threading.Event()
    calls = []

    def run(video_path, job_id, progress, deadline_s=240):
        calls.append(job_id)
        if len(calls) == 1:
            hang.wait(5)
        return make_result(job_id)

    kw = dict(watchdog_s=0.8, budget_s=0.7, job_cap_s=0.3, min_deadline_s=0.1, deadline_grace_s=0.1,
              zombie_release_gpu=True)
    try:
        with client_factory(spec(run), **kw) as c:
            j1 = upload(c, FTYP_ONLY).json()["job_id"]
            time.sleep(0.1)
            j2 = upload(c, FTYP_ONLY).json()["job_id"]
            assert wait_job(c, j1)["status"] == "failed"
            assert wait_job(c, j2)["status"] == "done"
    finally:
        hang.set()


def test_five_parallel_uploads(client_factory):
    with client_factory(spec(ok_run)) as c:
        with ThreadPoolExecutor(5) as ex:
            rs = list(ex.map(lambda i: upload(c, FTYP_ONLY, name=f"X-{i}.mov"), range(5)))
        assert [r.status_code for r in rs] == [200] * 5
        ids = [r.json()["job_id"] for r in rs]
        assert len(set(ids)) == 5
        for jid in ids:
            d = wait_job(c, jid)
            assert d["status"] == "done" and schema_errors(d) == []
            assert d["meta"]["api"]["deadline_s"] <= 120


def test_backlog_limit_503(client_factory):
    g = Gate()
    try:
        with client_factory(spec(g.run), max_queue=2) as c:
            assert upload(c, FTYP_ONLY).status_code == 200
            assert upload(c, FTYP_ONLY).status_code == 200
            r = upload(c, FTYP_ONLY)
            assert r.status_code == 503 and r.headers.get("retry-after")
    finally:
        g.release.set()


def test_fair_share_deadline(settings_factory, tmp_path):
    s = settings_factory(budget_s=270, job_cap_s=120, min_deadline_s=15)
    m = JobManager(s, JobStore(tmp_path / "db.sqlite3"), PipelineSpec(run=ok_run))
    now = time.time()
    assert m._deadline_for(now) == 120.0                       # одна задача: потолок
    assert m._deadline_for(now - 200) == 70.0                  # ждали 200 c: 270-200
    for i in range(7):                                         # жюри прислало 8 видео разом
        m.submit(f"job{i:08d}", queued_at=now)
    assert abs(m._deadline_for(now) - 270 / 8) < 0.2
    assert m._deadline_for(now - 265) == 15.0                  # ниже min_deadline не даём


def test_restart_requeues_unfinished(client_factory, settings_factory):
    g = Gate()
    with client_factory(spec(g.run)) as c:
        jid = upload(c, FTYP_ONLY).json()["job_id"]
        assert g.started.wait(5)
    # «процесс упал» посреди обработки: в БД running; новый процесс подхватывает
    with client_factory(spec(ok_run)) as c2:
        d = wait_job(c2, jid)
        assert d["status"] == "done"
    g.release.set()


def test_restart_gives_up_after_max_attempts(settings_factory, tmp_path):
    s = settings_factory(max_attempts=1)
    store = JobStore(s.db_path)
    video = tmp_path / "v.mov"
    video.write_bytes(FTYP_ONLY)
    store.create("deadbeef0001", video_path=str(video), filename="v.mov", size_bytes=1, sha256="x")
    assert store.mark_running("deadbeef0001")
    m = JobManager(s, store, PipelineSpec(run=ok_run))
    m._recover()
    row = store.get("deadbeef0001")
    assert row["status"] == "failed" and "crashed" in row["error"]


def test_cleanup_ttl(settings_factory):
    s = settings_factory(job_ttl_h=1)
    store = JobStore(s.db_path)
    m = JobManager(s, store, PipelineSpec(run=ok_run))
    jd = s.jobs_dir / "oldjob000001"
    jd.mkdir(parents=True)
    store.create("oldjob000001", video_path=str(jd / "video.mov"), filename="a", size_bytes=1,
                 sha256="x", now=1000)
    store.finalize("oldjob000001", "failed", error="x", now=1000)
    assert m.cleanup(now=1000 + 7200) == 1
    assert not jd.exists() and store.get("oldjob000001") is None


def test_unknown_job_404(client_factory):
    with client_factory(spec(ok_run)) as c:
        assert c.get("/api/v1/jobs/0123456789abcdef", headers=H).status_code == 404
        assert c.get("/api/v1/jobs/..%2F..%2Fetc", headers=H).status_code == 404


def test_timeout_calls_pipeline_cancel_and_frees_gpu(client_factory):
    """A03 cancel(job_id): после таймаута API просит пайплайн остановиться — run() возвращается,
    GPU освобождается без gpu_hung, следующая задача выполняется."""
    from backend.api.worker import PipelineSpec
    stop = threading.Event()
    cancelled = []

    def run(video_path, job_id, progress, deadline_s=240):
        if not cancelled:
            stop.wait(10)
        return make_result(job_id)

    def cancel(job_id):
        cancelled.append(job_id)
        stop.set()
        return True
    kw = dict(watchdog_s=0.8, budget_s=0.7, job_cap_s=0.3, min_deadline_s=0.1, deadline_grace_s=0.1,
              hung_gpu_s=0.2)
    try:
        with client_factory(PipelineSpec(run=run, cancel=cancel, mode="real"), **kw) as c:
            j1 = upload(c, FTYP_ONLY).json()["job_id"]
            d1 = wait_job(c, j1)
            assert d1["status"] == "failed" and "timeout" in d1["error"]
            end = time.time() + 3
            while time.time() < end and not cancelled:
                time.sleep(0.02)
            assert cancelled == [j1]
            assert wait_job(c, upload(c, FTYP_ONLY).json()["job_id"])["status"] == "done"
            assert c.get("/readyz").status_code == 200
    finally:
        stop.set()


def test_release_gpu_called_on_shutdown(client_factory):
    from backend.api.worker import PipelineSpec
    released = []
    with client_factory(PipelineSpec(run=ok_run, release_gpu=lambda: released.append(1) or {},
                                     mode="real")) as c:
        assert c.get("/healthz").status_code == 200
    assert released == [1]
