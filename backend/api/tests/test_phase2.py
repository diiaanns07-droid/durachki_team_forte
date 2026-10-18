"""Фаза 2: честный исход, полнота, partial, mock, дедлайн от приёма, идемпотентность, размер ответа,
жизненный цикл (обрыв загрузки, падение воркера, битый MOV), PTS-кадры, строгий профиль жюри."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
import time
from pathlib import Path

import pytest

from backend.api.results import extract_jpeg, schema_errors
from backend.api.store import JobStore
from backend.api.uploads import UploadError, receive_video

from .conftest import (FTYP_ONLY, H, KEY, Gate, make_result, ok_run, spec, upload, wait_job,
                       wait_ready, write_mov)

TZ_TOP = {"job_id", "status", "verdict", "violations"}


def _ui(c):
    return c.post("/ui/v1/session", json={"api_key": KEY}).json()["csrf_token"]


def result_with(job_id, *, verdict="violation", meta=None, violations=None):
    r = make_result(job_id)
    r.verdict = verdict
    if violations is not None:
        r.violations = violations
    r.meta = dict(meta or {})
    return r


# ------------------------------------------------------------------ обязательные поля и исход
def test_processing_and_failed_have_required_fields(client_factory):
    g = Gate()
    try:
        with client_factory(spec(g.run)) as c:
            jid = upload(c, FTYP_ONLY).json()["job_id"]
            g.started.wait(5)
            p = c.get(f"/api/v1/jobs/{jid}", headers=H).json()
            assert TZ_TOP <= set(p) and p["status"] == "processing"
            assert p["verdict"] is None and p["violations"] == [] and "stage" in p
            ps = c.get(f"/api/v1/jobs/{jid}?profile=strict", headers=H).json()
            assert set(ps) == TZ_TOP
    finally:
        g.release.set()

    def boom(video_path, job_id, progress, deadline_s=240):
        raise RuntimeError("x")
    with client_factory(spec(boom)) as c:
        d = wait_job(c, upload(c, FTYP_ONLY).json()["job_id"])
        assert TZ_TOP <= set(d) and d["verdict"] is None and d["violations"] == []


@pytest.mark.parametrize("verdict,meta,violations,expect", [
    (None, {}, [], "failed"),                                         # no_violation не выдумываем
    ("violation", {}, [], "failed"),                                  # violation без нарушения
    ("no_violation", {}, [], "failed"),                               # нет meta.coverage -> не подтверждено
    ("no_violation", {"coverage": 0.5}, [], "failed"),                # покрыта половина видео
    ("no_violation", {"coverage": {"analyzed_sec": 99}}, [], "failed"),  # нет ratio
    ("no_violation", {"coverage": {"ratio": 0.99, "complete": False}}, [], "failed"),
    ("no_violation", {"coverage": 1.0, "complete": False}, [], "failed"),            # флаг A03
    ("no_violation", {"coverage": 1.0, "required_stages": ["detect"],
                      "stages": {"detect": {"status": "timeout"}}}, [], "failed"),   # стадии A03
    ("no_violation", {"coverage": {"ratio": 1.0, "complete": True, "required": ["detect"],
                                   "stages": {"detect": "ok"}}}, [], "done"),
    ("no_violation", {"coverage": 0.99}, [], "done"),
    ("violation", {"complete": False}, None, "done"),                 # нарушение с кадром — отдаём
    (None, {}, None, "done"),                                          # есть нарушения -> violation
])
def test_outcome_policy(client_factory, verdict, meta, violations, expect):
    def run(video_path, job_id, progress, deadline_s=240):
        return result_with(job_id, verdict=verdict, meta=meta, violations=violations)
    with client_factory(spec(run)) as c:
        d = wait_job(c, upload(c, FTYP_ONLY).json()["job_id"])
        assert d["status"] == expect, d
        if expect == "done":
            assert schema_errors(d) == []


def test_no_violation_candidates_hidden_from_jury(client_factory):
    def run(video_path, job_id, progress, deadline_s=240):
        return result_with(job_id, verdict="no_violation", meta={"coverage": 1.0},
                           violations=make_result(job_id).violations)
    with client_factory(spec(run)) as c:
        jid = upload(c, FTYP_ONLY).json()["job_id"]
        d = wait_job(c, jid)
        assert d["verdict"] == "no_violation" and d["violations"] == [] and "candidates" not in d
        _ui(c)
        u = c.get(f"/ui/v1/jobs/{jid}").json()
        assert len(u["candidates"]) == 1


def test_violation_without_timecode_is_dropped(client_factory):
    def run(video_path, job_id, progress, deadline_s=240):
        r = make_result(job_id)
        r.violations[0].timecode_sec = float("nan")
        r.violations[0].t_start = float("nan")
        r.violations[0].evidence_frames = []
        return r
    with client_factory(spec(run)) as c:
        d = wait_job(c, upload(c, FTYP_ONLY).json()["job_id"])
        assert d["status"] == "failed" and "таймкод" in d["error"]


# ------------------------------------------------------------------ partial.json и полнота
def _slow_with_partial(partial):
    def run(video_path, job_id, progress, deadline_s=240):
        (Path(video_path).parent / "partial.json").write_text(json.dumps(partial), encoding="utf-8")
        for _ in range(300):
            time.sleep(0.01)
            progress("vlm", 0.6)
        return make_result(job_id)
    return run


TIGHT = dict(watchdog_s=0.6, budget_s=0.5, job_cap_s=0.3, min_deadline_s=0.1, deadline_grace_s=0.1)
FRAME = {"timestamp_sec": 1.0, "image_base64": base64.b64encode(
    base64.b64decode(make_result("x").violations[0].evidence_frames[0].image_base64)).decode()}
PARTIAL_OK = {"status": "processing", "verdict": "violation",
              "violations": [{"type": "Посторонний", "timecode_sec": 1.0, "description": "d",
                              "evidence_frames": [FRAME]}],
              "meta": {"partial": True, "complete": False, "coverage": 1.0}}


def test_partial_default_off_is_failed_with_reason(client_factory):
    with client_factory(spec(_slow_with_partial(PARTIAL_OK)), **TIGHT) as c:
        jid = upload(c, FTYP_ONLY).json()["job_id"]
        d = wait_job(c, jid)
        assert d["status"] == "failed" and "partial_accept=false" in d["error"]
        _ui(c)
        u = c.get(f"/ui/v1/jobs/{jid}").json()
        assert u["partial"]["partial"]["verdict"] == "violation"
        assert u["partial"]["partial"]["violations"][0]["evidence_frames"][0]["image_base64"] == "<jpeg omitted>"


def test_partial_accepted_only_with_coverage(client_factory):
    low = json.loads(json.dumps(PARTIAL_OK))
    low["meta"]["coverage"] = 0.4
    with client_factory(spec(_slow_with_partial(low)), partial_accept=True, **TIGHT) as c:
        d = wait_job(c, upload(c, FTYP_ONLY).json()["job_id"])
        assert d["status"] == "failed" and "incomplete" in d["error"]
    with client_factory(spec(_slow_with_partial(PARTIAL_OK)), partial_accept=True, **TIGHT) as c:
        d = wait_job(c, upload(c, FTYP_ONLY).json()["job_id"])
        assert d["status"] == "done" and d["verdict"] == "violation"
        assert d["meta"]["partial"] is True and d["meta"]["api"]["partial"] is True
        assert any("PARTIAL" in w for w in d["meta"]["warnings"])
        assert schema_errors(d) == []
    nothing = {"status": "processing", "verdict": None, "violations": [], "meta": {"coverage": 1.0}}
    with client_factory(spec(_slow_with_partial(nothing)), partial_accept=True, **TIGHT) as c:
        d = wait_job(c, upload(c, FTYP_ONLY).json()["job_id"])
        assert d["status"] == "failed"  # неполный анализ != «нарушений нет»


# ------------------------------------------------------------------ mock в production
def test_mock_env_refused_in_production(client_factory, monkeypatch):
    monkeypatch.setenv("SENTINEL_MOCK", "1")
    with client_factory(spec(ok_run)) as c:
        r = wait_ready(c)
        assert r.status_code == 503 and r.json()["state"] == "mock_not_allowed"
        d = wait_job(c, upload(c, FTYP_ONLY).json()["job_id"])
        assert d["status"] == "failed" and "mock" in d["error"].lower()
    with client_factory(spec(ok_run), allow_mock=True) as c:
        assert wait_ready(c).status_code == 200


def test_mock_result_refused(client_factory, monkeypatch):
    def run(video_path, job_id, progress, deadline_s=240):
        return result_with(job_id, meta={"mock": True})
    with client_factory(spec(run)) as c:
        d = wait_job(c, upload(c, FTYP_ONLY).json()["job_id"])
        assert d["status"] == "failed" and "MOCK" in d["error"]
    monkeypatch.setenv("SENTINEL_MOCK", "1")  # явный mock-режим (так его понимает и A01)
    with client_factory(spec(run), allow_mock=True) as c:
        assert wait_job(c, upload(c, FTYP_ONLY).json()["job_id"])["status"] == "done"


# ------------------------------------------------------------------ дедлайн от приёма
def test_deadline_at_passed_when_supported(client_factory):
    seen = {}

    def run(video_path, job_id, progress, deadline_s=240, *, deadline_at=None):
        seen["deadline_at"], seen["deadline_s"] = deadline_at, deadline_s
        return make_result(job_id)
    t0 = time.time()
    with client_factory(spec_with_kw(run), budget_s=270) as c:
        d = wait_job(c, upload(c, FTYP_ONLY).json()["job_id"])
        assert d["status"] == "done" and d["meta"]["api"]["deadline_at_passed"] is True
    assert abs(seen["deadline_at"] - (t0 + 270)) < 5 and seen["deadline_s"] <= 120
    with client_factory(spec(ok_run)) as c:  # run без deadline_at — не передаём, без TypeError
        assert wait_job(c, upload(c, FTYP_ONLY).json()["job_id"])["status"] == "done"


def spec_with_kw(run):
    from backend.api.worker import PipelineSpec
    return PipelineSpec(run=run, mode="real", accepts_deadline_at=True, version="t")


def test_restart_keeps_deadline_from_acceptance(client_factory, settings_factory, tmp_path):
    s = settings_factory()
    store = JobStore(s.db_path)
    now = time.time()
    for jid, age in (("oldjob0000001", 400), ("freshjob00001", 100)):
        jd = s.jobs_dir / jid
        jd.mkdir(parents=True)
        (jd / "video.mov").write_bytes(FTYP_ONLY)
        store.create(jid, video_path=str(jd / "video.mov"), filename="v.mov", size_bytes=1,
                     sha256=jid, now=now - age)
    store.close()
    with client_factory(spec(ok_run)) as c:
        old = wait_job(c, "oldjob0000001")
        assert old["status"] == "failed" and "deadline passed" in old["error"]
        fresh = wait_job(c, "freshjob00001")
        assert fresh["status"] == "done" and fresh["meta"]["api"]["total_s"] >= 99


# ------------------------------------------------------------------ идемпотентность
def test_dedup_same_file_same_result(client_factory, mov_bytes):
    with client_factory(spec(ok_run), dedup=True) as c:
        r1 = upload(c, FTYP_ONLY, name="Х-4.mov").json()
        d1 = wait_job(c, r1["job_id"])
        r2 = upload(c, FTYP_ONLY, name="Х-47.MOV").json()   # тот же файл под другим именем
        assert r2["job_id"] == r1["job_id"] and r2["deduplicated"] is True and r2["status"] == "done"
        assert wait_job(c, r2["job_id"]) == d1
        r3 = c.post("/api/v1/analyze?force=1", headers=H,
                    files={"file": ("a.mov", FTYP_ONLY, "video/quicktime")}).json()
        assert r3["job_id"] != r1["job_id"]
        r4 = upload(c, FTYP_ONLY + b"\x00").json()          # другие байты -> новая задача
        assert r4["job_id"] != r1["job_id"]


def test_dedup_does_not_reuse_failed(client_factory):
    calls = []

    def flaky(video_path, job_id, progress, deadline_s=240):
        calls.append(job_id)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return make_result(job_id)
    with client_factory(spec(flaky), dedup=True) as c:
        j1 = upload(c, FTYP_ONLY).json()["job_id"]
        assert wait_job(c, j1)["status"] == "failed"
        j2 = upload(c, FTYP_ONLY).json()["job_id"]
        assert j2 != j1 and wait_job(c, j2)["status"] == "done"


# ------------------------------------------------------------------ размер ответа
def _noise_jpeg_b64(seed: int, size=(900, 900)) -> str:
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    img = np.random.default_rng(seed).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return base64.b64encode(buf.tobytes()).decode()


def test_response_size_limit(client_factory):
    frames = [_noise_jpeg_b64(i) for i in range(4)]

    def run(video_path, job_id, progress, deadline_s=240):
        from .conftest import EvidenceFrame, Violation
        r = make_result(job_id)
        r.violations = [Violation(type="V", timecode_sec=float(i), description="d",
                                  evidence_frames=[EvidenceFrame(float(i), f) for f in frames])
                        for i in range(6)]
        return r
    with client_factory(spec(run), max_response_mb=1.0) as c:
        jid = upload(c, FTYP_ONLY).json()["job_id"]
        r = c.get(f"/api/v1/jobs/{jid}", headers=H)
        while r.json()["status"] == "processing":
            time.sleep(0.05)
            r = c.get(f"/api/v1/jobs/{jid}", headers=H)
        d = r.json()
        assert d["status"] == "done" and len(r.content) <= 1024 * 1024
        assert all(1 <= len(v["evidence_frames"]) <= 3 for v in d["violations"])
        assert len(d["violations"]) == 6 and schema_errors(d) == []
        for v in d["violations"]:
            raw = base64.b64decode(v["evidence_frames"][0]["image_base64"])
            assert raw[:3] == b"\xff\xd8\xff" and raw[-2:] == b"\xff\xd9"
        assert any("re-encoded" in w or "capped" in w for w in d["meta"]["warnings"])


# ------------------------------------------------------------------ жизненный цикл
def test_worker_crash_is_failed_and_queue_continues(client_factory, monkeypatch):
    from backend.api import worker as w
    orig = w.JobManager._complete
    state = {"n": 0}

    def crashing(self, *a, **kw):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("bug in finalize")
        return orig(self, *a, **kw)
    monkeypatch.setattr(w.JobManager, "_complete", crashing)
    with client_factory(spec(ok_run)) as c:
        d1 = wait_job(c, upload(c, FTYP_ONLY).json()["job_id"])
        assert d1["status"] == "failed" and "worker crashed" in d1["error"]
        assert wait_job(c, upload(c, FTYP_ONLY).json()["job_id"])["status"] == "done"


def test_upload_disconnect_leaves_nothing(tmp_path):
    from starlette.requests import ClientDisconnect

    async def stream():
        yield b"--B\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.mov\"\r\n\r\n"
        yield FTYP_ONLY + b"\x00" * 5000
        raise ClientDisconnect()

    async def run_sync(fn, *a):
        return fn(*a)
    with pytest.raises(ClientDisconnect):
        asyncio.run(receive_video(stream(), {"content-type": "multipart/form-data; boundary=B"},
                                  tmp_path, 10 << 20, run_sync))
    assert list(tmp_path.iterdir()) == []


def test_truncated_multipart_is_400_without_job(client_factory):
    with client_factory(spec(ok_run)) as c:
        body = (b"--B\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.mov\"\r\n"
                b"Content-Type: video/quicktime\r\n\r\n" + FTYP_ONLY)  # нет закрывающей границы
        r = c.post("/api/v1/analyze", headers={**H, "Content-Type": "multipart/form-data; boundary=B"},
                   content=body)
        assert r.status_code == 400
        assert c.get("/api/v1/jobs", headers=H).json()["jobs"] == []
        assert list(c.app.state.ctx.s.jobs_dir.iterdir()) == []


def test_truncated_mov_rejected_by_decoder(client_factory, mov_bytes):
    with client_factory(spec(ok_run), video_validation="required") as c:
        r = upload(c, mov_bytes[: len(mov_bytes) // 2])
        assert r.status_code == 415


def test_cyrillic_filename_star_and_upper_mov(client_factory):
    with client_factory(spec(ok_run)) as c:
        body = (b"--B\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.MOV\"; "
                b"filename*=UTF-8''%D0%A5-3.MOV\r\nContent-Type: video/quicktime\r\n\r\n"
                + FTYP_ONLY + b"\r\n--B--\r\n")
        r = c.post("/api/v1/analyze", headers={**H, "Content-Type": "multipart/form-data; boundary=B"},
                   content=body)
        assert r.status_code == 200
        d = wait_job(c, r.json()["job_id"])
        assert d["meta"]["api"]["filename"] == "Х-3.MOV"
        assert d["meta"]["api"]["video"]["filename_normalized"] == "X-3.MOV"


# ------------------------------------------------------------------ PTS-кадр и длительность
def test_extract_jpeg_uses_pts(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    p = write_mov(tmp_path / "v.mov", seconds=3.0)
    for t, k in ((0.0, 0), (1.0, 30), (2.5, 75)):
        img = cv2.imdecode(np.frombuffer(base64.b64decode(extract_jpeg(p, t)), np.uint8), cv2.IMREAD_COLOR)
        assert abs(float(img[400, 220].mean()) - (k * 5) % 255) < 20, (t, k)


def test_container_duration_preferred(tmp_path, mov_bytes):
    from backend.api.uploads import validate_video
    p = tmp_path / "v.mov"
    p.write_bytes(mov_bytes)
    info = validate_video(p, "required")
    assert abs(info["duration"] - 2.0) < 0.05 and "fps_nominal" in info


# ------------------------------------------------------------------ строгий профиль жюри
def test_strict_jury_profile_embeds_norm(client_factory):
    def run(video_path, job_id, progress, deadline_s=240):
        from .conftest import RuleRef
        r = make_result(job_id)
        r.violations[0].rules = [RuleRef(document="Правила X", clause="п. 5",
                                         url="https://adilet.zan.kz/rus/docs/TEST", quote="q")]
        return r
    with client_factory(spec(run), jury_profile="strict", jury_embed_rule=True) as c:
        d = wait_job(c, upload(c, FTYP_ONLY).json()["job_id"])
        assert set(d) == TZ_TOP
        v = d["violations"][0]
        assert set(v) == {"type", "timecode_sec", "description", "evidence_frames"}
        assert "Норма: Правила X, п. 5 — https://adilet.zan.kz/rus/docs/TEST" in v["description"]
        assert schema_errors(d) == []


def test_strict_profile_without_embedding_keeps_description(client_factory):
    with client_factory(spec(ok_run), jury_profile="strict") as c:
        d = wait_job(c, upload(c, FTYP_ONLY).json()["job_id"])
        assert d["violations"][0]["description"] == "Признаки возможного нарушения: тест."


_A03_STAGES = {k: {"status": "ok"} for k in ("ingest", "detect", "track", "rules")}
A03_STAGE_COVERAGE = {"coverage": {"ingest": 1.0, "detect": 1.0, "track": 1.0, "temporal": 0.0,
                                   "vlm_verify": None, "rules": 1.0},
                      "required_stages": ["ingest", "detect", "track", "rules"], "complete": True,
                      "stages": _A03_STAGES}  # как в реальном meta A03


def test_a03_stage_coverage_adapter_off_by_default(client_factory):
    def run(video_path, job_id, progress, deadline_s=240):
        return result_with(job_id, verdict="no_violation", violations=[],
                           meta=json.loads(json.dumps(A03_STAGE_COVERAGE)))
    with client_factory(spec(run)) as c:
        d = wait_job(c, upload(c, FTYP_ONLY).json()["job_id"])
        assert d["status"] == "failed" and "ratio" in d["error"]   # формат A03 != формат A01
    with client_factory(spec(run), coverage_adapt_stages=True) as c:
        d = wait_job(c, upload(c, FTYP_ONLY).json()["job_id"])
        assert d["status"] == "done" and d["verdict"] == "no_violation"
        assert d["meta"]["coverage"]["ratio"] == 1.0
        assert d["meta"]["coverage"]["source"] == "api-adapter:A03-per-stage"


def test_a03_stage_coverage_adapter_unknown_or_low(client_factory):
    for cov, need in (({"ingest": 1.0, "detect": None, "track": 1.0, "rules": 1.0}, "ratio"),
                      ({"ingest": 1.0, "detect": 0.6, "track": 1.0, "rules": 1.0}, "60%")):
        meta = {"coverage": cov, "required_stages": ["ingest", "detect", "track", "rules"],
                "stages": _A03_STAGES}

        def run(video_path, job_id, progress, deadline_s=240, meta=meta):
            return result_with(job_id, verdict="no_violation", violations=[], meta=json.loads(json.dumps(meta)))
        with client_factory(spec(run), coverage_adapt_stages=True) as c:
            d = wait_job(c, upload(c, FTYP_ONLY).json()["job_id"])
            assert d["status"] == "failed" and need in d["error"], d["error"]
