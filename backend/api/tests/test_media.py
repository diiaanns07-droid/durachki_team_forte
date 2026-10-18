"""Range-видео (API и cookie-UI для <video>), кадры, фронт, copilot, request-id."""
from __future__ import annotations

import sys
import types

from .conftest import FTYP_ONLY, H, KEY, ok_run, spec, upload, wait_job

PAYLOAD = FTYP_ONLY + bytes(range(256)) * 40  # 92 + 10240 байт


def _job(c):
    jid = upload(c, PAYLOAD).json()["job_id"]
    wait_job(c, jid)
    return jid


def test_range_requests_api(client_factory):
    with client_factory(spec(ok_run)) as c:
        jid = _job(c)
        url, size = f"/api/v1/jobs/{jid}/video", len(PAYLOAD)
        full = c.get(url, headers=H)
        assert full.status_code == 200 and full.content == PAYLOAD
        assert full.headers["accept-ranges"] == "bytes"
        r = c.get(url, headers={**H, "Range": "bytes=0-99"})
        assert r.status_code == 206 and r.content == PAYLOAD[:100]
        assert r.headers["content-range"] == f"bytes 0-99/{size}"
        assert r.headers["content-length"] == "100"
        r = c.get(url, headers={**H, "Range": "bytes=-10"})
        assert r.status_code == 206 and r.content == PAYLOAD[-10:]
        r = c.get(url, headers={**H, "Range": f"bytes={size - 5}-"})
        assert r.content == PAYLOAD[-5:]
        r = c.get(url, headers={**H, "Range": f"bytes=100-{size * 2}"})  # конец обрезается
        assert r.status_code == 206 and r.content == PAYLOAD[100:]
        r = c.get(url, headers={**H, "Range": f"bytes={size}-"})
        assert r.status_code == 416 and r.headers["content-range"] == f"bytes */{size}"
        h = c.head(url, headers={**H, "Range": "bytes=10-19"})
        assert h.status_code == 206 and h.headers["content-length"] == "10" and h.content == b""
        assert full.headers["content-type"].startswith("video/")


def test_range_video_for_ui_player_via_cookie(client_factory):
    """<video src="/ui/v1/jobs/<id>/video"> — браузер шлёт cookie сам, X-API-Key не нужен."""
    with client_factory(spec(ok_run)) as c:
        jid = _job(c)
        url = f"/ui/v1/jobs/{jid}/video"
        assert c.get(url, headers={"Range": "bytes=0-9"}).status_code == 401
        c.post("/ui/v1/session", json={"api_key": KEY})
        r = c.get(url, headers={"Range": "bytes=0-9"})
        assert r.status_code == 206 and r.content == PAYLOAD[:10]
        assert c.get(f"/ui/v1/jobs/{jid}/frames/0.jpg").content[:2] == b"\xff\xd8"


def test_frames_endpoint(client_factory):
    with client_factory(spec(ok_run)) as c:
        jid = _job(c)
        r = c.get(f"/api/v1/jobs/{jid}/frames/0.jpg", headers=H)
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        assert r.content[:2] == b"\xff\xd8"
        assert c.get(f"/api/v1/jobs/{jid}/frames/7.jpg", headers=H).status_code == 404
        d = c.get(f"/api/v1/jobs/{jid}", headers=H).json()
        assert d["meta"]["api"]["frames"][0]["path"] == "frames/0.jpg"


def test_copilot_501_then_router(client_factory, monkeypatch):
    # без copilot -> 501, даже если в дереве есть настоящий модуль A14 (интеграция)
    monkeypatch.setitem(sys.modules, "copilot", None)
    monkeypatch.setitem(sys.modules, "copilot.router", None)
    with client_factory(spec(ok_run)) as c:
        jid = _job(c)
        r = c.post(f"/api/v1/jobs/{jid}/ask", headers=H, json={"question": "почему?"})
        assert r.status_code == 501
    pkg = types.ModuleType("copilot")
    mod = types.ModuleType("copilot.router")

    def ask(job_id, question, result=None):
        return {"answer": f"{job_id}:{question}:{result['verdict']}"}
    mod.ask = ask
    pkg.router = mod
    monkeypatch.setitem(sys.modules, "copilot", pkg)
    monkeypatch.setitem(sys.modules, "copilot.router", mod)
    with client_factory(spec(ok_run)) as c:
        jid = _job(c)
        r = c.post(f"/api/v1/jobs/{jid}/ask", headers=H, json={"question": "почему?"})
        assert r.status_code == 200 and r.json()["answer"] == f"{jid}:почему?:violation"
        assert c.post(f"/api/v1/jobs/{jid}/ask", headers=H, json={}).status_code == 400


def test_frontend_dist_spa_fallback(client_factory, tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>SPA</html>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("nope", encoding="utf-8")
    with client_factory(spec(ok_run), frontend_dir=dist) as c:
        assert c.get("/").text == "<html>SPA</html>"
        assert c.get("/jobs/123/details").text == "<html>SPA</html>"  # SPA-маршрут
        a = c.get("/assets/app.js")
        assert a.text == "console.log(1)" and "immutable" in a.headers["cache-control"]
        assert c.get("/api/v1/nope").status_code == 404
        assert c.get("/api/v1/nope").headers["content-type"].startswith("application/json")
        assert "nope" not in c.get("/..%2Fsecret.txt").text


def test_builtin_page_without_frontend(client_factory):
    with client_factory(spec(ok_run)) as c:
        r = c.get("/")
        assert r.status_code == 200 and "Kassa Sentinel" in r.text
        assert KEY not in r.text


def test_request_id_and_no_cors_by_default(client_factory):
    with client_factory(spec(ok_run)) as c:
        r = c.get("/healthz", headers={"Origin": "https://evil.example"})
        assert r.headers.get("x-request-id")
        assert "access-control-allow-origin" not in r.headers
        assert c.get("/healthz", headers={"X-Request-ID": "abc-123"}).headers["x-request-id"] == "abc-123"
    with client_factory(spec(ok_run), cors_origins=["http://localhost:5173"]) as c:
        r = c.get("/healthz", headers={"Origin": "http://localhost:5173"})
        assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_copilot_make_router_from_a14(client_factory, monkeypatch):
    """Интерфейс A14: copilot.router.make_router(get_job, prefix, dependencies) -> APIRouter."""
    from fastapi import APIRouter

    def make_router(get_job=None, service=None, prefix="/api/v1", dependencies=None):
        r = APIRouter(prefix=prefix, dependencies=list(dependencies or []))

        @r.post("/jobs/{job_id}/ask")
        async def ask(job_id: str):
            job = get_job(job_id)
            if job is None:
                return {"error": "job_not_found"}
            return {"answer": "ok", "status": job["status"], "verdict": job.get("verdict")}

        @r.get("/copilot/status")
        async def status():
            return {"available": True}
        return r

    pkg = types.ModuleType("copilot")
    mod = types.ModuleType("copilot.router")
    mod.make_router = make_router
    pkg.router = mod
    monkeypatch.setitem(sys.modules, "copilot", pkg)
    monkeypatch.setitem(sys.modules, "copilot.router", mod)
    with client_factory(spec(ok_run)) as c:
        jid = _job(c)
        assert c.post(f"/api/v1/jobs/{jid}/ask", json={"question": "?"}).status_code == 401
        r = c.post(f"/api/v1/jobs/{jid}/ask", headers=H, json={"question": "?"})
        assert r.status_code == 200 and r.json() == {"answer": "ok", "status": "done", "verdict": "violation"}
        assert c.get("/api/v1/copilot/status", headers=H).json()["available"] is True
        assert c.get("/ui/v1/copilot/status").status_code == 401
        csrf = c.post("/ui/v1/session", json={"api_key": KEY}).json()["csrf_token"]
        assert c.post(f"/ui/v1/jobs/{jid}/ask", json={"question": "?"}).status_code == 403  # без CSRF
        r = c.post(f"/ui/v1/jobs/{jid}/ask", json={"question": "?"}, headers={"X-CSRF-Token": csrf})
        assert r.status_code == 200 and r.json()["status"] == "done"
