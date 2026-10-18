"""GET /jobs/{id}/observations — расширение контракта для оверлеев UI (ключ/сессия, 404 без файла)."""
from __future__ import annotations

import json

from .conftest import FTYP_ONLY, H, make_result, spec, upload, wait_job


def test_observations_need_auth_and_404_without_file(client_factory):
    with client_factory(spec(lambda v, j, p, d=240: make_result(j))) as c:
        assert c.get("/api/v1/jobs/abcdef123456/observations").status_code == 401
        assert c.get("/ui/v1/jobs/abcdef123456/observations").status_code == 401
        jid = upload(c, FTYP_ONLY).json()["job_id"]
        wait_job(c, jid)
        r = c.get(f"/api/v1/jobs/{jid}/observations", headers=H)
        assert r.status_code == 404 and "detail" in r.json()
        assert c.get("/api/v1/jobs/..%2F..%2Fsecret/observations", headers=H).status_code == 404


def test_observations_served_as_written_by_pipeline(client_factory, tmp_path):
    items = [{"kind": "meta", "engine": "seqrules", "rules": {"phone": {"status": "ok"}}, "zones": []},
             {"kind": "episode", "id": "A-1-1", "rule": "phone", "status": "event",
              "frames": [{"t": 1.0, "state": "observed", "person": [0.1, 0.1, 0.4, 0.5]}]}]

    def run(video_path, job_id, progress, deadline_s=240):
        d = tmp_path / "data" / "jobs" / job_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "observations.json").write_text(json.dumps({"job_id": job_id, "items": items}), encoding="utf-8")
        return make_result(job_id)

    with client_factory(spec(run)) as c:
        jid = upload(c, FTYP_ONLY).json()["job_id"]
        d = wait_job(c, jid)
        assert d["status"] == "done" and "observations" not in d       # схема ТЗ не меняется
        r = c.get(f"/api/v1/jobs/{jid}/observations", headers=H)
        assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
        body = r.json()
        assert body["job_id"] == jid and body["items"][1]["frames"][0]["state"] == "observed"
