"""jury_sim против локального фейкового API (http.server) — проверяет код симулятора, не сервис и не модели."""
from __future__ import annotations

import base64
import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from eval.common import load_config
from eval.jury_sim import inspect_result, jpeg_size

from .conftest import make_pred, make_violation, write_synthetic_video

KEY = "test-key"


def _jpeg(w: int, h: int) -> str:
    import cv2
    import numpy as np

    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), 100, np.uint8))
    return base64.b64encode(buf.tobytes()).decode()


def _result(jid: str, behaviour: str) -> dict:
    v = make_violation(0.5, frames=[0.5])
    v["evidence_frames"][0]["image_base64"] = _jpeg(16, 24)        # портрет, как видео 64×96
    meta = {"models": ["PekingU/rtdetr_v2_r18vd", "Qwen/Qwen3-VL-4B-Instruct"], "mock": False, "coverage": 1.0}
    if behaviour == "mock":
        meta = {"models": ["mock-detector"], "mock": True, "coverage": 1.0}
    if behaviour == "partial":
        return make_pred(jid, [], meta={**meta, "coverage": 0.3}) | {"job_id": jid}
    if behaviour == "failed_silent":
        return {"job_id": jid, "status": "failed", "verdict": None, "violations": []}
    if behaviour == "late_tc":
        v["timecode_sec"] = 99.0
    return make_pred(jid, [v], meta=meta) | {"job_id": jid}


def _server(enforce_auth: bool = True, max_bytes: int = 1 << 20, behaviour: str = "good"):
    jobs: dict[str, float] = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # тишина
            pass

        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authed(self) -> bool:
            return not enforce_auth or self.headers.get("X-API-Key") == KEY

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > max_bytes:
                self.rfile.read(n)
                return self._send(413, {"detail": "too large"})
            body = self.rfile.read(n)
            if not self._authed():
                return self._send(401, {"detail": "bad key"})
            m = re.search(rb'filename="([^"]+)"', body)
            if not m or not m.group(1).lower().endswith((b".mov", b".mp4")):
                return self._send(415, {"detail": "unsupported"})
            jid = uuid.uuid4().hex
            jobs[jid] = time.time()
            self._send(200, {"job_id": jid})

        def do_GET(self):
            if not self._authed():
                return self._send(401, {"detail": "bad key"})
            jid = self.path.rstrip("/").split("/")[-1]
            if jid not in jobs:
                return self._send(404, {"detail": "nf"})
            if behaviour == "hang" or time.time() - jobs[jid] < 0.15:
                return self._send(200, {"job_id": jid, "status": "processing"})
            self._send(200, _result(jid, behaviour))

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def videos(tmp_path: Path) -> list[Path]:
    pytest.importorskip("cv2")
    return [write_synthetic_video(tmp_path / f"X-{i + 1}.mov", seconds=1.0, fps=10, size=(64, 96)) for i in range(5)]


def _cfg(**over):
    js = {"poll_s": 0.05, "total_limit_s": 3, "grace_s": 0.5, "big_file_mb": 2}
    js.update(over)
    return load_config(overrides={"jury_sim": js})


def _session():
    import requests

    s = requests.Session()
    s.trust_env = False          # локальный сервер — мимо прокси окружения
    return s


def _sim(videos, **kw):
    pytest.importorskip("requests")
    from eval.jury_sim import run_jury_sim

    behaviour = kw.pop("behaviour", "good")
    auth = kw.pop("enforce_auth", True)
    cfg = kw.pop("cfg", _cfg())
    srv = _server(enforce_auth=auth, behaviour=behaviour)
    try:
        return run_jury_sim(f"http://127.0.0.1:{srv.server_port}", KEY, videos, cfg, session=_session(), **kw)
    finally:
        srv.shutdown()


def _st(res: dict, needle: str) -> list[str]:
    return [c["status"] for c in res["checks"] if needle in c["name"]]


def test_jpeg_size_and_inspect():
    raw = base64.b64decode(_jpeg(16, 24))
    assert jpeg_size(raw) == (16, 24) and jpeg_size(b"nope") is None
    r = _result("j", "good")
    assert inspect_result(r, {"duration": 1.0, "width": 64, "height": 96})["jpeg_issues"] == []
    land = inspect_result(r, {"duration": 1.0, "width": 96, "height": 64})       # горизонтальное видео
    assert any("ориентация" in x for x in land["jpeg_issues"])
    assert inspect_result(_result("j", "late_tc"), {"duration": 1.0, "width": 64, "height": 96})["pts_issues"]
    assert inspect_result(_result("j", "partial"), {})["partial_as_clean"] is True
    assert inspect_result(_result("j", "mock"), {})["mock"] is True


def test_good_server_pass(videos):
    from eval.jury_sim import format_table

    res = _sim(videos, concurrency=2)
    assert res["summary"]["overall"] == "PASS", format_table(res)
    assert _st(res, "sequential: ≥ 5") == ["PASS"]
    assert any(c["name"].startswith("batch") for c in res["checks"])
    assert "image_base64" not in json.dumps(res)            # в отчёте только статусы и тайминги


def test_broken_auth_detected(videos):
    res = _sim(videos[:1], enforce_auth=False, skip_stress=True, big_mb=0)
    assert any("401" in c["name"] for c in res["checks"] if c["status"] == "FAIL")
    assert res["summary"]["overall"] == "FAIL"


@pytest.mark.parametrize("behaviour,needle", [
    ("mock", "настоящие модели"),
    ("partial", "неполный анализ"),
    ("failed_silent", "явной причиной"),
    ("late_tc", "PTS"),
])
def test_acceptance_failures(videos, behaviour, needle):
    res = _sim(videos[:5], behaviour=behaviour, skip_stress=True, big_mb=0)
    assert "FAIL" in _st(res, needle), [(c["name"], c["status"]) for c in res["checks"]]
    assert res["summary"]["overall"] == "FAIL"


def test_unfinished_and_too_few(videos):
    res = _sim(videos[:2], behaviour="hang", skip_stress=True, big_mb=0, cfg=_cfg(total_limit_s=0.3, grace_s=0.2))
    assert "FAIL" in _st(res, "завершены") and "NOT RUN" in _st(res, "≥ 5")
    assert res["summary"]["overall"] == "FAIL"


def test_judge_zero_seconds_is_valid_not_timeout():
    """Регрессия: `(accept_to_done_s or 1e9)` превращал мгновенный ответ 0.0 с (dedup/кэш) в таймаут."""
    from eval.jury_sim import FAIL, NOT_RUN, PASS, judge_accept_to_done

    assert judge_accept_to_done([{"video": "X-1", "accept_to_done_s": 0.0}], 300)[0] == PASS
    assert judge_accept_to_done([{"video": "X-1", "accept_to_done_s": 0.0},
                                 {"video": "X-2", "accept_to_done_s": 12.5}], 300)[0] == PASS
    st, detail = judge_accept_to_done([{"video": "X-1", "accept_to_done_s": None}], 300)
    assert st == NOT_RUN and "не измерено" in detail and "X-1" in detail      # нет времени ≠ таймаут и ≠ PASS
    assert judge_accept_to_done([{"video": "X-1", "accept_to_done_s": 0.0},
                                 {"video": "X-2", "accept_to_done_s": None}], 300)[0] == NOT_RUN
    assert judge_accept_to_done([{"video": "X-1", "accept_to_done_s": 301.0},
                                 {"video": "X-2", "accept_to_done_s": None}], 300)[0] == FAIL
    assert judge_accept_to_done([], 300)[0] == NOT_RUN


def _dedup_server():
    """Сервис с dedup по содержимому: повторная загрузка того же файла сразу отдаёт done (0 с ожидания)."""
    import hashlib

    seen: dict[str, str] = {}
    done: dict[str, float] = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.headers.get("X-API-Key") != KEY:
                return self._send(401, {})
            if not re.search(rb'filename="[^"]+\.(mov|mp4)"', body):
                return self._send(415, {})
            boundary = self.headers.get("Content-Type", "").split("boundary=")[-1].encode()
            content = body.split(b"\r\n\r\n", 1)[-1].rsplit(b"\r\n--" + boundary, 1)[0]   # байты файла
            h = hashlib.md5(content).hexdigest()
            if h not in seen:
                seen[h] = uuid.uuid4().hex
                done[seen[h]] = time.time() + 0.2
            self._send(200, {"job_id": seen[h]})

        def do_GET(self):
            if self.headers.get("X-API-Key") != KEY:
                return self._send(401, {})
            jid = self.path.rstrip("/").split("/")[-1]
            if jid not in done:
                return self._send(404, {})
            if time.time() < done[jid]:
                return self._send(200, {"job_id": jid, "status": "processing"})
            self._send(200, _result(jid, "good"))

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_dedup_instant_answer_passes(videos):
    pytest.importorskip("requests")
    from eval.jury_sim import format_table, run_jury_sim

    srv = _dedup_server()
    try:
        same = [videos[0]] * 5                      # 5 загрузок одного файла: 2..5 — мгновенный ответ из dedup
        res = run_jury_sim(f"http://127.0.0.1:{srv.server_port}", KEY, same, _cfg(), session=_session(),
                           skip_stress=True, big_mb=0)
    finally:
        srv.shutdown()
    seq = next(c for c in res["checks"] if c["name"].startswith("sequential: ≤"))
    times = seq["data"]["accept_to_done_s"]
    assert min(times) < 0.1                          # действительно мгновенно
    assert seq["status"] == "PASS", format_table(res)
