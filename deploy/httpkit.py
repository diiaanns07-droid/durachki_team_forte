"""Минимальный HTTP-клиент на stdlib для healthcheck / tunnel_probe / sla_bench (без зависимостей)."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class Response:
    status: int
    headers: dict[str, str]
    body: bytes
    elapsed_s: float
    error: str = ""
    set_cookies: list[str] = field(default_factory=list)

    def json(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None


def _opener(url: str) -> urllib.request.OpenerDirector:
    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    if host in ("localhost", "127.0.0.1", "::1"):  # к себе — никогда через прокси
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def request(method: str, url: str, headers: Optional[dict[str, str]] = None, body: Optional[bytes] = None,
            timeout: float = 30.0) -> Response:
    req = urllib.request.Request(url, data=body, method=method, headers=dict(headers or {}))
    req.add_header("User-Agent", "kassa-sentinel-deploy/1")
    t0 = time.perf_counter()
    try:
        with _opener(url).open(req, timeout=timeout) as r:
            data = r.read()
            return Response(r.status, {k.lower(): v for k, v in r.headers.items()}, data,
                            time.perf_counter() - t0, set_cookies=r.headers.get_all("Set-Cookie") or [])
    except urllib.error.HTTPError as e:
        data = e.read() if e.fp else b""
        return Response(e.code, {k.lower(): v for k, v in (e.headers or {}).items()}, data,
                        time.perf_counter() - t0, set_cookies=(e.headers.get_all("Set-Cookie") or []) if e.headers else [])
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return Response(0, {}, b"", time.perf_counter() - t0, error=f"{type(e).__name__}: {getattr(e, 'reason', e)}")


def multipart(field_name: str, filename: str, data: bytes, content_type: str = "video/quicktime") -> tuple[bytes, str]:
    boundary = "----sentinel" + uuid.uuid4().hex
    head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field_name}\"; filename=\"{filename}\"\r\n"
            f"Content-Type: {content_type}\r\n\r\n").encode()
    return head + data + f"\r\n--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


def api_key_from_env() -> str:
    key = os.environ.get("SENTINEL_API_KEY") or ""
    if not key:
        keys = [k.strip() for k in (os.environ.get("SENTINEL_API_KEYS") or "").split(",") if k.strip()]
        key = keys[0] if keys else ""
    return key


def submit(base: str, video: Path, api_key: str, timeout: float = 300.0) -> Response:
    body, ctype = multipart("file", video.name, Path(video).read_bytes())
    return request("POST", base.rstrip("/") + "/api/v1/analyze", {"X-API-Key": api_key, "Content-Type": ctype},
                   body, timeout=timeout)


def poll(base: str, job_id: str, api_key: str, limit_s: float = 300.0, every_s: float = 1.0) -> tuple[Optional[dict], float]:
    t0 = time.perf_counter()
    url = base.rstrip("/") + f"/api/v1/jobs/{urllib.parse.quote(job_id)}"
    last = None
    while time.perf_counter() - t0 < limit_s:
        r = request("GET", url, {"X-API-Key": api_key}, timeout=30)
        last = r.json() if r.status == 200 else None
        if isinstance(last, dict) and last.get("status") in ("done", "failed"):
            return last, time.perf_counter() - t0
        time.sleep(every_s)
    return last, time.perf_counter() - t0


def validate_job(doc: Any) -> list[str]:
    """Обязательные поля ответа ТЗ (доп. поля разрешены)."""
    import base64

    if not isinstance(doc, dict):
        return ["ответ не JSON-объект"]
    errs = [f"нет поля {k}" for k in ("job_id", "status") if k not in doc]
    if doc.get("status") not in ("processing", "done", "failed"):
        errs.append(f"status={doc.get('status')!r}")
    if doc.get("status") == "done":
        if doc.get("verdict") not in ("violation", "no_violation"):
            errs.append(f"verdict={doc.get('verdict')!r}")
        vs = doc.get("violations")
        if not isinstance(vs, list):
            errs.append("violations не список")
            vs = []
        if doc.get("verdict") == "violation" and not vs:
            errs.append("verdict=violation, но violations пуст")
        for i, v in enumerate(vs):
            for k, typ in (("type", str), ("timecode_sec", (int, float)), ("description", str), ("evidence_frames", list)):
                if not isinstance(v.get(k) if isinstance(v, dict) else None, typ):
                    errs.append(f"violations[{i}].{k} отсутствует/неверного типа")
            for j, fr in enumerate((v.get("evidence_frames") or []) if isinstance(v, dict) else []):
                if not isinstance(fr, dict) or not isinstance(fr.get("timestamp_sec"), (int, float)):
                    errs.append(f"violations[{i}].evidence_frames[{j}].timestamp_sec")
                    continue
                try:
                    raw = base64.b64decode(fr.get("image_base64") or "", validate=True)
                except ValueError:
                    raw = b""
                if raw[:2] != b"\xff\xd8":
                    errs.append(f"violations[{i}].evidence_frames[{j}] не JPEG base64")
    return errs
