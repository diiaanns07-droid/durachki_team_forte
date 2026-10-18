"""Приёмка API одной командой по живому URL: схема ТЗ, авторизация, жизненный цикл.

    SENTINEL_API_KEY=... python -m backend.api.tools.acceptance --url http://127.0.0.1:8000 \
        [--video path.mov] [--timeout 300] [--out runs/acceptance.json]

Без --video генерирует синтетический MOV (cv2.VideoWriter) — реальные ролики не нужны.
Проверки: healthz/readyz; 401 без ключа/с неверным; 404 неизвестный job; 415 .txt / мусор с .mov /
обрезанный MOV; 400 пустой файл; 413 по Content-Length; обрыв загрузки не оставляет задач;
полный цикл (processing -> done|failed, схема на каждом опросе, ≤ лимита от приёма, ответ ≤ 3 МБ,
валидный JPEG); идемпотентность; Range 206/416; UI: вход только с секретом, HttpOnly cookie,
CSRF, видео по cookie с Range; CORS не открыт. В отчёт — только статусы и тайминги.
Код выхода 0 — нет FAIL."""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ._http import Client, api_key_from_env

TZ_TOP = ("job_id", "status", "verdict", "violations")


@dataclass
class Check:
    name: str
    status: str                 # PASS | WARN | FAIL | SKIP
    detail: str = ""
    seconds: Optional[float] = None
    data: dict[str, Any] = field(default_factory=dict)


def tz_errors(d: Any, status_expected: Optional[str] = None) -> list[str]:
    """Обязательные поля ТЗ на ЛЮБОМ статусе (схема A01) + типы нарушений и кадров."""
    errs: list[str] = []
    if not isinstance(d, dict):
        return ["response is not a JSON object"]
    for k in TZ_TOP:
        if k not in d:
            errs.append(f"missing {k}")
    st = d.get("status")
    if st not in ("processing", "done", "failed"):
        errs.append(f"bad status {st!r}")
    if status_expected and st != status_expected:
        errs.append(f"status {st!r} != {status_expected!r}")
    if st in ("processing", "failed") and d.get("verdict") is not None:
        errs.append(f"verdict must be null while {st}")
    if st == "done" and d.get("verdict") not in ("violation", "no_violation"):
        errs.append("done without verdict")
    if st == "done" and d.get("verdict") == "violation" and not d.get("violations"):
        errs.append("verdict=violation without violations")
    for i, v in enumerate(d.get("violations") or []):
        for k, t in (("type", str), ("timecode_sec", (int, float)), ("description", str),
                     ("evidence_frames", list)):
            if not isinstance(v.get(k), t) or isinstance(v.get(k), bool):
                errs.append(f"violations[{i}].{k} bad type")
        frames = v.get("evidence_frames") or []
        if not 1 <= len(frames) <= 3:
            errs.append(f"violations[{i}]: {len(frames)} evidence frames (expected 1–3)")
        for j, f in enumerate(frames):
            try:
                raw = base64.b64decode(f.get("image_base64") or "", validate=True)
            except Exception:
                raw = b""
            if raw[:3] != b"\xff\xd8\xff" or raw[-2:] != b"\xff\xd9":
                errs.append(f"violations[{i}].evidence_frames[{j}] is not a valid JPEG")
            if not isinstance(f.get("timestamp_sec"), (int, float)):
                errs.append(f"violations[{i}].evidence_frames[{j}].timestamp_sec bad type")
    return errs


def synth_mov(path: Path, seconds: float = 4.0, size: tuple[int, int] = (464, 848)) -> Optional[Path]:
    try:
        import cv2  # lazy
        import numpy as np
    except ImportError:
        return None
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
    if not w.isOpened():
        return None
    for i in range(int(seconds * 30)):
        f = np.full((size[1], size[0], 3), 40 + (i % 50), np.uint8)
        cv2.rectangle(f, (40 + i, 300), (200 + i, 600), (0, 160, 0), -1)
        w.write(f)
    w.release()
    return path


class Acceptance:
    def __init__(self, url: str, key: str, timeout_s: float = 300.0, limit_s: float = 300.0,
                 max_response_mb: float = 3.0, log: Callable[[str], None] = print):
        self.c = Client(url)
        self.url, self.key = url, key
        self.H = {"X-API-Key": key}
        self.timeout_s, self.limit_s = timeout_s, limit_s
        self.max_resp = int(max_response_mb * 1024 * 1024)
        self.checks: list[Check] = []
        self.log = log

    def add(self, name: str, ok: Optional[bool], detail: str = "", seconds: Optional[float] = None,
            warn: bool = False, **data: Any) -> None:
        st = "SKIP" if ok is None else ("PASS" if ok else ("WARN" if warn else "FAIL"))
        self.checks.append(Check(name, st, detail, None if seconds is None else round(seconds, 2), data))
        self.log(f"{st:5} {name} — {detail}")

    def _try(self, name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — любая ошибка проверки = FAIL с причиной
            self.add(name, False, f"{type(e).__name__}: {e}"[:300])

    # ---------------------------------------------------------------- проверки
    def health(self) -> None:
        r = self.c.get("/healthz")
        self.add("GET /healthz -> 200", r.status == 200, f"HTTP {r.status}")
        r = self.c.get("/readyz")
        d = r.json() or {}
        self.add("GET /readyz -> 200 (пайплайн загружен и прогрет)", r.status == 200,
                 f"HTTP {r.status} state={d.get('state')} pipeline={d.get('pipeline')} "
                 f"{d.get('detail') or ''}".strip(), warn=True)
        if d.get("pipeline") == "fake":
            self.add("pipeline не fake", False, "SENTINEL_FAKE_PIPELINE=1 — только dev/тест", warn=True)

    def auth(self, sample: bytes) -> None:
        for name, h in (("POST /analyze без ключа -> 401", {}),
                        ("POST /analyze с неверным ключом -> 401", {"X-API-Key": "wrong-key-xyz"})):
            r = self.c.post_file("/api/v1/analyze", sample, "a.mov", h)
            self.add(name, r.status == 401 and "detail" in (r.json() or {}), f"HTTP {r.status}")
        r = self.c.get("/api/v1/jobs/nonexistent-job", use_cookies=False)
        self.add("GET /jobs/{id} без ключа -> 401", r.status == 401, f"HTTP {r.status}")
        r = self.c.get("/api/v1/jobs/nonexistent-job-000", self.H)
        self.add("GET неизвестного job -> 404", r.status == 404, f"HTTP {r.status}")

    def bad_inputs(self, mov: Optional[bytes]) -> None:
        r = self.c.post_file("/api/v1/analyze", "это не видео".encode(), "notes.txt", self.H, "text/plain")
        self.add(".txt -> 415", r.status == 415, f"HTTP {r.status}")
        r = self.c.post_file("/api/v1/analyze", os.urandom(200_000), "random.mov", self.H)
        self.add("мусор с расширением .mov -> 415", r.status == 415, f"HTTP {r.status}")
        r = self.c.post_file("/api/v1/analyze", b"", "empty.mov", self.H)
        self.add("пустой файл -> 400", r.status == 400, f"HTTP {r.status}")
        if mov:
            r = self.c.post_file("/api/v1/analyze", mov[: len(mov) // 2], "truncated.mov", self.H)
            self.add("обрезанный MOV -> 415 (проверка декодером)", r.status == 415, f"HTTP {r.status}",
                     warn=r.status == 200)
        hdr = (f"POST {self.c.prefix}/api/v1/analyze HTTP/1.1\r\nHost: {self.c.host_header()}\r\n"
               f"X-API-Key: {self.key}\r\nContent-Type: multipart/form-data; boundary=B\r\n"
               f"Content-Length: {400 * 1024 * 1024}\r\nConnection: close\r\n\r\n").encode()
        out = self.c.raw(hdr + b"--B\r\n")
        line = out.split(b"\r\n", 1)[0].decode("latin-1", "replace")
        self.add("объявлено 400 МБ -> 413 до чтения тела", " 413 " in f"{line} ", line or "no response")

    def interrupted_upload(self, mov: bytes) -> None:
        before = len((self.c.get("/api/v1/jobs?limit=200", self.H).json() or {}).get("jobs") or [])
        body = (b"--B\r\nContent-Disposition: form-data; name=\"file\"; filename=\"cut.mov\"\r\n"
                b"Content-Type: video/quicktime\r\n\r\n" + mov[: len(mov) // 3])
        hdr = (f"POST {self.c.prefix}/api/v1/analyze HTTP/1.1\r\nHost: {self.c.host_header()}\r\n"
               f"X-API-Key: {self.key}\r\nContent-Type: multipart/form-data; boundary=B\r\n"
               f"Content-Length: {len(body) + 10_000_000}\r\n\r\n").encode()
        self.c.raw(hdr + body, close_after_send=True)
        time.sleep(1.0)
        after = len((self.c.get("/api/v1/jobs?limit=200", self.H).json() or {}).get("jobs") or [])
        self.add("обрыв загрузки -> задача не создаётся", after == before, f"jobs {before} -> {after}")

    def lifecycle(self, mov: bytes, name: str) -> Optional[str]:
        t0 = time.perf_counter()
        r = self.c.post_file("/api/v1/analyze", mov, name, self.H)
        t_acc = time.perf_counter()
        d = r.json() or {}
        ok = r.status == 200 and isinstance(d.get("job_id"), str)
        self.add("POST /analyze -> 200 {job_id} сразу после записи файла", ok,
                 f"HTTP {r.status} upload+accept {t_acc - t0:.2f}s", t_acc - t0)
        if not ok:
            return None
        jid = d["job_id"]
        bad: list[str] = []
        last: dict[str, Any] = {}
        size = 0
        while time.perf_counter() - t_acc < self.timeout_s:
            g = self.c.get(f"/api/v1/jobs/{jid}", self.H)
            last = g.json() or {}
            size = len(g.body)
            bad += [f"{last.get('status')}: {e}" for e in tz_errors(last)]
            if last.get("status") in ("done", "failed"):
                break
            time.sleep(1.0)
        el = time.perf_counter() - t_acc
        st = last.get("status")
        self.add("опросы GET /jobs/{id}: поля ТЗ на каждом статусе", not bad, "; ".join(sorted(set(bad)))[:300]
                 or "ok")
        self.add(f"задача завершилась (done|failed) за ≤ {self.limit_s:g} с от приёма",
                 st in ("done", "failed") and el <= self.limit_s, f"status={st} за {el:.1f}s", el,
                 status=st, verdict=last.get("verdict"), n_violations=len(last.get("violations") or []))
        if st == "failed":
            self.add("анализ синтетического видео выполнен (done)", False,
                     f"failed: {str(last.get('error'))[:200]}", warn=True)
        self.add(f"ответ ≤ {self.max_resp // (1024 * 1024)} МБ", size <= self.max_resp, f"{size} байт")
        self.first_status = st
        return jid

    def idempotency(self, mov: bytes, jid: str, first_status: Optional[str]) -> None:
        r = self.c.post_file("/api/v1/analyze", mov, "same-again.MOV", self.H)
        d = r.json() or {}
        same = d.get("job_id") == jid
        if first_status == "failed":  # упавшие задачи не переиспользуются — нужен новый прогон
            self.add("повтор после failed -> новый job (failed не кэшируется)", not same,
                     "новый job" if not same else "вернул упавший job")
            return
        self.add("повторная загрузка того же файла -> тот же job", same,
                 "deduplicated" if same else "новый job (SENTINEL_DEDUP=0?)", warn=True)

    def ranges(self, jid: str) -> None:
        base = f"/api/v1/jobs/{jid}/video"
        r = self.c.get(base, {**self.H, "Range": "bytes=0-99"})
        self.add("Range bytes=0-99 -> 206", r.status == 206 and len(r.body) == 100 and
                 r.headers.get("content-range", "").startswith("bytes 0-99/"), f"HTTP {r.status}")
        size = int(r.headers.get("content-range", "/0").split("/")[-1] or 0)
        r = self.c.get(base, {**self.H, "Range": "bytes=-10"})
        self.add("Range bytes=-10 -> 206", r.status == 206 and len(r.body) == 10, f"HTTP {r.status}")
        r = self.c.get(base, {**self.H, "Range": f"bytes={size}-"})
        self.add("Range за концом файла -> 416", r.status == 416, f"HTTP {r.status}")
        r = self.c.get(base, use_cookies=False)
        self.add("видео без ключа -> 401", r.status == 401, f"HTTP {r.status}")

    def ui(self, jid: Optional[str], mov: Optional[bytes]) -> None:
        self.c.cookies.clear()
        r = self.c.get("/ui/v1/jobs", {"Sec-Fetch-Site": "same-origin", "Origin": self.url})
        self.add("UI без сессии (даже same-origin) -> 401", r.status == 401, f"HTTP {r.status}")
        r = self.c.post_json("/ui/v1/session", {"api_key": "wrong"})
        self.add("UI вход с неверным секретом -> 401", r.status == 401, f"HTTP {r.status}")
        r = self.c.post_json("/ui/v1/session", {"api_key": self.key})
        sc = r.headers.get("set-cookie-all", "").lower()
        csrf = (r.json() or {}).get("csrf_token")
        self.add("UI вход по ключу -> HttpOnly SameSite cookie, токен не в теле",
                 r.status == 200 and "httponly" in sc and "samesite=strict" in sc and
                 bool(csrf) and self.c.cookies.get("ks_session", "x") not in r.body.decode("utf-8", "replace"),
                 f"HTTP {r.status}")
        if jid:
            v = self.c.get(f"/ui/v1/jobs/{jid}/video", {"Range": "bytes=0-9"})
            self.add("UI видео по cookie с Range -> 206 (<video src>)", v.status == 206, f"HTTP {v.status}")
        if mov:
            p = self.c.post_file("/ui/v1/analyze", mov, "a.mov")
            self.add("UI POST без X-CSRF-Token -> 403", p.status == 403, f"HTTP {p.status}")
        r = self.c.get("/healthz", {"Origin": "https://evil.example"})
        self.add("CORS не открыт для чужого Origin", "access-control-allow-origin" not in r.headers,
                 r.headers.get("access-control-allow-origin", "нет заголовка"))
        self.c.request("DELETE", "/ui/v1/session")
        self.c.cookies.clear()

    # ---------------------------------------------------------------- прогон
    def run(self, video: Optional[Path]) -> dict[str, Any]:
        tmp = Path(tempfile.mkdtemp(prefix="sentinel_accept_"))
        try:
            src = video or synth_mov(tmp / "synthetic.mov")
            mov = src.read_bytes() if src else None
            self._try("health", self.health)
            self._try("auth", lambda: self.auth(mov or b"\x00" * 64))
            self._try("bad inputs", lambda: self.bad_inputs(mov))
            jid = None
            if mov:
                self._try("interrupted upload", lambda: self.interrupted_upload(mov))
                name = src.name if src else "synthetic.mov"
                jid = self.lifecycle(mov, name)
                if jid:
                    self._try("idempotency", lambda: self.idempotency(mov, jid,
                                                                      getattr(self, "first_status", None)))
                    self._try("ranges", lambda: self.ranges(jid))
            else:
                self.add("жизненный цикл", None, "нет cv2 и не задан --video")
            self._try("ui", lambda: self.ui(jid, mov))
        finally:
            for p in tmp.glob("*"):
                p.unlink(missing_ok=True)
            tmp.rmdir()
        n = {s: sum(c.status == s for c in self.checks) for s in ("PASS", "WARN", "FAIL", "SKIP")}
        return {"summary": {**n, "ok": n["FAIL"] == 0}, "checks": [asdict(c) for c in self.checks]}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backend.api.tools.acceptance",
                                 description="Приёмка Kassa Sentinel API (ключ — из env SENTINEL_API_KEY)")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--video", type=Path, default=None, help="свой MOV вместо синтетического")
    ap.add_argument("--timeout", type=float, default=330.0)
    ap.add_argument("--limit", type=float, default=300.0, help="лимит от приёма до результата, с")
    ap.add_argument("--out", type=Path, default=None, help="JSON-отчёт (только статусы и тайминги)")
    a = ap.parse_args(argv)
    key = api_key_from_env()
    if not key:
        print("нужен ключ: export SENTINEL_API_KEY=... (в argv ключ не передаём)", file=sys.stderr)
        return 2
    res = Acceptance(a.url, key, a.timeout, a.limit).run(a.video)
    s = res["summary"]
    print(f"ИТОГО: PASS {s['PASS']} / WARN {s['WARN']} / FAIL {s['FAIL']} / SKIP {s['SKIP']}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if s["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
