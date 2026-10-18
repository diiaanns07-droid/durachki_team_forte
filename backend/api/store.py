"""SQLite (WAL) хранилище задач. Один коннект + RLock: нагрузка — десятки задач, не тысячи."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

ACTIVE = ("queued", "running")
FINAL = ("done", "failed")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    status       TEXT NOT NULL,          -- queued | running | done | failed
    stage        TEXT,
    progress     REAL NOT NULL DEFAULT 0,
    created_at   REAL NOT NULL,          -- момент загрузки
    queued_at    REAL NOT NULL,          -- база отсчёта бюджета (сбрасывается при рестарте)
    started_at   REAL,
    finished_at  REAL,
    error        TEXT,
    result_path  TEXT,
    video_path   TEXT,
    filename     TEXT,
    size_bytes   INTEGER,
    sha256       TEXT,
    attempts     INTEGER NOT NULL DEFAULT 0,
    deadline_s   REAL,
    client       TEXT,
    warnings     TEXT,                   -- JSON list
    video_meta   TEXT,                   -- JSON: длительность/fps/размер от декодера
    dedup_key    TEXT                    -- sha256:версия пайплайна (идемпотентность)
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS jobs_created ON jobs(created_at);
"""
_MIGRATIONS = {"video_meta": "TEXT", "dedup_key": "TEXT"}  # колонки, добавленные после v1

_COLUMNS = ("id", "status", "stage", "progress", "created_at", "queued_at", "started_at",
            "finished_at", "error", "result_path", "video_path", "filename", "size_bytes",
            "sha256", "attempts", "deadline_s", "client", "warnings", "video_meta", "dedup_key")


class JobStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False,
                                     isolation_level=None, timeout=30)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.executescript(_SCHEMA)
            have = {r[1] for r in self._conn.execute("PRAGMA table_info(jobs)").fetchall()}
            for col, typ in _MIGRATIONS.items():
                if col not in have:  # БД от прошлой версии API
                    self._conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {typ}")
            self._conn.execute("CREATE INDEX IF NOT EXISTS jobs_dedup ON jobs(dedup_key)")

    # ---- базовые операции: execute И fetch под одним локом (коннект общий для потоков) ----
    def _run(self, sql: str, args: Iterable[Any] = ()) -> int:
        with self._lock:
            return self._conn.execute(sql, tuple(args)).rowcount

    def _one(self, sql: str, args: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(args)).fetchone()

    def _all(self, sql: str, args: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(args)).fetchall()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    @staticmethod
    def _row(r: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
        if r is None:
            return None
        d = dict(r)
        for k, empty in (("warnings", "[]"), ("video_meta", "{}")):
            try:
                d[k] = json.loads(d.get(k) or empty)
            except ValueError:
                d[k] = json.loads(empty)
        return d

    def create(self, job_id: str, *, video_path: str, filename: str, size_bytes: int,
               sha256: str, client: str = "", video_meta: Optional[dict[str, Any]] = None,
               dedup_key: Optional[str] = None, now: Optional[float] = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        self._run(
            "INSERT INTO jobs (id, status, stage, progress, created_at, queued_at, video_path, "
            "filename, size_bytes, sha256, client, warnings, video_meta, dedup_key) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (job_id, "queued", "queued", 0.0, now, now, video_path, filename, size_bytes, sha256,
             client, "[]", json.dumps(video_meta or {}, ensure_ascii=False), dedup_key))
        return self.get(job_id) or {}

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        return self._row(self._one("SELECT * FROM jobs WHERE id=?", (job_id,)))

    def update(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        bad = set(fields) - set(_COLUMNS)
        if bad:
            raise KeyError(f"unknown columns: {bad}")
        if "warnings" in fields and not isinstance(fields["warnings"], str):
            fields["warnings"] = json.dumps(list(fields["warnings"]), ensure_ascii=False)
        if "video_meta" in fields and not isinstance(fields["video_meta"], str):
            fields["video_meta"] = json.dumps(fields["video_meta"], ensure_ascii=False)
        cols = ", ".join(f"{k}=?" for k in fields)
        self._run(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))

    def mark_running(self, job_id: str, now: Optional[float] = None) -> bool:
        """queued -> running (атомарно). False, если задачу уже финализировали."""
        now = time.time() if now is None else now
        return self._run(
            "UPDATE jobs SET status='running', started_at=?, attempts=attempts+1, stage=? "
            "WHERE id=? AND status='queued'", (now, "starting", job_id)) == 1

    def finalize(self, job_id: str, status: str, *, result_path: Optional[str] = None,
                 error: Optional[str] = None, warnings: Optional[list[str]] = None,
                 now: Optional[float] = None) -> bool:
        """Атомарный переход active -> done/failed. Выигрывает первый (воркер или watchdog)."""
        assert status in FINAL
        now = time.time() if now is None else now
        return self._run(
            "UPDATE jobs SET status=?, finished_at=?, result_path=?, error=?, warnings=?, "
            "progress=CASE WHEN ?='done' THEN 1.0 ELSE progress END, stage=? "
            "WHERE id=? AND status IN ('queued','running')",
            (status, now, result_path, error, json.dumps(warnings or [], ensure_ascii=False),
             status, status, job_id)) == 1

    def requeue(self, job_id: str, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        self._run("UPDATE jobs SET status='queued', stage='requeued', progress=0, queued_at=?, "
                   "started_at=NULL WHERE id=?", (now, job_id))

    def active(self) -> list[dict[str, Any]]:
        rows = self._all("SELECT * FROM jobs WHERE status IN ('queued','running') "
                         "ORDER BY queued_at, created_at")
        return [self._row(r) for r in rows if r is not None]  # type: ignore[misc]

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._all("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (int(limit),))
        return [self._row(r) for r in rows if r is not None]  # type: ignore[misc]

    def finished_before(self, ts: float) -> list[dict[str, Any]]:
        rows = self._all("SELECT * FROM jobs WHERE status IN ('done','failed') AND "
                         "COALESCE(finished_at, created_at) < ?", (ts,))
        return [self._row(r) for r in rows if r is not None]  # type: ignore[misc]

    def find_by_dedup(self, key: str) -> Optional[dict[str, Any]]:
        """Последняя не упавшая задача с тем же файлом и версией пайплайна."""
        r = self._one("SELECT * FROM jobs WHERE dedup_key=? AND status IN ('queued','running','done') "
                      "ORDER BY created_at DESC LIMIT 1", (key,))
        return self._row(r)

    def find_done_by_sha(self, sha256: str) -> Optional[dict[str, Any]]:
        r = self._one("SELECT * FROM jobs WHERE sha256=? AND status='done' "
                      "ORDER BY finished_at DESC LIMIT 1", (sha256,))
        return self._row(r)

    def delete(self, job_id: str) -> None:
        self._run("DELETE FROM jobs WHERE id=?", (job_id,))

    def count_active(self) -> int:
        r = self._one("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')")
        return int(r[0]) if r else 0

    def ids(self) -> set[str]:
        return {r[0] for r in self._all("SELECT id FROM jobs")}
