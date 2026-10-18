"""Очередь задач: FIFO, воркеры-потоки, GPU-семафор, бюджет времени, watchdog, рестарт.

Честность результата: done — только когда pipeline.run вернул результат и он прошёл политику исхода
(results.normalize). Нет пайплайна, mock в проде, исключение, таймаут -> failed с причиной.
partial.json становится done только при partial_accept=true И достаточной полноте (meta.coverage /
meta.completeness); иначе он — лишь подробности для UI."""
from __future__ import annotations

import base64
import dataclasses
import importlib
import inspect
import json
import logging
import math
import os
import shutil
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .config import Settings
from .results import (adapt_stage_coverage, analysis_complete, coverage_problem, coverage_ratio,
                      enforce_evidence, enforce_size, fill_missing_evidence, iter_frames, json_size,
                      normalize, to_api_dict)
from .store import JobStore

log = logging.getLogger("sentinel.api.worker")

ProgressFn = Callable[[str, float], None]
RunFn = Callable[..., Any]
_TRUE = {"1", "true", "yes", "on", "y"}


class JobCancelled(BaseException):
    """Бросается из progress(), если watchdog уже закрыл задачу: run() должен свернуться.
    BaseException (как asyncio.CancelledError), чтобы не глоталось `except Exception`."""


@dataclass
class PipelineSpec:
    run: Optional[RunFn] = None
    warmup: Optional[Callable[[], None]] = None
    mode: str = "real"            # real | fake | unavailable
    error: str = ""
    manages_gpu: bool = False     # pipeline сам держит GPU-лок -> API не оборачивает run()
    accepts_deadline_at: bool = False
    version: str = ""
    cancel: Optional[Callable[[str], Any]] = None      # A03: cancel(job_id) — остановить и освободить GPU
    release_gpu: Optional[Callable[[], Any]] = None    # A03: release_gpu() — при остановке сервиса
    warmup_report: Optional[Callable[[], Any]] = None  # отчёт прогрева, если warmup() его не вернул


def _accepts_kw(fn: Callable[..., Any], name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind == p.VAR_KEYWORD for p in params.values())


def resolve_pipeline(settings: Settings) -> PipelineSpec:
    """Ленивый импорт backend.pipeline. Фейк — только при явном SENTINEL_FAKE_PIPELINE=1."""
    if settings.fake_pipeline:
        from .fake_pipeline import make_fake_runner
        return PipelineSpec(run=make_fake_runner(settings.fake_pipeline_s), mode="fake",
                            version="fake")
    mod_name, _, fn = settings.pipeline.partition(":")
    try:
        mod = importlib.import_module(mod_name)
        run = getattr(mod, fn or "run")
        if not callable(run):
            raise TypeError(f"{settings.pipeline} is not callable")
    except Exception as e:
        return PipelineSpec(mode="unavailable", error=f"{type(e).__name__}: {e}"[:400])
    warm = getattr(mod, "warmup", None) if not fn else None
    manages = any(hasattr(mod, a) for a in ("GPU_LOCK", "GPU_SEMAPHORE", "gpu_lock"))
    version = str(getattr(mod, "PIPELINE_VERSION", None) or getattr(mod, "__version__", None) or "")
    cancel = getattr(mod, "cancel", None) if not fn else None
    release = getattr(mod, "release_gpu", None) if not fn else None
    return PipelineSpec(run=run, warmup=warm if callable(warm) else None, mode="real",
                        manages_gpu=manages, accepts_deadline_at=_accepts_kw(run, "deadline_at"),
                        version=version, cancel=cancel if callable(cancel) else None,
                        release_gpu=release if callable(release) else None,
                        warmup_report=None if fn else _report_reader(mod))


def _report_reader(mod: Any) -> Callable[[], Any]:
    """Отчёт прогрева пайплайна, если warmup() его не вернул: ``mod.warmup_report`` (функция или
    dict) либо ``ModelRegistry.instance().warmup_report`` (так хранит отчёт A03 на 2055fc4)."""
    def read() -> Any:
        rep = getattr(mod, "warmup_report", None)
        if callable(rep):
            return rep()
        if rep is not None:
            return rep
        try:
            reg = getattr(mod, "ModelRegistry")
            return getattr(reg.instance(), "warmup_report", None)
        except Exception:
            return None
    return read


def warmup_problem(report: Any) -> Optional[str]:
    """None — прогрев согласованно успешен (или отчёта нет); иначе причина для /readyz 503.
    Понимает: False; {"ok"/"ready": False}; {"status": failed|error}; {"error"|"errors"|
    "hooks_error"|"synthetic_error": ...}; {"synthetic": {"status": != done}}."""
    if report is False:
        return "warmup() returned False"
    if not isinstance(report, dict):
        return None
    for k in ("ok", "ready"):
        if report.get(k) is False:
            return f"warmup report: {k}=false" + (f" ({report.get('error')})" if report.get("error") else "")
    if str(report.get("status", "")).lower() in ("failed", "error"):
        return f"warmup report: status={report.get('status')}" + (
            f" ({report.get('error')})" if report.get("error") else "")
    for k in ("error", "hooks_error", "synthetic_error"):
        if report.get(k):
            return f"warmup {k}: {str(report[k])[:300]}"
    errs = report.get("errors")
    if isinstance(errs, (list, tuple, dict)) and errs:
        return f"warmup errors: {str(errs)[:300]}"
    syn = report.get("synthetic")
    if isinstance(syn, dict) and syn.get("status") not in (None, "done"):
        return f"warmup synthetic run: status={syn.get('status')}"
    return None


def mock_env_active() -> bool:
    return os.environ.get("SENTINEL_MOCK", "").strip().lower() in _TRUE


def result_is_mock(meta: Any) -> bool:
    """Признаки mock-результата (A03 meta.mock, A01 meta.provenance.mock, origin стадий/моделей)."""
    if not isinstance(meta, dict):
        return False
    if meta.get("mock") is True or meta.get("fake_pipeline") is True:
        return True
    prov = meta.get("provenance")
    if isinstance(prov, dict) and prov.get("mock") is True:
        return True
    for key in ("stages", "models"):
        block = meta.get(key)
        items = block.values() if isinstance(block, dict) else (block if isinstance(block, list) else [])
        for it in items:
            if isinstance(it, dict) and ("mock" in str(it.get("origin", "")).lower()
                                         or it.get("status") == "mock"):
                return True
    return False


class GpuGate:
    """Семафор с владельцами. Один владелец GPU: слот зависшей задачи не отдаётся другой
    (zombie_release_gpu=false), пока её run() не вернётся."""

    def __init__(self, slots: int):
        self.slots = max(1, int(slots))
        self._cv = threading.Condition()
        self._holders: set[str] = set()

    def acquire(self, owner: str, timeout: float) -> bool:
        end = time.monotonic() + max(0.0, timeout)
        with self._cv:
            while len(self._holders) >= self.slots:
                rem = end - time.monotonic()
                if rem <= 0:
                    return False
                self._cv.wait(rem)
            self._holders.add(owner)
            return True

    def release(self, owner: str) -> None:
        with self._cv:
            if owner in self._holders:
                self._holders.discard(owner)
                self._cv.notify_all()

    def holds(self, owner: str) -> bool:
        with self._cv:
            return owner in self._holders

    @property
    def busy(self) -> int:
        with self._cv:
            return len(self._holders)


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    os.replace(tmp, path)


def gpu_memory() -> Optional[dict[str, float]]:
    """VRAM процесса API — только если torch уже загружен пайплайном (сами torch не импортируем)."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return None
        mb = 1024 * 1024
        out = {"allocated_mb": round(torch.cuda.memory_allocated() / mb, 1),
               "reserved_mb": round(torch.cuda.memory_reserved() / mb, 1),
               "max_allocated_mb": round(torch.cuda.max_memory_allocated() / mb, 1)}
        try:
            free, total = torch.cuda.mem_get_info()
            out.update(free_mb=round(free / mb, 1), total_mb=round(total / mb, 1))
        except Exception:
            pass
        return out
    except Exception:
        return None


def _reset_gpu_peak() -> None:
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


class JobManager:
    def __init__(self, settings: Settings, store: JobStore,
                 pipeline: Optional[PipelineSpec | Callable[[], PipelineSpec]] = None):
        self.s = settings
        self.store = store
        self._pipeline_src = pipeline
        self.pipeline: Optional[PipelineSpec] = None
        # loading | warming | ready | unavailable | warmup_failed | mock_not_allowed
        self.state = "loading"
        self.state_detail = ""
        self._resolved = threading.Event()
        self._warm = threading.Event()
        self._stop = threading.Event()
        self._cv = threading.Condition()
        self._q: deque[str] = deque()
        self._queued_at: dict[str, float] = {}
        self._lock = threading.RLock()
        self._live: dict[str, dict[str, Any]] = {}
        self._running: dict[str, dict[str, Any]] = {}
        self._workers: set[threading.Thread] = set()
        self._zombies: dict[threading.Thread, dict[str, Any]] = {}
        self._threads: list[threading.Thread] = []
        self._seq = 0
        self.gpu = GpuGate(settings.gpu_slots)
        self.completed = 0

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.s.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._recover()
        self._spawn(self._init_pipeline, "sentinel-warmup")
        for _ in range(self.s.workers):
            self._spawn_worker()
        self._spawn(self._watchdog_loop, "sentinel-watchdog")
        if self.s.job_ttl_h > 0:
            self._spawn(self._cleanup_loop, "sentinel-cleanup")

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        spec = self.pipeline
        if spec is not None and spec.release_gpu is not None:  # не оставлять процесс-воркер A03
            t = threading.Thread(target=self._safe_call, args=(spec.release_gpu, "release_gpu"),
                                 name="sentinel-release-gpu", daemon=True)
            t.start()
            t.join(5.0)
        end = time.monotonic() + timeout
        for t in list(self._threads):
            t.join(max(0.0, end - time.monotonic()))

    def _spawn(self, target: Callable[[], None], name: str) -> threading.Thread:
        t = threading.Thread(target=target, name=name, daemon=True)
        self._threads.append(t)
        t.start()
        return t

    def _spawn_worker(self) -> None:
        with self._lock:
            self._seq += 1
            name = f"sentinel-worker-{self._seq}"
        t = threading.Thread(target=self._worker_loop, name=name, daemon=True)
        with self._lock:
            self._workers.add(t)
        self._threads.append(t)
        t.start()

    # ------------------------------------------------------------ pipeline/warmup
    def _init_pipeline(self) -> None:
        src = self._pipeline_src
        try:
            spec = src() if callable(src) else (src or resolve_pipeline(self.s))
        except Exception as e:
            spec = PipelineSpec(mode="unavailable", error=f"{type(e).__name__}: {e}"[:400])
        self.pipeline = spec
        self._resolved.set()
        if spec.run is None:
            self.state, self.state_detail = "unavailable", spec.error
            log.error("analysis pipeline unavailable: jobs will fail", extra={"error": spec.error})
            self._warm.set()
            return
        if spec.mode == "fake":
            log.warning("FAKE PIPELINE ENABLED (SENTINEL_FAKE_PIPELINE=1): results are synthetic")
        elif mock_env_active() and not self.s.allow_mock:
            self.state = "mock_not_allowed"
            self.state_detail = ("SENTINEL_MOCK=1 in production: mock models would produce synthetic "
                                 "results; set SENTINEL_ALLOW_MOCK=1 only for dev/tests")
            log.error("mock mode refused", extra={"error": self.state_detail})
            self._warm.set()
            return
        if spec.warmup and self.s.warmup:
            self.state = "warming"
            t0 = time.monotonic()
            try:
                ret = spec.warmup()
                report = ret if isinstance(ret, (dict, bool)) else (
                    spec.warmup_report() if spec.warmup_report else None)
                problem = warmup_problem(report)
            except Exception as e:
                problem = f"{type(e).__name__}: {e}"[:400]
            if problem:  # прогрев не удался -> /readyz 503, а не «готов» с незагруженными моделями
                self.state, self.state_detail = "warmup_failed", problem[:400]
                log.error("pipeline warmup failed", extra={"error": self.state_detail})
            else:
                self.state = "ready"
                log.info("pipeline warmed up", extra={"seconds": round(time.monotonic() - t0, 2)})
        else:
            self.state = "ready"
        self._warm.set()

    @property
    def ready(self) -> bool:
        return self.state == "ready" and not self.gpu_hung()

    def gpu_hung(self, now: Optional[float] = None) -> Optional[float]:
        """Секунды, сколько зависший run() держит GPU сверх hung_gpu_s (None — не держит)."""
        now = time.time() if now is None else now
        with self._lock:
            worst = None
            for z in self._zombies.values():
                if self.gpu.holds(z["job_id"]):
                    held = now - z["since"]
                    if held > self.s.hung_gpu_s:
                        worst = max(worst or 0.0, held)
            return worst

    def pipeline_tag(self) -> str:
        """Ключ идемпотентности: та же версия пайплайна -> тот же результат."""
        spec = self.pipeline
        if spec is None:
            return "loading"
        return f"{spec.mode}:{spec.version or 'unversioned'}"

    # ------------------------------------------------------------ queue
    def submit(self, job_id: str, queued_at: Optional[float] = None) -> None:
        with self._cv:
            self._q.append(job_id)
            self._queued_at[job_id] = time.time() if queued_at is None else queued_at
            self._cv.notify()

    def _take(self, timeout: float) -> Optional[str]:
        with self._cv:
            if not self._q:
                self._cv.wait(timeout)
            if not self._q:
                return None
            return self._q.popleft()

    def queue_position(self, job_id: str) -> Optional[int]:
        with self._cv:
            try:
                return list(self._q).index(job_id) + 1
            except ValueError:
                return None

    def queue_len(self) -> int:
        with self._cv:
            return len(self._q)

    def live(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            v = self._live.get(job_id)
            return dict(v) if v else None

    def stats(self) -> dict[str, Any]:
        with self._lock:
            out: dict[str, Any] = {
                "queued": self.queue_len(), "running": len(self._running),
                "workers": len(self._workers - set(self._zombies)), "zombies": len(self._zombies),
                "gpu_busy": self.gpu.busy, "gpu_slots": self.gpu.slots, "completed": self.completed}
        gm = gpu_memory()
        if gm:
            out["gpu_memory"] = gm
        return out

    # ------------------------------------------------------------ worker
    def _worker_loop(self) -> None:
        me = threading.current_thread()
        try:
            while not self._stop.is_set():
                job_id = self._take(0.5)
                if job_id is None:
                    continue
                try:
                    self._process(job_id)
                except Exception:
                    if self._stop.is_set():  # остановка сервера: хранилище уже закрыто
                        return
                    log.exception("worker crashed on job", extra={"job_id": job_id})
                    try:
                        self._fail(job_id, "worker crashed (see server log)", "worker_crash")
                    except Exception:
                        log.exception("could not mark crashed job failed", extra={"job_id": job_id})
                with self._lock:
                    if me in self._zombies:
                        self._zombies.pop(me, None)
                        if len(self._workers - set(self._zombies)) > self.s.workers:
                            return
        finally:
            with self._lock:
                self._workers.discard(me)
                self._zombies.pop(me, None)

    def job_dir(self, job_id: str) -> Path:
        return self.s.jobs_dir / job_id

    def _is_final(self, job_id: str) -> bool:
        row = self.store.get(job_id)
        return not row or row["status"] in ("done", "failed")

    def _set_live(self, job_id: str, stage: str, progress: float) -> None:
        with self._lock:
            self._live[job_id] = {"stage": stage, "progress": progress, "updated": time.time()}

    def _process(self, job_id: str) -> None:
        with self._cv:
            t0 = self._queued_at.pop(job_id, None)
        row = self.store.get(job_id)
        if not row or row["status"] != "queued":
            return
        t0 = row["queued_at"] if t0 is None else t0
        if not self.store.mark_running(job_id):
            return
        self._set_live(job_id, "starting", 0.0)

        if not self._warm.is_set():
            self._set_live(job_id, "waiting_warmup", 0.0)
            self.store.update(job_id, stage="waiting_warmup")
            self._warm.wait(max(0.0, min(self.s.warmup_wait_s, t0 + self.s.watchdog_s - time.time())))
        spec = self.pipeline
        if spec is None or spec.run is None:
            err = spec.error if spec else "pipeline not initialised (warmup timeout)"
            self._fail(job_id, f"analysis pipeline unavailable: {err}", "pipeline_unavailable")
            return
        if self.state == "mock_not_allowed":
            self._fail(job_id, self.state_detail, "mock_not_allowed")
            return

        gated = self.s.gpu_lock_scope == "job" and not spec.manages_gpu
        if gated:
            self._set_live(job_id, "waiting_gpu", 0.0)
            while not self.gpu.acquire(job_id, timeout=0.5):
                if self._stop.is_set() or self._is_final(job_id):
                    return
        try:
            if self._is_final(job_id):
                return
            deadline = self._deadline_for(t0)
            started = time.time()
            self.store.update(job_id, deadline_s=deadline, stage="running")
            self._set_live(job_id, "running", 0.0)
            with self._lock:
                self._running[job_id] = {"thread": threading.current_thread(),
                                         "started": started, "deadline": deadline}
            kwargs: dict[str, Any] = {}
            if self.s.pass_deadline_at and spec.accepts_deadline_at:
                kwargs["deadline_at"] = t0 + self.s.budget_s  # бюджет от приёма файла (A03)
            _reset_gpu_peak()
            try:
                result = spec.run(row["video_path"], job_id, self._progress_cb(job_id), deadline,
                                  **kwargs)
            except JobCancelled:
                return
            except BaseException as e:  # noqa: BLE001 — в т.ч. SystemExit из библиотек в потоке
                log.error("pipeline.run raised", extra={"job_id": job_id, "error": repr(e)[:300],
                                                         "trace": traceback.format_exc(limit=8)})
                self._fail(job_id, f"analysis error: {type(e).__name__}: {e}"[:500], "pipeline_error")
                return
            self._complete(job_id, result, spec, t0=t0, started=started, deadline=deadline)
        finally:
            if gated:
                self.gpu.release(job_id)
            with self._lock:
                self._running.pop(job_id, None)
                self._live.pop(job_id, None)

    def _deadline_for(self, t0: float) -> float:
        """deadline_s = budget − ожидание (очередь включена), не больше job_cap_s и честной доли
        для тех, кто ждёт за нами (оценка, а не обещание: пропускную способность меряет L2)."""
        s, now = self.s, time.time()
        d = min(s.budget_s - (now - t0), s.job_cap_s)
        if s.fair_share:
            par = min(s.workers, s.gpu_slots) if s.gpu_lock_scope == "job" else s.workers
            par = max(1, par)
            with self._cv:
                waiting = [self._queued_at.get(j, now) for j in self._q]
            for k, qt in enumerate(waiting, start=1):
                d = min(d, (s.budget_s - (now - qt)) / (math.ceil(k / par) + 1))
        return round(max(s.min_deadline_s, d), 1)

    def _progress_cb(self, job_id: str) -> ProgressFn:
        last = {"t": 0.0, "stage": None}

        def progress(stage: str, fraction: float = 0.0) -> None:
            with self._lock:
                cancelled = job_id not in self._running
            if cancelled and self._is_final(job_id):
                raise JobCancelled(job_id)
            try:
                f = float(fraction)
                f = 0.0 if not math.isfinite(f) else min(1.0, max(0.0, f))
            except (TypeError, ValueError):
                f = 0.0
            st = str(stage)[:64]
            self._set_live(job_id, st, f)
            now = time.monotonic()
            if st != last["stage"] or now - last["t"] >= 1.0:
                last.update(t=now, stage=st)
                try:
                    self.store.update(job_id, stage=st, progress=f)
                except Exception:
                    pass
        return progress

    # ------------------------------------------------------------ finalize
    def _complete(self, job_id: str, result: Any, spec: PipelineSpec, *, t0: float,
                  started: float, deadline: float) -> None:
        warns: list[str] = []
        if not (isinstance(result, dict) or dataclasses.is_dataclass(result)
                or hasattr(result, "violations")):
            self._fail(job_id, f"pipeline returned unusable result: {type(result).__name__}",
                       "bad_result")
            return
        if self.s.coverage_adapt_stages:
            rmeta = result.get("meta") if isinstance(result, dict) else getattr(result, "meta", None)
            note = adapt_stage_coverage(rmeta)
            if note:
                warns.append(note)
        try:
            api, w = to_api_dict(result)
            warns += w
        except Exception as e:
            self._fail(job_id, f"pipeline returned unusable result: {type(e).__name__}: {e}"[:500],
                       "bad_result")
            return
        if spec.mode != "fake" and not self.s.allow_mock and result_is_mock(api.get("meta")):
            self._fail(job_id, "pipeline returned a MOCK result (meta.mock) — refused in production",
                       "mock_result", details=api)
            return
        api, w = normalize(api, job_id, min_coverage=self.s.min_coverage, allow_mock=self.s.allow_mock)
        warns += w
        if api["status"] == "failed":
            self._fail(job_id, str(api.get("error") or "analysis failed"), "pipeline_failed",
                       details=api)
            return
        self._store_done(job_id, api, warns, spec, t0=t0, started=started, deadline=deadline)

    def _store_done(self, job_id: str, api: dict[str, Any], warns: list[str], spec: PipelineSpec,
                    *, t0: float, started: Optional[float], deadline: Optional[float],
                    partial: bool = False) -> bool:
        row = self.store.get(job_id) or {}
        if self.s.evidence_fallback and row.get("video_path"):
            warns += fill_missing_evidence(api, Path(row["video_path"]),
                                           tolerance_s=self.s.evidence_max_offset_s,
                                           time_budget_s=self.s.evidence_budget_s)
        ew, lost = enforce_evidence(api)  # нарушение без кадра -> candidates; ни одного -> failed
        warns += ew
        if lost:
            self._fail(job_id, lost, "evidence_missing", details=api)
            return False
        now = time.time()
        meta = api["meta"]
        mw = meta.get("warnings")
        meta["warnings"] = (list(mw) if isinstance(mw, list) else ([str(mw)] if mw else [])) + warns
        if spec.mode == "fake":
            meta["fake_pipeline"] = True
        if partial:
            meta["partial"] = True
        frames_meta = {
            "pipeline": spec.mode, "pipeline_version": spec.version or None,
            "queue_wait_s": round((started or now) - t0, 2),
            "processing_s": round(now - started, 2) if started else None,
            "total_s": round(now - t0, 2), "deadline_s": deadline,
            "deadline_at_passed": bool(self.s.pass_deadline_at and spec.accepts_deadline_at),
            "filename": row.get("filename"), "video": row.get("video_meta") or {},
            "coverage": coverage_ratio(meta), "analysis_complete": analysis_complete(
                meta, self.s.min_coverage, not partial, self.s.allow_mock),
            "partial": partial, "api_version": "2",
        }
        gm = gpu_memory()
        if gm:
            frames_meta["gpu_peak_mb"] = gm.get("max_allocated_mb")
        meta["api"] = frames_meta
        meta["warnings"] += enforce_size(api, self.s.max_response_bytes)
        frames_meta["frames"] = [{"i": i, "violation": vi, "timestamp_sec": f.get("timestamp_sec"),
                                  "path": f"frames/{i}.jpg"} for i, vi, f in iter_frames(api)]
        frames_meta["response_bytes"] = json_size(api)
        jd = self.job_dir(job_id)
        path = jd / f"result-{int(now * 1000)}-{threading.get_ident()}.json"
        try:
            write_json(path, api)
        except (OSError, ValueError) as e:
            self._fail(job_id, f"could not store result: {type(e).__name__}", "store_error")
            return False
        if self.store.finalize(job_id, "done", result_path=str(path), warnings=meta["warnings"]):
            with self._lock:
                self.completed += 1
            if spec.mode == "real" and self.state == "warmup_failed":
                self.state = "ready"  # рабочий прогон доказал готовность
            self._dump_frames(jd, api)
            log.info("job done", extra={"job_id": job_id, "verdict": api.get("verdict"),
                                         "violations": len(api.get("violations") or []),
                                         "total_s": frames_meta["total_s"], "partial": partial})
            return True
        try:  # watchdog успел раньше: результат поздний, наружу не меняем
            os.replace(path, jd / "late_result.json")
        except OSError:
            pass
        return False

    def _dump_frames(self, jd: Path, api: dict[str, Any]) -> None:
        try:
            fd = jd / "frames"
            fd.mkdir(exist_ok=True)
            for i, _, f in iter_frames(api):
                (fd / f"{i}.jpg").write_bytes(base64.b64decode(f["image_base64"]))
        except Exception:
            log.warning("could not dump evidence frames", exc_info=True)

    def _read_partial(self, job_id: str) -> Optional[dict[str, Any]]:
        p = self.job_dir(job_id) / "partial.json"
        if not p.is_file():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _try_partial(self, job_id: str, partial: dict[str, Any]) -> tuple[bool, str]:
        """partial.json -> done, только если включено и полнота достаточна. (принят?, причина)."""
        meta = partial.get("meta") if isinstance(partial.get("meta"), dict) else {}
        if self.s.coverage_adapt_stages:
            adapt_stage_coverage(meta)
        cov = coverage_ratio(meta)
        problem = coverage_problem(meta, self.s.min_coverage, allow_mock=self.s.allow_mock,
                                   pipeline_flags=False)
        why = (f"coverage={cov:.2f}" if cov is not None else "coverage not reported") + \
              (f"; {problem}" if problem else "")
        if not self.s.partial_accept:
            return False, f"partial result not accepted (partial_accept=false; {why})"
        if problem is not None:
            return False, f"partial result incomplete ({why}; need >= {self.s.min_coverage})"
        if not self.s.allow_mock and result_is_mock(meta):
            return False, "partial result is MOCK — refused"
        cand = dict(partial)
        cand.pop("status", None)  # A03 пишет processing
        cmeta = dict(meta)
        for k in ("complete", "partial"):  # флаги «не доделано» у partial по определению
            cmeta.pop(k, None)
        cand["meta"] = cmeta
        api, warns = normalize(cand, job_id, min_coverage=self.s.min_coverage,
                               allow_mock=self.s.allow_mock, pipeline_flags=False)
        if api["status"] != "done":
            return False, f"partial result rejected by outcome policy: {api.get('error')}"
        row = self.store.get(job_id) or {}
        spec = self.pipeline or PipelineSpec(mode="unknown")
        warns.append(f"PARTIAL: pipeline did not finish; result accepted by coverage ({why})")
        info = self._running_info(job_id)
        ok = self._store_done(job_id, api, warns, spec, t0=row.get("queued_at") or time.time(),
                              started=info.get("started") if info else row.get("started_at"),
                              deadline=info.get("deadline") if info else row.get("deadline_s"),
                              partial=True)
        return ok, why

    def _running_info(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            info = self._running.get(job_id)
            return dict(info) if info else None

    def _fail(self, job_id: str, error: str, code: str,
              details: Optional[dict[str, Any]] = None) -> bool:
        """failed — наружу без вердикта, с причиной. partial.json и отвергнутый результат —
        подробности только для UI (ui-маршрут, поле partial)."""
        partial = self._read_partial(job_id)
        if partial is not None and code in ("timeout", "pipeline_error", "worker_crash"):
            accepted, why = self._try_partial(job_id, partial)
            if accepted:
                return False  # задача завершена как done (partial) — это не failure
            error = f"{error}; {why}"
        result_path = None
        if partial is not None or details is not None:
            body = {"job_id": job_id, "complete": False, "code": code}
            if partial is not None:
                body["partial"] = partial
            if details is not None:
                body["rejected_result"] = details
            p = self.job_dir(job_id) / "failed_details.json"
            try:
                write_json(p, body)
                result_path = str(p)
            except (OSError, ValueError):
                result_path = None
        ok = self.store.finalize(job_id, "failed", error=f"{code}: {error}"[:600],
                                 result_path=result_path, warnings=[f"{code}: {error}"[:300]])
        if ok:
            log.warning("job failed", extra={"job_id": job_id, "code": code, "error": error[:300],
                                              "has_partial": partial is not None})
        return ok

    # ------------------------------------------------------------ watchdog
    def _watchdog_loop(self) -> None:
        while not self._stop.wait(0.25):
            try:
                self._watchdog_tick()
            except Exception:
                log.exception("watchdog tick failed")

    def _watchdog_tick(self, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        for row in self.store.active():
            jid = row["id"]
            info = self._running_info(jid)
            if info and not info["thread"].is_alive():  # поток умер мимо finally
                self._fail(jid, "worker thread died during analysis", "worker_crash")
                with self._lock:
                    self._running.pop(jid, None)
                self.gpu.release(jid)
                continue
            limit = row["queued_at"] + self.s.watchdog_s
            if info:
                limit = min(limit, info["started"] + info["deadline"] + self.s.deadline_grace_s)
            if now < limit:
                continue
            waited = now - row["queued_at"]
            closed = self._fail(jid, f"analysis did not finish in time ({waited:.0f}s since upload)",
                                "timeout")
            if closed or self._is_final(jid):
                self._abandon(jid, info, now)
        self._ensure_workers()

    def _ensure_workers(self) -> None:
        with self._lock:
            alive = {t for t in self._workers if t.is_alive()}
            self._workers = alive
            missing = self.s.workers - len(alive - set(self._zombies))
        for _ in range(max(0, missing)):
            if len(self._zombies) > self.s.max_zombie_workers:
                break
            log.warning("respawning worker thread")
            self._spawn_worker()

    def _abandon(self, job_id: str, info: Optional[dict[str, Any]], now: float) -> None:
        with self._cv:
            if job_id in self._q:
                self._q.remove(job_id)
            self._queued_at.pop(job_id, None)
        if not info:
            return
        with self._lock:
            self._running.pop(job_id, None)
            th = info.get("thread")
            if th is not None and th.is_alive():
                self._zombies[th] = {"job_id": job_id, "since": now}
        spec = self.pipeline
        if self.s.cancel_on_timeout and spec is not None and spec.cancel is not None:
            # A03 останавливает воркер и выгружает модели: run() вернётся, GPU освободится сам
            self._spawn(lambda: self._safe_call(spec.cancel, "cancel", job_id), f"sentinel-cancel-{job_id[:8]}")
        if self.s.zombie_release_gpu:
            self.gpu.release(job_id)
        else:
            log.warning("pipeline.run still running after timeout; GPU stays with it (single owner)",
                        extra={"job_id": job_id})

    @staticmethod
    def _safe_call(fn: Callable[..., Any], what: str, *args: Any) -> None:
        try:
            r = fn(*args)
            log.info(f"pipeline.{what} done", extra={"result": str(r)[:200]})
        except Exception as e:
            log.warning(f"pipeline.{what} failed", extra={"error": f"{type(e).__name__}: {e}"[:200]})

    # ------------------------------------------------------------ restart / cleanup
    def _recover(self) -> None:
        """Незавершённые задачи после рестарта: дедлайн по-прежнему от приёма файла (300 с),
        опоздавшие -> failed с причиной, остальные — в очередь с остатком бюджета."""
        now = time.time()
        for row in self.store.active():
            jid = row["id"]
            elapsed = now - row["created_at"]
            if row["attempts"] >= self.s.max_attempts:
                self.store.finalize(jid, "failed", error="crashed: server restarted during "
                                    f"processing {row['attempts']} times")
            elif not row.get("video_path") or not Path(row["video_path"]).is_file():
                self.store.finalize(jid, "failed", error="video file missing after restart")
            elif not self.s.restart_fresh_budget and \
                    elapsed > self.s.watchdog_s - self.s.min_deadline_s:
                self.store.finalize(jid, "failed", error=f"timeout: server restarted, deadline passed "
                                    f"({elapsed:.0f}s since upload)")
            else:
                base = now if self.s.restart_fresh_budget else row["created_at"]
                self.store.requeue(jid, base)
                self.submit(jid, base)
                log.info("requeued after restart", extra={"job_id": jid,
                                                           "elapsed_s": round(elapsed, 1)})

    def _cleanup_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.cleanup()
            except Exception:
                log.exception("cleanup failed")
            if self._stop.wait(self.s.cleanup_interval_s):
                return

    def cleanup(self, now: Optional[float] = None) -> int:
        now = time.time() if now is None else now
        ttl = self.s.job_ttl_h * 3600
        if ttl <= 0:
            return 0
        n = 0
        for row in self.store.finished_before(now - ttl):
            shutil.rmtree(self.job_dir(row["id"]), ignore_errors=True)
            self.store.delete(row["id"])
            n += 1
        known = self.store.ids()
        if self.s.jobs_dir.is_dir():
            for d in self.s.jobs_dir.iterdir():
                try:
                    if d.is_dir() and d.name not in known and now - d.stat().st_mtime > max(ttl, 3600):
                        shutil.rmtree(d, ignore_errors=True)
                        n += 1
                except OSError:
                    continue
        if n:
            log.info("cleanup removed jobs", extra={"count": n})
        return n
