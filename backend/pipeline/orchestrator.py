"""Оркестратор Kassa Sentinel: run(video) → AnalysisResult.

Порядок (всё, что трогает модели, — последовательно в одном исполнителе = один владелец GPU):
ingest → detect → track → pose → scene → rules → temporal → [partial] → vlm.verify → vlm.discover → fusion.

Обязательные стадии (config.required_stages) — «заявленный базовый анализ». Их сбой, таймаут, пропуск
по бюджету, отсутствие модуля или выключение НИКОГДА не дают успешный no_violation: результат
failed, либо done/violation с meta.degraded=true (если нарушение найдено другими источниками).
Дополнительные стадии (pose, temporal, VLM) пропускаются/отключаются с явным warning.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from . import serde
from .budget import TimeBudget, choose_sample_fps
from .cache import STAGE_FILES, StageCache, sha256_file, stable_hash
from .config import STAGE_ORDER, cache_root, cfg_get, data_root, load_config
from .contract import AnalysisResult, Event
from .coverage import canonical_coverage, completeness, coverage_gate, reference_duration
from .fallbacks import EvidenceClip, derive_risk, fallback_fuse, states_from_detections
from .media import ascii_safe_path, probe_video
from .registry import ModelRegistry
from .stages import STAGES, catalog_hash, is_required, load_catalog_safe
from .refine import Window, apply_refinement, apply_timecode_policy, build_windows
from .worker import CLIP, CallResult, ClipRef, WorkerCancelled, WorkerCrashed, WorkerTimeout, get_executor

log = logging.getLogger("sentinel.pipeline")

ProgressFn = Callable[[str, float], None]
OK_STATUSES = ("ok", "cached")

_ACTIVE: dict[str, threading.Event] = {}   # job_id → флаг отмены выполняющейся/ждущей задачи
_ACTIVE_LOCK = threading.Lock()

# kind для нормализации выходов (JSON-roundtrip) — совпадает с STAGE_FILES
_KINDS = {stage: [kind for _f, kind in files] for stage, files in STAGE_FILES.items()}


@dataclass
class StageRecord:
    status: str = "pending"          # ok | cached | error | timeout | crashed | missing | skipped | disabled
    seconds: float = 0.0
    origin: str = ""
    required: bool = False
    key: str = ""
    n_out: Optional[int] = None
    error: str = ""
    version: str = ""
    vram_peak_mb: Optional[float] = None
    cold: Optional[bool] = None      # первый вызов в процессе-воркере: время включает загрузку модели

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if v not in (None, "")}
        d["seconds"] = round(self.seconds, 3)
        return d


@dataclass
class RunContext:
    job_id: str
    video_path: str
    cfg: dict
    budget: TimeBudget
    progress_fn: Optional[ProgressFn]
    t_wall0: float = field(default_factory=time.perf_counter)
    warnings: list[str] = field(default_factory=list)
    records: dict[str, StageRecord] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    incomplete: list[str] = field(default_factory=list)
    keys: dict[str, str] = field(default_factory=dict)
    sha: str = ""
    sample_fps: float = 5.0
    device: str = "cpu"
    clip_info: dict = field(default_factory=dict)
    cache: Optional[StageCache] = None
    executor: Any = None
    catalog: dict = field(default_factory=dict)
    catalog_source: str = ""
    evidence_clip: Any = None
    work_path: str = ""
    ingest_identity: dict = field(default_factory=dict)
    probe: dict = field(default_factory=dict)
    identities: dict = field(default_factory=dict)
    _progress_done: float = 0.0
    partial_written: list[str] = field(default_factory=list)
    cancel: threading.Event = field(default_factory=threading.Event)
    queue_wait_s: float = 0.0

    def cancelled(self) -> bool:
        return self.cancel.is_set()

    # ---------------------------------------------------------------- прогресс
    def report(self, label: str, frac: float) -> None:
        frac = max(self._progress_done, min(1.0, float(frac)))
        self._progress_done = frac
        if self.progress_fn is None:
            return
        try:
            self.progress_fn(label, round(frac, 4))
        except Exception as exc:  # noqa: BLE001 — колбэк UI не должен ронять задачу
            log.warning("progress callback упал: %s", exc)

    def stage_progress_base(self, stage: str) -> float:
        total = sum(s.weight for s in STAGES.values())
        done = 0.0
        for name in STAGE_ORDER:
            if name == stage:
                break
            done += STAGES[name].weight
        return done / total

    def rpc_timeout(self, base: float) -> float:
        """Таймаут служебного вызова воркера: не дольше base и не дальше дедлайна (минус минимум на отчёт)."""
        left = self.budget.remaining() - float(cfg_get(self.cfg, "reserve_s.fusion_min", 3))
        return max(0.5, min(float(base), left))

    def out_of_time(self) -> bool:
        return self.budget.remaining() <= float(cfg_get(self.cfg, "reserve_s.essential_min", 1.5))

    def warn(self, msg: str) -> None:
        if msg and msg not in self.warnings:
            self.warnings.append(msg)
            log.warning("[%s] %s", self.job_id, msg)


# ============================================================== публичное API

def run(video_path: str, job_id: str, progress: Optional[ProgressFn] = None, deadline_s: float = 240,
        *, deadline_at: Optional[float] = None, config_path: Optional[str] = None,
        overrides: Optional[Mapping[str, Any]] = None) -> AnalysisResult:
    """Анализ видео. Никогда не бросает исключение: при невозможности анализа — status="failed".

    deadline_s — бюджет от момента вызова. deadline_at (time.time(), необязательный) — абсолютный
    дедлайн задачи (например, момент загрузки + SLA): ожидание в очереди не сбрасывает отсчёт.
    """
    result, _ctx = run_with_context(video_path, job_id, progress, deadline_s, deadline_at=deadline_at,
                                    config_path=config_path, overrides=overrides)
    return result


def run_with_context(video_path: str, job_id: str, progress: Optional[ProgressFn] = None,
                     deadline_s: float = 240, *, deadline_at: Optional[float] = None,
                     config_path: Optional[str] = None, overrides: Optional[Mapping[str, Any]] = None,
                     ) -> tuple[AnalysisResult, Optional[RunContext]]:
    """То же, что run(), но возвращает и контекст (промежуточные выходы для CLI/профилировщика)."""
    job_id = str(job_id or uuid.uuid4().hex[:12])
    try:
        cfg = load_config(config_path, overrides)
    except Exception as exc:  # noqa: BLE001
        return _failed(job_id, f"конфиг пайплайна не читается: {type(exc).__name__}: {exc}", {}), None
    if deadline_at is not None:
        deadline_s = min(float(deadline_s), float(deadline_at) - time.time()) if deadline_s else \
            float(deadline_at) - time.time()
    budget = TimeBudget(float(deadline_s), cfg)
    ctx = RunContext(job_id=job_id, video_path=str(video_path), cfg=cfg, budget=budget, progress_fn=progress)
    with _ACTIVE_LOCK:
        _ACTIVE[job_id] = ctx.cancel
    try:
        result = _run(ctx)
    except Exception as exc:  # noqa: BLE001 — последняя линия обороны: задача не падает исключением
        log.exception("pipeline crashed")
        ctx.warn(f"оркестратор: {type(exc).__name__}: {exc}")
        result = _failed(job_id, f"внутренняя ошибка оркестратора: {type(exc).__name__}: {exc}", _meta(ctx))
    finally:
        with _ACTIVE_LOCK:
            if _ACTIVE.get(job_id) is ctx.cancel:
                _ACTIVE.pop(job_id, None)
    if ctx.cancelled():
        result.status, result.verdict = "failed", None
        result.meta["cancelled"] = True
        result.meta["error"] = "задача отменена" + (f" ({result.meta['error']})" if result.meta.get("error") else "")
        result.meta.setdefault("completeness", {})["complete"] = False
    ctx.report("Готово" if result.status == "done" else ("Отменено" if ctx.cancelled() else "Ошибка"), 1.0)
    return result, ctx


def cancel(job_id: str) -> bool:
    """Отменить задачу (выполняющуюся или ждущую воркер). Текущий вызов модели прерывается
    остановкой процесса-воркера → модели выгружены, VRAM освобождена. True — задача найдена."""
    with _ACTIVE_LOCK:
        ev = _ACTIVE.get(str(job_id))
    if ev is None:
        return False
    ev.set()
    return True


def release_gpu() -> dict:
    """Освободить GPU между задачами: process — остановить воркер (модели умирают вместе с процессом);
    inline — хуки unload() модулей + реестр + сброс клипа. Следующая задача поднимет всё заново."""
    try:
        return get_executor(load_config()).release()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


class WarmupError(RuntimeError):
    """Прогрев не пройден — сервис НЕ готов. ``report`` — подробности (= ModelRegistry.warmup_report)."""

    def __init__(self, message: str, report: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.report = dict(report)


_OK_HOOK = ("ok", "no_hook")
_FAILED_STAGE = ("error", "timeout", "crashed", "missing", "cancelled")


def warmup(deadline_s: Optional[float] = None) -> None:
    """Прогрев при старте сервера. Успех → ``ModelRegistry.warm=True``; иначе ``WarmupError``
    (A02 ловит исключение → ``warmup_failed``, сервис не ``ready``).

    Успех требует ВСЕГО сразу, в пределах общего дедлайна ``warmup.deadline_s`` (включая ожидание
    воркера и каждый RPC):
    1. аренда воркера получена;
    2. хуки ``warmup()`` модулей обязательных стадий прошли (``ok``/нет хука); «нет весов» в хуке,
       отсутствие пакета, падение импорта — провал;
    3. синтетический прогон: ``status=done``, ``meta.complete``, без ``degraded``, каждая обязательная
       стадия ``ok`` из настоящего модуля (``builtin-mock``/запасной путь — только в явном mock-режиме
       или при ``warmup.allow_fallbacks``); ``status=failed`` — провал и без исключения.
    Дополнительные стадии — по явной политике ``warmup.optional_policy``: ``warn`` (по умолчанию —
    готовность с перечнем деградаций в отчёте) или ``fail``; выключенные/невостребованные стадии
    перечисляются в отчёте явно, а не считаются молча успешными.
    """
    cfg = load_config()
    reg = ModelRegistry.instance(cfg)
    reg.warm = False
    reg.warmup_error = None
    total = float(deadline_s if deadline_s is not None else cfg_get(cfg, "warmup.deadline_s", 120))
    t0 = time.monotonic()

    def left() -> float:
        return total - (time.monotonic() - t0)

    mock = bool(cfg.get("mock"))
    with_vlm = bool(cfg_get(cfg, "warmup.vlm", False))
    optional_policy = str(cfg_get(cfg, "warmup.optional_policy", "warn")).lower()
    allow_fallbacks = bool(cfg_get(cfg, "warmup.allow_fallbacks", False)) or mock
    required = [s for s in STAGE_ORDER if is_required(s, cfg)]
    enabled = [s for s in STAGE_ORDER if cfg_get(cfg, f"stages.{s}", True)
               and (not s.startswith("vlm") or with_vlm) and (s != "refine" or cfg_get(cfg, "refine.enabled", False))]
    req_modules = sorted({STAGES[s].module for s in required})
    modules = sorted({STAGES[s].module for s in enabled} | set(req_modules))
    report: dict[str, Any] = {
        "ok": False, "deadline_s": total, "mock": mock, "required_stages": required,
        "policy": {"optional": optional_policy, "allow_fallbacks": allow_fallbacks, "vlm": with_vlm},
        "hooks": {}, "stages": {}, "failures": [], "warnings": [], "seconds": None}
    failures: list[str] = report["failures"]
    warnings: list[str] = report["warnings"]

    # ---- 1-2. аренда воркера + хуки warmup() модулей (загрузка весов)
    try:
        ex = get_executor(cfg)
        lease_s = max(0.0, left())
        with ex.lease(timeout=lease_s) as leased:
            if not leased:
                failures.append(f"воркер занят другой задачей: аренда не получена за {lease_s:.1f} с")
            else:
                hooks = ex.request("warmup", {"modules": modules}, timeout=max(0.1, left()))
                report["hooks"] = hooks
                for name in modules:
                    h = hooks.get(name) or {"status": "missing", "error": "нет ответа"}
                    if h.get("status") in _OK_HOOK:
                        continue
                    msg = f"{name}: {h.get('status')}" + (f": {h.get('error')}" if h.get("error") else "")
                    if h.get("status") == "missing" and mock:
                        warnings.append(msg + " (mock-режим: встроенный фейк/запасной путь, проверит синтетика)")
                    elif name in req_modules or optional_policy == "fail":
                        failures.append(msg)
                    else:
                        warnings.append(msg + " (дополнительный модуль, политика warn)")
                if not failures:
                    dev = ex.request("device", {}, timeout=max(0.1, left()))
                    reg.set_device("cuda" if dev == "cuda" else "cpu")
    except WorkerTimeout as exc:
        failures.append(f"таймаут прогрева моделей (общий дедлайн {total:.0f} с): {exc}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"прогрев моделей: {type(exc).__name__}: {exc}")

    # ---- 3. синтетический прогон (только если модели поднялись)
    if not failures:
        if left() <= 0:
            failures.append(f"дедлайн прогрева {total:.0f} с истёк до синтетического прогона")
        else:
            try:
                failures += _warmup_synthetic(cfg, left(), required, mock, allow_fallbacks, optional_policy,
                                              with_vlm, report)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"синтетический прогон: {type(exc).__name__}: {exc}")

    # ---- 4. общий дедлайн
    elapsed = time.monotonic() - t0
    if elapsed > total and not any("дедлайн" in f for f in failures):
        failures.append(f"дедлайн прогрева {total:.0f} с истёк (прошло {elapsed:.1f} с)")
    report["seconds"] = round(elapsed, 3)
    report["ok"] = not failures
    reg.warmup_report = report
    if failures:
        msg = "прогрев не пройден: " + "; ".join(failures)
        reg.warmup_error = msg
        raise WarmupError(msg[:1500], report)
    reg.warm = True


def _warmup_synthetic(cfg: Mapping, budget_s: float, required: list[str], mock: bool, allow_fallbacks: bool,
                      optional_policy: str, with_vlm: bool, report: dict) -> list[str]:
    import tempfile

    from .media import make_synthetic_video

    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="sentinel-warmup-") as tmp:
        path = make_synthetic_video(Path(tmp) / "warmup.mp4", seconds=float(cfg_get(cfg, "warmup.synthetic_seconds", 2.0)))
        over: dict[str, Any] = {"cache": {"mode": "off"}, "partial": {"enabled": False}}
        if not with_vlm:
            over["stages"] = {"vlm_verify": False, "vlm_discover": False}
        res = run(path, "warmup", None, max(0.1, budget_s), overrides=over)
    meta = res.meta or {}
    stages = meta.get("stages") or {}
    report["synthetic"] = {"status": res.status, "verdict": res.verdict, "complete": meta.get("complete"),
                           "degraded": bool(meta.get("degraded")), "error": meta.get("error"),
                           "timings": meta.get("timings", {}), "warnings": (meta.get("warnings") or [])[:10]}
    if res.status != "done":
        failures.append(f"синтетический прогон: status={res.status}" + (f" ({meta['error']})" if meta.get("error") else ""))
    elif meta.get("complete") is not True:
        failures.append("синтетический прогон: анализ неполный (meta.complete=false)")
    if meta.get("degraded"):
        failures.append("синтетический прогон деградировал (обязательная стадия не отработала)")
    for st_name in STAGE_ORDER:
        st = stages.get(st_name) or {}
        status, origin = st.get("status"), st.get("origin", "")
        entry = {"status": status or "not_run", "origin": origin, "required": st_name in required}
        if st_name in required:
            if status not in OK_STATUSES:
                failures.append(f"{st_name}: {status or 'не запускалась'}" + (f": {st.get('error')}" if st.get("error") else ""))
            elif origin == "builtin-mock" and not mock:
                failures.append(f"{st_name}: встроенный фейк вне mock-режима")
            elif origin.startswith("fallback") and not allow_fallbacks:
                failures.append(f"{st_name}: сработал запасной путь ({origin}) — модуль стадии не работает")
        elif status in ("disabled",):
            entry["policy"] = "disabled"             # выключена флагом — явно, не «успех»
        elif status in ("skipped", None, "pending"):
            entry["policy"] = "not_exercised"        # синтетика её не задействовала (нет кандидатов и т.п.)
        elif status in _FAILED_STAGE:
            msg = f"{st_name}: {status}" + (f": {st.get('error')}" if st.get("error") else "")
            if optional_policy == "fail":
                failures.append(msg)
            else:
                report["warnings"].append(msg + " (дополнительная стадия, политика warn)")
                entry["policy"] = "degraded"
        report["stages"][st_name] = entry
    report["degraded_optional"] = [k for k, v in report["stages"].items() if v.get("policy") == "degraded"]
    return failures


# ============================================================== ядро

def _run(ctx: RunContext) -> AnalysisResult:
    cfg = ctx.cfg
    # mock НЕ включается мутацией os.environ: это глобальный эффект процесса API (следующие задачи
    # ушли бы в mock). Модулям mock передаётся только через env процесса-воркера (ProcessExecutor)
    # или явным SENTINEL_MOCK=1 / --mock на уровне процесса.
    ctx.report(STAGES["ingest"].label, 0.0)
    path = Path(ctx.video_path)
    if not path.is_file():
        return _failed(ctx.job_id, f"файл не найден: {ctx.video_path}", _meta(ctx))

    # ---- идентичность видео, кэш, каталог
    ctx.sha = sha256_file(path)
    ctx.cache = StageCache(cache_root(), ctx.sha, str(cfg_get(cfg, "cache.mode", "read_write")))
    ctx.catalog, ctx.catalog_source, cat_warn = load_catalog_safe()
    if cat_warn:
        ctx.warn(cat_warn)

    # ---- исполнитель (один владелец GPU); ждём аренду в пределах дедлайна
    ctx.executor = get_executor(cfg)
    lease_timeout = cfg_get(cfg, "executor.lease_timeout_s", None)
    lease_timeout = ctx.budget.remaining() - 2.0 if lease_timeout is None else float(lease_timeout)
    t_queue = time.perf_counter()
    with ctx.executor.lease(timeout=max(0.0, lease_timeout), cancel=ctx.cancel) as leased:
        ctx.queue_wait_s = time.perf_counter() - t_queue
        if not leased:
            if ctx.cancelled():
                return _failed(ctx.job_id, "задача отменена в очереди", _meta(ctx))
            return _failed(ctx.job_id, "исполнитель занят другой задачей до самого дедлайна", _meta(ctx))
        if ctx.executor.poisoned:
            ctx.warn("inline-исполнитель «отравлен» прошлым таймаутом: зависший поток может держать GPU")
        return _run_leased(ctx)


def _run_leased(ctx: RunContext) -> AnalysisResult:
    cfg = ctx.cfg
    ex = ctx.executor
    reg = ModelRegistry.instance(cfg)

    # ---- устройство — спрашиваем владельца GPU (родитель не импортирует torch)
    want = str(cfg_get(cfg, "gpu.device", "auto")).lower()
    if want == "cpu":
        ctx.device = "cpu"
    else:
        try:
            ctx.device = ex.request("device", {}, timeout=ctx.rpc_timeout(60))
        except Exception as exc:  # noqa: BLE001
            ctx.warn(f"не удалось определить устройство: {exc}")
            ctx.device = "cpu"
    reg.set_device(ctx.device)
    ctx.budget.device = ctx.device

    # ---- адаптивный sample_fps
    ctx.probe = probe = probe_video(ctx.video_path)
    ctx.sample_fps = choose_sample_fps(probe.get("duration"), ctx.budget.remaining(), ctx.device, cfg)
    if cfg_get(cfg, "refine.enabled", False):  # coarse→fine: обзор всей длительности реже, окна — плотно
        ctx.sample_fps = min(ctx.sample_fps, float(cfg_get(cfg, "refine.overview_fps", 2.0)))
    ctx.work_path = ascii_safe_path(ctx.video_path, data_root() / "jobs" / _safe_id(ctx.job_id),
                                    str(cfg_get(cfg, "paths.ascii_safe", "auto")))

    # ---- ingest (обязательная; встроенный декодер — допустимый fallback с warning)
    if not _ingest(ctx):
        return _finalize(ctx, None)
    ctx.budget.set_frames(len(ctx.clip_info.get("times", [])))

    # ---- CV-ядро
    dets = _stage(ctx, "detect", [("ml.detect", "detect", (CLIP,), {"every_n": int(cfg_get(cfg, "detect_every_n", 1))})],
                  upstream=["ingest"], stage_cfg={"every_n": cfg_get(cfg, "detect_every_n", 1)})
    dets = dets[0] if dets else []
    tz = _stage(ctx, "track", [("ml.track", "track", (CLIP, dets), {})], upstream=["detect"])
    tracks, zones = tz if tz else ([], [])
    pose_out = _stage(ctx, "pose", [("ml.pose", "estimate", (CLIP, tracks), {}),
                                    ("ml.pose", "pose_signals", ("$0", tracks, zones), {})], upstream=["track"])
    poses, pose_states = pose_out if pose_out else ([], [])
    sc = _stage(ctx, "scene", [("ml.scene", "scene_states", (CLIP, dets, tracks, zones, pose_states), {})],
                upstream=["detect", "track", "pose"])
    states = sc[0] if sc else []
    if ctx.records["scene"].status not in OK_STATUSES + ("disabled",) and dets and not ctx.cancelled():
        states = states_from_detections(ctx.clip_info.get("times", []), dets, pose_states)
        ctx.warn(f"scene: использованы грубые состояния из детекций ({len(states)} шт.) — анализ неполный")
    ru = _stage(ctx, "rules", [("ml.rules", "run_rules", (states, tracks, zones, ctx.catalog), {})],
                upstream=["scene", "track"], stage_cfg={"catalog": catalog_hash(ctx.catalog)},
                cacheable_in_from_cache=False)
    ev_rules = ru[0] if ru else []
    sq = _stage(ctx, "seqrules", [("ml.seqrules", "run", (CLIP, dets, tracks, zones, poses, ctx.catalog), {})],
                upstream=["detect", "track", "pose"], stage_cfg={"catalog": catalog_hash(ctx.catalog)})
    ev_seq = _seqrules_outputs(ctx, sq)
    ev_rules = list(ev_rules) + ev_seq

    # ---- временная модель (последовательно, тот же владелец GPU)
    tm = _stage(ctx, "temporal", [("ml.temporal", "score", (CLIP,), {})], upstream=["ingest"])
    risk, ev_temporal = tm if tm else ([], [])

    candidates = list(ev_rules) + list(ev_temporal)
    candidates = _refine(ctx, candidates, zones)                 # H1: выключено по умолчанию
    candidates = apply_timecode_policy(candidates, cfg)          # H2: по умолчанию "module" (без изменений)
    _partial(ctx, candidates, risk, "after_core")

    # ---- VLM (дополнительные)
    ev_verify: list[Event] = []
    if candidates or not cfg_get(cfg, "stages.vlm_verify", True):
        vv = _stage(ctx, "vlm_verify", [("ml.vlm", "verify", (CLIP, candidates, "$budget"), {})],
                    upstream=["rules", "seqrules", "temporal"])
        ev_verify = vv[0] if vv else []
    else:
        _record(ctx, "vlm_verify", StageRecord(status="skipped", required=False, error="нет кандидатов"))
    if ev_verify:
        _partial(ctx, candidates + ev_verify, risk, "after_vlm_verify")
    if ctx.budget.near_deadline():
        _partial(ctx, candidates + ev_verify, risk, "before_vlm_discover")
    dv = _stage(ctx, "vlm_discover", [("ml.vlm", "discover", (CLIP, "$budget"), {})], upstream=["ingest"])
    ev_discover = _filter_types(ctx, dv[0] if dv else [], cfg_get(cfg, "vlm_discover_types", None))

    events = list(candidates) + apply_timecode_policy(list(ev_verify) + list(ev_discover), cfg)
    ctx.outputs["events_all"] = events
    if ctx.budget.near_deadline():
        _partial(ctx, events, risk, "before_fusion")
    return _finalize(ctx, _fusion(ctx, events, risk, tracks))


# ============================================================== стадии

def _filter_types(ctx: RunContext, events: list[Event], types: Optional[list]) -> list[Event]:
    """Допуск типов из пути VLM (экспериментальные профили): после стадии, ключ кэша стадии не меняется."""
    if not types or not events:
        return list(events)
    keep = [e for e in events if e.type_id in set(types)]
    ctx.outputs["vlm_discover_filter"] = {"types": list(types), "in": len(events), "kept": len(keep)}
    ctx.warn(f"vlm_discover: в слияние допущены только типы {', '.join(types)} ({len(keep)} из {len(events)} событий)")
    return keep


def _seqrules_outputs(ctx: RunContext, sq: Optional[tuple]) -> list[Event]:
    """События seqrules + наблюдения для UI (data/jobs/<id>/observations.json). Ошибка правила при обязательной
    стадии делает анализ неполным: «правило не сработало» ≠ «нарушения нет»."""
    if not sq:
        return []
    events, observations = sq
    meta = observations[0] if observations and observations[0].get("kind") == "meta" else {}
    for w in meta.get("warnings", []) or []:
        ctx.warn(str(w))
    if is_required("seqrules", ctx.cfg):
        for name, inf in (meta.get("rules") or {}).items():
            if isinstance(inf, dict) and inf.get("status") == "error":
                ctx.incomplete.append(f"seqrules/{name} (error: {inf.get('error', '')})")
    ctx.outputs["observations"] = observations
    try:
        path = data_root() / "jobs" / _safe_id(ctx.job_id) / "observations.json"
        serde.write_json(path, {"job_id": ctx.job_id, "sha256": ctx.sha, "items": observations})
    except Exception as exc:  # noqa: BLE001
        ctx.warn(f"observations.json не записан: {type(exc).__name__}: {exc}")
    return list(events)


def _refine(ctx: RunContext, candidates: list[Event], zones: list) -> list[Event]:
    """coarse→fine: плотная выборка окон вокруг кандидатов, detect→track→scene→rules на окне,
    уточнение начала события. Кандидаты никогда не отбрасываются (см. refine.py)."""
    cfg = ctx.cfg
    rec = StageRecord(required=False)
    if not cfg_get(cfg, "refine.enabled", False):
        rec.status, rec.error = "disabled", "refine.enabled=false (режим по умолчанию выключен)"
        ctx.records["refine"] = rec
        return candidates
    if not candidates:
        rec.status, rec.error = "skipped", "нет кандидатов"
        ctx.records["refine"] = rec
        return candidates
    if ctx.cancelled() or ctx.out_of_time():
        rec.status, rec.error = "skipped", "отмена/дедлайн"
        _record(ctx, "refine", rec)
        return candidates
    can, why = ctx.budget.can_start("refine")
    if not can:
        rec.status, rec.error = "skipped", why
        _record(ctx, "refine", rec)
        return candidates
    ctx.report(STAGES["refine"].label, ctx.stage_progress_base("refine"))
    fps = float(cfg_get(cfg, "refine.fps", 8.0))
    max_frames = int(cfg_get(cfg, "refine.max_frames", 96))
    reuse_zones = bool(cfg_get(cfg, "refine.reuse_overview_zones", True))
    windows: list[Window] = build_windows(candidates, cfg, reference_duration(ctx.clip_info, ctx.probe))
    t0 = time.perf_counter()
    end = time.monotonic() + ctx.budget.timeout("refine")
    for w in windows:
        ref = ClipRef(window=(round(w.t0, 3), round(w.t1, 3), fps, max_frames))
        try:
            dets = _window_call(ctx, end, "ml.detect", "detect", (ref,), {"every_n": 1})
            tracks_w, zones_w = _window_call(ctx, end, "ml.track", "track", (ref, dets), {})
            zz = zones if (reuse_zones and zones) else zones_w
            states = _window_call(ctx, end, "ml.scene", "scene_states", (ref, dets, tracks_w, zz, []), {})
            evs = _window_call(ctx, end, "ml.rules", "run_rules", (states, tracks_w, zz, ctx.catalog), {})
            w.events = serde.roundtrip_list(list(evs or []), "event")
            w.processed = True
        except _WindowStop as exc:
            w.error = str(exc)
            if exc.fatal:
                break
    refined, stats = apply_refinement(candidates, windows)
    rec.seconds = time.perf_counter() - t0
    rec.n_out = len(refined)
    rec.origin = "module"
    errors = [w.error for w in windows if w.error]
    if stats["processed"]:
        rec.status = "ok"
    else:
        rec.status, rec.error = "error", "; ".join(errors)[:300] or "ни одно окно не обработано"
    _record(ctx, "refine", rec)
    ctx.outputs["refine_stats"] = {**stats, "fps": fps, "max_frames": max_frames, "errors": errors[:5],
                                   "windows_s": [[round(w.t0, 2), round(w.t1, 2)] for w in windows]}
    return refined


class _WindowStop(Exception):
    def __init__(self, msg: str, fatal: bool = False) -> None:
        super().__init__(msg)
        self.fatal = fatal


def _window_call(ctx: RunContext, end: float, module: str, func: str, args: tuple, kwargs: dict) -> Any:
    left = end - time.monotonic()
    if left <= 0.3 or ctx.cancelled() or ctx.out_of_time():
        raise _WindowStop("бюджет уточнения исчерпан", fatal=True)
    res = ctx.executor.call(module, func, args, kwargs, left)
    if not res.ok:
        raise _WindowStop(f"{module}.{func}: {res.status}: {res.error.splitlines()[0][:150] if res.error else ''}",
                          fatal=res.status in ("timeout", "cancelled", "crashed"))
    return res.value

def _ingest(ctx: RunContext) -> bool:
    rec = StageRecord(required=True)
    ctx.records["ingest"] = rec
    ctx.report(STAGES["ingest"].label, ctx.stage_progress_base("ingest"))
    t0 = time.perf_counter()
    timeout = ctx.rpc_timeout(float(cfg_get(ctx.cfg, "executor.load_clip_timeout_s", 60)))
    try:
        ctx.ingest_identity = ctx.executor.request("identity", {"module": "ml.ingest", "func": "load_video"},
                                                   ctx.rpc_timeout(30))
        info = ctx.executor.load_clip(ctx.work_path, ctx.sample_fps, timeout)
    except (WorkerTimeout, WorkerCrashed, WorkerCancelled) as exc:
        rec.status = {WorkerTimeout: "timeout", WorkerCrashed: "crashed"}.get(type(exc), "cancelled")
        rec.error = str(exc)
    except Exception as exc:  # noqa: BLE001
        rec.status, rec.error = "error", f"{type(exc).__name__}: {exc}".splitlines()[0]
    else:
        ctx.clip_info = info
        for w in info.get("warnings", []):
            ctx.warn(w)
        rec.status = "ok"
        rec.origin = info.get("source", "")
        rec.n_out = len(info.get("times", []))
        ctx.evidence_clip = EvidenceClip(ctx.work_path, info)
        if not rec.n_out:
            rec.status, rec.error = "error", "в клипе нет кадров"
    rec.seconds = time.perf_counter() - t0
    ctx.keys["ingest"] = stable_hash(ctx.sha, ctx.sample_fps, ctx.ingest_identity, ctx.clip_info.get("source"))
    rec.key = ctx.keys["ingest"]
    if rec.status != "ok":
        ctx.incomplete.append(f"ingest ({rec.status}: {rec.error})")
        return False
    ctx.cache.save_video_info({k: v for k, v in ctx.clip_info.items() if k not in ("times", "warnings", "path")}
                              | {"n_times": rec.n_out, "sample_fps": ctx.sample_fps})
    return True


def _record(ctx: RunContext, stage: str, rec: StageRecord) -> None:
    ctx.records[stage] = rec
    if rec.required and rec.status not in OK_STATUSES:
        ctx.incomplete.append(f"{stage} ({rec.status}{': ' + rec.error if rec.error else ''})")
    elif not rec.required and rec.status not in OK_STATUSES and rec.status != "disabled":
        ctx.warn(f"дополнительная стадия {stage}: {rec.status}{' — ' + rec.error if rec.error else ''}")


def _stage(ctx: RunContext, stage: str, calls: list[tuple[str, str, tuple, dict]], *, upstream: list[str],
           stage_cfg: Optional[dict] = None, cacheable_in_from_cache: bool = True) -> Optional[tuple]:
    """Запуск стадии: флаг → кэш → бюджет → VRAM → вызов(ы) в исполнителе → проверка → кэш.
    Возвращает кортеж выходов (как в STAGE_FILES) или None, если стадия не дала результата."""
    cfg = ctx.cfg
    spec = STAGES[stage]
    required = is_required(stage, cfg)
    rec = StageRecord(required=required)
    ctx.report(spec.label, ctx.stage_progress_base(stage))

    if ctx.cancelled():
        rec.status, rec.error = "cancelled", "задача отменена"
        _record(ctx, stage, rec)
        return None
    if not cfg_get(cfg, f"stages.{stage}", True):
        rec.status, rec.error = "disabled", "выключена в конфиге/флагах"
        _record(ctx, stage, rec)
        if not required:
            ctx.warn(f"дополнительная стадия {stage} выключена флагом")
        return None

    if ctx.out_of_time():
        rec.status, rec.error = "skipped", f"до дедлайна осталось {ctx.budget.remaining():.1f} с"
        _record(ctx, stage, rec)
        return None
    try:
        ident = ctx.executor.request("identity", {"module": spec.module, "func": spec.func}, ctx.rpc_timeout(30))
    except Exception as exc:  # noqa: BLE001
        ident = {"origin": "unknown", "error": str(exc)}
    rec.version = str(ident.get("version", ident.get("origin", "")))
    ctx.identities[stage] = ident
    key = stable_hash(stage, ctx.keys.get("ingest"), ident, stage_cfg or {}, bool(cfg.get("mock")),
                      [ctx.keys.get(u, "") for u in upstream])
    ctx.keys[stage] = key
    rec.key = key

    # ---- кэш
    cache_mode = ctx.cache.mode if ctx.cache else "off"
    if ctx.cache is not None and cache_mode != "off" and not (cache_mode == "from_cache" and not cacheable_in_from_cache):
        hit = ctx.cache.load(stage, key, allow_stale=(cache_mode == "from_cache"))
        if hit is not None:
            rec.status, rec.origin = "cached", "cache"
            rec.n_out = len(hit[0]) if hit and hasattr(hit[0], "__len__") else None
            _record(ctx, stage, rec)
            ctx.outputs[stage] = hit
            return hit

    if ident.get("origin") == "missing":
        rec.status, rec.error = "missing", str(ident.get("error", "модуль недоступен"))
        _record(ctx, stage, rec)
        return None

    # ---- бюджет
    can, why = ctx.budget.can_start(stage)
    if not can:
        rec.status, rec.error = "skipped", why
        _record(ctx, stage, rec)
        return None

    # ---- VRAM: один владелец GPU, перед тяжёлой моделью реально выгружаем другие
    if spec.gpu and ctx.device == "cuda":
        _prepare_vram(ctx, stage)

    # ---- вызов(ы)
    outputs: list[Any] = []
    t0 = time.perf_counter()
    res: Optional[CallResult] = None
    for module, func, args, kwargs in calls:
        args = tuple(_subst(a, ctx, stage, outputs) for a in args)
        res = ctx.executor.call(module, func, args, kwargs, ctx.budget.timeout(stage))
        rec.origin = res.origin or rec.origin
        rec.cold = bool(rec.cold) or bool(getattr(res, "cold", False))
        if res.vram_peak_mb is not None:
            rec.vram_peak_mb = max(rec.vram_peak_mb or 0.0, res.vram_peak_mb)
        if not res.ok:
            break
        outputs.append(res.value)
    rec.seconds = time.perf_counter() - t0
    ctx.budget.observe(stage, rec.seconds)
    if res is None or not res.ok:
        rec.status = res.status if res is not None else "error"
        rec.error = (res.error if res is not None else "нет вызовов").splitlines()[0][:500]
        if res is not None and res.trace:
            log.debug("%s trace:\n%s", stage, res.trace)
        _record(ctx, stage, rec)
        return None

    # ---- проверка формы и нормализация (живой выход ≡ выходу из кэша)
    try:
        values = _normalize(stage, outputs, bool(cfg_get(cfg, "cache.normalize_outputs", True)))
    except Exception as exc:  # noqa: BLE001 — нарушение контракта модулем
        rec.status, rec.error = "error", f"выход не соответствует контракту: {type(exc).__name__}: {exc}"
        _record(ctx, stage, rec)
        return None
    rec.status = "ok"
    rec.n_out = len(values[0]) if values and hasattr(values[0], "__len__") else None
    _record(ctx, stage, rec)
    ctx.outputs[stage] = values
    if ctx.cache is not None:
        ctx.cache.save(stage, key, values, {"origin": rec.origin, "version": rec.version,
                                            "sample_fps": ctx.sample_fps, "identity": ident})
    return values


def _subst(arg: Any, ctx: RunContext, stage: str, outputs: list[Any]) -> Any:
    if isinstance(arg, str) and arg == "$budget":
        return round(ctx.budget.stage_budget_for_call(stage), 2)
    if isinstance(arg, str) and arg.startswith("$") and arg[1:].isdigit():
        return outputs[int(arg[1:])]
    return arg


def _normalize(stage: str, outputs: list[Any], roundtrip: bool) -> tuple:
    """Приводит сырые выходы вызовов к кортежу в порядке STAGE_FILES и проверяет типы."""
    if stage == "track":
        if not isinstance(outputs[0], (tuple, list)) or len(outputs[0]) != 2:
            raise TypeError("track() должен вернуть (tracks, zones)")
        raw = (list(outputs[0][0]), list(outputs[0][1]))
    elif stage == "temporal":
        if not isinstance(outputs[0], (tuple, list)) or len(outputs[0]) != 2:
            raise TypeError("score() должен вернуть (risk, events)")
        raw = (list(outputs[0][0]), list(outputs[0][1]))
    elif stage == "pose":
        raw = (list(outputs[0]), list(outputs[1]))
    elif stage == "seqrules":
        if not isinstance(outputs[0], (tuple, list)) or len(outputs[0]) != 2:
            raise TypeError("seqrules.run() должен вернуть (events, observations)")
        raw = (list(outputs[0][0]), [dict(x) for x in outputs[0][1]])
    else:
        if not isinstance(outputs[0], (list, tuple)):
            raise TypeError(f"{stage}: ожидался список, получено {type(outputs[0]).__name__}")
        raw = (list(outputs[0]),)
    if not roundtrip:
        return raw
    out = []
    for value, kind in zip(raw, _KINDS[stage]):
        out.append(serde.roundtrip_risk(value) if kind == "risk" else serde.roundtrip_list(value, kind))
    return tuple(out)


def _prepare_vram(ctx: RunContext, stage: str) -> None:
    unload = list(cfg_get(ctx.cfg, f"executor.unload_before.{stage}", []) or [])
    if stage.startswith("vlm") and ModelRegistry.instance(ctx.cfg).vlm_policy() == "keep":
        unload = []
    need = float(cfg_get(ctx.cfg, f"gpu.vram_gb.{stage}", 1.0))
    try:
        info = ctx.executor.request("prepare_vram", {"need_gb": need, "unload_modules": unload},
                                    ctx.rpc_timeout(60))
    except Exception as exc:  # noqa: BLE001
        ctx.warn(f"{stage}: проверка VRAM не удалась: {exc}")
        return
    if info.get("action") == "unload":
        ctx.outputs.setdefault("vram_events", []).append({"stage": stage, **info})
    if not info.get("ok", True) and cfg_get(ctx.cfg, "executor.restart_on_vram_shortage", True):
        ctx.warn(f"{stage}: после unload() VRAM не хватает ({info.get('after', {}).get('free_gb')} ГБ) — "
                 f"перезапуск воркера")
        ctx.executor.restart()


def _fusion(ctx: RunContext, events: list[Event], risk: list[tuple[float, float]], tracks: list
            ) -> Optional[AnalysisResult]:
    rec = StageRecord(required=is_required("fusion", ctx.cfg))
    if ctx.cancelled():
        rec.status, rec.error = "cancelled", "задача отменена"
        _record(ctx, "fusion", rec)
        return None
    ctx.report(STAGES["fusion"].label, ctx.stage_progress_base("fusion"))
    t0 = time.perf_counter()
    respawn_s = float(cfg_get(ctx.cfg, "reserve_s.respawn_s", 8))
    worker_dead = getattr(ctx.executor, "mode", "") == "process" and not ctx.executor.alive()
    if worker_dead and ctx.budget.remaining() < respawn_s:
        res = CallResult("skipped", error=f"воркер остановлен, до дедлайна {ctx.budget.remaining():.1f} с — "
                                          f"без перезапуска")
    else:
        res = ctx.executor.call("ml.fusion", "fuse", (CLIP, events, risk, ctx.catalog, ctx.job_id), {},
                                ctx.budget.timeout("fusion"))
    result: Optional[AnalysisResult] = None
    if res.ok and hasattr(res.value, "violations") and getattr(res.value, "status", "") != "failed":
        result, rec.status, rec.origin = res.value, "ok", res.origin
    else:
        # Обязательная стадия не дала результата (исключение, таймаут, нет модуля или fuse() сам вернул
        # status=failed). Встроенное слияние строит отчёт ТОЛЬКО ради уже найденных нарушений
        # (degraded violation); «нет нарушения» по нему не выдаётся — _record заносит стадию в неполные.
        if res.ok:
            rec.status = "error"
            meta_err = (getattr(res.value, "meta", None) or {}).get("error") if res.value is not None else None
            rec.error = (f"fuse() вернул status=failed: {meta_err}" if getattr(res.value, "status", "") == "failed"
                         else "fuse() вернул не AnalysisResult")[:300]
        else:
            rec.status = res.status if res.status in _FAILED_STAGE or res.status == "skipped" else "error"
            rec.error = res.error.splitlines()[0][:300] if res.error else res.status
        rec.origin = "fallback"
        result = fallback_fuse(ctx.evidence_clip, events, risk, ctx.catalog, ctx.job_id, ctx.cfg, tracks)
        result.meta["fusion_fallback"] = True
    rec.seconds = time.perf_counter() - t0
    _record(ctx, "fusion", rec)
    if not result.timeline:
        result.timeline = risk or derive_risk(ctx.clip_info.get("times", []), events)
    return result


# ============================================================== partial и финал

def _partial(ctx: RunContext, events: list[Event], risk: list, milestone: str) -> None:
    if not cfg_get(ctx.cfg, "partial.enabled", True):
        return
    if milestone not in (cfg_get(ctx.cfg, "partial.milestones", []) or []) and not ctx.budget.near_deadline():
        return
    try:
        res = fallback_fuse(ctx.evidence_clip, events, risk, ctx.catalog, ctx.job_id, ctx.cfg)
        found = "violation" if res.violations else None
        res.status = "processing"  # незавершённый результат никогда не выдаётся за done
        res.verdict = None         # ...и не несёт вердикта (ни no_violation, ни violation)
        res.meta = {**_meta(ctx), "partial": True, "complete": False, "milestone": milestone,
                    "partial_verdict": found}
        res.meta["completeness"]["complete"] = False
        res.meta["completeness"]["reason"] = f"промежуточный результат ({milestone})"
        data = serde.result_to_dict(res)
        data.update({"partial": True, "partial_verdict": found,
                     "partial_violations": [{"type_id": v.type_id, "type": v.type, "timecode_sec": v.timecode_sec,
                                             "confidence": v.confidence} for v in res.violations]})
        path = data_root() / "jobs" / _safe_id(ctx.job_id) / "partial.json"
        serde.write_json(path, data)
        ctx.partial_written.append(milestone)
    except Exception as exc:  # noqa: BLE001
        ctx.warn(f"partial.json не записан: {type(exc).__name__}: {exc}")


def _finalize(ctx: RunContext, result: Optional[AnalysisResult]) -> AnalysisResult:
    if all(s in ctx.records and ctx.records[s].status in OK_STATUSES for s in _required(ctx)):
        cov = canonical_coverage(ctx.records, ctx.outputs, ctx.clip_info, ctx.probe, ctx.cfg, _required(ctx),
                                 {n: r.as_dict() for n, r in ctx.records.items()})
        for reason in coverage_gate(cov, ctx.cfg):
            ctx.incomplete.append(f"покрытие — {reason}")
    meta = _meta(ctx)
    complete = meta["completeness"]["complete"]
    policy = str(cfg_get(ctx.cfg, "on_required_failure", "violation_only"))
    if result is None:
        result = AnalysisResult(job_id=ctx.job_id, status="failed", verdict=None)
    base_meta = dict(result.meta or {})
    warnings = list(base_meta.pop("warnings", []) or []) + ctx.warnings
    result.meta = {**base_meta, **meta, "warnings": list(dict.fromkeys(warnings)), "complete": complete}
    result.job_id = ctx.job_id
    if complete:
        result.status = "done"
        if result.verdict not in ("violation", "no_violation"):
            result.verdict = "violation" if result.violations else "no_violation"
        return result
    reason = "Базовый анализ не завершён: " + "; ".join(ctx.incomplete)
    if result.violations and policy == "violation_only":
        result.status, result.verdict = "done", "violation"
        result.meta["degraded"] = True
        result.meta["warnings"].append(reason + " — нарушение найдено доступными стадиями, вердикт "
                                                 "«нет нарушения» был бы недостоверен")
    else:
        result.status, result.verdict = "failed", None
        result.meta["error"] = reason
    return result


def _vlm_usage(ctx: RunContext) -> dict:
    """Сводка VLM для замера L2: вызовы/токены (extra.vlm.budget модуля A11) и статусы проверки."""
    out: dict[str, Any] = {}
    for stage in ("vlm_verify", "vlm_discover"):
        evs = (ctx.outputs.get(stage) or ([],))[0]
        if not evs:
            continue
        statuses: dict[str, int] = {}
        budget = None
        for e in evs:
            v = (getattr(e, "extra", {}) or {}).get("vlm") or {}
            if isinstance(v, dict):
                st = str(v.get("status", "?"))
                statuses[st] = statuses.get(st, 0) + 1
                budget = budget or v.get("budget")
        out[stage] = {"events": len(evs), "statuses": statuses, "budget": budget}
    return out


def _required(ctx: RunContext) -> list[str]:
    return [s for s in STAGE_ORDER if is_required(s, ctx.cfg)]


def _is_complete(ctx: RunContext) -> bool:
    return not ctx.incomplete and all(s in ctx.records and ctx.records[s].status in OK_STATUSES
                                      for s in _required(ctx))


def _provenance(ctx: RunContext) -> dict:
    models: dict[str, dict] = {}
    for stage, ident in ctx.identities.items():
        for m in ident.get("models", []) or []:
            if isinstance(m, dict) and m.get("name"):
                models.setdefault(str(m["name"]), {**m, "stage": stage})
    sources = sorted({getattr(e, "source", "") for e in ctx.outputs.get("events_all", []) if getattr(e, "source", "")})
    return {"pipeline_version": str(cfg_get(ctx.cfg, "pipeline_version", "1.0.0")), "contract_version": "1",
            "mock": bool(ctx.cfg.get("mock")), "models": list(models.values()), "sources": sources}


def _meta(ctx: RunContext) -> dict:
    reg = ModelRegistry.peek()
    timings = {name: round(r.seconds, 3) for name, r in ctx.records.items()}
    timings["total"] = round(time.perf_counter() - ctx.t_wall0, 3)
    meta: dict[str, Any] = {
        "pipeline_version": str(cfg_get(ctx.cfg, "pipeline_version", "1.0.0")),
        "timings": timings,
        "stages": {name: r.as_dict() for name, r in ctx.records.items()},
        "sample_fps": ctx.sample_fps,
        "device": ctx.device,
        "warnings": list(ctx.warnings),
        "mock": bool(ctx.cfg.get("mock")),
        "budget": ctx.budget.snapshot(),
        "video": {"sha256": ctx.sha, **{k: ctx.clip_info.get(k) for k in ("duration", "fps", "width", "height")},
                  "n_sampled": len(ctx.clip_info.get("times", []) or []), "decoder": ctx.clip_info.get("source")},
        "catalog_source": ctx.catalog_source,
        "models": {name: {"origin": r.origin, "version": r.version} for name, r in ctx.records.items()
                   if r.origin and r.origin not in ("cache",)},
        "cache": {"mode": ctx.cache.mode if ctx.cache else "off", "hits": list(ctx.cache.hits) if ctx.cache else [],
                  "dir": str(ctx.cache.dir) if ctx.cache else None},
        "required_stages": _required(ctx),
        "partial_written": list(ctx.partial_written),
    }
    # ---- полнота (contracts/extensions.py, A01) + покрытие по стадиям
    cov = canonical_coverage(ctx.records, ctx.outputs, ctx.clip_info, ctx.probe, ctx.cfg, _required(ctx),
                             meta["stages"])
    duration = reference_duration(ctx.clip_info, ctx.probe)
    times = ctx.clip_info.get("times") or []
    complete = _is_complete(ctx)
    cov["complete"] = bool(cov.get("complete")) and complete
    reason = None if complete else ("Базовый анализ не завершён: " + "; ".join(ctx.incomplete) if ctx.incomplete
                                    else "обязательные стадии не выполнены")
    meta["coverage"] = cov
    meta["duration_sec"] = round(duration, 3) if duration else None
    meta["completeness"] = completeness(ctx.records, _required(ctx), complete, meta["duration_sec"],
                                        round(max(times), 3) if times else None, reason, bool(ctx.cfg.get("mock")))
    meta["provenance"] = _provenance(ctx)
    meta["complete"] = complete
    if ctx.executor is not None:
        meta["executor"] = ctx.executor.stats()
    if ctx.outputs.get("vram_events"):
        meta["vram_events"] = ctx.outputs["vram_events"]
    if ctx.outputs.get("refine_stats"):
        meta["refine"] = ctx.outputs["refine_stats"]
    meta["queue_wait_s"] = round(ctx.queue_wait_s, 3)
    vlm = _vlm_usage(ctx)
    if vlm:
        meta["vlm"] = vlm
    meta["modes"] = {"refine": bool(cfg_get(ctx.cfg, "refine.enabled", False)),
                     "timecode_policy": str(cfg_get(ctx.cfg, "timecode.policy", "module"))}
    if reg is not None:
        meta["registry"] = {"warm": reg.warm}
    if ctx.budget.remaining() < 0:
        meta["deadline_exceeded_s"] = round(-ctx.budget.remaining(), 2)
    return meta


def _failed(job_id: str, error: str, meta: dict) -> AnalysisResult:
    meta = dict(meta)
    meta["error"] = error
    meta["complete"] = False
    return AnalysisResult(job_id=job_id, status="failed", verdict=None, meta=meta)


def _safe_id(job_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", job_id)[:80] or "job"


def dump_intermediate(ctx: RunContext, out_dir: str | Path) -> list[str]:
    """--dump-intermediate: выходы стадий в формате кэша (+ все события и meta по стадиям)."""
    out_dir = Path(out_dir)
    written: list[str] = []
    for stage, values in ctx.outputs.items():
        if stage not in STAGE_FILES:
            continue
        for (fname, _kind), value in zip(STAGE_FILES[stage], values):
            target = out_dir / fname
            (serde.write_jsonl if fname.endswith(".jsonl") else serde.write_json)(target, value)
            written.append(str(target))
    if "events_all" in ctx.outputs:
        serde.write_json(out_dir / "events_all.json", ctx.outputs["events_all"], indent=1)
        written.append(str(out_dir / "events_all.json"))
    serde.write_json(out_dir / "stages.json", {n: r.as_dict() for n, r in ctx.records.items()}, indent=1)
    written.append(str(out_dir / "stages.json"))
    return written
