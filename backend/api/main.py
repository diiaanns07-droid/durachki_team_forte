"""Kassa Sentinel — FastAPI. Запуск: `python -m backend.api` или
`uvicorn backend.api.main:app --workers 1` (очередь и модели живут в процессе: ровно 1 процесс!).

/api/v1/*  — API жюри (X-API-Key), /ui/v1/* — то же для веб-UI по HttpOnly cookie-сессии,
которую выдаёт POST /ui/v1/session только после проверки API-ключа или пароля UI."""
from __future__ import annotations

import inspect
import json
import logging
import re
import shutil
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.security import APIKeyHeader

from .config import API_DIR, Settings, parse_rate
from .logging_utils import RequestContextMiddleware, setup_logging
from .media import range_response
from .results import iter_frames, jury_view, status_view
from .security import (ApiKeyAuth, RateLimiter, UiTokenSigner, check_secret, client_ip,
                       cross_site, fingerprint, ip_in, load_api_keys, load_ui_secret)
from .store import JobStore
from .uploads import UploadError, normalize_filename, receive_video, validate_video
from .worker import JobManager, PipelineSpec

try:
    from contracts.sentinel_contracts import AnalysisResult  # noqa: F401
except ImportError:  # пока ветка A01 не смерджена
    from contracts.sentinel_contracts import AnalysisResult  # noqa: F401

API_VERSION = "2.0.0"
SESSION_COOKIE = "ks_session"
_JOB_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
log = logging.getLogger("sentinel.api")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False,
                              description="Ключ из SENTINEL_API_KEYS")

_MULTIPART_DOC = {"requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
    "type": "object", "required": ["file"],
    "properties": {"file": {"type": "string", "format": "binary",
                            "description": ".mov/.mp4/.m4v, до 300 МБ"}}}}}}}


@dataclass
class Ctx:
    s: Settings
    store: JobStore
    manager: JobManager
    keys: ApiKeyAuth
    signer: UiTokenSigner
    rl: RateLimiter = field(default_factory=RateLimiter)
    cache: "OrderedDict[str, bytes]" = field(default_factory=OrderedDict)
    copilot: Any = None
    copilot_checked: bool = False
    dedup_lock: Any = None  # asyncio.Lock: поиск дубликата + создание задачи атомарны


def _ctx(request: Request) -> Ctx:
    ctx = getattr(request.app.state, "ctx", None)
    if ctx is None:
        raise HTTPException(503, "service starting")
    return ctx


def _headers(request: Request) -> dict[str, str]:
    return {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in request.scope["headers"]}


def _ip(request: Request, s: Settings) -> str:
    peer = request.client.host if request.client else None
    return client_ip(peer, _headers(request), s.trusted_proxies)


def _limit(ctx: Ctx, request: Request, bucket: str, spec: str) -> None:
    n, per = parse_rate(spec)
    ok, retry = ctx.rl.hit(bucket, _ip(request, ctx.s), n, per)
    if not ok:
        raise HTTPException(429, "too many requests", headers={"Retry-After": str(int(retry) + 1)})


def _is_https(request: Request, s: Settings) -> bool:
    """Secure-флаг cookie: https напрямую или через доверенный прокси (cloudflared)."""
    if request.url.scheme == "https":
        return True
    peer = request.client.host if request.client else ""
    if peer and ip_in(peer, s.trusted_proxies):
        h = _headers(request)
        return (h.get("x-forwarded-proto", "").split(",")[0].strip().lower() == "https"
                or '"https"' in h.get("cf-visitor", ""))
    return False


# ------------------------------------------------------------------ auth dependencies
async def require_api_key(request: Request, key: Optional[str] = Security(api_key_header)) -> str:
    ctx = _ctx(request)
    if not key:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            key = auth[7:].strip()
    if key and ctx.keys.check(key):
        principal = "api:" + fingerprint(key)
        request.state.principal = principal
        return principal
    _limit(ctx, request, "auth_fail", ctx.s.rl_auth_fail)
    raise HTTPException(401, "invalid or missing API key (header X-API-Key)",
                        headers={"WWW-Authenticate": "ApiKey"})


def _ui_dependency(bucket: str) -> Callable[..., Any]:
    async def require_ui_session(request: Request) -> str:
        ctx = _ctx(request)
        if not ctx.s.ui_enabled:
            raise HTTPException(404, "UI access disabled")
        token = request.cookies.get(SESSION_COOKIE, "")
        if ctx.signer.verify(token) is None:
            _limit(ctx, request, "auth_fail", ctx.s.rl_auth_fail)
            raise HTTPException(401, "UI session required: POST /ui/v1/session with api_key or password")
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            if cross_site(_headers(request), ctx.s.cors_origins):
                raise HTTPException(403, "cross-site request rejected")
            if not ctx.signer.check_csrf(token, request.headers.get("x-csrf-token")):
                raise HTTPException(403, "missing or invalid X-CSRF-Token")
        spec = {"upload": ctx.s.rl_ui_upload, "ask": ctx.s.rl_ui_ask}.get(bucket, ctx.s.rl_ui_read)
        _limit(ctx, request, "ui_" + bucket, spec)
        request.state.principal = "ui"
        return "ui"
    return require_ui_session


# ------------------------------------------------------------------ handlers
def _job_row(ctx: Ctx, job_id: str) -> dict[str, Any]:
    if not _JOB_ID.match(job_id or ""):
        raise HTTPException(404, "job not found")
    row = ctx.store.get(job_id)
    if not row:
        raise HTTPException(404, "job not found")
    return row


def _result_bytes(ctx: Ctx, row: dict[str, Any], view: str = "raw") -> bytes:
    """Сериализованный ответ из result.json (кэш LRU по задаче и представлению)."""
    key = f"{row['id']}:{row['result_path']}:{view}"
    data = ctx.cache.get(key)
    if data is None:
        raw = Path(row["result_path"]).read_bytes()
        if view == "raw":
            data = raw
        else:
            obj = json.loads(raw)
            data = json.dumps(jury_view(obj, view, ctx.s.jury_embed_rule), ensure_ascii=False,
                              separators=(",", ":")).encode("utf-8")
        ctx.cache[key] = data
        while len(ctx.cache) > 64:
            ctx.cache.popitem(last=False)
    else:
        ctx.cache.move_to_end(key)
    return data


def _profile(ctx: Ctx, request: Request, ui: bool) -> str:
    if ui:
        return "ui"
    q = request.query_params.get("profile", "")
    return q if q in ("extended", "strict") else ctx.s.jury_profile


def _job_payload(ctx: Ctx, row: dict[str, Any], profile: str) -> Response:
    """profile: extended|strict — для жюри (/api/v1), ui — полный отчёт (/ui/v1)."""
    jid, st = row["id"], row["status"]
    jprof = "extended" if profile == "ui" else profile
    if st in ("queued", "running"):
        live = ctx.manager.live(jid) or {}
        body = status_view(jid, "processing", jprof,
                           stage=live.get("stage") or row.get("stage") or "queued",
                           progress=round(float(live.get("progress", row.get("progress") or 0.0)), 3),
                           queue_position=ctx.manager.queue_position(jid),
                           elapsed_s=round(time.time() - row["created_at"], 1))
        return JSONResponse(body, headers={"Cache-Control": "no-store", "Retry-After": "2"})
    if st == "done" and row.get("result_path"):
        try:
            data = _result_bytes(ctx, row, "raw" if profile == "ui" else jprof)
            return Response(data, media_type="application/json", headers={"Cache-Control": "no-store"})
        except (OSError, ValueError):
            log.error("result file missing", extra={"job_id": jid})
            return JSONResponse(status_view(jid, "failed", jprof, error="result file missing"))
    body = status_view(jid, "failed", jprof, error=row.get("error") or "analysis failed")
    if profile == "ui" and row.get("result_path"):  # неполный/отвергнутый анализ — только для UI
        try:
            details = json.loads(Path(row["result_path"]).read_text(encoding="utf-8"))
            body["partial"] = _strip_frames(details)
        except (OSError, ValueError):
            pass
    return JSONResponse(body, headers={"Cache-Control": "no-store"})


def _strip_frames(d: Any) -> Any:
    """Подробности для UI без base64-кадров: ответ остаётся маленьким (лимит 3 МБ)."""
    if isinstance(d, dict):
        return {k: ("<jpeg omitted>" if k == "image_base64" else _strip_frames(v)) for k, v in d.items()}
    if isinstance(d, list):
        return [_strip_frames(v) for v in d]
    return d


async def _analyze(request: Request, principal: str) -> Response:
    import anyio
    from starlette.requests import ClientDisconnect

    ctx = _ctx(request)
    t_req = time.time()
    s = ctx.s
    if ctx.store.count_active() >= s.max_queue:
        return JSONResponse({"detail": f"queue is full ({s.max_queue} active jobs), retry later"},
                            status_code=503, headers={"Retry-After": "30"})
    try:
        free = shutil.disk_usage(s.jobs_dir).free
    except OSError:
        free = 1 << 62
    if free < s.min_free_disk_mb * 1024 * 1024:
        return JSONResponse({"detail": "insufficient disk space"}, status_code=507)
    job_id = uuid.uuid4().hex
    jd = s.jobs_dir / job_id
    jd.mkdir(parents=True, exist_ok=False)
    try:
        up = await receive_video(request.stream(), _headers(request), jd, s.max_upload_bytes,
                                 anyio.to_thread.run_sync,
                                 query_filename=request.query_params.get("filename", ""))
        vmeta = await anyio.to_thread.run_sync(validate_video, up.path, s.video_validation)
    except UploadError as e:
        shutil.rmtree(jd, ignore_errors=True)
        log.info("upload rejected", extra={"status": e.status, "reason": e.detail})
        return JSONResponse({"detail": e.detail}, status_code=e.status)
    except ClientDisconnect:
        shutil.rmtree(jd, ignore_errors=True)
        return JSONResponse({"detail": "client disconnected during upload"}, status_code=400)
    except BaseException:
        shutil.rmtree(jd, ignore_errors=True)
        raise
    vmeta.update(filename_normalized=normalize_filename(up.filename), size_bytes=up.size,
                 upload_s=round(time.time() - t_req, 2))
    force = request.query_params.get("force", "").lower() in ("1", "true", "yes")
    await anyio.to_thread.run_sync(ctx.manager._resolved.wait, 5.0)
    dkey = f"{up.sha256}:{ctx.manager.pipeline_tag()}"
    async with ctx.dedup_lock:
        if s.dedup and not force:
            prev = ctx.store.find_by_dedup(dkey)
            if prev is not None:  # тот же файл и та же версия пайплайна -> тот же результат
                shutil.rmtree(jd, ignore_errors=True)
                log.info("duplicate upload -> existing job", extra={"job_id": prev["id"]})
                st = "processing" if prev["status"] in ("queued", "running") else prev["status"]
                return JSONResponse({"job_id": prev["id"], "status": st, "deduplicated": True})
        row = ctx.store.create(job_id, video_path=str(up.path), filename=up.filename,
                               size_bytes=up.size, sha256=up.sha256, client=principal,
                               video_meta=vmeta, dedup_key=dkey, now=t_req)  # бюджет — от приёма
        ctx.manager.submit(job_id, queued_at=row.get("queued_at", t_req))
    log.info("job queued", extra={"job_id": job_id, "size": up.size, "principal": principal,
                                   "duration": vmeta.get("duration")})
    return JSONResponse({"job_id": job_id, "status": "processing"})


def _resolve_copilot(ctx: Ctx) -> Any:
    if ctx.copilot_checked:
        return ctx.copilot
    ctx.copilot_checked = True
    try:
        import importlib
        mod = importlib.import_module("copilot.router")
    except Exception as e:
        log.info("copilot not available", extra={"error": f"{type(e).__name__}: {e}"[:200]})
        return None
    for name in ("ask", "answer", "handle_ask", "answer_question"):
        fn = getattr(mod, name, None)
        if callable(fn):
            ctx.copilot = fn
            break
    return ctx.copilot


async def _call_copilot(fn: Any, payload: dict[str, Any], timeout: float) -> Any:
    import anyio
    params = inspect.signature(fn).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        args, kwargs = (), payload
    else:
        kwargs = {k: v for k, v in payload.items() if k in params}
        args = () if (kwargs or not params) else (payload,)
    with anyio.fail_after(timeout):
        if inspect.iscoroutinefunction(fn):
            return await fn(*args, **kwargs)
        return await anyio.to_thread.run_sync(lambda: fn(*args, **kwargs))


def build_jobs_router(auth: Callable[[str], Callable[..., Any]], ui: bool) -> APIRouter:
    """Одинаковые ручки для /api/v1 (X-API-Key) и /ui/v1 (cookie + CSRF)."""
    r = APIRouter()
    tag = ["ui"] if ui else ["api"]

    @r.post("/analyze", tags=tag, openapi_extra=_MULTIPART_DOC,
            summary="Загрузить видео (multipart поле file) -> job_id")
    async def analyze(request: Request, principal: str = Depends(auth("upload"))) -> Response:
        return await _analyze(request, principal)

    @r.get("/jobs", tags=tag, summary="Последние задачи")
    async def list_jobs(request: Request, limit: int = 50, _: str = Depends(auth("read"))) -> Any:
        ctx = _ctx(request)
        out = []
        for row in ctx.store.recent(max(1, min(200, limit))):
            out.append({"job_id": row["id"], "status": "processing" if row["status"] in
                        ("queued", "running") else row["status"], "stage": row.get("stage"),
                        "filename": row.get("filename"), "created_at": row["created_at"],
                        "finished_at": row.get("finished_at")})
        return {"jobs": out}

    @r.get("/jobs/{job_id}", tags=tag, summary="Статус/результат задачи (схема ТЗ)")
    async def get_job(job_id: str, request: Request, _: str = Depends(auth("read"))) -> Response:
        ctx = _ctx(request)
        return _job_payload(ctx, _job_row(ctx, job_id), _profile(ctx, request, ui))

    @r.api_route("/jobs/{job_id}/video", methods=["GET", "HEAD"], tags=tag,
                 summary="Исходное видео с поддержкой HTTP Range")
    async def get_video(job_id: str, request: Request, original: bool = False,
                        _: str = Depends(auth("read"))) -> Response:
        ctx = _ctx(request)
        row = _job_row(ctx, job_id)
        jd = ctx.manager.job_dir(job_id)
        path = jd / "preview.mp4"
        if original or not path.is_file():
            path = Path(row.get("video_path") or "")
        if not path.is_file():
            raise HTTPException(404, "video not available")
        mt = "video/mp4" if (ctx.s.mov_as_mp4 or path.suffix.lower() != ".mov") else None
        return range_response(path, request.headers.get("range"), request.method,
                              media_type=mt if path.suffix.lower() in (".mov", ".mp4", ".m4v") else None)

    @r.get("/jobs/{job_id}/frames/{index}.jpg", tags=tag, summary="Кадр-доказательство №index",
           response_class=Response)
    async def get_frame(job_id: str, index: int, request: Request,
                        _: str = Depends(auth("read"))) -> Response:
        ctx = _ctx(request)
        row = _job_row(ctx, job_id)
        p = ctx.manager.job_dir(job_id) / "frames" / f"{int(index)}.jpg"
        if index >= 0 and p.is_file():
            return FileResponse(p, media_type="image/jpeg",
                                headers={"Cache-Control": "private, max-age=86400"})
        if row["status"] == "done" and row.get("result_path"):
            import base64
            try:
                data = json.loads(_result_bytes(ctx, row))
                for i, _vi, f in iter_frames(data):
                    if i == index:
                        return Response(base64.b64decode(f["image_base64"]),
                                        media_type="image/jpeg")
            except (OSError, ValueError):
                pass
        raise HTTPException(404, "frame not found")

    @r.get("/jobs/{job_id}/observations", tags=tag,
           summary="Наблюдения правил по последовательностям: скелеты, предметы, зоны (расширение контракта)")
    async def get_observations(job_id: str, request: Request, _: str = Depends(auth("read"))) -> Response:
        """Только то, что использовали правила ml.seqrules (профиль pose_rules): meta + эпизоды с кадрами.
        Нет файла → 404 (профиль без seqrules или анализ не дошёл до стадии), а не пустой «всё чисто»."""
        ctx = _ctx(request)
        _job_row(ctx, job_id)
        p = ctx.manager.job_dir(job_id) / "observations.json"
        if not p.is_file():
            raise HTTPException(404, "observations not available for this job")
        return FileResponse(p, media_type="application/json", headers={"Cache-Control": "no-store"})

    @r.post("/jobs/{job_id}/ask", tags=tag, summary="Вопрос локальному ассистенту (copilot)")
    async def ask(job_id: str, request: Request, _: str = Depends(auth("ask"))) -> Any:
        ctx = _ctx(request)
        row = _job_row(ctx, job_id)
        fn = _resolve_copilot(ctx)
        if fn is None:
            raise HTTPException(501, "copilot is not available on this server")
        raw = await request.body()
        if len(raw) > 64 * 1024:
            raise HTTPException(413, "question too large")
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            raise HTTPException(400, "body must be JSON: {\"question\": \"...\"}") from None
        if not isinstance(body, dict) or not str(body.get("question") or "").strip():
            raise HTTPException(400, "field 'question' is required")
        result = None
        if row["status"] == "done" and row.get("result_path"):
            try:
                result = json.loads(_result_bytes(ctx, row))
            except (OSError, ValueError):
                result = None
        payload = {**body, "job_id": job_id, "question": str(body["question"]),
                   "history": body.get("history") or [], "result": result,
                   "job_dir": str(ctx.manager.job_dir(job_id)), "video_path": row.get("video_path")}
        try:
            out = await _call_copilot(fn, payload, ctx.s.copilot_timeout_s)
        except TimeoutError:
            raise HTTPException(504, "copilot timeout") from None
        except Exception as e:
            log.error("copilot failed", extra={"job_id": job_id, "error": repr(e)[:300]})
            raise HTTPException(500, f"copilot error: {type(e).__name__}") from None
        if isinstance(out, str):
            return {"job_id": job_id, "answer": out}
        from .results import jsonable
        return jsonable(out)

    @r.get("/catalog", tags=tag, summary="Каталог типов нарушений и норм (norms/catalog.yaml)")
    async def catalog(request: Request, _: str = Depends(auth("read"))) -> Any:
        try:
            from norms import load_catalog  # type: ignore
            from .results import jsonable
            return jsonable(load_catalog())
        except Exception as e:
            raise HTTPException(503, f"norms catalog unavailable ({type(e).__name__})") from None

    return r


def job_for_copilot(app: FastAPI, job_id: str) -> Optional[dict[str, Any]]:
    """Провайдер для copilot.make_router (A14): JSON ответа ТЗ, None если задачи нет."""
    ctx = getattr(app.state, "ctx", None)
    if ctx is None or not _JOB_ID.match(job_id or ""):
        return None
    row = ctx.store.get(job_id)
    if not row:
        return None
    if row["status"] in ("queued", "running"):
        return {"job_id": job_id, "status": "processing", "stage": row.get("stage")}
    if row["status"] == "done" and row.get("result_path"):
        try:
            return json.loads(_result_bytes(ctx, row))
        except (OSError, ValueError):
            pass
    return {"job_id": job_id, "status": "failed", "verdict": None, "violations": [],
            "error": row.get("error") or "analysis failed"}


def _mount_copilot(app: FastAPI) -> bool:
    """A14 отдаёт фабрику copilot.router.make_router(get_job, prefix, dependencies).
    Монтируем её дважды: /api/v1 под X-API-Key и /ui/v1 под cookie+CSRF. Нет модуля — 501."""
    try:
        from copilot.router import make_router  # type: ignore
    except Exception:
        return False
    provider = lambda job_id: job_for_copilot(app, job_id)  # noqa: E731
    try:
        app.include_router(make_router(get_job=provider, prefix="/api/v1",
                                       dependencies=[Depends(require_api_key)]))
        app.include_router(make_router(get_job=provider, prefix="/ui/v1",
                                       dependencies=[Depends(_ui_dependency("ask"))]))
    except Exception as e:
        log.warning("copilot router not mounted", extra={"error": f"{type(e).__name__}: {e}"[:200]})
        return False
    return True


# ------------------------------------------------------------------ app factory
def create_app(settings: Optional[Settings] = None,
               pipeline: Optional[PipelineSpec | Callable[[], PipelineSpec]] = None) -> FastAPI:
    """Ничего не запускает: потоки, БД и прогрев стартуют в lifespan."""
    s = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging(s.log_level, s.log_json)
        s.jobs_dir.mkdir(parents=True, exist_ok=True)
        keys, note = load_api_keys(s.api_keys, s.api_keys_file)
        if note:
            log.warning(note)
        store = JobStore(s.db_path)
        manager = JobManager(s, store, pipeline)
        import asyncio
        ctx = Ctx(s=s, store=store, manager=manager, keys=ApiKeyAuth(keys),
                  signer=UiTokenSigner(load_ui_secret(s.ui_secret, s.data_dir), s.ui_token_ttl_s),
                  dedup_lock=asyncio.Lock())
        app.state.ctx = ctx
        manager.start()
        log.info("api started", extra={"workers": s.workers, "gpu_slots": s.gpu_slots,
                                        "keys": len(ctx.keys), "data_dir": str(s.data_dir),
                                        "ui_password": bool(s.ui_password)})
        try:
            yield
        finally:
            manager.stop()
            store.close()
            app.state.ctx = None

    app = FastAPI(title="Kassa Sentinel API", version=API_VERSION, lifespan=lifespan,
                  docs_url="/docs" if s.docs else None, redoc_url=None,
                  openapi_url="/openapi.json" if s.docs else None,
                  description="Локальный анализ видео кассы. Результат — подсказка проверяющему, "
                              "не юридическое заключение.")
    app.state.ctx = None

    if s.cors_origins:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_credentials=True,
                           allow_methods=["GET", "POST", "DELETE", "HEAD", "OPTIONS"],
                           allow_headers=["X-API-Key", "X-CSRF-Token", "Range", "Content-Type"],
                           expose_headers=["Content-Range", "Accept-Ranges", "X-Request-ID"])

    def _scope_ip(scope: dict) -> str:
        peer = (scope.get("client") or ("unknown", 0))[0]
        hdrs = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers") or []}
        return client_ip(peer, hdrs, s.trusted_proxies)

    app.add_middleware(RequestContextMiddleware, ip_resolver=_scope_ip)

    api_auth = lambda bucket: require_api_key  # noqa: E731
    _mount_copilot(app)  # до наших ручек: его /jobs/{id}/ask приоритетнее нашего 501-fallback
    app.include_router(build_jobs_router(api_auth, ui=False), prefix="/api/v1")
    app.include_router(build_jobs_router(_ui_dependency, ui=True), prefix="/ui/v1")

    # ---------------- UI-сессия: вход по API-ключу или паролю -> HttpOnly cookie
    @app.post("/ui/v1/session", tags=["ui"], summary="Вход в UI: {api_key} или {password}")
    async def ui_login(request: Request) -> Response:
        ctx = _ctx(request)
        if not s.ui_enabled:
            raise HTTPException(404, "UI access disabled")
        _limit(ctx, request, "ui_login", s.rl_ui_login)
        if cross_site(_headers(request), s.cors_origins):
            raise HTTPException(403, "cross-site request rejected")
        try:
            body = json.loads((await request.body())[:8192] or b"{}")
        except ValueError:
            body = {}
        if not isinstance(body, dict):
            body = {}
        key = body.get("api_key") or body.get("key")
        ok = (isinstance(key, str) and ctx.keys.check(key)) or \
             (isinstance(body.get("password"), str) and check_secret(body["password"], s.ui_password))
        if not ok:
            log.info("ui login failed", extra={"ip": _ip(request, s)})
            raise HTTPException(401, "invalid credentials")
        token, exp = ctx.signer.issue()
        resp = JSONResponse({"authenticated": True, "expires_at": exp,
                             "csrf_token": ctx.signer.csrf_for(token)})
        resp.set_cookie(SESSION_COOKIE, token, max_age=int(s.ui_token_ttl_s), httponly=True,
                        samesite="strict", secure=_is_https(request, s), path="/ui/v1")
        return resp

    @app.get("/ui/v1/session", tags=["ui"], summary="Проверить UI-сессию, получить CSRF-токен")
    async def ui_session(request: Request) -> Response:
        ctx = _ctx(request)
        _limit(ctx, request, "ui_session", s.rl_ui_session)
        token = request.cookies.get(SESSION_COOKIE, "")
        exp = ctx.signer.verify(token)
        if exp is None:
            return JSONResponse({"authenticated": False,
                                 "login": "POST /ui/v1/session {\"api_key\"|\"password\": \"...\"}",
                                 "password_login": bool(s.ui_password)}, status_code=401)
        return JSONResponse({"authenticated": True, "expires_at": exp,
                             "csrf_token": ctx.signer.csrf_for(token)},
                            headers={"Cache-Control": "no-store"})

    @app.delete("/ui/v1/session", tags=["ui"], summary="Выход")
    async def ui_logout() -> Response:
        resp = JSONResponse({"authenticated": False})
        resp.delete_cookie(SESSION_COOKIE, path="/ui/v1")
        return resp

    # ---------------- health
    @app.get("/healthz", tags=["ops"])
    async def healthz() -> Any:
        return {"status": "ok", "version": API_VERSION}

    @app.get("/readyz", tags=["ops"])
    async def readyz(request: Request) -> Response:
        ctx = getattr(request.app.state, "ctx", None)
        if ctx is None:
            return JSONResponse({"ready": False, "state": "starting"}, status_code=503)
        m = ctx.manager
        body = {"ready": m.ready, "state": m.state,
                "pipeline": m.pipeline.mode if m.pipeline else "loading",
                "pipeline_version": (m.pipeline.version or None) if m.pipeline else None,
                "queue": m.stats(), "version": API_VERSION,
                "jury_profile": s.jury_profile, "partial_accept": s.partial_accept}
        if m.state_detail:
            body["detail"] = m.state_detail
        hung = m.gpu_hung()
        if hung is not None:
            body["state"] = "gpu_hung"
            body["detail"] = (f"pipeline.run has held the GPU {hung:.0f}s after its job timed out; "
                              "restart the service")
        return JSONResponse(body, status_code=200 if m.ready else 503)

    # ---------------- фронт: frontend/dist со SPA-fallback, иначе встроенная страница
    dist = s.frontend_dir
    fallback_page = API_DIR / "static" / "index.html"

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str) -> Response:
        if full_path.startswith(("api/", "ui/v1/")) or full_path in ("api", "ui/v1"):
            raise HTTPException(404, "not found")
        if dist.is_dir() and (dist / "index.html").is_file():
            if full_path:
                cand = (dist / full_path).resolve()
                try:
                    cand.relative_to(dist.resolve())
                except ValueError:
                    raise HTTPException(404, "not found") from None
                if cand.is_file():
                    cache = ("public, max-age=31536000, immutable" if "/assets/" in f"/{full_path}"
                             else "no-cache")
                    return FileResponse(cand, headers={"Cache-Control": cache})
            return FileResponse(dist / "index.html", headers={"Cache-Control": "no-cache"})
        if fallback_page.is_file() and full_path in ("", "index.html"):
            return HTMLResponse(fallback_page.read_text(encoding="utf-8"),
                                headers={"Cache-Control": "no-cache"})
        if full_path == "":
            return HTMLResponse("<h1>Kassa Sentinel API</h1><p>UI не собран. См. <a href='/docs'>"
                                "/docs</a>.</p>")
        raise HTTPException(404, "not found")

    return app


app = create_app()
