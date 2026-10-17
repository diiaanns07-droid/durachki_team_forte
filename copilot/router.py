"""FastAPI-роутер Sentinel Copilot (подключает A02 лениво).

    from copilot.router import make_router
    app.include_router(make_router(get_job=jobs.get_result), dependencies=[Depends(require_api_key)])

Эндпоинты (префикс /api/v1):
    GET  /copilot/status              -> {"enabled", "available", "backend", "reason", ...}  (UI скрывает панель, если available=false)
    POST /jobs/{job_id}/ask           -> {"answer", "citations", "grounded", ...}; ?stream=1 или Accept: text/event-stream -> SSE
    GET  /jobs/{job_id}/report        -> text/markdown «заключение для проверяющего» (без LLM); ?format=html
    GET  /jobs/{job_id}/suggestions   -> {"suggestions": [...], "by_type": {...}} — вопросы-подсказки для UI

Коды: 404 — задачи нет; 409 — анализ не завершён; 501 — copilot выключен/модели нет;
503 — GPU занят / провайдер задач не настроен; 504 — таймаут генерации.
Результат задачи берётся через инъецируемый провайдер ``get_job(job_id) -> dict | AnalysisResult | None``
(может быть async) — роутер не зависит от внутренностей API.
"""
from __future__ import annotations

import inspect
import json
from typing import Any, Awaitable, Callable, Literal, Optional, Sequence, Union

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import AliasChoices, BaseModel, Field
from starlette.concurrency import run_in_threadpool

JobProvider = Callable[[str], Union[Any, Awaitable[Any]]]

ERROR_STATUS = {
    "disabled": 501,
    "unavailable": 501,
    "not_ready": 409,
    "empty_question": 422,
    "busy": 503,
    "timeout": 504,
    "llm_error": 500,
    "internal_error": 500,
}

_job_provider: Optional[JobProvider] = None


def set_job_provider(fn: Optional[JobProvider]) -> None:
    """Зарегистрировать провайдер результата задачи для модульного ``router``."""
    global _job_provider
    _job_provider = fn


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    # A15 шлёт {role, text}; OpenAI-стиль — {role, content}. Принимаем оба.
    content: str = Field(default="", max_length=8000, validation_alias=AliasChoices("content", "text"))
    job_id: Optional[str] = Field(default=None, max_length=200)  # реплики другой задачи отбрасываются


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=40)


def _job_id_of(job: Any) -> Optional[str]:
    if isinstance(job, (str, bytes)):
        try:
            job = json.loads(job)
        except ValueError:
            return None
    jid = job.get("job_id") if isinstance(job, dict) else getattr(job, "job_id", None)
    return None if jid is None else str(jid)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def make_router(get_job: Optional[JobProvider] = None, service: Any = None, prefix: str = "/api/v1",
                dependencies: Optional[Sequence[Any]] = None) -> APIRouter:
    router = APIRouter(prefix=prefix, tags=["copilot"], dependencies=list(dependencies or []))

    def _svc() -> Any:
        if service is not None:
            return service
        from .service import get_service

        return get_service()

    async def _load(job_id: str) -> Any:
        provider = get_job or _job_provider
        if provider is None:
            raise HTTPException(status_code=503, detail={"code": "no_job_provider",
                                                         "message": "Провайдер результатов задач не настроен"})
        try:
            job = provider(job_id)
            if inspect.isawaitable(job):
                job = await job
        except HTTPException:
            raise
        except Exception as exc:  # сбой хранилища задач не превращаем в 500 без тела
            raise HTTPException(status_code=503, detail={"code": "job_provider_error",
                                                         "message": f"Не удалось получить результат задачи: {type(exc).__name__}"}) from exc
        if job is None or _job_id_of(job) not in (None, "", job_id):
            # провайдер вернул чужую задачу — не отвечаем по ней (не раскрываем её существование)
            raise HTTPException(status_code=404, detail={"code": "job_not_found", "message": "Задача не найдена"})
        return job

    def _status_of(job: Any) -> Optional[str]:
        return job.get("status") if isinstance(job, dict) else getattr(job, "status", None)

    @router.get("/copilot/status")
    async def copilot_status() -> dict[str, Any]:
        return await run_in_threadpool(lambda: _svc().status())

    @router.post("/jobs/{job_id}/ask")
    async def ask(job_id: str, body: AskRequest, request: Request,
                  stream: bool = Query(False, description="SSE-стриминг токенов")) -> Any:
        job = await _load(job_id)
        history = [t.model_dump(exclude_none=True) for t in body.history]
        want_sse = stream or "text/event-stream" in request.headers.get("accept", "")
        svc = _svc()
        if want_sse:
            gen = svc.stream(job, body.question, history)
            first = await run_in_threadpool(next, gen, None)
            if first is None or first.get("event") == "error":
                # ошибка до первого токена (выключен / нет модели / не готово) — обычный HTTP-код, не 200
                detail = dict(first or {"error": "llm_error"})
                detail.pop("event", None)
                return JSONResponse(status_code=ERROR_STATUS.get(str(detail.get("error")), 500),
                                    content={"detail": detail})

            def events() -> Any:
                ev = dict(first)
                yield _sse(ev.pop("event", "message"), ev)
                for ev in gen:
                    name = ev.pop("event", "message")
                    yield _sse(name, ev)
            return StreamingResponse(events(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        result = await run_in_threadpool(svc.answer, job, body.question, history)
        code = result.get("error")
        if code:
            return JSONResponse(status_code=ERROR_STATUS.get(code, 500), content={"detail": result})
        return result

    @router.get("/jobs/{job_id}/report", response_class=PlainTextResponse)
    async def job_report(job_id: str, format: str = Query("md", pattern="^(md|html)$")) -> Any:
        job = await _load(job_id)
        if _status_of(job) == "processing":
            raise HTTPException(status_code=409, detail={"code": "not_ready", "message": "Анализ ещё не завершён"})
        svc = _svc()

        def _render() -> str:
            from .report import report

            return report(job, corpus=svc.corpus, cfg=svc.cfg, fmt=format)

        text = await run_in_threadpool(_render)
        if format == "html":
            return HTMLResponse(text, headers={"Content-Security-Policy": "default-src 'none'; img-src data:; "
                                                                          "style-src 'unsafe-inline'"})
        return PlainTextResponse(text, media_type="text/markdown; charset=utf-8")

    @router.get("/jobs/{job_id}/suggestions")
    async def job_suggestions(job_id: str, limit: int = Query(4, ge=1, le=8)) -> dict[str, Any]:
        """3–4 вопроса-подсказки для панели (по типам нарушений в результате). Работает без модели."""
        job = await _load(job_id)
        from .suggestions import suggestions

        return await run_in_threadpool(suggestions, job, limit)

    return router


# модульный роутер для ленивого подключения: провайдер задаётся set_job_provider()
router = make_router()
