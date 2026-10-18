"""JSON-логи с request_id. Никакого содержимого видео, ключей, cookie и токенов в логах."""
from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
import time
import uuid
from typing import Any, Callable, Optional

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")

_STD = set(vars(logging.LogRecord("x", 0, "x", 0, "x", None, None))) | {"message", "asctime"}
_RID_OK = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_SECRETISH = re.compile(r"(key|token|secret|password|cookie|authorization)", re.I)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname, "logger": record.name, "msg": record.getMessage(),
        }
        rid = request_id_var.get()
        if rid:
            d["request_id"] = rid
        for k, v in record.__dict__.items():
            if k in _STD or k.startswith("_"):
                continue
            d[k] = "***" if _SECRETISH.search(k) else v
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d, ensure_ascii=False, default=str)


def setup_logging(level: str = "INFO", as_json: bool = True) -> None:
    """Идемпотентно вешает один handler на логгер sentinel.*"""
    logger = logging.getLogger("sentinel")
    logger.setLevel(level.upper())
    if any(getattr(h, "_sentinel", False) for h in logger.handlers):
        return
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(JsonFormatter() if as_json else
                   logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    h._sentinel = True  # type: ignore[attr-defined]
    logger.addHandler(h)
    logger.propagate = False


class RequestContextMiddleware:
    """Чистый ASGI (не буферизует стримы): X-Request-ID + строка access-лога в JSON."""

    def __init__(self, app: Callable[..., Any], ip_resolver: Optional[Callable[[dict], str]] = None):
        self.app = app
        self.ip_resolver = ip_resolver
        self.log = logging.getLogger("sentinel.api.access")

    async def __call__(self, scope: dict, receive: Callable[..., Any], send: Callable[..., Any]) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = ""
        for k, v in scope.get("headers") or []:
            if k == b"x-request-id":
                incoming = v.decode("latin-1")
                break
        rid = incoming if _RID_OK.match(incoming or "") else uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        scope.setdefault("state", {})["request_id"] = rid
        status = {"code": 500}
        t0 = time.perf_counter()

        async def _send(msg: dict) -> None:
            if msg["type"] == "http.response.start":
                status["code"] = msg["status"]
                msg.setdefault("headers", [])
                msg["headers"] = list(msg["headers"]) + [(b"x-request-id", rid.encode("ascii"))]
            await send(msg)

        try:
            await self.app(scope, receive, _send)
        finally:
            path = scope.get("path", "")
            if not path.startswith(("/assets/", "/favicon")):
                extra = {"method": scope.get("method"), "path": path, "status": status["code"],
                         "ms": round((time.perf_counter() - t0) * 1000, 1)}
                if self.ip_resolver:
                    try:
                        extra["ip"] = self.ip_resolver(scope)
                    except Exception:
                        pass
                self.log.info("request", extra=extra)
            request_id_var.reset(token)
