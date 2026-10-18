"""Минимальный HTTP-клиент на stdlib (без requests): multipart, Range, cookie ks_session."""
from __future__ import annotations

import http.client
import json
import os
import socket
import ssl
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlsplit


@dataclass
class Resp:
    status: int
    headers: dict[str, str]
    body: bytes
    seconds: float
    cookies: dict[str, str] = field(default_factory=dict)

    def json(self) -> Any:
        try:
            return json.loads(self.body or b"null")
        except ValueError:
            return None


def api_key_from_env() -> str:
    """Ключ — только из окружения (не из argv: он виден в списке процессов и истории shell)."""
    k = os.environ.get("SENTINEL_API_KEY", "").strip()
    if not k:
        k = os.environ.get("SENTINEL_API_KEYS", "").split(",")[0].strip()
    return k


class Client:
    def __init__(self, base_url: str, timeout: float = 120.0):
        u = urlsplit(base_url.rstrip("/"))
        self.scheme, self.host = u.scheme or "http", u.hostname or "127.0.0.1"
        self.port = u.port or (443 if self.scheme == "https" else 80)
        self.prefix = u.path.rstrip("/")
        self.timeout = timeout
        self.cookies: dict[str, str] = {}

    def _conn(self) -> http.client.HTTPConnection:
        if self.scheme == "https":
            return http.client.HTTPSConnection(self.host, self.port, timeout=self.timeout,
                                               context=ssl.create_default_context())
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def request(self, method: str, path: str, body: Any = None,
                headers: Optional[dict[str, str]] = None, use_cookies: bool = True) -> Resp:
        h = dict(headers or {})
        if use_cookies and self.cookies:
            h["Cookie"] = "; ".join(f"{k}={v}" for k, v in self.cookies.items())
        if self.scheme == "https" or self.port not in (80, 443):
            h.setdefault("Host", f"{self.host}:{self.port}")
        t0 = time.perf_counter()
        c = self._conn()
        try:
            c.request(method, self.prefix + path, body=body, headers=h)
            r = c.getresponse()
            data = r.read()
            hdrs = {k.lower(): v for k, v in r.getheaders()}
            got: dict[str, str] = {}
            for k, v in r.getheaders():
                if k.lower() == "set-cookie":
                    name, _, rest = v.partition("=")
                    got[name.strip()] = rest.split(";", 1)[0]
                    hdrs.setdefault("set-cookie-all", "")
                    hdrs["set-cookie-all"] += v + "\n"
            if use_cookies:
                for k, v in got.items():
                    if v:
                        self.cookies[k] = v
                    else:
                        self.cookies.pop(k, None)
            return Resp(r.status, hdrs, data, time.perf_counter() - t0, got)
        finally:
            c.close()

    def get(self, path: str, headers: Optional[dict[str, str]] = None, **kw: Any) -> Resp:
        return self.request("GET", path, None, headers, **kw)

    def post_json(self, path: str, obj: Any, headers: Optional[dict[str, str]] = None) -> Resp:
        return self.request("POST", path, json.dumps(obj).encode(),
                            {"Content-Type": "application/json", **(headers or {})})

    def post_file(self, path: str, data: bytes, filename: str, headers: Optional[dict[str, str]] = None,
                  ctype: str = "video/quicktime") -> Resp:
        b = uuid.uuid4().hex
        disp = f'form-data; name="file"; filename="{filename}"'.encode("utf-8")
        body = (f"--{b}\r\n".encode() + b"Content-Disposition: " + disp + b"\r\n" +
                f"Content-Type: {ctype}\r\n\r\n".encode() + data + f"\r\n--{b}--\r\n".encode())
        return self.request("POST", path, body,
                            {"Content-Type": f"multipart/form-data; boundary={b}", **(headers or {})})

    def raw(self, payload: bytes, read: bool = True, close_after_send: bool = False) -> bytes:
        """Сырые байты в сокет (для «обрыва загрузки» и «объявлен огромный Content-Length»)."""
        s = socket.create_connection((self.host, self.port), timeout=self.timeout)
        if self.scheme == "https":
            s = ssl.create_default_context().wrap_socket(s, server_hostname=self.host)
        try:
            s.sendall(payload)
            if close_after_send:
                return b""
            out = b""
            if read:
                s.settimeout(min(self.timeout, 30))
                try:
                    while b"\r\n\r\n" not in out:
                        chunk = s.recv(65536)
                        if not chunk:
                            break
                        out += chunk
                except (socket.timeout, OSError):
                    pass
            return out
        finally:
            s.close()

    def host_header(self) -> str:
        return f"{self.host}:{self.port}"
