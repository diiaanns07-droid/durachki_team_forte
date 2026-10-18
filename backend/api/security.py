"""API-ключи, подписанные UI-токены, same-origin проверка, rate-limit, IP клиента."""
from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlsplit

KEY_PREFIX = "ks_"


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def fingerprint(key: str) -> str:
    """Короткий отпечаток для логов (сам ключ в лог не пишем)."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]


def read_keys_file(path: Optional[Path]) -> list[str]:
    if not path or not path.is_file():
        return []
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def write_private(path: Path, text: str, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if append else "w", encoding="utf-8") as fh:
        fh.write(text)
    try:
        os.chmod(path, 0o600)
    except OSError:  # Windows
        pass


class ApiKeyAuth:
    """Набор допустимых ключей; сравнение только hmac.compare_digest по всем ключам."""

    def __init__(self, keys: Iterable[str]):
        self._keys = [k.encode("utf-8") for k in dict.fromkeys(k.strip() for k in keys) if k]

    def __len__(self) -> int:
        return len(self._keys)

    def check(self, presented: Optional[str]) -> bool:
        if not presented or not self._keys:
            return False
        p = presented.strip().encode("utf-8")
        ok = False
        for k in self._keys:  # без раннего выхода
            ok |= hmac.compare_digest(p, k)
        return ok


def load_api_keys(env_keys: list[str], keys_file: Optional[Path]) -> tuple[list[str], Optional[str]]:
    """Ключи из env + файла. Если нет ни одного — генерируем и кладём в файл (0600).
    Возвращает (ключи, сообщение-для-лога или None)."""
    keys = list(env_keys) + read_keys_file(keys_file)
    if keys:
        return keys, None
    if keys_file is None:
        return [], "no API keys configured: /api/v1 is closed"
    key = generate_key()
    write_private(keys_file, f"# Kassa Sentinel API keys, one per line\n{key}\n")
    return [key], f"no SENTINEL_API_KEYS: generated a key into {keys_file} (fp={fingerprint(key)})"


# ---------------------------------------------------------------- UI session
def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def check_secret(presented: Optional[str], expected: str) -> bool:
    """Сравнение пароля UI (пустой ожидаемый = вход по паролю выключен)."""
    if not presented or not expected:
        return False
    return hmac.compare_digest(presented.strip().encode("utf-8"), expected.encode("utf-8"))


class UiTokenSigner:
    """Сессия UI: ui1.<exp>.<nonce>.<hmac-sha256>. Выдаётся ТОЛЬКО после проверки API-ключа или
    пароля и живёт исключительно в HttpOnly SameSite=Strict cookie (JS её не видит).
    CSRF-токен выводится из сессии и обязателен в X-CSRF-Token для POST/DELETE."""

    def __init__(self, secret: bytes, ttl_s: float):
        self._secret = secret
        self.ttl_s = float(ttl_s)

    def _sig(self, body: str) -> str:
        return _b64(hmac.new(self._secret, body.encode("ascii"), hashlib.sha256).digest())

    def issue(self, now: Optional[float] = None) -> tuple[str, int]:
        exp = int((time.time() if now is None else now) + self.ttl_s)
        body = f"ui1.{exp}.{secrets.token_urlsafe(12)}"
        return f"{body}.{self._sig(body)}", exp

    def verify(self, token: Optional[str], now: Optional[float] = None) -> Optional[int]:
        """exp (unix) если токен валиден, иначе None."""
        if not token or len(token) > 200:
            return None
        parts = token.strip().split(".")
        if len(parts) != 4 or parts[0] != "ui1":
            return None
        body = ".".join(parts[:3])
        if not hmac.compare_digest(self._sig(body), parts[3]):
            return None
        try:
            exp = int(parts[1])
        except ValueError:
            return None
        return exp if exp >= (time.time() if now is None else now) else None

    def csrf_for(self, token: str) -> str:
        return _b64(hmac.new(self._secret, b"csrf|" + token.encode("ascii"),
                             hashlib.sha256).digest())[:32]

    def check_csrf(self, token: str, presented: Optional[str]) -> bool:
        return bool(presented) and hmac.compare_digest(self.csrf_for(token), presented.strip())


def load_ui_secret(configured: str, data_dir: Path) -> bytes:
    if configured:
        return configured.encode("utf-8")
    path = data_dir / "ui_secret"
    try:
        if path.is_file():
            s = path.read_text(encoding="utf-8").strip()
            if len(s) >= 32:
                return s.encode("utf-8")
        s = secrets.token_urlsafe(48)
        write_private(path, s)
        return s.encode("utf-8")
    except OSError:
        return secrets.token_bytes(48)


# ---------------------------------------------------------------- origin / ip
def _netloc(url: str) -> str:
    try:
        u = urlsplit(url)
        port = u.port
    except ValueError:
        return ""
    host = (u.hostname or "").lower()
    if not host:
        return ""
    default = {"http": 80, "https": 443}.get(u.scheme)
    return host if port in (None, default) else f"{host}:{port}"


def _host_netloc(host_header: str) -> str:
    h = (host_header or "").strip().lower()
    if h.endswith(":80") or h.endswith(":443"):
        h = h.rsplit(":", 1)[0]
    return h


def cross_site(headers: dict[str, str], allowed_origins: Iterable[str]) -> bool:
    """True, если браузер явно сообщает о кросс-сайтовом запросе. Это защита от CSRF
    (дополнительно к SameSite=Strict и X-CSRF-Token), а НЕ аутентификация."""
    sfs = headers.get("sec-fetch-site", "").lower()
    if sfs and sfs not in ("same-origin", "none"):
        return True
    allowed = {_netloc(o) for o in allowed_origins if o}
    own = {_host_netloc(headers.get("host", ""))}
    xfh = headers.get("x-forwarded-host", "")
    if xfh:
        own.add(_host_netloc(xfh.split(",")[0]))
    own.discard("")
    for h in ("origin", "referer"):
        v = headers.get(h, "")
        if v and v != "null":
            n = _netloc(v)
            return not (n and (n in own or n in allowed))
    return False


def client_ip(peer: Optional[str], headers: dict[str, str], trusted: Iterable[str]) -> str:
    """IP клиента. Заголовкам прокси верим только если пир — доверенный (cloudflared на localhost)."""
    peer = peer or "unknown"
    if ip_in(peer, trusted):
        for h in ("cf-connecting-ip", "x-real-ip"):
            v = headers.get(h, "").strip()
            if v:
                return v
        xff = headers.get("x-forwarded-for", "")
        if xff:
            return xff.split(",")[0].strip() or peer
    return peer


def ip_in(ip: str, nets: Iterable[str]) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return ip in set(nets)
    for n in nets:
        try:
            if addr in ipaddress.ip_network(n, strict=False):
                return True
        except ValueError:
            continue
    return False


# ---------------------------------------------------------------- rate limit
class RateLimiter:
    """Token bucket по (bucket, ip). Память чистится от давно неактивных ключей."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._b: dict[tuple[str, str], tuple[float, float]] = {}
        self._last_prune = time.monotonic()

    def hit(self, bucket: str, key: str, limit: int, per: float,
            now: Optional[float] = None) -> tuple[bool, float]:
        if limit <= 0:
            return True, 0.0
        now = time.monotonic() if now is None else now
        rate = limit / per
        with self._lock:
            tokens, last = self._b.get((bucket, key), (float(limit), now))
            tokens = min(float(limit), tokens + (now - last) * rate)
            if tokens >= 1.0:
                self._b[(bucket, key)] = (tokens - 1.0, now)
                allowed, retry = True, 0.0
            else:
                self._b[(bucket, key)] = (tokens, now)
                allowed, retry = False, (1.0 - tokens) / rate
            if now - self._last_prune > 300:
                self._prune(now)
        return allowed, retry

    def _prune(self, now: float) -> None:
        self._last_prune = now
        for k in [k for k, (_, last) in self._b.items() if now - last > 3600]:
            del self._b[k]
