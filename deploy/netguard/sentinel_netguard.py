"""Процессный запрет исходящей сети для Python-кода (без изменения firewall ОС).

Аудит-хук (PEP 578) видит socket.connect / sendto / getaddrinfo / gethostbyname ЛЮБОЙ Python-библиотеки
(urllib, requests, httpx, huggingface_hub, aiohttp). Режимы (SENTINEL_NETGUARD):
  block — не-локальный адрес → PermissionError + запись в лог;
  log   — только запись (аудит «пытался ли код выйти в сеть»).
Всегда разрешено: loopback, unix-сокеты, адреса привязки (0.0.0.0, ::), SENTINEL_NETGUARD_ALLOW
(имена хостов и CIDR через запятую, напр. "llm,10.0.0.0/8").
Прокси (HTTP(S)_PROXY, на Windows — системный из реестра) — это тоже путь наружу, даже если он на
loopback: его адреса запрещены, а в block proxy-переменные снимаются (клиенты идут напрямую → видны хуку).
Ограничение: нативный код со своими сокетами хук не видит → сильная гарантия только у
`docker run --network none` (ядро). Лог: SENTINEL_NETGUARD_LOG (JSONL).
"""
from __future__ import annotations

import ipaddress
import json
import os
import sys
import tempfile
import threading
import time

_LOCAL_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost", "", "0.0.0.0", "::", "<broadcast>"}
_state: dict = {"installed": False, "mode": None, "log": None, "attempts": 0, "proxies": set()}
_PROXY_VARS = ("http_proxy", "https_proxy", "all_proxy", "ftp_proxy", "ws_proxy", "wss_proxy")
_lock = threading.Lock()


def _allowlist() -> tuple[set[str], list]:
    names, nets = set(), []
    for item in (os.environ.get("SENTINEL_NETGUARD_ALLOW") or "").split(","):
        item = item.strip().lower()
        if not item:
            continue
        try:
            nets.append(ipaddress.ip_network(item, strict=False))
        except ValueError:
            names.add(item.split(":")[0])
    return names, nets


def is_local(host: object) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("utf-8", "replace")
    h = str(host).strip("[]").lower().split("%")[0]
    if h in _LOCAL_NAMES:
        return True
    names, nets = _allowlist()
    if h in names:
        return True
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return False  # имя хоста не из allowlist → резолв уже попытка выхода наружу
    return ip.is_loopback or ip.is_unspecified or any(ip in n for n in nets)


def _record(event: str, target: str, action: str) -> None:
    _state["attempts"] += 1
    rec = {"ts": round(time.time(), 3), "pid": os.getpid(), "event": event, "target": target, "action": action}
    with _lock:
        try:
            with open(_state["log"], "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass
    print(f"[netguard] {action}: {event} -> {target}", file=sys.stderr)


def _proxy_endpoints() -> set[tuple[str, int]]:
    import urllib.parse
    import urllib.request

    out = set()
    urls = [v for k, v in os.environ.items() if k.lower() in _PROXY_VARS and v]  # оба регистра
    urls += [v for k, v in urllib.request.getproxies().items() if k != "no"]  # + реестр Windows
    for url in urls:
        try:
            u = urllib.parse.urlsplit(url if "://" in url else "http://" + url)
            if u.hostname:
                out.add((u.hostname.lower(), u.port or 8080))
        except ValueError:
            continue
    return out


def _hook(event: str, args: tuple) -> None:
    if not event.startswith("socket."):
        return
    host = None
    if event in ("socket.connect", "socket.sendto"):
        addr = args[1] if len(args) > 1 else None
        if not isinstance(addr, tuple):
            return  # unix-сокет (путь) или нечто нестандартное
        host = addr[0]
        if (str(host).lower(), addr[1]) in _state["proxies"]:
            blocked = _state["mode"] == "block"
            _record(event, f"{host}:{addr[1]} (proxy)", "BLOCKED" if blocked else "logged")
            if blocked:
                raise PermissionError(f"netguard: соединение с прокси запрещено ({host}:{addr[1]})")
            return
    elif event == "socket.getaddrinfo":
        host = args[0] if args else None
    elif event in ("socket.gethostbyname", "socket.gethostbyaddr", "socket.gethostbyname_ex"):
        host = args[0] if args else None
    else:
        return
    if is_local(host):
        return
    blocked = _state["mode"] == "block"
    _record(event, str(host), "BLOCKED" if blocked else "logged")
    if blocked:
        raise PermissionError(f"netguard: исходящая сеть запрещена ({event} -> {host})")


def install(mode: str | None = None, log_path: str | None = None) -> dict:
    """Идемпотентно. Аудит-хук нельзя снять — это свойство, а не баг."""
    mode = (mode or os.environ.get("SENTINEL_NETGUARD") or "block").lower()
    if mode not in ("block", "log"):
        return status()
    _state["mode"] = mode
    _state["log"] = log_path or os.environ.get("SENTINEL_NETGUARD_LOG") or os.path.join(
        tempfile.gettempdir(), "sentinel_netguard.jsonl")
    try:
        _state["proxies"] = _proxy_endpoints()
    except Exception:  # noqa: BLE001 — защита не должна ронять процесс
        _state["proxies"] = set()
    if mode == "block":
        for k in list(os.environ):
            if k.lower() in _PROXY_VARS:
                os.environ.pop(k)
        os.environ["NO_PROXY"] = os.environ["no_proxy"] = "*"
    if not _state["installed"]:
        sys.addaudithook(_hook)
        _state["installed"] = True
        os.environ["SENTINEL_NETGUARD_ACTIVE"] = mode
    return status()


def status() -> dict:
    out = dict(_state)
    out["proxies"] = sorted(f"{h}:{p}" for h, p in _state["proxies"])
    return out
