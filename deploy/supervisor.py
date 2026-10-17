#!/usr/bin/env python3
"""Супервизор сервиса Kassa Sentinel: один экземпляр, один uvicorn-воркер, туннель — после готовности.

  python deploy/supervisor.py run [--no-tunnel] [--named]   # foreground (windows\\start.ps1 запускает скрыто)
  python deploy/supervisor.py status [--json]                # код 0 — ready, 1 — нет
  python deploy/supervisor.py stop

Гарантии:
* второй экземпляр не стартует: OS-блокировка state/supervisor.lock (снимается при смерти процесса);
* uvicorn перезапускается только после того, как прежнее дерево процессов завершено и порт свободен;
  чужой процесс на порту не трогаем — ошибка;
* убиваем только СВОИ процессы: PID из state.json + совпадение времени старта (защита от повторного PID);
* crash-loop: больше max_restarts за restart_window_s → остановка с понятной ошибкой, туннель закрыт;
* «ready» = /readyz 200 (warmup A02) + успешная пробная задача (реальный inference) — только потом туннель;
* пока супервизор жив, система не засыпает (Windows SetThreadExecutionState; дисплей гаснуть может).
Секреты (API-ключ, TUNNEL_TOKEN) передаются через env, не через командную строку, и не логируются.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deploy import deploy_config as dc  # noqa: E402
from deploy import healthcheck, httpkit  # noqa: E402

try:
    import psutil  # есть в lock L1
except ImportError:  # pragma: no cover - проверяется отдельным тестом
    psutil = None  # type: ignore

IS_WIN = os.name == "nt"
QUICK_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
OOM_RE = re.compile(r"CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|cudaErrorMemoryAllocation")
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
               "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1", "SENTINEL_WORKERS": "1", "PYTHONUTF8": "1"}


# ----------------------------------------------------------------------------- процессы
def _win_create_time(pid: int) -> Optional[float]:  # pragma: no cover - Windows без psutil
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    k32.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(ctypes.c_ulonglong),) * 4
    k32.CloseHandle.argtypes = (wintypes.HANDLE,)
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return None
    try:
        t = [ctypes.c_ulonglong() for _ in range(4)]
        if not k32.GetProcessTimes(h, *(ctypes.byref(x) for x in t)):
            return None
        return t[0].value / 1e7 - 11644473600.0  # FILETIME (100 нс с 1601) → epoch, как psutil
    finally:
        k32.CloseHandle(h)


def _win_ppids() -> dict[int, int]:  # pragma: no cover - Windows без psutil
    """PID → PPID всех процессов (Toolhelp32)."""
    import ctypes
    from ctypes import wintypes

    class Entry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k32.CreateToolhelp32Snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
    k32.Process32FirstW.argtypes = k32.Process32NextW.argtypes = (wintypes.HANDLE, ctypes.POINTER(Entry))
    k32.CloseHandle.argtypes = (wintypes.HANDLE,)
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    if not snap or snap == wintypes.HANDLE(-1).value:
        return {}
    out, e = {}, Entry()
    e.dwSize = ctypes.sizeof(Entry)
    try:
        ok = k32.Process32FirstW(snap, ctypes.byref(e))
        while ok:
            out[int(e.th32ProcessID)] = int(e.th32ParentProcessID)
            ok = k32.Process32NextW(snap, ctypes.byref(e))
    finally:
        k32.CloseHandle(snap)
    return out


def create_time(pid: int) -> Optional[float]:
    if psutil is None:
        return _win_create_time(pid) if IS_WIN else None
    try:
        return psutil.Process(pid).create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    if psutil is not None:
        try:
            p = psutil.Process(pid)
            return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    if IS_WIN:  # pragma: no cover
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
        return str(pid) in out.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # зомби (завершился, но не собран) — не жив
        with open(f"/proc/{pid}/stat", encoding="ascii") as f:
            return f.read().split(") ", 1)[1][:1] != "Z"
    except (OSError, IndexError):
        return True


def is_ours(rec: Optional[dict]) -> bool:
    """Процесс из записи жив И это тот же процесс (время старта совпадает), а не чужой с тем же PID."""
    if not rec or not pid_alive(rec.get("pid")):
        return False
    want, have = rec.get("create_time"), create_time(rec["pid"])
    return want is None or have is None or abs(want - have) < 1.0


def kill_tree(pid: int, timeout: float = 15.0) -> None:
    if psutil is not None:
        try:
            parent = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        procs = parent.children(recursive=True) + [parent]
        for p in procs:
            try:
                p.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(procs, timeout=timeout)
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(alive, timeout=5)
        return
    if IS_WIN:  # pragma: no cover
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        return
    try:
        os.killpg(pid, signal.SIGTERM)  # дети запущены с start_new_session → pgid == pid
    except OSError:
        return
    end = time.time() + timeout
    while time.time() < end and pid_alive(pid):
        time.sleep(0.1)
    if pid_alive(pid):
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass


def descendants(pid: int) -> dict[int, Optional[float]]:
    """PID → время старта всех потомков (spawn-воркер A03 — потомок uvicorn). Без psutil: Windows — Toolhelp32,
    иначе пусто (POSIX добивает группу процессов)."""
    if psutil is None:
        if not IS_WIN:
            return {}
        ppids, out, todo = _win_ppids(), {}, [(pid, create_time(pid))]  # pragma: no cover
        while todo:  # pragma: no cover
            parent, pct = todo.pop()
            for c, pp in ppids.items():
                if pp == parent and c not in out and c != pid:
                    ct = create_time(c)
                    if ct is not None and pct is not None and ct < pct:
                        continue  # PPID указывает на переиспользованный PID: «ребёнок» старше родителя — чужой
                    out[c] = ct
                    todo.append((c, ct))
        return out
    try:
        return {c.pid: c.create_time() for c in psutil.Process(pid).children(recursive=True)}
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return {}


def reap(pids: dict[int, Optional[float]], pgid: Optional[int] = None, timeout: float = 15.0) -> list[int]:
    """Добить СВОИХ потомков (PID + время старта), в т.ч. осиротевших после падения родителя.
    POSIX: дополнительно вся группа процессов (start_new_session → pgid = pid uvicorn, живёт без лидера)."""
    if pgid and not IS_WIN:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except OSError:
                break
            end = time.time() + (timeout / 2 if sig == signal.SIGTERM else 2)
            while time.time() < end and any(pid_alive(x) for x in pids):
                time.sleep(0.1)
    for pid, ct in pids.items():
        if is_ours({"pid": pid, "create_time": ct}):
            kill_tree(pid, timeout=5)
    end = time.time() + timeout
    while time.time() < end and any(is_ours({"pid": x, "create_time": c}) for x, c in pids.items()):
        time.sleep(0.1)
    return [x for x, c in pids.items() if is_ours({"pid": x, "create_time": c})]


class WinJob:
    """Windows Job Object: аналог killpg. Потомки api (spawn-воркер A03) попадают в job автоматически,
    даже если промежуточный родитель (venv-лаунчер python.exe) уже умер. KILL_ON_JOB_CLOSE — при смерти
    супервизора ОС сама добивает дерево, VRAM не остаётся занятой. Только ctypes, без pywin32 и psutil."""
    KILL_ON_JOB_CLOSE = 0x2000
    EXTENDED_LIMIT_INFORMATION = 9

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes
        self.k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.k32.CreateJobObjectW.restype = wintypes.HANDLE
        self.k32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
        self.k32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
        self.k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        self.k32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
        self.k32.CloseHandle.argtypes = (wintypes.HANDLE,)
        self.handle = self.k32.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW")

        class IoCounters(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in ("r_ops", "w_ops", "o_ops", "r_bytes", "w_bytes", "o_bytes")]

        class BasicLimit(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class ExtendedLimit(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BasicLimit), ("IoInfo", IoCounters),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        info = ExtendedLimit()
        info.BasicLimitInformation.LimitFlags = self.KILL_ON_JOB_CLOSE
        if not self.k32.SetInformationJobObject(self.handle, self.EXTENDED_LIMIT_INFORMATION,
                                                ctypes.byref(info), ctypes.sizeof(info)):
            err = ctypes.get_last_error()
            self.close()
            raise OSError(err, "SetInformationJobObject")

    def assign(self, p: subprocess.Popen) -> bool:
        return bool(self.handle) and bool(self.k32.AssignProcessToJobObject(self.handle, int(p._handle)))  # type: ignore[attr-defined]

    def terminate(self) -> None:
        if self.handle:
            self.k32.TerminateJobObject(self.handle, 1)

    def close(self) -> None:
        if self.handle:
            self.k32.CloseHandle(self.handle)
            self.handle = None


def _resume(p: subprocess.Popen) -> None:
    """Снять CREATE_SUSPENDED: процесс попал в job до того, как успел породить детей."""
    import ctypes
    ntdll = ctypes.WinDLL("ntdll")
    ntdll.NtResumeProcess.argtypes = (ctypes.c_void_p,)
    ntdll.NtResumeProcess(int(p._handle))  # type: ignore[attr-defined]


def bypass_venv_launcher(cmd: list[str], env: dict[str, str]) -> list[str]:
    """Windows venv: Scripts/python.exe — лаунчер, он кладёт интерпретатор в СВОЙ job с SILENT_BREAKAWAY_OK,
    и внуки (spawn-воркер A03) выпадают из нашего job. Как multiprocessing (bpo-35797): запускаем базовый
    python напрямую, venv передаём через __PYVENV_LAUNCHER__."""
    base = getattr(sys, "_base_executable", sys.executable)
    same = lambda a, b: os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))  # noqa: E731
    if cmd and sys.prefix != sys.base_prefix and not same(base, sys.executable) and same(cmd[0], sys.executable):
        env["__PYVENV_LAUNCHER__"] = sys.executable
        return [base, *cmd[1:]]
    return cmd


def gpu_pids() -> Optional[set[int]]:
    """PID процессов с CUDA-контекстом (nvidia-smi); None — проверить нельзя."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-compute-apps=pid", "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return {int(x) for x in re.findall(r"\d+", out)}


def port_busy(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def keep_awake(on: bool) -> None:
    """Windows: система не засыпает, пока жив процесс; дисплей гаснуть может (ES_DISPLAY_REQUIRED не ставим)."""
    if not IS_WIN:
        return
    try:  # pragma: no cover
        import ctypes
        es_continuous, es_system = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | (es_system if on else 0))
    except Exception:  # noqa: BLE001
        pass


# ----------------------------------------------------------------------------- состояние
class InstanceLock:
    def __init__(self, path: Path) -> None:
        self.path, self.fh = path, None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a+")
        try:
            if IS_WIN:  # pragma: no cover
                import msvcrt
                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            self.fh.close()
            self.fh = None
            return False

    def release(self) -> None:
        if self.fh:
            self.fh.close()
            self.fh = None


class Store:
    def __init__(self, state_dir: Path) -> None:
        self.dir = state_dir
        self.path = state_dir / "state.json"
        self.stop_flag = state_dir / "stop.request"

    def read(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def write(self, st: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for attempt in range(50):  # Windows: replace падает (WinError 5), пока читатель (status, тест) держит файл
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                if not IS_WIN or attempt == 49:
                    raise
                time.sleep(0.02)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _git_sha(root: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def api_key(environ: dict, data_dir: Path) -> str:
    """Ключ для пробной задачи: env или файл ключей A02 (data/api_keys.txt). Никогда не логируется."""
    keys = [k.strip() for k in (environ.get("SENTINEL_API_KEYS") or "").split(",") if k.strip()]
    key = keys[0] if keys else (environ.get("SENTINEL_API_KEY") or "").strip()
    if key:
        return key
    for f in (environ.get("SENTINEL_API_KEYS_FILE"), str(data_dir / "api_keys.txt")):
        if f and Path(f).is_file():
            for line in Path(f).read_text(encoding="utf-8").splitlines():
                line = line.split("#", 1)[0].strip()
                if line:
                    return line
    return ""


def _tail(path: Path, n: int = 65536) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


# ----------------------------------------------------------------------------- супервизор
class Supervisor:
    def __init__(self, cfg: dict[str, Any], args: argparse.Namespace, environ: Optional[dict] = None) -> None:
        self.cfg, self.args = cfg, args
        self.env = dict(os.environ if environ is None else environ)
        svc, paths = cfg.get("service") or {}, cfg.get("paths") or {}
        self.host = str(svc.get("host", "127.0.0.1"))
        self.port = int(args.port or self.env.get("SENTINEL_PORT") or svc.get("port", 8000))
        self.url = f"http://127.0.0.1:{self.port}"
        self.logs = Path(self.env.get("SENTINEL_LOG_DIR") or dc.repo_path(paths.get("logs", "runs/deploy")))
        self.data = Path(self.env.get("SENTINEL_DATA_DIR") or dc.repo_path(paths.get("data", "data")))
        self.store = Store(self.logs / "state")
        self.lock = InstanceLock(self.logs / "state" / "supervisor.lock")
        self.interval = float(args.interval)
        self.max_restarts = int(svc.get("max_restarts", 5))
        self.window = float(svc.get("restart_window_s", 600))
        self.ready_timeout = float(svc.get("ready_timeout_s", 900))
        self.trial_limit = float(svc.get("trial_job_limit_s", 300))
        tun = cfg.get("tunnel") or {}
        self.tunnel_mode = "off" if args.no_tunnel else (
            "named" if args.named else str(self.env.get("SENTINEL_TUNNEL_MODE") or tun.get("mode", "quick")))
        self.protocol = str(self.env.get("CLOUDFLARED_PROTOCOL") or tun.get("protocol") or "")
        self.netguard = str(self.env.get("SENTINEL_NETGUARD") or svc.get("netguard", "log"))
        self.app = str(args.app or svc.get("app", "backend.api.main:app"))
        self.api: Optional[subprocess.Popen] = None
        self.api_job: Optional[WinJob] = None  # Windows: всё дерево api (аналог pgid на POSIX)
        self.tunnel: Optional[subprocess.Popen] = None
        self.restarts: collections.deque[float] = collections.deque()
        self.tree: dict[int, Optional[float]] = {}  # потомки текущего api (snapshot каждый виток + watcher)
        self.tree_lock = threading.Lock()
        self.tree_saved: set[int] = set()
        self.watch_stop = threading.Event()
        self.st: dict[str, Any] = {}

    # ---- окружение детей
    def child_env(self) -> dict[str, str]:
        env = dict(self.env)
        env.update(OFFLINE_ENV)
        env["SENTINEL_PORT"] = str(self.port)
        env["SENTINEL_LOG_DIR"] = str(self.logs)
        pp = [str(REPO_ROOT)]
        if self.netguard in ("log", "block"):
            pp.append(str(REPO_ROOT / "deploy" / "netguard"))
            env["SENTINEL_NETGUARD"] = self.netguard
            env["SENTINEL_NETGUARD_LOG"] = str(self.logs / "netguard.jsonl")
        if env.get("PYTHONPATH"):
            pp.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pp)
        return env

    def _popen(self, cmd: list[str], name: str, env: dict[str, str], job: Optional["WinJob"] = None) -> subprocess.Popen:
        self.logs.mkdir(parents=True, exist_ok=True)
        out = open(self.logs / f"{name}.out.log", "ab")
        err = open(self.logs / f"{name}.err.log", "ab")
        kw: dict[str, Any] = {"stdout": out, "stderr": err, "stdin": subprocess.DEVNULL, "cwd": str(REPO_ROOT), "env": env}
        if IS_WIN:  # pragma: no cover
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if job is not None:  # в job до первой инструкции: иначе ребёнок успеет породить внука вне job
                kw["creationflags"] |= 0x4  # CREATE_SUSPENDED
                env = kw["env"] = dict(env)
                cmd = bypass_venv_launcher(cmd, env)
        else:
            kw["start_new_session"] = True
        try:
            p = subprocess.Popen(cmd, **kw)
            if IS_WIN and job is not None:  # pragma: no cover
                try:
                    if not job.assign(p):
                        self.log(f"WARN AssignProcessToJobObject не удался для pid={p.pid}")
                finally:
                    _resume(p)
            return p
        finally:
            out.close()
            err.close()

    def _rec(self, p: subprocess.Popen, **extra: Any) -> dict[str, Any]:
        return {"pid": p.pid, "create_time": create_time(p.pid), "started_at": _now(), **extra}

    def save(self, **kw: Any) -> None:
        self.st.update(kw)
        self.st["updated_at"] = _now()
        self.store.write(self.st)

    def log(self, msg: str) -> None:
        line = f"{_now()} {msg}"
        print(line, flush=True)
        with open(self.logs / "supervisor.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ---- api
    def api_cmd(self) -> list[str]:
        if self.args.api_cmd:
            return json.loads(self.args.api_cmd)
        return [sys.executable, "-m", "uvicorn", self.app, "--host", self.host, "--port", str(self.port),
                "--workers", "1", "--timeout-keep-alive", "75"]

    def start_api(self) -> bool:
        deadline = time.time() + 30
        while port_busy("127.0.0.1", self.port):  # прежний наш процесс уже убит — ждём освобождения порта
            if time.time() > deadline:
                self.save(phase="failed", last_error=f"порт {self.port} занят чужим процессом — не трогаем")
                self.log(f"ERROR порт {self.port} занят, второй сервер не запускаем")
                return False
            time.sleep(0.5)
        self.api_job = None
        if IS_WIN:  # pragma: no cover
            try:
                self.api_job = WinJob()
            except OSError as e:
                self.log(f"WARN Job Object недоступен ({e}) — потомков api добиваем только по snapshot")
        self.api = self._popen(self.api_cmd(), "api", self.child_env(), job=self.api_job)
        self.save(api=self._rec(self.api, cmd=self.app), phase="starting", api_started=time.time(), api_tree=[])
        self.log(f"api pid={self.api.pid} {self.app} на {self.host}:{self.port}")
        return True

    def stop_proc(self, p: Optional[subprocess.Popen]) -> None:
        if p is not None and p.poll() is None:
            if p is self.api:
                self.snapshot_tree()
            kill_tree(p.pid)
        if p is not None:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        if p is not None and p is self.api:
            self.reap_api_tree()

    def _grab_tree(self) -> None:
        api = self.api
        if api is not None and api.poll() is None:
            fresh = descendants(api.pid)
            if fresh:
                with self.tree_lock:
                    self.tree.update(fresh)

    def _watch_tree(self, period: float = 0.1) -> None:
        """Windows: killpg нет, а Job Object ловит не всё (дети Store-Python выходят из job) — фоновый snapshot
        каждые 100 мс, независимо от блокирующих check_ready (~2 с на отказ соединения)."""
        while not self.watch_stop.wait(period):
            try:
                self._grab_tree()
            except Exception:  # noqa: BLE001 - наблюдатель не должен ронять супервизор
                pass

    def snapshot_tree(self) -> None:
        self._grab_tree()
        with self.tree_lock:
            tree = dict(self.tree)
        if set(tree) - self.tree_saved:
            self.tree_saved = set(tree)
            self.save(api_tree=[{"pid": k, "create_time": v} for k, v in tree.items()])

    def reap_api_tree(self) -> None:
        """После выхода/убийства api: добить его потомков (GPU-воркер A03) и проверить, что GPU освобождён."""
        if self.api is None:
            return
        if self.api_job is not None:  # pragma: no cover - Windows: в job только наше дерево api
            self.api_job.terminate()
            self.api_job.close()
            self.api_job = None
        with self.tree_lock:
            tree, self.tree = self.tree, {}
        self.tree_saved = set()
        left = reap(tree, pgid=self.api.pid)
        ours = set(tree) | {self.api.pid}
        on_gpu = gpu_pids()
        still = sorted((on_gpu or set()) & ours)
        if left:
            self.log(f"ERROR не удалось завершить потомков api: {left}")
        if still:
            self.log(f"ERROR GPU всё ещё держат наши процессы {still}")
        self.save(gpu_released=None if on_gpu is None else not still, orphans_left=left, api_tree=[])

    # ---- tunnel
    def tunnel_cmd(self) -> list[str]:
        if self.args.tunnel_cmd:
            return json.loads(self.args.tunnel_cmd)
        exe = shutil.which("cloudflared") or "cloudflared"
        cmd = [exe, "tunnel", "--no-autoupdate"]
        if self.protocol:
            cmd += ["--protocol", self.protocol]
        if self.tunnel_mode == "named":
            return cmd + ["run"]  # токен — из env TUNNEL_TOKEN, не из argv
        return cmd + ["--url", self.url]

    def start_tunnel(self) -> None:
        if self.tunnel_mode == "off":
            return
        if self.tunnel_mode == "named" and not self.env.get("TUNNEL_TOKEN"):
            self.save(last_error="named tunnel без TUNNEL_TOKEN")
            self.log("ERROR named tunnel без TUNNEL_TOKEN")
            return
        env = dict(self.env)
        (self.logs / "tunnel.err.log").write_bytes(b"")  # новый quick URL ищем только в свежем логе
        self.tunnel = self._popen(self.tunnel_cmd(), "tunnel", env)
        self.save(tunnel=self._rec(self.tunnel, mode=self.tunnel_mode, url=None))
        self.log(f"tunnel pid={self.tunnel.pid} mode={self.tunnel_mode}")

    def poll_tunnel_url(self) -> None:
        t = self.st.get("tunnel") or {}
        if self.tunnel is None or t.get("url") or self.tunnel_mode != "quick":
            return
        m = QUICK_URL_RE.findall(_tail(self.logs / "tunnel.err.log") + _tail(self.logs / "tunnel.out.log"))
        if m:
            t["url"] = m[-1]
            (self.logs / "public_url.txt").write_text(m[-1] + "\n", encoding="utf-8")
            self.save(tunnel=t)
            self.log(f"PUBLIC URL: {m[-1]}")
            self.log(f"  проверка: python deploy/tunnel_probe.py real --url {m[-1]}")
            self.log(f"  приёмка A16: python -m eval.jury_sim --url {m[-1]} --key <ключ> --videos data/videos --n 6")

    # ---- готовность
    def check_ready(self) -> None:
        """Production gate: /readyz A02 (ready/real) → только тогда synthetic-задача → только тогда туннель."""
        rz = healthcheck.readiness(self.url, allow_stub=self.args.allow_stub)
        self.st["readyz"] = {k: rz.get(k) for k in ("status", "state", "pipeline", "verdict")}
        if rz["verdict"] == "waiting":
            self.save(phase="warming", detail=rz["why"])
            return
        if rz["verdict"] == "terminal" and rz.get("state") == "gpu_hung":
            self.log(f"readyz gpu_hung — перезапуск api (освобождение GPU): {rz['why']}")
            self.save(phase="restarting", detail=rz["why"], last_error=rz["why"])
            self.stop_proc(self.api)  # следующий виток цикла увидит выход и перезапустит
            return
        if rz["verdict"] in ("terminal", "bad"):
            if self.st.get("detail") != rz["why"]:
                self.log(f"NOT READY (без пробной задачи): {rz['why']}")
            self.save(phase="not_ready", detail=rz["why"], last_error=rz["why"])
            return
        key = api_key(self.env, self.data)
        video = self.logs / "trial_synth.mov"
        if not video.is_file():
            from deploy.synth_video import make_video
            make_video(video, seconds=10)
        self.save(phase="trial", detail="synthetic-задача POST/GET (тракт и контракт, не качество CV)")
        rep = healthcheck.ready(self.url, key, video, self.trial_limit, allow_stub=self.args.allow_stub)
        (self.logs / "ready.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if rep.get("ok"):
            self.save(phase="ready", ready_at=_now(),
                      detail=f"synthetic-задача: {rep.get('from_accept_s')} с от приёма до итога")
            self.log(f"READY: synthetic-задача {rep.get('from_accept_s')} с от приёма (server {rep.get('server')})")
            if self.tunnel is None or self.tunnel.poll() is not None:
                self.start_tunnel()  # открываем ОДИН раз; при рестарте api туннель (и quick URL) сохраняется
        else:
            self.save(phase="not_ready", detail="; ".join(rep.get("problems", []))[:500])
            self.log("NOT READY: " + "; ".join(rep.get("problems", [])))

    # ---- основной цикл
    def run(self) -> int:
        if not self.lock.acquire():
            prev = self.store.read().get("supervisor", {})
            print(f"[supervisor] уже запущен (pid {prev.get('pid')}); статус: deploy/supervisor.py status", file=sys.stderr)
            return 3
        try:
            self.store.stop_flag.unlink(missing_ok=True)
            self.st = {"supervisor": {"pid": os.getpid(), "create_time": create_time(os.getpid()), "started_at": _now()},
                       "sha": _git_sha(REPO_ROOT), "port": self.port, "url_local": self.url, "phase": "starting",
                       "tunnel_mode": self.tunnel_mode, "netguard": self.netguard, "restarts": 0, "last_error": None}
            self.save()
            keep_awake(True)
            if IS_WIN:  # pragma: no cover
                threading.Thread(target=self._watch_tree, name="tree-watch", daemon=True).start()
            if port_busy("127.0.0.1", self.port):
                self.save(phase="failed", last_error=f"порт {self.port} уже занят чужим процессом")
                self.log(f"ERROR порт {self.port} занят — сервис не запускаю, чужой процесс не трогаю")
                return 5
            if not self.start_api():
                return 5
            return self._loop()
        finally:
            self.stop_proc(self.tunnel)
            self.stop_proc(self.api)
            self.watch_stop.set()
            keep_awake(False)
            if self.st:
                self.save(phase=self.st.get("phase") if self.st.get("phase") in ("crash_loop", "failed")
                          else "stopped", stopped_at=_now())
            self.lock.release()

    def _loop(self) -> int:
        signal.signal(signal.SIGTERM, lambda *_: self.store.stop_flag.touch())
        live_fail, next_ready_try, next_readyz = 0, 0.0, 0.0
        while True:
            if self.store.stop_flag.exists():
                self.log("stop requested")
                self.store.stop_flag.unlink(missing_ok=True)
                return 0
            self.snapshot_tree()
            if self.api is not None and self.api.poll() is not None:
                code = self.api.returncode
                self.reap_api_tree()  # spawn-воркер A03 мог пережить родителя и держать VRAM
                tail = _tail(self.logs / "api.err.log")
                oom = OOM_RE.search(tail)
                err = f"api завершился с кодом {code}" + (" — GPU OOM (см. deploy/RUNBOOK.md#gpu-oom)" if oom else "")
                self.log(err)
                now = time.time()
                self.restarts.append(now)
                while self.restarts and now - self.restarts[0] > self.window:
                    self.restarts.popleft()
                # туннель НЕ трогаем: quick URL у жюри должен пережить рестарт api (на время прогрева — 502)
                self.save(last_error=err, restarts=self.st.get("restarts", 0) + 1)
                if len(self.restarts) > self.max_restarts:
                    self.save(phase="crash_loop", last_error=err + f"; {len(self.restarts)} падений за {self.window:.0f} с")
                    self.log("ERROR crash-loop: супервизор останавливается (deploy/RUNBOOK.md#crash-loop)")
                    return 4
                time.sleep(min(30.0, self.interval * 2 ** len(self.restarts)))
                if not self.start_api():
                    return 5
                continue
            phase = self.st.get("phase")
            if phase in ("starting", "warming", "not_ready", "trial", "restarting") and time.time() >= next_ready_try:
                if time.time() - float(self.st.get("api_started") or time.time()) > self.ready_timeout:
                    if phase != "ready_timeout":
                        self.save(phase="ready_timeout",
                                  last_error=f"нет готовности за {self.ready_timeout:.0f} с — туннель НЕ открыт")
                        self.log("ERROR ready timeout — туннель закрыт (runs/deploy/ready.json, api.err.log)")
                else:
                    self.check_ready()
                    next_ready_try = time.time() + (30.0 if self.st.get("phase") == "not_ready" else self.interval)
            elif phase == "ready":
                ok = healthcheck.live(self.url)["ok"]
                live_fail = 0 if ok else live_fail + 1
                if ok and time.time() >= next_readyz:
                    next_readyz = time.time() + 30.0
                    rz = healthcheck.readiness(self.url, allow_stub=self.args.allow_stub)
                    self.st["readyz"] = {k: rz.get(k) for k in ("status", "state", "pipeline", "verdict")}
                    if rz.get("state") == "gpu_hung":
                        self.log(f"readyz gpu_hung — перезапуск api: {rz['why']}")
                        self.save(phase="restarting", last_error=rz["why"])
                        self.stop_proc(self.api)
                        continue
                if live_fail >= max(3, int(30 / max(self.interval, 0.1))):
                    self.log("api не отвечает — перезапуск (зависание)")
                    self.stop_proc(self.api)
                    live_fail = 0
                    continue
                if self.tunnel is not None and self.tunnel.poll() is not None:
                    self.log(f"tunnel завершился ({self.tunnel.returncode}) — перезапуск")
                    time.sleep(self.interval)
                    self.start_tunnel()
                self.poll_tunnel_url()
            time.sleep(self.interval)


# ----------------------------------------------------------------------------- status / stop
def status(store: Store, as_json: bool = False) -> int:
    st = store.read()
    if not st:
        print("[status] сервис не запускался (нет state.json)")
        return 1
    sup_alive = is_ours(st.get("supervisor"))
    api_alive = is_ours(st.get("api"))
    tun_alive = is_ours(st.get("tunnel"))
    port = st.get("port", 8000)
    health = healthcheck.live(f"http://127.0.0.1:{port}") if api_alive else {"ok": False}
    ready = httpkit.request("GET", f"http://127.0.0.1:{port}/readyz", timeout=5) if api_alive else None
    out = {"phase": st.get("phase"), "supervisor_alive": sup_alive, "api_alive": api_alive, "tunnel_alive": tun_alive,
           "healthz": health.get("ok"), "readyz": ready.status if ready else None,
           "public_url": (st.get("tunnel") or {}).get("url"), "restarts": st.get("restarts"),
           "last_error": st.get("last_error"), "detail": st.get("detail"), "sha": st.get("sha"),
           "updated_at": st.get("updated_at")}
    out["ok"] = bool(sup_alive and api_alive and out["phase"] == "ready" and out["healthz"])
    if as_json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        for k, v in out.items():
            print(f"{k:<17} {v}")
    return 0 if out["ok"] else 1


def stop(store: Store, timeout: float = 60.0) -> int:
    st = store.read()
    sup = st.get("supervisor")
    if is_ours(sup):
        store.stop_flag.touch()
        end = time.time() + timeout
        while time.time() < end and pid_alive(sup["pid"]):
            time.sleep(0.3)
        if pid_alive(sup["pid"]) and is_ours(sup):
            kill_tree(sup["pid"])
    killed = []
    for name in ("tunnel", "api"):  # сироты (супервизор умер аварийно): только свои, по времени старта
        rec = store.read().get(name) or st.get(name)
        if is_ours(rec):
            kill_tree(rec["pid"])
            killed.append(f"{name}:{rec['pid']}")
    tree = {int(x["pid"]): x.get("create_time") for x in (store.read().get("api_tree") or st.get("api_tree") or [])}
    left = reap(tree) if tree else []
    killed += [f"worker:{x}" for x in tree if x not in left]
    print("[stop] остановлено" + (f"; добиты сироты {killed}" if killed else ""))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run", "status", "stop"))
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--app", default=None)
    ap.add_argument("--no-tunnel", action="store_true")
    ap.add_argument("--named", action="store_true")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--api-cmd", default=None, help=argparse.SUPPRESS)      # тесты deploy/
    ap.add_argument("--tunnel-cmd", default=None, help=argparse.SUPPRESS)   # тесты deploy/
    ap.add_argument("--allow-stub", action="store_true", help=argparse.SUPPRESS)  # тесты deploy/
    a = ap.parse_args(argv)
    cfg = dc.load()
    logs = Path(os.environ.get("SENTINEL_LOG_DIR") or dc.repo_path((cfg.get("paths") or {}).get("logs", "runs/deploy")))
    store = Store(logs / "state")
    if a.cmd == "status":
        return status(store, a.json)
    if a.cmd == "stop":
        return stop(store)
    return Supervisor(cfg, a).run()


if __name__ == "__main__":
    sys.exit(main())
