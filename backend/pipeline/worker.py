"""Исполнители стадий. Один владелец GPU: все вызовы модулей идут ПОСЛЕДОВАТЕЛЬНО в одном месте.

* ``ProcessExecutor`` (прод, по умолчанию): отдельный процесс-воркер (multiprocessing spawn — работает
  и на Windows) держит модули, тёплые модели и клип текущей задачи. Таймаут стадии → воркер
  убивается (kill), CUDA-контекст умирает вместе с процессом → VRAM гарантированно освобождена;
  следующий вызов поднимает новый воркер и заново открывает клип задачи.
* ``InlineExecutor`` (тесты/отладка): то же в текущем процессе; таймаут реализован потоком и НЕ
  останавливает CUDA → исполнитель помечается «отравленным», об этом пишется warning.

Задачи сериализуются арендой (``lease``): пока задача держит воркер, другая ждёт (отсчёт её дедлайна
идёт). Перед тяжёлой GPU-стадией ``prepare_vram`` выгружает другие модели через их хуки ``unload()``
(модуль сам обнуляет ссылки на модель/тензоры), меряет свободную VRAM и при нехватке
оркестратор перезапускает воркер.
"""
from __future__ import annotations

import atexit
import contextlib
import gc
import logging
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Optional

log = logging.getLogger("sentinel.pipeline")

_GB = 1024 ** 3
_MB = 1024 ** 2


class ClipRef:
    """Маркер аргумента: «подставь клип текущей задачи» (сам клип не пересылается между процессами).
    ``window=(t0, t1, fps, max_frames)`` — окно клипа с плотной выборкой (режим coarse→fine)."""

    def __init__(self, window: Optional[tuple] = None) -> None:
        self.window = tuple(window) if window else None

    def __repr__(self) -> str:  # pragma: no cover
        return f"<clip window={self.window}>" if self.window else "<clip>"


CLIP = ClipRef()


def window_times(t0: float, t1: float, fps: float, max_frames: int, duration: Optional[float] = None) -> list[float]:
    """Плотные таймстемпы окна [t0, t1]: шаг 1/fps, не больше max_frames (иначе шаг растягивается)."""
    t0 = max(0.0, float(t0))
    if duration:
        t1 = min(float(t1), float(duration))
    if t1 <= t0:
        return [round(t0, 4)]
    step = 1.0 / max(0.1, float(fps))
    n = int((t1 - t0) / step) + 1
    if max_frames and n > max_frames:
        step = (t1 - t0) / max(1, max_frames - 1)
        n = max_frames
    return [round(t0 + k * step, 4) for k in range(n)]


class WindowClip:
    """VideoClip-окно поверх клипа задачи: times — плотные таймстемпы окна, frame(i) — точный кадр
    по PTS (frame_at исходного клипа). Модули стадий не знают, что работают с окном."""

    def __init__(self, base: Any, times: list[float]) -> None:
        self._base = base
        self.times = list(times)
        self._frames: dict[int, Any] = {}
        for attr in ("path", "duration", "fps", "width", "height"):
            setattr(self, attr, getattr(base, attr, None))

    def __len__(self) -> int:
        return len(self.times)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def frame(self, i: int) -> Any:
        if i not in self._frames:
            self._frames[i] = self._base.frame_at(self.times[i])
        return self._frames[i]

    def frame_at(self, t: float) -> Any:
        return self._base.frame_at(t)


@dataclass
class CallResult:
    status: str                      # ok | error | timeout | missing | crashed | skipped
    value: Any = None
    error: str = ""
    seconds: float = 0.0
    origin: str = ""                 # module | builtin-mock | fallback
    identity: dict = field(default_factory=dict)
    vram_peak_mb: Optional[float] = None
    trace: str = ""
    cold: bool = False               # первый вызов функции в этом процессе-воркере (≈ загрузка модели)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


# ============================================================== среда выполнения (внутри воркера)

class Runtime:
    """Состояние исполнителя: клип задачи, разрешённые модули. Живёт в процессе-воркере."""

    def __init__(self, *, mock_missing: bool = False, allow_fallback_ingest: bool = True,
                 frame_cache_mb: float = 512.0) -> None:
        self.mock_missing = mock_missing
        self.allow_fallback_ingest = allow_fallback_ingest
        self.frame_cache_mb = frame_cache_mb
        self.clip: Any = None
        self.clip_key: Optional[tuple] = None
        self._windows: dict[tuple, WindowClip] = {}
        self._called: set[tuple[str, str]] = set()

    # ---------------------------------------------------------------- клип
    def load_clip(self, path: str, sample_fps: float) -> dict:
        from .media import LockedClip, SimpleClip
        from .stages import ModuleMissing, resolve

        try:
            st = os.stat(path)
            key = (path, float(sample_fps), st.st_size, st.st_mtime_ns)
        except OSError:
            key = (path, float(sample_fps), None, None)
        if self.clip is not None and self.clip_key == key:  # тот же файл/fps — клип переиспользуется
            return self._clip_info(self.clip, getattr(self, "_clip_source", "ml.ingest"),
                                   list(getattr(self, "_clip_warnings", [])))
        self.clip = None
        self._windows = {}
        gc.collect()
        warnings: list[str] = []
        source = "ml.ingest"
        clip = None
        try:
            fn, origin, _mod = resolve("ml.ingest", "load_video", mock_missing=False)
            clip = fn(path, sample_fps=sample_fps)
            if clip is None or not len(getattr(clip, "times", []) or []):
                raise RuntimeError("ml.ingest вернул пустой клип")
        except ModuleMissing as exc:
            if not self.allow_fallback_ingest:
                raise
            warnings.append(f"ingest: {exc} — использован встроенный декодер OpenCV")
        except Exception as exc:  # noqa: BLE001
            if not self.allow_fallback_ingest:
                raise
            warnings.append(f"ingest: ml.ingest упал ({type(exc).__name__}: {exc}) — встроенный декодер OpenCV")
            clip = None
        if clip is None:
            clip = SimpleClip(path, sample_fps=sample_fps)
            source = "fallback-opencv"
        self.clip = LockedClip(clip, cache_mb=self.frame_cache_mb)
        self.clip_key = key
        self._clip_source = source
        self._clip_warnings = list(warnings)
        return self._clip_info(self.clip, source, warnings)

    @staticmethod
    def _clip_info(clip: Any, source: str, warnings: list[str]) -> dict:
        return {"path": str(getattr(clip, "path", "")), "duration": float(getattr(clip, "duration", 0.0) or 0.0),
                "fps": float(getattr(clip, "fps", 0.0) or 0.0), "width": int(getattr(clip, "width", 0) or 0),
                "height": int(getattr(clip, "height", 0) or 0),
                "times": [float(t) for t in (getattr(clip, "times", []) or [])],
                "source": source, "warnings": warnings}

    def drop_clip(self) -> None:
        self.clip = None
        self.clip_key = None
        self._windows = {}
        gc.collect()

    def _resolve_clip(self, ref: ClipRef) -> Any:
        if ref.window is None or self.clip is None:
            return self.clip
        win = self._windows.get(ref.window)
        if win is None:
            t0, t1, fps, max_frames = ref.window
            win = WindowClip(self.clip, window_times(t0, t1, fps, int(max_frames),
                                                     getattr(self.clip, "duration", None)))
            if len(self._windows) >= 8:  # окна одной задачи; старые — вон
                self._windows.pop(next(iter(self._windows)))
            self._windows[ref.window] = win
        return win

    # ---------------------------------------------------------------- вызовы модулей
    def identity(self, module: str, func: str) -> dict:
        from .stages import ModuleMissing, module_identity, resolve

        try:
            _fn, origin, mod = resolve(module, func, mock_missing=self.mock_missing)
        except ModuleMissing as exc:
            return {"origin": "missing", "error": str(exc)}
        return module_identity(mod, origin)

    def call(self, module: str, func: str, args: tuple = (), kwargs: Optional[dict] = None) -> CallResult:
        from .stages import ModuleMissing, module_identity, resolve

        t0 = time.perf_counter()
        try:
            fn, origin, mod = resolve(module, func, mock_missing=self.mock_missing)
        except ModuleMissing as exc:
            return CallResult("missing", error=str(exc), seconds=time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001 — модуль есть, но его импорт падает
            return CallResult("error", error=f"импорт {module}: {type(exc).__name__}: {exc}",
                              seconds=time.perf_counter() - t0, trace=traceback.format_exc(limit=8))
        ident = module_identity(mod, origin)
        real_args = tuple(self._resolve_clip(a) if isinstance(a, ClipRef) else a for a in args)
        real_kwargs = {k: (self._resolve_clip(v) if isinstance(v, ClipRef) else v) for k, v in (kwargs or {}).items()}
        if any(isinstance(a, ClipRef) for a in args) and self.clip is None:
            return CallResult("error", error="клип не загружен", origin=origin, identity=ident)
        cold = (module, func) not in self._called
        self._called.add((module, func))
        torch = _torch_cuda()
        if torch is not None:
            with contextlib.suppress(Exception):
                torch.cuda.reset_peak_memory_stats()
        try:
            value = fn(*real_args, **real_kwargs)
        except Exception as exc:  # noqa: BLE001
            return CallResult("error", error=f"{type(exc).__name__}: {exc}", seconds=time.perf_counter() - t0,
                              origin=origin, identity=ident, trace=traceback.format_exc(limit=8), cold=cold)
        peak = None
        torch = _torch_cuda()
        if torch is not None:
            with contextlib.suppress(Exception):
                peak = round(torch.cuda.max_memory_allocated() / _MB, 1)
        return CallResult("ok", value=value, seconds=time.perf_counter() - t0, origin=origin,
                          identity=ident, vram_peak_mb=peak, cold=cold)

    # ---------------------------------------------------------------- память
    def vram(self) -> dict:
        torch = _torch_cuda()
        if torch is None:
            return {"cuda": False}
        try:
            free, total = torch.cuda.mem_get_info()
            return {"cuda": True, "free_gb": round(free / _GB, 3), "total_gb": round(total / _GB, 3),
                    "allocated_mb": round(torch.cuda.memory_allocated() / _MB, 1),
                    "reserved_mb": round(torch.cuda.memory_reserved() / _MB, 1)}
        except Exception as exc:  # noqa: BLE001
            return {"cuda": True, "error": str(exc)}

    def unload(self, modules: list[str]) -> dict:
        """Выгрузка через хуки модулей ``unload()`` (модуль обязан обнулить ссылки на модель)
        + модели, которые модули держат в ModelRegistry. Возвращает замеры до/после."""
        from .registry import ModelRegistry

        before = self.vram()
        called: list[str] = []
        failed: list[str] = []
        for name in modules:
            mod = sys.modules.get(name)
            hook = getattr(mod, "unload", None) if mod is not None else None
            if callable(hook):
                try:
                    hook()
                    called.append(name)
                except Exception as exc:  # noqa: BLE001
                    failed.append(f"{name}: {exc}")
        reg = ModelRegistry.peek()
        released = []
        if reg is not None:
            for key in list(reg.loaded()):
                if any(key == m or key.startswith(m + ":") for m in modules):
                    reg.release(key)
                    released.append(key)
        gc.collect()
        torch = _torch_cuda()
        if torch is not None:
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
        return {"before": before, "after": self.vram(), "hooks": called, "hook_errors": failed,
                "registry_released": released}

    def prepare_vram(self, need_gb: float, unload_modules: list[str], margin_gb: float = 0.3) -> dict:
        info = self.vram()
        if not info.get("cuda") or "free_gb" not in info:
            return {"ok": True, "cuda": bool(info.get("cuda")), "action": "none"}
        if info["free_gb"] >= need_gb + margin_gb:
            return {"ok": True, "cuda": True, "action": "none", "free_gb": info["free_gb"]}
        res = self.unload(unload_modules)
        free_after = res["after"].get("free_gb", 0.0)
        return {"ok": free_after >= need_gb + margin_gb, "cuda": True, "action": "unload", **res}

    def warmup(self, modules: list[str]) -> dict:
        """Хуки warmup() модулей (загрузка весов). По модулю: {"status": ok | no_hook | missing | error,
        "seconds", "error"}. missing — пакета нет; error — импорт/хук упал (например, нет весов)."""
        import importlib

        from .stages import _is_missing_module

        out: dict[str, dict] = {}
        for name in modules:
            t0 = time.perf_counter()
            try:
                mod = importlib.import_module(name)
            except ImportError as exc:
                out[name] = {"status": "missing" if _is_missing_module(exc, name) else "error",
                             "error": f"{type(exc).__name__}: {exc}"[:300], "seconds": round(time.perf_counter() - t0, 3)}
                continue
            except Exception as exc:  # noqa: BLE001
                out[name] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:300],
                             "seconds": round(time.perf_counter() - t0, 3)}
                continue
            hook = getattr(mod, "warmup", None)
            if not callable(hook):
                out[name] = {"status": "no_hook", "seconds": round(time.perf_counter() - t0, 3)}
                continue
            try:
                hook()
                out[name] = {"status": "ok", "seconds": round(time.perf_counter() - t0, 3)}
            except Exception as exc:  # noqa: BLE001
                out[name] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:300],
                             "seconds": round(time.perf_counter() - t0, 3)}
        return out

    def device(self) -> str:
        """Устройство глазами владельца GPU (torch импортируется здесь, а не в процессе API)."""
        if os.environ.get("SENTINEL_DEVICE", "").lower() == "cpu":
            return "cpu"
        try:
            import torch  # noqa: WPS433
        except Exception:  # noqa: BLE001
            return "cpu"
        try:
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"

    def ping(self) -> dict:
        return {"pid": os.getpid(), "clip": self.clip_key is not None, "torch": "torch" in sys.modules}


def _torch_cuda() -> Any:
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        return torch if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        return None


def _worker_main(conn: Any, init: dict) -> None:  # pragma: no cover — исполняется в дочернем процессе
    os.environ.update(init.get("env") or {})
    for p in reversed(init.get("sys_path") or []):
        if p not in sys.path:
            sys.path.insert(0, p)
    logging.basicConfig(level=logging.WARNING)
    rt = Runtime(mock_missing=init.get("mock_missing", False),
                 allow_fallback_ingest=init.get("allow_fallback_ingest", True),
                 frame_cache_mb=init.get("frame_cache_mb", 512.0))
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break  # родитель умер/закрыл канал → выходим, VRAM освобождается вместе с процессом
        cmd, payload = msg
        if cmd == "stop":
            break
        try:
            result = getattr(rt, cmd)(**payload)
            reply = ("ok", result)
        except BaseException as exc:  # noqa: BLE001
            reply = ("error", f"{type(exc).__name__}: {exc}", traceback.format_exc(limit=8))
        try:
            conn.send(reply)
        except Exception as exc:  # noqa: BLE001 — например, непиклуемый результат
            conn.send(("error", f"результат не сериализуется: {type(exc).__name__}: {exc}", ""))
    with contextlib.suppress(Exception):
        conn.close()


# ============================================================== исполнители (в процессе оркестратора)

class WorkerTimeout(Exception):
    pass


class WorkerCancelled(Exception):
    pass


class WorkerCrashed(Exception):
    pass


class _BaseExecutor:
    mode = "base"

    def __init__(self, cfg: Mapping) -> None:
        self.cfg = cfg
        self._lease = threading.Lock()
        self._job: Optional[tuple[str, float]] = None
        self.restarts = 0
        self.kills = 0
        self.poisoned = False
        self.cancel_event: Optional[threading.Event] = None

    def cancelled(self) -> bool:
        ev = self.cancel_event
        return ev is not None and ev.is_set()

    def release(self) -> dict:
        """Освободить GPU: остановить воркер (process) или выгрузить модели (inline)."""
        self.shutdown()
        return {"mode": self.mode, "action": "shutdown"}

    @contextlib.contextmanager
    def lease(self, timeout: Optional[float], cancel: Optional[threading.Event] = None) -> Iterator[bool]:
        """Аренда исполнителя задачей. Ожидание прерывается отменой задачи (cancel)."""
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        ok = False
        while not (cancel is not None and cancel.is_set()):
            if self._lease.acquire(blocking=False):   # свободный воркер берём сразу, даже при timeout=0
                ok = True
                break
            left = None if deadline is None else deadline - time.monotonic()
            if left is not None and left <= 0:
                break
            if self._lease.acquire(timeout=0.25 if left is None else max(0.001, min(0.25, left))):
                ok = True
                break
        if ok:
            self.cancel_event = cancel
        if ok and getattr(self, "_pending_restart", False):
            self._pending_restart = False
            self.shutdown()
        try:
            yield ok
        finally:
            if ok:
                self._job = None
                self.cancel_event = None
                self._lease.release()

    # интерфейс
    def load_clip(self, path: str, sample_fps: float, timeout: float) -> dict:
        raise NotImplementedError

    def call(self, module: str, func: str, args: tuple, kwargs: Optional[dict], timeout: float) -> CallResult:
        raise NotImplementedError

    def request(self, cmd: str, payload: dict, timeout: float) -> Any:
        raise NotImplementedError

    def restart(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def reconfigure(self, cfg: Mapping) -> None:
        self.cfg = cfg

    def stats(self) -> dict:
        return {"mode": self.mode, "restarts": self.restarts, "kills": self.kills, "poisoned": self.poisoned}


class InlineExecutor(_BaseExecutor):
    """В текущем процессе. Таймаут — поток-наблюдатель; зависший вызов продолжает работать (CUDA не
    останавливается) → исполнитель помечается poisoned. Только для тестов/отладки/CPU."""

    mode = "inline"

    def __init__(self, cfg: Mapping) -> None:
        super().__init__(cfg)
        from .config import cfg_get

        self.rt = Runtime(mock_missing=bool(cfg.get("mock")) and bool(cfg.get("mock_missing_modules", True)),
                          allow_fallback_ingest=bool(cfg_get(cfg, "executor.allow_fallback_ingest", True)),
                          frame_cache_mb=float(cfg_get(cfg, "frames.cache_mb", 512)))

    def _run(self, fn: Any, timeout: float) -> tuple[str, Any]:
        box: dict[str, Any] = {}

        def target() -> None:
            try:
                box["value"] = fn()
            except BaseException as exc:  # noqa: BLE001
                box["exc"] = exc
                box["trace"] = traceback.format_exc(limit=8)

        th = threading.Thread(target=target, name="sentinel-inline-stage", daemon=True)
        th.start()
        end = time.monotonic() + max(0.0, timeout)
        while th.is_alive() and time.monotonic() < end and not self.cancelled():
            th.join(timeout=min(0.25, max(0.0, end - time.monotonic())))
        if th.is_alive():
            self.poisoned = True
            return ("cancelled" if self.cancelled() else "timeout"), None
        if "exc" in box:
            return "error", (box["exc"], box.get("trace", ""))
        return "ok", box.get("value")

    def load_clip(self, path: str, sample_fps: float, timeout: float) -> dict:
        status, value = self._run(lambda: self.rt.load_clip(path, sample_fps), timeout)
        if status == "cancelled":
            raise WorkerCancelled("ingest")
        if status == "timeout":
            raise WorkerTimeout(f"ingest не уложился в {timeout:.1f} с")
        if status == "error":
            raise value[0]
        self._job = (path, sample_fps)
        return value

    def call(self, module: str, func: str, args: tuple, kwargs: Optional[dict], timeout: float) -> CallResult:
        t0 = time.perf_counter()
        status, value = self._run(lambda: self.rt.call(module, func, args, kwargs), timeout)
        if status == "cancelled":
            return CallResult("cancelled", error="задача отменена (inline: поток не остановлен)",
                              seconds=time.perf_counter() - t0)
        if status == "timeout":
            return CallResult("timeout", error=f"таймаут {timeout:.1f} с (inline: поток не остановлен, "
                                               f"GPU может быть занят)", seconds=time.perf_counter() - t0)
        if status == "error":
            exc, trace = value
            return CallResult("error", error=f"{type(exc).__name__}: {exc}", seconds=time.perf_counter() - t0,
                              trace=trace)
        return value

    def request(self, cmd: str, payload: dict, timeout: float) -> Any:
        status, value = self._run(lambda: getattr(self.rt, cmd)(**payload), timeout)
        if status == "cancelled":
            raise WorkerCancelled(cmd)
        if status == "timeout":
            raise WorkerTimeout(cmd)
        if status == "error":
            raise value[0]
        return value

    def restart(self) -> None:
        self.rt.drop_clip()
        self.restarts += 1

    def release(self) -> dict:
        """inline: выгрузить модели всех модулей ml.* (хуки unload + реестр), сбросить клип."""
        mods = sorted(m for m in sys.modules if m.startswith("ml.") and m.count(".") == 1)
        self.rt.drop_clip()
        return {"mode": self.mode, "action": "unload", **self.rt.unload(mods)}

    def reconfigure(self, cfg: Mapping) -> None:
        from .config import cfg_get

        self.cfg = cfg
        self.rt.mock_missing = bool(cfg.get("mock")) and bool(cfg.get("mock_missing_modules", True))
        self.rt.allow_fallback_ingest = bool(cfg_get(cfg, "executor.allow_fallback_ingest", True))


class ProcessExecutor(_BaseExecutor):
    """Отдельный процесс-воркер. Убиваемый: таймаут/краш → kill → новый процесс по требованию."""

    mode = "process"

    def __init__(self, cfg: Mapping) -> None:
        super().__init__(cfg)
        self._proc: Any = None
        self._conn: Any = None
        self._io = threading.Lock()
        atexit.register(self.shutdown)

    # ---------------------------------------------------------------- жизненный цикл
    def _init_payload(self) -> dict:
        from .config import cfg_get

        env = {k: v for k, v in os.environ.items() if k.startswith(("SENTINEL_", "HF_", "TRANSFORMERS_", "CUDA_"))}
        if self.cfg.get("mock"):
            env["SENTINEL_MOCK"] = "1"
        return {"env": env, "sys_path": list(cfg_get(self.cfg, "executor.extra_sys_path", []) or []) + sys.path,
                "mock_missing": bool(self.cfg.get("mock")) and bool(self.cfg.get("mock_missing_modules", True)),
                "allow_fallback_ingest": bool(cfg_get(self.cfg, "executor.allow_fallback_ingest", True)),
                "frame_cache_mb": float(cfg_get(self.cfg, "frames.cache_mb", 512))}

    def alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def reconfigure(self, cfg: Mapping) -> None:
        """Новый конфиг: если меняются параметры воркера (mock и т.п.) — воркер перезапустится лениво."""
        old = self._init_payload()
        self.cfg = cfg
        new = self._init_payload()
        if {k: v for k, v in old.items() if k != "sys_path"} != {k: v for k, v in new.items() if k != "sys_path"}:
            if self._lease.locked():
                self._pending_restart = True  # идёт задача — перезапуск при следующей аренде
            else:
                with self._io:
                    self.shutdown()

    def _spawn(self) -> None:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe(duplex=True)
        proc = ctx.Process(target=_worker_main, args=(child, self._init_payload()),
                           name="sentinel-gpu-worker", daemon=False)
        proc.start()
        child.close()
        self._proc, self._conn = proc, parent

    def kill(self) -> None:
        proc, conn = self._proc, self._conn
        self._proc = self._conn = None
        if proc is None:
            return
        with contextlib.suppress(Exception):
            if proc.is_alive():
                proc.kill()
            proc.join(timeout=10)
        with contextlib.suppress(Exception):
            conn.close()
        self.kills += 1

    def restart(self) -> None:
        self.kill()
        self.restarts += 1

    def shutdown(self) -> None:
        proc, conn = self._proc, self._conn
        if proc is None:
            return
        with contextlib.suppress(Exception):
            conn.send(("stop", {}))
            proc.join(timeout=5)
        if proc.is_alive():
            with contextlib.suppress(Exception):
                proc.kill()
                proc.join(timeout=5)
        self._proc = self._conn = None

    # ---------------------------------------------------------------- RPC
    def request(self, cmd: str, payload: dict, timeout: float) -> Any:
        with self._io:
            if not self.alive():
                self._spawn()
                if self._job is not None and cmd != "load_clip":  # воркер перезапущен посреди задачи
                    self._send_recv("load_clip", {"path": self._job[0], "sample_fps": self._job[1]}, timeout)
            return self._send_recv(cmd, payload, timeout)

    def _send_recv(self, cmd: str, payload: dict, timeout: float) -> Any:
        try:
            self._conn.send((cmd, payload))
            end = time.monotonic() + max(0.0, timeout)
            ready = False
            while not ready:
                if self.cancelled():
                    self.kill()  # отмена: процесс умирает → CUDA-контекст и модели выгружены
                    raise WorkerCancelled(f"{cmd}: задача отменена — воркер остановлен")
                left = end - time.monotonic()
                if left <= 0:
                    break
                ready = self._conn.poll(min(0.25, left))
        except (OSError, EOFError, BrokenPipeError) as exc:
            self.kill()
            raise WorkerCrashed(f"канал воркера: {exc}") from exc
        if not ready:
            self.kill()  # единственный надёжный способ остановить CUDA-работу и вернуть VRAM
            raise WorkerTimeout(f"{cmd}: нет ответа за {timeout:.1f} с — воркер убит")
        try:
            reply = self._conn.recv()
        except (EOFError, OSError) as exc:
            code = self._proc.exitcode if self._proc is not None else None
            self.kill()
            raise WorkerCrashed(f"воркер завершился (exit={code})") from exc
        if reply[0] == "ok":
            return reply[1]
        raise RuntimeError(reply[1] + (f"\n{reply[2]}" if len(reply) > 2 and reply[2] else ""))

    def load_clip(self, path: str, sample_fps: float, timeout: float) -> dict:
        info = self.request("load_clip", {"path": path, "sample_fps": sample_fps}, timeout)
        self._job = (path, sample_fps)
        return info

    def call(self, module: str, func: str, args: tuple, kwargs: Optional[dict], timeout: float) -> CallResult:
        t0 = time.perf_counter()
        try:
            return self.request("call", {"module": module, "func": func, "args": args, "kwargs": kwargs or {}},
                                timeout)
        except WorkerTimeout as exc:
            return CallResult("timeout", error=str(exc), seconds=time.perf_counter() - t0)
        except WorkerCancelled as exc:
            return CallResult("cancelled", error=str(exc), seconds=time.perf_counter() - t0)
        except WorkerCrashed as exc:
            return CallResult("crashed", error=str(exc), seconds=time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001
            return CallResult("error", error=f"{type(exc).__name__}: {exc}", seconds=time.perf_counter() - t0)

    def stats(self) -> dict:
        st = super().stats()
        st["pid"] = self._proc.pid if self.alive() else None
        return st


_EXECUTOR: Optional[_BaseExecutor] = None
_EXECUTOR_LOCK = threading.Lock()


def get_executor(cfg: Mapping) -> _BaseExecutor:
    """Исполнитель процесса (синглтон). Режим: cfg executor.mode / env SENTINEL_EXECUTOR."""
    from .config import cfg_get

    global _EXECUTOR
    mode = (os.environ.get("SENTINEL_EXECUTOR") or cfg_get(cfg, "executor.mode", "process") or "process").lower()
    with _EXECUTOR_LOCK:
        if _EXECUTOR is not None and _EXECUTOR.mode == mode:
            _EXECUTOR.reconfigure(cfg)
            return _EXECUTOR
        if _EXECUTOR is not None:
            _EXECUTOR.shutdown()
        _EXECUTOR = ProcessExecutor(cfg) if mode == "process" else InlineExecutor(cfg)
        return _EXECUTOR


def reset_executor() -> None:
    """Для тестов и смены конфигурации: остановить воркер и забыть исполнитель."""
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is not None:
            _EXECUTOR.shutdown()
        _EXECUTOR = None
