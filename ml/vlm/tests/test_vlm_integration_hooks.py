import sys
import types

import ml.vlm as vlm
from ml.vlm import runtime
from ml.vlm.backends.base import VLMBackend


class _Fake(VLMBackend):
    name = "fake"

    def __init__(self):
        self._loaded = False
        self.loads = 0
        self.unloads = 0

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        self.loads += 1
        self._loaded = True

    def unload(self):
        self.unloads += 1
        self._loaded = False


def _install_registry(monkeypatch):
    calls = []

    class Reg:
        def __init__(self):
            self.models = {}

        def get_or_load(self, name, loader, *, vram_gb=0.0, keep_warm=True, unload=None):
            calls.append((name, vram_gb))
            if name not in self.models:
                self.models[name] = (loader(), unload)
            return self.models[name][0]

        def release(self, name):
            obj, unload = self.models.pop(name)
            unload(obj)
            return True

    reg = Reg()
    mod = types.ModuleType("backend.pipeline.registry")
    mod.ModelRegistry = types.SimpleNamespace(peek=lambda: reg)
    monkeypatch.setitem(sys.modules, "backend", types.ModuleType("backend"))
    monkeypatch.setitem(sys.modules, "backend.pipeline", types.ModuleType("backend.pipeline"))
    monkeypatch.setitem(sys.modules, "backend.pipeline.registry", mod)
    return reg, calls


def test_load_goes_through_a03_registry(monkeypatch):
    reg, calls = _install_registry(monkeypatch)
    be = _Fake()
    ok, _ = runtime.ensure_loaded(be)
    assert ok and be.loaded and calls == [("ml.vlm:qwen", 3.5)]
    reg.release("ml.vlm:qwen")  # A03 вытесняет модель под детектор
    assert be.unloads == 1 and not be.loaded
    ok, _ = runtime.ensure_loaded(be)  # следующая проверка — перезагрузка через реестр
    assert ok and be.loaded and be.loads == 2


def test_no_registry_outside_a03(monkeypatch):
    monkeypatch.setitem(sys.modules, "backend.pipeline.registry", None)  # импорт падает
    be = _Fake()
    assert runtime.ensure_loaded(be)[0] and be.loads == 1


def test_model_info_for_cache_key():
    info = vlm.model_info()
    assert info["checkpoint"] == "Qwen/Qwen3-VL-4B-Instruct" and info["prompt_version"].startswith("v1+")
    assert info["verify_mode"] in ("json", "qa") and info["version"] == vlm.__version__


def test_verify_accepts_zones_kwarg(clip):
    from ml.vlm._contract import Event, Zone
    from ml.vlm.backends.mock import MockBackend
    runtime.set_backend(MockBackend(mode="confirm"))
    out = vlm.verify(clip, [Event("V05", 1, 2, 1.5, 0.5, "rules")], 30.0,
                     zones=[Zone("desk", [(0.1, 0.3), (0.9, 0.3), (0.9, 0.6)])])
    assert out[0].extra["vlm"]["status"] == "ok"


def test_stale_registry_entry_is_replaced_and_released(monkeypatch):
    """После unload/смены конфига реестр A03 не должен держать старый экземпляр, а новый — жить вне учёта VRAM."""
    reg, calls = _install_registry(monkeypatch)
    be1 = _Fake()
    assert runtime.ensure_loaded(be1)[0]
    be1.unload()  # выгрузили напрямую (vlm.unload / смена конфига)
    be2 = _Fake()
    assert runtime.ensure_loaded(be2)[0]
    assert reg.models["ml.vlm:qwen"][0] is be2 and be2.loaded and not be1.loaded
    monkeypatch.setattr(runtime, "_backend", be2)
    runtime.unload_backend()
    assert "ml.vlm:qwen" not in reg.models and not be2.loaded  # запись снята, VRAM освобождена


def test_gpu_locks_released_after_backend_exception(clip):
    import threading
    from ml.vlm import gpu
    from ml.vlm._contract import Event

    class Boom(_Fake):
        def generate(self, *a, **k):
            raise RuntimeError("CUDA error")

    ext = threading.Lock()
    vlm.set_gpu_lock(ext)
    try:
        be = Boom()
        be._loaded = True  # «тёплая» модель: падение — внутри generate под GPU-lock
        runtime.set_backend(be)
        out = vlm.verify(clip, [Event("V05", 1, 2, 1.5, 0.6, "rules")], 30.0)
        assert out[0].extra["vlm"]["status"] == "error" and out[0].confidence == 0.6
        assert ext.acquire(timeout=0.1)  # внешний lock A03 отпущен
        ext.release()
        assert gpu.priority_lock().stats()["held"] is False  # и приоритетный lock процесса
    finally:
        vlm.set_gpu_lock(None)
