"""Фикстуры тестов оркестратора: изоляция env/каталогов, синтетическое видео, фейковые модули ml.*."""
from __future__ import annotations

import sys
import textwrap
import types
from pathlib import Path
from typing import Any, Callable

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.pipeline import mocks  # noqa: E402
from backend.pipeline.media import make_synthetic_video  # noqa: E402
from backend.pipeline.registry import ModelRegistry  # noqa: E402
from backend.pipeline.worker import reset_executor  # noqa: E402

_ENV_CLEAR = ("SENTINEL_MOCK", "SENTINEL_PIPELINE_CONFIG", "SENTINEL_NO_VLM", "SENTINEL_CACHE", "SENTINEL_FOLD",
              "SENTINEL_SPLIT", "SENTINEL_SAMPLE_FPS", "SENTINEL_VLM_POLICY", "SENTINEL_FAKE_SLEEP")


def pytest_configure(config: Any) -> None:
    config.addinivalue_line("markers", "gpu: требует CUDA")
    config.addinivalue_line("markers", "slow: медленный тест (процесс-воркер, spawn)")


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import os

    saved = {k: v for k, v in os.environ.items() if k.startswith("SENTINEL_")}  # ничего не утекает в чужие тесты
    for k in _ENV_CLEAR:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SENTINEL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SENTINEL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SENTINEL_DEVICE", "cpu")
    monkeypatch.setenv("SENTINEL_EXECUTOR", "inline")
    reset_executor()
    ModelRegistry.reset()
    yield
    reset_executor()
    ModelRegistry.reset()
    for k in [k for k in os.environ if k.startswith("SENTINEL_")]:
        if k not in saved:
            del os.environ[k]
    os.environ.update(saved)


@pytest.fixture(scope="session")
def synth_video(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("video") / "synth.mp4"
    return make_synthetic_video(path, seconds=4.0, event_at=2.0)


FAST = {  # короткие таймауты для тестов дедлайна
    "timeouts": {"min_s": 0.6, "hard_factor": 1.0, "cpu_factor": 1.0},
    "reserve_s": {"fusion": 0.5, "fusion_min": 0.3, "vlm_min_budget": 0.0, "essential_min": 0.1},
}


@pytest.fixture
def fake_ml(monkeypatch: pytest.MonkeyPatch) -> Callable[..., dict]:
    """install(overrides={"ml.detect.detect": fn}, drop=("ml.vlm",), ingest=True) → фейковые ml.* в sys.modules."""

    def install(overrides: dict[str, Callable] | None = None, drop: tuple[str, ...] = (),
                versions: dict[str, str] | None = None, ingest: bool = False) -> dict[str, types.ModuleType]:
        funcs = dict(mocks.BUILTIN)
        if ingest:   # «настоящий» ml.ingest для тестов прогрева без mock-режима (декодер OpenCV)
            from backend.pipeline.media import SimpleClip

            funcs["ml.ingest.load_video"] = lambda path, sample_fps=5.0: SimpleClip(path, sample_fps)
        funcs.update(overrides or {})
        mods: dict[str, types.ModuleType] = {}
        root = types.ModuleType("ml")
        root.__path__ = []  # пакет
        mods["ml"] = root
        for full, fn in funcs.items():
            modname, func = full.rsplit(".", 1)
            if modname in drop:
                continue
            mod = mods.setdefault(modname, types.ModuleType(modname))
            mod.__version__ = (versions or {}).get(modname, "fake-1")
            setattr(mod, func, fn)
        for name in list(sys.modules):
            if name == "ml" or name.startswith("ml."):
                monkeypatch.delitem(sys.modules, name, raising=False)
        for name, mod in mods.items():
            monkeypatch.setitem(sys.modules, name, mod)
            if "." in name:
                setattr(root, name.split(".", 1)[1], mod)
        return mods

    return install


@pytest.fixture
def fake_ml_files(tmp_path: Path) -> Callable[..., Path]:
    """Фейковый пакет ml/ НА ДИСКЕ — для процесса-воркера (spawn не видит sys.modules родителя)."""

    def write(detect_sleep_env: bool = True) -> Path:
        base = tmp_path / "fakepkg"
        pkgs = {"ingest": None, "detect": "detect", "track": "track", "pose": "estimate, pose_signals",
                "scene": "scene_states", "rules": "run_rules", "temporal": "score", "vlm": "verify, discover"}
        (base / "ml").mkdir(parents=True, exist_ok=True)
        (base / "ml" / "__init__.py").write_text("", encoding="utf-8")
        for name, funcs in pkgs.items():
            if funcs is None:
                continue
            d = base / "ml" / name
            d.mkdir(exist_ok=True)
            body = f"__version__ = 'file-fake-1'\nfrom backend.pipeline.mocks import {funcs}\n"
            if name == "detect":
                body = textwrap.dedent("""\
                    import os, time
                    __version__ = 'file-fake-1'
                    from backend.pipeline.mocks import detect as _detect
                    def detect(clip, every_n=1):
                        s = float(os.environ.get('SENTINEL_FAKE_SLEEP') or 0)
                        if s:
                            time.sleep(s)
                        return _detect(clip, every_n)
                    """)
            (d / "__init__.py").write_text(body, encoding="utf-8")
        return base

    return write
