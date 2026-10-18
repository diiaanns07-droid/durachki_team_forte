"""Таблица стадий, разрешение функций модулей (ленивый импорт), идентичность модулей для ключей кэша."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from .config import REPO_ROOT, cfg_get


@dataclass(frozen=True)
class Stage:
    name: str
    label: str          # по-русски, для UI
    module: str
    func: str
    gpu: bool
    weight: float       # доля в общем прогрессе


STAGES: dict[str, Stage] = {s.name: s for s in [
    Stage("ingest", "Декодирование", "ml.ingest", "load_video", False, 0.06),
    Stage("detect", "Поиск людей и объектов", "ml.detect", "detect", True, 0.22),
    Stage("track", "Трекинг", "ml.track", "track", False, 0.06),
    Stage("pose", "Анализ поз", "ml.pose", "estimate", True, 0.10),
    Stage("scene", "Состояние сцены", "ml.scene", "scene_states", False, 0.06),
    Stage("rules", "Правила", "ml.rules", "run_rules", False, 0.03),
    Stage("seqrules", "Правила по последовательностям", "ml.seqrules", "run", True, 0.04),
    Stage("temporal", "Временная модель", "ml.temporal", "score", True, 0.15),
    Stage("refine", "Уточнение окон", "ml.detect", "detect", True, 0.06),
    Stage("vlm_verify", "Проверка VLM", "ml.vlm", "verify", True, 0.14),
    Stage("vlm_discover", "Проверка VLM", "ml.vlm", "discover", True, 0.10),
    Stage("fusion", "Формирование отчёта", "ml.fusion", "fuse", False, 0.08),
]}

DEFAULT_REQUIRED = ("ingest", "detect", "track", "scene", "rules", "fusion")


def required_stages(cfg: Mapping) -> list[str]:
    req = cfg_get(cfg, "required_stages", None)
    return list(req) if req else list(DEFAULT_REQUIRED)


def is_required(stage: str, cfg: Mapping) -> bool:
    return stage in required_stages(cfg)


class ModuleMissing(ImportError):
    """Модуль стадии не установлен/не смерджен."""


def resolve(module: str, func: str, *, mock_missing: bool = False) -> tuple[Callable[..., Any], str, Any]:
    """Возвращает (функция, происхождение, объект-модуль). Происхождение: 'module' | 'builtin-mock'.

    ImportError самого модуля (а не его зависимостей) → ModuleMissing, либо встроенный мок
    при mock_missing. Ошибки импорта зависимостей внутри модуля пробрасываются как есть.
    """
    try:
        mod = importlib.import_module(module)
        fn = getattr(mod, func)
        return fn, "module", mod
    except (ImportError, AttributeError) as exc:
        missing = isinstance(exc, AttributeError) or _is_missing_module(exc, module)
        if not missing:
            raise
        if mock_missing:
            from . import mocks
            fn = mocks.BUILTIN.get(f"{module}.{func}")
            if fn is not None:
                return fn, "builtin-mock", mocks
        raise ModuleMissing(f"{module}.{func} недоступен: {exc}") from exc


def _is_missing_module(exc: ImportError, module: str) -> bool:
    name = getattr(exc, "name", None) or ""
    return isinstance(exc, ModuleNotFoundError) and (module == name or module.startswith(name + "."))


# ---------------------------------------------------------------- идентичность модулей → ключи кэша

_FP_SUFFIXES = (".py", ".yaml", ".yml", ".json", ".txt", ".md", ".jinja", ".j2", ".toml", ".cfg", ".ini")


def code_fingerprint(module: Any) -> str:
    """Хэш содержимого кода/конфигов/промптов пакета модуля (без tests/ и __pycache__)."""
    file = getattr(module, "__file__", None)
    if not file:
        return "nofile"
    path = Path(file)
    root = path.parent if path.name == "__init__.py" else path
    files = sorted(p for p in (root.rglob("*") if root.is_dir() else [root])
                   if p.is_file() and p.suffix.lower() in _FP_SUFFIXES
                   and "tests" not in p.parts and "__pycache__" not in p.parts and p.name != "README.md")
    h = hashlib.sha1()
    for p in files:
        try:
            h.update(str(p.relative_to(root) if root.is_dir() else p.name).encode())
            h.update(p.read_bytes())
        except OSError:
            continue
    return h.hexdigest()[:12]


def _file_sig(path: str) -> Optional[str]:
    """Дешёвая подпись файла весов (без чтения гигабайт): путь + размер + mtime_ns."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return f"{Path(path).name}:{st.st_size}:{st.st_mtime_ns}"


def module_identity(module: Any, origin: str = "module") -> dict:
    """Всё, что определяет выход стадии, кроме входных данных:
    версия + код/конфиги/промпты + веса (checkpoint/revision) + fold/split обучаемых моделей.

    Модули могут (необязательно) экспортировать ``model_info() -> dict`` с ключами
    ``checkpoint`` (путь или список путей), ``revision``, ``fold``, ``split``, ``prompt_version``…
    """
    if module is None:
        return {"origin": origin}
    ident: dict[str, Any] = {
        "origin": origin,
        "version": str(getattr(module, "__version__", getattr(module, "VERSION", "0"))),
        "code": code_fingerprint(module) if origin == "module" else "builtin",
    }
    info_fn = getattr(module, "model_info", None)
    if origin == "module" and callable(info_fn):
        try:
            info = info_fn() or {}
        except Exception as exc:  # noqa: BLE001
            info = {"error": f"{type(exc).__name__}: {exc}"}
        ckpt = info.get("checkpoint") if isinstance(info, dict) else None
        paths = ckpt if isinstance(ckpt, (list, tuple)) else ([ckpt] if ckpt else [])
        ident["model_info"] = info
        ident["checkpoint_sig"] = [_file_sig(str(p)) for p in paths]
    if origin == "module":
        ident["models"] = module_models(module)
    for env in ("SENTINEL_FOLD", "SENTINEL_SPLIT"):
        if os.environ.get(env):
            ident[env.lower()] = os.environ[env]
    # Настройки модуля через env (например SENTINEL_DETECT_EXTRAS=open_vocab для ml.detect) меняют
    # выход его стадии, поэтому входят в её ключ кэша. Только переменные своего модуля, без секретов.
    if origin == "module":
        prefix = "SENTINEL_" + str(getattr(module, "__name__", "")).rsplit(".", 1)[-1].upper() + "_"
        behaviour_env = {
            k: v for k, v in sorted(os.environ.items())
            if k.startswith(prefix) and not any(s in k for s in _SECRET_ENV_MARKERS)
        }
        if behaviour_env:
            ident["env"] = behaviour_env
    return ident


_SECRET_ENV_MARKERS = ("KEY", "PASSWORD", "SECRET", "TOKEN")


def _hf_cache_dirs() -> list[Path]:
    dirs = []
    for env in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(env):
            dirs.append(Path(os.environ[env]))
    if os.environ.get("HF_HOME"):
        dirs.append(Path(os.environ["HF_HOME"]) / "hub")
    dirs += [REPO_ROOT / "weights" / "hf" / "hub", Path.home() / ".cache" / "huggingface" / "hub"]
    return dirs


def hf_revision(hf_id: str) -> Optional[str]:
    """Ревизия (commit) офлайн-снапшота модели в HF-кэше: models--org--name/refs/main. Нет — None."""
    if not hf_id or "://" in hf_id or "/" not in hf_id:
        return None
    name = "models--" + hf_id.strip("/").replace("/", "--")
    for d in _hf_cache_dirs():
        ref = d / name / "refs" / "main"
        try:
            return ref.read_text(encoding="utf-8").strip()[:40] or None
        except OSError:
            continue
    return None


def module_models(module: Any) -> list[dict]:
    """Модели из <пакет модуля>/models.json (+ ревизия весов из HF-кэша) — для ключа кэша и provenance."""
    file = getattr(module, "__file__", None)
    if not file:
        return []
    path = Path(file).parent / "models.json"
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for m in items if isinstance(items, list) else []:
        if not isinstance(m, dict):
            continue
        hf = str(m.get("hf_id_or_url") or m.get("hf_id") or "")
        out.append({"name": str(m.get("name", hf)), "hf_id_or_url": hf, "license": m.get("license"),
                    "revision": hf_revision(hf)})
    return out


def catalog_hash(catalog: Mapping) -> str:
    raw = json.dumps(catalog, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def load_catalog_safe() -> tuple[dict, str, Optional[str]]:
    """(каталог, источник, предупреждение). Нормы — только из norms/catalog.yaml (правило 5)."""
    try:
        norms = importlib.import_module("norms")
        cat = norms.load_catalog()
        if isinstance(cat, dict) and cat.get("types"):
            return cat, "norms.load_catalog", None
        warn = "norms.load_catalog() вернул пустой каталог"
    except Exception as exc:  # noqa: BLE001
        warn = f"norms.load_catalog недоступен: {type(exc).__name__}: {exc}"
    path = REPO_ROOT / "norms" / "catalog.yaml"
    if path.exists():
        try:
            import yaml  # noqa: WPS433

            with open(path, "r", encoding="utf-8") as fh:
                cat = yaml.safe_load(fh) or {}
            if isinstance(cat, dict) and cat.get("types"):
                return cat, str(path.relative_to(REPO_ROOT)), warn
        except Exception as exc:  # noqa: BLE001
            warn = f"{warn}; norms/catalog.yaml не читается: {exc}"
    return {"version": 1, "types": {}}, "empty", (warn or "") + "; каталог норм пуст — ссылки на нормы не выдаются"
