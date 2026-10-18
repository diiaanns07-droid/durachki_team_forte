"""Настройки API: config.yaml рядом с модулем + переопределение через env SENTINEL_*."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

API_DIR = Path(__file__).resolve().parent
REPO_ROOT = API_DIR.parent.parent
DEFAULT_CONFIG = API_DIR / "config.yaml"

_TRUE = {"1", "true", "yes", "on", "y"}


@dataclass
class Settings:
    data_dir: Path = REPO_ROOT / "data"
    job_ttl_h: float = 72.0
    cleanup_interval_s: float = 3600.0
    max_upload_mb: float = 300.0
    min_free_disk_mb: float = 512.0

    workers: int = 1
    gpu_slots: int = 1
    gpu_lock_scope: str = "job"
    budget_s: float = 270.0
    job_cap_s: float = 120.0
    min_deadline_s: float = 15.0
    fair_share: bool = True
    watchdog_s: float = 285.0
    deadline_grace_s: float = 20.0
    max_queue: int = 12
    max_attempts: int = 3
    max_zombie_workers: int = 2
    zombie_release_gpu: bool = False
    cancel_on_timeout: bool = True
    hung_gpu_s: float = 60.0
    restart_fresh_budget: bool = False
    pass_deadline_at: bool = True

    pipeline: str = "backend.pipeline"
    fake_pipeline: bool = False
    fake_pipeline_s: float = 3.0
    allow_mock: bool = False
    warmup: bool = True
    warmup_wait_s: float = 120.0
    evidence_fallback: bool = True
    evidence_max_offset_s: float = 0.5
    evidence_budget_s: float = 20.0
    video_validation: str = "required"
    copilot_timeout_s: float = 60.0
    min_coverage: float = 0.98
    coverage_adapt_stages: bool = False
    partial_accept: bool = False
    jury_profile: str = "extended"
    jury_embed_rule: bool = False
    max_response_mb: float = 3.0
    dedup: bool = True

    api_keys: list[str] = field(default_factory=list)
    api_keys_file: Path | None = None
    ui_secret: str = ""
    ui_password: str = ""
    ui_enabled: bool = True
    ui_token_ttl_s: float = 43200.0
    cors_origins: list[str] = field(default_factory=list)
    trusted_proxies: list[str] = field(default_factory=lambda: ["127.0.0.1", "::1"])
    docs: bool = True

    rl_ui_session: str = "30/60"
    rl_ui_login: str = "10/60"
    rl_ui_upload: str = "12/60"
    rl_ui_read: str = "600/60"
    rl_ui_ask: str = "20/60"
    rl_auth_fail: str = "300/60"

    frontend_dir: Path = REPO_ROOT / "frontend" / "dist"
    mov_as_mp4: bool = True

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    log_json: bool = True

    # ---- производные ----
    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "jobs.sqlite3"

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def max_response_bytes(self) -> int:
        return int(self.max_response_mb * 1024 * 1024)

    @classmethod
    def from_env(cls, config_path: Path | str | None = None, env: dict[str, str] | None = None,
                 **overrides: Any) -> "Settings":
        """Порядок: дефолты dataclass < config.yaml < env SENTINEL_* < overrides (для тестов)."""
        env = dict(os.environ if env is None else env)
        raw: dict[str, Any] = {}
        path = Path(config_path or env.get("SENTINEL_API_CONFIG") or DEFAULT_CONFIG)
        raw.update(_load_yaml(path))
        names = {f.name: f for f in fields(cls)}
        # синонимы из deploy/ (A17): применяются, только если основной ключ не задан
        for alias, name in _ALIASES.items():
            if env.get(alias) and not env.get("SENTINEL_" + name.upper()):
                raw[name] = env[alias]
        for name in names:
            key = "SENTINEL_" + name.upper()
            if key in env and env[key] != "":
                raw[name] = env[key]
        single = env.get("SENTINEL_API_KEY", "").strip()  # A17 healthcheck: один ключ
        if single:
            raw["api_keys"] = _split(raw.get("api_keys")) + [single]
        raw.update(overrides)
        kwargs = {k: _coerce(names[k].type, v, k) for k, v in raw.items() if k in names}
        s = cls(**kwargs)
        s.data_dir = _abs(s.data_dir)
        s.frontend_dir = _abs(s.frontend_dir)
        if s.api_keys_file is None:
            s.api_keys_file = s.data_dir / "api_keys.txt"
        else:
            s.api_keys_file = _abs(s.api_keys_file)
        s.workers = max(1, int(s.workers))
        s.gpu_slots = max(1, int(s.gpu_slots))
        if s.gpu_lock_scope not in ("job", "none"):
            s.gpu_lock_scope = "job"
        if s.video_validation not in ("required", "best_effort", "off"):
            s.video_validation = "required"
        if s.jury_profile not in ("extended", "strict"):
            s.jury_profile = "extended"
        return s


_ALIASES = {
    "SENTINEL_FRONTEND_DIST": "frontend_dir",   # deploy/.env.example, Dockerfile.gpu, start.ps1
    "SENTINEL_DEADLINE_S": "budget_s",          # deploy/.env.example (240): общий бюджет задачи
}


def _abs(p: Path | str) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        import yaml  # lazy: PyYAML опционален
    except ImportError:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _split(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        items = [str(x) for x in v]
    else:
        items = str(v).replace(";", ",").split(",")
    return [x.strip() for x in items if x and x.strip()]


def _coerce(tp: Any, v: Any, name: str) -> Any:
    t = str(tp)
    if v is None:
        return None
    if t.startswith("list"):
        return _split(v)
    if "Path" in t:
        return Path(str(v)) if str(v) else None
    if t == "bool":
        return v if isinstance(v, bool) else str(v).strip().lower() in _TRUE
    if t == "int":
        return int(float(v))
    if t == "float":
        return float(v)
    return str(v)


def parse_rate(spec: str) -> tuple[int, float]:
    """'30/60' -> (30 запросов, за 60 c). Пусто/0 -> без лимита (0, 0)."""
    try:
        n, per = str(spec).split("/", 1)
        return max(0, int(n)), max(0.001, float(per))
    except Exception:
        return 0, 0.0
