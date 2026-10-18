"""Запуск backend.pipeline с профилем deploy/profiles/<name>.env (воспроизводимо, без deploy/.env с ключами).

    python tools/l2/run_profile.py pose_rules run data/raw/X-15.mov --out runs/l2/pose_rules/X-15.json
    python tools/l2/run_profile.py pose_rules batch data/raw --out runs/l2/pose_rules/preds

Окружение как у сервера (deploy/windows/_env.ps1): офлайн HF, веса из weights/, один GPU-воркер.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def load_profile(name: str) -> dict[str, str]:
    path = REPO / "deploy" / "profiles" / f"{name}.env"
    if not path.exists():
        raise SystemExit(f"нет профиля {path}")
    env: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.split(" #", 1)[0].strip().strip('"').strip("'")
        if v.startswith("/app/"):
            v = str(REPO / v[5:])
        if v:
            env[k.strip()] = v
    return env


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    name, rest = sys.argv[1], sys.argv[2:]
    extra = None
    if rest[:1] == ["--extra-env"]:          # локальные уточнения поверх профиля (напр. путь конфига, под которым лежит кэш)
        extra, rest = rest[1], rest[2:]
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                       "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1", "PYTHONUTF8": "1"})
    os.environ.setdefault("SENTINEL_WEIGHTS", str(REPO / "weights"))
    os.environ.setdefault("SENTINEL_WEIGHTS_DIR", os.environ["SENTINEL_WEIGHTS"])
    os.environ.setdefault("HF_HOME", str(Path(os.environ["SENTINEL_WEIGHTS"]) / "hf"))
    os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
    os.environ["SENTINEL_PROFILE"] = name
    os.environ.update(load_profile(name))
    if extra:
        for raw in Path(extra).read_text(encoding="utf-8").splitlines():
            if raw.strip() and not raw.lstrip().startswith("#") and "=" in raw:
                k, v = raw.split("=", 1)
                os.environ[k.strip()] = v.strip()
    sys.path.insert(0, str(REPO))
    from backend.pipeline.cli import main as cli_main

    return cli_main(rest)


if __name__ == "__main__":
    sys.exit(main())
