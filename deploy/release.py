#!/usr/bin/env python3
"""Проверенная версия и откат — без reset --hard и без потери данных.

  python deploy/release.py mark-verified [--note "A16 acceptance ok"] [--report runs/eval/api1/metrics.json ...]
  python deploy/release.py diff                      # что изменилось с проверенной версии
  python deploy/release.py rollback [--to latest|<id>] [--dry-run]

mark-verified записывает runs/deploy/verified/<id>.json: SHA, грязные файлы, sha256 конфигов, lock-ов,
weights.lock.json, отчётов приёмки; для deploy/.env — ТОЛЬКО хэш (секреты не копируются).
rollback: незакоммиченное → `git stash push -u` (ничего не удаляется), затем `git switch --detach <SHA>`,
сверка хэшей конфигов с записью. data/, weights/, runs/ не трогаются (они вне git).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKED_CONFIGS = ("deploy/config/deploy.yaml", "backend/api/config.yaml", "backend/pipeline/config.yaml",
                   "ml/*/config.yaml", "copilot/config.yaml", "norms/catalog.yaml",
                   "tools/dataprep/requirements-l1.lock.txt", "deploy/weights.lock.json")
SECRET_FILES = ("deploy/.env",)


def git(root: Path, *args: str, check: bool = True) -> str:
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=60)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def config_hashes(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for pat in TRACKED_CONFIGS + SECRET_FILES:
        for p in sorted(root.glob(pat)):
            if p.is_file():
                out[p.relative_to(root).as_posix()] = sha256(p)
    extra = os.environ.get("SENTINEL_PIPELINE_CONFIG")  # оверлей вне git (если задан)
    if extra and Path(extra).is_file():
        out[f"env:SENTINEL_PIPELINE_CONFIG={extra}"] = sha256(Path(extra))
    return out


def verified_dir(root: Path) -> Path:
    return Path(os.environ.get("SENTINEL_LOG_DIR") or root / "runs" / "deploy") / "verified"


def mark_verified(root: Path, note: str = "", reports: Optional[list[Path]] = None) -> dict[str, Any]:
    rec = {
        "id": time.strftime("%Y%m%d_%H%M%S"), "marked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sha": git(root, "rev-parse", "HEAD"), "branch": git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": [line[3:] for line in git(root, "status", "--porcelain").splitlines() if line.strip()],
        "configs": config_hashes(root), "note": note,
        "reports": {str(p): sha256(p) for p in (reports or []) if Path(p).is_file()},
    }
    d = verified_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{rec['id']}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (d / "latest.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rec


def load_record(root: Path, which: str = "latest") -> dict[str, Any]:
    p = verified_dir(root) / f"{which}.json"
    if not p.is_file():
        raise FileNotFoundError(f"нет записи {p} — сначала: release.py mark-verified")
    return json.loads(p.read_text(encoding="utf-8"))


def diff(root: Path, rec: dict[str, Any]) -> dict[str, Any]:
    now = config_hashes(root)
    changed = sorted(k for k in set(now) | set(rec["configs"]) if now.get(k) != rec["configs"].get(k))
    return {"verified_sha": rec["sha"], "head": git(root, "rev-parse", "HEAD"),
            "commits_since": git(root, "rev-list", "--count", f"{rec['sha']}..HEAD", check=False) or "?",
            "changed_configs": changed,
            "dirty": [line[3:] for line in git(root, "status", "--porcelain").splitlines() if line.strip()]}


def rollback(root: Path, rec: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    steps: list[str] = []
    dirty = git(root, "status", "--porcelain")
    if dirty:
        steps.append(f"git stash push -u -m rollback-to-{rec['sha'][:12]}  (незакоммиченное сохраняется)")
    steps.append(f"git switch --detach {rec['sha']}")
    out: dict[str, Any] = {"to": rec["sha"], "steps": steps, "dry_run": dry_run}
    if dry_run:
        return out
    if dirty:
        git(root, "stash", "push", "-u", "-m", f"rollback-to-{rec['sha'][:12]}")
    git(root, "switch", "--detach", rec["sha"])
    now = config_hashes(root)
    out["config_mismatch"] = sorted(k for k, v in rec["configs"].items() if now.get(k) != v)
    out["note"] = ("перезапустите сервис: deploy\\windows\\stop.ps1 → run_all.ps1; "
                   "вернуться: git switch <ветка> && git stash pop")
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("mark-verified", "diff", "rollback"))
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    ap.add_argument("--note", default="")
    ap.add_argument("--report", type=Path, action="append", default=[])
    ap.add_argument("--to", default="latest")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "mark-verified":
            res = mark_verified(a.root, a.note, a.report)
        elif a.cmd == "diff":
            res = diff(a.root, load_record(a.root, a.to))
        else:
            res = rollback(a.root, load_record(a.root, a.to), a.dry_run)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"[release] {exc}", file=sys.stderr)
        return 1
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 1 if res.get("config_mismatch") else 0


if __name__ == "__main__":
    sys.exit(main())
