#!/usr/bin/env python3
"""Сверка РАБОЧЕЙ venv с lock L1 и требованиями модулей. Ничего не устанавливает и не обновляет.

  python deploy/envcheck.py                 # отчёт + код возврата (1 — блокирующие расхождения)
  python deploy/envcheck.py --json

Блокирует: несовпадение версии «критичного» пакета (torch, transformers, bitsandbytes, …) с lock,
отсутствие пакета из lock, неудовлетворённое требование модуля рантайма (fastapi, uvicorn, …).
Предупреждает: прочие расхождения с lock. Что доставить и КАК (без изменения pinned) — печатает.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name
    from packaging.version import InvalidVersion, Version
except ImportError:  # pragma: no cover
    from pip._vendor.packaging.requirements import InvalidRequirement, Requirement  # type: ignore
    from pip._vendor.packaging.utils import canonicalize_name  # type: ignore
    from pip._vendor.packaging.version import InvalidVersion, Version  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_lock(text: str) -> dict[str, str]:
    """Строки `name==version` (комментарии, опции и маркеры игнорируются)."""
    pins: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith(("#", "-")):
            continue
        try:
            req = Requirement(line.split("\\")[0].strip())
        except InvalidRequirement:
            continue
        specs = list(req.specifier)
        if len(specs) == 1 and specs[0].operator in ("==", "==="):
            pins[canonicalize_name(req.name)] = specs[0].version
    return pins


def installed_versions() -> dict[str, str]:
    from importlib import metadata

    out: dict[str, str] = {}
    for d in metadata.distributions():
        name = d.metadata.get("Name")
        if name:
            out[canonicalize_name(name)] = d.version
    return out


def _same(a: str, b: str) -> bool:
    try:
        return Version(a) == Version(b)  # локальная метка (+cu128) тоже сравнивается
    except InvalidVersion:
        return a == b


@dataclass
class EnvReport:
    lock_path: str
    blocking: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_runtime: list[str] = field(default_factory=list)
    checked: int = 0

    @property
    def ok(self) -> bool:
        return not self.blocking


def check(pins: dict[str, str], installed: dict[str, str], critical: list[str],
          module_reqs: list[str], lock_path: str = "") -> EnvReport:
    rep = EnvReport(lock_path=lock_path)
    crit = {canonicalize_name(c) for c in critical}
    for name, want in sorted(pins.items()):
        rep.checked += 1
        have = installed.get(name)
        if have is None:
            (rep.blocking if name in crit else rep.warnings).append(f"{name}: нет в venv (lock: {want})")
        elif not _same(have, want):
            msg = f"{name}: установлено {have}, в lock {want}"
            (rep.blocking if name in crit else rep.warnings).append(msg)
    for line in module_reqs:
        try:
            req = Requirement(line)
        except InvalidRequirement:
            continue
        if req.marker is not None and not req.marker.evaluate():
            continue
        name = canonicalize_name(req.name)
        have = installed.get(name)
        if have is None:
            rep.missing_runtime.append(str(req))
            rep.blocking.append(f"{name}: нужен модулям ({req}), не установлен")
        elif req.specifier and not req.specifier.contains(have, prereleases=True):
            rep.blocking.append(f"{name}: установлено {have}, модулям нужно {req.specifier}")
    return rep


def module_requirements(root: Path, env_lock: Optional[Path]) -> tuple[list[str], list[str]]:
    """Требования модулей (через merge_requirements, dev-инструменты исключены) + конфликты с lock."""
    try:
        from deploy import merge_requirements as mr
    except ImportError:
        import merge_requirements as mr  # type: ignore
    code, report, text = mr.run(root, "gpu", None, check_only=True, env_lock=env_lock)
    reqs = []
    for line in text.split("\n\n", 1)[-1].splitlines():
        body = line.split("  #", 1)[0].strip()
        if body and not body.startswith(("#", "-")):
            reqs.append(body)
    return reqs, report.conflicts + [f"запрещённый пакет: {b}" for b in report.banned]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    ap.add_argument("--lock", type=Path, default=None, help="по умолчанию env.lock из deploy.yaml")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        from deploy import deploy_config as dc
    except ImportError:
        import deploy_config as dc  # type: ignore
    cfg = dc.load()
    env = cfg.get("env") or {}
    lock = a.lock or dc.repo_path(env.get("lock", ""), a.root)
    if not lock.is_file():
        print(f"[envcheck] lock не найден: {lock} (ветка L1 data/labels должна быть смерджена)", file=sys.stderr)
        return 2
    pins = parse_lock(lock.read_text(encoding="utf-8-sig"))
    reqs, conflicts = module_requirements(a.root, lock)
    rep = check(pins, installed_versions(), env.get("critical") or [], reqs, str(lock))
    rep.blocking += [f"конфликт требований модулей с lock: {c}" for c in conflicts]
    out = {"ok": rep.ok, "lock": rep.lock_path, "checked": rep.checked, "blocking": rep.blocking,
           "warnings": rep.warnings, "missing_runtime": rep.missing_runtime}
    if rep.missing_runtime:
        out["install_hint"] = (f"uv pip install --python <venv python> {' '.join(repr(r) for r in rep.missing_runtime)} "
                               f"-c {lock.as_posix()}   # pinned из lock не меняются")
    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(f"[envcheck] {'OK' if rep.ok else 'FAIL'}: проверено {rep.checked} пинов lock {lock.name}")
        for b in rep.blocking:
            print(f"  BLOCK {b}")
        for w in rep.warnings:
            print(f"  warn  {w}")
        if out.get("install_hint"):
            print(f"  доставить: {out['install_hint']}")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
