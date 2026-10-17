#!/usr/bin/env python3
"""Сводит requirements.txt всех модулей в deploy/requirements.in и строит lock НАСТОЯЩИМ resolver-ом.

Скрипт НЕ выбирает версии сам. Он:
  1) собирает все */requirements.txt (+ deploy/platform.in);
  2) ищет конфликты: несовместимые диапазоны одного пакета, требование модуля против пина lock L1
     (tools/dataprep/requirements-l1.lock.txt — источник истины окружения), взаимоисключающие
     дистрибутивы (opencv-python vs opencv-python-headless), запрещённые лицензии (ultralytics …);
  3) пишет пересечение ограничений в deploy/requirements.in (без «максимальной версии»);
  4) с --lock TARGET вызывает `uv pip compile -c <lock L1>` (resolver) → deploy/locks/TARGET.lock.txt:
     пины L1 не меняются, добавляются только недостающие пакеты.

Конфликт = код возврата 1 и отчёт; решение принимают люди (владелец модуля / A01 / интегратор)
правкой requirements модуля или записью в deploy/constraints.txt (передаётся resolver-у как -c).

  python deploy/merge_requirements.py                       # сбор + проверка → requirements.in
  python deploy/merge_requirements.py --check               # только отчёт, файлы не пишет
  python deploy/merge_requirements.py --lock windows-cu128  # + lock для ноутбука (нужен uv и сеть)
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    from packaging.markers import Marker
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.utils import canonicalize_name
    from packaging.version import InvalidVersion, Version
except ImportError:  # голый python:3.11-slim — берём vendored-копию из pip
    from pip._vendor.packaging.markers import Marker  # type: ignore
    from pip._vendor.packaging.requirements import InvalidRequirement, Requirement  # type: ignore
    from pip._vendor.packaging.specifiers import SpecifierSet  # type: ignore
    from pip._vendor.packaging.utils import canonicalize_name  # type: ignore
    from pip._vendor.packaging.version import InvalidVersion, Version  # type: ignore

try:
    from deploy.license_policy import BANNED_PACKAGES
except ImportError:  # запуск как `python deploy/merge_requirements.py`
    from license_policy import BANNED_PACKAGES  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "env", "__pycache__", "weights", "data",
             "cache", "dist", "build", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
TORCH_FAMILY = {"torch", "torchvision", "torchaudio"}
DEFAULT_ENV_LOCK = Path("tools") / "dataprep" / "requirements-l1.lock.txt"
# Среда, для которой вычисляются маркеры (; python_version …, ; sys_platform …): прод L1 — Windows, py3.11.
TARGET_ENVS: dict[str, dict[str, str]] = {
    "windows": {"python_version": "3.11", "python_full_version": "3.11.9", "sys_platform": "win32",
                "platform_system": "Windows", "os_name": "nt", "platform_machine": "AMD64",
                "implementation_name": "cpython", "platform_python_implementation": "CPython",
                "platform_release": "10", "platform_version": "10.0", "extra": ""},
    "linux": {"python_version": "3.11", "python_full_version": "3.11.9", "sys_platform": "linux",
              "platform_system": "Linux", "os_name": "posix", "platform_machine": "x86_64",
              "implementation_name": "cpython", "platform_python_implementation": "CPython",
              "platform_release": "", "platform_version": "", "extra": ""},
}

# Цели lock-файлов. torch_backend → uv сам ходит в индекс PyTorch за torch/torchvision.
LOCK_TARGETS: dict[str, dict[str, str]] = {
    "windows-cu128": {"python_platform": "x86_64-pc-windows-msvc", "torch_backend": "cu128", "profile": "gpu"},
    "linux-cu128": {"python_platform": "x86_64-manylinux_2_28", "torch_backend": "cu128", "profile": "gpu"},
    # cu124 — только для torch <= 2.6 (lock L1: torch 2.11.0+cu128 → использовать cu128)
    "windows-cu124": {"python_platform": "x86_64-pc-windows-msvc", "torch_backend": "cu124", "profile": "gpu"},
    "windows-cpu": {"python_platform": "x86_64-pc-windows-msvc", "torch_backend": "cpu", "profile": "cpu"},
    "linux-cu124": {"python_platform": "x86_64-manylinux_2_28", "torch_backend": "cu124", "profile": "gpu"},
    "linux-cpu": {"python_platform": "x86_64-manylinux_2_28", "torch_backend": "cpu", "profile": "cpu"},
}

# Взаимоисключающие дистрибутивы одного import-пакета. Детерминированная замена (не выбор версии):
# сервер без GUI → headless; contrib ⊃ base. Каждая замена печатается в отчёте и шапке .in.
OPENCV = {"opencv-python", "opencv-python-headless", "opencv-contrib-python", "opencv-contrib-python-headless"}
ONNXRUNTIME = {"onnxruntime", "onnxruntime-gpu", "onnxruntime-directml", "onnxruntime-openvino"}

DEV_PATTERNS = ("pytest", "pytest-*", "ruff", "black", "mypy", "flake8", "isort", "pre-commit",
                "ipykernel", "ipython", "jupyter*", "notebook", "coverage", "types-*")

# Известные проблемы установки — только предупреждение, решение за владельцем модуля.
KNOWN_TROUBLE = {
    "flash-attn": "нет колёс под Windows/cu124 — сборка из исходников с nvcc; transformers умеет sdpa",
    "xformers": "колёса жёстко пинят конкретный torch — почти всегда конфликт с platform.in",
    "llama-cpp-python": "без готового CUDA-колеса компилируется; для copilot лучше llama.cpp server",
    "decord": "последний релиз 0.6.0 без колёс cp311 под Windows; альтернатива — PyAV/eva-decord",
    "bitsandbytes": "проверить версию с колёсами под Windows (>=0.43)",
}


@dataclass
class ReqLine:
    name: str                  # canonical
    extras: frozenset[str]
    specifier: str
    marker: str
    url: str
    source: str
    lineno: int
    raw: str


@dataclass
class Merged:
    name: str
    marker: str
    extras: set[str] = field(default_factory=set)
    specs: list[tuple[str, str]] = field(default_factory=list)  # (specifier, source)
    urls: set[str] = field(default_factory=set)
    sources: list[str] = field(default_factory=list)
    note: str = ""

    def spec_set(self) -> SpecifierSet:
        return SpecifierSet(",".join(s for s, _ in self.specs if s))

    def line(self) -> str:
        extras = f"[{','.join(sorted(self.extras))}]" if self.extras else ""
        if self.urls:
            body = f"{self.name}{extras} @ {sorted(self.urls)[0]}"
        else:
            body = f"{self.name}{extras}{self.spec_set()}"
        if self.marker:
            body += f" ; {self.marker}"
        srcs = ", ".join(sorted(set(self.sources)))
        return f"{body}  # {srcs}" + (f" ({self.note})" if self.note else "")


@dataclass
class Report:
    sources: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    banned: list[str] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    options: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)


def find_requirement_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        if "requirements.txt" in filenames:
            out.append(Path(dirpath) / "requirements.txt")
    platform_in = root / "deploy" / "platform.in"
    if platform_in.is_file():
        out.append(platform_in)
    return sorted(out)


def _logical_lines(text: str) -> list[tuple[int, str]]:
    lines, buf, start = [], "", 0
    for i, raw in enumerate(text.splitlines(), 1):
        line = re.sub(r"(^|\s)#.*$", "", raw).rstrip()
        if not buf:
            start = i
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        buf += line
        if buf.strip():
            lines.append((start, buf.strip()))
        buf = ""
    if buf.strip():
        lines.append((start, buf.strip()))
    return lines


def parse_file(path: Path, root: Path, report: Report, _seen: set[Path] | None = None) -> list[ReqLine]:
    seen = _seen if _seen is not None else set()
    path = path.resolve()
    if path in seen:
        return []
    seen.add(path)
    try:
        rel = path.relative_to(root.resolve()).as_posix()
    except ValueError:
        rel = path.as_posix()
    out: list[ReqLine] = []
    for lineno, line in _logical_lines(path.read_text(encoding="utf-8-sig")):
        where = f"{rel}:{lineno}"
        m = re.match(r"^(-r|--requirement|-c|--constraint)\s*=?\s*(\S+)$", line)
        if m:
            target = (path.parent / m.group(2)).resolve()
            if not target.is_file():
                report.warnings.append(f"{where}: не найден {m.group(2)}")
            elif m.group(1) in ("-r", "--requirement"):
                out += parse_file(target, root, report, seen)
            else:
                report.constraints.append(str(target))
            continue
        if line.startswith("-"):
            if line.startswith(("-e", "--editable")):
                report.warnings.append(f"{where}: editable пропущен в prod: {line}")
            elif "download.pytorch.org" in line:
                report.dropped.append(f"{where}: {line} (torch ставится через --torch-backend)")
            else:
                if line not in report.options:
                    report.options.append(line)
                report.warnings.append(f"{where}: опция индекса/сборки перенесена как есть: {line}")
            continue
        line = re.sub(r"\s--hash[=\s]\S+", "", line)
        try:
            req = Requirement(line)
        except InvalidRequirement as exc:
            report.warnings.append(f"{where}: не разобрано ({exc}): {line}")
            continue
        out.append(ReqLine(
            name=canonicalize_name(req.name), extras=frozenset(req.extras), specifier=str(req.specifier),
            marker=str(req.marker) if req.marker else "", url=req.url or "", source=rel, lineno=lineno, raw=line,
        ))
    return out


def _versions_in(spec: str) -> list[Version]:
    out = []
    for m in re.finditer(r"(?:===|==|!=|~=|>=|<=|>|<)\s*([^,;\s]+)", spec):
        v = m.group(1).rstrip(".*")
        try:
            out.append(Version(v))
        except InvalidVersion:
            pass
    return out


def _candidates(versions: list[Version]) -> list[Version]:
    """Пробные версии для проверки совместимости диапазонов (без индекса пакетов)."""
    cands: set[Version] = set()
    for v in versions:
        rel = list(v.release) + [0] * (3 - len(v.release))
        a, b, c = rel[:3]
        cands.add(v)
        for t in ((a, b, c + 1), (a, b + 1, 0), (a + 1, 0, 0), (a, b, c, 1)):
            cands.add(Version(".".join(map(str, t))))
        if c > 0:
            cands.add(Version(f"{a}.{b}.{c - 1}"))
        if b > 0:
            cands.add(Version(f"{a}.{b - 1}.9999"))
        if a > 0:
            cands.add(Version(f"{a - 1}.9999.9999"))
    if not cands:
        cands.add(Version("1.0"))
    return sorted(cands)


def satisfiable(specs: list[str]) -> bool:
    joined = ",".join(s for s in specs if s)
    if not joined:
        return True
    ss = SpecifierSet(joined)
    return any(ss.contains(v, prereleases=True) for v in _candidates(_versions_in(joined)))


def _is_dev(name: str) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in DEV_PATTERNS)


def _family_target(names: set[str], family: set[str], profile: str) -> str | None:
    present = names & family
    if not present:
        return None
    if family is OPENCV:
        return "opencv-contrib-python-headless" if any("contrib" in n for n in present) else "opencv-python-headless"
    return "onnxruntime-gpu" if profile == "gpu" else "onnxruntime"


def load_env_lock(path: Path) -> dict[str, str]:
    """Пины `name==version` lock-файла (комментарии/опции игнорируются)."""
    pins: dict[str, str] = {}
    for _, line in _logical_lines(path.read_text(encoding="utf-8-sig")):
        if line.startswith("-"):
            continue
        try:
            req = Requirement(re.sub(r"\s--hash[=\s]\S+", "", line))
        except InvalidRequirement:
            continue
        specs = list(req.specifier)
        if len(specs) == 1 and specs[0].operator in ("==", "==="):
            pins[canonicalize_name(req.name)] = specs[0].version
    return pins


def _applies(marker: str, env: dict[str, str]) -> bool:
    if not marker:
        return True
    try:
        return Marker(marker).evaluate(env)
    except Exception:  # noqa: BLE001 — непонятный маркер считаем применимым: лучше ложная тревога, чем маска
        return True


def check_target(merged: list["Merged"], pins: dict[str, str], lock_name: str, report: Report,
                 env: dict[str, str]) -> None:
    """Конфликты на ЦЕЛЕВОЙ платформе: маркеры вычисляются для env, группы одного пакета с разными маркерами
    проверяются совместно, прямой URL против пина lock — конфликт. Ничего не скрываем подбором версий."""
    by_name: dict[str, list[Merged]] = {}
    for g in merged:
        if _applies(g.marker, env):
            by_name.setdefault(g.name, []).append(g)
    for name, groups in sorted(by_name.items()):
        specs = [(sp, src) for g in groups for sp, src in g.specs]
        urls = sorted({u for g in groups for u in g.urls})
        if len(groups) > 1 and not satisfiable([sp for sp, _ in specs]):
            detail = "; ".join(f"{sp} ← {src}" for sp, src in specs)
            report.conflicts.append(f"{name}: несовместимо на целевой платформе (с учётом маркеров): {detail}")
        if name not in pins:
            continue
        if urls:
            report.conflicts.append(f"{name}: в {lock_name} {pins[name]}, а модуль требует прямую ссылку {urls}")
            continue
        try:
            pin = Version(pins[name])
        except InvalidVersion:
            continue
        bad = [(sp, src) for sp, src in specs if not SpecifierSet(sp).contains(pin, prereleases=True)]
        if bad:
            detail = "; ".join(f"{sp} ← {src}" for sp, src in bad)
            report.conflicts.append(f"{name}: в {lock_name} {pins[name]}, не совместимо: {detail}")
            for g in groups:
                g.note = "CONFLICT"


def merge(lines: list[ReqLine], profile: str, report: Report, keep_dev: bool = False) -> list[Merged]:
    names = {ln.name for ln in lines}
    rename: dict[str, str] = {}
    for family in (OPENCV, ONNXRUNTIME):
        target = _family_target(names, family, profile)
        if target:
            for n in sorted(names & family):
                if n != target:
                    rename[n] = target
                    report.replaced.append(f"{n} → {target} (взаимоисключающие дистрибутивы, профиль {profile})")
    groups: dict[tuple[str, str], Merged] = {}
    for ln in lines:
        name = rename.get(ln.name, ln.name)
        if name in BANNED_PACKAGES:
            report.banned.append(f"{name} ({BANNED_PACKAGES[name]}) — {ln.source}:{ln.lineno}")
            continue
        if not keep_dev and _is_dev(name):
            report.dropped.append(f"{name} — dev-инструмент ({ln.source})")
            continue
        if name in KNOWN_TROUBLE:
            msg = f"{name}: {KNOWN_TROUBLE[name]} ({ln.source})"
            if msg not in report.warnings:
                report.warnings.append(msg)
        g = groups.setdefault((name, ln.marker), Merged(name=name, marker=ln.marker))
        g.extras |= set(ln.extras)
        g.sources.append(ln.source)
        if ln.url:
            g.urls.add(ln.url)
        if ln.specifier:
            g.specs.append((ln.specifier, f"{ln.source}:{ln.lineno}"))
    for g in groups.values():
        if len(g.urls) > 1:
            report.conflicts.append(f"{g.name}: разные прямые ссылки {sorted(g.urls)}")
        if g.urls and g.specs:
            report.conflicts.append(f"{g.name}: прямая ссылка и версия одновременно {g.specs}")
        if not satisfiable([s for s, _ in g.specs]):
            detail = "; ".join(f"{s} ← {src}" for s, src in g.specs)
            report.conflicts.append(f"{g.name}{' ; ' + g.marker if g.marker else ''}: несовместимо: {detail}")
            g.note = "CONFLICT"
    return [groups[k] for k in sorted(groups)]


def render_in(merged: list[Merged], report: Report, profile: str) -> str:
    head = [
        "# AUTO-GENERATED: deploy/merge_requirements.py — не редактировать руками.",
        f"# Профиль: {profile}; источников: {len(report.sources)}; конфликтов: {len(report.conflicts)}.",
        "# Точные версии и хэши — ТОЛЬКО в deploy/locks/<target>.lock.txt (uv pip compile).",
    ]
    head += [f"# источник: {s}" for s in report.sources]
    head += [f"# замена: {r}" for r in report.replaced]
    head += [f"# исключено: {d}" for d in report.dropped]
    head += [f"# КОНФЛИКТ: {c}" for c in report.conflicts]
    body = list(report.options) + [m.line() for m in merged]
    return "\n".join(head + [""] + body) + "\n"


def print_report(report: Report, stream=sys.stderr) -> None:
    def block(title: str, items: list[str]) -> None:
        if items:
            print(f"[merge] {title} ({len(items)}):", file=stream)
            for it in items:
                print(f"    - {it}", file=stream)
    print(f"[merge] источников: {len(report.sources)}", file=stream)
    block("ЗАПРЕЩЁННЫЕ ЛИЦЕНЗИИ", report.banned)
    block("КОНФЛИКТЫ (нужно решение владельцев модулей / A01)", report.conflicts)
    block("замены дистрибутивов", report.replaced)
    block("исключено", report.dropped)
    block("предупреждения", report.warnings)


def lock(in_path: Path, target: str, out_dir: Path, constraints: list[str], uv: str | None = None) -> Path:
    """constraints: lock L1 (-c) — его пины resolver менять не может."""
    """Настоящий resolver: uv pip compile под платформу цели. Без uv — ошибка, а не «угадывание»."""
    cfg = LOCK_TARGETS[target]
    uv = uv or shutil.which("uv")
    if not uv:
        raise RuntimeError("uv не найден: установите https://docs.astral.sh/uv/ (setup.ps1 ставит его сам)")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{target}.lock.txt"
    cmd = [uv, "pip", "compile", str(in_path), "-o", str(out), "--python-version", "3.11",
           "--python-platform", cfg["python_platform"], "--torch-backend", cfg["torch_backend"],
           "--generate-hashes", "--emit-index-annotation", "--quiet"]
    for c in constraints:
        cmd += ["-c", c]
    print("[merge] resolver:", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)
    return out


def resolve_env_lock(root: Path, env_lock: Path | str | None) -> Path | None:
    if env_lock in ("none", ""):
        return None
    if env_lock is None:
        cand = root / DEFAULT_ENV_LOCK
        return cand if cand.is_file() else None
    p = Path(env_lock)
    return p if p.is_absolute() else root / p


def parse_pins(text: str) -> dict[str, str]:
    """Пины `name==version` из вывода uv pip compile / lock-файла."""
    pins = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==([^\s;\\]+)", line.strip())
        if m:
            pins[canonicalize_name(m.group(1))] = m.group(2)
    return pins


def resolve_check(in_path: Path, lock_path: Path, target: str, work: Path, no_torch_index: bool = False,
                  uv: str | None = None) -> dict:
    """Настоящий resolver: требования модулей + ВЕСЬ lock L1 (-c) на целевой платформе.
    ok — разрешилось и каждый пакет lock остался на своей версии (дрейфа нет).
    no_torch_index: индекс PyTorch недоступен (облако) → +cuXXX снимается с torch/torchvision, колёса с PyPI;
    это проверка совместимости диапазонов, НЕ проверка GPU-установки."""
    cfg = LOCK_TARGETS[target]
    uv = uv or shutil.which("uv")
    if not uv:
        return {"ok": False, "error": "uv не найден"}
    work.mkdir(parents=True, exist_ok=True)
    lock_pins = load_env_lock(lock_path)
    constraints = lock_path
    if no_torch_index:
        constraints = work / "constraints_pypi.txt"
        constraints.write_text("".join(f"{n}=={v.split('+')[0]}\n" for n, v in sorted(lock_pins.items())),
                               encoding="utf-8")
    out = work / f"resolve_{target}.txt"
    cmd = [uv, "pip", "compile", str(in_path), "-c", str(constraints), "-o", str(out), "--python-version", "3.11",
           "--python-platform", cfg["python_platform"], "--no-header", "--quiet"]
    if not no_torch_index:
        cmd += ["--torch-backend", cfg["torch_backend"]]
    r = subprocess.run(cmd, capture_output=True, text=True)
    rep: dict = {"target": target, "no_torch_index": no_torch_index, "cmd": " ".join(cmd)}
    if r.returncode != 0:
        rep.update(ok=False, error=(r.stderr or r.stdout).strip()[-3000:])
        return rep
    got = parse_pins(out.read_text(encoding="utf-8"))
    drift = {n: {"lock": v, "resolved": got.get(n)} for n, v in lock_pins.items()
             if n in got and got[n].split("+")[0] != v.split("+")[0]}
    rep.update(ok=not drift, drift=drift, extra=sorted(set(got) - set(lock_pins)),
               not_needed=sorted(set(lock_pins) - set(got)), resolved=len(got))
    return rep


def run(root: Path, profile: str, out: Path | None, check_only: bool = False, keep_dev: bool = False,
        allow_conflicts: bool = False, env_lock: Path | str | None = None,
        target: str = "windows") -> tuple[int, Report, str]:
    report = Report()
    files = find_requirement_files(root)
    lines: list[ReqLine] = []
    for f in files:
        report.sources.append(f.resolve().relative_to(root.resolve()).as_posix())
        lines += parse_file(f, root, report)
    merged = merge(lines, profile, report, keep_dev=keep_dev)
    lock_path = resolve_env_lock(root, env_lock)
    if lock_path is None:
        check_target(merged, {}, "", report, TARGET_ENVS[target])  # маркеры — и без lock
    if lock_path is not None:
        if lock_path.is_file():
            check_target(merged, load_env_lock(lock_path), lock_path.name, report, TARGET_ENVS[target])
            report.constraints.append(str(lock_path))
            report.sources.append(f"{lock_path.name} (constraints: пины не меняются)")
        else:
            report.warnings.append(f"env-lock не найден: {lock_path}")
    text = render_in(merged, report, profile)
    code = 2 if report.banned else (1 if report.conflicts else 0)
    if out is not None and not check_only and (code == 0 or (code == 1 and allow_conflicts)):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return code, report, text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    ap.add_argument("--profile", choices=("gpu", "cpu"), default=None,
                    help="по умолчанию — из --lock цели, иначе gpu")
    ap.add_argument("--out", type=Path, default=None, help="по умолчанию <root>/deploy/requirements.in")
    ap.add_argument("--check", action="store_true", help="только отчёт, ничего не писать")
    ap.add_argument("--keep-dev", action="store_true", help="не выкидывать pytest/ruff/...")
    ap.add_argument("--allow-conflicts", action="store_true",
                    help="всё равно записать .in (конфликт увидит resolver) — для диагностики")
    ap.add_argument("--lock", choices=sorted(LOCK_TARGETS), action="append", default=[],
                    help="построить lock через uv pip compile (можно несколько раз)")
    ap.add_argument("--locks-dir", type=Path, default=None)
    ap.add_argument("--resolve-check", choices=sorted(LOCK_TARGETS), default=None,
                    help="resolver: требования модулей + весь lock L1 на платформе цели, без дрейфа пинов")
    ap.add_argument("--no-torch-index", action="store_true",
                    help="resolve-check без индекса PyTorch (облако): torch/torchvision с PyPI, без +cuXXX")
    ap.add_argument("--target-env", choices=sorted(TARGET_ENVS), default="windows",
                    help="для какой платформы вычислять маркеры (прод L1 — windows)")
    ap.add_argument("--env-lock", default=None,
                    help=f"lock окружения (по умолчанию {DEFAULT_ENV_LOCK.as_posix()}, если есть); 'none' — без него")
    args = ap.parse_args(argv)

    root = args.root.resolve()
    profile = args.profile or (LOCK_TARGETS[args.lock[0]]["profile"] if args.lock else "gpu")
    out = args.out or root / "deploy" / "requirements.in"
    code, report, _ = run(root, profile, out, args.check, args.keep_dev, args.allow_conflicts, args.env_lock,
                          args.target_env)
    print_report(report)
    if code:
        print(f"[merge] код {code}: {'запрещённые пакеты' if code == 2 else 'конфликты'} — lock не строим",
              file=sys.stderr)
        return code
    if not args.check:
        print(f"[merge] записано: {out}", file=sys.stderr)
    if args.resolve_check:
        lock_path = resolve_env_lock(root, args.env_lock)
        if lock_path is None or not lock_path.is_file():
            print("[merge] resolve-check: нет lock окружения", file=sys.stderr)
            return 2
        import tempfile
        work = Path(tempfile.mkdtemp(prefix="sentinel_resolve_"))
        src = out if out.is_file() and not args.check else work / "requirements.in"
        if src != out:
            run(root, profile, src, allow_conflicts=False, env_lock=args.env_lock, target=args.target_env)
        res = resolve_check(src, lock_path, args.resolve_check, work, args.no_torch_index)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0 if res.get("ok") else 3
    constraints = list(report.constraints)
    agreed = root / "deploy" / "constraints.txt"
    if agreed.is_file():
        constraints.append(str(agreed))
    for target in args.lock:
        if LOCK_TARGETS[target]["profile"] != profile:
            print(f"[merge] пропуск {target}: профиль {LOCK_TARGETS[target]['profile']} != {profile}",
                  file=sys.stderr)
            continue
        try:
            path = lock(out, target, args.locks_dir or root / "deploy" / "locks", constraints)
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"[merge] resolver не смог построить {target}: {exc}", file=sys.stderr)
            return 3
        print(f"[merge] lock: {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
