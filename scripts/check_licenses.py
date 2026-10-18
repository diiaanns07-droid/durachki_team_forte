#!/usr/bin/env python3
"""Сторож правил дисквалификации (владелец A01). Код 1 — есть нарушение.

Проверяет:
1. ``**/models.json`` — лицензия только Apache-2.0 / MIT / BSD; запрещённые модели (YOLOv5–v11/ultralytics,
   YOLO-World, VideoMAE, Qwen2.5-VL-3B, Qwen2.5-3B, CC-BY-NC) — ошибка;
2. ``**/requirements*.txt`` — запрещённые пакеты (ultralytics, mmyolo) и облачные AI-SDK в рантайме;
3. ``*.py`` — обращения к облачным LLM/vision API (api.openai.com, api.anthropic.com, generativelanguage...);
4. ``git ls-files`` — видео, веса, кадры (JPEG вне docs/pitch/frontend assets) и файлы > 10 МБ.

    python scripts/check_licenses.py [--root .] [--no-git]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ownership import forbidden_reason  # noqa: E402

ALLOWED_LICENSE_RE = re.compile(r"^(apache[- ]?2(\.0)?|mit|bsd([- ]?[23][- ]?clause)?|bsd|isc)$", re.I)
FORBIDDEN_MODEL_RE = re.compile(
    r"ultralytics|yolo-?world|videomae|qwen2\.5-vl-3b|qwen2\.5-3b|yolo-?v?(5|8|9|10|11)\b|yolo11|cc-by-nc|"
    r"non-?commercial", re.I)
FORBIDDEN_PACKAGES = {"ultralytics": "AGPL-3.0 (YOLOv5–v11)", "mmyolo": "GPL-3.0", "yolov5": "AGPL-3.0"}
CLOUD_SDKS = {"anthropic", "google-generativeai", "google-genai", "google-cloud-vision", "google-cloud-aiplatform",
              "cohere", "mistralai", "groq", "replicate", "together", "azure-ai-vision", "boto3"}
SOFT_SDKS = {"openai": "допустимо только для локального OpenAI-совместимого сервера (llama.cpp); проверьте base_url"}
CLOUD_ENDPOINT_RE = re.compile(
    r"api\.openai\.com|api\.anthropic\.com|generativelanguage\.googleapis\.com|vision\.googleapis\.com|"
    r"aiplatform\.googleapis\.com|openai\.azure\.com|api\.cohere\.ai|api\.mistral\.ai|api\.groq\.com", re.I)
def license_ok(lic: str) -> bool:
    """«Apache-2.0», «MIT / BSD-3-Clause», «Apache-2.0 (собственные веса команды)» — ок; иное — нет."""
    base = re.split(r"[(;,]", lic, maxsplit=1)[0].strip()
    parts = [p.strip() for p in re.split(r"\s*(?:/|\+|\bor\b|\band\b)\s*", base, flags=re.I) if p.strip()]
    return bool(parts) and all(ALLOWED_LICENSE_RE.match(p) for p in parts)


SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "data", "weights", "cache", "__pycache__", "dist", "build"}
MAX_TRACKED_MB = 10.0


def _iter(root: Path, pattern: str):
    for p in root.rglob(pattern):
        # .venv-* — локальные окружения инструментов (например datasets/det/.venv-ls), не код репо
        if not any(part in SKIP_DIRS or part.startswith(".venv") for part in p.relative_to(root).parts):
            yield p


def check_models(root: Path) -> list[str]:
    errs = []
    for p in _iter(root, "models.json"):
        rel = p.relative_to(root).as_posix()
        try:
            entries = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            errs.append(f"{rel}: не читается как JSON ({e})")
            continue
        if isinstance(entries, dict):
            entries = entries.get("models", [entries])
        for e in entries if isinstance(entries, list) else []:
            if not isinstance(e, dict):
                continue
            name = f"{e.get('name', '')} {e.get('hf_id_or_url', '')}"
            lic = str(e.get("license", "")).strip()
            if FORBIDDEN_MODEL_RE.search(name) or FORBIDDEN_MODEL_RE.search(lic):
                errs.append(f"{rel}: запрещённая модель/лицензия: {name.strip()} [{lic}]")
            elif not license_ok(lic):
                errs.append(f"{rel}: лицензия '{lic}' у {name.strip()} — разрешены только Apache-2.0/MIT/BSD")
    return errs


def check_requirements(root: Path) -> tuple[list[str], list[str]]:
    errs, warns = [], []
    for p in _iter(root, "requirements*.txt"):
        rel = p.relative_to(root).as_posix()
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            m = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)", line)
            if not m:
                continue
            name = re.sub(r"[-_.]+", "-", m.group(1)).lower()
            if name in FORBIDDEN_PACKAGES:
                errs.append(f"{rel}: пакет {name} запрещён ({FORBIDDEN_PACKAGES[name]})")
            elif name in CLOUD_SDKS:
                errs.append(f"{rel}: облачный AI-SDK {name} в зависимостях (правило 1: только локально)")
            elif name in SOFT_SDKS:
                warns.append(f"{rel}: {name} — {SOFT_SDKS[name]}")
    return errs, warns


def check_code(root: Path) -> list[str]:
    errs = []
    me = Path(__file__).resolve()
    for p in _iter(root, "*.py"):
        if p.resolve() == me or "tests" in p.relative_to(root).parts:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if CLOUD_ENDPOINT_RE.search(line):
                errs.append(f"{p.relative_to(root).as_posix()}:{i}: обращение к облачному AI API (правило 1)")
    return errs


def check_tracked(root: Path) -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    errs = []
    for rel in filter(None, out.decode("utf-8", "replace").split("\0")):
        why = forbidden_reason(rel)
        if why:
            errs.append(f"{rel}: {why}")
            continue
        f = root / rel
        if f.is_file() and f.stat().st_size > MAX_TRACKED_MB * 1024 * 1024:
            errs.append(f"{rel}: файл больше {MAX_TRACKED_MB:.0f} МБ в git")
    return errs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Проверка лицензий, локальности и запрещённых файлов")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--no-git", action="store_true")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    errs = check_models(root)
    req_errs, warns = check_requirements(root)
    errs += req_errs + check_code(root)
    if not args.no_git:
        errs += check_tracked(root)
    for w in warns:
        print(f"WARN  {w}")
    for e in errs:
        print(f"ERROR {e}")
    print("OK: лицензии/локальность/файлы в порядке" if not errs else f"FAIL: {len(errs)} нарушений")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
