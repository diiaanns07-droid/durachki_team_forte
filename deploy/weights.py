#!/usr/bin/env python3
"""Офлайн-веса Kassa Sentinel: подготовка ОТДЕЛЬНЫМ шагом, проверка без сети, лицензии.

  python deploy/weights.py plan                 # что и сколько будет скачано (нужна сеть к HF)
  python deploy/weights.py prepare [--only a,b] # скачать ТОЧНЫЕ ревизии → weights/, записать
                                                # deploy/weights.lock.json (ревизии, sha256, лицензии)
  python deploy/weights.py adopt [--hash]       # БЕЗ сети: уже скачанный кэш HF (weights/hf/hub) → lock:
                                                # commit из refs, полнота шардов, .incomplete, лицензия из карточки
  python deploy/weights.py verify [--fast]      # без сети: каждый файл из lock на месте, размер+sha256
  python deploy/weights.py licenses             # deploy/LICENSES.md из lock (или models.json)

Источник списка — все **/models.json модулей: [{"name","hf_id_or_url","license","size_mb","purpose"}].
Необязательные поля: "revision" (commit/тег), "files" (glob-шаблоны внутри репо), "sha256"
(для URL), "optional": true (не требуется для verify).

Рантайм НИЧЕГО не скачивает: HF_HUB_OFFLINE=1, а verify работает только с файловой системой.
Лицензия решается по КОНКРЕТНОМУ repo@commit: заявленная (models.json) и из метаданных хаба
должны быть в allowlist (Apache-2.0/MIT/BSD) и совпадать; шаблоны имён — лишь доп. красный флаг.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol

try:
    from deploy import license_policy as policy
except ImportError:  # запуск как `python deploy/weights.py`
    import license_policy as policy  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = Path(os.environ.get("SENTINEL_WEIGHTS") or REPO_ROOT / "weights")
LOCK_PATH = REPO_ROOT / "deploy" / "weights.lock.json"
LICENSES_PATH = REPO_ROOT / "deploy" / "LICENSES.md"
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "weights", "data", "cache", "__pycache__", "dist"}
HF_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
IGNORE_ALWAYS = ("*.msgpack", "*.h5", "*.ot", "tf_model*", "flax_model*", "rust_model*", "*.tflite",
                 "coreml/*", "openvino/*", "*.mlmodel", ".gitattributes")


# ----------------------------------------------------------------------------- модели
@dataclass
class ModelSpec:
    name: str
    source: str                 # исходная строка hf_id_or_url
    kind: str                   # hf | hf_file | url | bundled
    license: str
    size_mb: float
    purpose: str
    module: str                 # папка, где лежит models.json
    repo_id: str = ""
    filename: str = ""          # для hf_file / url
    revision: str = ""
    files: list[str] = field(default_factory=list)
    sha256: str = ""
    optional: bool = False


def find_models_json(root: Path) -> list[Path]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        if "models.json" in filenames:
            out.append(Path(dirpath) / "models.json")
    return sorted(out)


def classify_source(src: str) -> tuple[str, str, str, str]:
    """→ (kind, repo_id, filename, revision)."""
    s = (src or "").strip()
    if not s or s.lower() in {"none", "builtin", "bundled", "pip"} or s.startswith(("pip:", "bundled:")):
        return "bundled", "", "", ""
    if s.startswith("local:"):
        path = s[6:].strip().split()
        return "local", "", path[0] if path else "", ""
    if s.startswith("hf://"):
        s = s[5:]
    if s.startswith(("http://", "https://")):
        u = urllib.parse.urlparse(s)
        if u.netloc in {"pypi.org", "www.pypi.org"}:
            return "bundled", "", "", ""  # модель внутри pip-колеса (напр. mediapipe) — качать нечего
        if u.netloc in {"huggingface.co", "www.huggingface.co"}:
            parts = [p for p in u.path.split("/") if p]
            if len(parts) >= 2 and parts[0] not in {"datasets", "spaces"}:
                repo = f"{parts[0]}/{parts[1]}"
                if len(parts) >= 5 and parts[2] in {"resolve", "blob"}:
                    return "hf_file", repo, "/".join(parts[4:]), parts[3]
                if len(parts) >= 4 and parts[2] == "tree":
                    return "hf", repo, "", parts[3]
                return "hf", repo, "", ""
        return "url", "", Path(u.path).name or "download.bin", ""
    if HF_ID_RE.match(s):
        return "hf", s, "", ""
    return "unknown", "", "", ""  # свободный текст вместо id/URL — проверить нельзя


def load_specs(root: Path, errors: Optional[list[str]] = None) -> list[ModelSpec]:
    errors = errors if errors is not None else []
    specs: list[ModelSpec] = []
    for path in find_models_json(root):
        module = path.parent.relative_to(root).as_posix()
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{module}/models.json: не читается: {exc}")
            continue
        if isinstance(data, dict):
            data = data.get("models", [data] if "name" in data else [])
        if not isinstance(data, list):
            errors.append(f"{module}/models.json: ожидается список")
            continue
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                errors.append(f"{module}/models.json[{i}]: не объект")
                continue
            item = {str(k).lower(): v for k, v in item.items()}
            missing = [k for k in ("name", "hf_id_or_url", "license") if not item.get(k)]
            if missing:
                errors.append(f"{module}/models.json[{i}]: нет полей {missing}")
            src = str(item.get("hf_id_or_url") or "")
            kind, repo, fname, rev = classify_source(src)
            try:
                size = float(item.get("size_mb") or 0)
            except (TypeError, ValueError):
                size = 0.0
            files = item.get("files") or item.get("allow_patterns") or []
            specs.append(ModelSpec(
                name=str(item.get("name") or f"{module}#{i}"), source=src, kind=kind,
                license=str(item.get("license") or ""), size_mb=size, purpose=str(item.get("purpose") or ""),
                module=module, repo_id=repo, filename=str(item.get("filename") or fname),
                revision=str(item.get("revision") or rev or ""),
                files=[files] if isinstance(files, str) else [str(f) for f in files],
                sha256=str(item.get("sha256") or "").lower(), optional=bool(item.get("optional", False)),
            ))
    return specs


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", text).strip("_") or "model"


# ----------------------------------------------------------------------------- хэши/кэш
def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    """Так хаб идентифицирует не-LFS файлы (blob_id)."""
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def repo_cache_dir(hub_cache: Path, repo_id: str) -> Path:
    return hub_cache / ("models--" + repo_id.replace("/", "--"))


def snapshot_dir(hub_cache: Path, repo_id: str, commit: str) -> Path:
    return repo_cache_dir(hub_cache, repo_id) / "snapshots" / commit


def hub_cache_dir(weights: Path) -> Path:
    env = os.environ.get("HF_HUB_CACHE")
    return Path(env) if env else weights / "hf" / "hub"


def select_files(all_files: Iterable[str], patterns: list[str]) -> list[str]:
    files = sorted(set(all_files))
    if patterns:
        return [f for f in files if any(fnmatch.fnmatch(f, p) for p in patterns)]
    keep = [f for f in files if not any(fnmatch.fnmatch(f, p) for p in IGNORE_ALWAYS)]
    if any(f.endswith(".safetensors") for f in keep):  # дубли форматов не тащим
        keep = [f for f in keep if not (fnmatch.fnmatch(f, "pytorch_model*.bin") or f.startswith("onnx/")
                                        or f.endswith(".gguf"))]
    ggufs = [f for f in keep if f.endswith(".gguf")]
    if len(ggufs) > 1:
        raise ValueError(f"в репозитории {len(ggufs)} GGUF-файлов — укажите 'files' в models.json")
    return keep


# ----------------------------------------------------------------------------- хаб
class Hub(Protocol):
    def info(self, repo_id: str, revision: str) -> dict[str, Any]: ...
    def download(self, repo_id: str, commit: str, files: list[str], cache_dir: Path) -> Path: ...


class HFHub:
    """Обёртка над huggingface_hub (импорт лениво; используется ТОЛЬКО в plan/prepare)."""

    def __init__(self, token: Optional[str] = None) -> None:
        os.environ["HF_HUB_OFFLINE"] = "0"
        from huggingface_hub import HfApi  # noqa: WPS433
        self.api = HfApi(token=token)

    def info(self, repo_id: str, revision: str) -> dict[str, Any]:
        mi = self.api.model_info(repo_id, revision=revision or None, files_metadata=True)
        card = getattr(mi, "card_data", None) or getattr(mi, "cardData", None) or {}
        lic = (card.get("license") if hasattr(card, "get") else getattr(card, "license", "")) or ""
        if not lic:
            lic = next((t.split(":", 1)[1] for t in (mi.tags or []) if t.startswith("license:")), "")
        files = []
        for s in mi.siblings or []:
            lfs = getattr(s, "lfs", None)
            sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
            files.append({"path": s.rfilename, "size": getattr(s, "size", None), "sha256": sha,
                          "blob_id": getattr(s, "blob_id", None)})
        return {"sha": mi.sha, "license": lic if isinstance(lic, str) else ",".join(lic), "files": files}

    def download(self, repo_id: str, commit: str, files: list[str], cache_dir: Path) -> Path:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(repo_id, revision=commit, allow_patterns=files or None,
                                      cache_dir=str(cache_dir)))


def http_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:  # noqa: S310
        shutil.copyfileobj(r, f, 8 << 20)
    tmp.replace(dest)


# ----------------------------------------------------------------------------- prepare
def license_verdict(declared: str, hub: Optional[str], banned: list[str]) -> tuple[bool, str]:
    if banned:
        return False, "запрещённое семейство: " + "; ".join(banned)
    if not policy.is_allowed(declared):
        return False, f"заявлена '{declared or '—'}' — не Apache/MIT/BSD"
    if hub is None:
        return False, "лицензия не подтверждена метаданными (URL) — нужна ручная проверка"
    if not hub:
        return False, "у ревизии на хабе не указана лицензия — нужна ручная проверка"
    if not policy.is_allowed(hub):
        return False, f"хаб для этой ревизии: '{hub}' — не Apache/MIT/BSD"
    if not policy.same_license(declared, hub):
        return False, f"заявлено '{declared}', на хабе '{hub}'"
    return True, "ok"


def load_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"version": 1, "models": []}


def _locked(lock: dict[str, Any], spec: ModelSpec) -> Optional[dict[str, Any]]:
    return next((m for m in lock.get("models", []) if m.get("name") == spec.name
                 and m.get("source") == spec.source), None)


def plan_entry(spec: ModelSpec, hub: Optional[Hub], lock: dict[str, Any], update: bool) -> dict[str, Any]:
    entry: dict[str, Any] = {"name": spec.name, "module": spec.module, "source": spec.source,
                             "kind": spec.kind, "purpose": spec.purpose, "license_declared": spec.license,
                             "declared_size_mb": spec.size_mb, "optional": spec.optional}
    banned = policy.banned_reasons(f"{spec.source} {spec.name}")
    prev = None if update else _locked(lock, spec)
    if spec.kind in ("hf", "hf_file"):
        if hub is None:
            raise RuntimeError("нужен доступ к хабу")
        rev = (prev or {}).get("revision") or spec.revision or "main"
        info = hub.info(spec.repo_id, rev)
        wanted = [spec.filename] if spec.kind == "hf_file" else spec.files
        names = select_files([f["path"] for f in info["files"]], wanted)
        meta = {f["path"]: f for f in info["files"]}
        entry.update(repo_id=spec.repo_id, revision=info["sha"], requested_revision=spec.revision or "main",
                     license_hub=info["license"],
                     files=[{"path": n, "size": meta[n].get("size"), "sha256": meta[n].get("sha256"),
                             "blob_id": meta[n].get("blob_id")} for n in names])
        ok, why = license_verdict(spec.license, info["license"], banned)
    elif spec.kind == "url":
        entry.update(url=spec.source, files=[{"path": f"url/{slug(spec.name)}/{spec.filename}",
                                              "size": None, "sha256": spec.sha256 or None}],
                     license_hub=None)
        ok, why = license_verdict(spec.license, None, banned)
    else:
        entry.update(files=[], license_hub=None)
        ok, why = license_verdict(spec.license, spec.license, banned)  # входит в pip-пакет
    entry["license_ok"], entry["license_note"] = ok, why
    entry["planned_bytes"] = sum(int(f.get("size") or 0) for f in entry["files"])
    return entry


def prepare(specs: list[ModelSpec], weights: Path, hub: Optional[Hub], lock_path: Path = LOCK_PATH,
            update: bool = False, dry_run: bool = False, allow_license_flags: bool = False,
            log=print) -> tuple[int, dict[str, Any]]:
    lock = load_lock(lock_path)
    hub_cache = hub_cache_dir(weights)
    entries = [plan_entry(s, hub, lock, update) for s in specs]
    total = sum(e["planned_bytes"] for e in entries)
    for e in entries:
        flag = "OK " if e["license_ok"] else "RED"
        log(f"[weights] {flag} {e['name']:<28} {e['planned_bytes'] / 2**20:9.1f} MB  {e['source']}"
            f"  license={e['license_declared']}/{e.get('license_hub')}  {e['license_note']}")
    log(f"[weights] всего к загрузке: {total / 2**30:.2f} GB → {weights}")
    bad = [e for e in entries if not e["license_ok"]]
    if bad and not allow_license_flags:
        log(f"[weights] СТОП: {len(bad)} модель(ей) с красным флагом лицензии (см. выше)")
        return 2, lock
    if dry_run:
        return 0, lock
    weights.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(weights).free
    if total * 1.1 > free:
        log(f"[weights] СТОП: на диске {free / 2**30:.1f} GB, нужно ~{total * 1.1 / 2**30:.1f} GB")
        return 3, lock
    code = 0
    for e, spec in zip(entries, specs):
        try:
            if e["kind"] in ("hf", "hf_file"):
                snap = hub.download(e["repo_id"], e["revision"], [f["path"] for f in e["files"]], hub_cache)
                ref = repo_cache_dir(hub_cache, e["repo_id"]) / "refs" / e["requested_revision"]
                if not re.fullmatch(r"[0-9a-f]{40}", e["requested_revision"]):
                    ref.parent.mkdir(parents=True, exist_ok=True)
                    ref.write_text(e["revision"], encoding="utf-8")  # offline from_pretrained(repo)
                for f in e["files"]:
                    p = snap / f["path"]
                    got = sha256_file(p)
                    if f.get("sha256") and got != f["sha256"]:
                        raise ValueError(f"{f['path']}: sha256 {got} != хаб {f['sha256']}")
                    if not f.get("sha256") and f.get("blob_id") and git_blob_sha1(p) != f["blob_id"]:
                        raise ValueError(f"{f['path']}: git blob не совпал с хабом")
                    f["sha256"], f["size"] = got, p.stat().st_size
                e["path"] = snap.relative_to(weights).as_posix() if snap.is_relative_to(weights) else str(snap)
            elif e["kind"] == "url":
                f = e["files"][0]
                dest = weights / f["path"]
                if not dest.is_file():
                    http_download(e["url"], dest)
                got = sha256_file(dest)
                if f.get("sha256") and got != f["sha256"]:
                    dest.unlink()
                    raise ValueError(f"sha256 {got} != заявленного {f['sha256']}")
                if not f.get("sha256"):
                    e["license_note"] += "; sha256 записан при первой загрузке (TOFU) — внесите в models.json"
                f["sha256"], f["size"] = got, dest.stat().st_size
                e["path"] = f["path"]
            e["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 — отчёт по каждой модели
            e["status"] = f"error: {exc}"
            code = 1 if not spec.optional else code
            log(f"[weights] ОШИБКА {e['name']}: {exc}")
    keep = {(e["name"], e["source"]) for e in entries}
    others = [m for m in lock.get("models", []) if (m.get("name"), m.get("source")) not in keep]
    lock = {"version": 1, "generated_at": _now(), "tool": "deploy/weights.py prepare",
            "models": sorted(others + [_lock_view(e) for e in entries], key=lambda m: (m["module"], m["name"]))}
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_manifest(weights, lock, entries)
    log(f"[weights] lock: {lock_path}")
    return code, lock


def _lock_view(e: dict[str, Any]) -> dict[str, Any]:
    keys = ("name", "module", "source", "kind", "repo_id", "revision", "requested_revision", "url",
            "license_declared", "license_hub", "license_ok", "license_note", "purpose", "optional", "path")
    out = {k: e[k] for k in keys if k in e}
    out["files"] = [{"path": f["path"], "size": f.get("size"), "sha256": f.get("sha256")} for f in e["files"]]
    return out


def write_manifest(weights: Path, lock: dict[str, Any], entries: list[dict[str, Any]]) -> Path:
    status = {(e["name"], e["source"]): e.get("status", "planned") for e in entries}
    models = []
    for m in lock["models"]:
        mm = dict(m)
        mm["status"] = status.get((m["name"], m["source"]), "from-lock")
        models.append(mm)
    path = weights / "MANIFEST.json"
    path.write_text(json.dumps({"version": 1, "generated_at": _now(), "hf_hub_cache": "hf/hub",
                                "models": models}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


# ----------------------------------------------------------------------------- adopt (без сети)
WEIGHT_EXT = (".safetensors", ".bin", ".pt", ".pth", ".onnx", ".gguf", ".ckpt")


def card_license(readme: Path) -> str:
    """license из YAML front matter README.md снапшота (карточка модели)."""
    try:
        text = readme.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    m = re.match(r"^---\s*\n(.*?)\n---", text, re.S)
    if not m:
        return ""
    lm = re.search(r"^license:\s*(.+?)\s*$", m.group(1), re.M)
    return lm.group(1).strip().strip("'\"") if lm else ""


def inspect_snapshot(hub_cache: Path, repo_id: str, ref: str = "main", hash_files: bool = False) -> dict[str, Any]:
    """Полнота конкретного снапшота в кэше huggingface_hub — только файловая система."""
    rdir = repo_cache_dir(hub_cache, repo_id)
    out: dict[str, Any] = {"repo_id": repo_id, "problems": [], "warnings": [], "files": [], "commit": ""}
    if not rdir.is_dir():
        out["problems"].append(f"нет в кэше ({rdir})")
        return out
    refp = rdir / "refs" / ref
    snaps = sorted(d.name for d in (rdir / "snapshots").glob("*") if d.is_dir()) if (rdir / "snapshots").is_dir() else []
    if refp.is_file():
        out["commit"] = refp.read_text(encoding="utf-8").strip()
    elif len(snaps) == 1:
        out["commit"] = snaps[0]
        out["warnings"].append(f"нет refs/{ref}; единственный снапшот {snaps[0][:12]}")
    else:
        out["problems"].append(f"нет refs/{ref}, снапшотов: {len(snaps)}")
        return out
    snap = rdir / "snapshots" / out["commit"]
    if not snap.is_dir():
        out["problems"].append(f"refs/{ref} → {out['commit'][:12]}, но снапшота нет")
        return out
    incomplete = sorted(b.name for b in (rdir / "blobs").glob("*.incomplete")) if (rdir / "blobs").is_dir() else []
    if incomplete:
        out["warnings"].append(f"{len(incomplete)} недокачанных blob (.incomplete)")
    out["incomplete_blobs"] = len(incomplete)
    names = []
    for f in sorted(snap.rglob("*")):
        if f.is_dir():
            continue
        rel = f.relative_to(snap).as_posix()
        names.append(rel)
        target = f.resolve() if f.is_symlink() else f
        if not target.is_file():
            out["problems"].append(f"{rel}: битая ссылка на blob")
            continue
        blob = target.name if f.is_symlink() else ""
        entry: dict[str, Any] = {"path": rel, "size": target.stat().st_size, "sha256": None, "blob_id": None}
        if re.fullmatch(r"[0-9a-f]{64}", blob):
            entry["sha256"] = blob                       # LFS: имя blob = sha256 содержимого
        elif re.fullmatch(r"[0-9a-f]{40}", blob):
            entry["blob_id"] = blob                      # не-LFS: git blob sha1
        if hash_files:
            got = sha256_file(target)
            if entry["sha256"] and got != entry["sha256"]:
                out["problems"].append(f"{rel}: sha256 содержимого не равен имени blob")
            entry["sha256"] = got
        if entry["size"] == 0 and rel.endswith(WEIGHT_EXT):
            out["problems"].append(f"{rel}: пустой файл весов")
        out["files"].append(entry)
    for idx in [n for n in names if n.endswith(".index.json")]:
        try:
            wmap = json.loads((snap / idx).read_text(encoding="utf-8")).get("weight_map", {})
        except (OSError, ValueError) as exc:
            out["problems"].append(f"{idx}: не читается ({exc})")
            continue
        missing = sorted(set(wmap.values()) - set(names))
        if missing:
            out["problems"].append(f"{idx}: нет шардов {missing}")
    if not any(n.endswith(WEIGHT_EXT) for n in names):
        out["problems"].append("в снапшоте нет файлов весов")
    out["license_card"] = card_license(snap / "README.md")
    return out


def adopt(models: list[dict[str, Any]], weights: Path, lock_path: Path = LOCK_PATH, hash_files: bool = False,
          exclude: Iterable[str] = (), declared: Optional[dict[str, str]] = None, log=print) -> tuple[int, dict]:
    """Уже скачанные веса → lock (точный commit, файлы, лицензия карточки). Сеть не нужна."""
    declared = declared or {}
    lock = load_lock(lock_path)
    hub = hub_cache_dir(weights)
    entries, code = [], 0
    for m in models:
        repo = m["repo"]
        if any(fnmatch.fnmatch(repo, pat) for pat in exclude):
            log(f"[adopt] SKIP {repo}: исключён шаблоном {list(exclude)}")
            continue
        info = inspect_snapshot(hub, repo, m.get("ref", "main"), hash_files)
        decl = declared.get(repo) or info.get("license_card", "")
        ok, why = license_verdict(decl, info.get("license_card", ""), policy.banned_reasons(repo))
        e = {"name": repo, "module": m.get("module", ""), "source": repo, "kind": "hf", "repo_id": repo,
             "revision": info["commit"], "requested_revision": m.get("ref", "main"),
             "license_declared": decl, "license_hub": info.get("license_card", ""),
             "license_ok": ok, "license_note": why + " (лицензия из README.md локального снапшота)",
             "purpose": m.get("purpose", ""), "optional": bool(m.get("optional")), "adopted": True,
             "files": [{"path": f["path"], "size": f["size"], "sha256": f["sha256"], "blob_id": f["blob_id"]}
                       for f in info["files"]],
             "status": "ok" if not info["problems"] else "error: " + "; ".join(info["problems"])}
        if info["problems"] and not e["optional"]:
            code = 1
        flag = "OK  " if not info["problems"] and ok else "FAIL"
        log(f"[adopt] {flag} {repo}@{info['commit'][:12]} файлов {len(info['files'])} "
            f"{sum(f['size'] for f in info['files']) / 2**30:.2f} GB license={e['license_hub'] or '—'} "
            + "; ".join(info["problems"] + info["warnings"]))
        entries.append(e)
    keep = {(e["name"], e["source"]) for e in entries}
    others = [x for x in lock.get("models", []) if (x.get("name"), x.get("source")) not in keep]
    lock = {"version": 1, "generated_at": _now(), "tool": "deploy/weights.py adopt",
            "models": sorted(others + entries, key=lambda x: (x.get("module", ""), x["name"]))}
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        write_manifest(weights, lock, entries)
    except OSError:
        pass
    return code, lock


# ----------------------------------------------------------------------------- verify (без сети)
def file_location(weights: Path, model: dict[str, Any], rel: str) -> Path:
    if model.get("kind") in ("hf", "hf_file"):
        return snapshot_dir(hub_cache_dir(weights), model["repo_id"], model["revision"]) / rel
    return weights / rel


def verify(weights: Path, lock: dict[str, Any], fast: bool = False, only: Optional[set[str]] = None,
           log=print) -> tuple[int, list[dict[str, Any]]]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    results, code = [], 0
    models = [m for m in lock.get("models", []) if not only or m["name"] in only or m.get("repo_id") in only]
    if not models:
        log("[verify] lock пуст — нечего проверять (сначала: weights.py prepare)")
    for m in models:
        problems = []
        if not m.get("license_ok"):
            problems.append(f"лицензия: {m.get('license_note')}")
        if m.get("kind") in ("hf", "hf_file") and not re.fullmatch(r"[0-9a-f]{40}", m.get("revision") or ""):
            problems.append(f"ревизия не закреплена коммитом: {m.get('revision')!r}")
        for f in m.get("files", []):
            p = file_location(weights, m, f["path"])
            if not p.is_file():
                problems.append(f"нет файла {f['path']}")
                continue
            size = p.stat().st_size
            if f.get("size") is not None and size != f["size"]:
                problems.append(f"{f['path']}: размер {size} != {f['size']}")
            elif not fast and f.get("sha256") and sha256_file(p) != f["sha256"]:
                problems.append(f"{f['path']}: sha256 не совпал")
            elif not fast and not f.get("sha256") and f.get("blob_id") and git_blob_sha1(p) != f["blob_id"]:
                problems.append(f"{f['path']}: git blob не совпал")
        ok = not problems
        if not ok and not m.get("optional"):
            code = 1
        results.append({"name": m["name"], "ok": ok, "optional": bool(m.get("optional")), "problems": problems})
        log(f"[verify] {'OK  ' if ok else 'FAIL'} {m['name']}" + ("" if ok else ": " + "; ".join(problems)))
    try:
        (weights / "VERIFY.json").write_text(json.dumps(
            {"checked_at": _now(), "fast": fast, "ok": code == 0, "results": results},
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass  # weights смонтированы read-only — отчёт только в stdout
    return code, results


# ----------------------------------------------------------------------------- LICENSES.md
def _md(text: Any) -> str:
    return str(text if text not in (None, "") else "—").replace("|", "\\|").replace("\n", " ")


def render_licenses(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Лицензии моделей Kassa Sentinel",
        "",
        "> Сгенерировано `python deploy/weights.py licenses` — не редактировать руками.",
        "> Правило 2: коммерческое использование только Apache-2.0 / MIT / BSD. 🔴 — модель НЕ допускается",
        "> к сдаче, пока владелец модуля не заменит её или не подтвердит лицензию вручную.",
        "> Источник истины — `deploy/weights.lock.json` (точная ревизия + лицензия из метаданных хаба).",
        "",
        "| Модель | Модуль | Источник@ревизия | Заявлено | По хабу | Коммерция | Примечание |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: (x.get("module", ""), x.get("name", ""))):
        rev = (r.get("revision") or "")[:12]
        src = f"{r.get('source', '')}@{rev}" if rev else r.get("source", "")
        verdict = "✅ да" if r.get("license_ok") else "🔴 НЕТ / проверить"
        lines.append(f"| {_md(r.get('name'))} | {_md(r.get('module'))} | {_md(src)} | "
                     f"{_md(r.get('license_declared'))} | {_md(r.get('license_hub'))} | {verdict} | "
                     f"{_md(r.get('license_note'))} |")
    if not rows:
        lines.append("| — | — | — | — | — | — | models.json модулей ещё не смерджены |")
    red = sum(1 for r in rows if not r.get("license_ok"))
    lines += ["", f"Итого моделей: {len(rows)}; красных флагов: {red}.", ""]
    return "\n".join(lines)


def license_rows(specs: list[ModelSpec], lock: dict[str, Any]) -> list[dict[str, Any]]:
    """Строка на модель: из lock (точная ревизия), иначе из models.json (не подтверждено).
    Записи adopt идут по repo id — сопоставляем и по нему; модели из lock без models.json тоже попадают."""
    rows, used = [], set()
    models = lock.get("models", [])
    for s in specs:
        locked = _locked(lock, s) or next((m for m in models if s.repo_id and m.get("repo_id") == s.repo_id), None)
        if locked:
            used.add(id(locked))
            rows.append({**locked, "module": locked.get("module") or s.module,
                         "license_declared": s.license or locked.get("license_declared")})
            continue
        banned = policy.banned_reasons(f"{s.source} {s.name}")
        ok, why = license_verdict(s.license, None if s.kind != "bundled" else s.license, banned)
        if s.kind in ("hf", "hf_file") and policy.is_allowed(s.license) and not banned:
            why = "не подтверждено хабом: запустите weights.py prepare"
        rows.append({"name": s.name, "module": s.module, "source": s.source, "license_declared": s.license,
                     "license_hub": None, "license_ok": ok and s.kind == "bundled", "license_note": why})
    rows += [m for m in models if id(m) not in used]
    return rows


def _config_models() -> Optional[tuple[list[dict], list[dict], list[str]]]:
    try:
        try:
            from deploy import deploy_config as dc
        except ImportError:
            import deploy_config as dc  # type: ignore
        cfg = dc.load()
    except (ImportError, OSError):
        return None
    req, opt = dc.required_models(cfg)
    return req, opt, list((cfg.get("models") or {}).get("exclude_patterns") or [])


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


# ----------------------------------------------------------------------------- CLI
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("plan", "prepare", "adopt", "verify", "licenses", "list"))
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--lock", type=Path, default=None, help="по умолчанию <root>/deploy/weights.lock.json")
    ap.add_argument("--only", default="", help="имена через запятую")
    ap.add_argument("--update", action="store_true", help="prepare: игнорировать закреплённые в lock ревизии")
    ap.add_argument("--fast", action="store_true", help="verify: только наличие и размер, без sha256")
    ap.add_argument("--allow-license-flags", action="store_true", help="prepare: качать и с красным флагом")
    ap.add_argument("--out", type=Path, default=None, help="licenses: путь LICENSES.md")
    ap.add_argument("--hash", action="store_true", help="adopt: полный sha256 каждого файла (долго)")
    ap.add_argument("--models-from", choices=("config", "models-json"), default="config",
                    help="verify/adopt: обязательный список из deploy.yaml (по умолчанию) или из models.json")
    args = ap.parse_args(argv)

    root = args.root.resolve()
    lock_path = args.lock or root / "deploy" / "weights.lock.json"
    errors: list[str] = []
    specs = load_specs(root, errors)
    for e in errors:
        print(f"[weights] models.json: {e}", file=sys.stderr)
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    if only:
        specs = [s for s in specs if s.name in only]

    if args.cmd == "list":
        for s in specs:
            print(f"{s.module:<14} {s.name:<28} {s.kind:<8} {s.license:<14} {s.size_mb:>8.0f} MB  {s.source}")
        return 1 if errors else 0
    if args.cmd == "licenses":
        out = args.out or root / "deploy" / "LICENSES.md"
        rows = license_rows(specs, load_lock(lock_path))
        out.write_text(render_licenses(rows), encoding="utf-8")
        red = sum(1 for r in rows if not r.get("license_ok"))
        print(f"[weights] {out}: моделей {len(rows)}, красных флагов {red}")
        return 1 if red or errors else 0
    cfg_models = _config_models() if args.models_from == "config" else None
    if args.cmd == "adopt":
        if cfg_models is None:
            print("[adopt] нужен deploy/config/deploy.yaml (models.required/optional)", file=sys.stderr)
            return 2
        req, opt, exclude = cfg_models
        declared = {s.repo_id: s.license for s in specs if s.repo_id}
        todo = [dict(m) for m in req] + [dict(m, optional=True) for m in opt]
        if only:
            todo = [m for m in todo if m["repo"] in only]
        code, _ = adopt(todo, args.weights, lock_path, hash_files=args.hash, exclude=exclude, declared=declared)
        return code
    if args.cmd == "verify" and cfg_models is not None:
        req, opt, exclude = cfg_models
        lock = load_lock(lock_path)
        have = {m.get("repo_id") or m.get("name") for m in lock.get("models", [])}
        missing = [m["repo"] for m in req if m["repo"] not in have]
        for name in missing:
            print(f"[verify] FAIL {name}: обязательной модели нет в weights.lock.json (weights.py adopt|prepare)",
                  file=sys.stderr)
        wanted = {m["repo"] for m in req + opt} | (only or set())
        sub = {**lock, "models": [m for m in lock.get("models", []) if (m.get("repo_id") or m["name"]) in wanted
                                  and not any(fnmatch.fnmatch(m.get("repo_id") or m["name"], p) for p in exclude)]}
        for m in sub["models"]:
            m["optional"] = (m.get("repo_id") or m["name"]) not in {x["repo"] for x in req}
        code, _ = verify(args.weights, sub, fast=args.fast, only=only or None)
        return 1 if missing else code
    if args.cmd == "verify":
        lock = load_lock(lock_path)
        missing = [s.name for s in specs if s.kind not in ("bundled", "unknown") and not s.optional
                   and not _locked(lock, s)]
        for name in missing:
            print(f"[verify] FAIL {name}: нет в weights.lock.json — запустите weights.py prepare", file=sys.stderr)
        code, _ = verify(args.weights, lock, fast=args.fast, only=only or None)
        return 1 if missing else code
    hub = HFHub(token=os.environ.get("HF_TOKEN")) if any(s.kind in ("hf", "hf_file") for s in specs) else None
    code, _ = prepare(specs, args.weights, hub, lock_path=lock_path, update=args.update,
                      dry_run=args.cmd == "plan", allow_license_flags=args.allow_license_flags)
    return code or (1 if errors else 0)


if __name__ == "__main__":
    sys.exit(main())
