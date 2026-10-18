"""Чтение/запись кэша состояний и разметки для ``python -m ml.rules.replay``.

Формат кэша (JSON, без pickle). Поддерживаются раскладки:

* ``<cache>/<video_id>/states.json`` (+ необязательные ``tracks.json``, ``zones.json``);
* ``<cache>/<video_id>.states.json`` / ``.states.jsonl`` (+ ``<video_id>.tracks.json``, ``.zones.json``);
* ``<cache>/**/<video_id>.json`` со словарём ``{"states": [...], "tracks": [...], "zones": [...]}``.
  Файлы ``*.json.gz`` тоже читаются.

Состояние: ``{"t": 1.2, "signals": {...}}`` или плоско ``{"t": 1.2, "safe_open": 0.9, ...}``.
Трек: ``{"track_id", "label", "role", "role_conf", "boxes": [[t, [x1, y1, x2, y2]], ...]}``.
Зона: ``{"name", "polygon": [[x, y], ...], "conf"}``. Это ``dataclasses.asdict`` контракта v1.
"""
from __future__ import annotations

import csv
import dataclasses
import gzip
import io
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

__all__ = ["save_cache", "load_cache_dir", "load_labels", "load_name_map", "hash_videos", "normalize_video_id",
           "parse_timecode", "parse_label"]

# Файлы кэша A03 (cache/<sha>/...), которые не являются состояниями сцены.
_NOT_STATES = ("dets.", "poses.", "pose_states.", "events_", "risk.", "manifest.", "summary.", "tracks.", "zones.")

_CYR2LAT = str.maketrans({
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "У": "Y", "Х": "X", "а": "a", "в": "b", "е": "e", "к": "k", "м": "m", "н": "h", "о": "o", "р": "p",
    "с": "c", "т": "t", "у": "y", "х": "x",
})
_VIDEO_EXT = re.compile(r"\.(mov|mp4|avi|mkv|m4v|webm)$", re.IGNORECASE)


def normalize_video_id(name: str) -> str:
    """``Х-3.MOV`` (кириллица) ≡ ``X-3`` ≡ ``x_3`` → ``x3``."""
    base = Path(str(name).strip()).name
    for suffix in (".gz", ".jsonl", ".json"):
        if base.lower().endswith(suffix):
            base = base[: -len(suffix)]
    for suffix in (".states", ".tracks", ".zones"):
        if base.lower().endswith(suffix):
            base = base[: -len(suffix)]
    base = _VIDEO_EXT.sub("", base)
    base = base.translate(_CYR2LAT).lower()
    return re.sub(r"[^0-9a-z]+", "", base)


def _to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return obj


def save_cache(cache_dir: str | Path, video_id: str, states: Sequence[Any], tracks: Sequence[Any] = (),
               zones: Sequence[Any] = ()) -> Path:
    """Записать ``<cache>/<video_id>/{states,tracks,zones}.json`` (для A03: кэш перед правилами)."""
    d = Path(cache_dir) / str(video_id)
    d.mkdir(parents=True, exist_ok=True)
    for name, items in (("states", states), ("tracks", tracks), ("zones", zones)):
        with open(d / f"{name}.json", "w", encoding="utf-8") as fh:
            json.dump([_to_jsonable(x) for x in items], fh, ensure_ascii=False)
    return d


def _read_json_any(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
        if path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz"):
            return [json.loads(line) for line in fh if line.strip()]
        return json.load(fh)


def _items(obj: Any, key: str) -> list:
    if isinstance(obj, dict):
        obj = obj.get(key, [])
    return list(obj) if isinstance(obj, list) else []


def _sibling(path: Path, video_stem: str, kind: str) -> Optional[Path]:
    for cand in (path.parent / f"{kind}.json", path.parent / f"{video_stem}.{kind}.json",
                 path.parent / f"{kind}.json.gz", path.parent / f"{video_stem}.{kind}.json.gz"):
        if cand.exists() and cand != path:
            return cand
    return None


def load_cache_dir(cache_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Найти кэши состояний: ``{norm_id: {"name", "states", "tracks", "zones", "path"}}``."""
    root = Path(cache_dir)
    out: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return out
    files = sorted(p for p in root.rglob("*") if p.is_file() and re.search(r"\.jsonl?(\.gz)?$", p.name))
    for p in files:
        lname = p.name.lower()
        if lname.startswith(_NOT_STATES) or re.search(r"\.(tracks|zones)\.json", lname):
            continue
        if lname.startswith("states."):
            name = p.parent.name
        elif ".states." in lname:
            name = p.name.split(".states.")[0]
        else:
            name = p.name.split(".json")[0]
        try:
            data = _read_json_any(p)
        except (OSError, ValueError) as exc:
            print(f"[replay] пропуск {p}: {exc}")
            continue
        states = _items(data, "states")
        if not states or not isinstance(states[0], dict) or "t" not in states[0]:
            continue
        tracks = _items(data, "tracks")
        zones = _items(data, "zones")
        stem = p.name.split(".")[0]
        for kind, holder in (("tracks", tracks), ("zones", zones)):
            if not holder:
                sib = _sibling(p, stem, kind)
                if sib is not None:
                    try:
                        holder.extend(_items(_read_json_any(sib), kind))
                    except (OSError, ValueError):
                        pass
        manifest = p.parent / "manifest.json"
        video_info: dict[str, Any] = {}
        if lname.startswith("states.") and manifest.exists():
            try:
                video_info = dict((_read_json_any(manifest) or {}).get("video") or {})
            except (OSError, ValueError, AttributeError):
                video_info = {}
        key = normalize_video_id(name)
        if key in out:
            print(f"[replay] дубликат id {key}: {p} (оставлен {out[key]['path']})")
            continue
        out[key] = {"name": name, "states": states, "tracks": tracks, "zones": zones, "path": str(p),
                    "video_info": video_info}
    return out


def load_name_map(path: str | Path) -> dict[str, str]:
    """sha256 → имя файла: ``summary.json`` батча A03 (``file``/``name`` + ``sha256``) или CSV ``sha256,file``."""
    p = Path(path)
    out: dict[str, str] = {}
    if p.suffix.lower() == ".json":
        rows = _read_json_any(p)
        rows = rows.get("rows", rows) if isinstance(rows, dict) else rows
        for r in rows or []:
            if isinstance(r, dict) and r.get("sha256") and (r.get("file") or r.get("name")):
                out[str(r["sha256"]).lower()] = str(r.get("file") or r.get("name"))
        return out
    text = p.read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    for r in rows:
        low = {k.strip().lower(): (v or "").strip() for k, v in r.items() if k}
        sha = low.get("sha256") or low.get("sha")
        name = low.get("file") or low.get("name") or low.get("video")
        if sha and name:
            out[sha.lower()] = name
    return out


def hash_videos(folder: str | Path) -> dict[str, str]:
    """sha256 → имя файла для видео в папке (локально у L2; содержимое видео никуда не передаётся)."""
    import hashlib

    out: dict[str, str] = {}
    for p in sorted(Path(folder).iterdir()):
        if p.is_file() and _VIDEO_EXT.search(p.name):
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            out[h.hexdigest()] = p.name
    return out


_POS = {"1", "true", "yes", "да", "violation", "нарушение", "есть", "y", "+"}
_NEG = {"0", "false", "no", "нет", "no_violation", "без нарушения", "безнарушения", "норма", "n", "-", "clean"}


def parse_label(value: Any) -> Optional[bool]:
    s = str(value or "").strip().lower().replace("ё", "е")
    if not s:
        return None
    s2 = re.sub(r"\s+", " ", s)
    if s2 in _NEG or s2.startswith("без") or s2.startswith("no_") or s2.startswith("нет"):
        return False
    if s2 in _POS or "наруш" in s2 or "violation" in s2:
        return True
    return None


def parse_timecode(value: Any) -> Optional[float]:
    """``22`` / ``22 секунда`` / ``0:22`` / ``00:01:05`` / ``22.5`` → секунды."""
    s = str(value or "").strip().lower().replace(",", ".")
    if not s:
        return None
    m = re.search(r"(\d+):(\d{1,2})(?::(\d{1,2}))?", s)
    if m:
        parts = [int(x) for x in m.groups() if x is not None]
        sec = 0
        for p in parts:
            sec = sec * 60 + p
        return float(sec)
    m = re.search(r"\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


_COL_FILE = ("file", "filename", "video", "name", "id", "файл", "видео", "имя", "ролик")
_COL_LABEL = ("label", "verdict", "class", "violation", "target", "gt", "метка", "нарушение", "класс", "вердикт")
_COL_TIME = ("time", "timecode", "sec", "second", "seconds", "t", "секунда", "время", "таймкод", "секунды")


def _pick(header: list[str], names: Iterable[str]) -> Optional[int]:
    low = [h.strip().lower() for h in header]
    for n in names:
        if n in low:
            return low.index(n)
    for i, h in enumerate(low):
        if any(n in h for n in names if len(n) > 2):
            return i
    return None


def load_labels(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Разметка CSV → ``{norm_id: [{"name", "violation", "t"}]}`` (список — на случай дублей)."""
    raw = Path(path).read_bytes()
    text = None
    for enc in ("utf-8-sig", "cp1251"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = [r for r in csv.reader(io.StringIO(text), dialect) if any(c.strip() for c in r)]
    if not rows:
        return {}
    header = rows[0]
    low_header = [h.strip().lower() for h in header]

    def _exact(name: str) -> Optional[int]:
        return low_header.index(name) if name in low_header else None

    i_tall, i_conflict, i_group, i_md5 = _exact("t_all"), _exact("conflict"), _exact("group"), _exact("md5")
    i_file = _exact("file") if _exact("file") is not None else _pick(header, _COL_FILE)
    i_label = _exact("label") if _exact("label") is not None else _pick(header, _COL_LABEL)
    i_time = _exact("t_sec") if _exact("t_sec") is not None else _pick(header, _COL_TIME)
    if i_file is None:
        i_file, i_label, i_time = 0, 1 if len(header) > 1 else None, 2 if len(header) > 2 else None
        body = rows
    else:
        body = rows[1:]
    out: dict[str, list[dict[str, Any]]] = {}
    for r in body:
        if i_file >= len(r):
            continue
        name = r[i_file].strip()
        if not name:
            continue
        label = parse_label(r[i_label]) if i_label is not None and i_label < len(r) else None
        t = parse_timecode(r[i_time]) if i_time is not None and i_time < len(r) else None
        if label is None and t is not None:
            label = True  # есть секунда события → нарушение
        entry: dict[str, Any] = {"name": name, "violation": label, "t": t}
        # схема L1 (eval.labels): md5, group, t_all, conflict — дубликаты и конфликт таймкодов
        if i_tall is not None and i_tall < len(r):
            alts = [parse_timecode(x) for x in re.split(r"[;|,/ ]+", r[i_tall]) if x.strip()]
            entry["t_all"] = sorted({a for a in alts if a is not None})
        if i_conflict is not None and i_conflict < len(r):
            entry["conflict"] = str(r[i_conflict]).strip().lower() in ("1", "true", "yes", "да", "y")
        if i_group is not None and i_group < len(r) and r[i_group].strip():
            entry["group"] = r[i_group].strip()
        if i_md5 is not None and i_md5 < len(r) and r[i_md5].strip():
            entry["md5"] = r[i_md5].strip().lower()
        out.setdefault(normalize_video_id(name), []).append(entry)
    return out
