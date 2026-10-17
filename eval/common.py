"""Общие утилиты eval: конфиг, нормализация имён видео, JSON-ввод/вывод."""
from __future__ import annotations

import copy
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Optional

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
DEFAULT_CONFIG = EVAL_DIR / "config.yaml"

# Латинская X/x, кириллическая Х/х, греческая Χ/χ — всё приводим к латинской X.
_X_CHARS = "XxХхΧχ"
_ID_RE = re.compile(rf"(?<![A-Za-zА-Яа-яЁё])[{_X_CHARS}]\s*[-‐‑‒–—_ .]*\s*0*(\d{{1,4}})(?!\d)")


def deep_merge(base: dict, override: Optional[dict]) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: Optional[str | os.PathLike] = None, overrides: Optional[dict] = None) -> dict:
    """config.yaml (+ необязательный пользовательский YAML поверх, + dict-переопределения)."""
    import yaml  # лениво

    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if path and Path(path).resolve() != DEFAULT_CONFIG:
        with open(path, encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    return deep_merge(cfg, overrides)


def normalize_video_id(name: Any) -> Optional[str]:
    """«Х-3.mov», «x_03», «X 3», «Х-3 (копия).MOV» → «X-3». Нет шаблона X<n> → None."""
    if name is None:
        return None
    s = unicodedata.normalize("NFKC", str(name)).strip()
    s = Path(s).name if ("/" in s or "\\" in s) else s
    m = _ID_RE.search(s)
    if not m:
        return None
    return f"X-{int(m.group(1))}"


def video_sort_key(video_id: str) -> tuple:
    m = re.search(r"(\d+)", video_id or "")
    return (0, int(m.group(1)), video_id) if m else (1, 0, video_id or "")


def list_videos(folder: str | os.PathLike, exts: Iterable[str] = (".mov", ".mp4", ".m4v", ".avi", ".mkv")) -> list[Path]:
    exts_l = {e.lower() for e in exts}
    p = Path(folder)
    if p.is_file():
        return [p]
    return sorted((x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in exts_l),
                  key=lambda x: video_sort_key(normalize_video_id(x.name) or x.stem))


def video_id_for_path(path: str | os.PathLike) -> str:
    """ID видео по имени файла; если шаблона X<n> нет — нормализованный stem."""
    p = Path(path)
    return normalize_video_id(p.name) or unicodedata.normalize("NFKC", p.stem).strip()


def to_jsonable(obj: Any) -> Any:
    """dataclass / tuple / numpy → чистый JSON."""
    import dataclasses

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "item") and callable(getattr(obj, "item")) and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: str | os.PathLike, data: Any, indent: Optional[int] = 1) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=False, indent=indent)
    os.replace(tmp, p)


def read_json(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ffprobe_info(path: str | os.PathLike) -> dict:
    """Длительность/fps/размер (с учётом rotation) через ffprobe; {} если ffprobe недоступен."""
    import shutil
    import subprocess

    if not shutil.which("ffprobe"):
        return {}
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=width,height,r_frame_rate,duration:stream_side_data=rotation:stream_tags=rotate:format=duration",
           "-of", "json", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False).stdout
        d = json.loads(out or "{}")
    except Exception:  # noqa: BLE001
        return {}
    st = (d.get("streams") or [{}])[0]
    w, h = int(st.get("width") or 0), int(st.get("height") or 0)
    rot = 0.0
    for sd in st.get("side_data_list") or []:
        if "rotation" in sd:
            rot = float(sd["rotation"])
    if not rot and (st.get("tags") or {}).get("rotate"):
        rot = float(st["tags"]["rotate"])
    if int(round(abs(rot))) % 180 == 90:
        w, h = h, w
    num, _, den = str(st.get("r_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    dur = st.get("duration") or (d.get("format") or {}).get("duration") or 0
    return {"width": w, "height": h, "fps": fps, "duration": float(dur), "rotation": rot}
