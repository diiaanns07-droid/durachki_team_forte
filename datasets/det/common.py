"""Общее для пайплайна разметки детектора: классы, пути, поиск видео, таймкоды, группы дубликатов.

Только stdlib: модуль импортируют и скрипты из venv OWLv2, и скрипты из venv Label Studio.
"""
from __future__ import annotations

import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

for _s in (sys.stdout, sys.stderr):              # консоль Windows (cp1251) не должна ронять скрипт на «→», «≡»
    try:
        _s.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

DET_DIR = Path(__file__).resolve().parent          # datasets/det
ROOT = DET_DIR.parents[1]                          # корень репо
DATA = Path(os.environ.get("DET_DATA") or ROOT / "data")   # DET_DATA — если данные на другом диске
RAW_DIR = DATA / "raw"
EXTRA_DIR = ROOT / "datasets" / "extra"
FRAMES_DIR = DATA / "frames"                       # кадры (НЕ в git)
INDEX_CSV = FRAMES_DIR / "index.csv"
PRED_JSONL = FRAMES_DIR / "_owlv2" / "predictions.jsonl"
LS_DIR = FRAMES_DIR / "_ls"
LABELS_CSV = DATA / "labels.csv"                   # схема eval.labels (tools/dataprep/build_labels.py, L1)

VIDEO_EXT = (".mov", ".mp4", ".avi", ".mkv", ".m4v")

# Единый список классов. Порядок = category_id в COCO (1..N), его не менять: на него завязаны веса.
# contract — имя в contracts.LABELS (у контракта закрытый сейф называется «safe»).
# prompts/thr/max — только для пред-разметки OWLv2; подобраны по реальным кадрам 17.10 (высокий угловой ракурс):
# OWLv2 хорошо находит, где объект, но путает классы — их правит человек, а пороги режут шум.
CLASSES: list[dict] = [
    {"name": "cash", "contract": "cash", "color": "#2ca02c", "thr": 0.10, "max": 6,
     "prompts": ["a stack of banknotes", "paper money", "cash banknotes on a desk", "banknotes in a hand"]},
    {"name": "cash_bag", "contract": "cash_bag", "color": "#8c564b", "thr": 0.10, "max": 3,
     "prompts": ["a cash collection bag", "a sealed money bag", "a plastic security bag", "a canvas sack"]},
    {"name": "safe_closed", "contract": "safe", "color": "#7f7f7f", "thr": 0.12, "max": 2,
     "prompts": ["a closed metal safe", "a steel safe box", "a closed safe door"]},
    {"name": "safe_open", "contract": "safe_open", "color": "#d62728", "thr": 0.12, "max": 2,
     "prompts": ["an open safe", "an open safe with shelves", "an open metal cabinet"]},
    {"name": "cell_phone", "contract": "cell_phone", "color": "#9467bd", "thr": 0.18, "max": 3,
     "prompts": ["a cell phone", "a smartphone", "a mobile phone in a hand"]},
    {"name": "cup", "contract": "cup", "color": "#17becf", "thr": 0.12, "max": 6,
     "prompts": ["a cup", "a mug", "a paper cup"]},
    {"name": "food", "contract": "food", "color": "#bcbd22", "thr": 0.12, "max": 6,
     "prompts": ["food", "a sandwich", "a snack", "fruit"]},
    {"name": "money_counter", "contract": "money_counter", "color": "#ff7f0e", "thr": 0.12, "max": 2,
     "prompts": ["a banknote counting machine", "a money counter machine", "a cash counting device"]},
    {"name": "door_open", "contract": "door_open", "color": "#e377c2", "thr": 0.20, "max": 1,
     "prompts": ["an open door", "an open doorway"]},
    {"name": "person", "contract": "person", "color": "#1f77b4", "thr": 0.22, "max": 6,
     "prompts": ["a person", "a person seen from above", "a head and shoulders seen from above"]},
]
CLASS_NAMES = [c["name"] for c in CLASSES]
CAT_ID = {n: i + 1 for i, n in enumerate(CLASS_NAMES)}
# Редкие на COCO-детекторах классы — ради них и дообучаем; кадры с ними приоритетны при отборе.
RARE = {"cash", "cash_bag", "safe_open", "safe_closed", "money_counter", "door_open"}
# Взаимоисключающие состояния одного объекта: из двух пересекающихся боксов пред-разметки остаётся один.
EXCLUSIVE = [("safe_open", "safe_closed")]


def norm_id(name: str) -> str | None:
    """«Х-3» (кириллическая Х), «x - 03» → «X-3»; не похоже на X-n → None."""
    m = re.fullmatch(r"\s*[XХxх]\s*-\s*(\d+)\s*", name)
    return f"X-{int(m.group(1))}" if m else None


def safe_id(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "video"


@dataclass
class Video:
    video_id: str
    path: Path
    src: str                                   # raw | extra
    md5: str = ""
    group: str = ""                            # id представителя группы одинаковых по md5 файлов
    times: list[float] = field(default_factory=list)   # таймкоды нарушений, сек


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _id_key(vid: str):
    m = re.fullmatch(r"X-(\d+)", vid)
    return (0, int(m.group(1)), "") if m else (1, 0, vid)


def discover_videos(raw_dir: Path = RAW_DIR, extra_dir: Path = EXTRA_DIR) -> list[Video]:
    """data/raw/*.mov + datasets/extra/**/<видео>. id: X-n для датасета, иначе имя файла (с префиксом при коллизии)."""
    out: dict[str, Video] = {}
    seen: set[Path] = set()
    for src, base, rec in (("raw", raw_dir, False), ("extra", extra_dir, True)):
        if not base.is_dir():
            continue
        files = base.rglob("*") if rec else base.glob("*")
        for p in sorted(files):
            if not p.is_file() or p.suffix.lower() not in VIDEO_EXT or p.resolve() in seen:
                continue
            seen.add(p.resolve())
            vid = norm_id(p.stem) or safe_id(p.stem)
            if vid in out:
                vid = f"extra_{vid}"
            if vid in out:
                raise SystemExit(f"неоднозначный id видео {vid}: {p} и {out[vid].path}")
            out[vid] = Video(vid, p, src)
    return sorted(out.values(), key=lambda v: _id_key(v.video_id))


def _parse_times(*cells: str) -> list[float]:
    ts: list[float] = []
    for c in cells:
        for part in re.split(r"[;,|\s]+", (c or "").strip()):
            if part:
                try:
                    ts.append(float(part))
                except ValueError:
                    pass
    return sorted(set(ts))


def read_timecodes(paths: list[Path]) -> dict[str, list[float]]:
    """CSV со столбцами video_id|file и t_sec/t_all (несколько таймкодов — через «;»). Ключ — нормализованный id."""
    res: dict[str, set[float]] = {}
    for p in paths:
        if not p.is_file():
            continue
        with open(p, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                key = (r.get("video_id") or Path(r.get("file") or "").stem or "").strip()
                if not key:
                    continue
                key = norm_id(key) or safe_id(key)
                ts = _parse_times(r.get("t_sec", ""), r.get("t_all", ""))
                if (r.get("label") or "").strip() in ("no_violation", "clean"):   # clean — datasets/extra
                    ts = []
                res.setdefault(key, set()).update(ts)
    return {k: sorted(v) for k, v in res.items()}


def assign_groups(videos: list[Video]) -> None:
    """Одинаковые файлы (md5) — одна группа: в один сплит и кадры только с одного (таймкоды объединяются)."""
    by_md5: dict[str, list[Video]] = {}
    for v in videos:
        v.md5 = v.md5 or _md5(v.path)
        by_md5.setdefault(v.md5, []).append(v)
    for members in by_md5.values():
        rep = min(members, key=lambda v: _id_key(v.video_id)).video_id
        for v in members:
            v.group = rep


def find_ffmpeg() -> str:
    """$FFMPEG → ffmpeg в PATH → бинарник из пакета imageio-ffmpeg."""
    exe = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("ffmpeg не найден: поставьте ffmpeg в PATH, задайте FFMPEG=... или pip install imageio-ffmpeg") from None


def probe_duration(ffmpeg: str, path: Path) -> float:
    """Длительность по выводу `ffmpeg -i` (ffprobe в imageio-ffmpeg нет)."""
    err = subprocess.run([ffmpeg, "-hide_banner", "-nostdin", "-i", str(path)],
                         capture_output=True, text=True, encoding="utf-8", errors="replace").stderr
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
    if not m:
        raise RuntimeError(f"не удалось определить длительность {path}: {err[-300:]}")
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def jpeg_size(path: Path) -> tuple[int, int]:
    """(width, height) из маркера SOF JPEG без PIL."""
    with open(path, "rb") as f:
        data = f.read(1 << 16)
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        seg = int.from_bytes(data[i + 2:i + 4], "big")
        if marker in (0xC0, 0xC1, 0xC2):
            return int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big")
        i += 2 + seg
    raise ValueError(f"не JPEG или нет SOF: {path}")


def frame_name(video_id: str, t: float) -> str:
    """Относительный путь кадра от data/frames: «X-1/X-1_t0012.250.jpg»."""
    return f"{video_id}/{video_id}_t{t:08.3f}.jpg"


INDEX_FIELDS = ["file", "video_id", "group", "t_sec", "dense", "src", "width", "height"]


def read_index(path: Path = INDEX_CSV) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"нет {path}: сначала extract_frames.py")
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["t_sec"] = float(r["t_sec"])
        r["dense"] = int(r["dense"])
        r["width"], r["height"] = int(r["width"]), int(r["height"])
    return rows
