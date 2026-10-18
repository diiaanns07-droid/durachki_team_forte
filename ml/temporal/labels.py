"""Разметка: нормализация имён (латинская X / кириллическая Х), парсинг labels.csv, группы дубликатов."""
from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

# кириллические буквы, визуально совпадающие с латинскими
_HOMOGLYPHS = str.maketrans({
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C",
    "Т": "T", "У": "Y", "Х": "X", "а": "A", "в": "B", "е": "E", "к": "K", "м": "M", "н": "H",
    "о": "O", "р": "P", "с": "C", "т": "T", "у": "Y", "х": "X",
})
VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv", ".m4v", ".webm"}

_POS_WORDS = ("наруш", "violation", "viol", "yes", "да", "true", "1", "positive", "pos", "есть")
_NEG_WORDS = ("без нар", "нет нар", "no_violation", "no violation", "none", "no", "нет", "false", "0",
              "clean", "negative", "neg", "чист", "норм")


def normalize_name(name: str) -> str:
    """'Х-3.mov' / 'x_3' / 'X 03' -> 'X-3'. Кириллица-омоглифы -> латиница, верхний регистр."""
    stem = Path(str(name).strip()).name
    if Path(stem).suffix.lower() in VIDEO_EXTS:
        stem = Path(stem).stem
    s = stem.translate(_HOMOGLYPHS).upper().strip()
    s = re.sub(r"[\s_]+", "-", s)
    m = re.fullmatch(r"([A-Z]+)-*0*(\d+)", s)
    if m:
        return f"{m.group(1)}-{int(m.group(2))}"
    return s


def parse_label_value(v: str) -> Optional[int]:
    s = str(v or "").strip().lower()
    if not s:
        return None
    for w in _NEG_WORDS:  # негативы первыми: «без нарушения» содержит «наруш»
        if s == w or (len(w) > 2 and w in s):
            return 0
    for w in _POS_WORDS:
        if s == w or (len(w) > 2 and w in s):
            return 1
    return None


def parse_times(v: str) -> list[float]:
    """'22' / '22 секунда' / '0:22' / '57 и 68' / '1:02, 1:10' -> [сек...]."""
    s = str(v or "").strip()
    if not s:
        return []
    out: list[float] = []
    for m in re.finditer(r"(?:(\d+):)?(\d+):(\d+(?:[.,]\d+)?)|(\d+(?:[.,]\d+)?)", s):
        if m.group(3) is not None:
            h = int(m.group(1) or 0)
            out.append(h * 3600 + int(m.group(2)) * 60 + float(m.group(3).replace(",", ".")))
        else:
            out.append(float(m.group(4).replace(",", ".")))
    return out


@dataclass
class VideoLabel:
    name: str                      # нормализованное имя
    label: int                     # 1 — нарушение, 0 — без нарушения
    times: list[float] = field(default_factory=list)
    type_id: str = ""              # опционально (если локальная команда разметит тип)
    files: list[str] = field(default_factory=list)  # исходные пути/имена
    group: str = ""                # id группы дубликатов (для LOO)
    md5: str = ""
    pseudo_type_id: str = ""       # предварительный тип (VLM/авто), в обучение — только по флагу use_pseudo_types
    sources: list[dict] = field(default_factory=list)  # исходные метки файлов группы (не теряются при слиянии)
    conflict: bool = False         # в группе дубликатов метки «нарушение»/«без» расходятся
    time_conflict: bool = False    # в группе дубликатов разные таймкоды (> 1.5 с) — решает человек


_NAME_KEYS = ("file", "name", "video", "filename", "файл", "имя", "видео", "ролик", "id")
_LABEL_KEYS = ("label", "violation", "verdict", "class", "target", "наруш", "вердикт", "класс", "метка")
_TIME_KEYS = ("t_all", "t_sec", "time", "sec", "second", "timecode", "ts", "секунд", "время", "таймкод",
              "момент")
_TYPE_KEYS = ("type_id", "type", "тип")
_TYPE_STATUS_KEYS = ("type_status", "type_source", "статус типа", "статус_типа")
_PSEUDO_WORDS = ("pseudo", "vlm", "auto", "предвар", "псевдо", "draft", "unverified", "непровер")


def is_pseudo_status(v: str) -> bool:
    """Статус метки типа: подтверждённая (человек) или предварительная (VLM/авто). Пусто — подтверждённая."""
    s = str(v or "").strip().lower()
    return any(w in s for w in _PSEUDO_WORDS)


def _pick(header: list[str], keys: Iterable[str], exclude: set[int]) -> Optional[int]:
    low = [h.strip().lower() for h in header]
    for k in keys:
        for i, h in enumerate(low):
            if i in exclude:
                continue
            if h == k or (len(k) > 3 and (h.startswith(k) or k in h)):
                return i
    return None


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def read_labels(path: str | Path, time_offset_s: float = 0.0) -> dict[str, VideoLabel]:
    """Гибкий парсер labels.csv (разделитель , ; или таб; заголовок по-русски или по-английски)."""
    text = _read_text(Path(path))
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = [r for r in csv.reader(io.StringIO(text), dialect) if any(c.strip() for c in r)]
    if not rows:
        return {}
    header = rows[0]
    used: set[int] = set()
    i_name = _pick(header, _NAME_KEYS, used)
    has_header = i_name is not None
    i_status = None
    if not has_header:  # без заголовка: имя, метка, время[, тип]
        i_name, i_label, i_time, i_type = 0, 1, 2, 3
        body = rows
    else:
        used.add(i_name)
        i_status = _pick(header, _TYPE_STATUS_KEYS, used)
        if i_status is not None:
            used.add(i_status)
        i_type = _pick(header, _TYPE_KEYS, used)
        if i_type is not None:
            used.add(i_type)
        i_time = _pick(header, _TIME_KEYS, used)  # время раньше метки: «таймкод нарушения» — это время
        if i_time is not None:
            used.add(i_time)
        i_label = _pick(header, _LABEL_KEYS, used)
        body = rows[1:]

    out: dict[str, VideoLabel] = {}
    for r in body:
        def cell(i: Optional[int]) -> str:
            return r[i].strip() if i is not None and i < len(r) else ""

        raw_name = cell(i_name)
        if not raw_name:
            continue
        name = normalize_name(raw_name)
        times = [t + time_offset_s for t in parse_times(cell(i_time))]
        lab_cell = cell(i_label)
        lab = parse_label_value(lab_cell)
        if not times and lab != 0 and lab_cell.strip() not in ("0", "1") and re.search(r"\d", lab_cell):
            times = [t + time_offset_s for t in parse_times(lab_cell)]  # «22 секунда» в столбце метки
        if lab is None:  # нет явной метки: есть таймкод -> нарушение
            lab = 1 if times else 0
        typ = cell(i_type).upper().translate(_HOMOGLYPHS)
        typ = typ if re.fullmatch(r"V\d{2}", typ) else ""
        pseudo = ""
        if typ and is_pseudo_status(cell(i_status)):  # предварительный тип не выдаём за подтверждённый
            typ, pseudo = "", typ
        if name in out:  # повтор строки — объединяем
            prev = out[name]
            prev.label = max(prev.label, lab)
            prev.times = sorted(set(prev.times + times))
            prev.type_id = prev.type_id or typ
            prev.pseudo_type_id = prev.pseudo_type_id or pseudo
            prev.files.append(raw_name)
        else:
            out[name] = VideoLabel(name=name, label=lab, times=times if lab else [], type_id=typ,
                                   files=[raw_name], pseudo_type_id=pseudo)
    return out


def file_md5(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def assign_groups(labels: dict[str, VideoLabel], md5s: dict[str, str] | None = None,
                  duplicate_groups: Iterable[Iterable[str]] = ()) -> dict[str, str]:
    """name -> group id. Дубликаты (одинаковый md5 или из списка) получают общую группу (union-find)."""
    parent = {n: n for n in labels}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            lo, hi = sorted((ra, rb), key=_natural_key)
            parent[hi] = lo

    by_md5: dict[str, str] = {}
    for n, h in (md5s or {}).items():
        if n not in parent or not h:
            continue
        if h in by_md5:
            union(n, by_md5[h])
        else:
            by_md5[h] = n
    for grp in duplicate_groups or ():
        names = [normalize_name(g) for g in grp]
        names = [g for g in names if g in parent]
        for a, b in zip(names, names[1:]):
            union(a, b)
    groups = {n: find(n) for n in labels}
    for n, lab in labels.items():
        lab.group = groups[n]
        if md5s and n in md5s:
            lab.md5 = md5s[n]
    return groups


def merge_duplicates(labels: dict[str, VideoLabel], conflict_policy: str = "positive",
                     time_conflict_policy: str = "both") -> dict[str, VideoLabel]:
    """Свернуть группу дубликатов в одну обучающую запись, НЕ выбирая метку произвольно:
    все исходные метки остаются в ``sources``; таймкоды позитивных файлов объединяются
    (Х-4≡Х-47 -> якоря [57, 68]); расхождение «нарушение/без» помечается ``conflict`` и решается
    явной политикой: positive (пропуск нарушения хуже) или exclude (группа исключается)."""
    grouped: dict[str, list[VideoLabel]] = {}
    for n in sorted(labels, key=_natural_key):
        grouped.setdefault(labels[n].group or n, []).append(labels[n])
    out: dict[str, VideoLabel] = {}
    for g, items in grouped.items():
        sources = [{"name": it.name, "label": it.label, "times": list(it.times), "type_id": it.type_id,
                    "files": list(it.files)} for it in items]
        conflict = len({it.label for it in items}) > 1
        if conflict and conflict_policy == "exclude":
            continue
        pos_times = [sorted(it.times) for it in items if it.label and it.times]
        time_conflict = any(min(abs(a - b) for b in other) > 1.5 for ts in pos_times for other in pos_times
                            for a in ts if other is not ts and other)
        label = max(it.label for it in items)
        times = sorted({t for it in items if it.label for t in it.times}) if label else []
        types = {it.type_id for it in items if it.label and it.type_id}
        ptypes = {it.pseudo_type_id for it in items if it.label and it.pseudo_type_id}
        out[g] = VideoLabel(name=g, label=label, times=times, type_id=types.pop() if len(types) == 1 else "",
                            pseudo_type_id=ptypes.pop() if len(ptypes) == 1 else "",
                            files=[f for it in items for f in it.files], group=g, md5=items[0].md5,
                            sources=sources, conflict=conflict, time_conflict=time_conflict)
        if time_conflict and time_conflict_policy == "video_only":
            out[g].times = []  # кадровый надзор снят, остаётся метка уровня видео (MIL); таймкод решает человек
    return out


def _natural_key(s: str) -> tuple:
    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s))


def find_videos(root: str | Path) -> dict[str, Path]:
    """Нормализованное имя -> путь для всех видео в каталоге (рекурсивно)."""
    out: dict[str, Path] = {}
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            out.setdefault(normalize_name(p.name), p)
    return out
