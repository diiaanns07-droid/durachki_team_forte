"""Разметка: open_labels.xlsx → data/labels.csv (+ data/labels_types.csv, когда появится).

* имена: латинская X / кириллическая Х / «x_03» → ``X-<n>``;
* таймкоды: «22 секунда» → 22.0, «1минута 8 секунда» → 68.0, «1:08» → 68.0, «без нарушения» → нет;
* md5-дедуп: одинаковые файлы (Х-4≡Х-47, Х-5≡Х-48) → одна группа, флаг ``conflict``,
  если в группе разные метки/таймкоды.

CLI::

    python -m eval.labels --xlsx data/open_labels.xlsx --videos data/videos --out data/labels.csv
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .common import list_videos, normalize_video_id, video_id_for_path, video_sort_key

log = logging.getLogger("eval.labels")

LABEL_VIOLATION = "violation"
LABEL_CLEAN = "no_violation"
CSV_COLUMNS = ["video_id", "file", "label", "t_sec", "md5", "group", "t_all", "conflict", "raw", "sha256"]
LABEL_CONFLICT = "conflict"   # в md5/sha256-группе разные метки — исключаем из бинарных метрик, не выбираем сами
LABEL_UNKNOWN = "unknown"     # пустая/непонятная метка в labels.csv — исключается (unknown ≠ false)

_NO_VIOLATION_RE = re.compile(
    r"(без\s*нарушени|нет\s*нарушени|нарушени[йяе]\s*нет|не\s*выявлен|не\s*обнаружен|отсутству|"
    r"^\s*нет\s*$|^\s*-\s*$|^\s*0\s*$|чист|норм[аы]?\b|no[\s_-]*violation|^\s*none\s*$|^\s*clean\s*$|^\s*ok\s*$)",
    re.I,
)
_VIOLATION_RE = re.compile(r"(нарушени|^\s*да\s*$|^\s*есть\s*$|^\s*1\s*$|violation|^\s*yes\s*$|^\s*true\s*$)", re.I)

_HDR_ID = re.compile(r"(видео|файл|назван|имя|ролик|запис|video|file|name|^id$|^№\s*видео)", re.I)
_HDR_LABEL = re.compile(r"(нарушен|метк|label|разметк|результат|статус|вердикт|класс|итог|наличи|violation|событи)", re.I)
_HDR_TIME = re.compile(r"(сек|время|тайм|time|момент|когда|second|таймкод)", re.I)

_UNIT_H = r"(?:час\w*|ч)"
_UNIT_M = r"(?:минут\w*|мин|м|min\w*|m)"
_UNIT_S = r"(?:секунд\w*|сек|с|sec\w*|s)"
_ORD = r"(?:-?(?:ой|ый|ий|ая|ую|ей|й|я|ю|е))?"
_TOKEN_RE = re.compile(
    rf"(?P<num>\d+(?:[.,]\d+)?)\s*{_ORD}\s*(?P<unit>{_UNIT_H}|{_UNIT_M}|{_UNIT_S})?(?![а-яa-z])", re.I)
_CLOCK_RE = re.compile(r"(?<![\d.,])(\d{1,2}):(\d{2})(?::(\d{2}))?(?:[.,](\d+))?(?![\d:])")
_JOINER_RE = re.compile(r"^[\s,;.и+]*$", re.I)


@dataclass
class LabelRow:
    video_id: str
    label: str                      # violation | no_violation
    times: list[float] = field(default_factory=list)
    file: str = ""
    md5: str = ""
    group: str = ""
    conflict: bool = False
    raw: str = ""                   # исходный текст ячеек — сохраняется как есть
    sha256: str = ""

    @property
    def t_sec(self) -> Optional[float]:
        return self.times[0] if self.times else None


@dataclass
class GroundTruth:
    """GT по видео (или по md5-группе при dedupe).

    ``events`` — список GT-событий, каждое = список альтернативных таймкодов. Конфликт таймкодов в группе
    дубликатов (Х-4: 57 с / Х-47: 68 с) не решаем выбором: это одно событие с альтернативами [57, 68]
    (политика ``conflict_policy`` в config.yaml), исходные строки — в ``raw``.
    """

    video_id: str
    label: str                       # violation | no_violation | conflict
    times: list[float] = field(default_factory=list)
    group: str = ""
    members: list[str] = field(default_factory=list)
    conflict: bool = False
    types: list[tuple[float, str]] = field(default_factory=list)  # (t_sec, type_id) из labels_types.csv
    file: str = ""
    md5: str = ""
    events: list[list[float]] = field(default_factory=list)
    raw: dict[str, str] = field(default_factory=dict)   # video_id члена → исходная строка разметки

    @property
    def positive(self) -> bool:
        return self.label == LABEL_VIOLATION


# ----------------------------------------------------------------------------- парсинг текста
def _norm_text(text: Any) -> str:
    s = unicodedata.normalize("NFKC", str(text)).lower().replace("ё", "е")
    return re.sub(r"\s+", " ", s).strip()


def _unit_kind(unit: Optional[str]) -> Optional[str]:
    if not unit:
        return None
    u = unit.lower()
    if u.startswith("ч") or u.startswith("час"):
        return "h"
    if u.startswith("м") or u.startswith("min") or u == "m":
        return "m"
    return "s"


def parse_timecodes(text: Any, bare_number_is_seconds: bool = True) -> list[float]:
    """Все таймкоды в тексте, сек. «1минута 8 секунда» → [68.0]; «22 и 45 сек» → [22.0, 45.0]."""
    if text is None:
        return []
    if isinstance(text, bool):
        return []
    if isinstance(text, (int, float)):
        return [float(text)] if bare_number_is_seconds and text >= 0 else []
    if isinstance(text, _dt.timedelta):
        return [text.total_seconds()]
    if isinstance(text, (_dt.time, _dt.datetime)):
        return [time_cell_to_seconds(text)]
    s = _norm_text(text)
    if not s:
        return []
    # ID видео («Х-3») — не таймкод
    s = re.sub(r"(?<![a-zа-я])[xхχ]\s*[-‐‑‒–—_ .]*\s*\d{1,4}(?!\d)", " ", s)
    out: list[tuple[int, float]] = []
    taken: list[tuple[int, int]] = []
    for m in _CLOCK_RE.finditer(s):
        a, b, c, frac = m.group(1), m.group(2), m.group(3), m.group(4)
        sec = int(a) * 3600 + int(b) * 60 + int(c) if c is not None else int(a) * 60 + int(b)
        if frac:
            sec += float("0." + frac)
        out.append((m.start(), float(sec)))
        taken.append(m.span())

    def _free(span: tuple[int, int]) -> bool:
        return all(span[1] <= a or span[0] >= b for a, b in taken)

    tokens = [m for m in _TOKEN_RE.finditer(s) if _free(m.span())]
    i = 0
    n_tokens = len(tokens)
    while i < n_tokens:
        m = tokens[i]
        num = float(m.group("num").replace(",", "."))
        kind = _unit_kind(m.group("unit"))
        if kind in ("h", "m"):
            total = num * (3600 if kind == "h" else 60)
            j = i
            # «1 час 2 минуты 3 секунды», «1минута 8 секунда», «1 мин и 8 с»
            while j + 1 < n_tokens:
                nxt = tokens[j + 1]
                nk = _unit_kind(nxt.group("unit"))
                gap = s[tokens[j].end():nxt.start()]
                if not _JOINER_RE.match(gap):
                    break
                order = {"h": 0, "m": 1, "s": 2}
                cur_kind = _unit_kind(tokens[j].group("unit")) or "s"
                if nk is None:  # «1 минута 8» → 68
                    nk = "s" if cur_kind == "m" else "m"
                if order[nk] <= order[cur_kind]:
                    break
                total += float(nxt.group("num").replace(",", ".")) * {"h": 3600, "m": 60, "s": 1}[nk]
                j += 1
                if nk == "s":
                    break
            out.append((m.start(), total))
            i = j + 1
            continue
        if kind == "s" or (kind is None and bare_number_is_seconds):
            out.append((m.start(), num))
        i += 1
    out.sort(key=lambda x: x[0])
    res: list[float] = []
    for _, v in out:
        if v not in res:
            res.append(round(v, 3))
    return res


def time_cell_to_seconds(v: Any, max_plausible_s: float = 3600.0) -> float:
    """datetime.time из Excel → сек. «1:08», набранное в Excel, читается как 01:08:00 → 68 с."""
    if isinstance(v, _dt.datetime):
        v = v.time()
    if isinstance(v, _dt.timedelta):
        return v.total_seconds()
    sec = v.hour * 3600 + v.minute * 60 + v.second + v.microsecond / 1e6
    if sec >= max_plausible_s and v.second == 0 and v.microsecond == 0:
        sec = v.hour * 60 + v.minute
    return float(sec)


def parse_label(text: Any, time_text: Any = None) -> tuple[Optional[str], list[float]]:
    """(label, times). label=None — непонятная ячейка."""
    parts = [t for t in (text, time_text) if t is not None and str(t).strip() != ""]
    if not parts:
        return None, []
    s = " ".join(_norm_text(p) if not isinstance(p, (_dt.time, _dt.datetime, _dt.timedelta)) else "" for p in parts)
    if text is not None and _NO_VIOLATION_RE.search(_norm_text(text)):
        return LABEL_CLEAN, []
    times: list[float] = []
    for p in parts:
        times += [t for t in parse_timecodes(p) if t not in times]
    if times:
        return LABEL_VIOLATION, times
    if _NO_VIOLATION_RE.search(s):
        return LABEL_CLEAN, []
    if _VIOLATION_RE.search(s):
        return LABEL_VIOLATION, []
    return None, []


# ----------------------------------------------------------------------------- xlsx
def _detect_header(rows: list[tuple]) -> tuple[int, dict[str, int]]:
    """Ищем строку-заголовок в первых 6 строках. → (index, {'id','label','time': col})."""
    for ri, row in enumerate(rows[:6]):
        cols: dict[str, int] = {}
        for ci, v in enumerate(row):
            if v is None or not isinstance(v, str):
                continue
            t = _norm_text(v)
            if normalize_video_id(v):
                cols = {}
                break
            if "id" not in cols and _HDR_ID.search(t):
                cols["id"] = ci
            elif "label" not in cols and _HDR_LABEL.search(t):
                cols["label"] = ci
                if _HDR_TIME.search(t) and "time" not in cols:
                    cols.setdefault("label_has_time", ci)
            elif "time" not in cols and _HDR_TIME.search(t):
                cols["time"] = ci
        if "label" in cols or ("id" in cols and "time" in cols):
            return ri, cols
    return -1, {}


def read_labels_xlsx(path: str | Path) -> tuple[list[LabelRow], list[str]]:
    """Все листы книги → строки разметки (без md5). Возвращает (rows, warnings)."""
    import openpyxl  # лениво

    wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)
    rows_out: dict[str, LabelRow] = {}
    warnings: list[str] = []
    for ws in wb.worksheets:
        rows = [tuple(r) for r in ws.iter_rows(values_only=True)]
        hdr_i, cols = _detect_header(rows)
        for ri, row in enumerate(rows):
            if ri <= hdr_i or not any(c is not None and str(c).strip() for c in row):
                continue
            id_col = cols.get("id")
            vid = normalize_video_id(row[id_col]) if id_col is not None and id_col < len(row) else None
            if vid is None:
                for ci, c in enumerate(row):
                    if isinstance(c, str) and normalize_video_id(c):
                        vid, id_col = normalize_video_id(c), ci
                        break
            if vid is None:
                if any(isinstance(c, str) and c.strip() for c in row):
                    warnings.append(f"{ws.title}!{ri + 1}: нет ID видео: {row}")
                continue
            if "label" in cols:
                lab_cell = row[cols["label"]] if cols["label"] < len(row) else None
                time_cell = row[cols["time"]] if "time" in cols and cols["time"] < len(row) else None
            else:
                others = [c for ci, c in enumerate(row) if ci != id_col and c is not None and str(c).strip()]
                texty = [c for c in others if not isinstance(c, (int, float)) or isinstance(c, bool)]
                nums = [c for c in others if isinstance(c, (int, float)) and not isinstance(c, bool)]
                lab_cell = " ".join(str(c) for c in texty) if texty else None
                time_cell = nums[-1] if nums and (len(others) == 1 or texty) else None
                if lab_cell is None and time_cell is not None:
                    lab_cell, time_cell = time_cell, None
            # Сам ID в той же ячейке, что и метка («Х-3: 22 секунда»)
            if isinstance(lab_cell, str) and id_col is not None and cols.get("label") == id_col:
                lab_cell = re.sub(r"(?<![A-Za-zА-Яа-я])[XxХхΧχ]\s*[-‐‑‒–—_ .]*\s*\d{1,4}(?!\d)", " ", lab_cell)
            label, times = parse_label(lab_cell, time_cell)
            raw = " | ".join(str(c) for c in (lab_cell, time_cell) if c is not None)
            if label is None:
                warnings.append(f"{ws.title}!{ri + 1}: {vid}: не распознана метка: {raw!r}")
                continue
            if label == LABEL_VIOLATION and not times:
                warnings.append(f"{ws.title}!{ri + 1}: {vid}: нарушение без таймкода: {raw!r}")
            if vid in rows_out:  # несколько строк на видео — объединяем
                prev = rows_out[vid]
                if label == LABEL_VIOLATION:
                    prev.label = LABEL_VIOLATION
                prev.times += [t for t in times if t not in prev.times]
                prev.raw += " || " + raw
                warnings.append(f"{vid}: несколько строк разметки — объединены")
            else:
                rows_out[vid] = LabelRow(video_id=vid, label=label, times=times, raw=raw)
    wb.close()
    return sorted(rows_out.values(), key=lambda r: video_sort_key(r.video_id)), warnings


# ----------------------------------------------------------------------------- md5 / группы
def md5_file(path: str | Path, chunk: int = 1 << 20) -> str:
    return hash_file(path, chunk)[0]


def hash_file(path: str | Path, chunk: int = 1 << 20) -> tuple[str, str]:
    """(md5, sha256) за один проход."""
    h5, h256 = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h5.update(b)
            h256.update(b)
    return h5.hexdigest(), h256.hexdigest()


def map_video_files(videos_dir: str | Path, exts: Iterable[str] = (".mov", ".mp4", ".m4v", ".avi", ".mkv")
                    ) -> tuple[dict[str, Path], list[str]]:
    """{video_id: path}; коллизии имён (X-3 латиницей и Х-3 кириллицей — разные файлы) → X-3#2."""
    out: dict[str, Path] = {}
    warnings: list[str] = []
    for p in list_videos(videos_dir, exts):
        vid = video_id_for_path(p)
        if vid in out:
            k = 2
            while f"{vid}#{k}" in out:
                k += 1
            warnings.append(f"коллизия имён: {out[vid].name} и {p.name} → {vid}; второй = {vid}#{k}")
            vid = f"{vid}#{k}"
        out[vid] = p
    return out, warnings


def assign_groups(rows: list[LabelRow]) -> list[str]:
    """group = минимальный ID среди файлов с одинаковым содержимым (sha256, иначе md5);
    conflict — если метки/таймкоды различаются (исходные строки не меняем)."""
    warnings: list[str] = []
    by_md5: dict[str, list[LabelRow]] = {}
    for r in rows:
        key = r.sha256 or r.md5
        if key:
            by_md5.setdefault(key, []).append(r)
        r.group = r.video_id
    for md5, members in by_md5.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda r: video_sort_key(r.video_id))
        gid = members[0].video_id
        labels = {m.label for m in members}
        times = {tuple(sorted(m.times)) for m in members}
        conflict = len(labels) > 1 or len(times) > 1
        for m in members:
            m.group = gid
            m.conflict = conflict
        ids = ", ".join(m.video_id for m in members)
        warnings.append(f"дубликаты по md5 {md5[:8]}: {ids}" + (
            f" — КОНФЛИКТ разметки: {[(m.video_id, m.label, m.times) for m in members]}" if conflict else ""))
    return warnings


def build_labels(xlsx: str | Path, videos_dir: Optional[str | Path] = None,
                 with_md5: bool = True) -> tuple[list[LabelRow], list[str]]:
    rows, warnings = read_labels_xlsx(xlsx)
    if videos_dir:
        files, w2 = map_video_files(videos_dir)
        warnings += w2
        labeled = {r.video_id for r in rows}
        for r in rows:
            p = files.get(r.video_id)
            if p is None:
                warnings.append(f"{r.video_id}: файл не найден в {videos_dir}")
                continue
            try:
                r.file = str(p.relative_to(Path(videos_dir)))
            except ValueError:
                r.file = str(p)
            if with_md5:
                r.md5, r.sha256 = hash_file(p)
        for vid, p in files.items():
            if vid not in labeled:
                warnings.append(f"{vid}: файл {p.name} без разметки")
        warnings += assign_groups(rows)
    else:
        for r in rows:
            r.group = r.video_id
    return rows, warnings


# ----------------------------------------------------------------------------- CSV
def _fmt_t(t: Optional[float]) -> str:
    if t is None:
        return ""
    return str(int(t)) + ".0" if float(t).is_integer() else f"{t:g}"


def write_labels_csv(rows: list[LabelRow], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in rows:
            w.writerow([r.video_id, r.file, r.label, _fmt_t(r.t_sec), r.md5, r.group or r.video_id,
                        ";".join(_fmt_t(t) for t in r.times), int(r.conflict), r.raw, r.sha256])


def _read_csv_dicts(path: str | Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        text = f.read()
    first = text.splitlines()[0] if text else ""
    delim = ";" if first.count(";") > first.count(",") else ","
    return [{(k or "").strip(): (v or "").strip() for k, v in row.items()}
            for row in csv.DictReader(text.splitlines(), delimiter=delim)]


def read_labels_csv(path: str | Path) -> list[LabelRow]:
    out: list[LabelRow] = []
    for d in _read_csv_dicts(path):
        vid = normalize_video_id(d.get("video_id")) or d.get("video_id", "")
        if not vid:
            continue
        lab = (d.get("label") or "").strip().lower()
        if lab in ("violation", "1", "true", "yes", "нарушение", "да"):
            label = LABEL_VIOLATION
        elif lab in ("no_violation", "0", "false", "no", "без нарушения", "нет", "clean"):
            label = LABEL_CLEAN
        else:  # unknown ≠ false: непонятная/пустая метка не становится «без нарушения»
            label = LABEL_UNKNOWN
        times: list[float] = []
        for chunk in re.split(r"[;,|]", d.get("t_all") or "") + [d.get("t_sec") or ""]:
            chunk = chunk.strip()
            if chunk:
                try:
                    t = float(chunk.replace(",", "."))
                except ValueError:
                    ts = parse_timecodes(chunk)
                    t = ts[0] if ts else None
                if t is not None and t not in times:
                    times.append(t)
        out.append(LabelRow(video_id=vid, label=label, times=times, file=d.get("file", ""),
                            md5=d.get("md5", ""), group=normalize_video_id(d.get("group")) or d.get("group") or vid,
                            conflict=(d.get("conflict") or "0") in ("1", "true", "True"), raw=d.get("raw", ""),
                            sha256=d.get("sha256", "")))
    return out


def read_types_csv(path: str | Path) -> dict[str, list[tuple[float, str]]]:
    """data/labels_types.csv: video_id,t_sec,type_id (несколько строк на видео допустимо)."""
    out: dict[str, list[tuple[float, str]]] = {}
    for d in _read_csv_dicts(path):
        vid = normalize_video_id(d.get("video_id")) or d.get("video_id", "")
        tid = (d.get("type_id") or "").strip().upper()
        if not vid or not tid:
            continue
        raw_t = d.get("t_sec") or ""
        try:
            t = float(raw_t.replace(",", "."))
        except ValueError:
            ts = parse_timecodes(raw_t)
            t = ts[0] if ts else float("nan")
        out.setdefault(vid, []).append((t, tid))
    return out


def load_ground_truth(labels_csv: str | Path, types_csv: Optional[str | Path] = None,
                      dedupe: bool = True) -> dict[str, GroundTruth]:
    """GT для метрик. dedupe=True: одна запись на группу дубликатов (ключ — ID группы).

    * метки в группе совпадают → эта метка; различаются → ``conflict`` (исключается из бинарных метрик);
    * таймкоды: если в группе конфликт и у каждого члена ≤1 таймкода → ОДНО событие с альтернативами
      (Х-4/Х-47: [57, 68]); иначе каждое время — отдельное событие. Исходные строки — в ``raw``.
    """
    rows = read_labels_csv(labels_csv)
    types: dict[str, list[tuple[float, str]]] = {}
    if types_csv and Path(types_csv).exists():
        types = read_types_csv(types_csv)
    grouped: dict[str, list[LabelRow]] = {}
    for r in rows:
        key = (r.group or r.video_id) if dedupe else r.video_id
        grouped.setdefault(key, []).append(r)
    out: dict[str, GroundTruth] = {}
    for key, members in grouped.items():
        members.sort(key=lambda r: video_sort_key(r.video_id))
        labels = {m.label for m in members}
        if LABEL_UNKNOWN in labels:
            label = LABEL_UNKNOWN
        else:
            label = labels.pop() if len(labels) == 1 else LABEL_CONFLICT
        rep = next((m for m in members if m.video_id == key), members[0])
        gt = GroundTruth(video_id=key, label=label, group=members[0].group or key, file=rep.file, md5=rep.md5,
                         members=[m.video_id for m in members],
                         conflict=any(m.conflict for m in members) or len({tuple(sorted(m.times)) for m in members}) > 1,
                         raw={m.video_id: m.raw for m in members})
        for m in members:
            gt.times += [t for t in m.times if t not in gt.times]
            gt.types += [x for x in types.get(m.video_id, []) if x not in gt.types]
        gt.times.sort()
        gt.types.sort(key=lambda x: (x[0] if x[0] == x[0] else 1e9))
        if gt.conflict and len(members) > 1 and all(len(m.times) <= 1 for m in members) and gt.times:
            gt.events = [list(gt.times)]
        else:
            gt.events = [[t] for t in gt.times]
        out[key] = gt
    return dict(sorted(out.items(), key=lambda kv: video_sort_key(kv[0])))


# ----------------------------------------------------------------------------- CLI
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="open_labels.xlsx → data/labels.csv")
    ap.add_argument("--xlsx", default="data/open_labels.xlsx")
    ap.add_argument("--videos", default=None, help="папка с роликами (для file/md5/дубликатов)")
    ap.add_argument("--out", default="data/labels.csv")
    ap.add_argument("--no-md5", action="store_true")
    ap.add_argument("--types", default="data/labels_types.csv", help="проверить labels_types.csv, если есть")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rows, warnings = build_labels(args.xlsx, args.videos, with_md5=not args.no_md5)
    write_labels_csv(rows, args.out)
    for w in warnings:
        log.warning(w)
    n_v = sum(r.label == LABEL_VIOLATION for r in rows)
    groups = {r.group for r in rows}
    print(f"{args.out}: {len(rows)} видео ({n_v} с нарушением, {len(rows) - n_v} без), "
          f"{len(groups)} уникальных по md5, конфликтов: {sum(r.conflict for r in rows)}")
    if args.types and Path(args.types).exists():
        t = read_types_csv(args.types)
        unknown = sorted(set(t) - {r.video_id for r in rows})
        print(f"{args.types}: {sum(len(v) for v in t.values())} событий у {len(t)} видео"
              + (f"; нет в labels: {unknown}" if unknown else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
