"""open_labels.xlsx → data/labels.csv (схема eval.labels: video_id,file,label,t_sec,md5,group,t_all,conflict,raw).

Имена: X-1, X-2 — латинская X; остальные — кириллическая Х (U+0425) → единый id «X-<n>» (латиница).
Таймкоды: «22 секунда» → 22, «1минута 8 секунда» → 68. Дубликаты по md5 → общая group (min id),
conflict=1, если внутри группы расходятся метки или таймкоды (решает человек, скрипт ничего не выбирает).

usage: python -I -X utf8 tools/dataprep/build_labels.py data [--out data/labels.csv] [--force]
Существующий labels.csv без --force не перезаписывается.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import zipfile
from pathlib import Path

COLUMNS = ["video_id", "file", "label", "t_sec", "md5", "group", "t_all", "conflict", "raw"]
LABEL_MAP = {"нарушение": "violation", "без нарушения": "no_violation"}


def read_xlsx(path: Path) -> list[dict[str, str]]:
    """Первый лист без openpyxl: {колонка: значение}; inline/shared strings не используются в исходнике."""
    with zipfile.ZipFile(path) as z:
        x = z.read("xl/worksheets/sheet1.xml").decode("utf-8")
    rows = []
    for row in re.findall(r"<row[^>]*>(.*?)</row>", x, re.S):
        cells = {}
        for attrs, inner in re.findall(r"<c ([^>]*?)(?:/>|>(.*?)</c>)", row, re.S):
            col = re.match(r'.*?r="([A-Z]+)\d+"', attrs).group(1)
            v = re.search(r"<v>(.*?)</v>", inner or "")
            cells[col] = v.group(1) if v else "".join(re.findall(r"<t[^>]*>(.*?)</t>", inner or "", re.S))
        rows.append(cells)
    return rows[1:]


def norm_id(name: str) -> str:
    m = re.fullmatch(r"\s*[XХxх]\s*-\s*(\d+)\s*", name)
    if not m:
        raise ValueError(f"неожиданное имя файла: {name!r}")
    return f"X-{int(m.group(1))}"


def parse_t(s: str | None) -> int | None:
    s = (s or "").strip().lower()
    if not s:
        return None
    mins = re.search(r"(\d+)\s*мин", s)
    secs = re.search(r"(\d+)\s*сек", s)
    if not (mins or secs):
        raise ValueError(f"неожиданный таймкод: {s!r}")
    return (int(mins.group(1)) * 60 if mins else 0) + (int(secs.group(1)) if secs else 0)


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_rows(xlsx_rows: list[dict[str, str]], md5_of) -> list[dict]:
    out = []
    for r in xlsx_rows:
        vid = norm_id(r["A"])
        label = LABEL_MAP[r["C"].strip().lower()]
        t = parse_t(r.get("B"))
        if (t is None) != (label == "no_violation"):
            raise ValueError(f"{vid}: метка {label} и таймкод {t!r} не согласованы")
        f = f"{vid}.mov"
        out.append(dict(video_id=vid, file=f, label=label, t_sec="" if t is None else t, md5=md5_of(f),
                        t_all="" if t is None else str(t),
                        raw=" | ".join(x.strip() for x in (r.get("B", ""), r["C"]) if x.strip())))
    assign_groups(out)
    out.sort(key=lambda o: int(o["video_id"].split("-")[1]))
    return out


def assign_groups(rows: list[dict]) -> None:
    """group = min id среди одинаковых md5 (для group k-fold); conflict — как eval.labels.assign_groups."""
    by_md5: dict[str, list[dict]] = {}
    for o in rows:
        by_md5.setdefault(o["md5"], []).append(o)
    for members in by_md5.values():
        group = min((m["video_id"] for m in members), key=lambda v: int(v.split("-")[1]))
        times = {m["t_sec"] for m in members if m["t_sec"] != ""}
        conflict = int(len({m["label"] for m in members}) > 1 or len(times) > 1)
        for m in members:
            m["group"], m["conflict"] = group, conflict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out_path = a.out or a.data / "labels.csv"
    if out_path.exists() and not a.force:
        raise SystemExit(f"{out_path} уже есть — не перезаписываю без --force")
    rows = build_rows(read_xlsx(a.data / "open_labels.xlsx"), lambda f: md5_file(a.data / "raw" / f))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows; violation {sum(o['label'] == 'violation' for o in rows)}; "
          f"md5 groups {len({o['md5'] for o in rows})}; conflict rows {sum(o['conflict'] for o in rows)}")


if __name__ == "__main__":
    main()
