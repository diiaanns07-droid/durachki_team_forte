from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import pytest

from eval.common import normalize_video_id
from eval.labels import (build_labels, load_ground_truth, main, parse_label, parse_timecodes, read_labels_xlsx,
                         read_types_csv, time_cell_to_seconds, write_labels_csv)


@pytest.mark.parametrize("text,expected", [
    ("22 секунда", [22.0]),
    ("1минута 8 секунда", [68.0]),
    ("1 мин. 8 сек.", [68.0]),
    ("1:08", [68.0]),
    ("на 22-й секунде", [22.0]),
    ("22с", [22.0]),
    ("2 мин 25 сек", [145.0]),
    ("22 и 45 секунда", [22.0, 45.0]),
    ("Х-3 22 секунда", [22.0]),          # номер видео — не таймкод
    (22, [22.0]),
    (dt.time(0, 1, 8), [68.0]),
])
def test_parse_timecodes(text, expected):
    assert parse_timecodes(text) == expected


@pytest.mark.parametrize("text,label,times", [
    ("без нарушения", "no_violation", []),
    ("Без нарушений", "no_violation", []),
    ("нарушений нет", "no_violation", []),
    ("нарушение", "violation", []),
    ("1минута 8 секунда", "violation", [68.0]),
    ("???", None, []),
])
def test_parse_label(text, label, times):
    assert parse_label(text) == (label, times)


def test_excel_time_typed_as_mm_ss():
    # «1:08», набранное в Excel, читается как 01:08:00 — это 68 с, а не 4080 с
    assert time_cell_to_seconds(dt.time(1, 8)) == 68.0
    assert time_cell_to_seconds(dt.time(0, 0, 22)) == 22.0


@pytest.mark.parametrize("name,vid", [
    ("X-1", "X-1"), ("Х-3", "X-3"), ("х-12 .mov", "X-12"), ("X_05.MOV", "X-5"), ("Х—47", "X-47"),
    ("МАХ-3", None), ("выход 12", None), ("", None), (None, None),
])
def test_normalize_video_id(name, vid):
    assert normalize_video_id(name) == vid


def test_read_xlsx(mini_xlsx: Path):
    rows, warnings = read_labels_xlsx(mini_xlsx)
    by = {r.video_id: r for r in rows}
    assert set(by) == {"X-1", "X-3", "X-4", "X-47", "X-5", "X-48", "X-12"}
    assert by["X-1"].label == "violation" and by["X-1"].times == [22.0]
    assert by["X-3"].label == "no_violation" and by["X-3"].times == []
    assert by["X-47"].times == [68.0]
    assert by["X-12"].times == [65.0]


def test_build_labels_md5_groups(mini_xlsx: Path, mini_videos: Path, tmp_path: Path):
    rows, warnings = build_labels(mini_xlsx, mini_videos)
    by = {r.video_id: r for r in rows}
    assert by["X-4"].group == by["X-47"].group == "X-4"
    assert by["X-4"].conflict and by["X-47"].conflict          # 57 vs 68 с
    assert by["X-5"].group == by["X-48"].group == "X-5"
    assert not by["X-5"].conflict
    assert by["X-1"].group == "X-1" and len(by["X-1"].md5) == 32
    assert any("X-99" in w and "без разметки" in w for w in warnings)

    out = tmp_path / "labels.csv"
    write_labels_csv(rows, out)
    with open(out, encoding="utf-8") as f:
        recs = list(csv.DictReader(f))
    assert list(recs[0].keys())[:6] == ["video_id", "file", "label", "t_sec", "md5", "group"]
    r1 = next(r for r in recs if r["video_id"] == "X-1")
    assert r1["label"] == "violation" and float(r1["t_sec"]) == 22.0 and r1["file"] == "X-1.mov"
    r3 = next(r for r in recs if r["video_id"] == "X-3")
    assert r3["t_sec"] == "" and r3["label"] == "no_violation"

    gt = load_ground_truth(out, dedupe=True)
    assert "X-47" not in gt and gt["X-4"].times == [57.0, 68.0] and gt["X-4"].conflict
    assert sorted(gt["X-4"].members) == ["X-4", "X-47"]
    gt_all = load_ground_truth(out, dedupe=False)
    assert "X-47" in gt_all and gt_all["X-47"].times == [68.0]


def test_types_csv(tmp_path: Path, mini_xlsx: Path):
    rows, _ = build_labels(mini_xlsx, None)
    lab = tmp_path / "labels.csv"
    write_labels_csv(rows, lab)
    types = tmp_path / "labels_types.csv"
    types.write_text("video_id,t_sec,type_id\nХ-1,22,v01\nX-1,40,V05\n", encoding="utf-8")
    t = read_types_csv(types)
    assert t["X-1"] == [(22.0, "V01"), (40.0, "V05")]
    gt = load_ground_truth(lab, types)
    assert gt["X-1"].types == [(22.0, "V01"), (40.0, "V05")]


def test_header_less_sheet(tmp_path: Path):
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Х-7", "нарушение", 31])
    ws.append(["X-8", "без нарушения"])
    ws.append(["X-9", "2 минуты 5 секунд"])
    p = tmp_path / "h.xlsx"
    wb.save(p)
    rows, _ = read_labels_xlsx(p)
    by = {r.video_id: r for r in rows}
    assert by["X-7"].times == [31.0] and by["X-7"].label == "violation"
    assert by["X-8"].label == "no_violation"
    assert by["X-9"].times == [125.0]


def test_cli(mini_xlsx: Path, mini_videos: Path, tmp_path: Path, capsys):
    out = tmp_path / "data" / "labels.csv"
    assert main(["--xlsx", str(mini_xlsx), "--videos", str(mini_videos), "--out", str(out)]) == 0
    assert out.exists()
    assert "конфликтов: 2" in capsys.readouterr().out
