"""build_labels / build_manifest / check_local — на синтетических входах."""
from __future__ import annotations

import csv
import zipfile

import pytest


def _xlsx(path, rows):
    cells = []
    for i, row in enumerate(rows, start=1):
        cs = "".join(f'<c r="{col}{i}" t="inlineStr"><is><t>{val}</t></is></c>' for col, val in row.items())
        cells.append(f'<row r="{i}">{cs}</row>')
    sheet = f'<worksheet><sheetData>{"".join(cells)}</sheetData></worksheet>'
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", sheet)


def test_norm_id_latin_and_cyrillic(dp):
    bl = dp("build_labels")
    assert bl.norm_id("X-1") == "X-1"
    assert bl.norm_id("Х-47") == "X-47"  # кириллическая Х
    assert bl.norm_id(" х - 05 ") == "X-5"
    with pytest.raises(ValueError):
        bl.norm_id("Y-3")


@pytest.mark.parametrize("raw,want", [("22 секунда", 22), ("1минута 8 секунда", 68), ("2 минуты", 120),
                                      ("", None), (None, None)])
def test_parse_t(dp, raw, want):
    assert dp("build_labels").parse_t(raw) == want


def test_parse_t_rejects_garbage(dp):
    with pytest.raises(ValueError):
        dp("build_labels").parse_t("примерно в середине")


def test_build_rows_groups_and_conflict(dp, tmp_path):
    bl = dp("build_labels")
    x = tmp_path / "l.xlsx"
    _xlsx(x, [{"A": "имя", "B": "время", "C": "метка"},
              {"A": "Х-10", "B": "57 секунда", "C": "нарушение"},
              {"A": "Х-2", "C": "без нарушения"},
              {"A": "X-3", "B": "1минута 8 секунда", "C": "нарушение"},
              {"A": "Х-4", "C": "без нарушения"}])
    md5 = {"X-10.mov": "aaa", "X-3.mov": "aaa", "X-2.mov": "bbb", "X-4.mov": "bbb"}
    rows = bl.build_rows(bl.read_xlsx(x), md5.__getitem__)
    by = {r["video_id"]: r for r in rows}
    assert [r["video_id"] for r in rows] == ["X-2", "X-3", "X-4", "X-10"]
    assert by["X-10"]["group"] == by["X-3"]["group"] == "X-3"
    assert by["X-10"]["conflict"] == by["X-3"]["conflict"] == 1  # один файл, разные таймкоды — не решаем
    assert by["X-2"]["conflict"] == by["X-4"]["conflict"] == 0
    assert by["X-3"]["t_sec"] == 68 and by["X-2"]["t_sec"] == ""


def test_build_rows_rejects_label_time_mismatch(dp, tmp_path):
    bl = dp("build_labels")
    x = tmp_path / "l.xlsx"
    _xlsx(x, [{"A": "h"}, {"A": "Х-1", "B": "5 секунда", "C": "без нарушения"}])
    with pytest.raises(ValueError):
        bl.build_rows(bl.read_xlsx(x), lambda f: "m")


def test_parse_probe_rotation_and_vfr(dp):
    bm = dp("build_manifest")
    j = {"streams": [{"width": 464, "height": 848, "codec_name": "h264", "avg_frame_rate": "21770/727",
                      "r_frame_rate": "30/1", "nb_frames": "2177", "side_data_list": [{"rotation": -90}]}],
         "format": {"duration": "72.700000"}}
    m = bm.parse_probe(j)
    assert m["rotation"] == -90 and m["nb_frames"] == 2177 and m["avg_fps"] == pytest.approx(29.945, abs=1e-3)


def test_mark_duplicates_needs_review_only_on_disagreement(dp):
    bm = dp("build_manifest")
    man = [dict(video_id="X-1", sha256="s1", label="violation", t_sec="57"),
           dict(video_id="X-2", sha256="s1", label="violation", t_sec="68"),
           dict(video_id="X-3", sha256="s2", label="no_violation", t_sec=""),
           dict(video_id="X-4", sha256="s2", label="no_violation", t_sec="")]
    rq = bm.mark_duplicates(man)
    assert [m["video_id"] for m in rq] == ["X-1", "X-2"]
    assert man[0]["dup_of"] == "X-2" and man[2]["needs_review"] == 0


def test_read_lock(dp, tmp_path):
    p = tmp_path / "lock.txt"
    p.write_text("# header\ntorch==2.11.0+cu128\nopencv_python_headless==4.14.0.94\n-e ./x\nfoo @ file:///x\n",
                 encoding="utf-8")
    assert dp("check_local").read_lock(p) == {"torch": "2.11.0+cu128", "opencv-python-headless": "4.14.0.94"}


def _write(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def test_check_labels_flags_vlm_leak(dp, tmp_path, capsys):
    cl = dp("check_local")
    (tmp_path / "data").mkdir()
    _write(tmp_path / "data/labels.csv", cl.LABELS_COLS,
           [dict(video_id="X-1", file="X-1.mov", label="violation", t_sec="5", md5="m", group="X-1", t_all="5",
                 conflict="0", raw="")])
    types_cols = cl.TYPES_REQUIRED
    _write(tmp_path / "data/labels_types.csv", types_cols,
           [dict(video_id="X-1", t_sec="5", type_id="V04", vlm_type_id="V04", source="vlm", confirmed="0",
                 conflict="", notes="")])
    r = cl.Report()
    cl.check_labels(tmp_path, r)
    assert r.n["FAIL"] == 1 and "неподтверждённых с type_id 1" in capsys.readouterr().out
