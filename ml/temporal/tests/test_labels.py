from __future__ import annotations

from ml.temporal.labels import (VideoLabel, assign_groups, merge_duplicates, normalize_name, parse_label_value,
                                parse_times, read_labels)


def test_normalize_latin_and_cyrillic_x():
    assert normalize_name("X-1.mov") == "X-1"
    assert normalize_name("Х-3.MOV") == "X-3"          # кириллическая Х
    assert normalize_name("х_03") == "X-3"
    assert normalize_name(" X 47 ") == "X-47"
    assert normalize_name("data/raw/Х-4.mov") == "X-4"


def test_parse_times_and_labels():
    assert parse_times("22 секунда") == [22.0]
    assert parse_times("0:22") == [22.0]
    assert parse_times("57 и 68") == [57.0, 68.0]
    assert parse_times("1:02, 1:10") == [62.0, 70.0]
    assert parse_times("") == []
    assert parse_label_value("нарушение") == 1
    assert parse_label_value("без нарушения") == 0
    assert parse_label_value("violation") == 1
    assert parse_label_value("no_violation") == 0
    assert parse_label_value("1") == 1 and parse_label_value("0") == 0


def test_read_labels_russian_semicolon(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("Видео;Нарушение;Таймкод нарушения\nX-1.mov;нарушение;22 секунда\nХ-3.mov;без нарушения;\n"
                 "X-4;да;57\n", encoding="utf-8")
    labs = read_labels(p)
    assert labs["X-1"].label == 1 and labs["X-1"].times == [22.0]
    assert labs["X-3"].label == 0 and labs["X-3"].times == []
    assert labs["X-4"].times == [57.0]


def test_read_labels_no_header_with_type(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text("X-1,1,22,V05\nX-2,0,,\n", encoding="utf-8")
    labs = read_labels(p)
    assert labs["X-1"].type_id == "V05" and labs["X-2"].label == 0


def test_duplicates_keep_original_labels():
    labs = {"X-4": VideoLabel("X-4", 1, [57.0]), "X-47": VideoLabel("X-47", 1, [68.0]),
            "X-5": VideoLabel("X-5", 1, [10.0]), "X-48": VideoLabel("X-48", 0, []),
            "X-1": VideoLabel("X-1", 0, [])}
    groups = assign_groups(labs, md5s={"X-1": "a"}, duplicate_groups=[["Х-4", "X-47"], ["X-5", "Х-48"]])
    assert groups["X-47"] == "X-4" and groups["X-48"] == "X-5"
    m = merge_duplicates(labs)
    assert set(m) == {"X-1", "X-4", "X-5"}
    assert m["X-4"].times == [57.0, 68.0]                      # обе исходные секунды сохранены
    assert [(s["name"], s["times"]) for s in m["X-4"].sources] == [("X-4", [57.0]), ("X-47", [68.0])]
    assert m["X-5"].conflict and m["X-5"].label == 1           # политика positive, конфликт помечен
    assert "X-5" not in merge_duplicates(labs, conflict_policy="exclude")


def test_md5_duplicates_grouped():
    labs = {"X-2": VideoLabel("X-2", 0), "X-9": VideoLabel("X-9", 0), "X-3": VideoLabel("X-3", 1, [5.0])}
    g = assign_groups(labs, md5s={"X-2": "h", "X-9": "h", "X-3": "z"})
    assert g["X-2"] == g["X-9"] == "X-2" and g["X-3"] == "X-3"


def test_time_inside_label_column(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text("Видео;Нарушение\nX-1;22 секунда\nX-2;нет\nX-3;1\n", encoding="utf-8")
    labs = read_labels(p)
    assert labs["X-1"].label == 1 and labs["X-1"].times == [22.0]
    assert labs["X-2"].label == 0 and labs["X-3"].label == 1 and labs["X-3"].times == []


def test_l1_schema_labels_csv(tmp_path):
    """Схема L1 (tools/dataprep/build_labels.py): video_id,file,label,t_sec,md5,group,t_all,conflict,raw."""
    p = tmp_path / "labels.csv"
    p.write_text("video_id,file,label,t_sec,md5,group,t_all,conflict,raw\n"
                 "X-1,X-1.mov,violation,22,aa,X-1,22,0,22 сек | нарушение\n"
                 "X-3,X-3.mov,no_violation,,bb,X-3,,0,без нарушения\n"
                 "X-4,X-4.mov,violation,57,cc,X-4,57,1,57 сек | нарушение\n"
                 "X-47,X-47.mov,violation,68,cc,X-4,68,1,1 мин 8 сек | нарушение\n", encoding="utf-8")
    labs = read_labels(p)
    assert labs["X-1"].label == 1 and labs["X-1"].times == [22.0]       # таймкод из t_all/t_sec, не из raw
    assert labs["X-3"].label == 0 and labs["X-3"].times == []
    assert labs["X-47"].times == [68.0]
    assert all(lab.type_id == "" for lab in labs.values())


def test_type_status_confirmed_vs_pseudo(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text("file,label,t_sec,type_id,type_status\nX-1,violation,10,V05,confirmed\n"
                 "X-2,violation,12,V03,vlm_pseudo\nX-3,violation,9,V06,\n", encoding="utf-8")
    labs = read_labels(p)
    assert labs["X-1"].type_id == "V05" and labs["X-1"].pseudo_type_id == ""
    assert labs["X-2"].type_id == "" and labs["X-2"].pseudo_type_id == "V03"     # VLM — только предварительно
    assert labs["X-3"].type_id == "V06"                                          # без статуса — разметка человека
