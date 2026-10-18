"""split_blinks на синтетике с «миганием» (реальных съёмок нет и не нужно)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from ml.ingest import load_video
from ml.ingest import split_blinks as SB
from ml.ingest.probe import probe
from ml.ingest.synthetic import write_blink_recording

from .conftest import FFMPEG, needs_ffmpeg

FPS = 30.0


def test_detect_blinks_unit_long_dark_and_flicker() -> None:
    t = np.arange(0, 30, 1 / FPS)
    b = np.full_like(t, 120.0) + np.random.default_rng(1).normal(0, 2, t.size)
    for a, z in [(2.0, 3.0), (9.0, 9.9), (14.0, 19.0), (24.0, 24.4), (24.6, 25.0)]:
        m = (t >= a - 1e-9) & (t < z - 1e-9)
        b[m] = 15 + 20 * (t[m] - a) / (z - a)  # автоэкспозиция
    blinks, long_dark = SB.detect_blinks(t, b)
    assert [round(x.off_t, 2) for x in blinks] == [2.0, 9.0, 24.0]
    assert [round(x.on_t, 2) for x in blinks] == [3.0, 9.9, 25.0]
    assert long_dark and all(not (14.0 - 0.1 <= x.off_t <= 19.0) for x in blinks)


def test_journal_parsing_helpers() -> None:
    assert SB.parse_seconds("12,5") == 12.5 and SB.parse_seconds("0:12.5") == 12.5
    assert SB.parse_seconds("1:02:03") == 3723.0 and SB.parse_seconds("7 с") == 7.0
    assert SB.parse_seconds("") is None and SB.parse_seconds("abc") is None
    assert SB._label("нарушение", "") == "violation" and SB._label("clean", "V02") == "no_violation"
    assert SB._label("", "V05") == "violation" and SB._label("??", "") == "unknown"


def _journal(path: Path, rows: list[dict[str, str]], delim: str = ";") -> str:
    fields = ["scene_id", "block", "kind", "code", "title", "priority", "done", "t_onset", "t_end", "layout",
              "light", "cashier", "notes", "shot_at"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, delimiter=delim)
        wr.writeheader()
        for r in rows:
            wr.writerow({f: r.get(f, "") for f in fields})
    return str(path)


JOURNAL = [  # порядок строк != порядок съёмки (shot_at)
    {"scene_id": "S3", "kind": "violation", "code": "V05", "done": "да", "t_onset": "1,0", "layout": "A",
     "light": "on", "shot_at": "2026-10-16 10:20", "title": "секретный заголовок", "notes": "секретная заметка"},
    {"scene_id": "S1", "kind": "violation", "code": "V02", "done": "1", "t_onset": "2.0", "layout": "A",
     "light": "on", "shot_at": "2026-10-16 10:00"},
    {"scene_id": "S9", "kind": "violation", "code": "V01", "done": "", "t_onset": "1", "shot_at": ""},  # не снята
    {"scene_id": "S2", "kind": "норма", "code": "V02", "done": "yes", "t_onset": "", "layout": "B",
     "light": "off", "shot_at": "2026-10-16 10:10"},
]


@pytest.fixture()
def shoot(tmp_path: Path) -> dict[str, object]:
    inc = tmp_path / "incoming"
    (inc / "C1").mkdir(parents=True)
    (inc / "C2").mkdir(parents=True)
    # C1 — портрет, два файла (сцены S1,S2 в первом, S3 — во втором); в S2 свет выключали по сценарию
    e1 = write_blink_recording(str(inc / "C1" / "IMG_0002.mov"), [6.0, 7.0], scenario_dark={1: (2.0, 3.5)})
    e2 = write_blink_recording(str(inc / "C1" / "IMG_0010.mov"), [4.0], preamble_s=1.0, flicker_scene=0)
    # C2 — ландшафт, один файл со всеми тремя сценами
    e3 = write_blink_recording(str(inc / "C2" / "cam2.mov"), [6.0, 7.0, 4.0], size=(432, 240),
                               scenario_dark={1: (2.0, 3.5)})
    journal = _journal(tmp_path / "journal.csv", JOURNAL)
    return {"inc": inc, "journal": journal, "out": tmp_path / "extra" / "clips", "expected": (e1, e2, e3)}


def test_dry_run_report(shoot: dict, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[type-arg]
    code = SB.main(["--videos", str(shoot["inc"]), "--journal", shoot["journal"], "--out", str(shoot["out"]), "--dry-run"])
    printed = capsys.readouterr().out
    assert code == 0
    rep = json.loads((shoot["out"] / "split_report.json").read_text(encoding="utf-8"))
    assert rep["scenes_expected"] == ["S1", "S2", "S3"]  # по shot_at, несъёмная S9 исключена
    for cam in ("C1", "C2"):
        assert rep["cameras"][cam]["status"] == "ok"
        assert [m["scene_id"] for m in rep["cameras"][cam]["matched"]] == ["S1", "S2", "S3"]
    e1, e2, e3 = shoot["expected"]
    segs = rep["cameras"]["C1"]["segments_detail"]
    assert segs[0]["start"] == pytest.approx(e1[0]["on_t"] - 0.5, abs=1 / FPS + 1e-6)
    assert segs[0]["end"] == pytest.approx(e1[1]["off_t"], abs=1 / FPS + 1e-6)
    assert segs[2]["start"] == pytest.approx(e2[0]["on_t"] - 0.5, abs=1 / FPS + 1e-6)  # мерцание не дробит метку
    c1_files = rep["cameras"]["C1"]["files"]
    assert c1_files[0]["long_dark"], "длинное затемнение по сценарию должно попасть в отчёт, но не в метки"
    assert not list(shoot["out"].glob("**/*.mov")) and not (shoot["out"].parent / "labels.csv").exists()
    text = json.dumps(rep, ensure_ascii=False) + printed
    assert "секрет" not in text  # title/notes журнала не утекают в отчёт и stdout


@needs_ffmpeg
def test_cut_clips_and_labels(shoot: dict) -> None:  # type: ignore[type-arg]
    assert SB.main(["--videos", str(shoot["inc"]), "--journal", shoot["journal"], "--out", str(shoot["out"])]) == 0
    labels = shoot["out"].parent / "labels.csv"
    rows = list(csv.DictReader(open(labels, encoding="utf-8")))
    assert list(rows[0].keys()) == SB.LABEL_FIELDS and len(rows) == 6
    by = {r["video_id"]: r for r in rows}
    assert by["S1-C1"]["label"] == "violation" and by["S1-C1"]["type_id"] == "V02" and float(by["S1-C1"]["t_sec"]) == 2.5
    assert by["S2-C2"]["label"] == "no_violation" and by["S2-C2"]["t_sec"] == "" and by["S2-C2"]["type_id"] == ""
    assert by["S3-C2"]["t_sec"] == "1.500" and by["S2-C1"]["light"] == "off" and by["S2-C1"]["layout"] == "B"
    for r in rows:
        clip_path = labels.parent / r["file"]
        info = probe(str(clip_path))
        assert info["codec"] == "h264" and info["fps"] == pytest.approx(30.0, rel=1e-3)
        assert min(info["display_width"], info["display_height"]) == 464
        portrait = r["camera"] == "C1"
        assert (info["display_height"] > info["display_width"]) == portrait
    # клип начинается за 0.5 с до включения света: в начале темно, после 0.5 с — светло
    clip = load_video(str(labels.parent / by["S1-C1"]["file"]))
    q = {round(x["t"], 1): x["brightness"] for x in clip.quality}
    assert q[0.2] < 0.5 * q[1.0]
    e1 = shoot["expected"][0]
    assert clip.duration == pytest.approx(e1[1]["off_t"] - (e1[0]["on_t"] - 0.5), abs=0.1)
    # повторный запуск по одной камере не теряет строки другой (merge по video_id)
    assert SB.main(["--videos", str(shoot["inc"]), "--journal", shoot["journal"], "--out", str(shoot["out"]),
                    "--cameras", "C2"]) == 0
    assert len(list(csv.DictReader(open(labels, encoding="utf-8")))) == 6


def test_count_mismatch_skips_camera(shoot: dict, tmp_path: Path) -> None:  # type: ignore[type-arg]
    extra = JOURNAL + [{"scene_id": "S4", "kind": "clean", "done": "1", "shot_at": "2026-10-16 10:30"}]
    j = _journal(tmp_path / "j4.csv", extra, delim=",")
    rep = SB.split(str(shoot["inc"]), j, str(shoot["out"]), dry_run=True)
    assert all(c["status"] == "count_mismatch" for c in rep["cameras"].values())
    assert rep["labels_rows"] == 0 and any("пропущена" in w for w in rep["warnings"])
    forced = SB.split(str(shoot["inc"]), j, str(shoot["out"]), dry_run=True, force=True)
    assert forced["labels_rows"] == 6


def test_manual_marks_override(shoot: dict, tmp_path: Path) -> None:  # type: ignore[type-arg]
    marks = tmp_path / "marks.csv"
    marks.write_text("camera,file,on_t,off_t\nC2,cam2.mov,3.0,2.0\nC2,cam2.mov,10.0,9.0\nC2,cam2.mov,18.0,17.0\n",
                     encoding="utf-8")
    rep = SB.split(str(shoot["inc"]), shoot["journal"], str(shoot["out"]), dry_run=True, marks=str(marks),
                   cameras=["C2"])
    segs = rep["cameras"]["C2"]["segments_detail"]
    assert [round(s["start"], 2) for s in segs] == [2.5, 9.5, 17.5] and rep["cameras"]["C2"]["files"][0]["manual_marks"]


def test_pyav_engine(shoot: dict) -> None:  # type: ignore[type-arg]
    import av

    if "libx264" not in av.codecs_available and "libopenh264" not in av.codecs_available:
        pytest.skip("в PyAV нет H.264-кодировщика")
    rep = SB.split(str(shoot["inc"]), shoot["journal"], str(shoot["out"]), cameras=["C2"], engine="pyav")
    assert rep["labels_rows"] == 3
    clip = load_video(str(shoot["out"] / "C2" / "S1-C2.mov"))
    assert min(clip.width, clip.height) == 464 and clip.width > clip.height
    assert clip.fps == pytest.approx(30.0, rel=1e-3)
    e3 = shoot["expected"][2]
    assert clip.duration == pytest.approx(e3[1]["off_t"] - (e3[0]["on_t"] - 0.5), abs=0.1)
