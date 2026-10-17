"""replay CLI: кэш состояний (синтетика) + разметка → таблица и метрики."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.rules.cache_io import load_cache_dir, load_labels, normalize_video_id, parse_label, parse_timecode, save_cache
from ml.rules.replay import main
from ml.rules.synth import RULE_CASES, make_states


def _write_cache(tmp_path: Path) -> Path:
    cache = tmp_path / "cache"
    pos = RULE_CASES["V03_safe_left_open"]["pos"]
    save_cache(cache, "X-1", make_states(60.0, segments=pos["segments"]))
    save_cache(cache, "Х-3", make_states(60.0))  # кириллическая Х
    # плоский формат + jsonl
    flat = [{"t": s.t, **s.signals} for s in make_states(30.0)]
    (cache / "x-9.states.jsonl").write_text("\n".join(json.dumps(r) for r in flat), encoding="utf-8")
    return cache


def test_normalize_and_parsers() -> None:
    assert normalize_video_id("Х-3.MOV") == normalize_video_id("X_3") == "x3"
    assert normalize_video_id("cache/X-47.states.json") == "x47"
    assert parse_timecode("22 секунда") == 22.0
    assert parse_timecode("0:22") == 22.0 and parse_timecode("01:05") == 65.0
    assert parse_timecode("") is None
    assert parse_label("нарушение") is True and parse_label("без нарушения") is False
    assert parse_label("violation") is True and parse_label("no_violation") is False
    assert parse_label("???") is None


def test_load_cache_layouts(tmp_path: Path) -> None:
    cache = _write_cache(tmp_path)
    found = load_cache_dir(cache)
    assert set(found) == {"x1", "x3", "x9"}
    assert found["x9"]["states"][0]["t"] == 0.0


def test_labels_csv_semicolon_cp1251(tmp_path: Path) -> None:
    p = tmp_path / "labels.csv"
    p.write_bytes("Файл;Нарушение;Секунда\nX-1.mov;нарушение;15 секунда\nХ-3.mov;без нарушения;\n".encode("cp1251"))
    lab = load_labels(p)
    assert lab["x1"][0]["violation"] is True and lab["x1"][0]["t"] == 15.0
    assert lab["x3"][0]["violation"] is False and lab["x3"][0]["t"] is None


def test_replay_cli_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache = _write_cache(tmp_path)
    labels = tmp_path / "labels.csv"
    labels.write_text("file,label,second\nX-1.mov,нарушение,15\nХ-3.mov,без нарушения,\nX-9.mov,без нарушения,\n",
                      encoding="utf-8")
    out_json = tmp_path / "res.json"
    out_csv = tmp_path / "res.csv"
    rc = main(["--cache", str(cache), "--labels", str(labels), "--assume-verified", "--json", str(out_json),
               "--csv", str(out_csv), "--events"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "TP=1 FP=0 FN=0 TN=2" in text
    res = json.loads(out_json.read_text(encoding="utf-8"))
    row = next(v for v in res["videos"] if v["id"] == "x1")
    # синтетика: наличные на столе + кассир ушёл при открытом сейфе → V02 и V03 в один кадр
    assert row["pred"] is True and row["type_id"] in {"V02", "V03"} and abs(row["dt"]) <= 0.25 and row["hit"] is True
    assert {e["type_id"] for e in row["events"]} >= {"V02", "V03"}
    assert out_csv.read_text(encoding="utf-8").startswith("name,gt")


def test_replay_verdict_policies_without_verified_norms(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    import ml.rules.catalog as catalog_mod

    monkeypatch.setattr(catalog_mod, "_try_norms_catalog", lambda: None)
    cache = _write_cache(tmp_path)
    labels = tmp_path / "labels.csv"
    labels.write_text("file,label,second\nX-1.mov,нарушение,15\n", encoding="utf-8")
    # по умолчанию — ось обнаружения (поведение по видео), норма отчитывается отдельно
    assert main(["--cache", str(cache), "--labels", str(labels)]) == 0
    assert "TP=1 FP=0 FN=0" in capsys.readouterr().out
    # юридическая политика: без подтверждённой нормы — только наблюдения
    assert main(["--cache", str(cache), "--labels", str(labels), "--verdict", "legal"]) == 0
    assert "TP=0 FP=0 FN=1" in capsys.readouterr().out


def test_replay_sweep_and_overrides(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache = _write_cache(tmp_path)
    labels = tmp_path / "labels.csv"
    labels.write_text("file,label,second\nX-1.mov,нарушение,15\nХ-3.mov,без нарушения,\n", encoding="utf-8")
    rc = main(["--cache", str(cache), "--labels", str(labels), "--assume-verified",
               "--set", "global.nms_window_s=3",
               "--sweep", "rules.V03_safe_left_open.min_duration_s=5,30"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "min_duration_s" in out and out.count("\n") >= 4


def test_rules_table_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--rules-table"]) == 0
    out = capsys.readouterr().out
    assert "| `V01_outsider_in_cashier_area` | V01 | violation | да |" in out
    assert "| `V99_placeholder` | V99 | observation | нет |" in out


def test_replay_empty_cache(tmp_path: Path) -> None:
    assert main(["--cache", str(tmp_path / "nothing")]) == 2


def _a03_cache(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Раскладка A03: cache/<sha>/states.jsonl (+ dets.jsonl, manifest.json) и summary.json батча."""
    cache = tmp_path / "cache"
    pos = RULE_CASES["V03_safe_left_open"]["pos"]
    shas = {"X-1.mov": "a" * 64, "Х-4.mov": "b" * 64, "X-47.mov": "c" * 64, "X-9.mov": "d" * 64}
    for fname, sha in shas.items():
        d = cache / sha
        d.mkdir(parents=True)
        segs = pos["segments"] if fname in ("X-1.mov", "Х-4.mov", "X-47.mov") else []
        with open(d / "states.jsonl", "w", encoding="utf-8") as fh:
            for st in make_states(60.0, segments=segs):
                fh.write(json.dumps({"t": st.t, "signals": st.signals}) + "\n")
        det = {"t": 0.0, "label": "person", "bbox": [0, 0, 1, 1], "score": 1}
        (d / "dets.jsonl").write_text(json.dumps(det) + "\n",
                                      encoding="utf-8")
        (d / "manifest.json").write_text(json.dumps({"sha256": sha, "video": {"duration": 60.0}}), encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps([{"file": f, "name": f.split(".")[0], "sha256": s} for f, s in shas.items()]),
                       encoding="utf-8")
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "video_id,file,label,t_sec,md5,group,t_all,conflict,raw\n"
        "X-1,X-1.mov,violation,15,m1,,15,0,\n"
        "Х-4,Х-4.mov,violation,15,m4,g4,15;27,1,\n"
        "X-47,X-47.mov,violation,27,m4,g4,15;27,1,\n"
        "X-9,X-9.mov,no_violation,,m9,,,0,\n", encoding="utf-8")
    return cache, summary, labels


def test_a03_layout_names_dedupe_and_conflict(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache, summary, labels = _a03_cache(tmp_path)
    found = load_cache_dir(cache)
    assert len(found) == 4  # dets.jsonl/manifest.json не принимаются за ролики
    out_json = tmp_path / "res.json"
    assert main(["--cache", str(cache), "--labels", str(labels), "--names", str(summary), "--json", str(out_json),
                 "--trace"]) == 0
    text = capsys.readouterr().out
    assert "TP=2 FP=0 FN=0 TN=1" in text          # Х-4 ≡ X-47 — один ролик
    assert "Правила (профиль baseline)" in text and "V09_personal_items_on_desk" in text  # enabled/disabled + причина
    assert "конфликт таймкода: 1" in text and "predicate" in text
    res = json.loads(out_json.read_text(encoding="utf-8"))
    assert res["summary"]["n_labeled"] == 3 and res["summary"]["joint"] is not None
    assert res["summary"]["time_hit_rate"]["1.0"] == 1.0   # конфликтная группа не входит в метрику времени
    assert res["summary"]["vram_mb"] == 0


def test_ablation_and_propose(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache, summary, labels = _a03_cache(tmp_path)
    out_json = tmp_path / "abl.json"
    assert main(["--cache", str(cache), "--labels", str(labels), "--names", str(summary),
                 "--hypotheses", "all", "--json", str(out_json)]) == 0
    rows = json.loads(out_json.read_text(encoding="utf-8"))
    names = [r["variant"] for r in rows]
    assert names[0] == "baseline" and "profile:phase2" in names and "+H1_left_after_departure" in names
    assert all("joint" in r and "d_fp" in r for r in rows)
    capsys.readouterr()
    assert main(["--cache", str(cache), "--labels", str(labels), "--names", str(summary), "--propose"]) == 0
    out = capsys.readouterr().out
    assert "joint" in out or "не найдено" in out


def test_rule_fires_count_only_verdict_events_as_fp_sources(tmp_path: Path) -> None:
    """insufficient_context на чистом видео — не «срабатывание на чистых» (иначе ложный источник FP)."""
    from ml.rules.replay import run_replay
    from ml.rules.config import load_rules_config

    cache = {"x1": {"name": "X-1", "states": [{"t": s.t, "signals": s.signals} for s in make_states(
        60.0, segments=[(20.0, 35.0, {"cashier_present": 0.0, "cashier_in_workplace": 0.0, "cash_visible": None})])],
        "tracks": [], "zones": []}}
    labels = {"x1": [{"name": "X-1", "violation": False, "t": None}]}
    res = run_replay(cache, labels, load_rules_config(), {"types": {}})
    fires = res["rule_fires"]["V02_cash_left_unattended"]
    assert fires["clean"] == 0 and fires["insufficient"] >= 1
    assert res["summary"]["fp"] == 0
