from __future__ import annotations

import json

import numpy as np
import pytest

from ml.temporal._video import SimpleClip
from ml.temporal.config import load_config
from ml.temporal.features import MockBackend, extract_clip

from .conftest import write_video

cv2 = pytest.importorskip("cv2")


def test_score_without_checkpoint_is_honest(tmp_path, monkeypatch):
    """Без обученной модели: не случайный score и не «пусто = чисто», а явный статус «не проанализировано»."""
    monkeypatch.setenv("SENTINEL_MOCK", "1")
    import ml.temporal as temporal

    clip = SimpleClip(str(write_video(tmp_path / "v.avi")), sample_fps=5.0)
    with pytest.raises(temporal.TemporalUnavailable) as ei:
        temporal.score(clip)
    assert ei.value.reason == "no_checkpoint" and "не означает «нарушений нет»" in str(ei.value)
    st = temporal.status()
    assert st["status"] == "no_checkpoint" and st["analyzed"] is False and st["coverage"]["complete"] is False
    assert temporal.model_info()["status"] == "no_checkpoint"
    over = tmp_path / "over.yaml"                           # старое поведение — только явно
    over.write_text("infer:\n  on_unavailable: empty\n", encoding="utf-8")
    monkeypatch.setenv("SENTINEL_TEMPORAL_CONFIG", str(over))
    assert temporal.score(clip) == ([], []) and temporal.status()["analyzed"] is False


def test_extract_clip_mock_shapes_and_focus(tmp_path):
    cfg = load_config()
    clip = SimpleClip(str(write_video(tmp_path / "v.avi", seconds=6.0)), sample_fps=4.0)
    fs = extract_clip(clip, MockBackend(32), cfg, fps=4.0, views=["full", "focus"])
    T = len(fs)
    assert T >= 20 and fs.emb.shape == (T, 2, 32) and fs.dyn.shape == (T, 4) and fs.boxes.shape == (T, 4)
    assert np.allclose(np.linalg.norm(fs.emb, axis=-1), 1.0, atol=2e-3)   # float16-точность кэша
    assert np.all(fs.boxes[:, 0] < fs.boxes[:, 2]) and np.all(fs.boxes[:, 1] < fs.boxes[:, 3])
    assert np.all((fs.boxes >= 0) & (fs.boxes <= 1))
    assert fs.meta["focus_mode"] == "motion"
    ev = (fs.times >= 3.2) & (fs.times <= 4.8)
    assert fs.dyn[ev, 0].mean() > fs.dyn[fs.times < 2.5, 0].mean()   # движение видно в окне события
    fs.save(tmp_path / "f.npz")
    from ml.temporal.features import FeatureSet

    fs2 = FeatureSet.load(tmp_path / "f.npz")
    assert fs2.views == fs.views and np.allclose(fs2.emb, fs.emb, atol=1e-3)


def test_extract_with_zone_and_deadline(tmp_path):
    from contracts.sentinel_contracts import Zone

    cfg = load_config()
    clip = SimpleClip(str(write_video(tmp_path / "v.avi", seconds=6.0)), sample_fps=4.0)
    zone = [Zone("cashier_area", [(0.1, 0.5), (0.9, 0.5), (0.9, 0.95), (0.1, 0.95)])]
    fs = extract_clip(clip, MockBackend(16), cfg, zones=zone, fps=4.0, views=["full", "focus"])
    assert fs.meta["focus_mode"] == "zone"
    fs_cut = extract_clip(clip, MockBackend(16), cfg, fps=4.0, views=["full"], deadline=0.0)
    assert "truncated_at" in fs_cut.meta and len(fs_cut) >= 1


def test_end_to_end_extract_cv_train_score(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_MOCK", "1")
    vids = tmp_path / "raw"
    vids.mkdir()
    rows = ["file;label;second"]
    for i in range(1, 9):
        pos = i <= 4
        name = f"Х-{i}.avi" if i % 2 else f"X-{i}.avi"          # кириллица и латиница вперемешку
        write_video(vids / name, seconds=8.0, event=(3.0, 5.0) if pos else None, shade=70 + 7 * i)
        rows.append(f"{name};{'нарушение' if pos else 'без нарушения'};{4 if pos else ''}")
    (vids / "X-9.avi").write_bytes((vids / "X-6.avi").read_bytes())   # точный дубликат по md5
    rows.append("Х-9.avi;нарушение;5")
    (tmp_path / "labels.csv").write_text("\n".join(rows), encoding="utf-8")

    from ml.temporal import cv as cvmod
    from ml.temporal import extract, train

    assert extract.main(["--videos", str(vids), "--labels", str(tmp_path / "labels.csv"), "--n-aug", "1"]) == 0
    cfg = load_config()
    from ml.temporal.config import cache_dir

    idx = json.loads((cache_dir(cfg) / "index.json").read_text(encoding="utf-8"))
    assert idx["groups"]["X-6"] == ["X-6", "X-9"] and idx["backend"] == "mock"
    out = tmp_path / "cv.json"
    assert cvmod.main(["--kind", "baseline", "--outer", "4", "--inner", "2", "--out", str(out)]) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    names = [v["name"] for v in rep["baseline"]["per_video"]]
    assert "X-9" not in names and "X-6" in names                       # дубликат свёрнут в группу
    x6 = next(v for v in rep["baseline"]["per_video"] if v["name"] == "X-6")
    # исходные метки обоих файлов сохранены, расхождение помечено (политика positive)
    assert x6["conflict"] and x6["label"] == 1
    assert {(s["name"], s["label"]) for s in x6["sources"]} == {("X-6", 0), ("X-9", 1)}
    from ml.temporal import bench, check_split

    bout = tmp_path / "bench.json"                        # та же команда, что запустит L2 (здесь — на mock)
    assert bench.main(["--experiments", "E0,E1", "--folds", "3", "--repeats", "1", "--inner", "2",
                       "--out", str(bout)]) == 0
    brep = json.loads(bout.read_text(encoding="utf-8"))
    assert set(brep["experiments"]) == {"E0", "E1"} and brep["n_groups"] == 8

    assert check_split.main(["--folds", "loo", "3", "--holdout", "0.25"]) == 0
    assert train.main(["--kind", "baseline", "--cv-report", str(out)]) == 0
    from ml.temporal.config import weights_dir

    tmeta = json.loads((weights_dir(cfg) / "baseline" / "meta.json").read_text(encoding="utf-8"))
    assert tmeta["reproducibility"]["feature_version"] >= 3 and tmeta["null"] and tmeta["pooling"]["pooling"]
    assert "inner grouped CV" in tmeta["threshold_source"] and tmeta["type_head_source"] is None
    from ml.temporal import predict

    assert predict.main(["--video", str(vids / "X-2.avi"), "--json", str(tmp_path / "pred.json")]) == 0
    assert json.loads((tmp_path / "pred.json").read_text(encoding="utf-8"))["status"]["status"] == "ok"

    import ml.temporal as temporal

    assert callable(temporal.score)                 # подмодуль predict не затеняет функцию контракта
    clip = SimpleClip(str(vids / "Х-1.avi"), sample_fps=5.0)
    risk, events = temporal.score(clip)
    st = temporal.status()
    assert st["status"] == "ok" and st["kind"] == "baseline"
    assert st["coverage"]["complete"] and st["coverage"]["fraction"] >= 0.98
    ts = [t for t, _ in risk]
    assert np.allclose(np.diff(ts), 0.5) and all(0.0 <= r <= 1.0 for _, r in risk)
    cands = temporal.propose(clip)                          # с чекпоинтом: кандидаты из риска + изменений
    assert 3 <= len(cands) <= 8 and any(c.extra["candidate_source"] == "risk" for c in cands)
    renamed = tmp_path / "renamed_copy.avi"                  # имя файла не признак: тот же контент -> тот же риск
    renamed.write_bytes((vids / "Х-1.avi").read_bytes())
    risk2, _ = temporal.score(SimpleClip(str(renamed), sample_fps=5.0))
    assert risk2 == risk
    for e in events:
        assert e.extra["role"] == "risk_support" and e.extra["standalone_evidence"] is False
        assert e.source == "temporal" and e.t_start <= e.t_peak <= e.t_end
        assert e.type_id == "V99" and e.extra["type_source"] == "none"
        assert e.extra["zero_shot_hint"]["trained"] is False
        assert 0.0 <= e.confidence <= 1.0 and e.facts


def test_simple_clip_uses_pts_and_monotonic_times(tmp_path):
    clip = SimpleClip(str(write_video(tmp_path / "v.avi", seconds=4.0, fps=10)), sample_fps=5.0)
    assert clip.time_source in ("pts", "index/fps")
    assert all(b > a for a, b in zip(clip.times, clip.times[1:]))
    assert abs(clip.times[1] - clip.times[0] - 0.2) < 0.05 and 3.5 <= clip.duration <= 4.2
