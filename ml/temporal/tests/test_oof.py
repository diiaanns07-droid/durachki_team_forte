"""Cross-fit чекпоинты для сравнения temporal off/on: без утечки групп, порог — только на train фолда."""
from __future__ import annotations

import csv
import json

import pytest

from .conftest import write_video

cv2 = pytest.importorskip("cv2")


@pytest.fixture
def mock_dataset(tmp_path, monkeypatch):
    """8 синтетических роликов (X/Х) + точный md5-дубликат X-9≡X-6, mock-признаки (только тест)."""
    monkeypatch.setenv("SENTINEL_MOCK", "1")
    vids = tmp_path / "raw"
    vids.mkdir()
    rows = ["file;label;second"]
    for i in range(1, 9):
        pos = i <= 4
        name = f"Х-{i}.avi" if i % 2 else f"X-{i}.avi"
        write_video(vids / name, seconds=8.0, event=(3.0, 5.0) if pos else None, shade=70 + 7 * i)
        rows.append(f"{name};{'нарушение' if pos else 'без нарушения'};{4 if pos else ''}")
    (vids / "X-9.avi").write_bytes((vids / "X-6.avi").read_bytes())
    rows.append("X-9.avi;без нарушения;")
    (tmp_path / "labels.csv").write_text("\n".join(rows), encoding="utf-8")
    from ml.temporal import extract

    assert extract.main(["--videos", str(vids), "--labels", str(tmp_path / "labels.csv"), "--n-aug", "1"]) == 0
    return vids


def _write_splits(path, folds: dict[str, int]):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "group", "split_group", "split", "fold", "label"])
        w.writeheader()
        for vid, k in folds.items():
            g = "X-6" if vid in ("X-6", "X-9") else vid
            w.writerow({"video_id": vid, "group": g, "split_group": g, "split": "holdout" if k == -1 else "train",
                        "fold": k, "label": ""})


def test_oof_folds_never_train_on_their_groups(mock_dataset, tmp_path):
    from ml.temporal import oof
    from ml.temporal.config import load_config, weights_dir

    splits = tmp_path / "splits.csv"
    _write_splits(splits, {"X-1": 0, "X-5": 0, "X-2": 1, "X-6": 1, "X-9": 1, "X-3": 2, "X-7": 2,
                           "X-4": -1, "X-8": -1})
    assert oof.main(["--splits", str(splits), "--kind", "baseline"]) == 0
    root = weights_dir(load_config()) / "oof"
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    assert "не независимый holdout" in idx["evaluation_status"]
    hold = {"X-4", "X-8"}
    for name, info in idx["folds"].items():
        assert info["trained"], name
        meta = json.loads((root / name / "baseline" / "meta.json").read_text(encoding="utf-8"))
        assert set(meta["train_groups"]).isdisjoint(info["val_groups"])       # своя группа не в обучении
        assert "inner grouped CV" in meta["threshold_source"]                   # порог — только на train фолда
        if name != "fold_holdout":
            assert set(meta["train_groups"]).isdisjoint(hold)                  # holdout не в обучении фолдов
    assert set(idx["folds"]["fold_holdout"]["train_groups"]) == {"X-1", "X-2", "X-3", "X-5", "X-6", "X-7"}
    by_name = {v["name"]: v["dir"] for v in idx["videos"].values()}
    assert by_name.get("X-6", by_name.get("X-9")) == "fold_1"                  # md5-дубликат — один фолд


def test_oof_inconsistent_duplicate_folds_raise(mock_dataset, tmp_path):
    from ml.temporal import oof

    splits = tmp_path / "bad.csv"
    _write_splits(splits, {"X-1": 0, "X-5": 0, "X-2": 1, "X-6": 1, "X-9": 2, "X-3": 2, "X-7": 2,
                           "X-4": 0, "X-8": 1})
    with pytest.raises(ValueError, match="утечка дубликатов"):
        oof.main(["--splits", str(splits)])


def test_oof_inference_uses_fold_model_and_is_honest_for_unknown(mock_dataset, tmp_path, monkeypatch):
    import ml.temporal as temporal
    from ml.temporal import oof
    from ml.temporal._video import SimpleClip

    assert oof.main(["--folds", "3", "--kind", "baseline"]) == 0
    monkeypatch.setenv("SENTINEL_TEMPORAL_OOF", "1")
    assert temporal.model_info()["mode"] == "oof" and temporal.model_info()["checkpoint"]
    idx_dir = {}
    from ml.temporal.config import load_config, weights_dir

    idx = json.loads((weights_dir(load_config()) / "oof" / "index.json").read_text(encoding="utf-8"))
    for v in idx["videos"].values():
        idx_dir[v["name"]] = v["dir"]
    temporal.score(SimpleClip(str(mock_dataset / "X-2.avi"), sample_fps=5.0))
    assert temporal.status()["checkpoint_dir"].endswith(idx_dir["X-2"])
    new = write_video(tmp_path / "new.avi", seconds=6.0, shade=33)              # не участвовало в cross-fit
    with pytest.raises(temporal.TemporalUnavailable) as ei:
        temporal.score(SimpleClip(str(new), sample_fps=5.0))
    assert ei.value.reason == "oof_unavailable"


def test_offon_dry_run_prints_single_ablation(capsys):
    from ml.temporal import offon

    assert offon.main(["--dry-run", "--splits", "data/splits.csv"]) == 0
    out = capsys.readouterr().out
    assert "-m eval.ablate" in out and "temporal_off.yaml" in out and "SENTINEL_TEMPORAL_OOF=1" in out
    assert "-m ml.temporal.oof" in out
