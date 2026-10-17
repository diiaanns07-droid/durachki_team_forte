"""Офлайн-разрешение кэша весов L1 и явная ошибка при их отсутствии: без сети, без mock, без тихой подмены детектора."""
from __future__ import annotations

import pytest

from ml.detect import DetectorUnavailable, detect, last_stats
from ml.detect.synth import synth_clip
from ml.detect.weights import find_cache_dir, hf_cache_candidates, is_cached, missing_message, pretrained_kwargs

PRIMARY = "PekingU/rtdetr_v2_r18vd"
DFINE = "ustc-community/dfine-medium-obj2coco"


def snapshot(cache, model_id, files=("config.json",)):
    root = cache / ("models--" + model_id.replace("/", "--"))
    (root / "refs").mkdir(parents=True, exist_ok=True)
    (root / "refs" / "main").write_text("abc123")
    snap = root / "snapshots" / "abc123"
    snap.mkdir(parents=True, exist_ok=True)
    for f in files:
        (snap / f).write_text("{}")
    return snap


def test_l1_hub_layout_resolved_offline(tmp_path, monkeypatch, no_network):
    w = tmp_path / "weights"
    monkeypatch.setenv("SENTINEL_WEIGHTS", str(w))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    snapshot(w / "hf" / "hub", PRIMARY)
    assert find_cache_dir(PRIMARY) == w / "hf" / "hub" and is_cached(PRIMARY)
    kw = pretrained_kwargs(PRIMARY)
    assert kw == {"cache_dir": str(w / "hf" / "hub"), "local_files_only": True}
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")           # даже без HF_HUB_OFFLINE инференс не скачивает
    assert pretrained_kwargs(PRIMARY)["local_files_only"] is True


def test_hf_hub_cache_and_hf_home_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_WEIGHTS", str(tmp_path / "w"))
    a, b = tmp_path / "a", tmp_path / "b"
    snapshot(a, PRIMARY)
    monkeypatch.setenv("HF_HUB_CACHE", str(a))
    assert find_cache_dir(PRIMARY) == a
    monkeypatch.delenv("HF_HUB_CACHE")
    snapshot(b / "hub", DFINE)
    monkeypatch.setenv("HF_HOME", str(b))
    assert find_cache_dir(DFINE) == b / "hub"
    assert hf_cache_candidates()[0] == b / "hub"


def test_empty_snapshot_is_not_weights(tmp_path, monkeypatch):
    w = tmp_path / "weights"
    monkeypatch.setenv("SENTINEL_WEIGHTS", str(w))
    snapshot(w / "hf" / "hub", PRIMARY, files=())        # прерванная загрузка: каталог есть, файлов нет
    assert not is_cached(PRIMARY)


def test_missing_weights_explicit_error_lists_search_paths(tmp_path, monkeypatch, no_network):
    w = tmp_path / "weights"
    monkeypatch.setenv("SENTINEL_WEIGHTS", str(w))
    clip, _ = synth_clip(duration_s=0.4)
    with pytest.raises(DetectorUnavailable) as ei:
        detect(clip)
    msg = str(ei.value)
    assert PRIMARY in msg and str(w / "hf" / "hub") in msg and "--download" in msg
    st = last_stats()
    assert st["status"] == "unavailable" and st["mock"] is False and st["backends"] == []
    assert missing_message(PRIMARY) in msg


def test_no_silent_switch_to_other_detector(tmp_path, monkeypatch, no_network):
    """Основной детектор отсутствует, D-FINE скачан, но не выбран явно → явная ошибка, а не тихая подмена."""
    w = tmp_path / "weights"
    monkeypatch.setenv("SENTINEL_WEIGHTS", str(w))
    snapshot(w / "hf" / "hub", DFINE, files=("config.json", "model.safetensors"))
    clip, _ = synth_clip(duration_s=0.4)
    with pytest.raises(DetectorUnavailable):
        detect(clip)
    st = last_stats()
    assert st["backends"] == [] and st["status"] == "unavailable"
    assert not any("резервный" in w for w in st["warnings"])      # подмены не было


def test_cached_but_unloadable_snapshot_is_explicit_error(tmp_path, monkeypatch, no_network):
    """Снапшот есть, но модель не грузится (битые файлы / нет torch) → DetectorUnavailable с причиной, не mock."""
    w = tmp_path / "weights"
    monkeypatch.setenv("SENTINEL_WEIGHTS", str(w))
    snapshot(w / "hf" / "hub", PRIMARY, files=("config.json",))
    clip, _ = synth_clip(duration_s=0.4)
    with pytest.raises(DetectorUnavailable) as ei:
        detect(clip)
    assert "not loaded" in str(ei.value)
    assert last_stats()["mock"] is False
