"""Инвалидация кэшей по видео/настройкам/модели и честный статус чекпоинта (синтетика, без весов)."""
from __future__ import annotations

import json
import os
import time

import pytest

from ml.temporal.config import feature_signature, load_config, stable_hash
from ml.temporal.dataset import fit_spec, prepare_samples
from ml.temporal.predictors import BaselineLR, save_checkpoint

from .conftest import make_bank, make_samples


def _train_ckpt(root, cfg, threshold=0.4, extra_meta=None):
    s, b = make_samples(n_pos=3, n_neg=3, T=60), make_bank()
    spec = fit_spec([fs for x in s for fs in x.versions], b, cfg)
    prepare_samples(s, b, spec)
    d = save_checkpoint(root, BaselineLR(cfg).fit(s, cfg), spec, {"threshold": threshold, **(extra_meta or {})})
    (root / "selected.json").write_text(json.dumps({"kind": "baseline"}), encoding="utf-8")
    return d


def _pipeline_sig(paths):
    """Как backend.pipeline.stages._file_sig: имя + размер + mtime_ns."""
    out = []
    for p in paths:
        st = os.stat(p)
        out.append(f"{os.path.basename(p)}:{st.st_size}:{st.st_mtime_ns}")
    return out


def test_model_info_reflects_checkpoint_and_config(tmp_path, monkeypatch):
    import ml.temporal as temporal
    from ml.temporal.config import weights_dir

    cfg = load_config()
    root = weights_dir(cfg)
    i0 = temporal.model_info()
    assert i0["status"] == "no_checkpoint" and i0["checkpoint"] == []
    d = _train_ckpt(root, cfg)
    i1 = temporal.model_info()
    assert i1["status"] == "trained" and str(d / "meta.json") in i1["checkpoint"] and i1["kind"] == "baseline"
    sig1 = _pipeline_sig(i1["checkpoint"])
    time.sleep(0.01)
    _train_ckpt(root, cfg, threshold=0.7)                     # переобучение -> подпись файлов другая
    assert _pipeline_sig(temporal.model_info()["checkpoint"]) != sig1
    over = tmp_path / "over.yaml"                             # внешний конфиг тоже меняет идентичность
    over.write_text("infer:\n  peak_rel: 0.7\n", encoding="utf-8")
    monkeypatch.setenv("SENTINEL_TEMPORAL_CONFIG", str(over))
    assert temporal.model_info()["config_hash"] != i1["config_hash"]


def test_in_process_checkpoint_refreshes_on_file_change():
    from ml.temporal import infer
    from ml.temporal.config import weights_dir

    cfg = load_config()
    d = _train_ckpt(weights_dir(cfg), cfg, threshold=0.4)
    assert infer._checkpoint(cfg)[2]["threshold"] == 0.4
    m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    m["threshold"] = 0.9
    time.sleep(0.01)
    (d / "meta.json").write_text(json.dumps(m), encoding="utf-8")      # selected.json не трогаем
    assert infer._checkpoint(cfg)[2]["threshold"] == 0.9


def test_feature_signature_tracks_settings_and_model_revision(tmp_path, monkeypatch):
    a = load_config({"features": {"max_frames": 1200}})
    b = load_config({"features": {"max_frames": 300}})
    assert feature_signature(a, "siglip2") != feature_signature(b, "siglip2")
    hub = tmp_path / "hub"
    ref = hub / "models--google--siglip2-base-patch16-224" / "refs" / "main"
    ref.parent.mkdir(parents=True)
    ref.write_text("rev-A", encoding="utf-8")
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    s_a = feature_signature(a, "siglip2")
    assert s_a["revision"] == "rev-A"
    ref.write_text("rev-B", encoding="utf-8")                 # другой снапшот весов -> другой ключ кэша
    assert feature_signature(a, "siglip2") != s_a
    assert feature_signature(a, "mock")["revision"] is None


def test_checkpoint_feature_mismatch_is_honest(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    monkeypatch.setenv("SENTINEL_MOCK", "1")
    import ml.temporal as temporal
    from ml.temporal._video import SimpleClip
    from ml.temporal.config import weights_dir

    from .conftest import write_video

    cfg = load_config()
    stale = stable_hash(feature_signature(load_config({"features": {"max_frames": 77}}), "mock"), 16)
    _train_ckpt(weights_dir(cfg), cfg, extra_meta={"feature_signature": stale})
    clip = SimpleClip(str(write_video(tmp_path / "v.avi")), sample_fps=5.0)
    with pytest.raises(temporal.TemporalUnavailable) as ei:
        temporal.score(clip)
    assert ei.value.reason == "checkpoint_mismatch" and temporal.status()["analyzed"] is False


def test_no_duplicate_group_leak_in_outer_and_inner_folds(monkeypatch, cfg):
    """Ни во внешних фолдах, ни во внутренней калибровке группа валидации не попадает в train."""
    from ml.temporal import cv as cvmod

    samples, bank = make_samples(n_pos=5, n_neg=5, T=60), make_bank()
    samples[1].group = samples[0].group                       # дубликаты: одна группа
    samples[6].group = samples[5].group
    seen = []
    real = cvmod.fit_and_score

    def spy(kind, train, val, *a, **k):
        seen.append(({x.group for x in train}, {x.group for x in val}))
        return real(kind, train, val, *a, **k)

    monkeypatch.setattr(cvmod, "fit_and_score", spy)
    cvmod.nested_cv(samples, bank, cfg, "baseline", outer=3, inner=2, seed=0)
    assert len(seen) >= 3 * 3                                  # внешние + внутренние вызовы
    assert all(tr.isdisjoint(va) for tr, va in seen)
