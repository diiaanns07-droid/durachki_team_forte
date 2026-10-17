"""Реальный путь загрузки (transformers) без сети: крошечная случайная RT-DETRv2 в SENTINEL_WEIGHTS/detect/."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.slow

COCO = {0: "person", 1: "bicycle", 2: "car", 39: "bottle", 41: "cup", 63: "laptop", 67: "cell phone", 73: "book"}


def _tiny_rtdetr(dst):
    from transformers import RTDetrImageProcessor, RTDetrResNetConfig, RTDetrV2Config, RTDetrV2ForObjectDetection

    id2label = {i: COCO.get(i, f"cls{i}") for i in range(80)}
    bb = RTDetrResNetConfig(embedding_size=16, hidden_sizes=[16, 32, 64, 128], depths=[1, 1, 1, 1],
                            layer_type="basic", out_features=["stage2", "stage3", "stage4"])
    cfg = RTDetrV2Config(backbone_config=bb, encoder_hidden_dim=32, encoder_in_channels=[32, 64, 128], d_model=32,
                         encoder_ffn_dim=64, decoder_ffn_dim=64, num_queries=30, decoder_layers=2, encoder_layers=1,
                         num_labels=80, decoder_n_points=2, num_denoising=0, decoder_in_channels=[32, 32, 32],
                         id2label=id2label, label2id={v: k for k, v in id2label.items()})
    RTDetrV2ForObjectDetection(cfg).save_pretrained(dst)
    RTDetrImageProcessor(size={"height": 320, "width": 320}).save_pretrained(dst)


def _hub_snapshot(cache: "Path", model_id: str) -> "Path":
    """Раскладка HF hub-кэша, как у L1: <cache>/models--org--name/{refs/main, snapshots/<rev>/...}."""
    root = cache / ("models--" + model_id.replace("/", "--"))
    (root / "refs").mkdir(parents=True)
    (root / "refs" / "main").write_text("0123456789abcdef")
    snap = root / "snapshots" / "0123456789abcdef"
    snap.mkdir(parents=True)
    return snap


@pytest.mark.parametrize("layout", ["hf_hub", "save_pretrained_dir"])
def test_detect_with_local_tiny_rtdetr(tmp_path, monkeypatch, no_network, layout):
    from ml.detect import available, detect, last_stats, warmup
    from contracts.sentinel_contracts import LABELS
    from ml.detect.synth import synth_clip
    from ml.detect.weights import find_cache_dir

    wdir = tmp_path / "w"
    if layout == "hf_hub":   # weights/hf/hub — как скачал L1 (deploy/weights.py)
        _tiny_rtdetr(_hub_snapshot(wdir / "hf" / "hub", "PekingU/rtdetr_v2_r18vd"))
    else:
        _tiny_rtdetr(wdir / "detect" / "rtdetr_v2_r18vd")
    monkeypatch.setenv("SENTINEL_WEIGHTS", str(wdir))
    if layout == "hf_hub":
        assert find_cache_dir("PekingU/rtdetr_v2_r18vd") == wdir / "hf" / "hub"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert available()["ready"] is True
    assert warmup(config={"runtime": {"device": "cpu"}})["status"] == "ok"
    clip, _ = synth_clip(duration_s=2.0, sample_fps=5.0)
    cfg = {"runtime": {"device": "cpu", "batch_size": 4},
           "closed_set": {"min_score": 0.0, "score_thr": {k: 0.0 for k in ["default", *LABELS]},
                          "max_dets_per_frame": 5}}
    dets = detect(clip, config=cfg)
    st = last_stats()
    assert st["status"] == "ok", st
    assert st["backends"] == ["rtdetr_v2_r18vd"]
    assert len({d.t for d in dets}) == len(clip.times)
    for d in dets:
        assert d.label in LABELS
        assert d.source.startswith("rtdetr_v2_r18vd:")
        x1, y1, x2, y2 = d.bbox
        assert 0 <= x1 <= x2 <= 1 and 0 <= y1 <= y2 <= 1
    # extras (флип+тайлы) на реальном бэкенде тоже отрабатывают
    detect(clip, config={**cfg, "tta": {"hflip": True, "hflip_on": "all"}, "tiling": {"enabled": True, "on": "all"}})
    st = last_stats()
    assert st["n_flip"] == len(clip.times) and st["n_tiles"] == 2 * len(clip.times)


def test_explicit_fallback_detector_is_flagged_degraded(tmp_path, monkeypatch, no_network):
    """Резервный детектор работает только если задан явно; тогда статус degraded и предупреждение, не «ok»."""
    from ml.detect import DetectorUnavailable, detect, last_stats
    from ml.detect.synth import synth_clip

    dfine = "ustc-community/dfine-medium-obj2coco"
    wdir = tmp_path / "w"
    _tiny_rtdetr(_hub_snapshot(wdir / "hf" / "hub", dfine))
    monkeypatch.setenv("SENTINEL_WEIGHTS", str(wdir))
    clip, _ = synth_clip(duration_s=0.4)
    base = {"runtime": {"device": "cpu"}}
    with pytest.raises(DetectorUnavailable):                      # по умолчанию резерв не подставляется
        detect(clip, config=base)
    detect(clip, config={**base, "closed_set": {"hf_fallback_ids": [dfine]}})
    st = last_stats()
    assert st["backends"] == ["dfine-medium-obj2coco"] and st["status"] == "degraded"
    assert any("резервный" in w for w in st["warnings"])
