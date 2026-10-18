"""Обучаемый вариант (на синтетике, только проверенные метки) и CLI графиков."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from ml.scene import learned
from ml.scene.core import scene_states_with_trace
from ml.scene.io import dump_states
from ml.scene.tests.synth import make_clip, zones_default


def _write_video(path, clip, fps=5.0):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (clip.width, clip.height))
    for f in clip.frames:
        vw.write(f)
    vw.release()


def test_lr_and_mlp_fit_separable():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0, 1, (80, 3)), rng.normal(2.5, 1, (80, 3))])
    y = np.r_[np.zeros(80), np.ones(80)]
    for fit in (learned.fit_lr, learned.fit_mlp):
        m = fit(x, y)
        assert learned.auc(y, learned.predict(m, x)) > 0.95
    assert learned.auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.75)


def test_read_labels_requires_verified(tmp_path):
    from ml.scene.train_signals import read_labels

    p = tmp_path / "labels.csv"
    p.write_text("video,signal,t_start,t_end,value,verified\nХ-3.mov,safe_open,0,5,0,1\nX-3,safe_open,5,10,1,0\nX-3,bogus,0,1,1,1\n", encoding="utf-8")
    lab = read_labels(p)
    assert list(lab) == ["x-3"] and len(lab["x-3"]) == 1
    assert len(read_labels(p, allow_unverified=True)["x-3"]) == 2
    q = tmp_path / "noverify.csv"
    q.write_text("video,signal,t_start,t_end,value\nX-1,safe_open,0,5,0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_labels(q)


def test_train_cli_end_to_end_and_model_is_used(tmp_path, monkeypatch):
    from ml.scene import train_signals

    vids = tmp_path / "videos"
    vids.mkdir()
    rows = ["video,signal,t_start,t_end,value,verified"]
    for k, t_open in enumerate((3.0, 5.0, 7.0)):
        clip = make_clip(10.0, safe_open=lambda t, o=t_open: t >= o)
        _write_video(vids / f"X-{k + 1}.avi", clip)
        rows.append(f"X-{k + 1},safe_open,0,{t_open - 0.2},0,1")
        rows.append(f"X-{k + 1},safe_open,{t_open},9.8,1,1")
    (tmp_path / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    zones = {"default": [{"name": z.name, "polygon": [list(p) for p in z.polygon]} for z in zones_default()]}
    (tmp_path / "zones.json").write_text(json.dumps(zones), encoding="utf-8")
    out = tmp_path / "weights" / "scene_signals.json"
    rc = train_signals.main(["--labels", str(tmp_path / "labels.csv"), "--videos", str(vids), "--out", str(out), "--zones-json", str(tmp_path / "zones.json"), "--no-upstream"])
    assert rc == 0 and out.is_file()
    doc = json.loads(out.read_text(encoding="utf-8"))
    m = doc["signals"]["safe_open"]
    assert m["n_videos"] == 3 and m["cv"]["folds"] >= 2 and m["cv"]["auc"] > 0.9
    # модель подхватывается ядром (и не подхватывается в mock-режиме)
    monkeypatch.setenv("SENTINEL_SCENE_MODEL", str(out))
    monkeypatch.delenv("SENTINEL_MOCK", raising=False)
    st, tr = scene_states_with_trace(make_clip(10.0, safe_open=lambda t: t >= 5.0), [], [], zones_default(), [])
    assert "learned" in st[0].provenance["safe_open"]
    t = np.array([s.t for s in st])
    p = np.array([s.signals["safe_open"] for s in st])
    assert (p[t >= 5.6] >= 0.5).all() and (p[t < 4.6] < 0.5).all()
    monkeypatch.setenv("SENTINEL_MOCK", "1")
    st2, _ = scene_states_with_trace(make_clip(4.0), [], [], zones_default(), [])
    assert "learned" not in st2[0].provenance["safe_open"]


def test_plot_cli_from_cache(tmp_path):
    pytest.importorskip("matplotlib")
    from ml.scene import plot

    st, tr = scene_states_with_trace(make_clip(6.0, safe_open=lambda t: t >= 3.0), [], [], zones_default(), [])
    cache = tmp_path / "cache" / "abc123"
    dump_states(st, cache / "scene_states.json", trace=tr, meta={"gt_sec": 3.0, "video": "synthetic"})
    out = tmp_path / "signals.png"
    assert plot.main(["--cache", str(cache), "--out", str(out)]) == 0
    assert out.is_file() and out.stat().st_size > 10_000
    assert plot._gt_from_meta({"gt_sec": [22, "57"]}) == [22.0, 57.0]
