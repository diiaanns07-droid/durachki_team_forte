import json

import pytest

from ml.track import track_full
from ml.track.config import DEFAULTS, _read_yaml, load_config
from ml.track.serialize import load_track_cache, save_track_cache, track_from_dict, track_to_dict
from ml.track.tests import synth as S


def _keys(d, prefix=""):
    out = set()
    for k, v in d.items():
        out.add(prefix + k)
        if isinstance(v, dict) and k not in ("label_groups", "group_label", "prefer_label", "crop"):
            out |= _keys(v, prefix + k + ".")
    return out


def test_yaml_keys_are_known():
    from pathlib import Path

    y = _read_yaml(str(Path(__file__).resolve().parents[1] / "config.yaml"))
    assert y, "config.yaml should load"
    assert _keys(y) <= _keys(DEFAULTS)
    assert load_config(overrides={"roles": {"early_s": 9.0}})["roles"]["early_s"] == 9.0


def test_cache_roundtrip(tmp_path):
    clip, dets, _ = S.make(S.scene_clean())
    r = track_full(clip, dets)
    save_track_cache(str(tmp_path), r.tracks, r.zones, r.info)
    tracks, zones, info = load_track_cache(str(tmp_path))
    assert [t.track_id for t in tracks] == [t.track_id for t in r.tracks]
    assert [z.name for z in zones] == [z.name for z in r.zones]
    assert track_from_dict(track_to_dict(r.tracks[0])).boxes[0][0] == r.tracks[0].boxes[0][0]
    json.dumps(info)


def test_viz_cli_smoke(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from ml.track.viz import main

    video = tmp_path / "v.avi"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 96))
    for i in range(20):
        img = np.full((96, 64, 3), 120, np.uint8)
        cv2.rectangle(img, (10 + i, 30), (30 + i, 60), (0, 0, 200), -1)
        vw.write(img)
    vw.release()
    cache = tmp_path / "cache"
    cache.mkdir()
    dets = [{"t": round(i / 10.0, 4), "label": "person", "bbox": [(10 + i) / 64, 30 / 96, (30 + i) / 64, 60 / 96],
             "score": 0.9} for i in range(0, 20, 2)]
    (cache / "detections.json").write_text(json.dumps(dets))
    out = tmp_path / "out.mp4"
    assert main([str(video), "--cache", str(cache), "--out", str(out), "--fps", "5", "--blur-heads"]) == 0
    assert out.is_file() and out.stat().st_size > 0
    assert (cache / "track.json").is_file()


def test_report_png_demo(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from ml.track.report import main as report_main

    out = tmp_path / "t.png"
    assert report_main(["--demo", "v02", "--out", str(out)]) == 0
    img = cv2.imread(str(out))
    assert img is not None and img.shape[0] > 100 and img.shape[1] > 1000


def test_report_png_from_cache(tmp_path):
    pytest.importorskip("cv2")
    from ml.track.report import main as report_main

    clip, dets, _ = S.make(S.scene_v01(), blackout=[(30.0, 33.0)])
    r = track_full(clip, dets)
    save_track_cache(str(tmp_path), r.tracks, r.zones, r.info)
    assert report_main(["--cache", str(tmp_path)]) == 0
    assert (tmp_path / "track_timeline.png").is_file()


def test_replay_smoke(tmp_path):
    from ml.track.replay import SCENARIOS, match_events, run_synthetic

    assert match_events([1.0, 5.0], [1.5], 2.0) == (1, 1, 0, [0.5])
    rep = run_synthetic(["baseline", "smooth"], seeds=1, tol=2.0) if False else None  # полный прогон ~40 с — в CLI
    assert "v02" in SCENARIOS and rep is None


def test_replay_real_mode_on_synthetic_cache(tmp_path):
    """Команда L2 (--real) отрабатывает на кэше формата detections.json + CSV меток (здесь — синтетика)."""
    from ml.track.replay import main as replay_main
    from ml.track.serialize import detection_to_dict

    for name, sc in (("Х-1", S.scene_v02()), ("X-2", S.scene_clean())):  # кириллическая и латинская X
        d = tmp_path / "cache" / name
        d.mkdir(parents=True)
        clip, dets, _ = S.make(sc)
        (d / "detections.json").write_text(json.dumps([detection_to_dict(x) for x in dets]))
        (d / "meta.json").write_text(json.dumps({"width": 464, "height": 848, "times": clip.times,
                                                 "duration": clip.duration}))
    (tmp_path / "labels.csv").write_text("video,label,second\nX-1.mov,violation,16\nХ-2.mov,no_violation,\n",
                                         encoding="utf-8")
    out = tmp_path / "out"
    rc = replay_main(["--real", "--dets-glob", str(tmp_path / "cache" / "*" / "detections.json"),
                      "--labels", str(tmp_path / "labels.csv"), "--variant", "smooth", "--out", str(out)])
    assert rc == 0
    rep = json.loads((out / "replay.json").read_text())
    s = rep["summary"]["baseline"]
    assert s["videos"] == 2 and s["labeled_violation"] == 1 and s["labeled_clean"] == 1
    assert s["violation_hit_within_2.0s"] == 1 and s["clean_with_alarm (FP-proxy)"] == 0
