"""Smoke замера L2 на синтетических роликах (демо): команда работает end-to-end без GPU и апстрима."""
from __future__ import annotations

import json

import cv2

from ml.scene import l2_bench
from ml.scene.tests.synth import make_clip


def test_l2_bench_smoke(tmp_path):
    vids = tmp_path / "videos"
    vids.mkdir()
    for k, opener in ((1, lambda t: t >= 3.0), (2, lambda t: False)):
        clip = make_clip(5.0, safe_open=opener)
        vw = cv2.VideoWriter(str(vids / f"X-{k}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (clip.width, clip.height))
        for f in clip.frames:
            vw.write(f)
        vw.release()
    labels = tmp_path / "labels.csv"
    labels.write_text("video_id,file,label,t_sec,md5,group,t_all,conflict,raw\n"
                      "X-1,X-1.avi,violation,3,a,X-1,3,0,x\nX-2,X-2.avi,no_violation,,b,X-2,,0,x\n", encoding="utf-8")
    out = tmp_path / "bench"
    assert l2_bench.main(["--videos", str(vids), "--labels", str(labels), "--out", str(out), "--variants", "baseline,H1,H3"]) == 0
    doc = json.loads((out / "bench.json").read_text(encoding="utf-8"))
    s = doc["summary"]
    assert s["exploratory"] is True and set(s["variants"]) == {"baseline", "H1", "H3"}
    assert s["variants"]["baseline"]["n"] == 2 and s["variants"]["baseline"]["latency_s"]["max"] > 0
    assert (out / "baseline" / "x-1" / "scene_states.json").is_file()
    assert (out / "_upstream" / "x-1.json").is_file()
    assert "baseline" in (out / "bench.md").read_text(encoding="utf-8")
    a09 = json.loads((out / "baseline" / "a09" / "X-1.states.json").read_text(encoding="utf-8"))
    assert set(a09) == {"states", "tracks", "zones"} and "known" in a09["states"][0] and "signals" in a09["states"][0]
