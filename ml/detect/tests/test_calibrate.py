"""Калибровка порогов: только на человеческой разметке; выбор порога по целевой точности; holdout по видео."""
from __future__ import annotations

import json

import pytest
import yaml

from ml.detect.calibrate import choose_threshold, main
from ml.detect.finetune import UnverifiedLabelsError, load_verified
from ml.detect.synth import SynthObject, synth_clip


def test_choose_threshold_max_recall_at_target_precision():
    s = [0.9, 0.8, 0.7, 0.6, 0.5]
    t = [True, True, False, True, False]
    assert choose_threshold(s, t, 4, 0.9) == 0.8
    assert choose_threshold(s, t, 4, 0.7) == 0.6
    assert choose_threshold(s, t, 4, 1.01) is None
    assert choose_threshold([], [], 3, 0.9) is None


def _dataset(root, n_videos=4, per_video=2, markers=False):
    import cv2

    objs = [SynthObject("cup", "ellipse", (0.20, 0.35, 0.26, 0.40)),
            SynthObject("person", "ellipse", (0.30, 0.60, 0.60, 0.85))]
    clip, gts = synth_clip(duration_s=per_video / 5.0, sample_fps=5.0, objects=objs)
    images, anns = [], []
    cats = [{"id": 1, "name": "cup"}, {"id": 2, "name": "person"}, {"id": 3, "name": "stapler"}]
    cid = {"cup": 1, "person": 2}
    for v in range(n_videos):
        for k, (f, gt) in enumerate(zip(clip.frames, gts)):
            rel = f"X-{v}/{k:04d}.png"
            p = root / "images" / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), f)
            iid = len(images) + 1
            h, w = f.shape[:2]
            images.append({"id": iid, "file_name": rel, "width": w, "height": h})
            for lab, (x1, y1, x2, y2) in gt:
                a = {"id": len(anns) + 1, "image_id": iid, "category_id": cid[lab], "iscrowd": 0,
                     "bbox": [x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h], "area": 1.0}
                if markers:
                    a.update(score=0.5, verified=False)
                anns.append(a)
    ann = root / "annotations.json"
    ann.write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))
    return ann


def test_calibrate_end_to_end_on_manual_labels(mock_mode, tmp_path):
    ann = _dataset(tmp_path / "datasets" / "det")
    out, rep = tmp_path / "calib.yaml", tmp_path / "calib.json"
    assert main(["--gt", str(ann), "--annotator", "Damir", "--min-support", "2", "--out", str(out),
                 "--report", str(rep), "--with-open-vocab"]) == 0
    doc = yaml.safe_load(out.read_text())
    thr = doc["closed_set"]["score_thr"]
    assert set(thr) == {"cup", "person"} and all(0.899 <= v <= 0.9 for v in thr.values())   # округление вниз
    assert 0.899 <= doc["inventory"]["rare_open_vocab"]["thr"]["cup"] <= 0.9
    assert "person" not in json.loads(rep.read_text())["open_vocab"]["classes"]          # OWL-группы без person
    r = json.loads(rep.read_text())
    assert r["label_status"] == "manual" and r["annotator"] == "Damir"
    assert r["closed_set"]["unmapped_categories"] == {"3": "stapler"}
    assert r["closed_set"]["split"]["holdout_images"] > 0
    cup = r["closed_set"]["classes"]["cup"]
    assert cup["status"] == "calibrated" and cup["holdout_P"] == 1.0 and cup["holdout_R"] == 1.0
    assert "generated_by" in out.read_text()


def test_calibration_and_training_refuse_pseudo_labels(tmp_path):
    ann = _dataset(tmp_path / "p", n_videos=2, per_video=1, markers=True)
    with pytest.raises(UnverifiedLabelsError, match="псевдоразметка"):
        load_verified(ann, annotator="Damir")
    manual = _dataset(tmp_path / "m", n_videos=2, per_video=1)
    with pytest.raises(UnverifiedLabelsError, match="--annotator"):
        load_verified(manual)
    assert load_verified(manual, annotator="Damir")["info"]["label_status"] == "manual"
