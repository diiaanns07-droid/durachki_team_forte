"""eval_det (COCO AP), autolabel (предложения, verified-гейт), finetune.load_verified/split."""
from __future__ import annotations

import json

import numpy as np
import pytest

from ml.detect.autolabel import (FINETUNE_CLASSES, PROPOSAL_WARNING, is_verified, mark_verified, propose,
                                 video_key)
from ml.detect.eval_det import UNVERIFIED_WARNING, average_precision, evaluate
from ml.detect.finetune import UnverifiedLabelsError, load_verified, split_by_video


def _gt(verified=True):
    return {
        "info": {"verified": verified, "kind": "verified" if verified else "proposals"},
        "images": [{"id": 1, "file_name": "X-1/a.jpg"}, {"id": 2, "file_name": "X-2/b.jpg"}],
        "categories": [{"id": 1, "name": "cash"}, {"id": 2, "name": "cell_phone"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "verified": verified},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [50, 50, 10, 10], "verified": verified},
            {"id": 3, "image_id": 2, "category_id": 2, "bbox": [0, 0, 5, 10], "verified": verified},
        ],
    }


def _pred(gt, score=0.9):
    return [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": score}
            for a in gt["annotations"]]


def test_ap_perfect_and_degraded():
    gt = _gt()
    rep = evaluate(gt, _pred(gt))
    assert rep["mAP50"] == pytest.approx(1.0) and rep["mAP"] == pytest.approx(1.0)
    assert rep["quality_claim"] is True and "warning" not in rep
    # высокоуверенный FP перед TP → AP < 1
    fp = {"image_id": 1, "category_id": 1, "bbox": [200, 200, 5, 5], "score": 0.99}
    rep2 = evaluate(gt, _pred(gt) + [fp])
    assert rep2["per_class"]["cash"]["AP50"] < 1.0
    # пропуск половины GT → AP50 ≈ 0.5
    rep3 = evaluate(gt, [p for p in _pred(gt) if p["image_id"] == 1])
    assert rep3["per_class"]["cash"]["AP50"] == pytest.approx(0.5, abs=0.01)
    assert rep3["per_class"]["cell_phone"]["AP50"] == 0.0


def test_ap_helper_edge_cases():
    assert np.isnan(average_precision(np.array([]), np.array([]), 0))
    assert average_precision(np.array([]), np.array([]), 3) == 0.0


def test_eval_on_unverified_gt_is_not_quality():
    gt = _gt(verified=False)
    rep = evaluate(gt, _pred(gt))
    assert rep["quality_claim"] is False and rep["warning"] == UNVERIFIED_WARNING


def test_finetune_refuses_unverified(tmp_path):
    p = tmp_path / "proposals.json"
    p.write_text(json.dumps(_gt(verified=False)))
    with pytest.raises(UnverifiedLabelsError):
        load_verified(p)
    partly = _gt(verified=True)
    partly["annotations"][0]["verified"] = False
    p.write_text(json.dumps(partly))
    with pytest.raises(UnverifiedLabelsError):
        load_verified(p)
    p.write_text(json.dumps(_gt(verified=True)))
    assert load_verified(p)["info"]["verified"] is True


def test_mark_verified_requires_reviewer():
    with pytest.raises(ValueError):
        mark_verified(_gt(False), "  ")
    v = mark_verified(_gt(False), "reviewer-1")
    assert is_verified(v) and v["info"]["reviewer"] == "reviewer-1"
    assert not is_verified(_gt(False))


def test_video_key_and_split_no_leakage():
    assert video_key("Х-3/000010.jpg") == "X-3"          # кириллическая Х → латинская
    assert video_key("X-1_t12.4.jpg") == "X-1"
    coco = {"images": [{"id": i, "file_name": f"X-{i % 5}/f{i}.jpg"} for i in range(50)]}
    tr, va = split_by_video(coco, 0.2, seed=1)
    vids = lambda ids: {video_key(coco["images"][i]["file_name"]) for i in ids}  # noqa: E731
    assert tr and va and not (vids(tr) & vids(va))
    assert sorted(tr + va) == list(range(50))


def test_propose_outputs_unverified_proposals(tmp_path):
    import cv2

    from ml.detect.backends.mock import MockColorDetector
    from ml.detect.config import load_config
    from ml.detect.synth import synth_clip

    clip, _ = synth_clip(duration_s=1.0, sample_fps=2.0)
    paths = []
    for k, f in enumerate(clip.frames):
        p = tmp_path / "X-9" / f"{k:04d}.png"
        p.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(p), f)
        paths.append(p)
    pal = load_config()["mock"]["palette"]
    be = MockColorDetector(pal, labels=FINETUNE_CLASSES, name="mock")
    coco = propose(paths, be, min_score={c: 0.5 for c in FINETUNE_CLASSES}, root=tmp_path)
    assert coco["info"]["verified"] is False and coco["info"]["warning"] == PROPOSAL_WARNING
    assert [c["name"] for c in coco["categories"]] == FINETUNE_CLASSES
    assert coco["annotations"], "ожидались предложения для cash/money_counter"
    for a in coco["annotations"]:
        assert a["verified"] is False and 0 < a["score"] <= 1 and a["source"]
    assert all(im["video"] == "X-9" for im in coco["images"])
    assert not is_verified(coco)
