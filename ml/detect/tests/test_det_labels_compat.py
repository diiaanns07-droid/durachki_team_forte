"""Совместимость с выгрузкой разметки det-labels (datasets/det/export_coco.py): images_root, contract_label,
группы дубликатов и фиксированный split. Только синтетика."""
from __future__ import annotations

import json

import pytest

from ml.detect.autolabel import FINETUNE_CLASSES, category_label, default_images_root, group_key
from ml.detect.finetune import CocoDetDataset, split_by_video
from ml.detect.weights import REPO_ROOT

CATS = [{"id": 1, "name": "cash", "supercategory": "object", "contract_label": "cash"},
        {"id": 3, "name": "safe_closed", "supercategory": "object", "contract_label": "safe"},
        {"id": 4, "name": "safe_open", "supercategory": "object", "contract_label": "safe_open"},
        {"id": 9, "name": "door_open", "supercategory": "object", "contract_label": "door_open"},
        {"id": 10, "name": "person", "supercategory": "object", "contract_label": "person"}]


def test_images_root_from_info(tmp_path):
    ann = tmp_path / "datasets" / "det" / "annotations.json"
    assert default_images_root(ann, {"info": {"images_root": "data/frames"}}) == REPO_ROOT / "data" / "frames"
    assert default_images_root(ann, {"info": {"images_root": str(tmp_path / "fr")}}) == tmp_path / "fr"
    assert default_images_root(ann, {"info": {}}) == ann.parent


def test_category_label_uses_contract_label():
    assert [category_label(c) for c in CATS] == ["cash", "safe", "safe_open", "door_open", "person"]
    assert category_label({"name": "stapler"}) is None
    assert category_label({"name": "banknotes"}) == "cash"          # без contract_label — по имени/синонимам


def test_split_respects_fixed_split_and_duplicate_groups():
    fixed = {"images": [{"id": 1, "file_name": "X-4/a.jpg", "group": "X-4", "split": "val"},
                        {"id": 2, "file_name": "X-47/a.jpg", "group": "X-4", "split": "val"},
                        {"id": 3, "file_name": "X-1/a.jpg", "group": "X-1", "split": "train"}]}
    assert split_by_video(fixed) == ([3], [1, 2])
    free = {"images": [{"id": i, "file_name": f"{v}/{i}.jpg", "group": g}
                       for i, (v, g) in enumerate([("X-4", "X-4"), ("X-47", "X-4"), ("X-1", "X-1"), ("X-2", "X-2"),
                                                   ("X-3", "X-3"), ("Х-5", "X-5")])]}
    for seed in range(5):
        tr, va = split_by_video(free, 0.4, seed)
        sides = {group_key(im): (im["id"] in va) for im in free["images"]}
        assert len({(im["id"] in va) for im in free["images"] if im["group"] == "X-4"}) == 1   # дубликаты вместе
        assert sorted(tr + va) == list(range(6)) and sides
    assert group_key({"file_name": "Х-5/1.jpg"}) == "X-5"


def test_finetune_dataset_maps_categories_to_contract_labels():
    coco = {"images": [{"id": 1, "file_name": "X-1/a.jpg", "width": 100, "height": 100}],
            "categories": CATS,
            "annotations": [{"id": 1, "image_id": 1, "category_id": 3, "bbox": [1, 1, 10, 10], "iscrowd": 0},
                            {"id": 2, "image_id": 1, "category_id": 9, "bbox": [20, 20, 10, 10], "iscrowd": 0},
                            {"id": 3, "image_id": 1, "category_id": 10, "bbox": [40, 40, 10, 10], "iscrowd": 0}]}
    ds = CocoDetDataset(coco, [1], ".", processor=None, classes=FINETUNE_CLASSES)
    got = sorted(FINETUNE_CLASSES[c] for _b, c in ds.anns[1])
    assert got == ["door_open", "safe"]                              # safe_closed → safe; person не в классах


def test_calibrate_on_det_labels_style_export(mock_mode, tmp_path):
    """Выгрузка в формате export_coco.py (images_root, contract_label, group/split) → калибровка находит кадры,
    safe_closed учитывается как safe, calib/holdout = train/val из разметки."""
    import cv2

    from ml.detect.calibrate import main
    from ml.detect.synth import SynthObject, synth_clip

    frames_root = tmp_path / "frames"
    clip, gts = synth_clip(duration_s=0.8, sample_fps=5.0, objects=[
        SynthObject("safe", "rect", (0.70, 0.65, 0.95, 0.85)), SynthObject("cash", "rect", (0.25, 0.35, 0.33, 0.40))])
    images, anns = [], []
    cid = {"safe": 3, "cash": 1}
    for v, split in (("X-1", "train"), ("X-2", "train"), ("X-3", "val")):
        for k, (f, gt) in enumerate(zip(clip.frames, gts)):
            rel = f"{v}/{k:04d}.png"
            (frames_root / v).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(frames_root / rel), f)
            iid = len(images) + 1
            h, w = f.shape[:2]
            images.append({"id": iid, "file_name": rel, "width": w, "height": h, "video_id": v, "group": v,
                           "split": split, "t_sec": k / 5.0, "dense": 0})
            for lab, (x1, y1, x2, y2) in gt:
                anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": cid[lab], "iscrowd": 0,
                             "bbox": [x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h], "area": 1.0})
    ann_path = tmp_path / "datasets" / "det" / "annotations.json"
    ann_path.parent.mkdir(parents=True)
    ann_path.write_text(json.dumps({"info": {"images_root": str(frames_root)}, "categories": CATS,
                                    "images": images, "annotations": anns}))
    out, rep = tmp_path / "c.yaml", tmp_path / "c.json"
    assert main(["--gt", str(ann_path), "--annotator", "Damir", "--min-support", "2", "--out", str(out),
                 "--report", str(rep)]) == 0
    r = json.loads(rep.read_text())["closed_set"]
    assert r["split"]["calib_images"] == 8 and r["split"]["holdout_images"] == 4
    assert r["split"]["holdout_videos"] == ["X-3"]
    assert r["classes"]["safe"]["n_gt_calib"] == 8 and r["classes"]["safe"]["n_gt_holdout"] == 4
    assert r["unmapped_categories"] == {}


def test_eval_maps_predictions_to_contract_categories():
    from ml.detect.eval_det import evaluate

    gt = {"info": {"verified": True}, "categories": CATS,
          "images": [{"id": 1, "file_name": "X-1/a.jpg", "width": 100, "height": 100}],
          "annotations": [{"id": 1, "image_id": 1, "category_id": 3, "bbox": [1, 1, 10, 10], "verified": True}]}
    rep = evaluate(gt, [{"image_id": 1, "category_id": 3, "bbox": [1, 1, 10, 10], "score": 0.9}])
    assert rep["per_class"]["safe_closed"]["AP50"] == pytest.approx(1.0)


def test_run_predictions_maps_detector_labels_to_gt_categories(mock_mode, tmp_path):
    """Детекция «safe» сопоставляется категории safe_closed (contract_label=safe), а не теряется."""
    import cv2

    from ml.detect.eval_det import run_predictions
    from ml.detect.synth import SynthObject, synth_clip

    clip, _ = synth_clip(duration_s=0.2, objects=[SynthObject("safe", "rect", (0.70, 0.65, 0.95, 0.85))])
    (tmp_path / "X-1").mkdir()
    cv2.imwrite(str(tmp_path / "X-1" / "a.png"), clip.frames[0])
    gt = {"categories": CATS, "images": [{"id": 7, "file_name": "X-1/a.png", "width": 464, "height": 848}],
          "annotations": []}
    preds = run_predictions(gt, tmp_path)
    assert preds and {p["category_id"] for p in preds} == {3} and all(p["image_id"] == 7 for p in preds)
