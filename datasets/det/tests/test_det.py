"""Тесты логики разметки детектора без видео, весов и Label Studio: pytest datasets/det/tests -q"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_tasks  # noqa: E402
import export_coco  # noqa: E402
import ls_project  # noqa: E402
from common import CAT_ID, CLASS_NAMES, CLASSES, frame_name, norm_id, read_timecodes  # noqa: E402
from extract_frames import schedule  # noqa: E402


def test_classes_fixed_order():
    assert CLASS_NAMES == ["cash", "cash_bag", "safe_closed", "safe_open", "cell_phone", "cup", "food",
                           "money_counter", "door_open", "person"]
    assert CAT_ID["cash"] == 1 and CAT_ID["person"] == 10
    contract = {"person", "cell_phone", "cup", "bottle", "food", "cash", "money_counter", "laptop", "safe",
                "safe_open", "door", "door_open", "cash_bag", "bag", "document", "keys"}   # contracts.LABELS
    assert all(c["contract"] in contract for c in CLASSES)


def test_norm_id_and_frame_name():
    assert norm_id("Х-3") == "X-3" and norm_id("x - 07") == "X-7" and norm_id("clip") is None
    assert frame_name("X-1", 12.25) == "X-1/X-1_t0012.250.jpg"


def test_schedule_grid_and_dense_window():
    s = schedule(10.0, [5], step=1.0, dense_fps=4.0, window=1.0)
    ts = [t for t, _ in s]
    assert ts == sorted(set(ts))
    assert {0.0, 1.0, 9.0} <= set(ts) and max(ts) <= 10.0 - 0.05
    dense = [t for t, d in s if d]
    assert dense == [4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0]
    assert schedule(10.0, [9.9], window=1.0)[-1][0] <= 9.95          # окно у конца ролика обрезается
    assert schedule(10.0, [], step=1.0) == [(float(i), 0) for i in range(10)]


def test_read_timecodes(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("video_id,file,label,t_sec,t_all\nХ-3,Х-3.mov,violation,12,12;30\n"
                 "X-4,X-4.mov,no_violation,,\n,clip_a.mp4,violation,7.5,\n,EXT-002.mov,clean,3,\n", encoding="utf-8")
    tc = read_timecodes([p, tmp_path / "missing.csv"])
    assert tc == {"X-3": [12.0, 30.0], "X-4": [], "clip_a": [7.5], "EXT-002": []}


def _task(file, results, cancelled=False, tid=1):
    return {"id": tid, "data": {"image": f"/data/local-files/?d=frames/{file}", "video_id": file.split("/")[0]},
            "annotations": [{"id": tid, "was_cancelled": cancelled, "updated_at": "2026-10-17T10:00:00",
                             "result": results}]}


def _rect(label, x, y, w, h, rot=0, W=464, H=848):
    return {"type": "rectanglelabels", "from_name": "label", "to_name": "image", "original_width": W,
            "original_height": H, "value": {"x": x, "y": y, "width": w, "height": h, "rotation": rot,
                                            "rectanglelabels": [label]}}


def test_convert_ls_to_coco():
    tasks = [
        _task("X-1/X-1_t0001.000.jpg", [_rect("cash", 10, 10, 50, 25), _rect("person", 90, 90, 20, 20)], tid=1),
        _task("X-1/X-1_t0002.000.jpg", [], tid=2),                                    # негатив
        _task("X-2/X-2_t0001.000.jpg", [_rect("cash", 1, 1, 5, 5)], cancelled=True, tid=3),   # Skip
        {"id": 4, "data": {"image": "/data/local-files/?d=frames/X-2/X-2_t0002.000.jpg"}, "annotations": []},
        _task("X-2/X-2_t0003.000.jpg", [_rect("bottle", 1, 1, 5, 5)], tid=5),
    ]
    index = {"X-1/X-1_t0002.000.jpg": {"width": "464", "height": "848", "video_id": "X-1", "group": "X-1",
                                       "t_sec": "2.0", "dense": "1"}}
    images, anns, errors = export_coco.convert(tasks, index)
    assert [im["file_name"] for im in images] == ["X-1/X-1_t0001.000.jpg", "X-1/X-1_t0002.000.jpg",
                                                   "X-2/X-2_t0003.000.jpg"]
    assert len(errors) == 1 and "bottle" in errors[0]
    cash, person = anns
    assert cash["category_id"] == 1 and cash["bbox"] == [46.4, 84.8, 232.0, 212.0]
    assert person["category_id"] == 10 and person["bbox"] == [417.6, 763.2, 46.4, 84.8]   # обрезан краем кадра
    coco = export_coco.build_coco(images, anns, {"X-1": "train", "X-2": "val"})
    assert [c["id"] for c in coco["categories"]] == list(range(1, 11))
    assert coco["images"][1]["dense"] == 1 and coco["info"]["images_root"] == "data/frames"
    assert {a["image_id"] for a in coco["annotations"]} == {1}
    assert not any(Path(im["file_name"]).is_absolute() for im in coco["images"])


def test_rotated_rect_is_axis_aligned_bbox():
    x, y, w, h = export_coco.rect_to_xywh({"x": 50, "y": 50, "width": 10, "height": 5, "rotation": 90}, 100, 100)
    assert (x, y, w, h) == (45.0, 50.0, 5.0, 10.0)


def _imgs(spec):
    """spec: {group: (n_images, [category_ids])}"""
    images, anns = [], []
    for g, (n, cats) in spec.items():
        for i in range(n):
            f = f"{g}/{g}_{i}.jpg"
            images.append({"file_name": f, "group": g})
            anns += [{"file": f, "category_id": c} for c in cats if i == 0]
    return images, anns


def test_split_by_video_covers_rare_classes_and_is_stable():
    spec = {f"X-{i}": (10, [10]) for i in range(1, 11)}
    spec["X-3"] = (10, [10, 2])
    spec["X-8"] = (10, [10, 2])          # cash_bag в 2 видео → одно в val, одно в train
    spec["X-5"] = (10, [10, 4])          # safe_open в одном видео → только train
    images, anns = _imgs(spec)
    split = export_coco.assign_split(images, anns, {}, 0.2)
    assert set(split) == set(spec) and set(split.values()) == {"train", "val"}
    assert {split["X-3"], split["X-8"]} == {"train", "val"}
    assert split["X-5"] == "train"
    assert 1 <= sum(v == "val" for v in split.values()) <= 3
    assert export_coco.assign_split(images, anns, {}, 0.2) == split
    # зафиксированные группы не переезжают, новая распределяется
    images2, anns2 = _imgs({**spec, "X-11": (10, [10])})
    split2 = export_coco.assign_split(images2, anns2, split, 0.2)
    assert {k: split2[k] for k in split} == split and "X-11" in split2


def test_stats_and_readme_block(tmp_path):
    images, anns = _imgs({"X-1": (2, [1, 10]), "X-2": (1, [1])})
    for im in images:
        im.update(width=464, height=848, video_id=im["group"], t_sec=0.0, dense=0)
    for a in anns:
        a.update(bbox=[0, 0, 10, 10], area=100, iscrowd=0)
    coco = export_coco.build_coco(images, anns, {"X-1": "train", "X-2": "val"})
    md = export_coco.stats_md(coco, {"X-1": "X-47"})
    assert "| 1 | `cash` | `cash` | 2 | 1 | 1 | 2 |" in md and "X-1 (≡X-47)" in md
    readme = tmp_path / "README.md"
    readme.write_text("# t\n\n<!-- STATS:BEGIN -->\nold\n<!-- STATS:END -->\n\nхвост\n", encoding="utf-8")
    export_coco.update_readme(readme, md)
    text = readme.read_text(encoding="utf-8")
    assert "old" not in text and "хвост" in text and text.count("STATS:BEGIN") == 1


def test_label_config_is_valid_xml_with_all_classes():
    root = ET.fromstring(ls_project.label_config())
    labels = [el.get("value") for el in root.iter("Label")]
    assert labels == CLASS_NAMES
    assert len({el.get("hotkey") for el in root.iter("Label")}) == len(CLASS_NAMES)


def test_owlv2_postprocess_pads_to_square_and_suppresses():
    np = pytest.importorskip("numpy")
    from prelabel_owlv2 import postprocess

    C = len(CLASSES)
    scores = np.zeros((4, C), dtype=np.float32)
    boxes = np.zeros((4, 4), dtype=np.float32)
    ci = {c["name"]: i for i, c in enumerate(CLASSES)}
    # кадр 464x848: квадрат 848, бокс по центру по x в долях квадрата
    scores[0, ci["cash"]], boxes[0] = 0.5, [0.25, 0.5, 0.1, 0.1]
    scores[1, ci["cash"]], boxes[1] = 0.4, [0.251, 0.5, 0.1, 0.1]         # дубль → NMS
    scores[2, ci["safe_open"]], boxes[2] = 0.3, [0.2, 0.2, 0.2, 0.2]
    scores[3, ci["safe_closed"]], boxes[3] = 0.6, [0.2, 0.2, 0.2, 0.2]    # тот же сейф, увереннее → остаётся он
    out = postprocess(scores, boxes, 464, 848)
    assert [b["label"] for b in out] == ["cash", "safe_closed"]
    scores[3, ci["safe_closed"]] = 0.0
    scores[3, ci["door_open"]], boxes[3] = 0.25, [0.2, 0.2, 0.2, 0.2]   # тот же бокс другим классом → один бокс
    assert [b["label"] for b in postprocess(scores, boxes, 464, 848)] == ["cash", "safe_open"]   # 0.3 > 0.25
    x1, y1, x2, y2 = out[0]["xyxy"]
    assert abs((x1 + x2) / 2 - 212) < 0.5 and abs((y1 + y2) / 2 - 424) < 0.5


def test_select_respects_target_min_per_video_and_previous():
    rows = []
    for v in ("X-1", "X-2", "X-3"):
        for k in range(40):
            rows.append({"file": f"{v}/{k}.jpg", "video_id": v, "t_sec": float(k), "dense": int(v == "X-1" and k > 30),
                         "dup": k % 5 == 4})
    preds = {"X-2/10.jpg": [{"label": "cash_bag", "score": 0.3}]}
    chosen = build_tasks.select(rows, preds, target=30, min_per_video=6, previous={"X-3/39.jpg"})
    files = {r["file"] for r in chosen}
    assert len(chosen) == 30 and "X-3/39.jpg" in files and "X-2/10.jpg" in files
    assert all(sum(r["video_id"] == v for r in chosen) >= 6 for v in ("X-1", "X-2", "X-3"))
    assert not any(r["dup"] for r in chosen if r["file"] != "X-3/39.jpg")
    task = build_tasks.to_task({**rows[0], "group": "X-1", "width": 464, "height": 848},
                               [{"label": "cash", "score": 0.5, "xyxy": [46.4, 84.8, 92.8, 169.6]}], "owlv2")
    v = task["predictions"][0]["result"][0]["value"]
    assert (round(v["x"], 6), round(v["y"], 6), round(v["width"], 6)) == (10.0, 10.0, 10.0)
    assert task["data"]["image"].startswith("/data/local-files/?d=frames/")
    json.dumps(task)


def test_apply_review_decisions():
    import apply_review

    cands = {"X-1/a.jpg": {"width": 464, "height": 848, "cands": [
        {"n": 1, "xyxy": [10, 10, 50, 50], "src": "owl"}, {"n": 2, "xyxy": [11, 11, 50, 50], "src": "extra"},
        {"n": 3, "xyxy": [100, 100, 200, 300], "src": "extra"}, {"n": 4, "xyxy": [240, 70, 255, 90], "src": "extra"},
        {"n": 5, "xyxy": [60, 350, 140, 430], "src": "owl"}, {"n": 6, "xyxy": [150, 410, 206, 446], "src": "owl"}]}}
    decisions = [
        {"file": "X-1/a.jpg", "keep": {"1": "cash", "2": "cash", "3": "person", "4": "person", "5": "money_counter",
                                    "6": "money_counter", "9": "cup"},
         "add": [{"label": "cell_phone", "xyxy": [500, 20, 440, 60]}, {"label": "bottle", "xyxy": [0, 0, 9, 9]}]},
        {"file": "X-9/zz.jpg", "keep": {}},
    ]
    out, errors = apply_review.apply(cands, decisions)
    boxes = out["X-1/a.jpg"]["boxes"]
    assert [(b["label"], b["src"]) for b in boxes] == [("cash", "owl"), ("person", "extra"), ("money_counter", "owl"),
                                                       ("cell_phone", "added")]
    assert [b["xyxy"] for b in boxes if b["label"] == "money_counter"] == [[60, 350, 140, 430]]   # крупная
    assert boxes[3]["xyxy"] == [440.0, 20.0, 464.0, 60.0]                     # упорядочено и обрезано краем
    assert len(errors) == 3 and any("#9" in e for e in errors) and any("X-9" in e for e in errors)
    res = apply_review.ls_result(boxes[:1], 464, 848)[0]["value"]
    assert res["rectanglelabels"] == ["cash"] and abs(res["x"] - 10 / 464 * 100) < 1e-9


def test_coco_status_in_info():
    images, anns = _imgs({"X-1": (1, [1])})
    for im in images:
        im.update(width=464, height=848, video_id="X-1", t_sec=0.0, dense=0)
    for a in anns:
        a.update(bbox=[0, 0, 10, 10], area=100, iscrowd=0)
    coco = export_coco.build_coco(images, anns, {"X-1": "train"}, status="черновик")
    assert coco["info"]["status"] == "черновик" and "черновик" in export_coco.stats_md(coco, {})
    human = export_coco.build_coco(images, anns, {"X-1": "train"})
    assert human["info"]["status"] == export_coco.HUMAN and "kind" not in human["info"]
    assert all("verified" not in a for a in human["annotations"])
    draft = export_coco.build_coco(images, anns, {"X-1": "train"}, status="черновик", proposals="owlv2+claude-review")
    # так ml/detect (A05) распознаёт псевдоразметку и не учится на ней
    assert draft["info"]["kind"] == "proposals" and draft["info"]["verified"] is False
    assert all(a["verified"] is False and a["source"] == "owlv2+claude-review" for a in draft["annotations"])
