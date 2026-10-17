from contracts.sentinel_contracts import LABELS
from ml.detect.labels import (COCO_TO_LABEL, build_index_map, is_proxy, map_label, normalize_name,
                              source_tag)


def test_all_mapped_labels_in_contract():
    assert set(COCO_TO_LABEL.values()) <= set(LABELS)


def test_coco_mapping_from_task():
    expect = {"person": "person", "cell phone": "cell_phone", "cup": "cup", "wine glass": "cup", "bowl": "cup",
              "bottle": "bottle", "banana": "food", "apple": "food", "sandwich": "food", "orange": "food",
              "donut": "food", "cake": "food", "pizza": "food", "laptop": "laptop", "handbag": "bag",
              "backpack": "bag", "suitcase": "bag", "book": "document"}
    for k, v in expect.items():
        assert map_label(k) == v, k


def test_unmapped_and_variants():
    assert map_label("car") is None
    assert map_label("tv") is None
    assert map_label("Cell_Phone") == "cell_phone"
    assert map_label("cash_bag") == "cash_bag"
    assert normalize_name("  Cell-Phone ") == "cell phone"


def test_bag_is_not_cash_bag():
    # COCO handbag/backpack не доказывает «инкассаторская сумка»
    for name in ("handbag", "backpack", "suitcase"):
        assert map_label(name) == "bag"
        assert map_label(name) != "cash_bag"


def test_proxy_semantics_kept_in_source():
    assert is_proxy("book") and is_proxy("bowl")
    assert not is_proxy("person") and not is_proxy("handbag")
    assert source_tag("rtdetr_v2_r18vd", "book") == "rtdetr_v2_r18vd:book~proxy"
    assert source_tag("rtdetr_v2_r18vd", "cell phone") == "rtdetr_v2_r18vd:cell phone"


def test_build_index_map_with_and_without_proxy():
    id2label = {0: "person", 1: "bicycle", 67: "cell phone", 73: "book", 45: "bowl"}
    m = build_index_map(id2label)
    assert m == {0: "person", 67: "cell_phone", 73: "document", 45: "cup"}
    m2 = build_index_map(id2label, allow_proxy=False)
    assert m2 == {0: "person", 67: "cell_phone"}
    assert build_index_map(["cash", "cash_bag", "zebra"]) == {0: "cash", 1: "cash_bag"}


COCO80 = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
          "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
          "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
          "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
          "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
          "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
          "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
          "teddy bear", "hair drier", "toothbrush"]

EXPECTED = {"person": "person", "backpack": "bag", "handbag": "bag", "suitcase": "bag", "bottle": "bottle",
            "wine glass": "cup", "cup": "cup", "bowl": "cup", "banana": "food", "apple": "food", "sandwich": "food",
            "orange": "food", "broccoli": "food", "carrot": "food", "hot dog": "food", "pizza": "food",
            "donut": "food", "cake": "food", "laptop": "laptop", "cell phone": "cell_phone", "book": "document"}


def test_full_coco80_table():
    """Таблица классов общего детектора: 21 класс COCO → метки контракта, остальные 59 игнорируются."""
    from ml.detect.labels import STANDARD_LABELS

    assert len(COCO80) == 80
    got = {n: map_label(n) for n in COCO80}
    assert {n: v for n, v in got.items() if v is not None} == EXPECTED
    assert set(EXPECTED.values()) == set(STANDARD_LABELS)


def test_general_detector_never_yields_nonstandard_labels():
    """Общий детектор не даёт наличных/сейфа/двери/ключей, даже если в его id2label есть похожие имена."""
    from ml.detect.labels import NONSTANDARD_LABELS, STANDARD_LABELS

    assert NONSTANDARD_LABELS == {"cash", "money_counter", "safe", "safe_open", "door", "door_open", "cash_bag",
                                  "keys"}
    id2label = {0: "person", 1: "safe", 2: "door", 3: "key", 4: "money", 5: "banknote", 6: "cash bag",
                7: "open door", 8: "Cell Phone"}
    general = build_index_map(id2label, allowed=STANDARD_LABELS)
    assert general == {0: "person", 8: "cell_phone"}
    # дообученная на проверенной разметке модель (ONNX) — свои классы домена разрешены
    finetuned = build_index_map(["cash", "cash_bag", "safe", "safe_open", "door_open"])
    assert finetuned == {0: "cash", 1: "cash_bag", 2: "safe", 3: "safe_open", 4: "door_open"}
