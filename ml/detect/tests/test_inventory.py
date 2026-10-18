"""desk_inventory / safe_door_state: present/unknown (никогда «нет»), зоны, контрасты, абляция baseline ↔ rare_open_vocab."""
from __future__ import annotations

import json

import numpy as np
import pytest

from ml.detect import desk_inventory, safe_door_state
from contracts.sentinel_contracts import Zone
from ml.detect.inventory import KINDS, box_zone_fraction, zone_occlusion, zone_polys
from ml.detect.synth import SynthObject, synth_clip

DESK = [(0.10, 0.30), (0.90, 0.30), (0.90, 0.60), (0.10, 0.60)]
SAFE = [(0.65, 0.62), (0.98, 0.62), (0.98, 0.88), (0.65, 0.88)]
ZONES = [Zone("desk", DESK), Zone("safe", SAFE)]
RARE = {"inventory": {"mode": "rare_open_vocab"}}


def scene(*objs):
    clip, _ = synth_clip(duration_s=3.0, sample_fps=5.0, objects=list(objs))
    return clip


def desk_scene():
    return scene(
        SynthObject("cup", "ellipse", (0.20, 0.35, 0.26, 0.40)),
        SynthObject("cell_phone", "rect", (0.30, 0.35, 0.34, 0.42)),
        SynthObject("bag", "rect", (0.40, 0.35, 0.55, 0.50)),
        SynthObject("cash", "rect", (0.60, 0.35, 0.68, 0.40)),
        SynthObject("cash_bag", "rect", (0.70, 0.42, 0.85, 0.55)),
        SynthObject("food", "ellipse", (0.02, 0.90, 0.08, 0.95)),       # вне зоны стола
        SynthObject("safe", "rect", (0.70, 0.65, 0.95, 0.85)),
    )


def test_baseline_reports_present_or_unknown_never_absent(mock_mode):
    inv = desk_inventory(desk_scene(), 1.0, ZONES)
    assert inv.mode == "baseline" and inv.zone == "desk" and inv.frames_checked == 3
    assert set(inv.items) == set(KINDS)
    assert {it.state for it in inv.items.values()} <= {"present", "unknown"}
    assert inv.present("cup") and inv.present("phone") and inv.present("bag")
    assert "не различаются" in inv.items["phone"].note
    assert "не считать инкассаторской" in inv.items["bag"].note
    assert inv.items["food"].state == "unknown"                  # вне зоны стола
    for k in ("cash", "cash_bag", "personal_item"):              # в COCO таких классов нет → unknown, не «нет»
        assert inv.items[k].state == "unknown" and "baseline" in inv.items[k].note
    it = inv.items["cup"]
    assert it.frames_seen == 3 and it.frames_checked == 3 and 0 < it.conf <= 1 and it.t in (0.5, 1.0, 1.5)
    json.dumps(inv.as_dict())


def test_baseline_safe_is_unknown(mock_mode):
    st = safe_door_state(desk_scene(), 1.0, ZONES)
    assert st.state == "unknown" and st.mode == "baseline" and st.votes["open"] == 0
    assert any("не наблюдаемо" in w for w in st.warnings)


def test_rare_open_vocab_adds_cash_and_confirmed_cash_bag(mock_mode):
    inv = desk_inventory(desk_scene(), 1.0, ZONES, config=RARE)
    assert inv.mode == "rare_open_vocab"
    assert inv.present("cash") and inv.present("cash_bag")
    assert inv.items["cash"].frames_checked == 2                   # OWL — не более max_frames кадров окна
    assert any("cash_bag>generic_bag" in s for s in inv.items["cash_bag"].sources)
    assert inv.present("personal_item")                            # обычная сумка → личная вещь
    assert inv.items["food"].state == "unknown"


def test_unconfirmed_cash_bag_stays_bag(mock_mode, monkeypatch):
    import ml.detect.backends as B

    groups = dict(B.MOCK_GROUP_COLORS)
    groups["generic_bag"] = ["bag", "cash_bag"]                     # «обычная сумка» скорит не хуже → контраст не пройден
    monkeypatch.setattr(B, "MOCK_GROUP_COLORS", groups)
    inv = desk_inventory(desk_scene(), 1.0, ZONES, config=RARE)
    assert inv.items["cash_bag"].state == "unknown"
    assert "контраст" in inv.items["cash_bag"].note
    assert inv.present("bag")


def test_landline_vetoes_coco_phone_only_in_rare_mode(mock_mode, monkeypatch):
    import ml.detect.backends as B

    groups = dict(B.MOCK_GROUP_COLORS)
    groups["phone"], groups["landline"] = [], ["cell_phone"]        # объект выглядит как стационарный телефон
    monkeypatch.setattr(B, "MOCK_GROUP_COLORS", groups)
    clip = desk_scene()
    assert desk_inventory(clip, 1.0, ZONES).present("phone")                    # baseline: COCO cell phone
    inv = desk_inventory(clip, 1.0, ZONES, config=RARE)
    assert inv.items["phone"].state == "unknown" and "стационарн" in inv.items["phone"].note


def test_safe_open_closed_unknown_in_rare_mode(mock_mode):
    open_clip = scene(SynthObject("safe_open", "rect", (0.70, 0.65, 0.95, 0.85)))
    closed_clip = scene(SynthObject("safe", "rect", (0.70, 0.65, 0.95, 0.85)))
    empty_clip = scene(SynthObject("safe", "rect", (0.05, 0.05, 0.20, 0.15)))      # «сейф» вне зоны сейфа
    so = safe_door_state(open_clip, 1.0, ZONES, config=RARE)
    assert so.state == "open" and so.votes["open"] == 2 and so.conf > 0 and so.bbox is not None
    assert safe_door_state(closed_clip, 1.0, ZONES, config=RARE).state == "closed"
    assert safe_door_state(empty_clip, 1.0, ZONES, config=RARE).state == "unknown"


def test_flicker_single_frame_is_unknown(mock_mode):
    # чашка видна только на одном кадре окна (0.5, 1.0, 1.5) → недостаточно
    clip = scene(SynthObject("cup", "ellipse", (0.20, 0.35, 0.26, 0.40), t_on=1.4, t_off=1.6))
    it = desk_inventory(clip, 1.0, ZONES).items["cup"]
    assert it.state == "unknown" and it.frames_seen == 1 and "1/3" in it.note


def test_no_zones_and_broken_frames(mock_mode):
    clip = desk_scene()
    inv = desk_inventory(clip, 1.0, None)
    assert inv.zone is None and any("зона стола не задана" in w for w in inv.warnings)
    assert inv.present("food")                                     # без зоны — весь кадр

    class Broken:
        times, duration, path = clip.times, clip.duration, "broken"

        def frame_at(self, t):
            raise IOError("decode error")

    inv2 = desk_inventory(Broken(), 1.0, ZONES)
    assert inv2.frames_checked == 0 and all(it.state == "unknown" for it in inv2.items.values())


def test_person_occlusion_warning(mock_mode):
    clip = scene(SynthObject("person", "rect", (0.05, 0.25, 0.95, 0.65)))
    inv = desk_inventory(clip, 1.0, ZONES)
    assert inv.person_occlusion > 0.9 and any("закрыта людьми" in w for w in inv.warnings)


def test_inventory_unavailable_without_weights(no_network):
    from ml.detect import DetectorUnavailable

    with pytest.raises(DetectorUnavailable):
        desk_inventory(desk_scene(), 1.0, ZONES)


def test_zone_helpers():
    polys = zone_polys(ZONES, ["desk"])
    assert len(polys) == 1
    assert zone_polys({"desk": DESK}, ["desk"])[0].shape == (4, 2)
    assert len(zone_polys({"desk": [DESK, SAFE]}, ["desk"])) == 2
    assert zone_polys([{"name": "safe", "polygon": SAFE}], ["safe"])[0].shape == (4, 2)
    fr = box_zone_fraction(np.array([[0.2, 0.35, 0.3, 0.45], [0.0, 0.0, 0.05, 0.05], [0.05, 0.3, 0.15, 0.4]]), polys)
    assert fr[0] == pytest.approx(1.0) and fr[1] == 0.0 and 0.3 < fr[2] < 0.7
    assert box_zone_fraction(np.array([[0, 0, 1, 1]]), [])[0] == 1.0
    assert zone_occlusion(polys, np.array([[0.0, 0.0, 1.0, 1.0]])) == pytest.approx(1.0)
    assert zone_occlusion(polys, np.zeros((0, 4))) == 0.0


def test_landline_veto_applies_to_all_window_frames(mock_mode, monkeypatch):
    import ml.detect.backends as B

    groups = dict(B.MOCK_GROUP_COLORS)
    groups["phone"], groups["landline"] = [], ["cell_phone"]
    monkeypatch.setattr(B, "MOCK_GROUP_COLORS", groups)
    # 5 кадров окна, OWL — только на 2: COCO-«телефон» на остальных 3 кадрах тоже должен погаснуть
    cfg = {"inventory": {"mode": "rare_open_vocab", "n_frames": 5, "window_s": 1.6}}
    inv = desk_inventory(desk_scene(), 1.0, ZONES, config=cfg)
    assert inv.items["phone"].state == "unknown" and inv.items["phone"].frames_seen == 0


def test_cache_not_reused_across_clip_objects(mock_mode):
    import gc

    from ml.detect.inventory import _CACHE

    a = scene(SynthObject("cup", "ellipse", (0.20, 0.35, 0.26, 0.40)))
    assert desk_inventory(a, 1.0, ZONES).present("cup")
    assert len(_CACHE.d) > 0
    del a
    gc.collect()
    b = scene()                                              # пустой стол, тот же path/времена
    assert desk_inventory(b, 1.0, ZONES).items["cup"].state == "unknown"
