"""Генератор синтетических рядов ``SceneState`` для тестов и отладки правил.

Только синтетика: никаких реальных кадров, описаний сцен или данных датасета.
``segments`` — список ``(t0, t1, {сигнал: значение | None})``: значение действует на
``[t0, t1)``; ``None`` удаляет ключ из состояния (сигнал «неизвестен» — окклюзия/пропуск).
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

try:
    from contracts.sentinel_contracts import SceneState, Track, Zone  # noqa
except ImportError:  # пока ветка A01 не смерджена
    from contracts.sentinel_contracts import SceneState, Track, Zone  # noqa

__all__ = ["CLEAN_BASE", "make_states", "make_tracks", "make_zones", "verified_catalog", "unverified_catalog",
           "CLEAN_SCENARIOS", "RULE_CASES"]

# Обычная работа: кассир за столом, наличные на столе, один человек в кадре.
CLEAN_BASE: dict[str, float] = {
    "cashier_present": 1.0,
    "cashier_in_workplace": 1.0,
    "n_persons": 1.0,
    "outsider_in_cashier_area": 0.0,
    "collector_present": 0.0,
    "safe_open": 0.0,
    "door_open": 0.0,
    "cash_visible": 1.0,
    "cash_bag_visible": 0.0,
    "phone_in_hand": 0.0,
    "phone_to_ear": 0.0,
    "eating_drinking": 0.0,
    "hands_hidden": 0.0,
    "cashier_facing_away": 0.0,
    "workplace_unattended_with_cash": 0.0,
    "handover_motion": 0.0,
    "motion_energy": 0.2,
}

_INT_SIGNALS = {"n_persons"}


def make_states(
    duration_s: float = 60.0,
    fps: float = 5.0,
    segments: Iterable[tuple[float, float, Mapping[str, Optional[float]]]] = (),
    base: Optional[Mapping[str, float]] = None,
    noise: float = 0.0,
    seed: int = 0,
    omit: Sequence[str] = (),
    start_t: float = 0.0,
) -> list[Any]:
    rng = np.random.default_rng(seed)
    base = dict(CLEAN_BASE if base is None else base)
    segs = list(segments)
    n = int(round(duration_s * fps))
    out = []
    for i in range(n):
        t = round(start_t + i / fps, 4)
        sig: dict[str, Optional[float]] = dict(base)
        for t0, t1, vals in segs:
            if t0 <= t < t1:
                sig.update(vals)
        clean: dict[str, float] = {}
        for k, v in sig.items():
            if v is None or k in omit:
                continue
            v = float(v)
            if noise > 0 and k not in _INT_SIGNALS:
                v = float(np.clip(v + rng.normal(0.0, noise), 0.0, 1.0))
            clean[k] = v
        out.append(SceneState(t=t, signals=clean))
    return out


def make_zones() -> list[Any]:
    """Синтетические зоны (координаты условные, только для тестов; в проде зоны даёт A06)."""
    return [
        Zone(name="cashier_area", polygon=[(0.0, 0.4), (1.0, 0.4), (1.0, 1.0), (0.0, 1.0)]),
        Zone(name="desk", polygon=[(0.2, 0.55), (0.8, 0.55), (0.8, 0.75), (0.2, 0.75)]),
        Zone(name="safe", polygon=[(0.8, 0.8), (1.0, 0.8), (1.0, 1.0), (0.8, 1.0)]),
        Zone(name="door", polygon=[(0.0, 0.0), (0.3, 0.0), (0.3, 0.2), (0.0, 0.2)]),
        Zone(name="client_area", polygon=[(0.3, 0.0), (1.0, 0.0), (1.0, 0.38), (0.3, 0.38)]),
    ]


Span = tuple[int, str, str, float, float, tuple[float, float, float, float]]


def make_tracks(duration_s: float = 60.0, fps: float = 5.0, spans: Sequence[Span] = ()) -> list[Any]:
    """``spans``: (track_id, label, role, t0, t1, bbox) — трек с постоянным боксом на [t0, t1)."""
    tracks = []
    for tid, label, role, t0, t1, box in spans:
        boxes = []
        n = int(round(duration_s * fps))
        for i in range(n):
            t = round(i / fps, 4)
            if t0 <= t < t1:
                boxes.append((t, box))
        tracks.append(Track(track_id=tid, label=label, role=role if label == "person" else "unknown",
                            role_conf=0.9 if label == "person" else 0.0, boxes=boxes))
    return tracks


def verified_catalog(types: Iterable[str] = ("V01", "V02", "V03", "V04", "V05", "V06", "V07", "V08")) -> dict:
    """Каталог-заглушка для тестов: у каждого типа одна «проверенная» запись (текст — заглушка, не норма)."""
    return {"version": 1, "types": {
        tid: {"name": f"Тип {tid}", "severity": "high",
              "rules": [{"document": "TEST-DOC", "clause": None, "url": "https://example.invalid/test",
                         "quote": "тестовая заглушка", "verified": True}]}
        for tid in types}}


def unverified_catalog(types: Iterable[str] = ("V01", "V02", "V03", "V04", "V05", "V06", "V07", "V08")) -> dict:
    cat = verified_catalog(types)
    for entry in cat["types"].values():
        for r in entry["rules"]:
            r["verified"] = False
    return cat


# Чистые сценарии: правила не должны выдать ни одного события (ни одной оценки).
CLEAN_SCENARIOS: dict[str, dict[str, Any]] = {
    "idle": {"duration_s": 60.0, "segments": []},
    "client_service": {"duration_s": 90.0, "segments": [
        (10.0, 40.0, {"n_persons": 2.0}),
        (15.0, 16.0, {"handover_motion": 0.7}),
        (30.0, 31.0, {"handover_motion": 0.7}),
        (50.0, 80.0, {"n_persons": 2.0}),
    ]},
    "proper_collection": {"duration_s": 90.0, "segments": [
        (19.0, 21.0, {"door_open": 1.0}),
        (20.0, 60.0, {"collector_present": 1.0, "n_persons": 2.0}),
        (22.0, 58.0, {"cash_bag_visible": 1.0}),
        (30.0, 45.0, {"safe_open": 1.0}),
        (40.0, 41.0, {"handover_motion": 0.8}),
        (59.0, 61.0, {"door_open": 1.0}),
    ]},
    "brief_absence": {"duration_s": 60.0, "segments": [
        (40.0, 42.0, {"cashier_present": 0.0, "cashier_in_workplace": 0.0}),
    ]},
    "safe_opened_by_cashier": {"duration_s": 60.0, "segments": [
        (10.0, 40.0, {"safe_open": 1.0}),
    ]},
    "single_frame_flicker": {"duration_s": 60.0, "segments": [
        (10.0, 10.2, {"outsider_in_cashier_area": 1.0}),
        (20.0, 20.2, {"safe_open": 1.0, "cashier_present": 0.0}),
        (30.0, 30.2, {"phone_in_hand": 1.0}),
        (40.0, 40.2, {"door_open": 1.0}),
    ]},
    "noisy": {"duration_s": 60.0, "segments": [], "noise": 0.08},
}

# Позитив/негатив для каждого включённого правила. onset — момент начала ситуации (сек).
RULE_CASES: dict[str, dict[str, Any]] = {
    "V01_outsider_in_cashier_area": {
        "type_id": "V01", "assessment": "candidate_violation", "onset": 20.0,
        "pos": {"duration_s": 60.0, "segments": [(20.0, 26.0, {"outsider_in_cashier_area": 1.0, "n_persons": 2.0})]},
        "neg": [
            {"duration_s": 60.0, "segments": [(20.0, 21.0, {"outsider_in_cashier_area": 1.0, "n_persons": 2.0})]},
            {"duration_s": 60.0, "segments": [(20.0, 30.0, {"outsider_in_cashier_area": 1.0, "collector_present": 1.0,
                                                             "n_persons": 2.0})]},
        ],
    },
    "V02_cash_left_unattended": {
        "type_id": "V02", "assessment": "candidate_violation", "onset": 30.0,
        "pos": {"duration_s": 60.0, "segments": [(30.0, 42.0, {"cashier_present": 0.0, "cashier_in_workplace": 0.0})]},
        "neg": [
            {"duration_s": 60.0, "segments": [(30.0, 33.0, {"cashier_present": 0.0, "cashier_in_workplace": 0.0})]},
            {"duration_s": 60.0, "segments": [(30.0, 42.0, {"cashier_present": 0.0, "cashier_in_workplace": 0.0,
                                                             "cash_visible": 0.0})]},
        ],
    },
    "V03_safe_open_with_outsider": {
        "type_id": "V03", "assessment": "candidate_violation", "onset": 22.0,
        "pos": {"duration_s": 60.0, "segments": [(20.0, 30.0, {"safe_open": 1.0}),
                                                  (22.0, 26.0, {"outsider_in_cashier_area": 1.0, "n_persons": 2.0})]},
        "neg": [
            {"duration_s": 60.0, "segments": [(20.0, 30.0, {"safe_open": 1.0}),
                                              (22.0, 26.0, {"outsider_in_cashier_area": 1.0, "collector_present": 1.0,
                                                            "n_persons": 2.0})]},
        ],
    },
    "V03_safe_left_open": {
        "type_id": "V03", "assessment": "candidate_violation", "onset": 15.0,
        "pos": {"duration_s": 60.0, "segments": [(10.0, 30.0, {"safe_open": 1.0}),
                                                  (15.0, 30.0, {"cashier_present": 0.0, "cashier_in_workplace": 0.0})]},
        "neg": [
            {"duration_s": 60.0, "segments": [(10.0, 30.0, {"safe_open": 1.0})]},
            {"duration_s": 60.0, "segments": [(10.0, 30.0, {"safe_open": 1.0}),
                                              (15.0, 18.0, {"cashier_present": 0.0, "cashier_in_workplace": 0.0})]},
        ],
    },
    "V04_door_open_with_valuables": {
        "type_id": "V04", "assessment": "review_observation", "onset": 20.0,
        "pos": {"duration_s": 60.0, "segments": [(20.0, 30.0, {"door_open": 1.0})]},
        "neg": [
            {"duration_s": 60.0, "segments": [(20.0, 23.0, {"door_open": 1.0})]},
            {"duration_s": 60.0, "segments": [(20.0, 30.0, {"door_open": 1.0, "cash_visible": 0.0})]},
        ],
    },
    "V05_phone_in_hand_at_workplace": {
        "type_id": "V05", "assessment": "review_observation", "onset": 20.0,
        "pos": {"duration_s": 60.0, "segments": [(20.0, 26.0, {"phone_in_hand": 1.0})]},
        "neg": [
            {"duration_s": 60.0, "segments": [(20.0, 21.0, {"phone_in_hand": 1.0})]},
            {"duration_s": 60.0, "segments": [(20.0, 26.0, {"phone_in_hand": 1.0, "cashier_in_workplace": 0.0,
                                                             "cash_visible": 0.0})]},
        ],
    },
    "V07_bag_left_unattended": {
        "type_id": "V07", "assessment": "candidate_violation", "onset": 22.0,
        "pos": {"duration_s": 60.0, "segments": [(10.0, 20.0, {"collector_present": 1.0, "n_persons": 2.0}),
                                                  (10.0, 40.0, {"cash_bag_visible": 1.0}),
                                                  (22.0, 40.0, {"cashier_present": 0.0, "cashier_in_workplace": 0.0,
                                                                "n_persons": 0.0})]},
        "neg": [
            {"duration_s": 60.0, "segments": [(10.0, 20.0, {"collector_present": 1.0, "n_persons": 2.0}),
                                              (10.0, 40.0, {"cash_bag_visible": 1.0})]},
        ],
    },
}


# Фаза 2: на каждую гипотезу — нарушение (pos, onset = момент перехода), чистый двойник (twin) и
# неизвестное (unknown: ожидание — ни одного violation_candidate; expect_insufficient — есть ли
# событие insufficient_context). Только синтетика, без реальных сцен.
AWAY = {"cashier_in_workplace": 0.0}
PHASE2_CASES: dict[str, dict[str, Any]] = {
    "H1_left_after_departure/V02": {
        "hypothesis": "H1_left_after_departure", "rule_id": "V02_left_cash_after_departure", "type_id": "V02",
        "detection": "violation_candidate", "onset": 20.0,
        "pos": [(20.0, 35.0, AWAY)],
        "twin": [(20.0, 35.0, {**AWAY, "cash_visible": 0.0, "cash_bag_visible": 0.0, "safe_open": 0.0})],
        "unknown": [(20.0, 35.0, {**AWAY, "cash_visible": None, "cash_bag_visible": None})],
        "expect_insufficient": True,
    },
    "H1_left_after_departure/V03": {
        "hypothesis": "H1_left_after_departure", "rule_id": "V03_left_safe_open_after_departure", "type_id": "V03",
        "detection": "violation_candidate", "onset": 20.0,
        "pos": [(10.0, 35.0, {"safe_open": 1.0}), (20.0, 35.0, {**AWAY, "cash_visible": 0.0})],
        "twin": [(20.0, 35.0, {**AWAY, "cash_visible": 0.0, "safe_open": 0.0})],
        "unknown": [(20.0, 35.0, {**AWAY, "cash_visible": 0.0, "safe_open": None})],
        "expect_insufficient": True,
    },
    "H2_work_with_safe_open": {
        "hypothesis": "H2_work_with_safe_open", "rule_id": "V03_work_with_safe_open", "type_id": "V03",
        "detection": "violation_candidate", "onset": 20.0,
        "pos": [(20.0, 40.0, {"safe_open": 1.0})],
        "twin": [(20.0, 26.0, {"safe_open": 1.0})],
        "twin2": [(15.0, 50.0, {"collector_present": 1.0, "n_persons": 2.0}), (20.0, 40.0, {"safe_open": 1.0})],
        "unknown": [(20.0, 40.0, {"safe_open": None})],
        "expect_insufficient": False,
    },
    "H3_unknown_role_observation": {
        "hypothesis": "H3_unknown_role_observation", "rule_id": "V01_unknown_role_in_cashier_area", "type_id": "V01",
        "detection": "observation", "onset": 20.0,
        "pos": [(20.0, 26.0, {"unknown_person_in_cashier_area": 1.0, "outsider_in_cashier_area": None})],
        "twin": [(20.0, 26.0, {"n_persons": 2.0})],
        "unknown": [(20.0, 26.0, {"outsider_in_cashier_area": None})],
        "expect_insufficient": False,
    },
    "H4_entry_while_valuables_open": {
        "hypothesis": "H4_entry_while_valuables_open", "rule_id": "V04_entry_while_valuables_open", "type_id": "V04",
        "detection": "violation_candidate", "onset": 20.0,
        "pos": [(10.0, 40.0, {"safe_open": 1.0}), (20.0, 40.0, {"door_open": 1.0, "n_persons": 2.0})],
        "twin": [(20.0, 40.0, {"door_open": 1.0, "n_persons": 2.0})],
        "unknown": [(10.0, 40.0, {"safe_open": 1.0}), (20.0, 40.0, {"door_open": None, "n_persons": 2.0})],
        "expect_insufficient": False,
    },
    "H5_second_outsider": {
        "hypothesis": "H5_second_outsider", "rule_id": "V01_second_outsider_during_service", "type_id": "V01",
        "detection": "violation_candidate", "onset": 20.0,
        "pos": [(20.0, 30.0, {"n_persons": 3.0})],
        "twin": [(20.0, 30.0, {"n_persons": 2.0})],
        "unknown": [(20.0, 30.0, {"n_persons": None})],
        "expect_insufficient": False,
    },
    "H6_phone_candidate": {
        "hypothesis": "H6_phone_candidate", "rule_id": "V05_phone_in_hand_at_workplace", "type_id": "V05",
        "detection": "violation_candidate", "onset": 20.0,
        "pos": [(20.0, 26.0, {"phone_in_hand": 1.0})],
        "twin": [(20.0, 26.0, {"phone_to_ear": 0.9})],   # рука у лица без телефона — не сигнал
        "unknown": [(20.0, 26.0, {"phone_in_hand": None})],
        "expect_insufficient": False,
    },
    "H7_food_with_valuables": {
        "hypothesis": "H7_food_with_valuables", "rule_id": "V06_food_drink_with_valuables", "type_id": "V06",
        "detection": "violation_candidate", "onset": 20.0,
        "pos": [(20.0, 26.0, {"eating_drinking": 0.9})],
        "twin": [(20.0, 26.0, {"eating_drinking": 0.9, "cash_visible": 0.0, "cash_bag_visible": 0.0,
                               "safe_open": 0.0})],
        "unknown": [(20.0, 26.0, {"eating_drinking": None})],
        "expect_insufficient": False,
    },
    "H8_personal_items": {
        "hypothesis": "H8_personal_items", "rule_id": "V09_personal_items_on_desk", "type_id": "V09",
        "detection": "violation_candidate", "onset": 20.0,
        "pos": [(0.0, 60.0, {"personal_items_on_desk": 0.0}), (20.0, 40.0, {"personal_items_on_desk": 1.0})],
        "twin": [(0.0, 60.0, {"personal_items_on_desk": 0.0})],
        "unknown": [],
        "expect_insufficient": False,
    },
    "H9_bag_left_after_handover": {
        "hypothesis": "H9_bag_left_after_handover", "rule_id": "V07_bag_left_after_handover", "type_id": "V07",
        "detection": "violation_candidate", "onset": 20.0,
        "pos": [(10.0, 20.0, {"collector_present": 1.0, "n_persons": 2.0}), (10.0, 45.0, {"cash_bag_visible": 1.0})],
        "twin": [(10.0, 20.0, {"collector_present": 1.0, "n_persons": 2.0}), (10.0, 24.0, {"cash_bag_visible": 1.0}),
                 (21.0, 26.0, {"safe_open": 1.0})],
        "unknown": [(10.0, 20.0, {"collector_present": 1.0, "n_persons": 2.0}), (10.0, 20.0, {"cash_bag_visible": 1.0}),
                    (20.0, 45.0, {"cash_bag_visible": None})],
        "expect_insufficient": True,
    },
}

# Чистые сценарии фазы 2 (профиль phase2 не должен дать ни одного события). Из базовых исключён
# safe_opened_by_cashier: по спецификации фазы 2 работа за столом при открытом сейфе — кандидат.
CLEAN_SCENARIOS_PHASE2: dict[str, dict[str, Any]] = {
    k: v for k, v in CLEAN_SCENARIOS.items() if k != "safe_opened_by_cashier"
}
CLEAN_SCENARIOS_PHASE2.update({
    "departure_with_clean_desk": {"duration_s": 60.0, "segments": [
        (20.0, 40.0, {"cashier_in_workplace": 0.0, "cashier_present": 0.0, "cash_visible": 0.0,
                      "cash_bag_visible": 0.0, "safe_open": 0.0})]},
    "brief_safe_work": {"duration_s": 60.0, "segments": [(20.0, 26.0, {"safe_open": 1.0})]},
    "bag_straight_into_safe": {"duration_s": 60.0, "segments": [
        (10.0, 20.0, {"collector_present": 1.0, "n_persons": 2.0}), (10.0, 23.0, {"cash_bag_visible": 1.0}),
        (20.5, 25.0, {"safe_open": 1.0})]},
    "client_opposite": {"duration_s": 60.0, "segments": [(5.0, 55.0, {"n_persons": 2.0})]},
})
