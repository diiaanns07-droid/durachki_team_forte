"""Конфиг ml.track: значения по умолчанию в коде + переопределение из config.yaml.

Порядок: DEFAULTS -> ml/track/config.yaml (или $SENTINEL_TRACK_CONFIG) -> overrides (аргумент).
Все пороги модуля живут здесь; YAML нужен для тюнинга на реальных данных без правки кода.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

DEFAULTS: dict[str, Any] = {
    "anchor": "center",          # точка человека для зон: center | bottom | torso
    "default_aspect": 1.0,       # width/height, если clip не сообщил размер
    "person_labels": ["person"],
    "tracker": {
        "high_thr": 0.45,        # ByteTrack: "уверенные" детекции (1-й этап сопоставления)
        "low_thr": 0.10,         # ниже — игнор; [low, high) — 2-й этап (только продление треков)
        "new_track_thr": 0.50,   # минимальный score для рождения трека
        "match_thr": 0.80,       # макс. стоимость 1-го этапа (1 - сходство)
        "low_match_thr": 0.60,   # макс. стоимость 2-го этапа (1 - IoU)
        "dist_weight": 0.30,     # вес близости центров в сходстве (остальное IoU)
        "dist_sigma_diag": 1.0,  # масштаб близости центров, в диагоналях бокса
        "gate_diag": 2.0,        # при IoU=0 дальше этого (в диагоналях) — запрет сопоставления
        "track_buffer_s": 2.0,   # сколько секунд держим потерянный трек живым
        "tentative_buffer_s": 0.6,
        "min_hits": 2,           # подтверждение трека (кроме треков с первого кадра)
        "speed_gate": False,        # H-SPEED (ВЫКЛ до замера L2): без перекрытия боксов — не дальше base + v·dt диагоналей
        "speed_gate_base_diag": 0.2,
        "speed_gate_diag_s": 1.5,
        "lost_cost_penalty": 0.1,  # штраф потерянным трекам при выборе пар: подмена ID при проходе перед сидящим (0 = как было)
        "kf_tau_s": 1.0,         # затухание скорости в фильтре Калмана
        "kf_q_pos": 0.5,
        "kf_q_vel": 1.0,
        "kf_r": 0.05,
        "fill_gaps_s": 0.0,      # 0 = не дорисовывать боксы в пропусках (в Track только реальные наблюдения)
        "min_track_obs": 2,
        "label_groups": {"safe": ["safe", "safe_open"], "door": ["door", "door_open"],
                         "bag": ["cash_bag", "bag"]},
        "group_label": {"safe": "safe", "door": "door"},
        "prefer_label": {"bag": {"label": "cash_bag", "min_frac": 0.3}},
    },
    "stitch": {
        "enabled": True,
        "short_gap_s": 2.5,          # короткий разрыв: окклюзия/пропуск детектора
        "long_gap_s": 180.0,         # длинный разрыв: ушёл и вернулся (только с внешностью)
        "base_dist_diag": 0.8,
        "max_speed_diag_s": 1.5,
        "size_ratio_max": 2.2,
        "app_thr_short": 0.50,       # Бхаттачарья по гистограмме одежды
        "app_thr_long": 0.32,
        "long_requires_appearance": True,
        "long_home_diag": 1.5,       # вернулся «на своё место» (медианы позиций)
        "long_entry_diag": 1.0,      # или появился там же, где исчез (дверь)
        "static_gap_s": 15.0,        # статичные предметы: склейка мерцания
        "static_iou": 0.3,
    },
    "appearance": {
        # Спецификация команды: цвет одежды/яркость — не признаки (одежда и свет меняются).
        # Гистограмма одежды оставлена как гипотеза H-APP для замера L2: по умолчанию ВЫКЛ.
        "enabled": False,
        "samples_per_track": 6,
        "max_frames": 150,
        "budget_s": 4.0,
        "bins": [8, 6, 4],
        "min_box_px": 10,
        # кроп одежды: без верхних 25% бокса (голова), без краёв — лица не используются
        "crop": {"x_margin": 0.2, "y_top": 0.25, "y_bottom": 0.9},
    },
    "roles": {
        "early_s": 3.0,                  # «присутствует с начала» / «пришёл позже»
        "min_presence_s": 1.0,           # короче — роль unknown
        "near_desk_diag": 0.9,
        "stationary_speed_diag_s": 0.35,
        # w_start = 0: роль НЕ зависит от порядка появления (кассир может войти через дверь в первые секунды)
        "w_desk": 0.40, "w_start": 0.0, "w_still": 0.20, "w_presence": 0.15, "w_safe_side": 0.10,
        "w_desk_center": 0.15,           # ближе к центру объектов стола (сидит перед ноутбуком/счётчиком)
        "w_seated": 0.45,                # сидит у стола со стороны кассы (относительно других людей ролика)
        "w_counter": 0.15,               # сидит у счётчика банкнот
        "speed_window_s": 1.0,
        "seated_speed_diag_s": 0.25,     # «сидит/стоит на месте»
        "min_seated_s": 2.0,             # минимум «сидит у стола со стороны кассы», чтобы быть кандидатом в кассиры
        "near_counter_diag": 0.8,
        "side_tol_diag": 0.1,
        "no_side_cashier_conf_cap": 0.7,     # сторона кассы не подтверждена сейфом
        "behavior_cashier_conf_cap": 0.65,   # кассир по собственному поведению после разрыва трека
        "min_cashier_score": 0.35,
        "fallback_cashier_conf_cap": 0.45,   # без объектов стола роль кассира — слабая
        "fallback_visitor_conf_cap": 0.6,
        "overlap_tol_s": 0.6,
        "app_same_thr": 0.35,
        "app_diff_thr": 0.50,
        "return_min_in_area": 0.5,
        "return_same_seat_diag": 0.8,
        "position_only_return_conf": 0.4,    # без внешности возврат «на то же место» -> unknown
        "shift_min_s": 20.0,
        "bag_near_expand": 1.3,
        "bag_contact_full_s": 2.0,
        "generic_bag_weight": 0.5,
        "bag_arrival_window_s": 4.0,
        "collector_thr": 0.5,
        "collector_conf_cap": 0.75,
        "w_bag": 0.5, "w_entered": 0.3, "w_bag_arrived": 0.2,
        "companion_window_s": 4.0,
        "companion_dist_diag": 2.5,
        "second_person_role": "unknown",     # второй человек в зоне кассира: не «посторонний» автоматически
    },
    "queries": {
        "min_role_conf": 0.5,            # роль ниже — передаётся как unknown (fallback-кассир <= 0.45)
        "min_zone_conf": 0.35,           # зона ниже — «не определена» (fallback-зоны <= 0.3)
        "max_dt": 0.6,
        "start_grace_s": 0.5,            # кто в кадре с начала — не «вошёл»
        "vis_grace_s": 1.0,              # появление сразу после потери видимости — не вход
        "entry_requires_boundary": True,  # вход = появление у двери/края; в глубине кадра — не вход
        "entry_window_s": 1.0,
        "edge_margin": 0.06,
        "door_expand": 1.3,
        "safe_expand": 1.3,               # «у ценностей» = в расширенной зоне сейфа
        "low_quality_min_s": 0.6,         # трек короче — не считается человеком в n_outsiders/входах
        "min_gap_s": 0.6,
        "min_loss_s": 0.6,                # потеря видимости (все статичные предметы пропали)
        "smoothing": "off",               # H-SMOOTH: off | hysteresis — по умолчанию ВЫКЛ до замера L2
        "min_state_s": 0.4,               # гистерезис состояний (режим off)
        "merge_gap_s": 0.0,
        "smooth_min_state_s": 2.0,        # режим hysteresis
        "smooth_merge_gap_s": 3.0,
    },
    "visibility": {
        "min_loss_s": 0.6,
        "bridge_tracks": False,           # H-VIS: склейка треков через потерю видимости по положению (ВЫКЛ)
        "bridge_dist_diag": 1.0,
        "bridge_max_s": 20.0,
    },
    "zones": {
        "desk_labels": ["laptop", "money_counter"],
        "desk_static_labels": ["cell_phone", "keys", "document"],
        "static_min_presence_frac": 0.4,
        "static_max_motion_diag": 0.15,
        "min_object_presence_s": 0.6,
        "desk_expand": 1.3,
        "desk_min_size_person": 0.8,
        "cashier_shrink": 0.6,
        "cashier_expand": 1.15,
        "cashier_keep_pct": 90.0,
        "cashier_max_dist_diag": 1.5,
        "cashier_band": True,            # cashier_area = место кассира ∪ «отпечаток» стола на его стороне
        "client_expand": 1.15,
        "client_min_points": 5,
        "client_max_dist_diag": 2.5,
        "fallback_zone_conf_cap": 0.3,
        "safe_expand": 1.15,
        "door_expand": 1.1,
        "door_cluster_diag": 1.2,
        "door_min_points": 2,
        "edge_margin": 0.06,             # точка у края кадра: вероятно ушёл из поля зрения
        "cash_cluster_diag": 1.5,
        "cash_expand": 1.25,
        "max_vertices": 12,
    },
}


def deep_merge(base: dict, upd: Optional[dict]) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: str) -> dict:
    """YAML с кэшем по (путь, mtime, размер): правка файла в долгоживущем воркере подхватывается."""
    try:
        st = os.stat(path)
    except OSError:
        return {}
    return _read_yaml_cached(path, st.st_mtime_ns, st.st_size)


@lru_cache(maxsize=16)
def _read_yaml_cached(path: str, mtime_ns: int, size: int) -> dict:
    try:
        import yaml
    except ImportError:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def config_path(path: Optional[str] = None) -> str:
    return path or os.environ.get("SENTINEL_TRACK_CONFIG") or str(Path(__file__).with_name("config.yaml"))


def load_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> dict:
    """Собрать конфиг. Неизвестные ключи YAML не ломают загрузку (проверяются тестом)."""
    cfg = deep_merge(DEFAULTS, _read_yaml(config_path(path)))
    return deep_merge(cfg, overrides)


def config_fingerprint(cfg: Optional[dict] = None) -> dict[str, Any]:
    """Хэш ДЕЙСТВУЮЩЕГО конфига (в т.ч. внешнего $SENTINEL_TRACK_CONFIG) — для ключа кэша A03."""
    c = cfg if cfg is not None else load_config()
    blob = json.dumps(c, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return {"config_sha": hashlib.sha1(blob).hexdigest()[:12], "config_path": config_path(),
            "config_from_env": bool(os.environ.get("SENTINEL_TRACK_CONFIG"))}
