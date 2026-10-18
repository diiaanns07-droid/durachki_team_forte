"""ml.seqrules — правила по последовательностям наблюдений (профиль pose_rules, L2).

Цепочка: кадр с реальным PTS → люди/предметы (ml.detect) → track_id (ml.track) → суставы (ml.pose) →
зоны и взаимодействия → эпизоды во времени → события контракта (source="rules_seq") + наблюдения для UI.

    from ml.seqrules import run
    events, observations = run(clip, dets, tracks, zones, poses, catalog)

Правила: А — мобильный телефон у запястья/головы конкретного человека (V05); Б — дверца сейфа распознана
открытой, а в рабочей зоне никого после наблюдаемого ухода (V03); В — вход в рабочую зону (V01, только
наблюдение). Пороги — технические параметры (config.yaml), не нормы.
"""
from __future__ import annotations

__version__ = "1.0.0"

from .api import run  # noqa: E402
from .config import config_hash, load_config  # noqa: E402


def model_info() -> dict:
    """Для ключа кэша A03: итоговый конфиг (включая файл из SENTINEL_SEQRULES_CONFIG)."""
    return {"config_hash": config_hash(load_config())}


__all__ = ["run", "load_config", "model_info", "__version__"]
