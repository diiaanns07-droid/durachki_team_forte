"""ml.track (A06): трекинг людей и предметов, роли без лиц, авто-зоны без хардкода координат.

Публичное API (контракт v1):
    track(clip, dets) -> tuple[list[Track], list[Zone]]
Дополнительно:
    track_full(clip, dets, cfg=None) -> TrackResult(tracks, zones, info)   # + provenance
    events: entries_exits, dwell, proximity, gaps, presence_state, person_signals
    queries: TrackQueries(tracks, zones, info).cashier_left_workplace(t) / cashier_left_room(t) /
             person_entered(t, role) / n_outsiders(t) / outsider_at_cashier_side(t) / events()
"""
from __future__ import annotations

from .api import TrackResult, track, track_full
from .events import dwell, entries_exits, gaps, person_signals, presence_state, proximity
from .queries import Answer, TrackEvent, TrackQueries

__all__ = ["track", "track_full", "TrackResult", "entries_exits", "dwell", "proximity", "gaps",
           "presence_state", "person_signals", "TrackQueries", "Answer", "TrackEvent", "model_info"]


def model_info() -> dict:
    """Для идентичности кэша стадии в A03 (module_identity): хэш действующего конфига.

    Отпечаток кода A03 видит только файлы пакета; внешний $SENTINEL_TRACK_CONFIG без этого
    не менял бы ключ кэша. Моделей и весов у ml.track нет (checkpoint не указывается).
    """
    from .config import config_fingerprint

    return config_fingerprint()
