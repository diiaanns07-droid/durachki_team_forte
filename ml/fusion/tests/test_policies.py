"""Поведенческие политики фазы 2 (по умолчанию выключены): «ушла и оставила», клиент напротив, контр-факты."""
from __future__ import annotations

import pytest

from ml.fusion.catalog import resolve_catalog
from ml.fusion.config import load_config, load_thresholds
from ml.fusion.core import fuse_scores

from .conftest import ev

LEFT = ["кассир вышел из зоны рабочего места"]


def cfg_with(**flags):
    cfg = load_config()
    for name, on in flags.items():
        cfg["policies"][name]["enabled"] = on
    return cfg


def run(events, cfg, catalog):
    return fuse_scores(events, [], 60.0, resolve_catalog(catalog)[0], cfg, load_thresholds())


def v02(state=None, trace=None, conf=0.8):
    extra = {}
    if state is not None:
        extra["state_after"] = state
    if trace is not None:
        extra["trace"] = {"signals": trace}
    return ev("V02", t=10.0, conf=conf, facts=LEFT, extra=extra)


def test_policies_off_by_default(catalog):
    cfg = load_config()
    assert not any(p["enabled"] for p in cfg["policies"].values())
    out = run([v02({"cash_visible": 0.0, "cash_bag_visible": 0.0, "safe_open": 0.0})], cfg, catalog)
    assert out.reported and "policies" not in out.candidates[0].extra


def test_v02_valuables_removed_is_norm(catalog):
    out = run([v02({"cash_visible": 0.05, "cash_bag_visible": 0.0, "safe_open": 0.1})],
              cfg_with(v02_after_window=True), catalog)
    c = out.candidates[0]
    assert not c.reported and "valuables_removed" in c.gate and c.confidence < 0.8


def test_v02_cash_left_stays_candidate(catalog):
    out = run([v02(trace={"cash_visible": {"mean": 0.9, "max": 1.0}, "safe_open": {"mean": 0.0}})],
              cfg_with(v02_after_window=True), catalog)
    c = out.candidates[0]
    assert c.reported and c.extra["policies"][0]["effect"] == "valuables_remain"


def test_v02_unknown_state_is_not_false(catalog):
    # свет выключен: сигналы неизвестны → без изменений (unknown ≠ «стол пуст»)
    out = run([v02({"cash_visible": None, "cash_bag_visible": None, "safe_open": None})],
              cfg_with(v02_after_window=True), catalog)
    c = out.candidates[0]
    assert c.reported and c.confidence == pytest.approx(0.8) and c.extra["policies"][0]["effect"] == "unknown"
    out = run([v02({"cash_visible": 0.0, "safe_open": None})], cfg_with(v02_after_window=True), catalog)
    assert out.reported  # известно не всё → «убрано» не выводится


def test_v01_client_side_suppressed(catalog):
    e = ev("V01", t=10.0, extra={"person_side": "client_area", "n_outsiders": 1})
    out = run([e], cfg_with(v01_client_side=True), catalog)
    assert not out.reported and out.candidates[0].gate == ["client_side"]


@pytest.mark.parametrize("extra", [
    {"person_side": "cashier_area"},                              # посторонний на стороне кассира
    {"person_side": "client", "n_outsiders": 2},                  # второй посторонний во время обслуживания
    {"person_side": "client", "valuables_open": True},            # вход при открытых ценностях
    {},                                                           # сторона неизвестна → без изменений
])
def test_v01_kept(catalog, extra):
    out = run([ev("V01", t=10.0, extra=extra)], cfg_with(v01_client_side=True), catalog)
    assert out.reported


def test_negative_evidence_downweights_once(catalog):
    events = [ev("V03", t=10.0, conf=0.8, facts=["дверца сейфа открыта"], extra={"negative": {"сейф закрыт": 0.9}}),
              ev("V03", t=10.5, conf=0.7, facts=["рядом нет кассира"], extra={"negative": [{"fact": "стол пуст",
                                                                                               "strength": 0.6}]})]
    out = run(events, cfg_with(negative_evidence=True), catalog)
    c = out.candidates[0]
    assert c.confidence == pytest.approx(0.8 * (1 - 0.5 * 0.9), abs=1e-4)
    assert c.extra["counter_facts"] == ["сейф закрыт", "стол пуст"]


def test_v02_low_known_fraction_is_unknown(catalog):
    trace = {"cash_visible": {"mean": 0.0, "known_frac": 0.3}, "cash_bag_visible": {"mean": 0.0, "known_frac": 0.9},
             "safe_open": {"mean": 0.0, "known_frac": 0.9}}
    out = run([v02(trace=trace)], cfg_with(v02_after_window=True), catalog)
    c = out.candidates[0]
    assert c.reported and c.extra["policies"][0]["effect"] == "partial_unknown"
