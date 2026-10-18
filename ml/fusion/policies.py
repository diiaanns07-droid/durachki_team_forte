"""Поведенческие политики слияния (спецификация команды, фаза 2). Каждая — отдельная гипотеза под флагом
``policies.<имя>.enabled`` в config.yaml; по умолчанию ВЫКЛЮЧЕНЫ, пока L2 не измерил выигрыш на реальных данных.

Fusion не видит SceneState — политики читают то, что upstream кладёт в ``Event.extra`` (форматы — в README):

* ``v02_after_window`` — «ушла и оставила»: решение по состоянию стола и сейфа в окне ПОСЛЕ ухода
  (``extra.state_after`` или статистика сигналов ``extra.trace.signals`` A09). Ценности убраны и сейф закрыт →
  норма (подсказка); на столе наличные/сумка или сейф открыт → остаётся кандидатом; не видно → неизвестно,
  без изменений (unknown ≠ false: при выключенном свете «стол пуст» не выводится).
* ``v01_client_side`` — клиент напротив кассира со своей стороны — норма (подсказка), если нет второго
  постороннего и ценности не открыты. Сторона неизвестна → без изменений.
* ``negative_evidence`` — явные отрицательные свидетельства (``extra.negative``: сейф закрыт, стол пуст…)
  снижают скор: ``conf *= 1 − weight·max(strength)``; один раз на кластер (источники коррелированы).
"""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence

_CLIENT = {"client", "client_area", "client_side", "customer", "клиент", "клиентская"}
_CASHIER = {"cashier", "cashier_area", "cashier_side", "desk", "safe", "cash_tray", "кассир", "касса"}
_SIGNALS_AFTER = {
    "cash": ("cash_on_desk", "cash_visible", "cash_left"),
    "bag": ("cash_bag_on_desk", "cash_bag_visible", "cash_bag_left"),
    "safe": ("safe_open",),
}


def _num(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, dict):
        known = v.get("known_frac")
        if known is not None and _num(known) is not None and float(_num(known)) < 0.6:
            return None  # сигнал известен лишь в части кадров (окклюзия, свет выключен) — неизвестно
        for k in ("mean", "value", "at_peak", "max"):
            if k in v:
                return _num(v[k])
        return None
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _enabled(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    pcfg = ((cfg.get("policies") or {}).get(name) or {})
    return pcfg if pcfg.get("enabled", False) else {}


def _state_after(events: Sequence[Any]) -> dict[str, Optional[float]]:
    """{cash, bag, safe} → значение 0..1 или None (неизвестно) по самому информативному событию."""
    out: dict[str, Optional[float]] = {"cash": None, "bag": None, "safe": None}
    for e in events:
        extra = getattr(e, "extra", {}) or {}
        sources = []
        if isinstance(extra.get("state_after"), dict):
            sources.append(extra["state_after"])
        trace = extra.get("trace")
        if isinstance(trace, dict) and isinstance(trace.get("signals"), dict):
            sources.append(trace["signals"])
        for src in sources:
            for key, names in _SIGNALS_AFTER.items():
                for n in names:
                    if n in src:
                        val = _num(src[n])
                        if val is not None:
                            out[key] = val if out[key] is None else max(out[key], val)
                        break
    return out


def _side(events: Sequence[Any]) -> Optional[str]:
    sides = set()
    for e in events:
        extra = getattr(e, "extra", {}) or {}
        for k in ("person_side", "side", "outsider_side", "person_zone", "zone"):
            v = extra.get(k)
            if isinstance(v, str) and v.strip():
                v = v.strip().lower().replace("zone:", "")
                sides.add("client" if v in _CLIENT else "cashier" if v in _CASHIER else "unknown")
                break
    # одна согласованная известная сторона; неизвестно или противоречие → None (без изменений)
    return sides.pop() if len(sides) == 1 and "unknown" not in sides else None


def _max_extra(events: Sequence[Any], keys: Sequence[str]) -> Optional[float]:
    vals = []
    for e in events:
        extra = getattr(e, "extra", {}) or {}
        for k in keys:
            if k in extra:
                v = _num(extra[k])
                if v is not None:
                    vals.append(v)
    return max(vals) if vals else None


def _negative_strength(events: Sequence[Any]) -> tuple[float, list[str]]:
    best, facts = 0.0, []
    for e in events:
        neg = (getattr(e, "extra", {}) or {}).get("negative")
        items: list[tuple[str, float]] = []
        if isinstance(neg, dict):
            items = [(str(k), _num(v) or 0.0) for k, v in neg.items()]
        elif isinstance(neg, list):
            for it in neg:
                if isinstance(it, dict):
                    items.append((str(it.get("fact") or it.get("name") or ""), _num(it.get("strength", 1.0)) or 0.0))
                elif isinstance(it, str):
                    items.append((it, 1.0))
        for name, val in items:
            val = max(0.0, min(1.0, val))
            if val > best:
                best = val
            if name and val >= 0.5 and name not in facts:
                facts.append(name)
    return best, facts


def apply_policies(cand: Any, events: Sequence[Any], cfg: dict[str, Any]) -> None:
    """Меняет cand.confidence и добавляет причины в cand.extra["policy_gate"]; след — в cand.extra["policies"]."""
    trace: list[dict[str, Any]] = []
    gate: list[str] = []
    positives = [e for e in events if getattr(e, "positive", True)]

    p = _enabled(cfg, "v02_after_window")
    if p and cand.type_id == "V02" and cand.origin == "events":
        st = _state_after(positives)
        known = {k: v for k, v in st.items() if v is not None}
        pos, neg = float(p.get("remain_min", 0.5)), float(p.get("removed_max", 0.2))
        if not known:
            trace.append({"policy": "v02_after_window", "effect": "unknown", "state": st})
        elif max(known.values()) >= pos:
            trace.append({"policy": "v02_after_window", "effect": "valuables_remain", "state": st})
        elif len(known) == 3 and max(known.values()) <= neg:
            cand.confidence = round(cand.confidence * float(p.get("removed_factor", 0.35)), 4)
            gate.append("valuables_removed")
            trace.append({"policy": "v02_after_window", "effect": "valuables_removed", "state": st})
        else:
            trace.append({"policy": "v02_after_window", "effect": "partial_unknown", "state": st})

    p = _enabled(cfg, "v01_client_side")
    if p and cand.type_id == "V01" and cand.origin == "events":
        side = _side(positives)
        n_out = _max_extra(positives, ("n_outsiders", "n_visitors"))
        opened = _max_extra(positives, ("valuables_open", "safe_open", "cash_open"))
        if side == "client" and (n_out is None or n_out <= 1) and (opened is None or opened < 0.5):
            gate.append("client_side")
            trace.append({"policy": "v01_client_side", "effect": "client_side", "n_outsiders": n_out,
                          "valuables_open": opened})
        else:
            trace.append({"policy": "v01_client_side", "effect": "kept", "side": side, "n_outsiders": n_out,
                          "valuables_open": opened})

    p = _enabled(cfg, "negative_evidence")
    if p and cand.origin == "events":
        strength, facts = _negative_strength(positives)
        if strength > 0:
            factor = 1.0 - float(p.get("weight", 0.5)) * strength
            cand.confidence = round(max(0.0, cand.confidence * factor), 4)
            cand.extra["counter_facts"] = facts
            trace.append({"policy": "negative_evidence", "effect": "downweight", "strength": strength,
                          "factor": round(factor, 4), "facts": facts})

    if trace:
        cand.extra["policies"] = trace
    if gate:
        cand.extra["policy_gate"] = gate
