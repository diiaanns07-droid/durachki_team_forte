"""discover.mode=states (L2): VLM по кадрам отвечает на короткие вопросы о состоянии сцены
(кто на стороне кассира, открыт ли сейф, что лежит на столе, что в руках), а нарушения выводятся
временной логикой по этим состояниям. Таймкод — момент перехода (начало нарушения).

Кадры и ответы модели остаются в процессе (в extra кладём только флаги и агрегаты).
"""
from __future__ import annotations

import logging
import math
from typing import Any, Optional

from ._contract import Event
from .backends import VLMBackend
from .budget import Budget, VLMLimits
from .config import load_config, preset_cfg
from .frames import clip_duration, get_frame, prepare_images
from .prompting import retry_suffix
from .runtime import call_json, ensure_loaded, fit_images, get_backend

log = logging.getLogger(__name__)

# ключ -> (вопрос, максимум). 1 — да/нет, 3 — число людей.
QUESTIONS: dict[str, tuple[str, int]] = {
    "cs": ("сколько людей находится на стороне кассира (за кассовым столом со стороны кассы или у сейфа)?", 3),
    "cl": ("сколько людей находится с клиентской стороны стола или в остальной части комнаты?", 3),
    "so": ("дверца металлического шкафа-сейфа открыта?", 1),
    "as": ("кто-то стоит вплотную у сейфа и работает с ним?", 1),
    "ca": ("на столе открыто лежат наличные (пачки или купюры)?", 1),
    "gb": ("на столе лежит зелёная (инкассаторская) сумка?", 1),
    "ph": ("человек на стороне кассира держит в руке мобильный телефон (не стационарный телефон на столе)?", 1),
    "ed": ("человек на стороне кассира ест или пьёт (подносит еду, кружку или бутылку ко рту)?", 1),
    "pi": ("на кассовом столе лежит верхняя одежда (куртка, пальто) или личная сумка/рюкзак?", 1),
}
KEYS = list(QUESTIONS)

SYSTEM = ("Ты анализируешь кадры потолочной камеры банковской кассы. Отвечай только по тому, что видно на "
          "кадрах. Если не видно или сомневаешься — 0. Ответ — только JSON, без пояснений.")


def _prompt(times: list[float]) -> str:
    """Компактный ответ (строка из чисел на кадр) — в ~3 раза меньше токенов, чем JSON-объекты."""
    n = len(times)
    qs = "\n".join(f"{j}) {q}" + (" (число 0..3)" if mx > 1 else " (0 или 1)")
                   for j, (_k, (q, mx)) in enumerate(QUESTIONS.items(), start=1))
    tl = ", ".join(f"{t:.1f}" for t in times)
    return (f"Перед тобой {n} кадр(ов) по порядку; время кадров (с): {tl}.\n"
            "На кадрах касса: стол кассира, рядом металлический шкаф-сейф с дверцей, напротив — место клиента.\n"
            f"Для каждого кадра ответь на {len(QUESTIONS)} вопросов:\n{qs}\n"
            f'Ответь одним JSON: {{"f": [[...], ...]}} — {n} списков по {len(QUESTIONS)} чисел, '
            "по одному списку на кадр в порядке кадров, числа в порядке вопросов.")


def _coerce(v: Any, mx: int) -> int:
    try:
        v = int(round(float(v))) if not isinstance(v, bool) else int(v)
    except (TypeError, ValueError):
        v = 1 if str(v).strip().lower() in ("да", "yes", "true") else 0
    return max(0, min(mx, v))


def _normalize(data: dict[str, Any], n: int) -> tuple[Optional[dict[str, Any]], str]:
    raw = None
    if isinstance(data, dict):
        raw = data.get("f", data.get("frames"))
    if not isinstance(raw, list) or not raw:
        return None, "missing_frames"
    out: dict[int, dict[str, int]] = {}
    for k, fr in enumerate(raw, start=1):
        if k > n:
            break
        if isinstance(fr, list) and len(fr) >= len(KEYS):
            out[k] = {key: _coerce(fr[j], QUESTIONS[key][1]) for j, key in enumerate(KEYS)}
        elif isinstance(fr, dict):  # на случай, если модель ответила объектами
            out[k] = {key: _coerce(fr.get(key, 0), QUESTIONS[key][1]) for key in KEYS}
    if not out:
        return None, "no_valid_frames"
    return {"frames": out}, "ok"


def sample_times(duration: float, cfg: dict[str, Any]) -> list[float]:
    """1 к/с для коротких роликов, реже для длинных (бюджет 300 с у жюри)."""
    if duration <= 0:
        return []
    fps = 1.0 if duration <= 40 else (0.67 if duration <= 90 else 0.5)
    fps = float(cfg.get("fps", fps))
    step = 1.0 / fps
    n = max(2, int(math.floor(duration / step)))
    return [round(min(duration - 0.05, (k + 0.5) * step), 2) for k in range(n)]


def _run(seq: list[tuple[float, dict[str, int]]], key: str, pred) -> list[tuple[int, int]]:
    """Отрезки [i0, i1] подряд идущих кадров, где pred(state) истинно."""
    runs, start = [], None
    for i, (_t, s) in enumerate(seq):
        if pred(s):
            start = i if start is None else start
        elif start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(seq) - 1))
    return runs


def _ev(type_id: str, t0: float, t1: float, tp: float, conf: float, facts: list[str], info: dict) -> Event:
    return Event(type_id=type_id, t_start=round(t0, 2), t_end=round(t1, 2), t_peak=round(tp, 2),
                 confidence=round(min(0.95, conf), 3), source="vlm", facts=facts,
                 extra={"vlm": {"status": "ok", "stage": "states", "agreement": "not_checked"},
                        "vlm_states": info})


def infer_events(seq: list[tuple[float, dict[str, int]]]) -> list[Event]:
    """Временная логика по последовательности (t, состояние)."""
    ev: list[Event] = []
    if len(seq) < 2:
        return ev
    t = [x[0] for x in seq]
    S = [x[1] for x in seq]
    n = len(seq)
    mid = lambda a, b: (t[a] + t[b]) / 2.0  # noqa: E731

    # 1) ушла с рабочего места, а на столе ценности / сейф открыт
    for k in range(1, n):
        if S[k - 1]["cs"] >= 1 and S[k]["cs"] == 0 and (k + 1 >= n or S[k + 1]["cs"] == 0):
            win = range(k, min(n, k + 4))
            cash = sum(S[j]["ca"] for j in win)
            bag = sum(S[j]["gb"] for j in win)
            safe = sum(S[j]["so"] for j in win)
            tl = mid(k - 1, k)
            info = {"rule": "left_workplace", "frames": len(win), "cash": cash, "bag": bag, "safe": safe}
            if bag:
                ev.append(_ev("V07" if not cash else "V02", tl, t[min(n - 1, k + 3)], tl, 0.55 + 0.12 * min(3, bag),
                              [f"Около {tl:.0f} с кассир покидает рабочее место; на столе остаётся зелёная (инкассаторская) сумка."], info))
            elif cash:
                ev.append(_ev("V02", tl, t[min(n - 1, k + 3)], tl, 0.55 + 0.12 * min(3, cash),
                              [f"Около {tl:.0f} с кассир покидает рабочее место; на столе остаются наличные."], info))
            if safe:
                ev.append(_ev("V03", tl, t[min(n - 1, k + 3)], tl, 0.55 + 0.12 * min(3, safe),
                              [f"Около {tl:.0f} с кассир покидает рабочее место; дверца сейфа остаётся открытой."], info))

    # 2) сейф открыт, у сейфа никого, кассир на месте (работает спиной к открытому сейфу)
    for a, b in _run(seq, "so", lambda s: s["so"] == 1 and s["as"] == 0 and s["cs"] >= 1):
        if t[b] - t[a] >= 2.5 or b - a >= 2:
            tp = mid(a - 1, a) if a > 0 else t[a]
            ev.append(_ev("V03", t[a], t[b], tp, 0.5 + 0.1 * min(4, b - a + 1),
                          [f"С {t[a]:.0f} с дверца сейфа открыта, у сейфа никого нет, кассир находится за столом."],
                          {"rule": "safe_open_unattended", "frames": b - a + 1}))

    # 3) посторонний на стороне кассира: двое и больше на стороне кассы
    for a, b in _run(seq, "cs", lambda s: s["cs"] >= 2):
        if b - a >= 1:
            tp = mid(a - 1, a) if a > 0 else t[a]
            ev.append(_ev("V01", t[a], t[b], tp, 0.5 + 0.1 * min(4, b - a + 1),
                          [f"С {t[a]:.0f} с на стороне кассира одновременно находятся два человека."],
                          {"rule": "two_at_cashier_side", "frames": b - a + 1}))

    # 4) вход человека, когда ценности открыты; второй посетитель при обслуживании
    for k in range(1, n):
        before = S[k - 1]["cs"] + S[k - 1]["cl"]
        now = S[k]["cs"] + S[k]["cl"]
        nxt = S[k + 1]["cs"] + S[k + 1]["cl"] if k + 1 < n else now
        if now > before and nxt >= now:
            exposed = S[k - 1]["ca"] or S[k]["ca"] or S[k - 1]["so"] or S[k]["so"]
            second = S[k]["cl"] >= 2
            if exposed or second:
                tp = mid(k - 1, k)
                what = "наличные на столе" if (S[k - 1]["ca"] or S[k]["ca"]) else ("открыт сейф" if exposed else "")
                fact = (f"Около {tp:.0f} с в помещение входит человек, когда {what}." if exposed else
                        f"Около {tp:.0f} с во время обслуживания в помещение входит второй посетитель.")
                ev.append(_ev("V04" if exposed else "V01", tp, t[min(n - 1, k + 2)], tp, 0.55 + (0.15 if exposed and second else 0.0),
                              [fact], {"rule": "entry_exposed" if exposed else "second_visitor"}))

    # 5) мобильный, еда/питьё, личные вещи
    for key, tid, mn, txt in (("ph", "V05", 2, "держит в руке мобильный телефон"),
                              ("ed", "V06", 1, "ест или пьёт на рабочем месте"),
                              ("pi", "V09", 2, "на кассовом столе лежит верхняя одежда или личная сумка")):
        for a, b in _run(seq, key, lambda s, k_=key: s[k_] == 1):
            if b - a + 1 >= mn:
                tp = mid(a - 1, a) if a > 0 else t[a]
                subj = "Человек на рабочем месте кассира " if key != "pi" else ""
                ev.append(_ev(tid, t[a], t[b], tp, 0.5 + 0.1 * min(4, b - a + 1),
                              [f"С {t[a]:.0f} с {subj}{txt}.".replace("с Ч", "с ч")],
                              {"rule": key, "frames": b - a + 1}))
    return _merge(ev)


def _merge(events: list[Event], gap: float = 3.0) -> list[Event]:
    events = sorted(events, key=lambda e: (e.type_id, e.t_start))
    out: list[Event] = []
    for e in events:
        last = out[-1] if out else None
        if last is not None and last.type_id == e.type_id and e.t_start - last.t_end <= gap:
            last.t_end = max(last.t_end, e.t_end)
            last.confidence = max(last.confidence, e.confidence)
            continue
        out.append(e)
    return sorted(out, key=lambda e: -e.confidence)


def states_events(clip: Any, budget_s: float, *, cfg: Optional[dict[str, Any]] = None,
                  backend: Optional[VLMBackend] = None, limits: Optional[dict[str, Any]] = None) -> list[Event]:
    cfg = cfg or load_config()
    scfg = (cfg.get("discover") or {}).get("states") or {}
    if backend is None:
        backend, _reason = get_backend(cfg)
        if backend is None:
            return []
    duration = clip_duration(clip)
    times = sample_times(duration, scfg)
    per_call = int(scfg.get("frames_per_call", 4))
    n_calls = math.ceil(len(times) / per_call)
    lim_over = {"max_calls": n_calls + 2, "max_frames_per_call": per_call,
                "max_visual_tokens_per_call": 2400, "max_visual_tokens_total": 2400 * (n_calls + 2),
                "max_output_tokens_per_call": 300, "max_output_tokens_total": 300 * (n_calls + 2),
                "max_retries_total": 3, **(limits or {})}
    lim = VLMLimits.from_cfg(cfg, "discover", lim_over)
    budget = Budget(float(budget_s), lim)
    rcfg = cfg.get("runtime") or {}
    ok, _reason = ensure_loaded(backend, budget, float(rcfg.get("lazy_load_min_budget_s", 0.0)),
                                float(rcfg.get("load_retry_cooldown_s", 300.0)))
    if not ok:
        return []
    budget.px_per_token = int(preset_cfg(cfg, "transformers").get("px_per_token", 28))
    fcfg = dict(cfg.get("frames") or {})
    fcfg["max_side"] = int(scfg.get("max_side", fcfg.get("max_side", 448)))
    lang, version = str(cfg.get("lang", "ru")), str(cfg.get("prompt_version", "v1"))
    seq: list[tuple[float, dict[str, int]]] = []
    calls_ok = 0
    for c in range(n_calls):
        chunk = times[c * per_call:(c + 1) * per_call]
        frames, ft = [], []
        for tt in chunk:
            f = get_frame(clip, tt)
            if f is not None:
                frames.append(f)
                ft.append(tt)
        if not frames or not budget.can_call(len(frames)):
            continue
        images = fit_images(prepare_images(frames, ft, None, None, fcfg), len(frames),
                            lim.max_visual_tokens_per_call, budget.px_per_token)
        nf = len(ft)
        outcome = call_json(backend, images, SYSTEM, _prompt(ft), budget,
                            normalize=lambda d, n=nf: _normalize(d, n),
                            max_new_tokens=int(scfg.get("max_new_tokens", 160)),
                            retry_suffix=retry_suffix(lang, version), temperature=0.0)
        if not outcome.ok:
            continue
        calls_ok += 1
        for i, st in (outcome.data.get("frames") or {}).items():
            seq.append((ft[int(i) - 1], st))
    seq.sort(key=lambda x: x[0])
    events = infer_events(seq)[: int(scfg.get("max_events", 4))]
    summary = budget.summary()
    summary.update({"frames_planned": len(times), "frames_answered": len(seq), "calls_ok": calls_ok})
    for e in events:
        e.extra["vlm"]["budget"] = summary
    log.info("vlm states: %d/%d frames, %d events, %.1fs", len(seq), len(times), len(events), budget.elapsed())
    return sorted(events, key=lambda e: e.t_peak)
