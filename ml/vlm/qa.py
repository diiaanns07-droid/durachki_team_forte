"""verify.mode=qa: сначала нейтральные наблюдения (вероятности из логитов), потом сопоставление с кандидатом в коде.

1. Наблюдения. Для типа кандидата из qa_spec.yaml берутся вопросы и кадры (t−2, t, t+2, t+4 от anchor). Каждый
   вопрос — один forward без генерации: softmax логитов первого токена ответа по вариантам «Да/Нет/Не видно»
   (или 0/1/2/3/«Не видно» для счёта людей). К кадру добавляется кроп зоны (стол/сейф/дверь) из зон A06 или
   боксов события; зон нет — только полный кадр (crop_used=false). Боксы события НЕ рисуются: рамка подсказала
   бы модели, что что-то не так.
2. Сопоставление. decide_<тип>() по спецификации команды превращает вероятности в p_support / p_contra.
   «Не видно» и непрозвучавшие вопросы уменьшают обе величины: неизвестность ≠ «нет нарушения».
3. Факты — шаблонные русские фразы из уверенных ответов (без свободного текста модели).
"""
from __future__ import annotations

import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

from ._contract import Event
from .backends import VLMBackend
from .budget import Budget, visual_tokens
from .frames import get_frame_ts, resize_max_side, to_pil_rgb
from .runtime import gpu_guard

log = logging.getLogger(__name__)

QA_SPEC_PATH = Path(__file__).resolve().parent / "qa_spec.yaml"
Answers = dict[tuple[str, str], dict[str, float]]  # (вопрос, кадр) -> {вариант: p, "_mass": m}

# Шаблоны фактов: (вопрос, вариант) -> фраза; {t} — время кадра.
FACT_TEMPLATES: dict[tuple[str, str], str] = {
    ("cashier_present", "yes"): "Кассир находится на рабочем месте ({t:.1f} с)",
    ("cashier_present", "no"): "Кассира на рабочем месте не видно ({t:.1f} с)",
    ("cash_on_desk", "yes"): "На кассовом столе видны наличные ({t:.1f} с)",
    ("cash_on_desk", "no"): "Наличных на кассовом столе не видно ({t:.1f} с)",
    ("bag_on_desk", "yes"): "На кассовом столе видна инкассаторская сумка ({t:.1f} с)",
    ("bag_on_desk", "no"): "Инкассаторской сумки на кассовом столе не видно ({t:.1f} с)",
    ("safe_open", "yes"): "Дверца сейфа открыта ({t:.1f} с)",
    ("safe_open", "no"): "Дверца сейфа закрыта ({t:.1f} с)",
    ("phone_in_hand", "yes"): "Кассир держит в руке предмет, похожий на мобильный телефон ({t:.1f} с)",
    ("phone_in_hand", "no"): "Мобильного телефона в руке кассира не видно ({t:.1f} с)",
    ("eating_drinking", "yes"): "Кассир подносит ко рту еду или напиток ({t:.1f} с)",
    ("eating_drinking", "no"): "Еды или напитка у рта кассира не видно ({t:.1f} с)",
    ("personal_items", "yes"): "На кассовом столе лежат личные вещи (куртка, сумка или пакет) ({t:.1f} с)",
    ("personal_items", "no"): "Личных вещей на кассовом столе не видно ({t:.1f} с)",
    ("door_open", "yes"): "Дверь в кассовое помещение открыта ({t:.1f} с)",
    ("door_open", "no"): "Дверь в кассовое помещение закрыта ({t:.1f} с)",
    ("bag_unattended", "yes"): "Инкассаторская сумка лежит, её никто не держит ({t:.1f} с)",
    ("bag_unattended", "no"): "Инкассаторская сумка в руках у человека или её не видно лежащей ({t:.1f} с)",
    ("hands_visible", "yes"): "Руки кассира с деньгами видны ({t:.1f} с)",
    ("hands_visible", "no"): "Руки кассира с деньгами не видны камере ({t:.1f} с)",
}
COUNT_FACT = {
    "cashier_side_count": "На стороне кассира видно человек: {n} ({t:.1f} с)",
    "client_side_count": "Со стороны клиентов видно человек: {n} ({t:.1f} с)",
}


@lru_cache(maxsize=4)
def _load_spec(path: str) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_spec(path: Optional[str] = None) -> dict[str, Any]:
    return _load_spec(str(path or QA_SPEC_PATH))


def supported_types(spec: Optional[dict[str, Any]] = None) -> set[str]:
    return set((spec or load_spec()).get("types") or {})


# --------------------------------------------------------------------------- доступ к ответам

def _get(a: Answers, q: str, at: str, key: str) -> float:
    """p(key) для вопроса на кадре; не задан/не в формате -> 0 (это «неизвестно», а не «нет»)."""
    return float((a.get((q, at)) or {}).get(key, 0.0))


def _yes(a: Answers, q: str, at: str) -> float:
    return _get(a, q, at, "yes")


def _no(a: Answers, q: str, at: str) -> float:
    return _get(a, q, at, "no")


def _asked(a: Answers, q: str) -> list[str]:
    return [at for (qq, at) in a if qq == q]


def _p_count_ge(a: Answers, q: str, at: str, n: int) -> float:
    d = a.get((q, at)) or {}
    return float(sum(v for k, v in d.items() if k.isdigit() and int(k) >= n))


def _p_count_le(a: Answers, q: str, at: str, n: int) -> float:
    d = a.get((q, at)) or {}
    return float(sum(v for k, v in d.items() if k.isdigit() and int(k) <= n))


def _max(xs: list[float]) -> float:
    return max(xs) if xs else 0.0


def _min(xs: list[float]) -> float:
    return min(xs) if xs else 0.0


def _any_yes(a: Answers, q: str) -> float:
    return _max([_yes(a, q, at) for at in _asked(a, q)])


def _all_no(a: Answers, q: str) -> float:
    ats = _asked(a, q)
    return _min([_no(a, q, at) for at in ats]) if ats else 0.0


# --------------------------------------------------------------------------- сопоставление по спецификации

def decide_V02(a: Answers) -> tuple[float, float]:
    """Уход — норма, если ценности убраны и сейф закрыт; нарушение — если после ухода на столе наличные/сумка
    или сейф открыт. Решение — по состоянию стола и сейфа в окне ПОСЛЕ ухода."""
    absent = _no(a, "cashier_present", "after2")
    stayed = _yes(a, "cashier_present", "after2")
    left_open = [_yes(a, q, "after2") for q in ("cash_on_desk", "bag_on_desk", "safe_open") if (q, "after2") in a]
    tidy = [_no(a, q, "after2") for q in ("cash_on_desk", "bag_on_desk", "safe_open") if (q, "after2") in a]
    p_support = absent * _max(left_open)
    p_contra = max(stayed, absent * _min(tidy) if len(tidy) == 3 else 0.0)  # «всё убрано» — только если спросили всё
    return p_support, p_contra


def decide_V03(a: Answers) -> tuple[float, float]:
    """Открытый сейф (в т.ч. за спиной работающего кассира) — кандидат; закрыт на всех кадрах — нет."""
    return _any_yes(a, "safe_open"), _all_no(a, "safe_open")


def _presence(q: str) -> Callable[[Answers], tuple[float, float]]:
    def f(a: Answers) -> tuple[float, float]:
        return _any_yes(a, q), _all_no(a, q)
    return f


def decide_V01(a: Answers) -> tuple[float, float]:
    """Норма — клиент напротив кассира; нарушение — второй человек на стороне кассира или второй посторонний
    со стороны клиентов во время обслуживания."""
    sup = [_p_count_ge(a, "cashier_side_count", at, 2) for at in _asked(a, "cashier_side_count")]
    sup += [_p_count_ge(a, "client_side_count", at, 2) for at in _asked(a, "client_side_count")]
    con_cash = [_p_count_le(a, "cashier_side_count", at, 1) for at in _asked(a, "cashier_side_count")]
    con_client = [_p_count_le(a, "client_side_count", at, 1) for at in _asked(a, "client_side_count")]
    p_contra = _min(con_cash) * (_min(con_client) if con_client else 1.0) if con_cash else 0.0
    return _max(sup), p_contra


def decide_V04(a: Answers) -> tuple[float, float]:
    """Вход/открытая дверь при открытых ценностях (сейф открыт или наличные на столе)."""
    door = _any_yes(a, "door_open")
    valuables = max(_any_yes(a, "safe_open"), _any_yes(a, "cash_on_desk"))
    closed = _all_no(a, "door_open")
    hidden = min(_all_no(a, "safe_open"), _all_no(a, "cash_on_desk"))
    return door * valuables, max(closed, door * hidden)


def decide_V07(a: Answers) -> tuple[float, float]:
    """Из рук в руки и сразу в сейф — норма; сумка оставлена (никто не держит) — кандидат."""
    return _any_yes(a, "bag_unattended"), _all_no(a, "bag_unattended")


def decide_V08(a: Answers) -> tuple[float, float]:
    """Руки с деньгами не видны камере — кандидат."""
    hidden = _max([_no(a, "hands_visible", at) for at in _asked(a, "hands_visible")])
    visible = _min([_yes(a, "hands_visible", at) for at in _asked(a, "hands_visible")])
    return hidden, visible


DECIDERS: dict[str, Callable[[Answers], tuple[float, float]]] = {
    "V01": decide_V01, "V02": decide_V02, "V03": decide_V03, "V04": decide_V04,
    "V05": _presence("phone_in_hand"), "V06": _presence("eating_drinking"), "V07": decide_V07,
    "V08": decide_V08, "V09": _presence("personal_items"),
}


def decide(type_id: str, answers: Answers, spec: dict[str, Any]) -> dict[str, Any]:
    dcfg = spec.get("decision") or {}
    fn = DECIDERS.get(type_id)
    if fn is None:
        return {"agreement": "uncertain", "p_support": 0.0, "p_contra": 0.0, "p_vlm": 0.5, "evidence": 0.0}
    ps, pc = fn(answers)
    ps, pc = max(0.0, min(1.0, ps)), max(0.0, min(1.0, pc))
    evidence = ps + pc
    if evidence < float(dcfg.get("min_evidence", 0.5)):
        return {"agreement": "uncertain", "p_support": round(ps, 4), "p_contra": round(pc, 4), "p_vlm": 0.5,
                "evidence": round(evidence, 4)}
    pv = ps / evidence
    return {"agreement": None, "p_support": round(ps, 4), "p_contra": round(pc, 4), "p_vlm": round(pv, 4),
            "evidence": round(min(1.0, evidence), 4)}


def facts_from_answers(answers: Answers, times: dict[str, float], spec: dict[str, Any]) -> tuple[list[str], list[str]]:
    """-> (факты «за» нарушение, факты «против/норма»). Только уверенные ответы (answer_threshold)."""
    thr = float((spec.get("decision") or {}).get("answer_threshold", 0.6))
    pro: list[str] = []
    con: list[str] = []
    violation_dir = {"cashier_present": "no", "cash_on_desk": "yes", "bag_on_desk": "yes", "safe_open": "yes",
                     "phone_in_hand": "yes", "eating_drinking": "yes", "personal_items": "yes", "door_open": "yes",
                     "bag_unattended": "yes", "hands_visible": "no"}
    for (q, at), d in sorted(answers.items(), key=lambda kv: times.get(kv[0][1], 0.0)):
        t = times.get(at, 0.0)
        if q in COUNT_FACT:
            best = max((k for k in d if k.isdigit()), key=lambda k: d[k], default=None)
            if best is not None and d[best] >= thr:
                (pro if int(best) >= 2 else con).append(COUNT_FACT[q].format(n=best, t=t))
            continue
        for key in ("yes", "no"):
            if d.get(key, 0.0) >= thr and (q, key) in FACT_TEMPLATES:
                (pro if violation_dir.get(q) == key else con).append(FACT_TEMPLATES[(q, key)].format(t=t))
    return pro, con


def transition_time(answers: Answers, times: dict[str, float]) -> Optional[float]:
    """Момент смены состояния по вопросу, заданному на ≥2 кадрах: середина первой пары кадров со сменой
    доминирующего ответа. Только запись в extra (подмена t_peak — флаг qa.refine_t_peak, по умолчанию выкл.)."""
    by_q: dict[str, list[tuple[float, str]]] = {}
    for (q, at), d in answers.items():
        opts = {k: v for k, v in d.items() if not k.startswith("_") and k != "unknown"}
        if not opts or at not in times:
            continue
        by_q.setdefault(q, []).append((times[at], max(opts, key=opts.get)))
    for seq in by_q.values():
        seq.sort()
        for (t1, a1), (t2, a2) in zip(seq, seq[1:]):
            if a1 != a2:
                return round((t1 + t2) / 2.0, 3)
    return None


# --------------------------------------------------------------------------- кадры и кропы

def _zone_bbox(name: str, zones: list[Any], ev: Event) -> Optional[tuple[float, float, float, float]]:
    for z in zones or []:
        zname = getattr(z, "name", None) if not isinstance(z, dict) else z.get("name")
        poly = getattr(z, "polygon", None) if not isinstance(z, dict) else z.get("polygon")
        if zname == name and poly:
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            return min(xs), min(ys), max(xs), max(ys)
    for key, bb in (ev.bboxes or {}).items():  # запасной вариант: боксы события вида «desk», «safe:2»
        if str(key).split(":")[0].lower() == name:
            return tuple(float(v) for v in bb)  # type: ignore[return-value]
    return None


def crop_zone(frame_bgr: Any, bbox: tuple[float, float, float, float], pad: float = 0.1, max_side: int = 448) -> Any:
    img = to_pil_rgb(frame_bgr)
    w, h = img.size
    x1, y1, x2, y2 = bbox
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    box = (max(0, int((x1 - dx) * w)), max(0, int((y1 - dy) * h)), min(w, int(math.ceil((x2 + dx) * w))),
           min(h, int(math.ceil((y2 + dy) * h))))
    if box[2] - box[0] < 8 or box[3] - box[1] < 8:
        return None
    return resize_max_side(img.crop(box), max_side)


def _prompt(spec: dict[str, Any], q: str, t: float, has_crop: bool, lang: str = "ru") -> str:
    qs = spec["questions"][q]
    suffix = (spec.get("suffix") or {}).get(qs.get("suffix", qs.get("options", "yn")), "")
    crop_note = " Второе изображение — увеличенный фрагмент этого же кадра." if has_crop else ""
    return f"Кадр потолочной камеры кассы, время {t:.1f} с.{crop_note}\nВопрос: {qs['text']}\n{suffix}".strip()


# --------------------------------------------------------------------------- проверка одного события

def qa_verify_event(clip: Any, ev: Event, be: VLMBackend, budget: Budget, cfg: dict[str, Any],
                    system: str, zones: Optional[list[Any]] = None,
                    spec: Optional[dict[str, Any]] = None, dump: Optional[Callable[[dict[str, Any]], None]] = None
                    ) -> Optional[dict[str, Any]]:
    """-> результат для _apply_qa или None (тип не поддержан qa / нет кадров). Бюджет режет хвост вопросов."""
    spec = spec or load_spec()
    plan = (spec.get("types") or {}).get(ev.type_id)
    if not plan:
        return None
    qcfg = cfg.get("qa") or {}
    max_side = int((cfg.get("frames") or {}).get("max_side", 448))
    anchor = float(getattr(ev, str(spec.get("anchor", "t_peak")), ev.t_peak))
    duration = float(getattr(clip, "duration", 0.0) or 0.0)
    offsets = spec.get("offsets_s") or {}
    times = {k: round(min(max(0.0, anchor + float(v)), duration if duration > 0 else anchor + float(v)), 3)
             for k, v in offsets.items()}
    min_mass = float((spec.get("decision") or {}).get("min_mass", 0.2))
    frames_cache: dict[str, Any] = {}
    used_times: dict[str, float] = {}  # только кадры, реально показанные модели, с фактическим PTS
    answers: Answers = {}
    asked = skipped = low_mass = 0
    crops_used = 0
    lat = 0.0
    for q, at in plan:
        if q not in (spec.get("questions") or {}) or at not in times:
            continue
        if at not in frames_cache:
            f0, ts = get_frame_ts(clip, times[at])
            frames_cache[at] = f0
            if ts is not None:
                times[at] = round(ts, 4)  # время — фактический PTS показанного кадра
        f = frames_cache[at]
        if f is not None:
            used_times[at] = times[at]
        if f is None:
            skipped += 1
            continue
        images = [resize_max_side(to_pil_rgb(f), max_side)]
        crop = None
        for zname in spec["questions"][q].get("crop") or []:
            bb = _zone_bbox(zname, zones or [], ev)
            if bb is not None:
                crop = crop_zone(f, bb, float(qcfg.get("crop_pad", 0.1)), max_side)
                if crop is not None:
                    break
        if crop is not None:
            images.append(crop)
            crops_used += 1
        vis = sum(visual_tokens(im.size, budget.px_per_token) for im in images)
        if not budget.can_call(len(images), vis):
            skipped += 1
            continue
        prompt = _prompt(spec, q, times[at], crop is not None)
        options = (spec.get("options") or {})[spec["questions"][q].get("options", "yn")]
        try:
            with gpu_guard():
                sc = be.option_probs(images, system, prompt, options, timeout_s=budget.call_timeout())
        except NotImplementedError:
            return None  # бэкенд без логитов — вызывающий откатится на json
        except Exception as e:  # noqa: BLE001
            budget.record(0.0, vis, 1, 1)
            log.warning("qa option_probs failed: %s", type(e).__name__)
            skipped += 1
            continue
        budget.record(sc.latency_s, vis, 1, 1)
        lat += sc.latency_s
        asked += 1
        if dump is not None:
            dump({"kind": "qa", "type_id": ev.type_id, "question": q, "frame": at, "t": times[at], "images": images,
                  "prompt": prompt, "probs": sc.probs, "mass": sc.mass, "mode": sc.mode})
        if sc.mass < min_mass:
            low_mass += 1  # модель отвечала не вариантами — вопрос не учитываем («неизвестно»)
            continue
        answers[(q, at)] = {**{k: float(v) for k, v in sc.probs.items()}, "_mass": float(sc.mass)}
    if asked == 0:
        return {"status": "skipped_budget" if skipped else "no_frames", "answers": {}, "times": used_times}
    dec = decide(ev.type_id, answers, spec)
    pro, con = facts_from_answers(answers, used_times, spec)
    return {"status": "ok", "answers": answers, "times": used_times, "decision": dec, "facts_pro": pro,
            "facts_con": con, "t_transition": transition_time(answers, used_times), "asked": asked,
            "skipped": skipped,
            "low_mass": low_mass, "crops_used": crops_used, "latency_s": round(lat, 3), "spec_version": spec.get("version")}
