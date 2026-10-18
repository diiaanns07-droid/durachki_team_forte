"""Абляция state machine / сглаживания на синтетических случаях (ДЕМО-данные, не реальные ролики).

    python -m ml.scene.ablation [--out outputs/scene_ablation]

Варианты: hmm (по умолчанию), hysteresis, none (без сглаживания), наивный режим без защит (не увидели = нет,
пропавший трек = ушёл, первые кадры = закрытый сейф, «не кассир» = посторонний). Случаи:

* отрицательные (нарушения быть не должно): N1 сейф открыт с начала и закрыт до ухода кассира;
  N2 кассир заслонил закрытый сейф и ушёл; N3 трек кассира пропал посреди рабочего места при деньгах
  на столе; N4 два сотрудника на стороне кассы; N5 клиент напротив наклоняется над столом;
* положительные контроли: P1 кассир ушёл, деньги на столе; P2 сейф открыт и кассир ушёл.

Метрики: FP — запрещённый предикат/сигнал стал true; FN — ожидаемый предикат не стал true;
ошибка таймкода — |t_start первого интервала − ожидаемый момент|. Это проверка логики, а не точности
на реальных данных: цифры точности даёт только замер L2 (python -m ml.scene.l2_bench).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

VARIANTS: dict[str, dict[str, Any]] = {
    "hmm": {},
    "hysteresis": {"ablation": {"smoothing": {"method": "hysteresis", "hi": 0.6, "lo": 0.4, "alpha": 0.5}}},
    "none": {"ablation": {"smoothing": {"method": "none"}}},
    "naive (без защит unknown/ролей/эталона)": {"ablation": {"naive": True}},
    "naive + none": {"ablation": {"naive": True, "smoothing": {"method": "none"}}},
}
FORBIDDEN = ["cash_left_after_leave", "bag_left_after_leave", "safe_open_after_leave", "outsider_cashier_side",
             "entry_while_valuables_open", "bag_left_unattended", "personal_item_on_desk"]
LEGACY_FORBIDDEN = ["workplace_unattended_with_cash", "outsider_in_cashier_area"]


def _cases() -> dict[str, dict[str, Any]]:
    """Синтетические сцены (ДЕМО). Импорт здесь, чтобы модуль не тянул тестовые генераторы при импорте пакета."""
    from ml.scene._contract import Detection
    from ml.scene.tests import test_phase2 as S
    from ml.scene.tests.synth import SAFE as SYN_SAFE
    from ml.scene.tests.synth import make_clip, person_dets, person_track

    T = S.TIMES
    cases: dict[str, dict[str, Any]] = {}

    def leaving(t_walk: float):
        return S.leaving_cashier(t_walk=t_walk)

    # N1: сейф (на кадрах!) открыт с начала и закрыт в 5 с, детекций сейфа нет; кассир ушёл в 8 с, стол пуст
    tr = leaving(8.0)
    cases["N1_safe_open_from_start"] = {"dets": person_dets([tr]), "tracks": [tr], "neg": True, "legacy_forbid_after": {"safe_open": 6.0},
                                        "clip_kw": {"safe_open": lambda t: t < 5.0}, "safe_zone": SYN_SAFE}
    # N2: кассир заслонил закрытый сейф 3–6 с, затем ушёл; сейф только «закрыт» по детекциям
    boxes = {}
    for t in T:
        if t < 3.0:
            boxes[t] = S.SEAT
        elif t < 6.0:
            boxes[t] = (S.SAFE[0] - 0.04, S.SAFE[1] - 0.03, S.SAFE[2] + 0.03, S.SAFE[3] + 0.05)
        elif t < 6.8:
            a = min(1.0, (t - 6.0) / 0.6)
            boxes[t] = (0.70 + 0.22 * a, 0.60, 0.80 + 0.20 * a, 0.76)
    tr = person_track(1, boxes, role="cashier", conf=0.9)
    dets = person_dets([tr]) + [Detection(t=t, label="safe", bbox=S.SAFE, score=0.6) for t in T if not 3.0 <= t < 6.0]
    cases["N2_safe_occluded_then_leave"] = {"dets": dets, "tracks": [tr], "neg": True, "legacy_forbid_after": {"safe_open": 0.0}}
    # N3: трек кассира пропал посреди рабочего места 4–9 с; деньги на столе всё время
    t1 = person_track(1, {t: S.SEAT for t in T if t < 4.0}, role="cashier", conf=0.9)
    t2 = person_track(2, {t: S.SEAT for t in T if t >= 9.0}, role="cashier", conf=0.9)
    cases["N3_track_lost_mid_workplace"] = {"dets": person_dets([t1, t2]) + S.cash(), "tracks": [t1, t2], "neg": True}
    # N4: два сотрудника на стороне кассы, ролей нет
    a = person_track(1, {t: S.SEAT for t in T})
    b = person_track(2, {t: (0.60, 0.62, 0.76, 0.78) for t in T if t >= 2.0})
    cases["N4_two_employees"] = {"dets": person_dets([a, b]) + S.cash(), "tracks": [a, b], "neg": True}
    # N5: клиент напротив наклоняется над столом
    c = person_track(2, {t: (S.CLIENT_LEAN if 4.0 <= t < 7.0 else S.CLIENT_SEAT) for t in T})
    cases["N5_client_opposite"] = {"dets": person_dets([a, c]) + S.cash(), "tracks": [a, c], "neg": True}
    # P1: кассир ушёл (3–3.8 с), деньги на столе
    tr = leaving(3.0)
    cases["P1_cash_left"] = {"dets": person_dets([tr]) + S.cash(), "tracks": [tr], "neg": False, "expect": "cash_left_after_leave", "t_expect": 3.8}
    # P2: сейф открыт (детекции), кассир ушёл в 6 с
    tr = leaving(6.0)
    dets = person_dets([tr]) + [Detection(t=t, label="safe_open", bbox=S.SAFE, score=0.8) for t in T]
    cases["P2_safe_open_left"] = {"dets": dets, "tracks": [tr], "neg": False, "expect": "safe_open_after_leave", "t_expect": 6.8}
    for v in cases.values():
        v["clip"] = make_clip  # кадры синтетики (без реальных данных)
    return cases


def run(variants: Optional[dict[str, dict[str, Any]]] = None, progress: Optional[Callable[[str], None]] = None) -> dict[str, Any]:
    from ml.scene.core import scene_states_with_trace
    from ml.scene.intervals import intervals
    from ml.scene.tests import test_phase2 as S

    variants = variants or VARIANTS
    cases = _cases()
    base_clip = cases[next(iter(cases))]["clip"](S.DUR)  # синтетическая картинка по умолчанию
    res: dict[str, Any] = {}
    for vname, override in variants.items():
        rows = {}
        fp = fn = 0
        errs = []
        for cname, c in cases.items():
            clip = c["clip"](S.DUR, **c["clip_kw"]) if c.get("clip_kw") else base_clip
            zones = S.zones()
            if c.get("safe_zone"):
                from ml.scene._contract import Zone

                zones = [z for z in zones if z.name != "safe"] + [Zone("safe", S.poly(c["safe_zone"]))]
            states, trace = scene_states_with_trace(clip, c["dets"], c["tracks"], zones, [], config=override or None)
            t = np.array([s.t for s in states])
            bad = sorted({p for s in states for p in FORBIDDEN if s.pred_status.get(p) == "true"})
            bad += sorted({k for s in states for k in LEGACY_FORBIDDEN if s.status.get(k) == "true"})
            for k, after in (c.get("legacy_forbid_after") or {}).items():
                if any(s.status.get(k) == "true" for s, tt in zip(states, t) if tt >= after):
                    bad.append(k)
            row: dict[str, Any] = {}
            if c["neg"]:
                row["false_positives"] = bad
                fp += int(bool(bad))
            else:
                iv = intervals(states, [c["expect"]])[c["expect"]]
                row["detected"] = bool(iv)
                fn += int(not iv)
                if iv:
                    row["onset_err_s"] = round(abs(iv[0]["t_start"] - c["t_expect"]), 2)
                    errs.append(row["onset_err_s"])
            rows[cname] = row
            if progress:
                progress(f"{vname}:{cname}")
        res[vname] = {"cases": rows, "fp_cases": fp, "fn_cases": fn, "mean_onset_err_s": round(float(np.mean(errs)), 2) if errs else None}
    return res


def to_markdown(res: dict[str, Any]) -> str:
    lines = ["| вариант | FP (из 5 отриц.) | FN (из 2 полож.) | ошибка onset, с | какие FP |", "|---|---|---|---|---|"]
    for v, r in res.items():
        fps = "; ".join(f"{c}: {', '.join(x['false_positives'])}" for c, x in r["cases"].items() if x.get("false_positives"))
        lines.append(f"| {v} | {r['fp_cases']} | {r['fn_cases']} | {r['mean_onset_err_s']} | {fps or '—'} |")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Абляция сглаживания/state machine на синтетике (демо)")
    ap.add_argument("--out", default="outputs/scene_ablation")
    args = ap.parse_args(argv)
    res = run()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    md = to_markdown(res)
    (out / "ablation.md").write_text("# ml.scene — абляция (синтетика, демо)\n\n" + md + "\n", encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
