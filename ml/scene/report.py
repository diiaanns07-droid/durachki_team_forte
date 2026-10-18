"""Отчёт для человека (только локальный диск): дорожки предикатов + графики сигналов + линия GT.

    python -m ml.scene.report --cache-root cache/ --labels data/labels.csv --out outputs/scene_report
    python -m ml.scene.report --states cache/<sha>/scene_states.json --gt 22 --out outputs/scene_report

Рисует OpenCV (matplotlib не нужен — его нет в lock L1). Цвета статусов: true — красный, false — зелёный,
unknown — серый, occluded — фиолетовый. Сплошная красная вертикаль — GT-секунда (все t_all для дублей).
Пишет <out>/<video>.png, <out>/index.md и <out>/summary.json. Ничего не отправляет в сеть.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .io import find_states_file, load_states, normalize_video_name
from .predicates import PREDICATES, TYPE_HINT

SIGNAL_PANELS = ["cashier_in_workplace", "outsider_in_cashier_area", "safe_open", "door_open", "cash_visible",
                 "cash_bag_visible", "phone_in_hand", "eating_drinking", "motion_energy"]
COLORS = {"true": (40, 40, 220), "false": (90, 170, 70), "unknown": (175, 175, 175), "occluded": (170, 80, 170)}


def read_gt(labels: Optional[str]) -> dict[str, dict[str, Any]]:
    """data/labels.csv (схема eval.labels: video_id,file,label,t_sec,...,t_all,conflict) → {нормализованный id: {...}}."""
    if not labels or not Path(labels).is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with open(labels, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            vid = row.get("video_id") or row.get("video") or row.get("file") or ""
            ts: list[float] = []
            for key in ("t_all", "t_sec", "gt_sec"):
                raw = (row.get(key) or "").replace("|", ",").replace(";", ",")
                for x in raw.split(","):
                    try:
                        ts.append(float(x))
                    except ValueError:
                        continue
                if ts:
                    break
            out[normalize_video_name(vid)] = {"label": row.get("label", ""), "t": sorted(set(ts)), "conflict": str(row.get("conflict", "0")) in ("1", "true")}
    return out


def render(states: Sequence[Any], out_path: str | Path, gt: Sequence[float] = (), title: str = "",
           events: Sequence[dict[str, Any]] = (), signals: Sequence[str] = SIGNAL_PANELS, preds: Sequence[str] = PREDICATES) -> Path:
    import cv2

    t = np.array([s.t for s in states], dtype=np.float64)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.2
    t0, t1 = (float(t[0]), float(t[-1] + dt)) if len(t) else (0.0, 1.0)
    W, left, right = 1500, 290, 20
    lane_h, panel_h, gap = 16, 46, 6
    preds = [p for p in preds if states and p in getattr(states[0], "pred_status", {})]
    H = 60 + len(preds) * (lane_h + 4) + 24 + len(signals) * (panel_h + gap) + 50
    img = np.full((H, W, 3), 255, np.uint8)

    def x(tt: float) -> int:
        return int(left + (tt - t0) / max(t1 - t0, 1e-6) * (W - left - right))

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, title[:110], (10, 26), font, 0.65, (20, 20, 20), 1, cv2.LINE_AA)
    leg_x = W - 520
    for k, (name, col) in enumerate(COLORS.items()):
        cv2.rectangle(img, (leg_x + k * 125, 12), (leg_x + k * 125 + 18, 26), col, -1)
        cv2.putText(img, name, (leg_x + k * 125 + 24, 25), font, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    y = 50
    for e in events:
        xe = x(float(e["t"]))
        cv2.drawMarker(img, (xe, y - 4), (0, 120, 255) if e.get("type") == "leave" else (255, 120, 0), cv2.MARKER_TRIANGLE_DOWN, 10, 2)
        cv2.putText(img, str(e.get("type", ""))[:6], (xe + 4, y - 2), font, 0.35, (60, 60, 60), 1, cv2.LINE_AA)
    y += 8
    for name in preds:
        st = [s.pred_status.get(name, "unknown") for s in states]
        hint = TYPE_HINT.get(name, "")
        cv2.putText(img, f"{name} {hint}"[:38], (8, y + 12), font, 0.4, (30, 30, 30), 1, cv2.LINE_AA)
        for i, s in enumerate(st):
            cv2.rectangle(img, (x(t[i]), y), (max(x(t[i] + dt), x(t[i])), y + lane_h), COLORS.get(s, COLORS["unknown"]), -1)
        y += lane_h + 4
    y += 24
    for name in signals:
        if not states or name not in states[0].signals:
            continue
        v = np.array([s.signals.get(name, 0.0) for s in states], dtype=np.float64)
        st = [getattr(s, "status", {}).get(name, "unknown") for s in states]
        cv2.putText(img, name[:34], (8, y + panel_h // 2 + 4), font, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.rectangle(img, (left, y), (W - right, y + panel_h), (225, 225, 225), 1)
        cv2.line(img, (left, y + panel_h // 2), (W - right, y + panel_h // 2), (235, 235, 235), 1)
        vmax = max(1.0, float(v.max())) if name == "n_persons" else 1.0
        pts = [(x(t[i]), int(y + panel_h - np.clip(v[i] / vmax, 0, 1) * panel_h)) for i in range(len(v))]
        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], COLORS.get(st[i], COLORS["unknown"]), 2, cv2.LINE_AA)
        y += panel_h + gap
    for g in gt:
        xg = x(float(g))
        cv2.line(img, (xg, 40), (xg, y), (0, 0, 200), 2)
        cv2.putText(img, f"GT {g:g}s", (xg + 4, 44), font, 0.45, (0, 0, 200), 1, cv2.LINE_AA)
    step = 5.0 if t1 - t0 <= 90 else 10.0
    tk = np.arange(np.ceil(t0 / step) * step, t1 + 1e-6, step)
    for tt in tk:
        cv2.line(img, (x(tt), y), (x(tt), y + 6), (60, 60, 60), 1)
        cv2.putText(img, f"{tt:g}", (x(tt) - 8, y + 22), font, 0.42, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(img, "time, s", (W - 90, y + 40), font, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img)
    return p


def summarize(states: Sequence[Any], gt: dict[str, Any], tol_s: float = 2.0) -> dict[str, Any]:
    """Сводка для человека: первые true-интервалы предикатов, расстояние до GT, покрытие."""
    from .intervals import coverage, intervals

    iv = intervals(states, kind="pred")
    first = sorted(((v[0]["t_start"], k) for k, v in iv.items() if v and k in TYPE_HINT))
    gts = gt.get("t", []) if gt else []
    hit = None
    if first and gts:
        hit = min(abs(first[0][0] - g) for g in gts) <= tol_s
    cov = coverage(states)
    return {
        "label": gt.get("label", "") if gt else "",
        "gt_t": gts,
        "conflict": gt.get("conflict", False) if gt else False,
        "candidate": bool(first),
        "first_onset": first[0][0] if first else None,
        "first_predicate": first[0][1] if first else None,
        "type_hint": TYPE_HINT.get(first[0][1]) if first else None,
        "onset_within_tol": hit,
        "intervals": {k: v for k, v in iv.items() if v},
        "known_frac": {k: v["known_frac"] for k, v in cov["per_name"].items()},
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Локальный отчёт ml.scene: PNG + index.md с линией GT")
    ap.add_argument("--cache-root", help="папка, где рекурсивно искать scene_states.json")
    ap.add_argument("--states", action="append", default=[], help="конкретные scene_states.json (можно несколько)")
    ap.add_argument("--labels", default=None, help="data/labels.csv (eval.labels)")
    ap.add_argument("--gt", type=float, action="append", default=[], help="GT-секунда для одиночного --states")
    ap.add_argument("--out", default="outputs/scene_report")
    ap.add_argument("--tol", type=float, default=2.0, help="допуск таймкода в сводке, с")
    args = ap.parse_args(argv)
    files = [Path(f) for f in args.states]
    if args.cache_root:
        files += sorted(Path(args.cache_root).rglob("scene_states.json"))
    if not files:
        ap.error("нет файлов состояний: --cache-root или --states")
    gts = read_gt(args.labels)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    summary: dict[str, Any] = {}
    for f in files:
        f = find_states_file(f) or f
        states, extra = load_states(f)
        meta = extra.get("meta") or {}
        vid = str(meta.get("video_id") or meta.get("video") or f.parent.name)
        key = normalize_video_name(vid)
        gt = gts.get(key, {})
        gt_t = gt.get("t", []) or (args.gt if len(files) == 1 else [])
        if gt_t and not gt:
            gt = {"t": gt_t}
        events = (extra.get("trace") or {}).get("events", []) or meta.get("events", [])
        name = f"{key}__{f.parent.name}" if (out / f"{key}.png").exists() and key in summary else key
        png = render(states, out / f"{name}.png", gt_t, f"{vid}  label={gt.get('label', '?')}  conflict={gt.get('conflict', False)}", events)
        s = summarize(states, gt, args.tol)
        summary[name] = s
        rows.append((name, s, png.name))
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    lines = ["# ml.scene — локальный отчёт", "", "Разведочный просмотр (не независимый тест). Таймкод: первый true-интервал предиката.", "",
             "| видео | GT метка | GT t | кандидат | onset | предикат (тип) | ≤ tol | график |", "|---|---|---|---|---|---|---|---|"]
    for name, s, png in rows:
        lines.append(f"| {name} | {s['label']} | {','.join(f'{g:g}' for g in s['gt_t'])} | {'да' if s['candidate'] else 'нет'} | "
                     f"{'' if s['first_onset'] is None else round(s['first_onset'], 1)} | {s['first_predicate'] or ''} {s['type_hint'] or ''} | "
                     f"{'' if s['onset_within_tol'] is None else ('да' if s['onset_within_tol'] else 'нет')} | [{png}]({png}) |")
    (out / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {len(rows)} report(s) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
