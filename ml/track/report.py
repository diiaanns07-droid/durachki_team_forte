"""PNG-таймлайн ролей, зон и событий для человека (пишется на ЛОКАЛЬНЫЙ диск; кадров видео нет).

    python -m ml.track.report --cache cache/<sha> [--out cache/<sha>/track_timeline.png] [--smoothing hysteresis]
    python -m ml.track.report --demo v02 --out timeline.png     # СИНТЕТИЧЕСКОЕ демо, помечено на картинке

Строки: люди (цвет = роль, тёмная полоса = в cashier_area, серые засечки = пропуски детекций),
события (вход, уход с рабочего места, уход из помещения, посторонний / unknown у кассы),
n_outsiders (сплошная — visitor, пунктир — верхняя граница с unknown). Серая заливка — потеря видимости.
Подписи латиницей (шрифты OpenCV не умеют кириллицу).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

ROLE_BGR = {"cashier": (70, 170, 60), "visitor": (60, 60, 220), "collector": (0, 160, 255), "unknown": (150, 150, 150)}
EVENT_ROWS = [("person_entered", "entered", (200, 120, 0)), ("cashier_left_workplace", "left workplace", (0, 90, 230)),
              ("cashier_left_room", "left room", (0, 0, 170)), ("outsider_at_cashier_side", "outsider @cashier", (40, 40, 255)),
              ("unknown_at_cashier_side", "unknown @cashier", (110, 110, 110))]


def render_timeline(tracks: Sequence[Any], zones: Sequence[Any], info: Optional[dict] = None,
                    cfg: Optional[dict] = None, title: str = "", width: int = 1600) -> np.ndarray:
    import cv2

    from .events import box_at, inside, zone_by_name
    from .queries import TrackQueries

    q = TrackQueries(tracks, zones, info or {}, cfg)
    persons = [t for t in tracks if t.label == "person"]
    t0, t1 = q.t0, max(q.t1, q.t0 + 1e-3)
    L, R, top = 190, 30, 92
    row_h, ev_h, plot_h = 26, 22, 70
    H = top + row_h * max(1, len(persons)) + 20 + ev_h * len(EVENT_ROWS) + 20 + plot_h + 60
    img = np.full((H, width, 3), 255, np.uint8)
    pw = width - L - R

    def X(t: float) -> int:
        return int(L + (t - t0) / (t1 - t0) * pw)

    for a, b in q.vis:  # потеря видимости
        cv2.rectangle(img, (X(a), top - 10), (X(b), H - 50), (225, 225, 225), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, title[:120], (10, 24), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    step = max(1, int(round((t1 - t0) / 15)))
    for s in range(int(t0), int(t1) + 1, step):
        cv2.line(img, (X(s), top - 12), (X(s), H - 50), (235, 235, 235), 1)
        cv2.putText(img, f"{s}s", (X(s) - 8, top - 18), font, 0.4, (80, 80, 80), 1, cv2.LINE_AA)
    wp = q.workplace
    y = top
    for tr in persons:
        col = ROLE_BGR.get(tr.role, (0, 0, 0))
        cv2.putText(img, f"#{tr.track_id} {tr.role} {tr.role_conf:.2f}", (8, y + 17), font, 0.45, col, 1, cv2.LINE_AA)
        rows = sorted(tr.boxes, key=lambda r: r[0])
        for (ta, ba), (tb, _) in zip(rows[:-1], rows[1:]):
            if tb - ta <= 2.5 * q.frame_dt:
                cv2.rectangle(img, (X(ta), y + 4), (max(X(tb), X(ta) + 1), y + row_h - 4), col, -1)
                if wp is not None and inside(ba, wp):
                    cv2.rectangle(img, (X(ta), y + 10), (max(X(tb), X(ta) + 1), y + row_h - 10), (20, 60, 20), -1)
            else:
                cv2.line(img, (X(ta), y + row_h // 2), (X(tb), y + row_h // 2), (190, 190, 190), 1)
        y += row_h
    y += 20
    evs = q.events()
    for kind, label, col in EVENT_ROWS:
        cv2.putText(img, label, (8, y + 15), font, 0.45, col, 1, cv2.LINE_AA)
        for e in [e for e in evs if e.kind == kind]:
            xa = X(e.t)
            if e.t_end is not None and e.t_end > e.t:
                cv2.rectangle(img, (xa, y + 7), (X(e.t_end), y + ev_h - 7), col, 1)
            cv2.drawMarker(img, (xa, y + ev_h // 2), col, cv2.MARKER_TRIANGLE_DOWN, 12, 2)
            cv2.putText(img, f"{e.t:.1f}s {e.conf:.2f}", (xa + 6, y + 9), font, 0.33, col, 1, cv2.LINE_AA)
        y += ev_h
    y += 20
    cv2.putText(img, "n_outsiders", (8, y + plot_h // 2), font, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(img, (L, y), (L + pw, y + plot_h), (200, 200, 200), 1)
    vals = [q.n_outsiders(float(t)) for t in q.grid]
    vmax = max([int(a.detail.get("upper_bound", 0) or 0) for a in vals] + [2])
    pts_v, pts_u = [], []
    for t, a in zip(q.grid, vals):
        yy = lambda v: int(y + plot_h - 4 - (plot_h - 8) * v / vmax)  # noqa: E731
        if a.value is None:
            continue
        pts_v.append((X(float(t)), yy(a.value)))
        pts_u.append((X(float(t)), yy(a.detail.get("upper_bound", a.value))))
    if len(pts_u) > 1:
        for i in range(0, len(pts_u) - 1, 2):
            cv2.line(img, pts_u[i], pts_u[i + 1], (150, 150, 150), 1)
    if len(pts_v) > 1:
        cv2.polylines(img, [np.array(pts_v, np.int32)], False, (60, 60, 220), 2)
    cv2.putText(img, f"max {vmax}", (L - 60, y + 12), font, 0.35, (80, 80, 80), 1, cv2.LINE_AA)
    leg = "cashier=green visitor=red collector=orange unknown=gray | dark inner bar = in cashier_area | gray fill = visibility loss"
    cv2.putText(img, leg, (8, H - 20), font, 0.42, (60, 60, 60), 1, cv2.LINE_AA)
    zl = ", ".join(f"{z.name}:{z.conf:.2f}" for z in zones)
    cv2.putText(img, f"zones: {zl}", (8, 46), font, 0.42, (60, 60, 60), 1, cv2.LINE_AA)
    del box_at, zone_by_name
    return img


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ml.track.report")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cache", help="каталог кэша cache/<sha> (track.json или detections.json)")
    g.add_argument("--demo", help="СИНТЕТИЧЕСКИЙ сценарий из ml.track.synth: clean|v01|v02|v02_bounce|v07|two_staff|fidget|cashier_enters")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoothing", default=None, choices=["off", "hysteresis"], help="режим H-SMOOTH для событий")
    args = ap.parse_args(argv)

    import cv2

    from .api import track_full
    from .config import load_config
    from .serialize import load_detections, load_track_cache

    cfg = load_config(overrides={"queries": {"smoothing": args.smoothing}} if args.smoothing else None)
    if args.demo:
        from . import synth as S
        from .replay import SCENARIOS

        if args.demo not in SCENARIOS:
            print(f"unknown demo {args.demo}; known: {sorted(SCENARIOS)}", file=sys.stderr)
            return 2
        clip, dets, _ = S.make(SCENARIOS[args.demo][0]())
        res = track_full(clip, dets, cfg=cfg)
        tracks, zones, info = res.tracks, res.zones, res.info
        title = f"SYNTHETIC DEMO (not real data): {args.demo}"
        out = Path(args.out or f"ml/track/_out/timeline_demo_{args.demo}.png")
    else:
        cached = load_track_cache(args.cache)
        if cached is None:
            dets = load_detections(args.cache)
            if dets is None:
                print(f"нет track.json и детекций в {args.cache}", file=sys.stderr)
                return 2
            res = track_full(None, dets, cfg=cfg)
            tracks, zones, info = res.tracks, res.zones, res.info
        else:
            tracks, zones, info = cached
        title = f"ml.track timeline: {Path(args.cache).name}"
        out = Path(args.out or Path(args.cache) / "track_timeline.png")
    img = render_timeline(tracks, zones, info, cfg, title)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
