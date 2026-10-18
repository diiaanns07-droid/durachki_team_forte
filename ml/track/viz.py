"""Отладочная отрисовка треков/ролей/зон на видео (локально; результат не коммитить).

    python -m ml.track.viz video.mov --cache cache/<sha> [--out out.mp4] [--fps 5] [--recompute] [--blur-heads]

Детекции берутся из кэша (detections.json | dets.json | *.pkl); если их нет — пробуем
ml.detect.detect(clip). Треки/зоны — из cache/<sha>/track.json, иначе считаем и сохраняем.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

ROLE_COLORS = {"cashier": (60, 200, 60), "collector": (0, 200, 255), "visitor": (60, 60, 230),
               "unknown": (180, 180, 180)}
ZONE_COLORS = {"cashier_area": (60, 200, 60), "desk": (200, 160, 60), "safe": (160, 60, 200),
               "door": (0, 140, 255), "client_area": (60, 60, 230), "cash_tray": (0, 220, 220)}


class SimpleClip:
    """Минимальный VideoClip на cv2 (если ml.ingest недоступен). Поворот — CAP_PROP_ORIENTATION_AUTO."""

    def __init__(self, path: str, sample_fps: float = 5.0) -> None:
        import cv2

        self.path = path
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        try:
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        except Exception:
            pass
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frames, times = [], []
        period = 1.0 / max(sample_fps, 1e-6)
        next_t = 0.0
        last_t = 0.0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            t = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0  # время только из PTS (переменный fps)
            last_t = max(last_t, t)
            if t + 1e-6 >= next_t:
                frames.append(fr)
                times.append(round(t, 4))
                next_t = t + period
        cap.release()
        self._frames = frames
        self.times = times
        self.duration = last_t
        self.height, self.width = (frames[0].shape[:2] if frames else (0, 0))

    def frame(self, i: int) -> Any:
        return self._frames[i]

    def frame_at(self, t: float) -> Any:
        j = int(np.argmin(np.abs(np.asarray(self.times) - t)))
        return self._frames[j]


def load_clip(path: str, sample_fps: float) -> Any:
    try:
        from ml.ingest import load_video  # type: ignore

        return load_video(path, sample_fps=sample_fps)
    except Exception:
        return SimpleClip(path, sample_fps)


def _px(poly: Sequence[Sequence[float]], w: int, h: int) -> np.ndarray:
    return np.array([[int(round(x * w)), int(round(y * h))] for x, y in poly], np.int32)


def draw_frame(img: np.ndarray, t: float, tracks: Sequence[Any], zones: Sequence[Any],
               max_dt: float = 0.25, blur_heads: bool = False, trail_s: float = 3.0) -> np.ndarray:
    import cv2

    from .events import box_at

    out = img.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    for z in zones:
        col = ZONE_COLORS.get(z.name, (255, 255, 255))
        pts = _px(z.polygon, w, h)
        if len(pts) >= 3:
            cv2.fillPoly(overlay, [pts], col)
            cv2.polylines(out, [pts], True, col, 2)
            cv2.putText(out, f"{z.name} {z.conf:.2f}", tuple(int(v) for v in pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, col, 1, cv2.LINE_AA)
    out = cv2.addWeighted(overlay, 0.15, out, 0.85, 0)
    for tr in tracks:
        b = box_at(tr, t, max_dt)
        if b is None:
            continue
        x1, y1, x2, y2 = int(b[0] * w), int(b[1] * h), int(b[2] * w), int(b[3] * h)
        if tr.label == "person":
            col = ROLE_COLORS.get(tr.role, (255, 255, 255))
            if blur_heads:
                hy2 = y1 + max(1, (y2 - y1) // 4)
                roi = out[max(0, y1):max(0, hy2), max(0, x1):max(0, x2)]
                if roi.size:
                    out[max(0, y1):max(0, hy2), max(0, x1):max(0, x2)] = cv2.GaussianBlur(roi, (31, 31), 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            cv2.putText(out, f"#{tr.track_id} {tr.role} {tr.role_conf:.2f}", (x1, max(10, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            trail = [((bb[0] + bb[2]) / 2 * w, (bb[1] + bb[3]) / 2 * h) for tt, bb in tr.boxes if t - trail_s <= tt <= t]
            for (ax, ay), (bx, by) in zip(trail[:-1], trail[1:]):
                cv2.line(out, (int(ax), int(ay)), (int(bx), int(by)), col, 1)
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), (230, 230, 230), 1)
            cv2.putText(out, f"{tr.label}#{tr.track_id}", (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(out, f"t={t:6.2f}s", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ml.track.viz", description=__doc__.splitlines()[0])
    ap.add_argument("video")
    ap.add_argument("--cache", required=True, help="каталог кэша cache/<sha>")
    ap.add_argument("--out", default=None, help="выходной .mp4 (по умолчанию <cache>/track_viz.mp4)")
    ap.add_argument("--fps", type=float, default=5.0, help="частота сэмплинга кадров")
    ap.add_argument("--recompute", action="store_true", help="пересчитать треки, игнорируя track.json")
    ap.add_argument("--blur-heads", action="store_true", help="размыть верх боксов людей")
    args = ap.parse_args(argv)

    import cv2

    from .api import track_full
    from .serialize import load_detections, load_track_cache, save_track_cache

    clip = load_clip(args.video, args.fps)
    cached = None if args.recompute else load_track_cache(args.cache)
    if cached is None:
        dets = load_detections(args.cache)
        if dets is None:
            try:
                from ml.detect import detect  # type: ignore

                dets = detect(clip)
            except Exception as e:  # noqa: BLE001
                print(f"нет детекций в {args.cache} и ml.detect недоступен: {e}", file=sys.stderr)
                return 2
        res = track_full(clip, dets)
        tracks, zones, info = res.tracks, res.zones, res.info
        save_track_cache(args.cache, tracks, zones, info)
    else:
        tracks, zones, info = cached
    out = Path(args.out) if args.out else Path(args.cache) / "track_viz.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    times = list(clip.times)
    if not times:
        print("пустое видео", file=sys.stderr)
        return 3
    first = clip.frame(0)
    h, w = first.shape[:2]
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), (w, h))
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.2
    for i, t in enumerate(times):
        vw.write(draw_frame(clip.frame(i), float(t), tracks, zones, max_dt=0.6 * dt, blur_heads=args.blur_heads))
    vw.release()
    roles = {t.track_id: (t.role, t.role_conf) for t in tracks if t.label == "person"}
    print(f"written {out}  persons={roles}  zones={[z.name for z in zones]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
