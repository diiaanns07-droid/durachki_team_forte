"""Графики сигналов сцены во времени с вертикальной линией GT-секунды (локальная отладка).

    python -m ml.scene.plot --cache cache/<sha> --out signals.png [--gt 22] [--gt 57]
    python -m ml.scene.plot --video data/X-1.mov --out signals.png --gt 22     # посчитать на лету

Сплошная линия — значение подтверждено наблюдением (known), пунктир — неизвестно (conf < 0.5).
Кэш: файл scene_states.json (``ml.scene.io.dump_states``) в папке кэша или путь к JSON напрямую;
GT берётся из --gt или из meta (ключи gt_sec / gt_seconds / gt / label_sec / event_sec).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .io import dump_states, find_states_file, load_states

PANELS: list[tuple[str, list[str]]] = [
    ("люди", ["cashier_present", "cashier_in_workplace", "outsider_in_cashier_area", "collector_present"]),
    ("число людей", ["n_persons"]),
    ("сейф / дверь", ["safe_open", "door_open"]),
    ("наличные", ["cash_visible", "cash_bag_visible", "workplace_unattended_with_cash"]),
    ("поза", ["phone_in_hand", "phone_to_ear", "eating_drinking", "hands_hidden", "cashier_facing_away", "handover_motion"]),
    ("кадр", ["motion_energy", "_quality"]),
]
GT_KEYS = ("gt_sec", "gt_seconds", "gt", "label_sec", "event_sec", "gt_second")


def _gt_from_meta(meta: dict[str, Any]) -> list[float]:
    for k in GT_KEYS:
        v = meta.get(k)
        if v is None:
            continue
        vals = v if isinstance(v, (list, tuple)) else [v]
        out = []
        for x in vals:
            try:
                out.append(float(x))
            except (TypeError, ValueError):
                continue
        if out:
            return out
    return []


def load_for_plot(cache: str) -> tuple[list[Any], dict[str, Any]]:
    f = find_states_file(cache)
    if f is None:
        raise FileNotFoundError(f"no scene states JSON in {cache} (expected scene_states.json)")
    states, extra = load_states(f)
    meta = dict(extra.get("meta") or {})
    mfile = Path(cache) / "meta.json" if Path(cache).is_dir() else None
    if mfile is not None and mfile.is_file():
        try:
            meta = {**json.loads(mfile.read_text(encoding="utf-8")), **meta}
        except ValueError:
            pass
    return states, {"meta": meta, "trace": extra.get("trace") or {}}


def plot_states(states: Sequence[Any], out: str | Path, gt: Sequence[float] = (), title: str = "") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array([s.t for s in states])
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(13, 2.0 * len(PANELS)), sharex=True)
    for ax, (label, names) in zip(axes, PANELS):
        for k, name in enumerate(names):
            color = f"C{k}"
            if name == "_quality":
                y = np.array([getattr(s, "quality", 1.0) for s in states])
                ax.plot(t, y, color="0.4", lw=1.2, label="quality")
                continue
            y = np.array([s.signals.get(name, np.nan) for s in states])
            known = np.array([bool(getattr(s, "known", {}).get(name, True)) for s in states])
            ax.plot(t, y, color=color, lw=0.9, ls=":", alpha=0.8)
            ax.plot(t, np.where(known, y, np.nan), color=color, lw=1.8, label=name)
        for g in gt:
            ax.axvline(g, color="crimson", lw=1.6, alpha=0.85)
        if label != "число людей":
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(0.5, color="0.8", lw=0.6, zorder=0)
        ax.set_ylabel(label)
        ax.legend(loc="upper right", fontsize=7, ncol=3, framealpha=0.6)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("время, с   (сплошная — подтверждено наблюдением, пунктир — неизвестно; красная — GT)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def _compute_from_video(video: str, cache: Optional[str]) -> list[Any]:
    from .core import scene_states_with_trace
    from .train_signals import load_inputs

    clip, dets, tracks, zones = load_inputs(Path(video), None, use_upstream=True)
    pose: list[Any] = []
    try:
        from ml.pose import estimate, pose_signals  # type: ignore

        pose = pose_signals(estimate(clip, tracks), tracks, zones)
    except Exception:  # noqa: BLE001 — поза необязательна для отладочного графика
        pose = []
    states, trace = scene_states_with_trace(clip, dets, tracks, zones, pose)
    if cache:
        dump_states(states, Path(cache) / "scene_states.json", trace=trace, meta={"video": Path(video).name})
    return states


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Графики сигналов ml.scene")
    ap.add_argument("--cache", help="cache/<sha> или путь к scene_states.json")
    ap.add_argument("--video", help="посчитать по видео (модули команды, если доступны)")
    ap.add_argument("--out", default="signals.png")
    ap.add_argument("--gt", type=float, action="append", default=[], help="GT-секунда (можно несколько)")
    ap.add_argument("--title", default="")
    args = ap.parse_args(argv)
    if not args.cache and not args.video:
        ap.error("нужен --cache или --video")
    meta: dict[str, Any] = {}
    if args.video and (not args.cache or find_states_file(args.cache) is None):
        states = _compute_from_video(args.video, args.cache)
    else:
        states, extra = load_for_plot(args.cache)
        meta = extra["meta"]
    gt = args.gt or _gt_from_meta(meta)
    title = args.title or str(meta.get("video") or args.video or args.cache or "")
    out = plot_states(states, args.out, gt, title)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
