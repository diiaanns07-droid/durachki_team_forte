"""Замер детектора для L2 (локальный GPU): latency, VRAM, валидные детекции по классам, абляция inventory.

    python -m ml.detect.probe --frames <dir> --out <локальная папка> \
        [--modes baseline,rare_open_vocab] [--zones zones.json] [--limit 300] [--no-vis]
    python -m ml.detect.probe --video clip.mov --fps 2 --out <локальная папка> ...

Выход в --out:
  probe_report.json — окружение, latency (ms/кадр: mean/p50/p95), пик VRAM, детекции по классам после порогов
                      (n, доля кадров, квантили score), по режимам inventory: present/unknown по видам и
                      open/closed/unknown сейфа, latency вызова;
  vis/<mode>/*.jpg  — кадры с боксами и зонами, ТОЛЬКО для просмотра человеком на локальном диске;
  .gitignore («*»)  — содержимое папки никогда не попадёт в git.
Кадры и картинки никуда не отправляются; в отчёте нет описаний сцен — только числа.
Метрики качества (FP/FN, попадание таймкода, тип) здесь не считаются: для них нужна разметка событий
(замер всего пайплайна, см. README «Замер L2»).
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .autolabel import list_images
from .boxes import RawDets, classwise_nms
from .config import deep_merge, label_threshold, load_config

MODES = ("baseline", "rare_open_vocab")


def _env() -> dict[str, Any]:
    info: dict[str, Any] = {"python": platform.python_version()}
    try:
        import torch

        info.update(torch=torch.__version__, cuda=bool(torch.cuda.is_available()))
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            info.update(device=torch.cuda.get_device_name(0), vram_free_mb=round(free / 2**20),
                        vram_total_mb=round(total / 2**20))
    except Exception:  # noqa: BLE001
        info["torch"] = None
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001
        info["transformers"] = None
    return info


def _sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass


def _vram_reset() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass


def _vram_peak() -> Optional[dict[str, float]]:
    try:
        import torch

        if torch.cuda.is_available():
            return {"peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                    "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1)}
    except Exception:  # noqa: BLE001
        pass
    return None


def _lat(ms: Sequence[float]) -> dict[str, float]:
    if not ms:
        return {"n": 0}
    a = np.asarray(ms, np.float64)
    return {"n": int(a.size), "mean": round(float(a.mean()), 2), "p50": round(float(np.percentile(a, 50)), 2),
            "p95": round(float(np.percentile(a, 95)), 2), "max": round(float(a.max()), 2)}


def _load_frames(a: argparse.Namespace) -> list[tuple[str, np.ndarray]]:
    import cv2

    if a.video:
        from .synth import ArrayClip

        clip = ArrayClip.from_file(a.video, sample_fps=a.fps)
        items = [(f"t{t:08.3f}", f) for t, f in zip(clip.times, clip.frames)]
    else:
        items = []
        for p in list_images(a.frames):
            img = cv2.imread(str(p))
            if img is not None:
                items.append((str(p.relative_to(a.frames)).replace("/", "__").rsplit(".", 1)[0], img))
    return items[: a.limit] if a.limit else items


def _load_zones(path: Optional[str]) -> Any:
    """[{name, polygon}] | {name: polygon} | {"zones": [...]} (например, выгрузка зон A06)."""
    if not path:
        return None
    z = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(z, dict) and "zones" in z:
        z = z["zones"]
    return z


def _draw(img: np.ndarray, d: RawDets, zones: Any, extra: Sequence[tuple[str, Sequence[float]]] = ()) -> np.ndarray:
    import cv2

    from .inventory import zone_polys

    out = img.copy()
    h, w = out.shape[:2]
    for poly in zone_polys(zones, ["desk", "cash_tray", "safe", "door", "cashier_area", "client_area"]):
        pts = (poly * np.array([w, h], np.float32)).astype(np.int32)
        cv2.polylines(out, [pts], True, (255, 200, 0), 1)
    for i, lab in enumerate(d.labels):
        x1, y1, x2, y2 = (int(round(v)) for v in d.boxes[i] * np.array([w, h, w, h]))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 1)
        cv2.putText(out, f"{lab} {d.scores[i]:.2f}", (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 0), 1)
    for name, box in extra:
        x1, y1, x2, y2 = (int(round(v)) for v in np.asarray(box) * np.array([w, h, w, h]))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, name, (x1, min(h - 3, y2 + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    return out


def run_probe(frames: list[tuple[str, np.ndarray]], out: Path, modes: Sequence[str], zones: Any = None,
              batch: int = 8, vis: bool = True, vis_max: int = 50, config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    import cv2

    from .backends import get_backends, is_mock
    from .inventory import clear_cache, desk_inventory, safe_door_state
    from .synth import ArrayClip

    out.mkdir(parents=True, exist_ok=True)
    (out / ".gitignore").write_text("*\n", encoding="utf-8")   # кадры/картинки — только локально, не в git
    cfg = load_config(config)
    score_thr = cfg["closed_set"].get("score_thr", {"default": 0.4})
    report: dict[str, Any] = {"env": _env(), "mock": is_mock(), "n_frames": len(frames),
                              "frame_shapes": dict(Counter(f"{f.shape[1]}x{f.shape[0]}" for _, f in frames)),
                              "zones": bool(zones), "detect": {}, "inventory": {}, "warnings": []}
    if is_mock():
        report["warnings"].append("SENTINEL_MOCK=1: цифры НЕ относятся к реальным моделям (только тест/демо)")
    # ---- closed-set baseline (одинаков для обоих режимов inventory) ----
    _vram_reset()
    t0 = time.perf_counter()
    be = get_backends(cfg)
    report["detect"]["load_s"] = round(time.perf_counter() - t0, 3)
    report["detect"]["backends"] = be.names
    report["warnings"] += be.warnings
    batches_ms: list[list[float]] = []
    dets: list[RawDets] = []
    for s in range(0, len(frames), max(1, batch)):
        chunk = [f for _, f in frames[s:s + batch]]
        _sync()
        t1 = time.perf_counter()
        parts = [b.predict(chunk) for b in be.closed]
        _sync()
        dt = (time.perf_counter() - t1) * 1000.0
        batches_ms.append([dt / len(chunk)] * len(chunk))
        for k in range(len(chunk)):
            d = classwise_nms(RawDets.concat([p[k] for p in parts]), cfg["merge"]["iou_thr"])
            keep = [i for i, lab in enumerate(d.labels) if d.scores[i] >= label_threshold(score_thr, lab)]
            dets.append(d.select(keep))
    # первый батч — холодный старт (cudnn autotune, аллокации): отдельно, в статистику не входит
    warm = [x for b in batches_ms[1:] for x in b] if len(batches_ms) > 1 else [x for b in batches_ms for x in b]
    report["detect"]["latency_ms_per_frame"] = _lat(warm)
    report["detect"]["cold_first_batch_ms_per_frame"] = round(batches_ms[0][0], 2) if batches_ms else None
    report["detect"]["batch"] = batch
    report["detect"]["vram"] = _vram_peak()
    by_cls: dict[str, list[float]] = defaultdict(list)
    frames_with: Counter = Counter()
    for d in dets:
        for lab in set(d.labels):
            frames_with[lab] += 1
        for i, lab in enumerate(d.labels):
            by_cls[lab].append(float(d.scores[i]))
    report["detect"]["classes"] = {
        lab: {"n_dets": len(sc), "n_frames": frames_with[lab], "frac_frames": round(frames_with[lab] / max(1, len(frames)), 4),
              "score_p10": round(float(np.percentile(sc, 10)), 3), "score_p50": round(float(np.percentile(sc, 50)), 3),
              "score_p90": round(float(np.percentile(sc, 90)), 3)}
        for lab, sc in sorted(by_cls.items())}
    if vis:
        vdir = out / "vis" / "detect"
        vdir.mkdir(parents=True, exist_ok=True)
        for (name, f), d in list(zip(frames, dets))[:vis_max]:
            cv2.imwrite(str(vdir / f"{name}.jpg"), _draw(f, d, zones), [cv2.IMWRITE_JPEG_QUALITY, 85])
    # ---- inventory: абляция baseline ↔ rare_open_vocab ----
    for mode in modes:
        clear_cache()
        _vram_reset()
        icfg = {"inventory": {"mode": mode, "n_frames": 1, "min_frames_present": 1}}
        items: dict[str, Counter] = defaultdict(Counter)
        safe: Counter = Counter()
        lat: list[float] = []
        warns: Counter = Counter()
        eff = Counter()
        vdir = out / "vis" / mode
        if vis:
            vdir.mkdir(parents=True, exist_ok=True)
        for k, (name, f) in enumerate(frames):
            clip = ArrayClip(frames=[f], times=[0.0], path=f"probe://{name}", duration=0.0)
            _sync()
            t1 = time.perf_counter()
            inv = desk_inventory(clip, 0.0, zones, config=deep_merge(config or {}, icfg))
            sd = safe_door_state(clip, 0.0, zones, config=deep_merge(config or {}, icfg))
            _sync()
            lat.append((time.perf_counter() - t1) * 1000.0)
            eff[inv.mode] += 1
            for kind, it in inv.items.items():
                items[kind][it.state] += 1
            safe[sd.state] += 1
            for w in inv.warnings + sd.warnings:
                warns[w] += 1
            if vis and k < vis_max:
                extra = [(f"{kind}:{it.conf:.2f}", it.bbox) for kind, it in inv.items.items()
                         if it.state == "present" and it.bbox is not None]
                if sd.bbox is not None and sd.state != "unknown":
                    extra.append((f"safe:{sd.state}", sd.bbox))
                cv2.imwrite(str(vdir / f"{name}.jpg"), _draw(f, RawDets(), zones, extra), [cv2.IMWRITE_JPEG_QUALITY, 85])
        report["inventory"][mode] = {
            "effective_mode": dict(eff), "latency_ms_per_call": _lat(lat[1:] if len(lat) > 1 else lat),
            "vram": _vram_peak(), "items": {k: dict(v) for k, v in items.items()}, "safe": dict(safe),
            "warnings_top": [{"text": w, "n": n} for w, n in warns.most_common(8)],
        }
    (out / "probe_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _summary(rep: dict[str, Any]) -> str:
    lines = [f"frames={rep['n_frames']} shapes={rep['frame_shapes']} mock={rep['mock']} env={rep['env']}",
             f"detect: backends={rep['detect']['backends']} load={rep['detect']['load_s']}s "
             f"latency_ms/frame={rep['detect']['latency_ms_per_frame']} vram={rep['detect']['vram']}"]
    for lab, c in rep["detect"]["classes"].items():
        lines.append(f"  {lab:14s} n={c['n_dets']:5d} frames={c['frac_frames']:.0%} score p50={c['score_p50']}")
    for mode, r in rep["inventory"].items():
        lines.append(f"inventory[{mode}] effective={r['effective_mode']} latency_ms/call={r['latency_ms_per_call']} vram={r['vram']}")
        for kind, c in sorted(r["items"].items()):
            lines.append(f"  {kind:14s} {c}")
        lines.append(f"  safe           {r['safe']}")
    for w in rep["warnings"]:
        lines.append(f"! {w}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ml.detect.probe")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames", help="папка с кадрами (jpg/png, рекурсивно)")
    src.add_argument("--video", help="видео: кадры берутся с шагом --fps по PTS")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--out", required=True, help="ЛОКАЛЬНАЯ папка для отчёта и картинок (в git не попадает)")
    ap.add_argument("--modes", default="baseline,rare_open_vocab")
    ap.add_argument("--zones", default="", help="JSON зон A06: [{name, polygon}] или {name: polygon}")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--no-vis", action="store_true")
    ap.add_argument("--vis-max", type=int, default=50)
    a = ap.parse_args(argv)
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODES]
    if bad:
        ap.error(f"неизвестные режимы {bad}; допустимы {MODES}")
    if a.frames:
        a.frames = Path(a.frames)
    frames = _load_frames(a)
    if not frames:
        ap.error("не найдено ни одного кадра")
    rep = run_probe(frames, Path(a.out), modes, _load_zones(a.zones), batch=a.batch, vis=not a.no_vis,
                    vis_max=a.vis_max)
    print(_summary(rep))
    print(f"\n→ {Path(a.out) / 'probe_report.json'} (картинки — только локально, папка в .gitignore)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
