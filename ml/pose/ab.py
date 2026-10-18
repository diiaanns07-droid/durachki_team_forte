"""A/B бэкендов позы на одинаковых кадрах (для L2; локально, без сети).

    python -m ml.pose.ab --frames <dir> --out <локальная папка> [--boxes boxes.json]
                         [--backends rtmpose_onnx,mediapipe,mediapipe_noup] [--max-frames N]
    python -m ml.pose.ab --video a.mov [--video b.mov ...] --out <локальная папка>   # кадры+bbox из пайплайна

Кадры — изображения в <dir> (сортировка по имени = порядок времени). bbox людей — из --boxes (JSON:
{"<имя кадра>": [[x1,y1,x2,y2] | {"bbox": [...], "track_id": 1, "role": "cashier"}]}, нормированные 0..1
или пиксели); без --boxes кроп = весь кадр (для RTMPose хуже, чем bbox трекера). С --video кадры берутся
через ml.ingest.load_video (время из PTS), bbox и роли — из ml.detect + ml.track (person-треки); кадры
без людей пропускаются.

Метрики на бэкенд: доля кропов с позой; уверенность и доля видимых точек по группам (голова, плечи,
локти, запястья, таз); дрожание (скорость и «ускорение» точек между соседними кадрами в масштабах тела);
задержка на кроп (p50/p95), батч-пропускная способность; VRAM (pynvml / nvidia-smi, если есть).
Попарно — согласие бэкендов (PCK@0.5 плеча; эталона нет — это не точность).

Выход: <out>/ab_report.json, ab_report.md, per_crop.csv и (по умолчанию) <out>/overlays/<бэкенд>/*.jpg —
скелеты. Голова размывается только по найденным точкам головы; если их нет, размывается весь bbox человека
(место головы не угадывается); счётчики — в ab_report.json → overlay_blur. Картинки — только на локальный
диск: в <out> кладётся .gitignore с «*».
Реальные кадры и картинки никуда не отправляются; JSON/MD содержат только числа и имена файлов.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .config import deep_merge, load_config

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
GROUPS: dict[str, tuple[int, ...]] = {
    "head": (0, 1, 2, 3, 4), "shoulders": (5, 6), "elbows": (7, 8), "wrists": (9, 10), "hips": (11, 12)}
JITTER_KP = (0, 7, 8, 9, 10)  # нос, локти, запястья
SKELETON = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4))

# Варианты: имя → (бэкенд, override конфига). mediapipe_noup — тот же MediaPipe без апскейла кропа.
VARIANTS: dict[str, tuple[str, dict[str, Any]]] = {
    "rtmpose_onnx": ("rtmpose_onnx", {}),
    "mediapipe": ("mediapipe", {}),
    "mediapipe_noup": ("mediapipe", {"backend": {"mediapipe": {"upscale_min_side": 0}}}),
    "mock": ("mock", {}),  # только для тестов/демо: позы НЕ настоящие
}


# ---------------------------------------------------------------- I/O


def list_frames(frames_dir: str, max_frames: Optional[int] = None) -> list[Path]:
    files = sorted(p for p in Path(frames_dir).iterdir() if p.suffix.lower() in IMG_EXT)
    return files[:max_frames] if max_frames else files


FrameItem = tuple[str, Callable[[], Any]]  # (имя кадра, ленивый загрузчик BGR-кадра)


def frames_from_dir(frames_dir: str, max_frames: Optional[int] = None) -> list[FrameItem]:
    import cv2

    return [(p.name, (lambda p=p: cv2.imread(str(p), cv2.IMREAD_COLOR))) for p in list_frames(frames_dir, max_frames)]


def frames_from_videos(videos: Sequence[str], sample_fps: float = 5.0, max_frames: Optional[int] = None
                       ) -> tuple[list[FrameItem], dict[str, list[dict[str, Any]]]]:
    """Кадры и bbox людей из пайплайна команды (ingest → detect → track); всё локально."""
    from ml.detect import detect  # type: ignore[import-not-found]
    from ml.ingest import load_video  # type: ignore[import-not-found]
    from ml.track import track  # type: ignore[import-not-found]

    from .estimator import BoxLookup

    items: list[FrameItem] = []
    boxes: dict[str, list[dict[str, Any]]] = {}
    for v in videos:
        clip = load_video(v, sample_fps)
        tracks, _zones = track(clip, detect(clip))
        persons = [(tr, BoxLookup(tr, 0.25, 1.0)) for tr in tracks if tr.label == "person" and tr.boxes]
        stem = Path(v).stem
        n = len(clip.times) if not max_frames else min(len(clip.times), max_frames)
        for i in range(n):
            entry = []
            for tr, lk in persons:
                b = lk.at(float(clip.times[i]))
                if b is not None:
                    entry.append({"bbox": [float(x) for x in b], "track_id": f"{stem}:{tr.track_id}", "role": tr.role})
            if not entry:
                continue
            name = f"{stem}__{i:05d}.jpg"
            boxes[name] = entry
            items.append((name, (lambda c=clip, k=i: c.frame(k))))
    return items, boxes


def load_boxes(path: Optional[str]) -> dict[str, list[dict[str, Any]]]:
    if not path:
        return {}
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    out: dict[str, list[dict[str, Any]]] = {}
    for name, items in raw.items():
        lst = []
        for k, it in enumerate(items or []):
            # id без явного track_id — своё пространство имён, чтобы не совпасть с явными track_id
            if isinstance(it, dict):
                lst.append({"bbox": list(map(float, it["bbox"])), "track_id": it.get("track_id", f"box{k}"),
                            "role": it.get("role", "unknown")})
            else:
                lst.append({"bbox": list(map(float, it)), "track_id": f"box{k}", "role": "unknown"})
        out[Path(name).name] = lst
    return out


def boxes_px(entry: list[dict[str, Any]], w: int, h: int) -> list[dict[str, Any]]:
    out = []
    for b in entry:
        x1, y1, x2, y2 = b["bbox"]
        if max(x1, y1, x2, y2) <= 1.5:  # нормированные
            x1, x2, y1, y2 = x1 * w, x2 * w, y1 * h, y2 * h
        out.append({**b, "px": np.array([x1, y1, x2, y2], dtype=np.float32)})
    return out


def protect_out_dir(out: Path) -> None:
    """Папка вывода никогда не должна попасть в git: кладём .gitignore с «*»."""
    out.mkdir(parents=True, exist_ok=True)
    gi = out / ".gitignore"
    if not gi.exists():
        gi.write_text("# ml.pose.ab: локальные результаты (кадры/скелеты) — не коммитить\n*\n", encoding="utf-8")


# ---------------------------------------------------------------- GPU memory

_NVML: dict[str, Any] = {}


def gpu_used_mib() -> Optional[float]:
    """Занятая память GPU 0 (MiB) через pynvml, иначе nvidia-smi; None — нет GPU/инструментов."""
    try:
        if "h" not in _NVML:
            import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            _NVML["m"], _NVML["h"] = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
        return float(_NVML["m"].nvmlDeviceGetMemoryInfo(_NVML["h"]).used) / 2 ** 20
    except Exception:  # noqa: BLE001
        pass
    try:
        res = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
        if res.returncode == 0 and res.stdout.strip():
            return float(res.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        pass
    return None


# ---------------------------------------------------------------- metrics


def _scale(kp: np.ndarray, sc: np.ndarray, box: np.ndarray, thr: float) -> float:
    if sc[5] >= thr and sc[6] >= thr:
        s = float(np.linalg.norm(kp[5] - kp[6]))
        if s > 1.0:
            return s
    return 0.35 * float(math.sqrt(max((box[2] - box[0]) * (box[3] - box[1]), 1.0)))


def group_stats(scores: np.ndarray, thr: float) -> dict[str, dict[str, float]]:
    """scores: (N, 17) → по группам: средний score и доля видимых (score ≥ thr)."""
    out = {}
    for g, idx in GROUPS.items():
        sub = scores[:, list(idx)] if len(scores) else np.zeros((0, len(idx)))
        out[g] = {"mean_score": round(float(sub.mean()), 4) if sub.size else None,
                  "visible_frac": round(float((sub >= thr).mean()), 4) if sub.size else None}
    return out


def jitter_stats(seqs: dict[Any, list[tuple[int, np.ndarray, np.ndarray, float]]], thr: float) -> dict[str, Any]:
    """Скорость |p_t − p_{t−1}| и ускорение |p_{t+1} − 2p_t + p_{t−1}| в масштабах тела (соседние кадры)."""
    vel, acc = [], []
    for items in seqs.values():
        items = sorted(items, key=lambda x: x[0])
        for a, b in zip(items, items[1:]):
            if b[0] - a[0] != 1:
                continue
            for k in JITTER_KP:
                if a[2][k] >= thr and b[2][k] >= thr:
                    vel.append(float(np.linalg.norm(b[1][k] - a[1][k])) / max(a[3], 1e-6))
        for a, b, c in zip(items, items[1:], items[2:]):
            if c[0] - a[0] != 2:
                continue
            for k in JITTER_KP:
                if a[2][k] >= thr and b[2][k] >= thr and c[2][k] >= thr:
                    acc.append(float(np.linalg.norm(c[1][k] - 2 * b[1][k] + a[1][k])) / max(b[3], 1e-6))

    def q(x: list[float], p: float) -> Optional[float]:
        return round(float(np.percentile(x, p)), 4) if x else None

    return {"velocity_p50": q(vel, 50), "velocity_p90": q(vel, 90), "accel_p50": q(acc, 50),
            "accel_p90": q(acc, 90), "n_pairs": len(vel), "n_triples": len(acc)}


def agreement(a: dict[tuple[int, Any], tuple[np.ndarray, np.ndarray, float]],
              b: dict[tuple[int, Any], tuple[np.ndarray, np.ndarray, float]], thr: float) -> dict[str, Any]:
    """PCK@0.5 масштаба тела между двумя бэкендами по общим кропам (эталона нет — это согласие)."""
    out: dict[str, Any] = {}
    common = sorted(set(a) & set(b), key=lambda k: (k[0], str(k[1])))  # track_id бывает int и str
    for g, idx in GROUPS.items():
        d = []
        for key in common:
            ka, sa, s = a[key]
            kb, sb, _ = b[key]
            for k in idx:
                if sa[k] >= thr and sb[k] >= thr:
                    d.append(float(np.linalg.norm(ka[k] - kb[k])) / max(s, 1e-6))
        out[g] = {"pck_0.5": round(float(np.mean(np.array(d) <= 0.5)), 4) if d else None,
                  "median_dist": round(float(np.median(d)), 4) if d else None, "n": len(d)}
    out["n_common_crops"] = len(common)
    return out


# ---------------------------------------------------------------- overlays


def draw_overlay(frame: np.ndarray, results: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]], thr: float,
                 blur_heads: bool = True, counts: Optional[dict[str, int]] = None) -> np.ndarray:
    """Скелеты поверх кадра (только для локального диска).

    Приватность: голова размывается по видимым точкам головы (COCO 0..4). Если точек головы нет, место головы
    НЕ угадывается («верх bbox» на потолочной камере — не обязательно голова): размывается весь bbox
    человека. counts (если передан) получает {"head_keypoints": n, "whole_bbox": m}.
    """
    import cv2

    img = frame.copy()
    h, w = img.shape[:2]

    def _blur(x0: float, y0: float, x1: float, y1: float, r: float) -> None:
        xa, ya, xb, yb = int(max(0, x0)), int(max(0, y0)), int(min(w, x1)), int(min(h, y1))
        if xb > xa and yb > ya:
            k = max(3, int(r) | 1)
            img[ya:yb, xa:xb] = cv2.GaussianBlur(img[ya:yb, xa:xb], (k, k), 0)

    if blur_heads:
        for kp, sc, box in results:
            head = [kp[k] for k in GROUPS["head"] if sc[k] >= thr]
            if head:
                c = np.mean(head, axis=0)
                r = max(12.0, 0.6 * _scale(kp, sc, box, thr))
                _blur(c[0] - r, c[1] - r, c[0] + r, c[1] + r, r)
                key = "head_keypoints"
            else:  # точек головы нет — не гадаем, где голова: размываем всего человека
                _blur(box[0], box[1], box[2], box[3], max(12.0, 0.1 * float(box[3] - box[1])))
                key = "whole_bbox"
            if counts is not None:
                counts[key] = counts.get(key, 0) + 1
    for kp, sc, box in results:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (200, 200, 200), 1)
        for a, b in SKELETON:
            if sc[a] >= thr and sc[b] >= thr:
                cv2.line(img, tuple(int(v) for v in kp[a]), tuple(int(v) for v in kp[b]), (0, 220, 0), 2)
        for k in range(17):
            if sc[k] >= thr:
                col = (0, 0, 255) if k in (9, 10) else (255, 160, 0)
                cv2.circle(img, (int(kp[k][0]), int(kp[k][1])), 3, col, -1)
    return img


# ---------------------------------------------------------------- run


def run_variant(name: str, frames: Sequence[FrameItem], boxes: dict[str, list[dict[str, Any]]], cfg: dict[str, Any],
                out: Path, overlays: bool, blur_heads: bool, roles: Optional[set[str]],
                make_backend: Optional[Callable[[str, dict[str, Any]], Any]] = None) -> tuple[dict[str, Any], dict]:
    import cv2

    from .backends import _make

    be_name, override = VARIANTS[name]
    vcfg = deep_merge(cfg, override) if override else cfg
    thr = float(vcfg["signals"].get("kp_min_score", 0.3))
    mem0 = gpu_used_mib()
    t0 = time.perf_counter()
    try:
        backend = (make_backend or _make)(be_name, vcfg)
    except Exception as exc:  # noqa: BLE001 — недоступный бэкенд не роняет A/B
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}, {}
    init_s = time.perf_counter() - t0
    mem_init = gpu_used_mib()
    lat, rows, seqs = [], [], {}
    per_crop: dict[tuple[int, Any], tuple[np.ndarray, np.ndarray, float]] = {}
    all_scores, detected, mem_peak = [], 0, mem_init
    blur_counts: dict[str, int] = {}
    items_for_batch: list[tuple[np.ndarray, np.ndarray]] = []
    warm = 0
    whole = [{"bbox": [0.0, 0.0, 1.0, 1.0], "track_id": 0, "role": "unknown"}]
    for fi, (fname, load) in enumerate(frames):
        frame = load()
        if frame is None:
            continue
        h, w = frame.shape[:2]
        entry = boxes.get(fname, []) if boxes else whole  # bbox заданы → кадр без людей пропускаем
        bxs = [b for b in boxes_px(entry, w, h) if roles is None or b.get("role") in roles]
        res_frame = []
        for bi, b in enumerate(bxs):
            tid = b.get("track_id")
            meta = [(fi / 5.0, tid if isinstance(tid, int) else None)]
            if warm < 3:  # прогрев — вне замера
                backend.infer([(frame, b["px"])], meta)
                warm += 1
            t1 = time.perf_counter()
            (kp, sc), = backend.infer([(frame, b["px"])], meta)
            dt = (time.perf_counter() - t1) * 1000.0
            lat.append(dt)
            kp, sc = np.asarray(kp, np.float32), np.asarray(sc, np.float32)
            s = _scale(kp, sc, b["px"], thr)
            all_scores.append(sc)
            detected += int((sc >= thr).any())
            seqs.setdefault(b.get("track_id", bi), []).append((fi, kp, sc, s))
            per_crop[(fi, b.get("track_id", bi))] = (kp, sc, s)
            res_frame.append((kp, sc, b["px"]))
            if len(items_for_batch) < 64:
                items_for_batch.append((frame, b["px"]))
            row = {"frame": fname, "box": bi, "track_id": b.get("track_id", bi), "backend": name,
                   "latency_ms": round(dt, 3)}
            for g, idx in GROUPS.items():
                row[f"{g}_score"] = round(float(sc[list(idx)].mean()), 4)
            rows.append(row)
        if len(lat) % 20 == 0:
            m = gpu_used_mib()
            if m is not None:
                mem_peak = max(mem_peak or m, m)
        if overlays and res_frame:
            od = out / "overlays" / name
            od.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(od / (Path(fname).stem + ".jpg")),
                        draw_overlay(frame, res_frame, thr, blur_heads, blur_counts))
    batch_ms = None
    bs = int(getattr(backend, "batch_size", 1) or 1)
    if bs > 1 and items_for_batch:
        t1 = time.perf_counter()
        for i in range(0, len(items_for_batch), bs):
            backend.infer(items_for_batch[i:i + bs], [(0.0, None)] * len(items_for_batch[i:i + bs]))
        batch_ms = round((time.perf_counter() - t1) * 1000.0 / len(items_for_batch), 3)
    m = gpu_used_mib()
    if m is not None:
        mem_peak = max(mem_peak or m, m)
    scores = np.stack(all_scores) if all_scores else np.zeros((0, 17))
    rep = {
        "status": "ok", "backend": getattr(backend, "name", be_name), "mock": be_name == "mock",
        "providers": list(getattr(backend, "providers", []) or []), "init_s": round(init_s, 3),
        "n_crops": len(lat), "pose_found_frac": round(detected / len(lat), 4) if lat else None,
        "groups": group_stats(scores, thr), "jitter": jitter_stats(seqs, thr),
        "latency_ms": {"mean": round(float(np.mean(lat)), 3) if lat else None,
                       "p50": round(float(np.percentile(lat, 50)), 3) if lat else None,
                       "p95": round(float(np.percentile(lat, 95)), 3) if lat else None,
                       "batched_per_crop": batch_ms, "batch_size": bs},
        "overlay_blur": ({"enabled": blur_heads, **blur_counts} if overlays else None),
        "vram_mib": {"before": mem0, "after_init": mem_init, "peak": mem_peak,
                     "delta_peak": None if (mem0 is None or mem_peak is None) else round(mem_peak - mem0, 1)},
    }
    if hasattr(backend, "close"):
        backend.close()
    return rep, {"rows": rows, "per_crop": per_crop}


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = ["# A/B бэкендов позы (ml.pose.ab)", "",
             f"Кадров: {report['n_frames']}; bbox: {report['boxes_source']}; порог видимости: {report['kp_min_score']}", "",
             "| вариант | статус | поза найдена | запястья vis | локти vis | голова vis | дрожание p50/p90 | мс/кроп p50/p95 | мс/кроп батч | VRAM Δ MiB |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in report["backends"].items():
        if r.get("status") != "ok":
            lines.append(f"| {name} | {r.get('status')}: {r.get('error', '')[:80]} |  |  |  |  |  |  |  |  |")
            continue
        g, j, la = r["groups"], r["jitter"], r["latency_ms"]
        lines.append(f"| {name}{' (MOCK)' if r['mock'] else ''} | ok | {r['pose_found_frac']} | "
                     f"{g['wrists']['visible_frac']} | {g['elbows']['visible_frac']} | {g['head']['visible_frac']} | "
                     f"{j['velocity_p50']}/{j['velocity_p90']} | {la['p50']}/{la['p95']} | {la['batched_per_crop']} | "
                     f"{r['vram_mib']['delta_peak']} |")
    if report.get("agreement"):
        lines += ["", "## Согласие бэкендов (PCK@0.5 ширины плеч; эталона нет)", "",
                  "| пара | голова | плечи | локти | запястья | общих кропов |", "|---|---|---|---|---|---|"]
        for pair, a in report["agreement"].items():
            lines.append(f"| {pair} | {a['head']['pck_0.5']} | {a['shoulders']['pck_0.5']} | {a['elbows']['pck_0.5']} | "
                         f"{a['wrists']['pck_0.5']} | {a['n_common_crops']} |")
    lines += ["", "Дрожание — медиана/90-й перцентиль смещения носа/локтей/запястий между соседними кадрами "
              "в ширинах плеч (включает реальное движение; сравнивать варианты между собой)."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None,
         make_backend: Optional[Callable[[str, dict[str, Any]], Any]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ml.pose.ab", description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames", help="папка с кадрами (jpg/png), порядок = имя файла")
    src.add_argument("--video", action="append", help="видео (можно несколько): кадры+bbox через ingest/detect/track")
    ap.add_argument("--sample-fps", type=float, default=5.0, help="для --video")
    ap.add_argument("--out", required=True, help="локальная папка результатов (в неё кладётся .gitignore '*')")
    ap.add_argument("--boxes", default=None, help="JSON с bbox людей по именам кадров")
    ap.add_argument("--backends", default="rtmpose_onnx,mediapipe,mediapipe_noup",
                    help=f"через запятую из {sorted(VARIANTS)} (mock — только тесты/демо)")
    ap.add_argument("--roles", default=None, help="оставить только bbox с этими ролями (через запятую)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-overlays", action="store_true", help="не сохранять картинки со скелетами")
    ap.add_argument("--no-blur", action="store_true", help="не размывать головы на картинках (не рекомендуется)")
    args = ap.parse_args(argv)

    names = [n.strip() for n in args.backends.split(",") if n.strip()]
    bad = [n for n in names if n not in VARIANTS]
    if bad:
        ap.error(f"неизвестные варианты: {bad}")
    out = Path(args.out)
    protect_out_dir(out)
    if args.video:
        try:
            frames, boxes = frames_from_videos(args.video, args.sample_fps, args.max_frames)
        except ImportError as exc:
            print(f"--video требует модули пайплайна (ml.ingest, ml.detect, ml.track): {exc}", file=sys.stderr)
            return 2
        src_desc = "ml.track (person-треки)"
    else:
        frames = frames_from_dir(args.frames, args.max_frames)
        boxes = load_boxes(args.boxes)
        src_desc = args.boxes or "весь кадр"
    if not frames:
        print("нет кадров (или нет людей в кадрах)", file=sys.stderr)
        return 2
    roles = {r.strip() for r in args.roles.split(",")} if args.roles else None
    cfg = load_config()
    report: dict[str, Any] = {"n_frames": len(frames), "boxes_source": src_desc,
                              "kp_min_score": float(cfg["signals"].get("kp_min_score", 0.3)),
                              "backends": {}, "agreement": {}}
    all_rows, crops = [], {}
    for name in names:
        rep, extra = run_variant(name, frames, boxes, cfg, out, not args.no_overlays, not args.no_blur, roles,
                                 make_backend)
        report["backends"][name] = rep
        all_rows += extra.get("rows", [])
        if extra.get("per_crop"):
            crops[name] = extra["per_crop"]
        print(f"[ml.pose.ab] {name}: {rep.get('status')} {rep.get('error', '')}", file=sys.stderr)
    ok = list(crops)
    for i in range(len(ok)):
        for j in range(i + 1, len(ok)):
            report["agreement"][f"{ok[i]}~{ok[j]}"] = agreement(crops[ok[i]], crops[ok[j]], report["kp_min_score"])
    (out / "ab_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, out / "ab_report.md")
    if all_rows:
        with open(out / "per_crop.csv", "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            wr.writeheader()
            wr.writerows(all_rows)
    print(f"[ml.pose.ab] отчёт: {out / 'ab_report.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
