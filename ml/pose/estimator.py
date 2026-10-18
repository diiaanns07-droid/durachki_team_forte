"""estimate(clip, tracks): top-down поза COCO-17 по bbox-ам person-треков (батчи кропов, бюджет, сглаживание)."""
from __future__ import annotations

import bisect
import logging
import math
import time
from typing import Any, Optional, Sequence

import numpy as np

from ._contract import PoseFrameX, Track
from .backends import WeightsNotFound, get_backend
from .config import deep_merge, enabled_mode, is_mock_mode, load_config
from .filters import smooth_keypoint_track
from .status import set_status

log = logging.getLogger("ml.pose")


class BoxLookup:
    """bbox трека в момент t: ближайшее наблюдение в ±tol или линейная интерполяция (разрыв ≤ max_gap)."""

    def __init__(self, track: Track, tol: float, max_gap: float) -> None:
        items = []
        for t, b in track.boxes:
            try:
                tt, bb = float(t), np.asarray(b, dtype=np.float64).reshape(4)
            except (TypeError, ValueError):
                continue  # битое наблюдение трека — пропуск (проверка геометрии — в valid_box)
            if np.isfinite(tt):
                items.append((tt, bb))
        items.sort(key=lambda x: x[0])
        self.ts = [t for t, _ in items]
        self.bb = [b for _, b in items]
        self.tol = tol
        self.max_gap = max_gap

    def at(self, t: float) -> Optional[np.ndarray]:
        if not self.ts:
            return None
        i = bisect.bisect_left(self.ts, t)
        best, best_dt = None, self.tol + 1e-9
        for k in (i - 1, i):
            if 0 <= k < len(self.ts) and abs(self.ts[k] - t) <= best_dt:
                best, best_dt = self.bb[k], abs(self.ts[k] - t)
        if best is not None:
            return best
        if 0 < i < len(self.ts) and self.ts[i] - self.ts[i - 1] <= self.max_gap:
            a = (t - self.ts[i - 1]) / (self.ts[i] - self.ts[i - 1])
            return (1 - a) * self.bb[i - 1] + a * self.bb[i]
        return None


_BACKEND_CFG_KEY = {"mediapipe": "mediapipe", "rtmpose-onnx": "rtmpose", "mock": "mock"}


def backend_selection(cfg: dict[str, Any], backend_name: str) -> dict[str, Any]:
    """selection.* ⊕ backend.<имя>.selection (например, MediaPipe — только кассир, 1 кроп на кадр)."""
    base = cfg.get("selection", {}) or {}
    key = _BACKEND_CFG_KEY.get(backend_name, backend_name)
    over = ((cfg.get("backend", {}) or {}).get(key, {}) or {}).get("selection", {}) or {}
    return deep_merge(base, over) if over else dict(base)


def allowed_tracks(tracks: dict[int, Track], sel: dict[str, Any]) -> Optional[set[int]]:
    """Фильтр по ролям (selection.roles); None — без фильтра. roles_fallback=top1: ролей нет ни у кого → все."""
    roles = sel.get("roles")
    if not roles:
        return None
    ids = {tid for tid, tr in tracks.items() if tr.role in set(roles)}
    if not ids and str(sel.get("roles_fallback", "top1")) == "top1":
        return None
    return ids


def valid_box(b: Any, min_area: float) -> Optional[np.ndarray]:
    """Нормированный bbox x1<x2, y1<y2 в пределах кадра (допуск 2%) и не меньше min_area; иначе None."""
    try:
        arr = np.asarray(b, dtype=np.float64).reshape(4)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(arr)):
        return None
    if arr.min() < -0.02 or arr.max() > 1.02:  # не нормирован (пиксели) или далеко за кадром
        return None
    arr = np.clip(arr, 0.0, 1.0)
    if arr[2] <= arr[0] or arr[3] <= arr[1]:
        return None
    if float((arr[2] - arr[0]) * (arr[3] - arr[1])) < min_area:
        return None
    return arr


def _select(lookups: dict[int, BoxLookup], tracks: dict[int, Track], t: float, sel: dict[str, Any],
            allowed: Optional[set[int]] = None, invalid: Optional[dict[int, int]] = None
            ) -> list[tuple[int, np.ndarray]]:
    pri_map = sel.get("role_priority", {}) or {}
    cands = []
    for tid, lk in lookups.items():
        if allowed is not None and tid not in allowed:
            continue
        b = lk.at(t)
        if b is None:
            continue
        b = valid_box(b, float(sel.get("min_box_area", 0.002)))
        if b is None:  # NaN/inf, перевёрнутый, вне 0..1 (пиксели?), нулевой площади — не гадаем позу
            if invalid is not None:
                invalid[tid] = invalid.get(tid, 0) + 1
            continue
        area = float((b[2] - b[0]) * (b[3] - b[1]))
        tr = tracks[tid]
        pri = float(pri_map.get(tr.role, 1.0)) + float(tr.role_conf or 0.0) + 0.5 * math.sqrt(area)
        cands.append((pri, tid, b))
    cands.sort(key=lambda x: (-x[0], x[1]))
    return [(tid, b) for _, tid, b in cands[: int(sel.get("max_persons_per_frame", 3))]]


def estimate_impl(clip: Any, tracks: Sequence[Track], cfg: Optional[dict[str, Any]] = None) -> list[PoseFrameX]:
    cfg = cfg or load_config()
    sel = cfg.get("selection", {}) or {}
    t_start = time.perf_counter()
    persons = {int(t.track_id): t for t in tracks or [] if t.label == "person" and t.boxes}
    times = [float(t) for t in (getattr(clip, "times", None) or [])]
    mode = enabled_mode(cfg)
    if mode == "false":
        set_status(state="disabled", stage_status="skipped", available=False,
                   reason="ml/pose отключена (enabled: false)", coverage=_coverage(times, {}, 0, 1, persons))
        return []
    if not persons or not times:
        set_status(state="no_persons", stage_status="ok", n_crops=0, elapsed_s=0.0,
                   coverage=_coverage(times, {}, 0, 1, persons))
        return []
    exc_list: list[BaseException] = []
    backend, errors = get_backend(cfg, mock=is_mock_mode(), errors_out=exc_list)
    if backend is None:
        only_missing = bool(exc_list) and all(isinstance(e, WeightsNotFound) for e in exc_list)
        msg = "pose backend unavailable: " + "; ".join(errors)
        log.warning("%s — позы пропущены, сигналы позы будут unknown", msg)
        stage = "skipped" if (mode == "auto" and only_missing) else "failed"
        set_status(state="unavailable", stage_status=stage, available=False, error=msg, warnings=[msg], n_crops=0,
                   elapsed_s=time.perf_counter() - t_start, coverage=_coverage(times, {}, 0, 1, persons))
        return []
    sel = backend_selection(cfg, backend.name)
    allowed = allowed_tracks(persons, sel)
    tol = float(sel.get("box_time_tolerance_s", 0.25))
    lookups = {tid: BoxLookup(tr, tol, float(sel.get("max_interp_gap_s", 1.0))) for tid, tr in persons.items()}
    budget = float(sel.get("budget_s", 10.0))
    probe = max(1, int(sel.get("budget_probe_frames", 20)))
    max_stride = max(1, int(sel.get("max_stride", 5)))
    max_total = int(sel.get("max_crops_total", 2500))
    flush_n = max(8, int(getattr(backend, "batch_size", 16)) * 2)
    warnings: list[str] = []
    results: dict[int, list[tuple[float, np.ndarray, np.ndarray, tuple[int, int]]]] = {}
    pending: list[tuple[np.ndarray, np.ndarray, float, int, np.ndarray, tuple[int, int]]] = []
    state = "mock" if backend.name == "mock" else "ok"
    error: Optional[str] = None
    n_crops = 0
    n_frames = 0
    stride = 1
    invalid: dict[int, int] = {}
    min_pose_score = float(sel.get("min_pose_score", cfg.get("signals", {}).get("kp_min_score", 0.3)))
    geo_margin = float(sel.get("geometry_margin", 0.35))
    empty_poses = 0

    def flush() -> bool:
        nonlocal error, state, empty_poses
        if not pending:
            return True
        items = [(p[0], p[1]) for p in pending]
        meta = [(p[2], p[3]) for p in pending]
        try:
            outs = backend.infer(items, meta)
        except Exception as exc:  # noqa: BLE001 — не роняем пайплайн, но и не скрываем
            error = f"{backend.name}.infer: {type(exc).__name__}: {exc}"
            log.warning("pose inference failed — %s", error)
            state = "failed"  # даже если часть поз уже есть: стадия не дошла до конца
            pending.clear()
            return False
        for (frame, box_px, t, tid, bbox_n, wh), (kp, sc) in zip(pending, outs):
            w, h = wh
            kp = np.asarray(kp, dtype=np.float32).reshape(-1, 2)[:17]
            sc = np.nan_to_num(np.asarray(sc, dtype=np.float32).reshape(-1)[:17], nan=0.0)
            if kp.shape[0] < 17 or sc.shape[0] < 17 or not np.all(np.isfinite(kp)) or float(sc.max()) < min_pose_score:
                empty_poses += 1  # поза не найдена / неуверенна: не выдаём скелет (unknown), а не нули
                continue
            kn = np.zeros((17, 3), dtype=np.float32)
            kn[:, 0] = kp[:, 0] / w
            kn[:, 1] = kp[:, 1] / h
            kn[:, 2] = sc
            outside = (kn[:, 0] < -0.02) | (kn[:, 0] > 1.02) | (kn[:, 1] < -0.02) | (kn[:, 1] > 1.02)
            kn[outside, 2] *= 0.5
            # точки далеко за bbox своего трека не принадлежат этому человеку → score 0 (unknown)
            bw, bh = float(bbox_n[2] - bbox_n[0]), float(bbox_n[3] - bbox_n[1])
            far = ((kn[:, 0] < bbox_n[0] - geo_margin * bw) | (kn[:, 0] > bbox_n[2] + geo_margin * bw) |
                   (kn[:, 1] < bbox_n[1] - geo_margin * bh) | (kn[:, 1] > bbox_n[3] + geo_margin * bh))
            kn[far, 2] = 0.0
            kn[:, :2] = np.clip(kn[:, :2], 0.0, 1.0)
            if float(kn[:, 2].max()) < min_pose_score:
                empty_poses += 1
                continue
            results.setdefault(tid, []).append((t, kn, bbox_n, wh))
        pending.clear()
        return True

    i = 0
    while i < len(times):
        t = times[i]
        chosen = _select(lookups, persons, t, sel, allowed, invalid)
        if chosen:
            if n_crops + len(chosen) > max_total:
                warnings.append(f"max_crops_total={max_total} достигнут на t={t:.2f}s")
                break
            try:
                frame = clip.frame(i)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"clip.frame({i}) failed: {type(exc).__name__}: {exc}")
                i += stride
                continue
            h, w = frame.shape[:2]
            for tid, b in chosen:
                box_px = np.array([b[0] * w, b[1] * h, b[2] * w, b[3] * h], dtype=np.float32)
                pending.append((frame, box_px, t, tid, b, (w, h)))
            n_crops += len(chosen)
            n_frames += 1
            if len(pending) >= flush_n and not flush():
                break
            if n_frames % probe == 0 and i + 1 < len(times):
                elapsed = time.perf_counter() - t_start
                per_frame = elapsed / n_frames
                remaining = (len(times) - i - 1) / stride
                if elapsed + remaining * per_frame > budget and stride < max_stride:
                    left = max(budget - elapsed, 1e-3)
                    stride = min(max_stride, max(stride + 1, math.ceil(remaining * per_frame * stride / left)))
                    warnings.append(f"бюджет {budget:.0f}s: шаг по сэмплам → {stride}")
        i += stride
    if error is None:
        flush()
    smooth_cfg = cfg.get("smoothing", {}) or {}
    poses: list[PoseFrameX] = []
    for tid, items in results.items():
        items.sort(key=lambda x: x[0])
        ts = np.array([x[0] for x in items])
        kps = np.stack([x[1] for x in items])
        raw = kps.copy()
        smoothed = bool(smooth_cfg.get("enabled", True)) and len(items) > 1
        if smoothed:
            kps = smooth_keypoint_track(ts, kps, smooth_cfg)
        role = persons[tid].role
        for j, (t, _, bbox_n, wh) in enumerate(items):
            extra: dict[str, Any] = {"bbox": tuple(float(v) for v in bbox_n), "source": backend.name,
                                     "frame_wh": (int(wh[0]), int(wh[1])), "role": role, "smoothed": smoothed,
                                     "schema": "coco17"}
            if smooth_cfg.get("keep_raw"):
                extra["raw_keypoints"] = [tuple(float(v) for v in row) for row in raw[j]]
            poses.append(PoseFrameX(t=float(t), track_id=tid,
                                    keypoints=[(float(x), float(y), float(s)) for x, y, s in kps[j]], extra=extra))
    poses.sort(key=lambda p: (p.t, p.track_id if p.track_id is not None else -1))
    capped = any("max_crops_total" in w for w in warnings)
    if invalid:
        warnings.append(f"невалидные bbox пропущены: {sum(invalid.values())} (треки {sorted(invalid)[:10]})")
    if state == "ok" and n_crops and not poses:
        state = "no_poses"
        warnings.append("поза не найдена ни на одном кропе — сигналы позы unknown")
    elif state == "ok" and (stride > 1 or capped):
        state = "partial"
    if state == "failed":
        stage = "failed"
    elif state == "mock":
        stage = "mock"
    elif capped:
        stage = "timeout"  # лимит кропов: конец видео без позы (см. coverage.analyzed_until_sec)
    else:
        stage = "ok"
    set_status(state=state, stage_status=stage, available=True, backend=backend.name, mock=backend.name == "mock",
               error=error, warnings=warnings + errors, n_crops=n_crops, n_frames=n_frames, stride=stride,
               poses_emitted=len(poses), empty_poses=empty_poses, invalid_boxes=sum(invalid.values()),
               elapsed_s=round(time.perf_counter() - t_start, 3),
               providers=list(getattr(backend, "providers", []) or []),
               coverage=_coverage(times, results, n_frames, stride, persons, capped=capped,
                                  stopped_early=error is not None))
    return poses


def _coverage(times: Sequence[float], results: dict[int, list], n_frames: int, stride: int,
              persons: dict[int, Track], capped: bool = False, stopped_early: bool = False) -> dict[str, Any]:
    """Полнота стадии позы — для meta.completeness у A03 (частичный анализ ≠ «нарушений нет»)."""
    cashier_ids = [tid for tid, tr in persons.items() if tr.role == "cashier"]
    cashier_frames = len({round(x[0], 3) for tid in cashier_ids for x in results.get(tid, [])})
    posed_frames = len({round(x[0], 3) for items in results.values() for x in items})
    analyzed_until = max((x[0] for items in results.values() for x in items), default=None)
    return {
        "frames_total": len(times),
        "frames_with_persons_processed": int(n_frames),
        "frames_posed": posed_frames,
        "stride": int(stride),
        "persons_tracked": len(persons),
        "persons_posed": len(results),
        "cashier_frames_posed": cashier_frames,
        "crops_capped": bool(capped),
        "stopped_early": bool(stopped_early),
        "analyzed_until_sec": None if analyzed_until is None else round(float(analyzed_until), 3),
    }


def warmup_impl(cfg: Optional[dict[str, Any]] = None) -> bool:
    """Загрузить бэкенд и прогнать пустой кроп (держим модель тёплой). False — бэкенд недоступен."""
    cfg = cfg or load_config()
    mode = enabled_mode(cfg)
    if mode == "false":
        set_status(state="disabled", stage_status="skipped", reason="ml/pose отключена (enabled: false)")
        return False
    exc_list: list[BaseException] = []
    backend, errors = get_backend(cfg, mock=is_mock_mode(), errors_out=exc_list)
    if backend is None:
        only_missing = bool(exc_list) and all(isinstance(e, WeightsNotFound) for e in exc_list)
        set_status(state="failed", stage_status="skipped" if (mode == "auto" and only_missing) else "failed",
                   error="; ".join(errors), warnings=errors)
        return False
    try:
        backend.infer([(np.zeros((256, 192, 3), dtype=np.uint8), np.array([0, 0, 192, 256], dtype=np.float32))],
                      [(0.0, None)])
    except Exception as exc:  # noqa: BLE001
        set_status(state="failed", backend=backend.name, error=f"warmup: {type(exc).__name__}: {exc}")
        return False
    return True
