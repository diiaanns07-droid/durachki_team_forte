"""MediaPipe Pose Landmarker (Tasks API, Apache-2.0) — переключаемый вариант бэкенда, по умолчанию выключен.

Включение: SENTINEL_POSE_BACKEND=mediapipe или backend.order: [mediapipe]. Работает по кропу человека
(по умолчанию только кассира, см. backend.mediapipe.selection) с апскейлом: короткая сторона кропа
доводится до upscale_min_side px — детектор BlazePose плохо видит мелких людей.

Выход ЯВНО перекладывается из 33 landmark-ов BlazePose в порядок COCO-17 (MP33_TO_COCO17); остальные
16 точек отбрасываются. score = visibility (× presence) — шкала не та же, что у RTMPose
(visibility_scale в config). Поза не найдена → нулевые score (unknown), а не выдуманные точки.
Модель (.task, офлайн): backend.mediapipe.task_path | $SENTINEL_POSE_MP_TASK |
$SENTINEL_WEIGHTS_DIR/{pose,}/**/pose_landmarker_<variant>.task | ./weights/pose/.
"""
from __future__ import annotations

import atexit
import glob
import os
from typing import Any, Optional, Sequence

import numpy as np

from .base import PoseBackendError, WeightsNotFound

MP33_NAMES: tuple[str, ...] = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky",
    "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
)

# COCO-17 индекс → индекс landmark-а MediaPipe Pose (33 точки, BlazePose topology)
MP33_TO_COCO17: tuple[int, ...] = (
    0,   # 0 nose            ← 0 NOSE
    2,   # 1 left_eye        ← 2 LEFT_EYE
    5,   # 2 right_eye       ← 5 RIGHT_EYE
    7,   # 3 left_ear        ← 7 LEFT_EAR
    8,   # 4 right_ear       ← 8 RIGHT_EAR
    11,  # 5 left_shoulder   ← 11 LEFT_SHOULDER
    12,  # 6 right_shoulder  ← 12 RIGHT_SHOULDER
    13,  # 7 left_elbow      ← 13 LEFT_ELBOW
    14,  # 8 right_elbow     ← 14 RIGHT_ELBOW
    15,  # 9 left_wrist      ← 15 LEFT_WRIST
    16,  # 10 right_wrist    ← 16 RIGHT_WRIST
    23,  # 11 left_hip       ← 23 LEFT_HIP
    24,  # 12 right_hip      ← 24 RIGHT_HIP
    25,  # 13 left_knee      ← 25 LEFT_KNEE
    26,  # 14 right_knee     ← 26 RIGHT_KNEE
    27,  # 15 left_ankle     ← 27 LEFT_ANKLE
    28,  # 16 right_ankle    ← 28 RIGHT_ANKLE
)


def mp33_to_coco17(landmarks_xyv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(33, 3) [x, y, score] → (17, 2) xy, (17,) score — в порядке COCO-17."""
    lm = np.asarray(landmarks_xyv, dtype=np.float32)
    if lm.shape != (33, 3):
        raise ValueError(f"ожидались 33 landmark-а MediaPipe, получено {lm.shape}")
    idx = np.array(MP33_TO_COCO17)
    return lm[idx, :2].copy(), np.clip(lm[idx, 2], 0.0, 1.0)


def find_task(cfg: dict[str, Any]) -> Optional[str]:
    mc = cfg.get("backend", {}).get("mediapipe", {}) or {}
    explicit = mc.get("task_path") or os.environ.get("SENTINEL_POSE_MP_TASK")
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    name = f"pose_landmarker_{mc.get('variant', 'full')}.task"
    dirs: list[str] = []
    wd = os.environ.get("SENTINEL_WEIGHTS_DIR")
    if wd:
        dirs += [os.path.join(wd, "pose"), wd]
    dirs.append(os.path.join(os.getcwd(), "weights", "pose"))
    for d in dirs:
        hits = sorted(glob.glob(os.path.join(os.path.expanduser(d), "**", name), recursive=True))
        if hits:
            return hits[0]
    return None


def crop_upscale(frame: np.ndarray, box_px: Sequence[float], padding: float, min_side: float,
                 max_scale: float) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Кроп bbox×padding (за краем кадра — чёрное поле) с апскейлом только вверх.

    Возвращает (изображение, (x0, y0, w, h) кропа в пикселях кадра) — для обратного маппинга точек.
    """
    import cv2

    x1, y1, x2, y2 = (float(v) for v in box_px)
    w = max((x2 - x1) * padding, 2.0)
    h = max((y2 - y1) * padding, 2.0)
    x0 = (x1 + x2) / 2.0 - w / 2.0
    y0 = (y1 + y2) / 2.0 - h / 2.0
    s = float(min(max(min_side / min(w, h), 1.0), max(max_scale, 1.0)))
    out_w, out_h = max(1, int(round(w * s))), max(1, int(round(h * s)))
    mat = np.array([[s, 0.0, -x0 * s], [0.0, s, -y0 * s]], dtype=np.float32)
    img = cv2.warpAffine(frame, mat, (out_w, out_h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return img, (x0, y0, out_w / s, out_h / s)


def to_frame(xy_norm: np.ndarray, geom: tuple[float, float, float, float]) -> np.ndarray:
    """Нормированные координаты кропа (0..1) → пиксели кадра."""
    x0, y0, cw, ch = geom
    return np.stack([x0 + xy_norm[:, 0] * cw, y0 + xy_norm[:, 1] * ch], axis=1).astype(np.float32)


def pick_pose(cands: Sequence[tuple[np.ndarray, np.ndarray]], box_px: Sequence[float], min_score: float = 0.3,
              min_inside: float = 0.5) -> int:
    """Из поз в кропе — та, чьи видимые точки больше лежат в bbox трека (×1.1), с учётом score.

    -1 — ни одна поза не принадлежит треку (меньше min_inside видимых точек внутри bbox): в кроп с паддингом
    попал сосед, и приписывать его скелет этому треку нельзя.
    """
    x1, y1, x2, y2 = (float(v) for v in box_px)
    mx, my = 0.05 * (x2 - x1), 0.05 * (y2 - y1)
    best, best_v = -1, -1.0
    for i, (xy, sc) in enumerate(cands):
        vis = sc >= min_score
        if not vis.any():
            continue
        inside = ((xy[:, 0] >= x1 - mx) & (xy[:, 0] <= x2 + mx) & (xy[:, 1] >= y1 - my) & (xy[:, 1] <= y2 + my))
        frac = float((inside & vis).sum()) / float(vis.sum())
        if frac < min_inside:
            continue
        v = frac * float(sc[vis].mean())
        if v > best_v:
            best, best_v = i, v
    return best


class MediaPipePose:
    """running_mode=image — один landmarker, кадры независимы (как было у A07).
    running_mode=video — отдельный поток VIDEO-режима на каждый track_id: время кадра — PTS из clip.times
    (монотонно внутри потока), поза внутри кропа выбирается по bbox трека (pick_pose), поэтому порядок
    скелетов в ответе модели не используется как ID. Число потоков ограничено (max_streams, LRU)."""

    name = "mediapipe"

    def __init__(self, cfg: dict[str, Any], landmarker: Any = None, landmarker_factory: Any = None) -> None:
        mc = cfg.get("backend", {}).get("mediapipe", {}) or {}
        self.pad = float(mc.get("crop_padding", 1.4))
        self.min_side = float(mc.get("upscale_min_side", 384))
        self.max_scale = float(mc.get("upscale_max", 3.0))
        self.vis_scale = float(mc.get("visibility_scale", 1.0))
        self.use_presence = bool(mc.get("use_presence", True))
        self.min_score = float(cfg.get("signals", {}).get("kp_min_score", 0.3))
        self.mode = str(mc.get("running_mode", "image")).lower()
        self.max_streams = max(1, int(mc.get("max_streams", 8)))
        self.batch_size = 1
        self._mp = None
        self._factory = landmarker_factory
        self._streams: dict[Any, Any] = {}          # track_id → VIDEO-landmarker (порядок вставки = LRU)
        self._last_ts: dict[Any, int] = {}
        self.stats = {"calls": 0, "streams_opened": 0, "streams_evicted": 0, "ts_bumped": 0, "no_track_id": 0}
        if landmarker is not None or landmarker_factory is not None:  # тесты: фейки без mediapipe и модели
            self.landmarker = landmarker if landmarker is not None else landmarker_factory()
            self.providers = ["injected"]
            return
        path = find_task(cfg)
        if not path:
            raise WeightsNotFound(f"модель MediaPipe pose_landmarker_{mc.get('variant', 'full')}.task не найдена")
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except Exception as exc:  # noqa: BLE001
            raise PoseBackendError(f"mediapipe недоступен: {type(exc).__name__}: {exc}") from exc
        delegate = str(mc.get("delegate", "cpu")).lower()
        base = mp_python.BaseOptions(
            model_asset_path=path,
            delegate=(mp_python.BaseOptions.Delegate.GPU if delegate == "gpu" else mp_python.BaseOptions.Delegate.CPU))

        def _make(mode: Any) -> Any:
            opts = vision.PoseLandmarkerOptions(
                base_options=base, running_mode=mode, num_poses=int(mc.get("num_poses", 2)),
                min_pose_detection_confidence=float(mc.get("min_pose_detection_confidence", 0.3)),
                min_pose_presence_confidence=float(mc.get("min_pose_presence_confidence", 0.3)),
                min_tracking_confidence=float(mc.get("min_tracking_confidence", 0.3)),
                output_segmentation_masks=False)
            return vision.PoseLandmarker.create_from_options(opts)

        try:
            self.landmarker = _make(vision.RunningMode.IMAGE)  # для кропов без track_id и warmup
        except Exception as exc:  # noqa: BLE001 — например, нет libGLESv2 в контейнере
            raise PoseBackendError(f"PoseLandmarker не создан: {type(exc).__name__}: {exc}") from exc
        self._factory = (lambda: _make(vision.RunningMode.VIDEO)) if self.mode == "video" else None
        self._mp = mp
        self.path = path
        self.providers = [f"mediapipe-{delegate}-{self.mode}"]
        atexit.register(self.close)

    def close(self) -> None:
        streams = list(getattr(self, "_streams", {}).values())
        self._streams = {}
        lm, self.landmarker = getattr(self, "landmarker", None), None
        for x in streams + [lm]:
            if x is not None and hasattr(x, "close"):
                try:
                    x.close()
                except Exception:  # noqa: BLE001
                    pass

    def _stream(self, tid: Any) -> Any:
        """VIDEO-landmarker трека: создаётся при первом кадре трека; старейший поток закрывается сверх лимита."""
        lm = self._streams.pop(tid, None)
        if lm is None:
            if len(self._streams) >= self.max_streams:
                old_tid, old = next(iter(self._streams.items()))
                self._streams.pop(old_tid)
                self._last_ts.pop(old_tid, None)
                self.stats["streams_evicted"] += 1
                if hasattr(old, "close"):
                    old.close()
            lm = self._factory()
            self.stats["streams_opened"] += 1
        self._streams[tid] = lm  # в конец — недавно использованный
        return lm

    def _timestamp_ms(self, tid: Any, t: float) -> int:
        """PTS кадра в мс; VIDEO-режим требует строго возрастающих отметок внутри потока."""
        ts = int(round(float(t) * 1000.0))
        last = self._last_ts.get(tid)
        if last is not None and ts <= last:
            ts = last + 1
            self.stats["ts_bumped"] += 1
        self._last_ts[tid] = ts
        return ts

    def _detect(self, rgb: np.ndarray, tid: Any = None, t: Optional[float] = None) -> list[np.ndarray]:
        """→ список (33, 3) [x, y, score] нормированных к входному изображению."""
        video = self._factory is not None and tid is not None and t is not None
        lm = self._stream(tid) if video else self.landmarker
        self.stats["calls"] += 1
        if self._mp is not None:
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            res = lm.detect_for_video(image, self._timestamp_ms(tid, t)) if video else lm.detect(image)
        else:
            res = lm.detect_for_video(rgb, self._timestamp_ms(tid, t)) if video else lm.detect(rgb)
        out = []
        for pose in getattr(res, "pose_landmarks", None) or []:
            rows = []
            for p in pose:
                vis = getattr(p, "visibility", None)
                pres = getattr(p, "presence", None)
                score = vis if vis is not None else (pres if pres is not None else 0.0)
                if self.use_presence and vis is not None and pres is not None:
                    score = min(vis, pres)
                rows.append((float(p.x), float(p.y), float(score)))
            if len(rows) == 33:
                out.append(np.array(rows, dtype=np.float32))
        return out

    def infer(self, items: Sequence[tuple[np.ndarray, np.ndarray]], meta: Any = None) -> list[tuple[np.ndarray, np.ndarray]]:
        """meta (от estimate): [(t PTS, track_id)] на каждый элемент — нужен VIDEO-режиму."""
        out = []
        metas = list(meta) if meta is not None else [(None, None)] * len(items)
        for (frame, box), m in zip(items, metas):
            t, tid = (m[0], m[1]) if isinstance(m, (tuple, list)) and len(m) >= 2 else (None, None)
            if self._factory is not None and tid is None:
                self.stats["no_track_id"] += 1
            img, geom = crop_upscale(frame, box, self.pad, self.min_side, self.max_scale)
            cands = []
            for lm in self._detect(img[..., ::-1], tid, t):
                xy, sc = mp33_to_coco17(lm)
                cands.append((to_frame(xy, geom), np.clip(sc * self.vis_scale, 0.0, 1.0).astype(np.float32)))
            k = pick_pose(cands, box, self.min_score) if cands else -1
            if k < 0:  # поза не найдена или не этого трека → нулевые score (estimate() её не выдаст)
                out.append((np.zeros((17, 2), np.float32), np.zeros(17, np.float32)))
                continue
            out.append(cands[k])
        return out
