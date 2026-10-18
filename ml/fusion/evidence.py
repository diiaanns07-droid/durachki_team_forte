"""Кадры-доказательства для нарушений.

* Ответ жюри (``Violation.evidence_frames``) — ЧИСТЫЙ исходный кадр (без рамок/подписей) в JPEG ≤ 200 КБ,
  ``timestamp_sec`` = действительный PTS этого кадра: время берётся из ``clip.times`` (PTS сэмплированных
  кадров от ml.ingest) или из ``clip.frame_at_pts(t) -> (img, pts)``, если ingest его предоставляет.
* Для UI — отдельно: JPEG с оверлеями (боксы с русскими подписями, штамп времени, подпись типа, рамка)
  и векторные фигуры (боксы/области размытия) — ``meta.ui.evidence_overlays``.
* Приватность: размытие голов только по надёжной локализации (ключевые точки головы COCO 0..4 или
  явные head_boxes из Event.extra). Эвристика «верх bbox» выключена по умолчанию: на потолочной камере
  верх бокса — не обязательно голова и может скрыть руки/действия.
"""
from __future__ import annotations

import base64
import math
import re
import time
from typing import Any, Optional, Sequence

import numpy as np

from ._types import EvidenceFrame
from .timing import clip_times, nearest_index, snap_time

HEAD_KP = (0, 1, 2, 3, 4)          # COCO: нос, глаза, уши
SHOULDERS = (5, 6)

ROLE_RU = {
    "cashier": "кассир", "visitor": "посетитель", "collector": "инкассатор", "unknown": "человек",
    "person": "человек", "outsider": "посторонний", "client": "клиент", "customer": "клиент",
}
LABEL_RU = {
    "cell_phone": "телефон", "phone": "телефон", "cup": "чашка", "bottle": "бутылка", "food": "еда",
    "cash": "наличные", "money_counter": "счётчик банкнот", "laptop": "ноутбук", "safe": "сейф",
    "safe_open": "сейф открыт", "door": "дверь", "door_open": "дверь открыта",
    "cash_bag": "инкассаторская сумка", "bag": "сумка", "document": "документ", "keys": "ключи",
    "hand": "рука", "hands": "руки", "desk": "стол", "cash_tray": "лоток",
}
ZONE_RU = {
    "cashier_area": "зона кассира", "desk": "зона стола", "safe": "зона сейфа", "door": "зона двери",
    "client_area": "клиентская зона", "cash_tray": "зона лотка",
}
PERSON_KEYS = set(ROLE_RU)
# BGR
ROLE_COLOR = {
    "cashier": (70, 190, 60), "visitor": (50, 50, 230), "outsider": (50, 50, 230), "client": (50, 50, 230),
    "customer": (50, 50, 230), "collector": (0, 150, 255), "unknown": (230, 190, 40),
    "person": (230, 190, 40),
}
LABEL_COLOR = {
    "cash": (0, 215, 255), "cash_bag": (0, 150, 255), "cell_phone": (220, 60, 220), "phone": (220, 60, 220),
    "safe": (255, 140, 30), "safe_open": (255, 140, 30), "door": (255, 140, 30), "door_open": (255, 140, 30),
}
SEVERITY_COLOR = {"high": (50, 50, 230), "medium": (0, 140, 255), "low": (0, 200, 230)}
_SUFFIX = re.compile(r"^(?P<base>.*?)[\s_:#\-]*(?P<num>\d+)$")


def parse_box_key(key: str) -> tuple[str, Optional[int]]:
    """'visitor:3' → ('visitor', 3); 'cash_bag' → ('cash_bag', None); 'cashier_hand' → ('cashier', None)."""
    k = str(key).strip().lower()
    num: Optional[int] = None
    if k.startswith("zone:") or k.startswith("zone_"):
        return "zone:" + k[5:].strip(), None
    m = _SUFFIX.match(k)
    if m and m.group("base"):
        k, num = m.group("base").strip(" _:#-"), int(m.group("num"))
    if k in ROLE_RU or k in LABEL_RU:
        return k, num
    known = sorted(list(ROLE_RU) + list(LABEL_RU), key=len, reverse=True)
    for cand in known:
        if k.startswith(cand) or k.endswith(cand):
            return cand, num
    return k, num


def ru_label(key: str) -> str:
    base, num = parse_box_key(key)
    if base.startswith("zone:"):
        name = base[5:]
        return ZONE_RU.get(name) or f"зона {LABEL_RU.get(name, name)}"
    text = ROLE_RU.get(base) or LABEL_RU.get(base) or base
    return f"{text} #{num}" if num is not None else text


def box_color(key: str) -> tuple[int, int, int]:
    base, _ = parse_box_key(key)
    if base.startswith("zone:"):
        return (170, 170, 170)
    return ROLE_COLOR.get(base) or LABEL_COLOR.get(base) or (255, 220, 0)


def is_person_key(key: str) -> bool:
    return parse_box_key(key)[0] in PERSON_KEYS


def to_pixels(box: Sequence[float], w: int, h: int) -> Optional[tuple[int, int, int, int]]:
    """Нормированный bbox (0..1) → пиксели; значения > 1.5 считаем уже пиксельными."""
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if any(math.isnan(v) for v in (x1, y1, x2, y2)):
        return None
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1, x2, y1, y2 = x1 * w, x2 * w, y1 * h, y2 * h
    xa, xb = sorted((int(round(x1)), int(round(x2))))
    ya, yb = sorted((int(round(y1)), int(round(y2))))
    xa, xb = max(0, xa), min(w - 1, xb)
    ya, yb = max(0, ya), min(h - 1, yb)
    if xb - xa < 2 or yb - ya < 2:
        return None
    return xa, ya, xb, yb


def ensure_bgr(img: Any) -> Optional[np.ndarray]:
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return None
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr) if arr.size else 0) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(np.nan_to_num(arr), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def sharpness(img: np.ndarray) -> float:
    """Дисперсия лапласиана на уменьшенном сером кадре."""
    import cv2  # lazy

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    if w > 320:
        gray = cv2.resize(gray, (320, max(2, int(h * 320 / w))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def thumb(img: np.ndarray) -> np.ndarray:
    import cv2  # lazy

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32)


def is_duplicate(a: np.ndarray, b: np.ndarray, block_thr: float, block: int = 12) -> bool:
    """Дубль = почти идентичные кадры: максимум средней разности по блокам мал (локальные изменения — не дубль)."""
    d = np.abs(a - b)
    h, w = d.shape[:2]
    hb, wb = h // block, w // block
    if hb == 0 or wb == 0:
        return float(d.mean()) < block_thr
    blocks = d[: hb * block, : wb * block].reshape(hb, block, wb, block).mean(axis=(1, 3))
    return float(blocks.max()) < block_thr


def encode_jpeg(img: np.ndarray, max_bytes: int, qualities: Sequence[int], max_side: int = 1280) -> bytes:
    """JPEG ≤ max_bytes: понижаем качество, затем размер."""
    import cv2  # lazy

    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (max(2, int(w * s)), max(2, int(h * s))), interpolation=cv2.INTER_AREA)
    qs = [int(q) for q in qualities] or [85, 70, 55, 40]
    data = b""
    for _ in range(6):
        for q in qs:
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if not ok:
                raise RuntimeError("cv2.imencode failed")
            data = buf.tobytes()
            if len(data) <= max_bytes:
                return data
        h, w = img.shape[:2]
        img = cv2.resize(img, (max(2, int(w * 0.75)), max(2, int(h * 0.75))), interpolation=cv2.INTER_AREA)
    return data


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _kp_list(raw: Any) -> list[tuple[float, float, float]]:
    pts: list[tuple[float, float, float]] = []
    for p in raw or []:
        try:
            x, y = float(p[0]), float(p[1])
            s = float(p[2]) if len(p) > 2 else 1.0
        except (TypeError, ValueError, IndexError):
            pts.append((0.0, 0.0, 0.0))
            continue
        pts.append((x, y, s))
    return pts


def head_boxes_from_keypoints(kps: Sequence[tuple[float, float, float]], w: int, h: int, min_score: float,
                              grow: float = 1.0) -> Optional[tuple[int, int, int, int]]:
    """Квадрат вокруг видимых точек головы; размер — по разбросу точек и ширине плеч."""
    def px(p: tuple[float, float, float]) -> tuple[float, float]:
        x, y = p[0], p[1]
        if max(abs(x), abs(y)) <= 1.5:
            x, y = x * w, y * h
        return x, y

    head = [px(kps[i]) for i in HEAD_KP if i < len(kps) and kps[i][2] >= min_score]
    if not head:
        return None
    xs, ys = [p[0] for p in head], [p[1] for p in head]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    sh = [px(kps[i]) for i in SHOULDERS if i < len(kps) and kps[i][2] >= min_score]
    shoulder_w = abs(sh[0][0] - sh[1][0]) if len(sh) == 2 else 0.0
    side = max(span * 2.2, shoulder_w * 0.75, 0.045 * min(w, h)) * grow
    x1, y1 = int(cx - side / 2), int(cy - side / 2)
    x2, y2 = int(cx + side / 2), int(cy + side / 2)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def blur_regions(img: np.ndarray, regions: Sequence[tuple[int, int, int, int]]) -> np.ndarray:
    """Пикселизация + гаусс по прямоугольникам (in-place на копии)."""
    import cv2  # lazy

    out = img.copy()
    for x1, y1, x2, y2 in regions:
        roi = out[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        rh, rw = roi.shape[:2]
        small = cv2.resize(roi, (max(1, rw // 6), max(1, rh // 6)), interpolation=cv2.INTER_AREA)
        pix = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
        k = max(3, (min(rw, rh) // 4) | 1)
        out[y1:y2, x1:x2] = cv2.GaussianBlur(pix, (k, k), 0)
    return out


def _f(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _hex(bgr: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(bgr[2], bgr[1], bgr[0])


class EvidenceRenderer:
    """Выбор и рендер кадров для всех нарушений одного видео (общие кэши и бюджет времени)."""

    def __init__(self, clip: Any, cfg: dict[str, Any]):
        self.clip = clip
        self.cfg = cfg.get("evidence") or {}
        self.warnings: list[str] = []
        self.elapsed = 0.0
        self.times: list[float] = clip_times(clip)
        self.fps = float(getattr(clip, "fps", 0.0) or 0.0)
        self.duration = float(getattr(clip, "duration", 0.0) or 0.0)
        self._sharp: dict[int, float] = {}
        self._frames: dict[float, tuple[np.ndarray, float]] = {}
        self._warned: set[str] = set()

    # ------------------------------------------------------------------ доступ к кадрам
    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            self.warnings.append(msg)

    def nearest_index(self, t: float) -> Optional[int]:
        return nearest_index(self.times, t)

    def snap(self, t: float) -> float:
        """Время ближайшего существующего кадра: PTS сэмпла из clip.times или кадровая сетка fps."""
        return snap_time(self.times, self.fps, self.duration, t)

    def grab(self, t: float) -> Optional[tuple[np.ndarray, float]]:
        """(кадр BGR полного разрешения, действительный PTS). Кэшируется по PTS."""
        pts = self.snap(t)
        key = round(pts, 4)
        if key in self._frames:
            return self._frames[key]
        img: Optional[np.ndarray] = None
        # A04: frame_at_ts(t) -> (img, фактический PTS); read_exact(t) -> FrameRef(image, timestamp_sec)
        for name in ("frame_at_ts", "frame_at_pts", "frame_with_pts", "read_exact"):
            exact = getattr(self.clip, name, None)
            if img is not None or not callable(exact):
                continue
            try:
                res = exact(pts)
                if isinstance(res, tuple) and len(res) == 2:
                    img, real = ensure_bgr(res[0]), _f(res[1])
                else:
                    img, real = ensure_bgr(getattr(res, "image", None)), _f(getattr(res, "timestamp_sec", None))
                if img is not None and real is not None:
                    pts = real
                elif img is not None:
                    img = None  # кадр без фактического PTS — не годится для доказательства
            except Exception as exc:  # noqa: BLE001
                self._warn_once(name, f"{name} failed: {exc!r}")
        if img is None and callable(getattr(self.clip, "frame_at", None)):
            try:
                img = ensure_bgr(self.clip.frame_at(pts))
            except Exception as exc:  # noqa: BLE001
                self._warn_once("frame_at", f"frame_at failed, using sampled frames: {exc!r}")
        if img is None:
            i = self.nearest_index(pts)
            if i is not None:
                try:
                    img = ensure_bgr(self.clip.frame(i))
                    pts = self.times[i]
                except Exception as exc:  # noqa: BLE001
                    self._warn_once("frame", f"clip.frame failed: {exc!r}")
        if img is None:
            return None
        if len(self._frames) > 24:
            self._frames.pop(next(iter(self._frames)))
        self._frames[key] = (img, float(pts))
        return self._frames[key]

    def _sharpness_at(self, i: int) -> float:
        if i not in self._sharp:
            try:
                img = ensure_bgr(self.clip.frame(i))
                self._sharp[i] = sharpness(img) if img is not None else -1.0
            except Exception:  # noqa: BLE001
                self._sharp[i] = -1.0
        return self._sharp[i]

    # ------------------------------------------------------------------ выбор моментов
    def select_times(self, t_peak: float, t_start: float, t_end: float, degraded: bool) -> list[float]:
        """Политика sharpest (фаза 1): t_peak, самый резкий кадр из t_peak ± window, контекст (начало/конец)."""
        return [t for t, _ in self._sharpest_moments(t_peak, t_start, t_end, degraded)]

    def _sharpest_moments(self, t_peak: float, t_start: float, t_end: float, degraded: bool
                          ) -> list[tuple[float, str]]:
        max_frames = max(1, min(3, int(self.cfg.get("max_frames", 3))))
        min_sep = float(self.cfg.get("min_sep_s", 0.4))
        out = [(self.snap(t_peak), "key")]
        if degraded or max_frames == 1:
            return out
        window = float(self.cfg.get("window_s", 1.0))
        idx = [i for i, t in enumerate(self.times) if abs(t - t_peak) <= window and abs(t - out[0][0]) >= min_sep]
        if idx:
            best = max(idx, key=lambda i: (self._sharpness_at(i), -abs(self.times[i] - t_peak)))
            if self._sharp.get(best, -1.0) >= 0:
                out.append((self.times[best], "sharp"))
        if self.cfg.get("context_frame", True) and len(out) < max_frames:
            gap = float(self.cfg.get("context_min_gap_s", 1.5))
            ctx = t_start if t_peak - t_start >= gap else t_end if t_end - t_peak >= gap else None
            if ctx is not None:
                c = self.snap(ctx)
                if all(abs(c - o) >= min_sep for o, _ in out):
                    out.append((c, "before" if ctx < t_peak else "after"))
        return out[:max_frames]

    def _local_best(self, target: float, radius: float) -> float:
        """Самый резкий сэмпл в ±radius вокруг target (резкость — только локальный тай-брейк)."""
        idx = [i for i, t in enumerate(self.times) if abs(t - target) <= radius]
        if not idx:
            return self.snap(target)
        best = max(idx, key=lambda i: (self._sharpness_at(i), -abs(self.times[i] - target)))
        return self.times[best]

    def _action_moments(self, cand: Any, degraded: bool) -> list[tuple[float, str]]:
        """Политика action: ключевой момент (t_peak) + «до» / «после» реального действия.

        Источники моментов: extra.evidence_t (A09: фактические начало/пик/максимум), для V02 —
        state_after.t_after («стол после ухода»), иначе t_start − before_s и t_end + after_s.
        """
        max_frames = max(1, min(3, int(self.cfg.get("max_frames", 3))))
        min_sep = float(self.cfg.get("min_sep_s", 0.4))
        acfg = self.cfg.get("action") or {}
        before_s, after_s = float(acfg.get("before_s", 1.0)), float(acfg.get("after_s", 2.0))
        local = float(acfg.get("local_s", 0.2))
        t_peak, t_start, t_end = float(cand.t_peak), float(cand.t_start), float(cand.t_end)
        key = self.snap(t_peak)
        out = [(key, "key")]
        if degraded or max_frames == 1:
            return out
        extra = getattr(cand, "extra", {}) or {}
        hi = self.duration if self.duration > 0 else max(t_end + after_s, t_peak)
        # (время, роль, точный момент от upstream?) — моменты A09/state_after уже фактические кадры, их не двигаем
        targets: list[tuple[float, str, bool]] = []
        state = extra.get("state_after") if isinstance(extra.get("state_after"), dict) else {}
        if state.get("t_after") is not None:
            targets.append((float(state["t_after"]), "after", True))
        for t in sorted(float(x) for x in extra.get("evidence_t") or []):
            targets.append((t, "before" if t < key else "after", True))
        targets.append((max(0.0, t_start - before_s), "before", False))
        targets.append((min(hi, max(t_end, t_peak) + after_s), "after", False))
        roles_taken: set[str] = set()
        for t, role, exact in targets:
            if len(out) >= max_frames:
                break
            if role in roles_taken or not 0.0 <= t <= hi:
                continue
            c = self.snap(t) if exact else self._local_best(t, local)
            if all(abs(c - o) >= min_sep for o, _ in out):
                out.append((c, role))
                roles_taken.add(role)
        return [out[0]] + sorted(out[1:])

    def select_moments(self, cand: Any, degraded: bool) -> list[tuple[float, str]]:
        if str(self.cfg.get("policy", "sharpest")).lower() == "action":
            return self._action_moments(cand, degraded)
        return self._sharpest_moments(float(cand.t_peak), float(cand.t_start), float(cand.t_end), degraded)

    # ------------------------------------------------------------------ приватность
    def head_regions(self, cand: Any, img_shape: tuple[int, ...], pts: float) -> list[tuple[int, int, int, int]]:
        if not self.cfg.get("blur_heads", True):
            return []
        h, w = img_shape[:2]
        extra = getattr(cand, "extra", {}) or {}
        min_s = float(self.cfg.get("head_kp_min_score", 0.3))
        dt = abs(pts - float(getattr(cand, "t_peak", pts)))
        grow = 1.0 if dt <= 0.3 else min(2.0, 1.0 + dt)
        regions: list[tuple[int, int, int, int]] = []
        heads = extra.get("head_boxes")
        for box in (heads.values() if isinstance(heads, dict) else heads or []):
            px = to_pixels(box, w, h)
            if px:
                if grow > 1.0:
                    cx, cy = (px[0] + px[2]) / 2, (px[1] + px[3]) / 2
                    hw, hh = (px[2] - px[0]) * grow / 2, (px[3] - px[1]) * grow / 2
                    px = (max(0, int(cx - hw)), max(0, int(cy - hh)), min(w - 1, int(cx + hw)),
                          min(h - 1, int(cy + hh)))
                regions.append(px)
        kp_sets: list[list[tuple[float, float, float]]] = []
        kps = extra.get("keypoints")
        if isinstance(kps, dict):
            kp_sets += [_kp_list(v) for v in kps.values()]
        elif isinstance(kps, list) and kps and isinstance(kps[0], (list, tuple)) and kps[0] and \
                isinstance(kps[0][0], (list, tuple)):
            kp_sets += [_kp_list(v) for v in kps]
        elif isinstance(kps, list) and kps:
            kp_sets.append(_kp_list(kps))
        for pose in extra.get("poses") or []:
            raw = pose.get("keypoints") if isinstance(pose, dict) else getattr(pose, "keypoints", None)
            pt = pose.get("t") if isinstance(pose, dict) else getattr(pose, "t", None)
            if pt is not None and abs(float(pt) - pts) > 0.6:
                continue
            kp_sets.append(_kp_list(raw))
        for kp in kp_sets:
            hb = head_boxes_from_keypoints(kp, w, h, min_s, grow)
            if hb:
                regions.append(hb)
        if str(self.cfg.get("head_fallback", "none")).lower() == "top_frac":
            frac = float(self.cfg.get("head_frac", 0.18))
            persons = [b for k, b in (getattr(cand, "bboxes", {}) or {}).items() if is_person_key(k)]
            persons += list(extra.get("person_bboxes") or [])
            for box in persons:
                px = to_pixels(box, w, h)
                if not px:
                    continue
                if any(r[0] >= px[0] - 2 and r[2] <= px[2] + 2 and r[1] >= px[1] - 2 and r[3] <= px[3] + 2
                       for r in regions):
                    continue
                regions.append((px[0], px[1], px[2], px[1] + max(2, int((px[3] - px[1]) * frac))))
        return regions

    # ------------------------------------------------------------------ UI-оверлей
    def overlay(self, img: np.ndarray, cand: Any, info: dict[str, Any], pts: float, idx: int, n: int,
                draw_boxes: bool, role: str = "key", boxes: Optional[dict[str, Any]] = None
                ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """UI-кадр: тонкие рамки; подписи — легендой под кадром (ui_label_mode=legend), чтобы не закрывать действие."""
        import cv2  # lazy

        from .describe import format_time
        from .draw import TextPainter

        h, w = img.shape[:2]
        painter = TextPainter(tuple(self.cfg.get("font_paths") or ()))
        shapes: list[dict[str, Any]] = []
        canvas_img = img.copy()
        size = max(11, int(round(min(w, h) * 0.032)))
        legend_mode = str(self.cfg.get("ui_label_mode", "legend")).lower() == "legend"
        legend: list[str] = []
        boxes = boxes if boxes is not None else (getattr(cand, "bboxes", {}) or {})
        header_h = size + 12   # текст рисуется на итоговом холсте с полем сверху — смещаем подписи рамок
        if draw_boxes:
            for key, box in boxes.items():
                px = to_pixels(box, w, h)
                if not px:
                    continue
                color = box_color(key)
                th = max(1, int(round(min(w, h) / 320)))
                cv2.rectangle(canvas_img, (px[0], px[1]), (px[2], px[3]), color, th, cv2.LINE_AA)
                label = ru_label(key)
                lsize = max(10, size - 2)
                if legend_mode:
                    num = str(len(legend) + 1)
                    tw = painter.text_width(num, lsize) + 6
                    lx = px[0] if px[0] - tw < 0 else px[0] - tw   # номер — снаружи рамки, слева
                    ly = px[1]
                    cv2.rectangle(canvas_img, (lx, ly), (lx + tw, ly + lsize + 4), color, -1)
                    painter.add(num, (lx + 3, ly + 1 + header_h), lsize, (255, 255, 255))
                    legend.append(f"{num} — {label}")
                else:
                    tw = painter.text_width(label, lsize) + 6
                    ly = px[1] - lsize - 6 if px[1] - lsize - 6 >= 0 else px[1] + 1
                    lx = min(px[0], max(0, w - tw))
                    cv2.rectangle(canvas_img, (lx, ly), (lx + tw, ly + lsize + 5), color, -1)
                    painter.add(label, (lx + 3, ly + 1 + header_h), lsize, (255, 255, 255))
                shapes.append({"kind": "box", "key": str(key), "label": label, "color": _hex(color),
                               "frame_role": role,
                               "bbox": [round(px[0] / w, 4), round(px[1] / h, 4), round(px[2] / w, 4),
                                        round(px[3] / h, 4)]})
        # Поля сверху/снизу — оверлей не закрывает содержимое кадра.
        type_line = f"{cand.type_id} · {info.get('name', cand.type_id)}"
        lines = painter.wrap(type_line, size, w - 12, max_lines=2)
        legend_lines = painter.wrap("; ".join(legend), max(10, size - 2), w - 12, max_lines=2) if legend else []
        footer_lines = lines + legend_lines + [f"скор {cand.confidence:.2f} · признаки возможного нарушения"]
        footer_h = (size + 6) * len(footer_lines) + 8
        sev = str(info.get("severity") or "medium").lower()
        border = SEVERITY_COLOR.get(sev, SEVERITY_COLOR["medium"])
        canvas = np.full((h + header_h + footer_h, w, 3), 24, dtype=np.uint8)
        canvas[header_h:header_h + h] = canvas_img
        cv2.rectangle(canvas, (0, header_h), (w - 1, header_h + h - 1), border, 2)
        mark = {"key": " · ключевой момент", "before": " · до", "after": " · после"}.get(role, "")
        painter.add(f"кадр {idx + 1}/{n} · t = {pts:.2f} с · {format_time(pts)}{mark}", (6, 5), size,
                    (255, 255, 255))
        y = header_h + h + 4
        for i, line in enumerate(footer_lines):
            color = border if i < len(lines) else (210, 210, 210)
            painter.add(line, (6, y), size if i < len(lines) else max(10, size - 2), color)
            y += size + 6
        return painter.flush(canvas), shapes

    # ------------------------------------------------------------------ главный метод
    def render(self, cand: Any, info: dict[str, Any]) -> tuple[list[EvidenceFrame], dict[str, Any]]:
        """→ (кадры для жюри, UI-данные: оверлеи JPEG + фигуры)."""
        t0 = time.perf_counter()
        budget = float(self.cfg.get("budget_s", 20.0))
        degraded = self.elapsed > budget
        if degraded:
            self._warn_once("budget", "evidence budget exceeded: 1 кадр на нарушение без поиска резкости")
        max_bytes = int(self.cfg.get("max_bytes", 200_000))
        qualities = list(self.cfg.get("jpeg_quality") or [88, 80, 72, 64, 56, 48, 40, 32])
        max_side = int(self.cfg.get("max_side", 1280))
        dup_block = float(self.cfg.get("dup_block", 4.0))
        min_sep = float(self.cfg.get("min_sep_s", 0.4))
        moments = self.select_moments(cand, degraded)

        chosen: list[tuple[np.ndarray, float, str]] = []
        thumbs: list[np.ndarray] = []
        for t, role in moments:
            got = self.grab(t)
            if got is None:
                continue
            img, pts = got
            if any(abs(pts - p) < min_sep * 0.5 for _, p, _ in chosen):
                continue
            th = thumb(img)
            if any(is_duplicate(th, o, dup_block) for o in thumbs):
                continue  # визуальный дубль
            chosen.append((img, pts, role))
            thumbs.append(th)
        if not chosen:
            self._warn_once(f"noframe:{cand.type_id}:{cand.t_peak}", f"no frame available for {cand.type_id} "
                            f"@ {cand.t_peak:.1f}s")
            self.elapsed += time.perf_counter() - t0
            return [], {}

        jury: list[EvidenceFrame] = []
        ui_frames: list[dict[str, Any]] = []
        all_shapes: list[dict[str, Any]] = []
        n_ui = int(self.cfg.get("ui_overlay_frames", 1)) if self.cfg.get("ui_overlays", True) else 0
        box_dt = float(self.cfg.get("box_max_dt_s", 1.5))
        any_blur = False
        state = (getattr(cand, "extra", {}) or {}).get("state_after")
        after_boxes = state.get("valuables_bboxes") if isinstance(state, dict) else None
        for k, (img, pts, role) in enumerate(chosen):
            regions = self.head_regions(cand, img.shape, pts)
            any_blur = any_blur or bool(regions)
            clean = blur_regions(img, regions) if regions else img
            jury.append(EvidenceFrame(timestamp_sec=round(float(pts), 3),
                                      image_base64=b64(encode_jpeg(clean, max_bytes, qualities, max_side))))
            if k < n_ui and not degraded:
                boxes = None
                if role == "after" and isinstance(after_boxes, dict) and after_boxes:
                    boxes, draw = after_boxes, bool(self.cfg.get("draw_boxes", True))  # подсветка оставленного
                else:
                    draw = bool(self.cfg.get("draw_boxes", True)) and abs(pts - float(cand.t_peak)) <= box_dt
                ov, shapes = self.overlay(clean, cand, info, pts, k, len(chosen), draw, role, boxes)
                ui_frames.append({"timestamp_sec": round(float(pts), 3), "role": role,
                                  "image_base64": b64(encode_jpeg(ov, max_bytes, qualities, max_side))})
                all_shapes.extend(sh for sh in shapes if sh not in all_shapes)
            for r in regions:
                h, w = img.shape[:2]
                all_shapes.append({"kind": "blur", "timestamp_sec": round(float(pts), 3),
                                   "bbox": [round(r[0] / w, 4), round(r[1] / h, 4), round(r[2] / w, 4),
                                            round(r[3] / h, 4)]})
        if not any_blur and self.cfg.get("blur_heads", True):
            self._warn_once("noblur", "privacy: нет ключевых точек головы — головы на кадрах не размыты "
                                      "(эвристика верха bbox выключена)")
        self.elapsed += time.perf_counter() - t0
        return jury, {"frames": ui_frames, "shapes": all_shapes}
