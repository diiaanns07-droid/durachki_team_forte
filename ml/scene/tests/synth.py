"""Синтетическая касса сверху: стол, сейф (закрыт/открыт), дверь, купюры, «люди» — без реальных данных."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import cv2
import numpy as np

from ml.scene._contract import Detection, Track, Zone

W, H = 232, 424  # портрет, как 464×848 / 2

SAFE = (0.68, 0.30, 0.90, 0.46)
DOOR = (0.02, 0.05, 0.22, 0.30)
DESK = (0.10, 0.55, 0.62, 0.78)
CASHIER_AREA = (0.05, 0.35, 0.95, 0.95)
CLIENT_AREA = (0.05, 0.0, 0.95, 0.33)
BANKNOTE = (0.30, 0.62, 0.42, 0.68)


def poly(b: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    return [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]


def zones_default() -> list[Zone]:
    return [
        Zone("safe", poly(SAFE)),
        Zone("door", poly(DOOR)),
        Zone("desk", poly(DESK)),
        Zone("cashier_area", poly(CASHIER_AREA)),
        Zone("client_area", poly(CLIENT_AREA)),
    ]


def _px(b: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    return int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H)


_rng = np.random.default_rng(7)
_TEXTURE = cv2.GaussianBlur((_rng.normal(0, 18, (H, W))).astype(np.float32), (3, 3), 0)


def base_frame(safe_open: bool = False, door_open: bool = False, cash: bool = False) -> np.ndarray:
    img = np.full((H, W, 3), 118, np.float32)
    img += _TEXTURE[..., None]
    for k in range(0, H, 37):
        img[k : k + 1, :, :] -= 25  # плитка пола
    x1, y1, x2, y2 = _px(DESK)
    img[y1:y2, x1:x2] = (150, 175, 190)  # светлое дерево (BGR)
    img[y1:y2, x1:x2] += _TEXTURE[y1:y2, x1:x2, None] * 0.4
    sx1, sy1, sx2, sy2 = _px(SAFE)
    img[sy1:sy2, sx1:sx2] = 135
    img[sy1:sy2, sx1:sx2] += _TEXTURE[sy1:sy2, sx1:sx2, None] * 0.5
    cv2.rectangle(img, (sx1 + 3, sy1 + 3), (sx2 - 3, sy2 - 3), (80, 80, 80), 2)
    cv2.circle(img, ((sx1 + sx2) // 2 + 8, (sy1 + sy2) // 2), 4, (60, 60, 60), -1)
    if safe_open:
        # тёмный проём и открытая дверца, выступающая за контур слева
        img[sy1 + 5 : sy2 - 5, sx1 + 5 : sx2 - 5] = 28
        cv2.line(img, (sx1 + 8, (sy1 + sy2) // 2), (sx2 - 8, (sy1 + sy2) // 2), (90, 90, 90), 2)  # полка
        dw = int((sx2 - sx1) * 0.55)
        img[sy1:sy2, sx1 - dw : sx1] = 175
        cv2.line(img, (sx1 - dw, sy1), (sx1 - dw, sy2), (40, 40, 40), 2)
    dx1, dy1, dx2, dy2 = _px(DOOR)
    if door_open:
        img[dy1:dy2, dx1:dx2] = (60, 70, 70)
        cv2.line(img, (dx2, dy1), (dx2 + 25, dy1 + 30), (30, 30, 30), 3)
    else:
        img[dy1:dy2, dx1:dx2] = (190, 200, 205)
        cv2.rectangle(img, (dx1 + 2, dy1 + 2), (dx2 - 2, dy2 - 2), (120, 120, 120), 2)
    if cash:
        bx1, by1, bx2, by2 = _px(BANKNOTE)
        img[by1:by2, bx1:bx2] = (190, 150, 110)  # пастельно-голубая купюра
    return np.clip(img, 0, 255).astype(np.uint8)


def draw_person(img: np.ndarray, b: tuple[float, float, float, float]) -> None:
    x1, y1, x2, y2 = _px(b)
    cv2.ellipse(img, ((x1 + x2) // 2, (y1 + y2) // 2), (max(2, (x2 - x1) // 2), max(2, (y2 - y1) // 2)), 0, 0, 360, (40, 35, 70), -1)
    cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), max(2, (x2 - x1) // 4), (20, 20, 20), -1)


@dataclass
class FakeClip:
    frames: list[np.ndarray]
    times: list[float]
    path: str = "synthetic.mov"
    fps: float = 30.0
    quality: Any = None
    width: int = W
    height: int = H
    duration: float = field(default=0.0)

    def __post_init__(self) -> None:
        self.duration = self.times[-1] + 0.2 if self.times else 0.0
        if self.frames:
            self.height, self.width = self.frames[0].shape[:2]

    def frame(self, i: int) -> np.ndarray:
        return self.frames[i]

    def frame_at(self, t: float) -> np.ndarray:
        return self.frames[int(np.argmin(np.abs(np.asarray(self.times) - t)))]


def make_clip(
    duration: float = 10.0,
    fps: float = 5.0,
    safe_open: Callable[[float], bool] = lambda t: False,
    door_open: Callable[[float], bool] = lambda t: False,
    cash: Callable[[float], bool] = lambda t: False,
    persons: Optional[Callable[[float], list[tuple[float, float, float, float]]]] = None,
    post: Optional[Callable[[np.ndarray, float, int], np.ndarray]] = None,
) -> FakeClip:
    times = [round(k / fps, 4) for k in range(int(duration * fps))]
    frames = []
    for i, t in enumerate(times):
        f = base_frame(safe_open(t), door_open(t), cash(t))
        for b in (persons(t) if persons else []):
            draw_person(f, b)
        if post is not None:
            f = post(f, t, i)
        frames.append(f)
    return FakeClip(frames=frames, times=times)


def person_track(tid: int, boxes: dict[float, tuple[float, float, float, float]], role: str = "unknown", conf: float = 0.0) -> Track:
    return Track(track_id=tid, label="person", role=role, role_conf=conf, boxes=sorted(boxes.items()))


def person_dets(tracks: list[Track]) -> list[Detection]:
    return [Detection(t=t, label="person", bbox=b, score=0.9, track_id=tr.track_id) for tr in tracks for t, b in tr.boxes]


def affine_point(m: np.ndarray, x: float, y: float) -> tuple[float, float]:
    px, py = x * W, y * H
    nx = m[0, 0] * px + m[0, 1] * py + m[0, 2]
    ny = m[1, 0] * px + m[1, 1] * py + m[1, 2]
    return float(nx / W), float(ny / H)


def affine_zones(m: np.ndarray, zones: list[Zone]) -> list[Zone]:
    return [Zone(z.name, [affine_point(m, x, y) for x, y in z.polygon], z.conf) for z in zones]
