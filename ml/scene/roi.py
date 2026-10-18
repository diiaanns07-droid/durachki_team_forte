"""Работа с кадрами: качество, ROI-кропы с компенсацией сдвига камеры, CS-SSIM, края, тёмная площадь.

cv2 импортируется лениво. Яркость ROI нормируется (z-score), поэтому общий сдвиг освещения
и мягкие тени слабо влияют на признаки; окклюзия людьми маскируется.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .geometry import BBox, bbox_to_px


def _cv2() -> Any:
    import cv2

    return cv2


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.float32)
    cv2 = _cv2()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)


def resize_w(img: np.ndarray, width: int) -> np.ndarray:
    cv2 = _cv2()
    h, w = img.shape[:2]
    if w == width:
        return img
    nh = max(8, int(round(h * width / float(w))))
    return cv2.resize(img, (width, nh), interpolation=cv2.INTER_AREA)


def frame_quality(gray_small: np.ndarray) -> dict[str, float]:
    """Качество кадра 0..1: резкость (дисперсия лапласиана), экспозиция, контраст."""
    cv2 = _cv2()
    g = gray_small.astype(np.float32)
    lap = float(cv2.Laplacian(g, cv2.CV_32F).var())
    sharp = float(1.0 - np.exp(-lap / 60.0))
    mean = float(g.mean())
    expo = float(np.clip(min((mean - 15.0) / 35.0, (240.0 - mean) / 35.0), 0.0, 1.0))
    std = float(g.std())
    contrast = float(np.clip((std - 4.0) / 16.0, 0.0, 1.0))
    q = float(min(sharp * 1.2, expo, contrast, 1.0))
    return {"quality": max(0.0, q), "sharpness": sharp, "exposure": expo, "contrast": contrast}


def phase_shift(ref: np.ndarray, cur: np.ndarray, max_shift_px: float, min_response: float) -> tuple[float, float]:
    """Глобальный сдвиг cur относительно ref (px малого кадра); 0, если отклик слабый."""
    cv2 = _cv2()
    if ref.shape != cur.shape:
        return 0.0, 0.0
    win = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(ref.astype(np.float32), cur.astype(np.float32), win)
    if resp < min_response or not np.isfinite(dx) or not np.isfinite(dy):
        return 0.0, 0.0
    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return 0.0, 0.0
    if np.hypot(dx, dy) > max_shift_px:
        return 0.0, 0.0
    return float(dx), float(dy)


def crop(frame: np.ndarray, b: BBox, shift: tuple[float, float] = (0.0, 0.0), out_wh: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Кроп нормированного бокса (со сдвигом в норм. координатах), края — replicate."""
    cv2 = _cv2()
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = b
    sx, sy = shift
    px1, py1 = int(round((x1 + sx) * w)), int(round((y1 + sy) * h))
    px2, py2 = int(round((x2 + sx) * w)), int(round((y2 + sy) * h))
    px2, py2 = max(px2, px1 + 2), max(py2, py1 + 2)
    pad = max(0, -px1, -py1, px2 - w, py2 - h)
    src = frame
    if pad > 0:
        src = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        px1, py1, px2, py2 = px1 + pad, py1 + pad, px2 + pad, py2 + pad
    out = src[py1:py2, px1:px2]
    if out_wh is not None:
        out = cv2.resize(out, out_wh, interpolation=cv2.INTER_AREA)
    return out


def mask_from_boxes(roi: BBox, boxes: list[BBox], wh: tuple[int, int], pad: float = 0.0) -> np.ndarray:
    """Маска валидных пикселей ROI (1 — не закрыт людьми)."""
    w, h = wh
    m = np.ones((h, w), dtype=np.float32)
    rx1, ry1, rx2, ry2 = roi
    rw, rh = max(rx2 - rx1, 1e-6), max(ry2 - ry1, 1e-6)
    for b in boxes:
        bx1 = (b[0] - pad - rx1) / rw
        by1 = (b[1] - pad - ry1) / rh
        bx2 = (b[2] + pad - rx1) / rw
        by2 = (b[3] + pad - ry1) / rh
        if bx2 <= 0 or by2 <= 0 or bx1 >= 1 or by1 >= 1:
            continue
        x1, y1, x2, y2 = bbox_to_px((max(bx1, 0), max(by1, 0), min(bx2, 1), min(by2, 1)), w, h)
        m[y1:y2, x1:x2] = 0.0
    return m


def occlusion_fraction(roi: BBox, boxes: list[BBox], pad: float = 0.0, res: int = 24) -> float:
    return float(1.0 - mask_from_boxes(roi, boxes, (res, res), pad).mean())


def znorm(g: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    m = mask if mask is not None and mask.sum() > 8 else None
    vals = g[m > 0] if m is not None else g
    mu, sd = float(vals.mean()), float(vals.std())
    return (g - mu) / max(sd, 4.0)


def cs_ssim(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None, win: int = 7) -> float:
    """Контрастно-структурная часть SSIM (без яркостного члена) по маске валидных пикселей."""
    cv2 = _cv2()
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    k = (win, win)
    mu_a, mu_b = cv2.blur(a, k), cv2.blur(b, k)
    va = cv2.blur(a * a, k) - mu_a * mu_a
    vb = cv2.blur(b * b, k) - mu_b * mu_b
    cov = cv2.blur(a * b, k) - mu_a * mu_b
    c2 = 0.03 ** 2 * 4.0  # данные z-нормированы (~±2)
    cs = (2 * cov + c2) / (va + vb + c2)
    if mask is not None:
        mk = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        if mk.sum() < 16:
            return float("nan")
        return float(cs[mk].mean())
    return float(cs.mean())


def grads(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cv2 = _cv2()
    gx = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy


def edge_density(g: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    gx, gy = grads(g)
    mag = np.hypot(gx, gy)
    if mask is not None and mask.sum() > 8:
        return float(mag[mask > 0].mean())
    return float(mag.mean())


def vertical_edge_energy(g: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Энергия преимущественно вертикальных кромок (|gx| − |gy|)+."""
    gx, gy = grads(g)
    v = np.clip(np.abs(gx) - np.abs(gy), 0.0, None)
    if mask is not None and mask.sum() > 8:
        return float(v[mask > 0].mean())
    return float(v.mean())


def dark_fraction(gz: np.ndarray, mask: Optional[np.ndarray] = None, thr: float = -1.0) -> float:
    """Доля «тёмных» пикселей z-нормированного ROI (ниже thr σ от среднего)."""
    d = gz < thr
    if mask is not None and mask.sum() > 8:
        return float(d[mask > 0].mean())
    return float(d.mean())


def masked_median(stack: list[np.ndarray], masks: list[np.ndarray]) -> np.ndarray:
    arr = np.stack([s.astype(np.float32) for s in stack])
    mk = np.stack([m > 0 for m in masks])
    data = np.where(mk, arr, np.nan)
    import warnings

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)  # пиксель закрыт людьми во всех кадрах → фолбэк ниже
        med = np.nanmedian(data, axis=0)
    fallback = np.median(arr, axis=0)
    return np.where(np.isfinite(med), med, fallback).astype(np.float32)


def ring_mask(outer: BBox, inner: BBox, wh: tuple[int, int]) -> np.ndarray:
    """Маска кольца outer∖inner в координатах кропа outer."""
    w, h = wh
    m = np.ones((h, w), dtype=np.float32)
    ox1, oy1, ox2, oy2 = outer
    ow, oh = max(ox2 - ox1, 1e-6), max(oy2 - oy1, 1e-6)
    ib = ((inner[0] - ox1) / ow, (inner[1] - oy1) / oh, (inner[2] - ox1) / ow, (inner[3] - oy1) / oh)
    x1, y1, x2, y2 = bbox_to_px((max(ib[0], 0), max(ib[1], 0), min(ib[2], 1), min(ib[3], 1)), w, h)
    m[y1:y2, x1:x2] = 0.0
    return m
