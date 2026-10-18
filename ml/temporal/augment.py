"""Аугментация «другой угол камеры» для извлечения обучающих признаков.

Одна случайная перспектива/поворот/кроп/отражение/яркость на ВСЁ видео (как будто камеру повесили иначе).
Если в ml.ingest.augment есть фабрика ``make_view_augmenter(seed)`` / ``random_view_augmenter(seed)`` —
используем её (единая аугментация на весь проект), иначе — своя реализация ниже.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

Augmenter = Callable[[Any], Any]


def _ingest_factory() -> Optional[Callable[[int], Augmenter]]:
    try:
        from ml.ingest import augment as ing  # type: ignore
    except ImportError:
        return None
    for name in ("make_view_augmenter", "random_view_augmenter", "make_augmenter"):
        fn = getattr(ing, name, None)
        if callable(fn):
            return fn
    if hasattr(ing, "ClipAugmenter") and hasattr(ing, "apply"):  # API A04: одни параметры на весь клип
        def factory(seed: int) -> Augmenter:
            import numpy as np

            ca = ing.ClipAugmenter(seed=seed)
            aug = ca.sample()
            noise_rng = np.random.default_rng(seed + 1)
            return lambda frame: ing.apply(frame, aug, noise_rng)
        return factory
    return None


def make_augmenter(seed: int, cfg: dict, prefer_ingest: bool = True) -> Augmenter:
    if prefer_ingest:
        fac = _ingest_factory()
        if fac is not None:
            try:
                return fac(seed)
            except Exception:
                pass
    return ViewAugmenter(seed, cfg)


class ViewAugmenter:
    """Фиксированное на видео случайное преобразование BGR-кадра (размер кадра сохраняется)."""

    def __init__(self, seed: int, cfg: dict) -> None:
        import numpy as np

        a = cfg.get("augment", cfg)
        rng = np.random.default_rng(seed)
        self.shift = float(a.get("max_corner_shift", 0.12))
        self.rot = float(rng.uniform(-1, 1) * a.get("max_rotate_deg", 12))
        self.crop = float(rng.uniform(a.get("min_crop", 0.8), 1.0))
        self.flip = bool(rng.random() < a.get("flip_prob", 0.5))
        self.brightness = float(rng.uniform(-1, 1) * a.get("brightness", 0.25))
        self.contrast = float(1.0 + rng.uniform(-1, 1) * a.get("contrast", 0.25))
        self.corners = rng.uniform(-1, 1, size=(4, 2)) * self.shift
        self.offset = rng.uniform(0, 1, size=2)
        self._M: Any = None
        self._shape: Optional[tuple[int, int]] = None

    def _matrix(self, h: int, w: int) -> Any:
        import cv2
        import numpy as np

        if self._M is not None and self._shape == (h, w):
            return self._M
        cw, ch = w * self.crop, h * self.crop
        ox, oy = (w - cw) * self.offset[0], (h - ch) * self.offset[1]
        src = np.float32([[ox, oy], [ox + cw, oy], [ox + cw, oy + ch], [ox, oy + ch]])
        src = src + (self.corners * np.float32([cw, ch])).astype(np.float32)
        dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        M = cv2.getPerspectiveTransform(src.astype(np.float32), dst)
        R = cv2.getRotationMatrix2D((w / 2, h / 2), self.rot, 1.0)
        R3 = np.vstack([R, [0, 0, 1]])
        self._M, self._shape = R3 @ M, (h, w)
        return self._M

    def __call__(self, frame: Any) -> Any:
        import cv2
        import numpy as np

        h, w = frame.shape[:2]
        out = cv2.warpPerspective(frame, self._matrix(h, w), (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)
        if self.flip:
            out = out[:, ::-1]
        out = out.astype(np.float32) * self.contrast + 255.0 * self.brightness * 0.5
        return np.clip(out, 0, 255).astype(np.uint8)
