"""Отрисовка русских подписей на кадрах: Pillow + TTF с кириллицей, иначе cv2 + транслитерация.

Порядок поиска шрифта: env ``SENTINEL_FONT`` → ``evidence.font_paths`` из config.yaml →
системные DejaVu/Liberation/Noto (Linux), Arial/Segoe/Tahoma (Windows), Arial (macOS) →
шрифт matplotlib (если установлен). Без шрифта текст транслитерируется и рисуется cv2.putText.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional, Sequence

import numpy as np

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/verdana.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
)

_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z", "и": "i",
    "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
    "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "",
    "э": "e", "ю": "yu", "я": "ya", "ә": "a", "ғ": "g", "қ": "q", "ң": "n", "ө": "o", "ұ": "u", "ү": "u",
    "һ": "h", "і": "i", "«": '"', "»": '"', "—": "-", "–": "-", "…": "...", "·": "|", "№": "N",
}


def transliterate(text: str) -> str:
    out = []
    for ch in text:
        low = ch.lower()
        if low in _TRANSLIT:
            tr = _TRANSLIT[low]
            out.append(tr.capitalize() if ch != low and tr else tr)
        elif ord(ch) < 128:
            out.append(ch)
        else:
            out.append("?")
    return "".join(out)


def _matplotlib_font() -> Optional[str]:
    try:
        import matplotlib  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return None
    path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
    return path if os.path.exists(path) else None


@lru_cache(maxsize=4)
def find_font_path(extra_paths: tuple[str, ...] = ()) -> Optional[str]:
    env = os.environ.get("SENTINEL_FONT")
    for path in ((env,) if env else ()) + tuple(extra_paths) + _FONT_CANDIDATES:
        if path and os.path.exists(path):
            return path
    return _matplotlib_font()


@lru_cache(maxsize=32)
def _pil_font(path: str, size: int) -> Any:
    from PIL import ImageFont  # lazy

    return ImageFont.truetype(path, size)


class TextPainter:
    """Копит текстовые элементы и рисует их за один проход (Pillow) или через cv2 (fallback)."""

    def __init__(self, font_paths: Sequence[str] = ()):
        self.font_path: Optional[str] = None
        try:
            import PIL  # noqa: F401  # lazy

            self.font_path = find_font_path(tuple(font_paths))
        except ImportError:
            self.font_path = None
        self.items: list[tuple[str, tuple[int, int], int, tuple[int, int, int]]] = []

    @property
    def cyrillic(self) -> bool:
        return self.font_path is not None

    def text_width(self, text: str, size: int) -> int:
        if self.font_path:
            font = _pil_font(self.font_path, size)
            left, _, right, _ = font.getbbox(text)
            return int(right - left)
        import cv2  # lazy

        scale = size / 22.0
        (w, _), _ = cv2.getTextSize(transliterate(text), cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        return int(w)

    def fit(self, text: str, size: int, max_width: int, min_size: int = 10) -> int:
        """Подбирает кегль, чтобы строка влезла по ширине (не меньше min_size)."""
        while size > min_size and self.text_width(text, size) > max_width:
            size -= 1
        return size

    def wrap(self, text: str, size: int, max_width: int, max_lines: int = 2) -> list[str]:
        words = text.split()
        lines: list[str] = []
        cur = ""
        for w in words:
            trial = f"{cur} {w}".strip()
            if self.text_width(trial, size) <= max_width or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            last = lines[-1]
            while last and self.text_width(last + "…", size) > max_width:
                last = last[:-1]
            lines[-1] = last.rstrip() + "…"
        return lines

    def add(self, text: str, org: tuple[int, int], size: int, color: tuple[int, int, int]) -> None:
        """org — левый верхний угол строки; color — BGR."""
        self.items.append((text, (int(org[0]), int(org[1])), int(size), color))

    def flush(self, img: np.ndarray) -> np.ndarray:
        if not self.items:
            return img
        if self.font_path:
            from PIL import Image, ImageDraw  # lazy

            pil = Image.fromarray(img[:, :, ::-1].copy())
            draw = ImageDraw.Draw(pil)
            for text, org, size, color in self.items:
                font = _pil_font(self.font_path, size)
                draw.text(org, text, font=font, fill=(color[2], color[1], color[0]))
            img = np.ascontiguousarray(np.asarray(pil)[:, :, ::-1])
        else:
            import cv2  # lazy

            for text, org, size, color in self.items:
                scale = size / 22.0
                baseline_y = org[1] + int(size * 0.8)
                cv2.putText(img, transliterate(text), (org[0], baseline_y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                            color, 1, cv2.LINE_AA)
        self.items = []
        return img
