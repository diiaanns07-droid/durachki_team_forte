"""Общий интерфейс бэкендов детекции и утилиты препроцессинга."""
from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np

from ..boxes import RawDets


class FrameDetector(Protocol):
    name: str

    def predict(self, frames: Sequence[np.ndarray]) -> list[RawDets]:
        """frames — BGR uint8 (H, W, 3); возвращает по RawDets на кадр, боксы нормированы к кадру."""
        ...


def resolve_device(pref: str = "auto") -> str:
    if pref in ("cpu", "cuda") or pref.startswith("cuda:"):
        if pref.startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    return "cpu"
            except Exception:  # noqa: BLE001
                return "cpu"
        return pref
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def group_by_shape(frames: Sequence[np.ndarray]) -> dict[tuple[int, ...], list[int]]:
    groups: dict[tuple[int, ...], list[int]] = {}
    for i, f in enumerate(frames):
        groups.setdefault(tuple(f.shape), []).append(i)
    return groups


def torch_preprocess(frames: Sequence[np.ndarray], size: tuple[int, int], device: str, dtype: Any,
                     rescale: float = 1.0 / 255.0, mean: Sequence[float] | None = None,
                     std: Sequence[float] | None = None, pad_square: bool = False,
                     pad_value: float = 0.5) -> Any:
    """BGR uint8 кадры одного размера → тензор (B, 3, H, W) на device: RGB, resize (bilinear+antialias),
    rescale, normalize. pad_square — паддинг справа/снизу до квадрата (как у OWLv2)."""
    import torch
    import torch.nn.functional as F

    arr = np.ascontiguousarray(np.stack(frames)[..., ::-1])  # BGR→RGB
    x = torch.from_numpy(arr).to(device, non_blocking=True)
    x = x.permute(0, 3, 1, 2).float() * float(rescale)
    if pad_square:
        h, w = x.shape[-2:]
        s = max(h, w)
        if (h, w) != (s, s):
            x = F.pad(x, (0, s - w, 0, s - h), value=float(pad_value))
    if tuple(x.shape[-2:]) != tuple(size):
        down = x.shape[-2] > size[0] or x.shape[-1] > size[1]
        x = F.interpolate(x, size=tuple(size), mode="bilinear", align_corners=False, antialias=bool(down))
    if mean is not None and std is not None:
        m = torch.tensor(list(mean), device=device).view(1, 3, 1, 1)
        s_ = torch.tensor(list(std), device=device).view(1, 3, 1, 1)
        x = (x - m) / s_
    return x.to(dtype)
