"""Ввод-вывод: дамп/загрузка состояний (JSON), простой cv2-клип (если ml.ingest недоступен), имена видео."""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .types import SceneStateEx, state_from_dict, state_to_dict

STATES_FILENAMES = ("scene_states.json", "scene.json", "states.json")


def normalize_video_name(name: str) -> str:
    """'Х-3.MOV' (кириллица) → 'x-3'; 'X-1' → 'x-1'. Для сопоставления разметки и файлов."""
    s = unicodedata.normalize("NFC", Path(str(name).strip()).name)
    s = s.replace("Х", "X").replace("х", "x")
    stem = s.rsplit(".", 1)[0] if "." in s else s
    return stem.strip().lower().replace(" ", "")


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, (np.floating,)):
        f = float(v)
        return f if np.isfinite(f) else None
    if isinstance(v, float):
        return v if np.isfinite(v) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def dump_states(states: Sequence[Any], path: str | Path, trace: Optional[dict[str, Any]] = None, meta: Optional[dict[str, Any]] = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {"version": 1, "states": [state_to_dict(s) for s in states]}
    if trace is not None:
        doc["trace"] = _jsonable({k: v for k, v in trace.items() if k in ("raw", "conf_raw", "provenance", "info", "warnings", "rois", "events", "intervals")})
    if meta:
        doc["meta"] = _jsonable(meta)
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return p


def find_states_file(cache: str | Path) -> Optional[Path]:
    c = Path(cache)
    if c.is_file():
        return c
    for name in STATES_FILENAMES:
        if (c / name).is_file():
            return c / name
    for f in sorted(c.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and isinstance(d.get("states"), list):
            return f
    return None


def load_states(path: str | Path) -> tuple[list[SceneStateEx], dict[str, Any]]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(d, list):
        return [state_from_dict(x) for x in d], {}
    states = [state_from_dict(x) for x in d.get("states", [])]
    return states, {"trace": d.get("trace", {}), "meta": d.get("meta", {})}


class SimpleClip:
    """Минимальный клип по контракту VideoClip (cv2), когда ml.ingest недоступен (обучение/отладка)."""

    def __init__(self, path: str, sample_fps: float = 5.0) -> None:
        import cv2

        self.path = str(path)
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise IOError(f"cannot open {path}")
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frames: list[np.ndarray] = []
        self._all_t: list[float] = []
        # переменный fps: время кадра — из PTS (CAP_PROP_POS_MSEC), а не k/fps
        next_t, k, last_t = 0.0, 0, 0.0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            t = pts if pts > 0 or k == 0 else k / self.fps
            last_t = t
            if t + 1e-6 >= next_t:
                frames.append(fr)
                self._all_t.append(round(t, 4))
                next_t += 1.0 / sample_fps
            k += 1
        cap.release()
        self._frames = frames
        self.times = list(self._all_t)
        self.duration = last_t + (1.0 / self.fps if self.fps else 0.0)
        h, w = (frames[0].shape[:2] if frames else (0, 0))
        self.width, self.height = int(w), int(h)

    def frame(self, i: int) -> np.ndarray:
        return self._frames[i]

    def frame_at(self, t: float) -> np.ndarray:
        j = int(np.argmin(np.abs(np.asarray(self.times) - t)))
        return self._frames[j]
