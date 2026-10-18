"""Mock-бэкенд (SENTINEL_MOCK=1): синтетическая поза-шаблон внутри bbox + скрипт действий.

Скрипт (config mock.script или $SENTINEL_POSE_MOCK_SCRIPT): "action:t0-t1[@track_id];..."
action ∈ phone_to_ear | eating | hands_hidden | facing_away. Веса и кадры не нужны.
"""
from __future__ import annotations

import os
import zlib
from typing import Any, Optional, Sequence

import numpy as np

# шаблон «человек лицом к камере» в координатах bbox (u, v ∈ 0..1), порядок COCO-17
TEMPLATE = np.array([
    [0.50, 0.12], [0.54, 0.10], [0.46, 0.10], [0.58, 0.11], [0.42, 0.11],
    [0.68, 0.25], [0.32, 0.25], [0.74, 0.42], [0.26, 0.42], [0.62, 0.55], [0.38, 0.55],
    [0.60, 0.60], [0.40, 0.60], [0.60, 0.78], [0.40, 0.78], [0.60, 0.95], [0.40, 0.95],
], dtype=np.float32)
BASE_SCORE = np.array([0.9, 0.85, 0.85, 0.7, 0.7, 0.9, 0.9, 0.85, 0.85, 0.8, 0.8,
                       0.6, 0.6, 0.4, 0.4, 0.2, 0.2], dtype=np.float32)


def parse_script(script: str) -> list[tuple[str, float, float, Optional[int]]]:
    out: list[tuple[str, float, float, Optional[int]]] = []
    for part in (script or "").split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        action, rest = part.split(":", 1)
        tid: Optional[int] = None
        if "@" in rest:
            rest, tid_s = rest.split("@", 1)
            tid = int(tid_s)
        t0, t1 = (float(x) for x in rest.split("-", 1))
        out.append((action.strip(), t0, t1, tid))
    return out


def scripted_pose(t: float, tid: Optional[int], actions: Sequence[tuple[str, float, float, Optional[int]]]
                  ) -> tuple[np.ndarray, np.ndarray]:
    uv = TEMPLATE.copy()
    sc = BASE_SCORE.copy()
    for action, t0, t1, a_tid in actions:
        if a_tid is not None and tid != a_tid:
            continue
        if not (t0 <= t <= t1):
            continue
        if action == "phone_to_ear":
            uv[10] = [0.43, 0.14]
            uv[8] = [0.28, 0.24]
        elif action == "eating":
            phase = ((t - t0) % 4.0) / 4.0           # цикл 4 с: 0.6 с у рта
            if phase < 0.15:
                uv[10] = [0.49, 0.15]
                uv[8] = [0.30, 0.33]
        elif action == "hands_hidden":
            sc[9] = sc[10] = 0.05
        elif action == "facing_away":
            uv[[5, 6]] = uv[[6, 5]]
            uv[[7, 8]] = uv[[8, 7]]
            uv[[9, 10]] = uv[[10, 9]]
            sc[:5] = 0.1
    return uv, sc


class MockPose:
    name = "mock"

    def __init__(self, cfg: dict[str, Any]) -> None:
        mc = cfg.get("mock", {}) or {}
        self.seed = int(mc.get("seed", 7))
        self.jitter = float(mc.get("jitter", 0.004))
        self.actions = parse_script(os.environ.get("SENTINEL_POSE_MOCK_SCRIPT", mc.get("script", "")))
        self.batch_size = 64

    def infer(self, items: Sequence[tuple[np.ndarray, np.ndarray]],
              meta: Optional[Sequence[tuple[float, Optional[int]]]] = None) -> list[tuple[np.ndarray, np.ndarray]]:
        out = []
        for k, (frame, box) in enumerate(items):
            t, tid = meta[k] if meta is not None else (0.0, None)
            uv, sc = scripted_pose(float(t), tid, self.actions)
            tid_i = tid if isinstance(tid, int) else (zlib.crc32(str(tid).encode()) if tid is not None else 0)
            rng = np.random.default_rng((self.seed, int(round(float(t) * 1000)), abs(tid_i) + 10_000))
            x1, y1, x2, y2 = (float(v) for v in box)
            h_img = float(frame.shape[0]) if hasattr(frame, "shape") and len(frame.shape) >= 2 else 1.0
            uv = uv + rng.normal(0.0, self.jitter * h_img / max(y2 - y1, 1e-6), uv.shape).astype(np.float32)
            kp = np.stack([x1 + uv[:, 0] * (x2 - x1), y1 + uv[:, 1] * (y2 - y1)], axis=1)
            out.append((kp.astype(np.float32), sc.astype(np.float32)))
        return out
