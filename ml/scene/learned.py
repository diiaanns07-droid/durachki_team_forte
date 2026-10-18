"""Обучаемый вариант (опционально): логрегрессия / маленький MLP на признаках ROI.

Модель — JSON (без pickle): {"version": 1, "signals": {name: {"type": "lr"|"mlp", "features": [...],
"mean": [...], "std": [...], ...}}}. Обучение — ``python -m ml.scene.train_signals`` по проверенным меткам.
Нет файла / SENTINEL_MOCK=1 / learned.enabled=false → используется только ручная фузия.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import is_mock

# имена признаков для каждого обучаемого сигнала (все берутся из trace["features"][signal])
FEATURES: dict[str, list[str]] = {
    "safe_open": ["change", "change_b", "state_b", "dark_rel", "edge", "ring_vert", "visibility", "quality", "det_lo", "det_c"],
    "door_open": ["change", "change_b", "state_b", "edge", "ring_vert", "visibility", "quality", "det_lo", "det_c", "traffic_lo", "traffic_c"],
    "cash_visible": ["det_score", "colour_lo", "colour_c", "visibility", "quality"],
}
TRAINABLE = tuple(FEATURES)
_REPO_ROOT = Path(__file__).resolve().parents[2]


def model_path(cfg: dict[str, Any]) -> Path:
    p = Path(os.environ.get("SENTINEL_SCENE_MODEL") or cfg["learned"]["path"])
    return p if p.is_absolute() else _REPO_ROOT / p


def load_model(cfg: dict[str, Any]) -> Optional[dict[str, Any]]:
    if is_mock() or not cfg.get("learned", {}).get("enabled", True):
        return None
    p = model_path(cfg)
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    sigs = doc.get("signals") if isinstance(doc, dict) else None
    return sigs if isinstance(sigs, dict) else None


def feature_matrix(feats: dict[str, np.ndarray], names: list[str], n: int) -> np.ndarray:
    cols = []
    for k in names:
        v = np.asarray(feats.get(k, np.full(n, np.nan)), dtype=np.float64)
        cols.append(v if len(v) == n else np.full(n, np.nan))
    return np.stack(cols, axis=1) if cols else np.zeros((n, 0))


def _prep(x: np.ndarray, m: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(m["mean"], dtype=np.float64)
    std = np.asarray(m["std"], dtype=np.float64)
    x = np.where(np.isfinite(x), x, mean)          # пропуски → среднее (нейтрально)
    return (x - mean) / np.where(std > 1e-9, std, 1.0)


def predict(m: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """P(state=1) по модели сигнала."""
    z = _prep(x, m)
    if m.get("type", "lr") == "mlp":
        w1, b1 = np.asarray(m["w1"]), np.asarray(m["b1"])
        w2, b2 = np.asarray(m["w2"]), float(m["b2"])
        h = np.tanh(z @ w1 + b1)
        logits = h @ w2 + b2
    else:
        logits = z @ np.asarray(m["coef"], dtype=np.float64) + float(m["intercept"])
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))


def fit_lr(x: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> dict[str, Any]:
    """L2-логрегрессия (IRLS/Ньютон) на стандартизованных признаках."""
    mean = np.nanmean(x, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.nanstd(x, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-9), std, 1.0)
    m = {"type": "lr", "mean": mean.tolist(), "std": std.tolist()}
    z = _prep(x, m)
    a = np.hstack([z, np.ones((len(z), 1))])
    w = np.zeros(a.shape[1])
    reg = np.eye(a.shape[1]) * l2
    reg[-1, -1] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(a @ w, -30, 30)))
        g = a.T @ (p - y) + reg @ w
        h = (a * (p * (1 - p))[:, None]).T @ a + reg + np.eye(len(w)) * 1e-6
        step = np.linalg.solve(h, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    m.update({"coef": w[:-1].tolist(), "intercept": float(w[-1])})
    return m


def fit_mlp(x: np.ndarray, y: np.ndarray, hidden: int = 8, epochs: int = 400, lr: float = 0.03, l2: float = 1e-3, seed: int = 0) -> dict[str, Any]:
    """Маленький MLP (tanh, 1 скрытый слой), full-batch Adam — numpy."""
    base = fit_lr(x, y)
    z = _prep(x, base)
    rng = np.random.default_rng(seed)
    d = z.shape[1]
    params = {"w1": rng.normal(0, 1 / np.sqrt(max(d, 1)), (d, hidden)), "b1": np.zeros(hidden), "w2": rng.normal(0, 0.3, hidden), "b2": np.zeros(1)}
    mom = {k: np.zeros_like(v) for k, v in params.items()}
    vel = {k: np.zeros_like(v) for k, v in params.items()}
    for ep in range(1, epochs + 1):
        h = np.tanh(z @ params["w1"] + params["b1"])
        p = 1.0 / (1.0 + np.exp(-np.clip(h @ params["w2"] + params["b2"][0], -30, 30)))
        e = (p - y) / len(y)
        grads = {
            "w2": h.T @ e + l2 * params["w2"],
            "b2": np.array([e.sum()]),
        }
        dh = np.outer(e, params["w2"]) * (1 - h ** 2)
        grads["w1"] = z.T @ dh + l2 * params["w1"]
        grads["b1"] = dh.sum(axis=0)
        for k in params:
            mom[k] = 0.9 * mom[k] + 0.1 * grads[k]
            vel[k] = 0.999 * vel[k] + 0.001 * grads[k] ** 2
            mh = mom[k] / (1 - 0.9 ** ep)
            vh = vel[k] / (1 - 0.999 ** ep)
            params[k] = params[k] - lr * mh / (np.sqrt(vh) + 1e-8)
    return {"type": "mlp", "mean": base["mean"], "std": base["std"], "w1": params["w1"].tolist(), "b1": params["b1"].tolist(), "w2": params["w2"].tolist(), "b2": float(params["b2"][0])}


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC AUC через ранги (ties — средний ранг)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))
