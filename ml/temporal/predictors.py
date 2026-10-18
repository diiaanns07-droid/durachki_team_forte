"""Обучаемые предсказатели на замороженных признаках и их чекпоинты.

* ``baseline`` — логистическая регрессия по кадрам (+ скользящее среднее как временной контекст), чистый numpy.
  Это первый обучаемый результат и опорная точка для сравнения.
* ``temporal`` — TCN(+BiGRU) голова (model.py, torch) с MIL-потерей уровня видео — сравнивается с baseline.

Обе модели учатся одинаково слабо: гауссиана вокруг размеченной секунды (локально), остальная часть
позитивного ролика — маска (неизвестно), чистые ролики — негатив. Голова типа — только при наличии
проверенных type_id в разметке.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .dataset import InputSpec, Sample, collate, frame_targets, mixup, sample_window

log = logging.getLogger("sentinel.temporal")
KINDS = ("baseline", "temporal")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def moving_average(X: np.ndarray, win: int) -> np.ndarray:
    """Скользящее среднее по времени (окно win шагов, с краевым дополнением)."""
    if win <= 1 or len(X) == 0:
        return X
    pad = win // 2
    Xp = np.pad(X, ((pad, win - 1 - pad), (0, 0)) if X.ndim == 2 else (pad, win - 1 - pad), mode="edge")
    c = np.cumsum(np.concatenate([np.zeros((1,) + Xp.shape[1:], Xp.dtype), Xp], axis=0), axis=0)
    return ((c[win:] - c[:-win]) / win).astype(np.float32)


def has_type_labels(samples: Sequence[Sample], cfg: Optional[dict] = None) -> bool:
    if cfg is None:
        return any(s.label and s.type_id for s in samples)
    from .dataset import train_type_id

    return any(s.label and train_type_id(s, cfg) for s in samples)


def type_head_source(samples: Sequence[Sample], cfg: dict) -> Optional[str]:
    """confirmed — только подтверждённые метки; pseudo — есть предварительные (VLM); None — головы нет."""
    if not has_type_labels(samples, cfg):
        return None
    from .dataset import train_type_id

    return "pseudo" if any(s.label and not s.type_id and train_type_id(s, cfg) for s in samples) else "confirmed"


# ----------------------------------------------------------------------------- baseline (numpy)
class BaselineLR:
    kind = "baseline"

    def __init__(self, cfg: dict) -> None:
        b = cfg.get("baseline", {})
        self.context_s = float(b.get("context_s", 2.0))
        self.l2 = float(b.get("l2", 1e-3))
        self.iters = int(b.get("iters", 400))
        self.lr = float(b.get("lr", 0.05))
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.type_W: Optional[np.ndarray] = None   # (2F, K) — только при проверенных type_id

    def _ctx(self, X: np.ndarray, fps: float) -> np.ndarray:
        win = max(1, int(round(self.context_s * fps)))
        return np.concatenate([X, moving_average(X, win)], axis=1)

    def fit(self, samples: Sequence[Sample], cfg: dict, seed: int = 0) -> "BaselineLR":
        X, y, wt, tt, _ = self.gather(samples, cfg)
        return self._fit_arrays(X, y, wt, tt, samples, cfg, seed)

    def gather(self, samples: Sequence[Sample], cfg: dict) -> tuple:
        """Кадры обучения: X (+контекст), мягкие цели y, веса wt (сумма 1), type idx, индекс группы."""
        from .dataset import train_type_id

        tr = cfg["train"]
        group_mode = str(tr.get("weighting", "group")) == "group"
        n_pos = max(1, sum(1 for s in samples if s.label))
        n_neg = max(1, sum(1 for s in samples if not s.label))
        Xs, ys, ws, ts, gs = [], [], [], [], []
        for gi, s in enumerate(samples):
            nv = max(1, len(s.X))
            for X, g in zip(s.X, s.grid):
                fps = 1.0 / max(float(g[1] - g[0]), 1e-6) if len(g) > 1 else 4.0
                y, w = frame_targets(g, s.label, s.times, tr["sigma_s"], tr["far_radius_s"], tr["far_weight"])
                if group_mode:  # группа весит 0.5/n_класса независимо от длины ролика и числа версий
                    tot = float(w.sum())
                    if tot <= 0:
                        continue
                    w = w / tot / nv * (0.5 / (n_pos if s.label else n_neg))
                Xs.append(self._ctx(X, fps)); ys.append(y); ws.append(w)
                ts.append(np.full(len(y), _type_idx(train_type_id(s, cfg)), np.int64))
                gs.append(np.full(len(y), gi, np.int64))
        X = np.concatenate(Xs).astype(np.float32)
        y = np.concatenate(ys).astype(np.float32)
        w = np.concatenate(ws).astype(np.float32)
        keep = w > 0
        X, y, w, tt, gg = X[keep], y[keep], w[keep], np.concatenate(ts)[keep], np.concatenate(gs)[keep]
        if group_mode:
            wt = w
        else:  # старый режим: каждый кадр = 1, баланс классов по массе кадров
            pos = float((w * y).sum())
            neg = float((w * (1 - y)).sum())
            pw = float(np.clip(neg / max(pos, 1e-6), 1.0, tr.get("pos_weight_max", 20.0))) if pos > 0 else 1.0
            wt = w * (1 + (pw - 1) * y)
        wt = wt / max(wt.sum(), 1e-6)
        return X, y, wt, tt, gg

    def _fit_arrays(self, X: np.ndarray, y: np.ndarray, wt: np.ndarray, tt: np.ndarray, samples: Sequence[Sample],
                    cfg: dict, seed: int) -> "BaselineLR":
        F = X.shape[1]
        rng = np.random.default_rng(seed)
        theta = np.concatenate([rng.normal(0, 0.01, F), [0.0]]).astype(np.float64)
        m = np.zeros_like(theta)
        v = np.zeros_like(theta)
        Xa = np.concatenate([X, np.ones((len(X), 1), np.float32)], axis=1).astype(np.float64)
        for it in range(1, self.iters + 1):  # Adam, полный батч, взвешенная логистическая потеря + L2
            p = _sigmoid(Xa @ theta)
            grad = Xa.T @ (wt * (p - y))
            grad[:-1] += self.l2 * theta[:-1]
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad ** 2
            theta -= self.lr * (m / (1 - 0.9 ** it)) / (np.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
        self.w, self.b = theta[:-1].astype(np.float32), float(theta[-1])
        sel = (tt >= 0) & (y > 0.5)
        if has_type_labels(samples, cfg) and sel.sum() >= 2:
            self.type_W = _fit_softmax(X[sel], tt[sel], n=int(cfg["model"].get("n_types", 9)), l2=self.l2)
        return self

    def predict(self, X: np.ndarray, fps: float) -> tuple[np.ndarray, Optional[np.ndarray]]:
        Z = self._ctx(X, fps)
        p = _sigmoid(Z @ self.w + self.b).astype(np.float32)
        tp = None
        if self.type_W is not None:
            logits = np.concatenate([Z, np.ones((len(Z), 1), np.float32)], axis=1) @ self.type_W
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            tp = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
        return p, tp

    @property
    def type_trained(self) -> bool:
        return self.type_W is not None

    def save(self, d: Path) -> None:
        arrs = {"w": self.w, "b": np.asarray([self.b], np.float32),
                "cfg": np.asarray(json.dumps({"context_s": self.context_s, "l2": self.l2}))}
        if self.type_W is not None:
            arrs["type_W"] = self.type_W
        np.savez(d / "baseline.npz", **arrs)

    @classmethod
    def load(cls, d: Path, cfg: dict) -> "BaselineLR":
        with np.load(d / "baseline.npz", allow_pickle=False) as z:
            c = json.loads(str(z["cfg"]))
            m = cls({"baseline": c, "train": cfg.get("train", {})})
            m.w, m.b = z["w"].astype(np.float32), float(z["b"][0])
            m.type_W = z["type_W"].astype(np.float32) if "type_W" in z else None
        return m


def _type_idx(type_id: str) -> int:
    from .config import TYPE_IDS
    return TYPE_IDS.index(type_id) if type_id in TYPE_IDS else -1


def _fit_softmax(X: np.ndarray, t: np.ndarray, n: int, l2: float, iters: int = 300, lr: float = 0.05) -> np.ndarray:
    Xa = np.concatenate([X, np.ones((len(X), 1), np.float32)], axis=1).astype(np.float64)
    W = np.zeros((Xa.shape[1], n))
    Y = np.eye(n)[t]
    m = np.zeros_like(W); v = np.zeros_like(W)
    for it in range(1, iters + 1):
        L = Xa @ W
        P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        g = Xa.T @ (P - Y) / len(Xa) + l2 * W
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g ** 2
        W -= lr * (m / (1 - 0.9 ** it)) / (np.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
    return W.astype(np.float32)


# ----------------------------------------------------------------------------- temporal head (torch)
class TemporalHead:
    kind = "temporal"

    def __init__(self, in_dim: int, cfg: dict) -> None:
        from .model import build_model

        self.cfg = cfg
        self.in_dim = in_dim
        self.net = build_model(in_dim, cfg)
        self.type_trained = False
        self.device = "cpu"

    def fit(self, samples: Sequence[Sample], cfg: dict, spec: InputSpec, epochs: Optional[int] = None,
            seed: int = 0, deadline: Optional[float] = None) -> "TemporalHead":
        import torch

        from .model import loss_fn

        tr = cfg["train"]
        dev = tr.get("device", "auto")
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if dev == "auto" else dev
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        self.type_trained = has_type_labels(samples, cfg) and float(tr.get("type_weight", 0)) > 0
        cfg_loss = cfg if self.type_trained else {**cfg, "train": {**tr, "type_weight": 0.0}}
        net = self.net.to(self.device).train()
        opt = torch.optim.AdamW(net.parameters(), lr=float(tr["lr"]), weight_decay=float(tr["weight_decay"]))
        n_ep = int(epochs or tr["epochs"])
        steps = n_ep * int(tr["steps_per_epoch"])
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(tr["lr"]), total_steps=max(steps, 1))
        pos = [s for s in samples if s.label]
        neg = [s for s in samples if not s.label]
        view_cols = spec.view_columns()
        B = int(tr["batch_size"])
        t0 = time.monotonic()
        for step in range(steps):
            items = []
            for i in range(B):  # балансировка: половина позитивов, половина негативов
                pool = pos if (i % 2 == 0 and pos) or not neg else neg
                s = pool[int(rng.integers(0, len(pool)))]
                items.append(sample_window(s, cfg, rng, use_aug=rng.random() < tr.get("aug_prob", 0.7),
                                           view_cols=view_cols))
            batch = collate(items)
            if rng.random() < tr.get("mixup_prob", 0.0):
                batch = mixup(batch, rng, float(tr.get("mixup_alpha", 0.4)))
            tb = {k: torch.from_numpy(v).to(self.device) for k, v in batch.items()}
            logits, type_logits = net(tb["X"], tb["mask"])
            loss, parts = loss_fn(logits, type_logits, tb, cfg_loss)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            if deadline is not None and time.monotonic() > deadline:
                log.warning("temporal: остановка по бюджету времени на шаге %d/%d", step + 1, steps)
                break
        self.train_s = round(time.monotonic() - t0, 2)
        net.eval()
        return self

    def predict(self, X: np.ndarray, fps: float) -> tuple[np.ndarray, Optional[np.ndarray]]:
        import torch

        with torch.no_grad():
            x = torch.from_numpy(X[None].astype(np.float32)).to(self.device)
            logits, tl = self.net.eval()(x)
            p = torch.sigmoid(logits)[0].float().cpu().numpy()
            tp = torch.softmax(tl, dim=-1)[0].float().cpu().numpy() if self.type_trained else None
        return p.astype(np.float32), tp

    def save(self, d: Path) -> None:
        import torch

        torch.save({k: v.detach().cpu() for k, v in self.net.state_dict().items()}, d / "temporal.pt")
        (d / "temporal.json").write_text(json.dumps({"in_dim": self.in_dim, "type_trained": self.type_trained,
                                                      "model": self.cfg["model"]}), encoding="utf-8")

    @classmethod
    def load(cls, d: Path, cfg: dict) -> "TemporalHead":
        import torch

        info = json.loads((d / "temporal.json").read_text(encoding="utf-8"))
        m = cls(int(info["in_dim"]), {**cfg, "model": {**cfg["model"], **info["model"]}})
        m.net.load_state_dict(torch.load(d / "temporal.pt", map_location="cpu", weights_only=True))
        m.device = "cuda" if torch.cuda.is_available() else "cpu"
        m.net.to(m.device).eval()
        m.type_trained = bool(info.get("type_trained"))
        return m


# ----------------------------------------------------------------------------- fit / checkpoint
def fit_predictor(kind: str, samples: Sequence[Sample], spec: InputSpec, cfg: dict, seed: int = 0,
                  epochs: Optional[int] = None, deadline: Optional[float] = None) -> Any:
    if kind == "baseline":
        return BaselineLR(cfg).fit(samples, cfg, seed=seed)
    if kind == "temporal":
        return TemporalHead(spec.in_dim, cfg).fit(samples, cfg, spec, epochs=epochs, seed=seed, deadline=deadline)
    raise ValueError(f"неизвестная модель: {kind}")


def save_checkpoint(out_dir: str | Path, model: Any, spec: InputSpec, meta: dict) -> Path:
    d = Path(out_dir) / model.kind
    d.mkdir(parents=True, exist_ok=True)
    spec.to_npz(d / "spec.npz")
    model.save(d)
    meta = {**meta, "kind": model.kind, "type_head_trained": bool(model.type_trained),
            "backend": spec.backend, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "format": 1}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return d


def load_checkpoint(root: str | Path, cfg: dict, kind: Optional[str] = None) -> Optional[tuple[Any, InputSpec, dict]]:
    """root/<kind>/{spec.npz, meta.json, baseline.npz|temporal.pt}; kind=None -> root/selected.json или baseline."""
    root = Path(root)
    if kind is None:
        sel = root / "selected.json"
        kind = json.loads(sel.read_text(encoding="utf-8")).get("kind") if sel.exists() else None
        candidates = ([kind] if kind else []) + [k for k in KINDS if k != kind]
    else:
        candidates = [kind]
    for k in candidates:
        d = root / k
        if not (d / "meta.json").exists():
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if kind is not None and k != kind:
            log.warning("temporal: выбранный чекпоинт '%s' не найден, использую '%s'", kind, k)
        spec = InputSpec.from_npz(d / "spec.npz")
        model = BaselineLR.load(d, cfg) if k == "baseline" else TemporalHead.load(d, cfg)
        return model, spec, meta
    return None
