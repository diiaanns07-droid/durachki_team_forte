"""Подготовка обучающих данных: входной вектор (sims/emb/dyn), нормализация+PCA (fit только на train),
слабые цели (гауссиана вокруг размеченной секунды), окна, аугментации в пространстве признаков."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .config import TYPE_IDS
from .features import FeatureSet, TextBank, type_scores


@dataclass
class Sample:
    name: str
    group: str
    label: int
    times: list[float]                      # размеченные секунды (может быть несколько у дубликатов)
    type_id: str = ""                        # только проверенная локальная разметка типа, иначе ""
    pseudo_type_id: str = ""                 # предварительный тип (VLM) — только по флагу train.use_pseudo_types
    md5: str = ""
    sources: list[dict] = field(default_factory=list)          # исходные метки файлов группы дубликатов
    conflict: bool = False
    time_conflict: bool = False              # в группе дубликатов разные таймкоды (решает человек)
    versions: list[FeatureSet] = field(default_factory=list)   # [оригинал, аугментации...]
    X: list[np.ndarray] = field(default_factory=list)           # входы на сетке модели для каждой версии
    grid: list[np.ndarray] = field(default_factory=list)        # таймстемпы сетки для каждой версии


# ----------------------------------------------------------------------------- input spec
@dataclass
class InputSpec:
    views: list[str]
    groups: list[str]
    fps: float
    n_prompts: int
    dim: int
    pca_dim: int
    pca_mean: Optional[np.ndarray] = None    # (V, D)
    pca_comp: Optional[np.ndarray] = None    # (V, D, k)
    mean: Optional[np.ndarray] = None        # (F,)
    std: Optional[np.ndarray] = None         # (F,)
    backend: str = ""
    n_type_scores: int = 0
    change_lags_s: tuple = (2.0, 8.0)
    change_deltas_s: tuple = (1.0, 3.0)

    @property
    def n_change(self) -> int:
        return len(self.change_lags_s) + len(self.change_deltas_s)

    def view_columns(self) -> list[np.ndarray]:
        """Индексы столбцов входа, относящихся к каждому виду (для view-dropout на обучении)."""
        V, P = len(self.views), self.n_prompts
        cols: list[list[int]] = [[] for _ in range(V)]
        off = 0
        for g in self.groups:
            if g == "sims":
                for v in range(V):
                    cols[v] += list(range(off + v * P, off + (v + 1) * P))
                off += V * P + self.n_type_scores
            elif g == "emb":
                k = self.pca_dim if self.pca_comp is not None else self.dim
                for v in range(V):
                    cols[v] += list(range(off + v * k, off + (v + 1) * k))
                off += V * k
            elif g == "dyn":
                for v in range(V):
                    cols[v].append(off + 2 + v)
                off += 2 + V
            elif g == "change":
                nc = self.n_change
                for v in range(V):
                    cols[v] += list(range(off + v * nc, off + (v + 1) * nc))
                off += V * nc
            elif g == "avail":
                for v in range(V):
                    cols[v].append(off + v)
                off += V
        return [np.asarray(c, dtype=np.int64) for c in cols]

    @property
    def in_dim(self) -> int:
        return 0 if self.mean is None else int(self.mean.shape[0])

    def to_npz(self, path: str | Path) -> None:
        arrs = {k: getattr(self, k) for k in ("pca_mean", "pca_comp", "mean", "std") if getattr(self, k) is not None}
        meta = {"views": self.views, "groups": self.groups, "fps": self.fps, "n_prompts": self.n_prompts,
                "dim": self.dim, "pca_dim": self.pca_dim, "backend": self.backend,
                "n_type_scores": self.n_type_scores, "change_lags_s": list(self.change_lags_s),
                "change_deltas_s": list(self.change_deltas_s)}
        np.savez(path, meta=np.asarray(json.dumps(meta)), **arrs)

    @classmethod
    def from_npz(cls, path: str | Path) -> "InputSpec":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            arrs = {k: z[k] for k in ("pca_mean", "pca_comp", "mean", "std") if k in z}
        for k in ("change_lags_s", "change_deltas_s"):
            if k in meta:
                meta[k] = tuple(meta[k])
        return cls(**meta, **arrs)


def _view_emb(fs: FeatureSet, views: Sequence[str]) -> np.ndarray:
    """(T, V_spec, D); недостающий вид = NaN (после нормализации станет 0 — как view-dropout на обучении)."""
    T, _, D = fs.emb.shape
    out = np.full((T, len(views), D), np.nan, np.float32)
    for i, v in enumerate(views):
        if v in fs.views:
            out[:, i] = fs.emb[:, fs.views.index(v)]
    return out


def raw_blocks(fs: FeatureSet, bank: TextBank, spec: InputSpec) -> dict[str, np.ndarray]:
    """Сырые группы признаков до нормализации (на сетке клипа)."""
    emb = _view_emb(fs, spec.views)
    blocks: dict[str, np.ndarray] = {}
    if "sims" in spec.groups:
        T, V, D = emb.shape
        s = (np.nan_to_num(emb, nan=0.0).reshape(T * V, D) @ bank.emb.T).reshape(T, V, -1)
        s[np.isnan(emb[:, :, 0])] = np.nan
        ts, _ = type_scores(fs, bank)
        blocks["sims"] = np.concatenate([s.reshape(len(s), -1), ts], axis=1)
    if "emb" in spec.groups:
        blocks["emb"] = emb
    if "change" in spec.groups:  # изменения во времени: лаг-разность и контраст «до/после» по каждому виду
        # не зависит от подгонки (PCA/скейлер) -> считаем один раз на объект признаков, а не в каждом фолде
        key = ("change", tuple(spec.views), tuple(spec.change_lags_s), tuple(spec.change_deltas_s))
        cache = fs.__dict__.setdefault("_cache", {})
        if key not in cache:
            cache[key] = change_block(fs.times, emb, spec)
        blocks["change"] = cache[key]
    if "avail" in spec.groups:   # вид известен (1) / неизвестен (0): unknown ≠ «среднее значение»
        blocks["avail"] = np.isfinite(emb[:, :, 0]).astype(np.float32)
    if "dyn" in spec.groups:
        V = len(spec.views)
        d = np.full((len(fs), 2 + V), np.nan, np.float32)
        d[:, :2] = fs.dyn[:, :2]
        for i, v in enumerate(spec.views):
            if v in fs.views:
                d[:, 2 + i] = fs.dyn[:, 2 + fs.views.index(v)]
        blocks["dyn"] = d
    return blocks


def change_block(times: np.ndarray, emb: np.ndarray, spec: InputSpec) -> np.ndarray:
    """(T, V·n_change): для каждого вида 1−cos(e_t, e_{t−Δ}) по change_lags_s и 1−cos(среднее до, среднее после)
    по change_deltas_s. Неизвестный вид / начало ролика без истории — NaN (после нормализации 0)."""
    from .candidates import _cos_dist, _window_means

    T, V, _ = emb.shape
    dt = float(np.median(np.diff(times))) if T > 1 else 0.25
    out = np.full((T, V, spec.n_change), np.nan, np.float32)
    for v in range(V):
        E = emb[:, v].astype(np.float64)
        if np.isnan(E).any():
            continue
        j = 0
        for lag in spec.change_lags_s:
            d = max(1, int(round(float(lag) / max(dt, 1e-6))))
            if d < T:
                out[d:, v, j] = 1.0 - np.sum(E[d:] * E[:-d], axis=1)
            j += 1
        for delta in spec.change_deltas_s:
            d = max(1, int(round(float(delta) / max(dt, 1e-6))))
            b, a = _window_means(E, d)
            out[:, v, j] = _cos_dist(b, a)
            j += 1
    return out.reshape(T, -1)


def _project(blocks: dict[str, np.ndarray], spec: InputSpec) -> np.ndarray:
    parts = []
    for g in spec.groups:
        b = blocks[g]
        if g == "emb":
            if spec.pca_comp is not None:
                b = np.stack([(b[:, v] - spec.pca_mean[v]) @ spec.pca_comp[v] for v in range(b.shape[1])], axis=1)
            b = b.reshape(len(b), -1)
        parts.append(b.astype(np.float32))
    return np.concatenate(parts, axis=1) if parts else np.zeros((0, 0), np.float32)


def fit_spec(sets: Sequence[FeatureSet], bank: TextBank, cfg: dict) -> InputSpec:
    m = cfg["model"]
    fs0 = sets[0]
    views = list(cfg["features"]["views"]) + [f"zone:{z}" for z in (cfg["features"].get("zone_views") or [])]

    def has_data(v: str) -> bool:
        return any(v in fs.views and np.isfinite(fs.emb[:, fs.views.index(v), 0]).any() for fs in sets)

    views = [v for v in views if has_data(v)] or list(fs0.views)
    spec = InputSpec(views=views, groups=list(m.get("input_groups", ["sims", "emb", "dyn"])),
                     fps=float(fs0.meta.get("fps", cfg["features"]["fps"])), n_prompts=len(bank.texts_en),
                     dim=int(fs0.emb.shape[-1]), pca_dim=int(m.get("emb_pca", 32)),
                     backend=str(fs0.meta.get("backend", "")),
                     n_type_scores=len([k for k in TYPE_IDS if k in bank.type_index]),
                     change_lags_s=tuple(m.get("change_lags_s", [2.0, 8.0])),
                     change_deltas_s=tuple(m.get("change_deltas_s", [1.0, 3.0])))
    if any(v.startswith("zone:") for v in views) and "avail" not in spec.groups:
        spec.groups.append("avail")  # с зонными видами всегда отдаём флаг «вид известен»
    if "emb" in spec.groups and spec.pca_dim > 0:
        k = min(spec.pca_dim, spec.dim)
        means, comps = [], []
        for v in views:
            parts = [fs.emb[:, fs.views.index(v)] for fs in sets if v in fs.views]
            E = np.concatenate(parts, axis=0).astype(np.float32) if parts else np.zeros((0, spec.dim), np.float32)
            E = E[np.isfinite(E).all(axis=1)]  # неизвестные кадры/виды не участвуют в PCA
            mu = E.mean(axis=0) if len(E) else np.zeros(spec.dim, np.float32)
            if len(E) > k:  # PCA через ковариацию (D×D) — быстрее SVD по всем кадрам
                Ec = E - mu
                evals, evecs = np.linalg.eigh((Ec.T @ Ec).astype(np.float64))
                comp = evecs[:, ::-1][:, :k].astype(np.float32)
            else:
                comp = np.eye(spec.dim, k, dtype=np.float32)
            means.append(mu); comps.append(comp)
        spec.pca_mean = np.stack(means).astype(np.float32)
        spec.pca_comp = np.stack(comps).astype(np.float32)
        spec.pca_dim = k
    X = np.concatenate([_project(raw_blocks(fs, bank, spec), spec) for fs in sets], axis=0)
    import warnings

    with warnings.catch_warnings():  # полностью неизвестный столбец (напр. лаг длиннее роликов) -> среднее 0, std 1
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
    spec.mean = np.nan_to_num(mean).astype(np.float32)
    std = np.nan_to_num(std, nan=1.0)
    spec.std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return spec


def model_grid(duration: float, fps: float) -> np.ndarray:
    n = max(1, int(np.floor(duration * fps + 1e-6)) + 1)
    return (np.arange(n) / fps).astype(np.float32)


def build_inputs(fs: FeatureSet, bank: TextBank, spec: InputSpec) -> tuple[np.ndarray, np.ndarray]:
    """Нормализованные входы на равномерной сетке spec.fps (линейная интерполяция, если fps клипа другой)."""
    X = _project(raw_blocks(fs, bank, spec), spec)
    X = (X - spec.mean) / spec.std
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    t = fs.times.astype(np.float64)
    grid = model_grid(float(t[-1]) if len(t) else 0.0, spec.fps)
    if len(t) >= 2 and (len(t) != len(grid) or np.max(np.abs(t - grid[: len(t)])) > 0.5 / spec.fps):
        X = np.stack([np.interp(grid, t, X[:, j]) for j in range(X.shape[1])], axis=1)
    elif len(t) == 1:
        X = np.repeat(X, len(grid), axis=0)
    return X.astype(np.float32), grid


# ----------------------------------------------------------------------------- targets
def frame_targets(grid: np.ndarray, label: int, times: Sequence[float], sigma: float, far_radius: float,
                  far_weight: float) -> tuple[np.ndarray, np.ndarray]:
    """y (T,) — гауссиана(ы) вокруг размеченных секунд; w (T,) — веса (слабые вдали от метки в позитиве)."""
    T = len(grid)
    if not label or not times:
        return np.zeros(T, np.float32), np.ones(T, np.float32)
    d = np.min(np.abs(grid[:, None] - np.asarray(times, np.float32)[None, :]), axis=1)
    y = np.exp(-0.5 * (d / max(sigma, 1e-3)) ** 2).astype(np.float32)
    w = np.where(d <= far_radius, 1.0, far_weight).astype(np.float32)
    return y, w


def type_index(type_id: str) -> int:
    return TYPE_IDS.index(type_id) if type_id in TYPE_IDS else -1


def train_type_id(s: "Sample", cfg: dict) -> str:
    """Тип для обучения головы: подтверждённый; предварительный (VLM) — только если use_pseudo_types."""
    if s.type_id:
        return s.type_id
    return s.pseudo_type_id if cfg["train"].get("use_pseudo_types", False) else ""


# ----------------------------------------------------------------------------- windows / augs
def time_warp(X: np.ndarray, y: np.ndarray, w: np.ndarray, factor: float) -> tuple[np.ndarray, ...]:
    T = len(X)
    n = max(2, int(round(T * factor)))
    src = np.linspace(0, T - 1, n)
    Xw = np.stack([np.interp(src, np.arange(T), X[:, j]) for j in range(X.shape[1])], axis=1)
    return Xw.astype(np.float32), np.interp(src, np.arange(T), y).astype(np.float32), \
        np.interp(src, np.arange(T), w).astype(np.float32)


def sample_window(s: Sample, cfg: dict, rng: np.random.Generator, use_aug: bool,
                  view_cols: Optional[list[np.ndarray]] = None) -> dict:
    tr = cfg["train"]
    vi = 0
    if use_aug and len(s.X) > 1:
        vi = int(rng.integers(1, len(s.X)))
    X, grid = s.X[vi], s.grid[vi]
    y, w = frame_targets(grid, s.label, s.times, tr["sigma_s"], tr["far_radius_s"], tr["far_weight"])
    fps = 1.0 / max(float(grid[1] - grid[0]), 1e-6) if len(grid) > 1 else 4.0
    L = max(8, int(round(tr["window_s"] * fps)))
    T = len(X)
    contains = True
    if T > L:
        if s.label and s.times and rng.random() < tr["pos_window_prob"]:
            tc = float(rng.choice(s.times))
            c = int(np.clip(round(tc * fps), 0, T - 1))
            start = int(np.clip(c - int(rng.uniform(0.15, 0.85) * L), 0, T - L))
        else:
            start = int(rng.integers(0, T - L + 1))
        X, y, w = X[start:start + L], y[start:start + L], w[start:start + L]
        if s.label:
            contains = bool(y.max() > 0.5)
    if rng.random() < tr.get("time_warp_prob", 0.0):
        lo, hi = tr.get("time_warp", [0.8, 1.25])
        X, y, w = time_warp(X, y, w, float(rng.uniform(lo, hi)))
    if tr.get("noise_std", 0) > 0:
        X = X + rng.normal(0, tr["noise_std"], size=X.shape).astype(np.float32)
    vd = float(cfg["model"].get("view_dropout", 0.0))
    if view_cols and len(view_cols) > 1 and rng.random() < vd:
        X = X.copy()
        X[:, view_cols[int(rng.integers(0, len(view_cols)))]] = 0.0
    return {"X": X, "y": y, "w": w, "label": float(s.label if contains else 0),
            "mil_mask": float(1.0 if (not s.label or contains) else 0.0), "type": type_index(train_type_id(s, cfg))}


def collate(items: list[dict]) -> dict:
    L = max(len(it["X"]) for it in items)
    F = items[0]["X"].shape[1]
    B = len(items)
    X = np.zeros((B, L, F), np.float32)
    y = np.zeros((B, L), np.float32)
    w = np.zeros((B, L), np.float32)
    m = np.zeros((B, L), np.float32)
    for i, it in enumerate(items):
        n = len(it["X"])
        X[i, :n], y[i, :n], w[i, :n], m[i, :n] = it["X"], it["y"], it["w"], 1.0
    return {"X": X, "y": y, "w": w, "mask": m,
            "label": np.asarray([it["label"] for it in items], np.float32),
            "mil_mask": np.asarray([it["mil_mask"] for it in items], np.float32),
            "type": np.asarray([it["type"] for it in items], np.int64)}


def mixup(batch: dict, rng: np.random.Generator, alpha: float) -> dict:
    lam = float(rng.beta(alpha, alpha))
    perm = rng.permutation(len(batch["X"]))
    out = dict(batch)
    for k in ("X", "y"):
        out[k] = lam * batch[k] + (1 - lam) * batch[k][perm]
    out["w"] = np.minimum(batch["w"], batch["w"][perm])
    out["mask"] = batch["mask"] * batch["mask"][perm]
    out["label"] = lam * batch["label"] + (1 - lam) * batch["label"][perm]
    out["mil_mask"] = batch["mil_mask"] * batch["mil_mask"][perm]
    out["type"] = np.where(lam >= 0.5, batch["type"], batch["type"][perm])
    return out


def prepare_samples(samples: Sequence[Sample], bank: TextBank, spec: InputSpec) -> None:
    for s in samples:
        s.X, s.grid = [], []
        for fs in s.versions:
            X, g = build_inputs(fs, bank, spec)
            s.X.append(X)
            s.grid.append(g)
