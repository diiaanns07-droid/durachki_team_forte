"""Признаки для временной модели: эмбеддинги SigLIP2 (весь кадр + адаптивный кроп), zero-shot сходства,
динамика (движение, смена кадра). Кэш: cache/features/<sha>.npz. Бэкенды: siglip2 | mock (SENTINEL_MOCK=1)."""
from __future__ import annotations

import functools
import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .config import TYPE_IDS, cache_dir, feature_signature, is_mock, load_prompts, stable_hash

log = logging.getLogger("sentinel.temporal")
IMG = 224
FEATURE_VERSION = 3  # поднимать при любом изменении извлечения: старый кэш перестанет совпадать по ключу


# ----------------------------------------------------------------------------- backends
class MockBackend:
    """Детерминированные «эмбеддинги» из уменьшенного кадра (без весов, для тестов и SENTINEL_MOCK=1)."""

    name = "mock"

    def __init__(self, dim: int = 64) -> None:
        self.dim = int(dim)
        rng = np.random.default_rng(1234)
        self._proj = rng.standard_normal((8 * 8 * 3, self.dim)).astype(np.float32)
        self.logit_scale, self.logit_bias = 10.0, -5.0
        self.device = "cpu"

    def encode_images(self, rgb: np.ndarray) -> np.ndarray:
        import cv2

        small = np.stack([cv2.resize(im, (8, 8), interpolation=cv2.INTER_AREA) for im in rgb]).astype(np.float32)
        x = small.reshape(len(rgb), -1) / 255.0
        x = x - x.mean(axis=1, keepdims=True)
        return _l2(x @ self._proj)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        out = []
        for t in texts:
            seed = int(stable_hash(t, 8), 16)
            out.append(np.random.default_rng(seed).standard_normal(self.dim))
        return _l2(np.asarray(out, dtype=np.float32))


class SiglipBackend:
    """google/siglip2-base-patch16-224 (Apache-2.0) через transformers; fp16 на CUDA, иначе CPU fp32."""

    name = "siglip2"

    def __init__(self, model_id: str, device: str = "auto", fp16: bool = True, cache_dir: Optional[str] = None) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        kw = {"cache_dir": cache_dir} if cache_dir else {}

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = torch.float16 if (fp16 and device.startswith("cuda")) else torch.float32
        self.model_id = model_id
        try:
            self.processor = AutoProcessor.from_pretrained(model_id, **kw)
            ip = getattr(self.processor, "image_processor", None)
        except Exception:  # нет токенизатора рядом с весами — картинки всё равно считаем
            self.processor = None
            try:
                from transformers import AutoImageProcessor

                ip = AutoImageProcessor.from_pretrained(model_id, **kw)
            except Exception:
                ip = None
        try:  # transformers 5.x: dtype=; 4.x: torch_dtype=
            model = AutoModel.from_pretrained(model_id, dtype=self.dtype, **kw)
        except TypeError:
            model = AutoModel.from_pretrained(model_id, torch_dtype=self.dtype, **kw)
        self.model = model.to(device).eval()
        self._kw = kw
        self.mean = np.asarray(getattr(ip, "image_mean", None) or [0.5, 0.5, 0.5], dtype=np.float32)
        self.std = np.asarray(getattr(ip, "image_std", None) or [0.5, 0.5, 0.5], dtype=np.float32)
        self.logit_scale = float(self.model.logit_scale.exp().item()) if hasattr(self.model, "logit_scale") else 1.0
        self.logit_bias = float(self.model.logit_bias.item()) if hasattr(self.model, "logit_bias") else 0.0
        with torch.no_grad():
            self.dim = int(self._img(torch.zeros(1, 3, IMG, IMG, device=device, dtype=self.dtype)).shape[-1])
        self._torch = torch

    @staticmethod
    def _pool(out: Any) -> Any:
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "image_embeds"):
            return out.image_embeds
        if hasattr(out, "text_embeds"):
            return out.text_embeds
        return out

    def _img(self, pix: Any) -> Any:
        return self._pool(self.model.get_image_features(pixel_values=pix))

    def encode_images(self, rgb: np.ndarray) -> np.ndarray:
        torch = self._torch
        x = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        pix = torch.from_numpy(x.transpose(0, 3, 1, 2).copy()).to(self.device, dtype=self.dtype)
        with torch.no_grad():
            e = self._img(pix).float().cpu().numpy()
        return _l2(e)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        if self.processor is not None:
            inp = self.processor(text=list(texts), padding="max_length", max_length=64, truncation=True,
                                 return_tensors="pt")
        else:
            from transformers import AutoTokenizer

            inp = AutoTokenizer.from_pretrained(self.model_id, **self._kw)(list(texts), padding="max_length",
                                                                           max_length=64, truncation=True,
                                                                           return_tensors="pt")
        inp = {k: v.to(self.device) for k, v in inp.items() if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            try:
                e = self.model.get_text_features(**inp)
            except TypeError:
                e = self.model.get_text_features(input_ids=inp["input_ids"])
            e = self._pool(e).float().cpu().numpy()
        return _l2(e)


_BACKENDS: dict[str, Any] = {}
_LOCK = threading.Lock()


def get_backend(cfg: dict, force: Optional[str] = None) -> Any:
    """Тёплый синглтон бэкенда. При ошибке загрузки SigLIP2 — исключение (вызывающий решает про fallback)."""
    f = cfg["features"]
    name = force or ("mock" if is_mock() else f.get("backend", "siglip2"))
    key = f"{name}:{f.get('model_id')}:{f.get('mock_dim')}"
    with _LOCK:
        if key not in _BACKENDS:
            if name == "mock":
                _BACKENDS[key] = MockBackend(f.get("mock_dim", 64))
            else:
                if os.environ.get("SENTINEL_OFFLINE", "1") != "0":
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                _BACKENDS[key] = SiglipBackend(f["model_id"], str(f.get("device", "auto")),
                                               bool(f.get("fp16", True)), hf_cache_dir(cfg))
        return _BACKENDS[key]


def hf_cache_dir(cfg: dict) -> Optional[str]:
    """Каталог офлайн-весов HF: env HF_HUB_CACHE/HF_HOME имеют приоритет, иначе features.hf_cache (если есть)."""
    if os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME"):
        return None  # transformers сам возьмёт из env
    p = cfg["features"].get("hf_cache")
    if not p:
        return None
    from .config import REPO_ROOT

    path = Path(p) if Path(p).is_absolute() else REPO_ROOT / p
    return str(path) if path.is_dir() else None


def hf_revision(model_id: str, cfg: dict) -> Optional[str]:
    """Коммит офлайн-снапшота модели в HF-кэше (models--org--name/refs/main) или None. Входит в ключ кэша
    признаков: другая ревизия весов -> другие эмбеддинги -> кэш и чекпоинт не переиспользуются молча."""
    dirs: list[Path] = []
    for env in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(env):
            dirs.append(Path(os.environ[env]))
    if os.environ.get("HF_HOME"):
        dirs.append(Path(os.environ["HF_HOME"]) / "hub")
    p = cfg["features"].get("hf_cache")
    if p:
        from .config import REPO_ROOT

        dirs.append(Path(p) if Path(p).is_absolute() else REPO_ROOT / p)
    dirs.append(Path.home() / ".cache" / "huggingface" / "hub")
    name = "models--" + str(model_id).replace("/", "--")
    for d in dirs:
        ref = d / name / "refs" / "main"
        try:
            if ref.is_file():
                return ref.read_text(encoding="utf-8").strip() or None
        except OSError:
            continue
    return None


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


# ----------------------------------------------------------------------------- text bank
@dataclass
class TextBank:
    texts_en: list[str]
    texts_ru: list[str]
    owner: list[str]          # type_id или "normal"
    emb: np.ndarray           # (P, D)
    logit_scale: float = 1.0
    logit_bias: float = 0.0

    @property
    def type_index(self) -> dict[str, list[int]]:
        d: dict[str, list[int]] = {}
        for i, o in enumerate(self.owner):
            d.setdefault(o, []).append(i)
        return d


def flatten_prompts(prompts: dict) -> tuple[list[str], list[str], list[str]]:
    en, ru, owner = [], [], []
    for tid, items in (prompts.get("types") or {}).items():
        for it in items:
            en.append(it["en"]); ru.append(it.get("ru", it["en"])); owner.append(str(tid))
    for it in prompts.get("normal") or []:
        en.append(it["en"]); ru.append(it.get("ru", it["en"])); owner.append("normal")
    return en, ru, owner


def get_text_bank(cfg: dict, backend: Any, use_cache: bool = True) -> TextBank:
    en, ru, owner = flatten_prompts(load_prompts(cfg))
    return text_bank_from_lists(cfg, backend, en, ru, owner, use_cache)


def text_bank_from_lists(cfg: dict, backend: Any, en: Sequence[str], ru: Sequence[str], owner: Sequence[str],
                         use_cache: bool = True) -> TextBank:
    """Банк подсказок из явных списков (на инференсе — из чекпоинта, чтобы совпадал с обучением)."""
    sig = feature_signature(cfg, backend.name)
    key = stable_hash({"sig": sig["model_id"], "texts": list(en)})
    return _text_bank_cached(key, tuple(en), tuple(ru), tuple(owner), backend, str(cache_dir(cfg)), use_cache)


@functools.lru_cache(maxsize=16)
def _text_bank_cached(key: str, en: tuple, ru: tuple, owner: tuple, backend: Any, cdir: str,
                      use_cache: bool) -> TextBank:
    path = Path(cdir) / f"text_{key}.npz"
    emb = None
    if use_cache and path.exists():
        try:
            emb = np.load(path)["emb"].astype(np.float32)
        except Exception:
            emb = None
    if emb is None:
        emb = backend.encode_texts(list(en))
        if use_cache:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, emb=emb.astype(np.float16))
            except OSError:
                pass
    return TextBank(list(en), list(ru), list(owner), emb.astype(np.float32),
                    float(getattr(backend, "logit_scale", 1.0)), float(getattr(backend, "logit_bias", 0.0)))


# ----------------------------------------------------------------------------- feature set
@dataclass
class FeatureSet:
    times: np.ndarray                 # (T,) сек
    emb: np.ndarray                   # (T, V, D) L2-нормированные
    dyn: np.ndarray                   # (T, 2 + V): motion_energy, motion_area, emb_delta[view]
    boxes: np.ndarray                 # (T, 4) нормированный bbox кропа focus (x1,y1,x2,y2)
    views: list[str]
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(len(self.times))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, times=self.times.astype(np.float32), emb=self.emb.astype(np.float16),
                            dyn=self.dyn.astype(np.float32), boxes=self.boxes.astype(np.float32),
                            meta=np.asarray(json.dumps({**self.meta, "views": self.views}, ensure_ascii=False)))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureSet":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            return cls(times=z["times"].astype(np.float32), emb=z["emb"].astype(np.float32),
                       dyn=z["dyn"].astype(np.float32), boxes=z["boxes"].astype(np.float32),
                       views=list(meta.pop("views")), meta=meta)


def sims(fs: FeatureSet, bank: TextBank) -> np.ndarray:
    """Косинусные сходства (T, V, P)."""
    T, V, D = fs.emb.shape
    return (fs.emb.reshape(T * V, D) @ bank.emb.T).reshape(T, V, -1).astype(np.float32)


def type_scores(fs: FeatureSet, bank: TextBank) -> tuple[np.ndarray, list[str]]:
    """Zero-shot отрыв типа от «нормы»: (T, K) = max_view max_prompt(type) − max_view max_prompt(normal)."""
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            s = np.nanmax(sims(fs, bank), axis=1)  # (T, P); неизвестные виды (NaN) пропускаются
    s = np.nan_to_num(s, nan=0.0)
    idx = bank.type_index
    normal = s[:, idx["normal"]].max(axis=1) if "normal" in idx else np.zeros(len(s), np.float32)
    keys = [k for k in TYPE_IDS if k in idx]
    out = np.stack([s[:, idx[k]].max(axis=1) - normal for k in keys], axis=1) if keys else \
        np.zeros((len(s), 0), np.float32)
    return out.astype(np.float32), keys


# ----------------------------------------------------------------------------- extraction
def _square_box(cx: float, cy: float, side_w: float, side_h: float, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = int(round(min(max(cx - side_w / 2, 0), max(w - side_w, 0))))
    y1 = int(round(min(max(cy - side_h / 2, 0), max(h - side_h, 0))))
    return x1, y1, int(round(min(x1 + side_w, w))), int(round(min(y1 + side_h, h)))


def _zone_box(zones: Any, names: Sequence[str] = ("cashier_area", "desk", "cash_tray")
              ) -> Optional[tuple[float, float, float, float]]:
    if not zones:
        return None
    for name in names:
        for z in zones:
            zn = getattr(z, "name", None) if not isinstance(z, dict) else z.get("name")
            poly = getattr(z, "polygon", None) if not isinstance(z, dict) else z.get("polygon")
            if zn == name and poly:
                xs, ys = [p[0] for p in poly], [p[1] for p in poly]
                return (max(min(xs), 0.0), max(min(ys), 0.0), min(max(xs), 1.0), min(max(ys), 1.0))
    return None


def sample_grid(times: Sequence[float], duration: float, fps: float, max_frames: int) -> tuple[list[int], np.ndarray]:
    """Индексы кадров клипа, ближайшие к равномерной сетке fps; повторы схлопываются."""
    tarr = np.asarray(times, dtype=np.float64)
    if len(tarr) == 0:
        return [], np.zeros(0, np.float32)
    end = max(float(duration or 0.0), float(tarr[-1]))
    step = 1.0 / max(fps, 1e-6)
    n = int(np.floor(end / step)) + 1
    if n > max_frames:
        step = end / max(max_frames - 1, 1)
        n = max_frames
    grid = np.arange(n) * step
    idx = np.clip(np.searchsorted(tarr, grid), 0, len(tarr) - 1)
    prev = np.clip(idx - 1, 0, len(tarr) - 1)
    idx = np.where(np.abs(tarr[prev] - grid) <= np.abs(tarr[idx] - grid), prev, idx)
    uniq = sorted(set(int(i) for i in idx))
    return uniq, tarr[uniq].astype(np.float32)


def extract_clip(clip: Any, backend: Any, cfg: dict, zones: Any = None,
                 augmenter: Optional[Callable[[Any], Any]] = None, fps: Optional[float] = None,
                 views: Optional[Sequence[str]] = None, deadline: Optional[float] = None) -> FeatureSet:
    """Извлечь признаки из VideoClip. ``deadline`` — time.monotonic() граница: при нехватке времени шаг редеет."""
    import cv2

    f = cfg["features"]
    fc = f["focus"]
    on_cuda = str(getattr(backend, "device", "cpu")).startswith("cuda") or backend.name == "mock"
    fps = float(fps or (f["fps"] if on_cuda else f.get("cpu_fps", 1.0)))
    views = list(views or (f["views"] if on_cuda else f.get("cpu_views", ["full"])))
    idxs, times = sample_grid(clip.times, getattr(clip, "duration", 0.0), fps, int(f.get("max_frames", 1200)))
    if not idxs:
        raise ValueError("пустой клип")
    mode = fc.get("mode", "auto")
    zbox = _zone_box(zones) if mode in ("auto", "zone") else None
    use_motion = "focus" in views and zbox is None and mode in ("auto", "motion")
    W = max(1, int(round(fc.get("motion_window_s", 1.0) * fps)))
    ds = int(fc.get("motion_downscale", 96))
    bs = int(f.get("batch_size", 64))

    n = len(idxs)
    prev_g: Optional[np.ndarray] = None
    diffs: list[np.ndarray] = []
    buf: deque = deque()
    boxes = np.zeros((n, 4), np.float32)
    energy = np.zeros(n, np.float32)
    area = np.zeros(n, np.float32)
    pending: list[np.ndarray] = []
    pending_idx: list[tuple[int, int]] = []
    emb_arr = np.full((n, len(views), int(backend.dim)), np.nan, np.float32)   # NaN = вид неизвестен (не «пусто»)
    emitted = [0]
    zone_boxes = {v: _zone_box(zones, names=(v.split(":", 1)[1],)) for v in views if v.startswith("zone:")}
    center: Optional[np.ndarray] = None
    meta: dict = {"fps": fps, "views": views, "focus_mode": "zone" if zbox else ("motion" if use_motion else "none")}
    t0 = time.monotonic()

    def flush(final: bool = False) -> None:
        while len(pending) >= bs or (final and pending):
            batch = np.stack(pending[:bs])
            where = pending_idx[:bs]
            del pending[:bs]
            del pending_idx[:bs]
            e = backend.encode_images(batch)
            for (kk, vi), row in zip(where, e):
                emb_arr[kk, vi] = row

    def emit(k: int, frame: np.ndarray) -> None:
        nonlocal center
        h, w = frame.shape[:2]
        crops: list[tuple[int, np.ndarray]] = []
        for vi, v in enumerate(views):
            if v == "full":
                crops.append((vi, frame))
                continue
            if v.startswith("zone:"):  # кроп по зоне A06; зоны нет -> вид неизвестен (NaN), а не «ноль»
                zb = zone_boxes.get(v)
                if zb is None:
                    continue
                x1, y1, x2, y2 = zb
                side_min = fc.get("min_side", 0.45) * min(w, h) * 0.5
                bw, bh = max((x2 - x1) * w, side_min), max((y2 - y1) * h, side_min)
                side = min(max(bw, bh), min(w, h))
                X1, Y1, X2, Y2 = _square_box((x1 + x2) / 2 * w, (y1 + y2) / 2 * h, side, side, w, h)
                crops.append((vi, frame[Y1:Y2, X1:X2]))
                continue
            if v != "focus":
                continue
            # focus
            if zbox is not None:
                x1, y1, x2, y2 = zbox
                cx, cy = (x1 + x2) / 2 * w, (y1 + y2) / 2 * h
                bw, bh = (x2 - x1) * w, (y2 - y1) * h
            else:
                lo, hi = max(0, k - W + 1), min(len(diffs) - 1, k + W)
                acc = np.max(np.stack(diffs[lo:hi + 1]), axis=0) if hi >= lo and diffs else None
                cx, cy, bw, bh = w / 2, h / 2, float(w), float(h)
                if acc is not None and acc.max() > 0:
                    thr = max(float(np.percentile(acc, 97)), 12.0)
                    ys, xs = np.nonzero(acc >= thr)
                    if len(xs) >= 3:
                        sy, sx = h / acc.shape[0], w / acc.shape[1]
                        cx, cy = (xs.mean() + 0.5) * sx, (ys.mean() + 0.5) * sy
                        bw = (xs.max() - xs.min() + 1) * sx * fc.get("expand", 1.35)
                        bh = (ys.max() - ys.min() + 1) * sy * fc.get("expand", 1.35)
                c = np.array([cx, cy], np.float32)
                a = float(fc.get("smooth_alpha", 0.6))
                center = c if center is None else a * center + (1 - a) * c
                cx, cy = float(center[0]), float(center[1])
            side_min = fc.get("min_side", 0.45) * min(w, h)
            side = min(max(bw, bh, side_min), min(w, h))
            x1, y1, x2, y2 = _square_box(cx, cy, side, side, w, h)
            boxes[k] = (x1 / w, y1 / h, x2 / w, y2 / h)
            crops.append((vi, frame[y1:y2, x1:x2]))
        if "focus" not in views:
            boxes[k] = (0, 0, 1, 1)
        for vi, c in crops:
            rgb = cv2.cvtColor(cv2.resize(np.ascontiguousarray(c), (IMG, IMG), interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2RGB)
            pending.append(rgb)
            pending_idx.append((k, vi))
        emitted[0] = k + 1
        flush()

    for j, ci in enumerate(idxs):
        frame = clip.frame(ci)
        if augmenter is not None:
            frame = augmenter(frame)
        h, w = frame.shape[:2]
        s = ds / max(h, w)
        g = cv2.cvtColor(cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))),
                                    interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (3, 3), 0).astype(np.float32)
        d = np.abs(g - prev_g) if prev_g is not None and prev_g.shape == g.shape else np.zeros_like(g)
        prev_g = g
        diffs.append(d)
        energy[j] = float(d.mean() / 255.0)
        area[j] = float((d > 15).mean())
        buf.append((j, frame))
        while buf and (not use_motion or buf[0][0] + W <= j):
            k, fr = buf.popleft()
            emit(k, fr)
        if deadline is not None and time.monotonic() > deadline:
            meta["truncated_at"] = float(times[j])
            break
    while buf:
        k, fr = buf.popleft()
        emit(k, fr)
    flush(final=True)
    n_done = int(emitted[0])
    # точность кэша (float16) сразу: свежий и кэшированный пути дают бит-в-бит одинаковый вход (и как на обучении)
    emb = emb_arr[:n_done].astype(np.float16).astype(np.float32)
    meta["zones_found"] = sorted(v for v, b in zone_boxes.items() if b is not None)
    times, boxes, energy, area = times[:n_done], boxes[:n_done], energy[:n_done], area[:n_done]
    delta = np.zeros((n_done, len(views)), np.float32)
    if n_done > 1:
        delta[1:] = 1.0 - np.sum(emb[1:] * emb[:-1], axis=-1)
    dyn = np.concatenate([energy[:, None] * 10.0, area[:, None], delta * 5.0], axis=1).astype(np.float32)
    meta.update({"backend": backend.name, "duration": float(getattr(clip, "duration", times[-1] if n_done else 0)),
                 "width": int(getattr(clip, "width", 0)), "height": int(getattr(clip, "height", 0)),
                 "extract_s": round(time.monotonic() - t0, 3), "n": n_done})
    return FeatureSet(times=times, emb=emb.astype(np.float32), dyn=dyn, boxes=boxes, views=views, meta=meta)


def cache_path(cfg: dict, key: dict) -> Path:
    return cache_dir(cfg) / f"{stable_hash(key, 16)}.npz"
