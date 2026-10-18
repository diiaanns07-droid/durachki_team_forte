"""Бонус: кластеризация окон вокруг размеченных секунд (k-means по эмбеддингам) — помогает людям быстро
разметить ТИПЫ нарушений. Сетки кадров по кластерам пишутся в outputs/temporal/clusters/ (в git не попадают).

    python -m ml.temporal.cluster [--k 5] [--videos data/raw] [--no-images]

Важно: кадры из датасета остаются локально; в отчёт попадают только имена видео, секунды и номера кластеров.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .config import load_config, outputs_dir

log = logging.getLogger("sentinel.temporal")


def kmeans(X: np.ndarray, k: int, seed: int = 0, iters: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """k-means++ (косинус: X L2-нормированы). Возвращает (labels, centers)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    k = max(1, min(k, n))
    centers = [X[int(rng.integers(0, n))]]
    for _ in range(1, k):
        d = np.min(np.stack([np.sum((X - c) ** 2, axis=1) for c in centers]), axis=0)
        p = d / d.sum() if d.sum() > 0 else np.full(n, 1.0 / n)
        centers.append(X[int(rng.choice(n, p=p))])
    C = np.stack(centers)
    lab = np.zeros(n, int)
    for _ in range(iters):
        lab_new = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        if np.array_equal(lab_new, lab) and _ > 0:
            break
        lab = lab_new
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return lab, C


def window_embeddings(samples: Sequence, half_s: float = 1.5) -> tuple[np.ndarray, list[dict]]:
    """Средний эмбеддинг (все виды) в окне ±half_s вокруг каждой размеченной секунды (только оригиналы)."""
    X, meta = [], []
    for s in samples:
        if not s.label:
            continue
        fs = s.versions[0]
        for t in s.times:
            m = np.abs(fs.times - t) <= half_s
            if not m.any():
                continue
            e = fs.emb[m].reshape(int(m.sum()), -1).mean(0)
            X.append(e / max(np.linalg.norm(e), 1e-8))
            meta.append({"name": s.name, "t": float(t)})
    return (np.stack(X) if X else np.zeros((0, 1), np.float32)), meta


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ml.temporal: кластеры окон нарушений для разметки типов")
    ap.add_argument("--index", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--videos", default="data/raw")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    from .data_index import load_samples

    samples, _ = load_samples(cfg, a.index, max_aug=0)
    X, meta = window_embeddings(samples)
    if len(X) == 0:
        print("нет позитивных окон")
        return 1
    lab, _ = kmeans(X, a.k, seed=int(cfg["train"]["seed"]))
    out = Path(a.out) if a.out else outputs_dir(cfg) / "clusters"
    out.mkdir(parents=True, exist_ok=True)
    clusters: dict[int, list[dict]] = {}
    for m, l in zip(meta, lab):
        clusters.setdefault(int(l), []).append(m)
    (out / "clusters.json").write_text(json.dumps({"k": int(lab.max()) + 1, "clusters": clusters},
                                                  ensure_ascii=False, indent=2), encoding="utf-8")
    if not a.no_images:
        _grids(clusters, Path(a.videos), out)
    for c, items in sorted(clusters.items()):
        print(f"кластер {c}: " + ", ".join(f"{m['name']}@{m['t']:.0f}s" for m in items))
    print(f"-> {out} (локально, не коммитить)")
    return 0


def _grids(clusters: dict[int, list[dict]], videos: Path, out: Path, tile: int = 160) -> None:
    import cv2

    from .labels import find_videos

    vids = find_videos(videos)
    for c, items in clusters.items():
        tiles = []
        for m in items[:16]:
            p = vids.get(m["name"])
            if p is None:
                continue
            cap = cv2.VideoCapture(str(p))
            cap.set(cv2.CAP_PROP_POS_MSEC, m["t"] * 1000.0)
            ok, fr = cap.read()
            cap.release()
            if ok:
                h, w = fr.shape[:2]
                s = tile / max(h, w)
                fr = cv2.resize(fr, (int(w * s), int(h * s)))
                canvas = np.zeros((tile, tile, 3), np.uint8)
                canvas[: fr.shape[0], : fr.shape[1]] = fr
                cv2.putText(canvas, f"{m['name']} {m['t']:.0f}s", (3, tile - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 255), 1)
                tiles.append(canvas)
        if tiles:
            cols = 4
            rows = [np.hstack(tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols])))
                    for i in range(0, len(tiles), cols)]
            cv2.imwrite(str(out / f"cluster_{c}.jpg"), np.vstack(rows))


if __name__ == "__main__":
    raise SystemExit(main())
