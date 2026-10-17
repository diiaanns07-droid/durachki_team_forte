"""Геометрия боксов: IoU, NMS, WBF, флип, тайлы (SAHI-стиль, своя реализация), нормировка.

Все боксы — нормированные (x1, y1, x2, y2) в долях кадра, numpy float32 массивы формы (N, 4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass
class RawDets:
    """Сырые детекции одного кадра от одного бэкенда/вида (до слияния)."""

    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))  # (N,4) норм. xyxy
    scores: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.float32))   # (N,)
    labels: list[str] = field(default_factory=list)                                   # (N,) из LABELS
    sources: list[str] = field(default_factory=list)                                  # (N,)

    def __len__(self) -> int:
        return len(self.labels)

    @staticmethod
    def concat(parts: Sequence["RawDets"]) -> "RawDets":
        parts = [p for p in parts if len(p)]
        if not parts:
            return RawDets()
        return RawDets(
            boxes=np.concatenate([p.boxes for p in parts]).astype(np.float32),
            scores=np.concatenate([p.scores for p in parts]).astype(np.float32),
            labels=[lab for p in parts for lab in p.labels],
            sources=[s for p in parts for s in p.sources],
        )

    def select(self, idx: Sequence[int] | np.ndarray) -> "RawDets":
        idx = np.asarray(idx, dtype=np.int64)
        return RawDets(
            boxes=self.boxes[idx].reshape(-1, 4),
            scores=self.scores[idx].reshape(-1),
            labels=[self.labels[i] for i in idx],
            sources=[self.sources[i] for i in idx],
        )


def clip01(boxes: np.ndarray) -> np.ndarray:
    b = np.clip(np.asarray(boxes, dtype=np.float32).reshape(-1, 4), 0.0, 1.0)
    # гарантируем x1<=x2, y1<=y2
    x1 = np.minimum(b[:, 0], b[:, 2])
    x2 = np.maximum(b[:, 0], b[:, 2])
    y1 = np.minimum(b[:, 1], b[:, 3])
    y2 = np.maximum(b[:, 1], b[:, 3])
    return np.stack([x1, y1, x2, y2], axis=1)


def cxcywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def area(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = area(a)[:, None] + area(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0).astype(np.float32)


def ios_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over area(a): доля бокса a, покрытая боксом b (для подавления «кусков» из тайлов)."""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return (inter / np.maximum(area(a)[:, None], 1e-12)).astype(np.float32)


def suppress_contained(extra: RawDets, base: RawDets, ios_thr: float = 0.8) -> RawDets:
    """Убирает из extra боксы, почти целиком лежащие внутри бокса той же метки из base."""
    if len(extra) == 0 or len(base) == 0:
        return extra
    ios = ios_matrix(extra.boxes, base.boxes)
    bl = np.asarray(base.labels)
    keep = [i for i, lab in enumerate(extra.labels) if not np.any((ios[i] >= ios_thr) & (bl == lab))]
    return extra.select(keep)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Классический greedy NMS. Возвращает индексы оставленных боксов по убыванию score."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def classwise_nms(d: RawDets, iou_thr: float) -> RawDets:
    if len(d) == 0:
        return d
    keep: list[int] = []
    labels = np.asarray(d.labels)
    for lab in sorted(set(d.labels)):
        idx = np.nonzero(labels == lab)[0]
        k = nms(d.boxes[idx], d.scores[idx], iou_thr)
        keep.extend(idx[k].tolist())
    keep.sort(key=lambda i: -float(d.scores[i]))
    return d.select(keep)


def weighted_boxes_fusion(views: Sequence[RawDets], iou_thr: float = 0.55,
                          single_view_penalty: float = 0.85) -> RawDets:
    """WBF (идея Solovyev et al., своя реализация) поверх нескольких «видов» (оригинал/флип/тайлы).

    Боксы одной метки кластеризуются жадно по IoU; координаты усредняются с весами score,
    score = среднее по видам (отсутствие в виде = 0), но не ниже max*single_view_penalty —
    чтобы TTA не «убивал» объекты, видимые только в одном виде (важно для полноты).
    """
    n_views = max(1, len(views))
    allb = RawDets.concat(list(views))
    if len(allb) == 0:
        return RawDets()
    view_ids = np.concatenate([np.full(len(v), i) for i, v in enumerate(views) if len(v)]) if len(allb) else np.zeros(0)
    out_boxes, out_scores, out_labels, out_sources = [], [], [], []
    labels = np.asarray(allb.labels)
    for lab in sorted(set(allb.labels)):
        idx = np.nonzero(labels == lab)[0]
        idx = idx[np.argsort(-allb.scores[idx], kind="stable")]
        clusters: list[list[int]] = []
        fused: list[np.ndarray] = []
        for i in idx:
            b = allb.boxes[i]
            best, best_iou = -1, iou_thr
            if fused:
                ious = iou_matrix(b[None], np.stack(fused))[0]
                j = int(np.argmax(ious))
                if ious[j] > best_iou:
                    best = j
            if best < 0:
                clusters.append([int(i)])
                fused.append(b.copy())
            else:
                clusters[best].append(int(i))
                members = clusters[best]
                w = allb.scores[members][:, None]
                fused[best] = (allb.boxes[members] * w).sum(0) / max(float(w.sum()), 1e-12)
        for members, fb in zip(clusters, fused):
            sc = allb.scores[members]
            per_view: dict[int, float] = {}
            for m in members:
                v = int(view_ids[m])
                per_view[v] = max(per_view.get(v, 0.0), float(allb.scores[m]))
            mean_score = sum(per_view.values()) / n_views
            if len(per_view) < n_views:  # виден не во всех видах: не ниже max*penalty (полнота важнее)
                score = min(max(mean_score, float(sc.max()) * single_view_penalty), float(sc.max()))
            else:
                score = mean_score
            out_boxes.append(fb)
            out_scores.append(score)
            out_labels.append(lab)
            srcs = sorted({allb.sources[m] for m in members})
            out_sources.append("+".join(srcs))
    res = RawDets(boxes=clip01(np.asarray(out_boxes)), scores=np.asarray(out_scores, np.float32),
                  labels=out_labels, sources=out_sources)
    order = np.argsort(-res.scores, kind="stable")
    return res.select(order)


def hflip_boxes(boxes: np.ndarray) -> np.ndarray:
    """Перевод боксов, найденных на горизонтально отражённом кадре, обратно в исходные координаты."""
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    x1 = 1.0 - b[:, 2]
    x2 = 1.0 - b[:, 0]
    b[:, 0], b[:, 2] = x1, x2
    return b


@dataclass(frozen=True)
class Tile:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0


def _axis_starts(length: int, tile: int, overlap: float) -> list[int]:
    """Минимальное число окон с перекрытием >= overlap, равномерно распределённых по оси."""
    if tile >= length:
        return [0]
    step = max(1.0, tile * (1.0 - float(overlap)))
    n = int(np.ceil((length - tile) / step)) + 1
    return [int(round(v)) for v in np.linspace(0, length - tile, n)]


def make_tiles(width: int, height: int, tile_frac: float = 0.6, overlap: float = 0.25) -> list[Tile]:
    """Квадратные тайлы со стороной tile_frac × короткой стороны кадра, с перекрытием, покрывающие кадр целиком."""
    side = max(16, int(round(min(width, height) * float(tile_frac))))
    side_w, side_h = min(side, width), min(side, height)
    tiles = [Tile(x, y, x + side_w, y + side_h)
             for y in _axis_starts(height, side_h, overlap)
             for x in _axis_starts(width, side_w, overlap)]
    return tiles


def tile_to_frame(boxes: np.ndarray, tile: Tile, width: int, height: int) -> np.ndarray:
    """Нормированные к тайлу боксы → нормированные к кадру."""
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    b[:, [0, 2]] = (b[:, [0, 2]] * tile.w + tile.x0) / float(width)
    b[:, [1, 3]] = (b[:, [1, 3]] * tile.h + tile.y0) / float(height)
    return clip01(b)


def touches_inner_edge(boxes: np.ndarray, tile: Tile, width: int, height: int, margin: float) -> np.ndarray:
    """Маска боксов (в координатах тайла 0..1), касающихся границы тайла, которая НЕ является границей кадра."""
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    m = np.zeros(len(b), dtype=bool)
    if tile.x0 > 0:
        m |= b[:, 0] <= margin
    if tile.y0 > 0:
        m |= b[:, 1] <= margin
    if tile.x1 < width:
        m |= b[:, 2] >= 1.0 - margin
    if tile.y1 < height:
        m |= b[:, 3] >= 1.0 - margin
    return m


def filter_by_area(d: RawDets, min_area: float = 0.0, max_area: Optional[dict[str, float]] = None) -> RawDets:
    if len(d) == 0:
        return d
    a = area(d.boxes)
    keep = []
    for i, lab in enumerate(d.labels):
        mx = 1.0
        if max_area:
            mx = float(max_area.get(lab, max_area.get("default", 1.0)))
        if min_area <= a[i] <= mx:
            keep.append(i)
    return d.select(keep)
