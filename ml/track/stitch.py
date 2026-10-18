"""Склейка треклетов после окклюзий / пропусков детектора / ухода-возврата.

Люди:
* короткий разрыв (<= short_gap_s): по положению (допуск растёт со временем разрыва),
  внешность (если есть) — вето;
* длинный разрыв: ТОЛЬКО при совпадении внешности (гистограмма одежды) и правдоподобном
  месте (вернулся «на своё место» или появился там же, где исчез). Без внешности длинные
  разрывы не склеиваются: лучше два трека, чем чужой человек под ID кассира.
Статичные предметы: склейка мерцания по IoU медианных боксов.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .appearance import bhattacharyya
from .bytetrack import Tracklet
from .geometry import box_center, box_diag, iou, iso_dist


def _chains(tracklets: Sequence[Tracklet], nxt: dict[int, int], prv: dict[int, int]) -> list[list[Tracklet]]:
    by = {t.tid: t for t in tracklets}
    out = []
    for t in sorted(tracklets, key=lambda x: (x.t_start, x.tid)):
        if t.tid in prv:
            continue
        chain = [t]
        while chain[-1].tid in nxt:
            chain.append(by[nxt[chain[-1].tid]])
        out.append(chain)
    return out


def _link(cands: list[tuple[float, int, int]]) -> tuple[dict[int, int], dict[int, int]]:
    nxt: dict[int, int] = {}
    prv: dict[int, int] = {}
    for _, a, b in sorted(cands):
        if a in nxt or b in prv:
            continue
        nxt[a] = b
        prv[b] = a
    return nxt, prv


def stitch_persons(tracklets: Sequence[Tracklet], app: dict[int, np.ndarray], cfg: dict,
                   aspect: float, bridges: Sequence[Sequence[float]] = (),
                   vcfg: Optional[dict] = None) -> tuple[list[list[Tracklet]], list[dict]]:
    if not cfg.get("enabled", True) or len(tracklets) < 2:
        return [[t] for t in sorted(tracklets, key=lambda x: (x.t_start, x.tid))], []
    cands: list[tuple[float, int, int]] = []
    why: dict[tuple[int, int], dict] = {}
    for a in tracklets:
        ma = a.median_box()
        ca_med = box_center(ma)
        for b in tracklets:
            if a is b:
                continue
            gap = b.t_start - a.t_end
            if gap <= 1e-6 or gap > cfg["long_gap_s"]:
                continue
            mb = b.median_box()
            sa = max(box_diag(ma, aspect), 1e-6)
            sb = max(box_diag(mb, aspect), 1e-6)
            if max(sa / sb, sb / sa) > cfg["size_ratio_max"]:
                continue
            diag = 0.5 * (box_diag(a.boxes[-1], aspect) + box_diag(b.boxes[0], aspect)) + 1e-6
            d = iso_dist(box_center(a.boxes[-1]), box_center(b.boxes[0]), aspect) / diag
            ad: Optional[float] = bhattacharyya(app.get(a.tid), app.get(b.tid))
            bridged = False
            if bridges and vcfg and gap <= vcfg["bridge_max_s"]:
                for la, lb in bridges:  # H-VIS: разрыв совпал с потерей видимости
                    if la - 1.5 <= a.t_end <= la + 0.5 and lb - 0.5 <= b.t_start <= lb + 1.5 and d <= vcfg["bridge_dist_diag"]:
                        bridged = True
                        break
            if bridged:
                cost = 0.5 + 0.1 * d
                kind = "visibility_bridge"
            elif gap <= cfg["short_gap_s"]:
                dmax = cfg["base_dist_diag"] + cfg["max_speed_diag_s"] * gap
                if d > dmax or (ad is not None and ad > cfg["app_thr_short"]):
                    continue
                cost = 0.5 * d / dmax + (ad if ad is not None else 0.25)
                kind = "short_gap"
            else:
                if ad is None:
                    if cfg.get("long_requires_appearance", True):
                        continue
                    ad_eff = 0.5
                else:
                    ad_eff = ad
                if ad_eff > cfg["app_thr_long"] and ad is not None:
                    continue
                dm = iso_dist(ca_med, box_center(mb), aspect) / diag
                if dm > cfg["long_home_diag"] and d > cfg["long_entry_diag"]:
                    continue
                cost = 1.0 + ad_eff + 0.05 * min(d, 10.0)
                kind = "long_gap_appearance" if ad is not None else "long_gap_position"
            cands.append((cost, a.tid, b.tid))
            why[(a.tid, b.tid)] = {"from": a.tid, "to": b.tid, "kind": kind, "gap_s": round(gap, 3),
                                   "dist_diag": round(d, 3), "app_dist": None if ad is None else round(ad, 3)}
    nxt, prv = _link(cands)
    log = [why[(a, b)] for a, b in nxt.items()]
    return _chains(tracklets, nxt, prv), log


def stitch_static(tracklets: Sequence[Tracklet], cfg: dict) -> list[list[Tracklet]]:
    if len(tracklets) < 2:
        return [[t] for t in tracklets]
    cands = []
    for a in tracklets:
        ma = a.median_box()
        for b in tracklets:
            if a is b:
                continue
            gap = b.t_start - a.t_end
            if gap <= 1e-6 or gap > cfg["static_gap_s"]:
                continue
            ov = iou(ma, b.median_box())
            if ov >= cfg["static_iou"]:
                cands.append((1.0 - ov + 0.01 * gap, a.tid, b.tid))
    nxt, prv = _link(cands)
    return _chains(tracklets, nxt, prv)


def merge_chain(chain: Sequence[Tracklet]) -> Tracklet:
    head = chain[0]
    out = Tracklet(tid=head.tid, group=head.group, label=head.label)
    for t in sorted(chain, key=lambda x: x.t_start):
        out.times += t.times
        out.boxes += t.boxes
        out.scores += t.scores
        out.labels += t.labels
        out.refs += t.refs
    return out
