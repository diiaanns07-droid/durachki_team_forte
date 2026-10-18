from __future__ import annotations

from ml.track import track_full
from ml.track.config import load_config
from ml.track.tests import synth as S


def run(scene, cfg=None, **kw):
    clip, dets, gt = S.make(scene, **kw)
    res = track_full(clip, dets, cfg=cfg)
    ids: dict[str, set] = {}
    for d in dets:
        if id(d) in gt and d.track_id is not None:
            ids.setdefault(gt[id(d)]["key"], set()).add(d.track_id)
    by_id = {t.track_id: t for t in res.tracks}
    zones = {z.name: z for z in res.zones}
    return res, ids, by_id, zones, clip


def role_of(ids, by_id, key):
    tids = ids[key]
    assert len(tids) >= 1
    return [by_id[t].role for t in sorted(tids)], [by_id[t].role_conf for t in sorted(tids)]


def cfg_with(**sections):
    """load_config() + переопределения секций: cfg_with(appearance={"enabled": True})."""
    return load_config(overrides=sections)
