"""Регрессия A06 после аудита (синтетика): роли, потерянные треки, окно refine, передача конфига.

Потеря трека сама по себе не означает ни ухода кассира, ни нарушения: ответы — None («неизвестно»).
"""
import os
import time

import numpy as np
import pytest

from ml.track import TrackQueries, model_info, track_full
from ml.track.config import load_config
from ml.track.contract import Detection
from ml.track.tests import synth as S
from ml.track.tests._util import role_of, run


def _q(res):
    return TrackQueries(res.tracks, res.zones, res.info)


def _kinds(q, kind):
    return [e for e in q.events() if e.kind == kind]


@pytest.mark.parametrize("seed", range(5))
def test_crossing_in_front_of_cashier_keeps_cashier_and_no_leave(seed):
    # посетитель проходит прямо перед кассиром; кассир перекрыт и не детектируется 1.4 с
    # (без tracker.lost_cost_penalty потерянный трек кассира забирал детекцию посетителя — подмена ID)
    res, ids, by_id, _, _ = run(S.scene_crossing_front(), seed=seed)
    assert len(ids["cashier"]) == 1 and len(ids["visitor"]) == 1 and ids["cashier"] != ids["visitor"]
    assert role_of(ids, by_id, "cashier")[0] == ["cashier"]
    q = _q(res)
    assert not _kinds(q, "cashier_left_workplace") and not _kinds(q, "cashier_left_room")


def test_hidden_behind_occlusion_new_id_is_not_leave_nor_entry():
    sc = S.scene_clean(60.0)
    sc.persons[0].miss = [(20.0, 30.0)]  # 10 с в глубине кадра без детекций -> трек рвётся (смена ID)
    res, ids, by_id, _, _ = run(sc)
    roles, _ = role_of(ids, by_id, "cashier")
    assert len(roles) == 2 and set(roles) == {"cashier"}  # новый фрагмент — кассир по поведению (сторона из сейфа)
    q = _q(res)
    for t in (21.0, 25.0, 29.0):
        lw, lr = q.cashier_left_workplace(t), q.cashier_left_room(t)
        assert lw.value is None and lr.value is None and "lost_in_view" in lw.source
    assert not _kinds(q, "cashier_left_workplace") and not _kinds(q, "cashier_left_room")
    assert [e.track_ids for e in _kinds(q, "person_entered")] == [sorted(ids["visitor"])]  # фрагмент — не вход


def test_reentry_via_door_is_entry_and_cashier_role_kept():
    res, ids, by_id, _, _ = run(S.scene_v02())
    q = _q(res)
    lr = _kinds(q, "cashier_left_room")
    assert len(lr) == 1 and abs(lr[0].t - 19.0) < 0.5 and lr[0].source == "last_seen_in_door_zone"
    ent = _kinds(q, "person_entered")
    assert len(ent) == 1 and abs(ent[0].t - 40.0) < 0.5 and ent[0].role == "cashier"
    assert q.cashier_left_workplace(50.0).value is False


@pytest.mark.parametrize("every_n", [5, 10])
def test_sparse_detections_no_false_events(every_n):
    clip, dets, gt = S.make(S.scene_clean(40.0), every_n=every_n)
    res = track_full(clip, dets)
    roles = {t.role for t in res.tracks if t.label == "person"}
    assert "cashier" in roles and roles <= {"cashier", "visitor", "unknown"}
    q = _q(res)
    assert not _kinds(q, "cashier_left_workplace") and not _kinds(q, "outsider_at_cashier_side")
    assert len(_kinds(q, "person_entered")) <= 1


def test_very_few_detections_are_unknown_not_violation():
    dets = [Detection(t, "person", (0.40, 0.66, 0.60, 0.80), 0.9) for t in (1.0, 1.2, 1.4)]
    clip = S.SynthClip(S.scene_clean(10.0))
    res = track_full(clip, dets)
    assert res.info["coverage"]["frames_processed"] == 3
    persons = [t for t in res.tracks if t.label == "person"]
    assert all(t.role in ("unknown", "cashier") for t in persons)
    q = _q(res)
    assert q.events() == [] or all(e.kind != "cashier_left_workplace" for e in q.events())
    assert q.cashier_left_workplace(5.0).value in (None, False)


def test_cashier_is_not_the_first_person_to_appear():
    client = S.Person("client", [(0, *S.CLIENT), (40, *S.CLIENT)], [(0.0, 40.0)], color=(40, 40, 220))
    cas = S.Person("cashier", [(8.0, *S.DOOR_PT), (11.0, *S.SEAT), (40, *S.SEAT)], [(8.0, 40.0)])
    res, ids, by_id, _, _ = run(S.Scene(40.0, [client, cas], S.base_objects(40.0)))
    assert role_of(ids, by_id, "client")[0] == ["visitor"]
    assert role_of(ids, by_id, "cashier")[0] == ["cashier"]
    assert "present_from_start" not in res.info["roles"][sorted(ids["cashier"])[0]]["reasons"]


class _WindowLike:
    """Как backend/pipeline/worker.py WindowClip: times — только окно, duration — весь ролик."""

    def __init__(self, base, times):
        self.times, self.duration, self.width, self.height = list(times), base.duration, base.width, base.height

    def frame(self, i):
        raise RuntimeError("no frames")


def _window_dets(scene, times):
    _, dets, _ = S.make(scene)
    by_t: dict = {}
    for d in dets:
        by_t.setdefault(round(d.t, 4), []).append(d)
    keys = np.array(sorted(by_t))
    out = []
    for t in times:
        k = keys[int(np.argmin(np.abs(keys - t)))]
        out += [Detection(float(t), d.label, d.bbox, d.score) for d in by_t[k]]
    return out


def test_refine_window_normalizes_by_sampled_span_not_full_duration():
    sc = S.scene_clean(60.0)
    sc.objects.append(S.Obj("cell_phone", (0.62, 0.47, 0.66, 0.50), [(0.0, 60.0)], 0.7))  # стационарный телефон
    base = S.SynthClip(sc)
    wt = [round(20.0 + k / 8.0, 4) for k in range(81)]  # окно 20..30 с, 8 fps
    win = _WindowLike(base, wt)
    full_like = _WindowLike(base, wt)
    full_like.duration = 10.125
    r_win = track_full(win, _window_dets(sc, wt))
    r_ref = track_full(full_like, _window_dets(sc, wt))
    zw = {z.name: z.conf for z in r_win.zones}
    zr = {z.name: z.conf for z in r_ref.zones}
    assert zw.keys() == zr.keys()
    for k in zw:
        assert abs(zw[k] - zr[k]) < 0.02, (k, zw[k], zr[k])
    assert "cell_phone" in r_win.info["zones"]["desk"]["evidence"]
    cov = r_win.info["coverage"]
    assert cov["span_frac"] == 1.0 and cov["declared_duration"] == 60.0 and cov["t_window"] == [20.0, 30.0]
    assert all(t.boxes[0][0] >= 20.0 for t in r_win.tracks)  # время абсолютное (с начала ролика)


def test_config_reload_and_model_info_follow_external_yaml(tmp_path, monkeypatch):
    p = tmp_path / "track.yaml"
    p.write_text("queries:\n  smoothing: 'off'\n", encoding="utf-8")
    monkeypatch.setenv("SENTINEL_TRACK_CONFIG", str(p))
    a, ia = load_config()["queries"]["smoothing"], model_info()
    time.sleep(0.01)
    p.write_text("queries:\n  smoothing: hysteresis\n", encoding="utf-8")
    os.utime(p, None)
    b, ib = load_config()["queries"]["smoothing"], model_info()
    assert (a, b) == ("off", "hysteresis")
    assert ia["config_sha"] != ib["config_sha"] and ib["config_from_env"] is True


@pytest.mark.parametrize("seed", range(8))
def test_speed_gate_flag_prevents_jump_to_neighbour(seed):
    # H-SPEED (флаг, по умолчанию ВЫКЛ): двое сидят рядом, детекции выпадают; трек не «прыгает» на соседа
    cfg = load_config(overrides={"tracker": {"speed_gate": True}})
    res, ids, by_id, _, _ = run(S.scene_two_staff(), cfg=cfg, seed=seed, drop=0.15)
    assert not (ids["cashier"] & ids["staff2"])


def test_new_flags_defaults():
    t = load_config()["tracker"]
    assert t["lost_cost_penalty"] > 0 and t["speed_gate"] is False


def _a03_cache(tmp_path):
    """Кэш стадий в формате A03 (backend/pipeline/cache.py): cache/<sha256>/dets.jsonl + manifest.json.
    «Видео» — синтетические байты только для sha256 (реальных роликов нет)."""
    import hashlib
    import json

    from ml.track.serialize import detection_to_dict

    vids = tmp_path / "videos"
    vids.mkdir()
    for name, sc in (("Х-1.mov", S.scene_v02()), ("X-2.mov", S.scene_clean())):  # кириллическая и латинская X
        raw = f"SYNTHETIC-{name}".encode()
        (vids / name).write_bytes(raw)
        d = tmp_path / "cache" / hashlib.sha256(raw).hexdigest()
        d.mkdir(parents=True)
        clip, dets, _ = S.make(sc)
        (d / "dets.jsonl").write_text("\n".join(json.dumps(detection_to_dict(x)) for x in dets), encoding="utf-8")
        (d / "manifest.json").write_text(json.dumps({"video": {"duration": clip.duration, "fps": 30.0, "width": 464,
                                                               "height": 848, "n_times": len(clip.times),
                                                               "sample_fps": 5.0}, "stages": {}}), encoding="utf-8")
    (tmp_path / "labels.csv").write_text("video,label,second\nX-1.mov,violation,16\nХ-2.mov,no_violation,\n",
                                         encoding="utf-8")
    return tmp_path


def test_replay_real_reads_a03_stage_cache_and_maps_sha_to_names(tmp_path):
    import json

    from ml.track.replay import main as replay_main

    root = _a03_cache(tmp_path)
    out = root / "out"
    rc = replay_main(["--real", "--dets-glob", str(root / "cache" / "*" / "dets.jsonl"), "--videos-dir",
                      str(root / "videos"), "--labels", str(root / "labels.csv"), "--variant", "smooth",
                      "--out", str(out)])
    assert rc == 0
    s = json.loads((out / "replay.json").read_text())["summary"]["baseline"]
    assert s["videos"] == 2 and s["unlabeled_videos"] == 0
    assert s["labeled_violation"] == 1 and s["violation_hit_within_2.0s"] == 1
    assert s["clean_with_alarm (FP-proxy)"] == 0


def test_replay_real_without_files_fails_loudly(tmp_path):
    from ml.track.replay import main as replay_main

    with pytest.raises(SystemExit) as e:
        replay_main(["--real", "--dets-glob", str(tmp_path / "nope" / "*" / "dets.jsonl"), "--out", str(tmp_path)])
    assert "не найдено" in str(e.value)


def test_report_and_loaders_read_a03_tracks_zones(tmp_path):
    import json

    pytest.importorskip("cv2")
    from ml.track.report import main as report_main
    from ml.track.serialize import load_clip_meta, load_detections, load_track_cache, track_to_dict, zone_to_dict

    root = _a03_cache(tmp_path)
    d = sorted((root / "cache").iterdir())[0]
    dets = load_detections(str(d))
    assert dets and all(x.t >= 0 for x in dets)
    assert load_clip_meta(str(d))["width"] == 464
    res = track_full(None, dets)
    rows = [track_to_dict(t) for t in res.tracks]
    for r in rows:  # форма serde A03: boxes = [[t, [x1, y1, x2, y2]], ...]
        r["boxes"] = [[b[0], b[1:5]] for b in r["boxes"]]
    (d / "tracks.json").write_text(json.dumps(rows), encoding="utf-8")
    (d / "zones.json").write_text(json.dumps([zone_to_dict(z) for z in res.zones]), encoding="utf-8")
    tracks, zones, info = load_track_cache(str(d))
    assert [t.track_id for t in tracks] == [t.track_id for t in res.tracks] and info == {}
    assert report_main(["--cache", str(d), "--out", str(tmp_path / "t.png")]) == 0


def test_replay_real_reads_eval_labels_schema(tmp_path):
    """Метки в схеме eval.labels (tools/dataprep/build_labels.py): t_sec / t_all / conflict — синтетические строки."""
    import json

    from ml.track.replay import load_labels, main as replay_main

    root = _a03_cache(tmp_path)
    (root / "labels.csv").write_text(
        "video_id,file,label,t_sec,md5,group,t_all,conflict,raw\n"
        "X-1,Х-1.mov,violation,16,0,X-1,16,0,synthetic\n"
        "X-2,X-2.mov,no_violation,,0,X-2,,0,synthetic\n"
        "X-9,X-9.mov,violation,57,0,X-9,57;68,1,synthetic\n", encoding="utf-8")
    lab = load_labels(str(root / "labels.csv"))
    assert lab["x-1"]["seconds"] == [16.0] and lab["x-9"]["seconds"] == [57.0, 68.0] and lab["x-9"]["conflict"]
    out = root / "out"
    assert replay_main(["--real", "--dets-glob", str(root / "cache" / "*" / "dets.jsonl"), "--videos-dir",
                        str(root / "videos"), "--labels", str(root / "labels.csv"), "--variant", "no_lost_penalty",
                        "--out", str(out)]) == 0
    s = json.loads((out / "replay.json").read_text())["summary"]["baseline"]
    assert s["videos"] == 2 and s["unlabeled_videos"] == 0 and s["violation_hit_within_2.0s"] == 1
    assert s["label_conflicts"] == 0
