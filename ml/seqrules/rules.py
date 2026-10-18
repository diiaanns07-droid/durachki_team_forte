"""Правила по последовательностям: А (телефон у руки/головы), Б (открытый сейф без людей в рабочей зоне),
В (вход в рабочую зону — наблюдение), «рука — деньги» (только наблюдение).

Каждое правило строит покадровые наблюдения (observed / absent / unknown) по реальным PTS, собирает эпизоды
с допуском коротких пропусков и выдаёт эпизоды для UI и события контракта. Пороги — из config.yaml
(технические параметры, не нормы). Решения не зависят от имени файла, порядка или длительности ролика.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .checks import Checker
from .observe import (WRISTS, Frames, ObjIndex, PersonTrack, PoseIndex, box_height_px, box_list, center, iso_dist,
                      joints, lower_half_in_zone, points_in_polygon, r4, zone_polygon)

try:
    from contracts.sentinel_contracts import Event
except ImportError:  # pragma: no cover
    from backend.pipeline.contract import Event  # type: ignore

ROLE_RU = {"cashier": "кассир", "collector": "инкассатор", "visitor": "посетитель", "unknown": "роль не определена"}
SOURCE = "rules_seq"            # fusion: canonical_source("rules_seq") == "rules"
NOTE = "пороги — технические параметры распознавания, не требования норм"


@dataclass
class Ctx:
    cfg: dict
    fr: Frames
    persons: dict[int, PersonTrack]
    poses: PoseIndex
    zones: list
    work_zone: Optional[np.ndarray]
    checker: Optional[Checker]
    dets: list

    @property
    def tol(self) -> float:
        return float(self.cfg["person"].get("box_time_tolerance_s", 0.12))

    def in_zone(self, box: Optional[np.ndarray]) -> Optional[bool]:
        if box is None or self.work_zone is None:
            return None
        return lower_half_in_zone(box, self.work_zone) >= float(self.cfg["person"].get("zone_min_overlap", 0.25))

    def staff(self, p: PersonTrack) -> bool:
        return p.role in set(self.cfg["person"].get("staff_roles", ["cashier"]))


def build_episodes(obs: Sequence[dict], max_gap_s: float, breakers: Iterable[str] = ()) -> list[list[dict]]:
    """Эпизод — от первого до последнего observed. Внутри допускаются прочие состояния, пока между соседними
    observed не больше max_gap_s. Состояние из breakers закрывает эпизод сразу. Хвост после последнего
    observed в эпизод не входит."""
    br = set(breakers)
    out: list[list[dict]] = []
    cur: list[dict] = []
    pending: list[dict] = []
    last_t: Optional[float] = None
    for o in sorted(obs, key=lambda x: x["t"]):
        st = o["state"]
        if st == "observed":
            if cur and last_t is not None and o["t"] - last_t <= max_gap_s + 1e-6:
                cur.extend(pending)
                cur.append(o)
            else:
                if cur:
                    out.append(cur)
                cur = [o]
            pending, last_t = [], o["t"]
        elif st in br:
            if cur:
                out.append(cur)
            cur, pending, last_t = [], [], None
        elif cur:
            pending.append(o)
    if cur:
        out.append(cur)
    return out


def _counts(ep: Sequence[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for o in ep:
        out[o["state"]] = out.get(o["state"], 0) + 1
    return out


def _kp_list(kp: Optional[np.ndarray]) -> Optional[list]:
    if kp is None:
        return None
    return [[round(float(x), 4), round(float(y), 4), round(float(s), 3)] for x, y, s in kp[:17]]


def _role(p: PersonTrack) -> str:
    return ROLE_RU.get(p.role, p.role)


def _event(type_id: str, t0: float, t1: float, anchor: float, conf: float, facts: list[str], bboxes: dict,
           tids: list[int], extra: dict) -> Event:
    extra = {"engine": "seqrules", "anchor_policy": "onset", "t_anchor": r4(anchor), "thresholds_note": NOTE, **extra}
    return Event(type_id=type_id, t_start=r4(t0), t_end=r4(max(t1, t0)), t_peak=r4(anchor),
                 confidence=round(float(min(max(conf, 0.0), 1.0)), 4), source=SOURCE, facts=facts,
                 bboxes={k: tuple(v) for k, v in bboxes.items() if v is not None}, track_ids=tids, extra=extra)


# =============================================================== А: телефон у руки/головы

def rule_phone(c: Ctx) -> tuple[list[dict], list[Event], dict]:
    pc = c.cfg["phone"]
    phones = ObjIndex(c.dets, pc.get("labels", ["cell_phone"]), float(pc.get("min_score", 0.3)))
    info: dict[str, Any] = {"status": "ok", "object_dets": phones.n, "object_sources": phones.sources}
    if not c.persons:
        info["status"] = "no_persons"
        return [], [], info
    if c.poses.n == 0:
        info.update(status="unavailable", reason="нет поз: запястья и голова не наблюдаемы")
        return [], [], info
    tol = c.tol
    obs_by: dict[int, list[dict]] = {tid: [] for tid in c.persons}
    for k, t in enumerate(c.fr.times):
        t = float(t)
        present = [(p, p.box_at(t, tol)) for p in c.persons.values() if p.spans(t)]
        if not present:
            continue
        processed = bool(c.fr.processed[k])
        cands = phones.at(t, tol) if processed else []
        J: dict[int, tuple[Any, Optional[np.ndarray]]] = {}
        for p, box in present:
            kp = c.poses.at(p.tid, t, tol) if box is not None else None
            J[p.tid] = (joints(kp, float(pc.get("wrist_min", 0.3)), float(pc.get("head_min", 0.3))), kp)
        pairs = []
        for oi, o in enumerate(cands):
            oc = center(o.box)
            for p, box in present:
                if box is None:
                    continue
                jn = J[p.tid][0]
                h = box_height_px(box, c.fr)
                best = None
                for name, xy, sc in jn.wrists:
                    d = iso_dist(oc, xy, c.fr) / h
                    if d <= float(pc["near_hand_rel"]) and (best is None or d < best[0]):
                        best = (d, name, xy, sc, False)
                if jn.head is not None:
                    d = iso_dist(oc, jn.head[0], c.fr) / h
                    if d <= float(pc["near_head_rel"]) and (best is None or d < best[0]):
                        best = (d, "head", jn.head[0], jn.head[1], True)
                if best is not None:
                    pairs.append((best[0], oi, p.tid, best))
        pairs.sort(key=lambda x: x[0])
        used_o: set[int] = set()
        assigned: dict[int, tuple[Any, tuple]] = {}
        for _d, oi, tid, best in pairs:          # телефон — одному человеку, ближайшему по суставу
            if oi in used_o or tid in assigned:
                continue
            used_o.add(oi)
            assigned[tid] = (cands[oi], best)
        for p, box in present:
            jn, kp = J[p.tid]
            if not processed:
                state, why = "unknown", "кадр не обработан детектором"
            elif box is None:
                state, why = "unknown", "человек не детектирован на кадре"
            elif p.tid in assigned:
                state, why = "observed", ""
            elif not jn.any:
                state, why = "unknown", "запястья и голова не видны"
            else:
                state, why = "absent", ""
            rec: dict[str, Any] = {"t": r4(t), "state": state}
            if why:
                rec["why"] = why
            if box is not None:
                rec["person"] = box_list(box)
                rec["in_zone"] = c.in_zone(box)
            if kp is not None:
                rec["kp"] = _kp_list(kp)
            if p.tid in assigned:
                o, (d, name, xy, sc, at_head) = assigned[p.tid]
                rec.update(object=box_list(o.box), object_label=o.label, object_score=round(o.score, 3),
                           object_source=o.source, joint=name, joint_xy=[round(float(xy[0]), 4), round(float(xy[1]), 4)],
                           joint_score=round(float(sc), 3), at_head=bool(at_head), d_rel=round(float(d), 3))
            obs_by[p.tid].append(rec)

    episodes: list[dict] = []
    events: list[Event] = []
    per_ep = int((c.cfg.get("checks", {}).get("phone_contrast", {}) or {}).get("per_episode", 2))
    for tid, obs in obs_by.items():
        p = c.persons[tid]
        for n, ep in enumerate(build_episodes(obs, float(pc["max_gap_s"])), 1):
            seen = [o for o in ep if o["state"] == "observed"]
            dur = seen[-1]["t"] - seen[0]["t"]
            hs = float(np.median([box_height_px(np.asarray(o["person"]), c.fr) for o in seen]))
            cs = np.array([center(np.asarray(o["object"])) for o in seen])
            med = np.median(cs, axis=0)
            dists = np.array([iso_dist(x, med, c.fr) for x in cs]) / max(hs, 1e-6)
            move = float(np.percentile(dists, 80)) * 2.0 if len(dists) > 1 else 0.0
            head_frac = float(np.mean([o.get("at_head", False) for o in seen]))
            zf = [o.get("in_zone") for o in seen if o.get("in_zone") is not None]
            zone_frac = float(np.mean(zf)) if zf else None
            at_work = c.staff(p) or (zone_frac is not None and zone_frac >= float(pc.get("staff_zone_fraction", 0.5)))
            epi = {"kind": "episode", "id": f"A-{tid}-{n}", "rule": "phone", "type_id": pc["type_id"],
                   "track_id": tid, "role": p.role, "t_start": seen[0]["t"], "t_end": seen[-1]["t"],
                   "t_anchor": seen[0]["t"], "duration_s": round(dur, 3), "counts": _counts(ep),
                   "move_rel": round(move, 3), "head_frac": round(head_frac, 3),
                   "zone_frac": None if zone_frac is None else round(zone_frac, 3), "frames": ep}
            if dur < float(pc["min_duration_s"]) or len(seen) < int(pc["min_observed_frames"]):
                epi.update(status="too_short", reason=f"наблюдение {dur:.1f} с / {len(seen)} кадров — меньше порога")
            elif move < float(pc["carry_move_rel"]) and head_frac < float(pc["head_fraction"]):
                epi.update(status="static_object", reason="предмет не перемещается и не у головы — вероятно, лежит рядом с рукой")
            elif not at_work:
                epi.update(status="not_at_workplace", reason=f"человек #{tid} ({_role(p)}) не на рабочем месте кассира")
            else:
                epi["status"] = "candidate"
            if epi["status"] == "candidate" and c.checker is not None:
                order = sorted(seen, key=lambda o: (-o.get("object_score", 0.0), o["t"]))[:per_ep]
                epi["contrast"] = [c.checker.phone_contrast(o["t"], o["object"]) for o in order]
                votes = [x["state"] for x in epi["contrast"]]
                if votes.count("not_mobile") > votes.count("mobile"):
                    epi.update(status="rejected_not_mobile",
                               reason="контрастная проверка: похоже на другой предмет стола, не на мобильный телефон")
            episodes.append(epi)
            if epi["status"] != "candidate":
                continue
            cc = pc.get("confidence", {})
            conf = min(float(cc.get("max", 0.8)), float(cc.get("base", 0.55)) + float(cc.get("per_s", 0.03)) * dur)
            votes = [x["state"] for x in epi.get("contrast", [])]
            mobile = votes.count("mobile") > 0 and votes.count("mobile") >= votes.count("not_mobile")
            if mobile:
                conf = min(1.0, conf + float(cc.get("mobile_bonus", 0.05)))
            epi["status"] = "event"
            src = seen[0].get("object_source", "")
            facts = [f"У {'головы' if head_frac >= 0.5 else 'руки'} человека #{tid} ({_role(p)}) виден предмет, "
                     f"распознанный детектором как мобильный телефон: {dur:.1f} с, {len(seen)} кадров наблюдения"
                     + (f" (модель: {src.split(':')[0]})" if src else "") + "."]
            if head_frac > 0:
                facts.append(f"Предмет у головы на {head_frac:.0%} кадров наблюдения.")
            if votes:
                facts.append("Контрастная проверка «мобильный / стационарный»: "
                             + ("мобильный." if mobile else "не дала однозначного ответа."))
            first = seen[0]
            events.append(_event(pc["type_id"], seen[0]["t"], seen[-1]["t"], seen[0]["t"], conf, facts,
                                 {"person": first.get("person"), "phone": first.get("object")}, [tid],
                                 {"rule": "A_phone", "episode_id": epi["id"], "assessment": "candidate_violation",
                                  "n_observed": len(seen), "counts": epi["counts"], "duration_s": round(dur, 3),
                                  "move_rel": epi["move_rel"], "head_frac": epi["head_frac"], "zone_frac": epi["zone_frac"],
                                  "contrast": epi.get("contrast", []), "catalog_type": pc["type_id"],
                                  "last_box": seen[-1].get("object")}))
    info["episodes"] = len(episodes)
    info["by_status"] = _status_counts(episodes)
    return episodes, events, info


def _status_counts(eps: Sequence[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in eps:
        out[e["status"]] = out.get(e["status"], 0) + 1
    return out


# =============================================================== В: вход в рабочую зону (наблюдение)

def _zone_obs(c: Ctx, p: PersonTrack) -> list[dict]:
    obs = []
    for k, t in enumerate(c.fr.times):
        t = float(t)
        if not p.spans(t):
            continue
        box = p.box_at(c.fr.times[k], c.tol) if c.fr.processed[k] else None
        if box is None:
            obs.append({"t": r4(t), "state": "unknown"})
            continue
        inside = c.in_zone(box)
        obs.append({"t": r4(t), "state": "observed" if inside else "absent", "person": box_list(box)})
    return obs


def rule_entry(c: Ctx) -> tuple[list[dict], list[Event], dict]:
    ec = c.cfg["entry"]
    info: dict[str, Any] = {"status": "ok"}
    if c.work_zone is None:
        info.update(status="unavailable", reason="рабочая зона кассира не построена")
        return [], [], info
    episodes, events = [], []
    for tid, p in c.persons.items():
        if c.staff(p):
            continue
        obs = _zone_obs(c, p)
        for n, ep in enumerate(build_episodes(obs, float(ec["max_gap_s"])), 1):
            seen = [o for o in ep if o["state"] == "observed"]
            dwell = seen[-1]["t"] - seen[0]["t"]
            before = [o for o in obs if o["t"] < seen[0]["t"] and o["state"] == "absent"]
            epi = {"kind": "episode", "id": f"C-{tid}-{n}", "rule": "entry", "type_id": ec["type_id"], "track_id": tid,
                   "role": p.role, "t_start": seen[0]["t"], "t_end": seen[-1]["t"], "t_anchor": seen[0]["t"],
                   "duration_s": round(dwell, 3), "counts": _counts(ep), "frames": ep}
            if not before:
                epi.update(status="no_entry_observed", reason="человек появился уже внутри зоны — вход не наблюдался")
            elif dwell < float(ec["dwell_s"]):
                epi.update(status="too_short", reason=f"в зоне {dwell:.1f} с — меньше порога")
            else:
                epi["status"] = "event"
            episodes.append(epi)
            if epi["status"] != "event":
                continue
            facts = [f"Человек #{tid} (роль по трекингу: {_role(p)}) вошёл в рабочую зону кассира в {seen[0]['t']:.1f} с "
                     f"и оставался там {dwell:.1f} с.",
                     "Это наблюдение для проверяющего, а не вывод о нарушении: полномочия человека камера не видит."]
            events.append(_event(ec["type_id"], seen[0]["t"], seen[-1]["t"], seen[0]["t"], float(ec["confidence"]), facts,
                                 {"person": seen[0].get("person")}, [tid],
                                 {"rule": "C_entry", "episode_id": epi["id"], "assessment": "review_observation",
                                  "assessment_reasons": ["наблюдение: вход в рабочую зону, без вывода о нарушении"],
                                  "duration_s": round(dwell, 3), "counts": epi["counts"], "catalog_type": ec["type_id"]}))
    info["episodes"] = len(episodes)
    info["by_status"] = _status_counts(episodes)
    return episodes, events, info


# =============================================================== Б: открытый сейф, в рабочей зоне никого

def rule_safe(c: Ctx) -> tuple[list[dict], list[Event], dict]:
    sc = c.cfg["safe"]
    info: dict[str, Any] = {"status": "ok"}
    if c.work_zone is None:
        info.update(status="unavailable", reason="рабочая зона кассира не построена")
        return [], [], info
    occ: list[dict] = []
    for k, t in enumerate(c.fr.times):
        t = float(t)
        if not c.fr.processed[k]:
            occ.append({"t": r4(t), "state": "unknown"})
            continue
        inside = []
        for p in c.persons.values():
            b = p.box_at(t, c.tol)
            if b is not None and c.in_zone(b):
                inside.append(p.tid)
        occ.append({"t": r4(t), "state": "occupied" if inside else "observed", "occupants": inside})
    episodes, events = [], []
    runs = build_episodes(occ, float(sc["max_gap_s"]), breakers=("occupied",))
    info["absence_runs"] = len(runs)
    checked, unchecked = [], []
    for n, run in enumerate(runs, 1):
        seen = [o for o in run if o["state"] == "observed"]
        t0, t1 = seen[0]["t"], seen[-1]["t"]
        dur = t1 - t0
        if dur < float(sc["absent_min_s"]):
            continue
        prev = [o for o in occ if o["t"] < t0 and o["state"] == "occupied"]
        epi: dict[str, Any] = {"kind": "episode", "id": f"B-{n}", "rule": "safe", "type_id": sc["type_id"],
                               "track_id": None, "t_start": t0, "t_end": t1, "t_anchor": t0,
                               "duration_s": round(dur, 3), "counts": _counts(run)}
        if not prev:
            epi.update(status="uncertain", reason="зона пуста с начала записи — уход сотрудника не наблюдался",
                       absence="empty_from_start")
        else:
            last = prev[-1]["occupants"]
            how = None
            for tid in last:
                p = c.persons[tid]
                for tk in c.fr.times:
                    tk = float(tk)
                    if prev[-1]["t"] < tk <= t0 + float(sc["exit_window_s"]):
                        b = p.box_at(tk, c.tol)
                        if b is not None and c.in_zone(b) is False:
                            how = ("exit_observed", tid, tk, b)
                            break
                if how:
                    break
            if how is None:
                epi.update(status="uncertain", reason="сотрудник пропал из детекций внутри зоны — потеря ≠ уход",
                           absence="lost_in_zone", last_occupants=last)
            else:
                epi.update(absence=how[0], track_id=how[1], exit_t=r4(how[2]), exit_box=box_list(how[3]))
                epi["track_role"] = c.persons[how[1]].role
                # вышел из зоны, но остаётся в кадре рядом (например, у сейфа) — это не «без присмотра»
                p = c.persons[how[1]]
                proc = [float(t) for k, t in enumerate(c.fr.times) if c.fr.processed[k] and t0 <= float(t) <= t1]
                vis = sum(1 for t in proc if p.box_at(t, c.tol) is not None)
                epi["seen_after_exit_frac"] = round(vis / max(1, len(proc)), 3)
                if epi["seen_after_exit_frac"] > float(sc.get("max_seen_after_exit_frac", 0.3)):
                    epi.update(status="uncertain", absence="nearby",
                               reason=f"человек #{how[1]} вне зоны, но виден в кадре {epi['seen_after_exit_frac']:.0%} "
                                      "времени отсутствия — возможно, рядом с сейфом")
        if epi.get("status") == "uncertain":
            episodes.append(epi)
            continue
        # сейф: проверки внутри подтверждённого отсутствия
        times = []
        tk = t0 + float(sc["first_check_after_s"])
        per = int((c.cfg.get("checks", {}).get("safe_state", {}) or {}).get("per_interval", 4))
        while tk <= t1 + 1e-6 and len(times) < per:
            times.append(tk)
            tk += float(sc["check_every_s"])
        res = [c.checker.safe_state(x, c.zones) if c.checker is not None else
               {"t": r4(x), "state": "skipped", "reason": "нет доступа к кадрам"} for x in times]
        epi["safe_checks"] = res
        states = [r["state"] for r in res]
        # самая длинная серия «open» без «closed» между ними (unknown/skipped серию не рвут и не продлевают)
        run_open, run_start, best, best_start = 0, 0, 0, None
        for i, r in enumerate(res):
            if r["state"] == "open":
                if run_open == 0:
                    run_start = i
                run_open += 1
                if run_open > best:
                    best, best_start = run_open, run_start
            elif r["state"] == "closed":
                run_open = 0
        did = [r for r in res if r["state"] in ("open", "closed", "unknown")]
        (checked if did else unchecked).append([t0, t1])
        if best >= int(sc["min_open_checks"]):
            epi["status"] = "event"
        elif "closed" in states:
            epi.update(status="safe_closed", reason="дверца сейфа распознана закрытой")
        elif did:
            epi.update(status="safe_unknown", reason="состояние сейфа не определено (модель не дала свидетельства)")
        else:
            epi.update(status="not_checked", reason="проверки сейфа не выполнены (лимит/бюджет/нет кадров) — не «проверено»")
        epi["frames"] = run
        episodes.append(epi)
        if epi["status"] != "event":
            continue
        onset = t0 if best_start == 0 else float(res[best_start]["t"])
        closed_after = [r["t"] for r in res[best_start:] if r["state"] == "closed"]
        t_stop = float(closed_after[0]) if closed_after else t1
        cc = sc.get("confidence", {})
        conf = min(float(cc.get("max", 0.85)), float(cc.get("base", 0.6)) + float(cc.get("per_check", 0.05)) * (best - 1))
        opened = [r for r in res if r["state"] == "open"]
        facts = [f"Дверца сейфа распознана открытой на {len(opened)} проверках кадров (OWLv2, контраст «открыт / закрыт»).",
                 f"В рабочей зоне кассира людей не видно {dur:.1f} с; перед этим человек #{epi['track_id']} "
                 f"({ROLE_RU.get(epi.get('track_role', ''), 'роль не определена')}) виден выходящим из зоны."]
        events.append(_event(sc["type_id"], onset, t_stop, onset, conf, facts,
                             {"safe": opened[0].get("bbox"), "person": epi.get("exit_box")},
                             [epi["track_id"]] if epi.get("track_id") is not None else [],
                             {"rule": "B_safe", "episode_id": epi["id"], "assessment": "candidate_violation",
                              "safe_checks": res, "absence": epi.get("absence"), "duration_s": round(dur, 3),
                              "catalog_type": sc["type_id"]}))
    info.update(episodes=len(episodes), by_status=_status_counts(episodes), checked_intervals=checked,
                unchecked_intervals=unchecked)
    return episodes, events, info


# =============================================================== Б2: место хранения, найденное по рукам (два прохода)

def _in_any(pt: np.ndarray, polys: list) -> bool:
    return any(bool(points_in_polygon(np.array([pt[0]]), np.array([pt[1]]), z)[0]) for z in polys)


def _staff_reaches(c: Ctx, sc: dict) -> list[dict]:
    """Проход 1: эпизоды «рука вытянута за пределы стола и зоны клиента» у сотрудников (суставы, PTS)."""
    excl = [z for z in (zone_polygon(c.zones, n) for n in sc.get("exclude_zones", ["desk", "client_area"])) if z is not None]
    wmin, smin, ext = float(sc.get("wrist_min", 0.3)), float(sc.get("shoulder_min", 0.3)), float(sc["reach_ext"])
    out = []
    for tid, p in c.persons.items():
        if not c.staff(p):
            continue
        rows = []
        for t in c.fr.times:
            t = float(t)
            if not p.spans(t):
                continue
            box = p.box_at(t, c.tol)
            kp = c.poses.at(tid, t, c.tol) if box is not None else None
            if kp is None or kp[5, 2] < smin or kp[6, 2] < smin:
                rows.append((t, box, None, None, None, []))
                continue
            S = (kp[5, :2] + kp[6, :2]) / 2.0
            sw = iso_dist(kp[5, :2], kp[6, :2], c.fr)
            if sw < 4.0:
                rows.append((t, box, None, None, None, []))
                continue
            ws = [(iso_dist(kp[i, :2], S, c.fr) / sw, name, kp[i, :2].copy()) for name, i in WRISTS.items() if kp[i, 2] >= wmin]
            rows.append((t, box, kp, S, sw, ws))
        # порог — от обычного положения рук этого человека: стоящий с опущенными руками не «тянется» всё время
        es = [max(w[0] for w in r[5]) for r in rows if r[5]]
        thr = max(ext, (float(np.median(es)) if es else 0.0) + float(sc.get("reach_delta", 0.6)))
        obs = []
        for t, box, kp, S, sw, ws in rows:
            if kp is None:
                obs.append({"t": r4(t), "state": "unknown"})
                continue
            cand = [w for w in ws if w[0] >= thr and not _in_any(w[2], excl)]
            if not cand:
                obs.append({"t": r4(t), "state": "absent" if ws else "unknown"})
                continue
            e, name, xy = max(cand, key=lambda w: w[0])
            best = (e, name, xy)
            obs.append({"t": r4(t), "state": "observed", "person": box_list(box), "kp": _kp_list(kp), "joint": best[1],
                        "joint_xy": [round(float(best[2][0]), 4), round(float(best[2][1]), 4)], "ext": round(best[0], 3),
                        "sw": round(sw, 2)})
        for ep in build_episodes(obs, float(sc.get("max_gap_s", 0.6))):
            seen = [o for o in ep if o["state"] == "observed"]
            dur = seen[-1]["t"] - seen[0]["t"]
            if len(seen) < 2 or dur < float(sc.get("min_reach_s", 0.4)) or dur > float(sc.get("max_reach_s", 8.0)):
                continue
            top = sorted(seen, key=lambda o: -o["ext"])[:3]
            out.append({"tid": tid, "t0": seen[0]["t"], "t1": seen[-1]["t"], "ext": max(o["ext"] for o in seen),
                        "target": np.mean([o["joint_xy"] for o in top], axis=0),
                        "sw": float(np.median([o["sw"] for o in seen])), "frames": ep})
    return sorted(out, key=lambda r: r["t0"])


def _sites(c: Ctx, reaches: list[dict], sc: dict) -> list[dict]:
    rad = float(sc.get("site_radius_rel", 0.06)) * c.fr.height
    sites: list[dict] = []
    for r in reaches:
        hit = None
        for s in sites:
            if iso_dist(s["center"], r["target"], c.fr) <= rad:
                hit = s
                break
        if hit is None:
            sites.append({"center": np.asarray(r["target"], np.float32), "visits": [r]})
        else:
            n = len(hit["visits"])
            hit["center"] = (hit["center"] * n + np.asarray(r["target"], np.float32)) / (n + 1)
            hit["visits"].append(r)
    for i, s in enumerate(sites, 1):
        side_px = float(sc.get("crop_scale", 3.0)) * float(np.median([v["sw"] for v in s["visits"]]))
        hw, hh = side_px / 2.0 / c.fr.width, side_px / 2.0 / c.fr.height
        cx, cy = float(s["center"][0]), float(s["center"][1])
        s["box"] = [round(max(0.0, cx - hw), 5), round(max(0.0, cy - hh), 5),
                    round(min(1.0, cx + hw), 5), round(min(1.0, cy + hh), 5)]
        s["id"] = i
    return sorted(sites, key=lambda s: (-len(s["visits"]), s["visits"][0]["t0"]))


def _hands_in(c: Ctx, box: list, t: float) -> Optional[bool]:
    """Есть ли чьё-то запястье в месте в момент t (None — суставов в этот момент нет ни у кого)."""
    any_pose = False
    for tid in c.persons:
        kp = c.poses.at(tid, t, c.tol)
        if kp is None:
            continue
        any_pose = True
        for i in WRISTS.values():
            if kp[i, 2] >= 0.3 and box[0] <= kp[i, 0] <= box[2] and box[1] <= kp[i, 1] <= box[3]:
                return True
    return False if any_pose else None


def _site_checks(c: Ctx, sc: dict, box: list, v: dict, t_next: float, t_end: float, app0) -> list[dict]:
    checks = []
    for dt in sc.get("checks_after_s", [2, 4, 7, 10]):
        tk = v["t1"] + float(dt)
        if tk > t_end or tk >= t_next - 0.5:
            break
        if _hands_in(c, box, tk):
            checks.append({"t": r4(tk), "state": "occluded", "reason": "рука снова у места", "bbox": box})
            continue
        if c.checker is not None and hasattr(c.checker, "site_state"):
            r = c.checker.site_state(tk, box)
        else:
            r = {"t": r4(tk), "state": "skipped", "reason": "нет доступа к кадрам"}
        if app0 is not None and r.get("state") not in ("skipped", "unavailable") and hasattr(c.checker, "site_appearance"):
            a = c.checker.site_appearance(tk, box)
            if a is not None:
                r["appearance_diff"] = round(float(1.0 - np.clip(np.dot(app0, a) / len(a), -1.0, 1.0)), 3)
        r.setdefault("bbox", box)
        checks.append(r)
    return checks


def rule_storage(c: Ctx) -> tuple[list[dict], list[Event], dict]:
    sc = c.cfg["storage"]
    info: dict[str, Any] = {"status": "ok", "note": "место найдено по движению рук; что это (сейф, ящик) — не утверждается"}
    if c.poses.n == 0:
        info.update(status="unavailable", reason="нет поз: движения рук не наблюдаемы")
        return [], [], info
    reaches = _staff_reaches(c, sc)
    sites = _sites(c, reaches, sc)[: int(sc.get("max_sites", 3))]
    info.update(reaches=len(reaches), sites=len(sites))
    episodes, events = [], []
    t_end = float(c.fr.times[-1]) if len(c.fr.times) else 0.0
    for s in sites:
        for v in s["visits"][-int(sc.get("visits_per_site", 2)):]:
            nxt = [w["t0"] for w in s["visits"] if w["t0"] > v["t1"]]
            t_next = min(nxt) if nxt else float("inf")
            p = c.persons[v["tid"]]
            for f in v["frames"]:
                if f["state"] == "observed":
                    f.update(object=s["box"], object_label="место хранения")
            epi: dict[str, Any] = {"kind": "episode", "id": f"S-{s['id']}-{len(episodes) + 1}", "rule": "storage",
                                   "type_id": sc["type_id"], "track_id": v["tid"], "role": p.role, "t_start": v["t0"],
                                   "t_end": v["t1"], "t_anchor": v["t1"], "duration_s": round(v["t1"] - v["t0"], 3),
                                   "counts": _counts(v["frames"]), "site_box": s["box"], "site_visits": len(s["visits"]),
                                   "reach_ext": round(v["ext"], 2), "frames": v["frames"]}
            # проход 2: вид места до касания и состояние после ухода руки
            tb = v["t0"] - float(sc.get("baseline_before_s", 1.0))
            app0 = None
            if tb >= 0 and _hands_in(c, s["box"], tb) is False and c.checker is not None \
                    and hasattr(c.checker, "site_appearance"):
                app0 = c.checker.site_appearance(tb, s["box"])
            checks = _site_checks(c, sc, s["box"], v, t_next, t_end, app0)
            epi["safe_checks"] = checks
            epi["appearance_baseline"] = app0 is not None
            states = [x["state"] for x in checks]
            run_open, best = 0, 0
            for st in states:
                if st == "open":
                    run_open += 1
                    best = max(best, run_open)
                elif st == "closed":
                    run_open = 0
            diffs = [x["appearance_diff"] for x in checks if "appearance_diff" in x]
            reverted = bool(diffs) and float(np.median(diffs)) < float(sc.get("revert_dist", 0.25))
            need = int(sc.get("min_open_checks", 2)) + (0 if app0 is not None else 1)
            if not checks:
                epi.update(status="not_checked", reason="после касания не было момента без рук у места до конца записи")
            elif best >= need and not reverted:
                epi["status"] = "event"
            elif reverted:
                epi.update(status="reverted", reason="вид места вернулся к исходному — вероятно, закрыто")
            elif "closed" in states:
                epi.update(status="closed", reason="место распознано закрытым")
            elif any(x in ("open", "unsure", "occluded", "unknown") for x in states):
                epi.update(status="uncertain", reason="открытость места не подтверждена проверками подряд")
            else:
                epi.update(status="not_checked", reason="проверки места не выполнены (лимит, бюджет, нет кадров)")
            episodes.append(epi)
            if epi["status"] != "event":
                continue
            opened = [x for x in checks if x["state"] == "open"]
            cc = sc.get("confidence", {})
            conf = min(float(cc.get("max", 0.8)), float(cc.get("base", 0.55)) + float(cc.get("per_check", 0.05)) * (best - need))
            person = next((f.get("person") for f in v["frames"] if f.get("person")), None)
            facts = [f"Сотрудник #{v['tid']} ({_role(p)}) тянулся к месту вне стола и зоны клиента "
                     f"({v['t0']:.1f}–{v['t1']:.1f} с; к этому месту — {len(s['visits'])} раз).",
                     f"После ухода руки место распознано открытым на {len(opened)} проверках подряд (OWLv2 по увеличенной "
                     "вырезке), рук у места нет" + (", вид места не вернулся к исходному." if diffs else "."),
                     "Место найдено по движению рук; что это (сейф, ящик, шкаф), система не утверждает."]
            events.append(_event(sc["type_id"], v["t1"], opened[-1]["t"], v["t1"], conf, facts,
                                 {"storage": tuple(s["box"]), "person": person}, [v["tid"]],
                                 {"rule": "B2_storage", "episode_id": epi["id"], "assessment": "candidate_violation",
                                  "site_visits": len(s["visits"]), "checks": checks, "reach_t0": v["t0"],
                                  "catalog_type": sc["type_id"]}))
    info["episodes"] = len(episodes)
    info["by_status"] = _status_counts(episodes)
    return episodes, events, info


# =============================================================== рука — деньги (только наблюдение)

def obs_hand_cash(c: Ctx) -> tuple[list[dict], list[Event], dict]:
    hc = c.cfg["hand_cash"]
    cash = ObjIndex(c.dets, hc.get("labels", ["cash"]), float(hc.get("min_score", 0.3)))
    info: dict[str, Any] = {"status": "ok", "object_dets": cash.n, "object_sources": cash.sources,
                            "note": "сближение руки и денег — кандидат на контакт, не вывод о хищении"}
    if cash.n == 0 or c.poses.n == 0:
        info.update(status="unavailable", reason="нет детекций денег или поз")
        return [], [], info
    episodes = []
    for tid, p in c.persons.items():
        obs = []
        for k, t in enumerate(c.fr.times):
            t = float(t)
            if not p.spans(t):
                continue
            box = p.box_at(t, c.tol)
            kp = c.poses.at(tid, t, c.tol) if box is not None else None
            jn = joints(kp, 0.3, 1.1)
            if not c.fr.processed[k] or box is None or not jn.wrists:
                obs.append({"t": r4(t), "state": "unknown"})
                continue
            h = box_height_px(box, c.fr)
            near = [(min(iso_dist(center(o.box), xy, c.fr) for _n, xy, _s in jn.wrists) / h, o) for o in cash.at(t, c.tol)]
            near = [x for x in near if x[0] <= float(hc["near_rel"])]
            if near:
                d, o = min(near, key=lambda x: x[0])
                obs.append({"t": r4(t), "state": "observed", "person": box_list(box), "object": box_list(o.box),
                            "object_score": round(o.score, 3), "d_rel": round(float(d), 3), "kp": _kp_list(kp)})
            else:
                obs.append({"t": r4(t), "state": "absent"})
        for n, ep in enumerate(build_episodes(obs, float(hc["max_gap_s"])), 1):
            seen = [o for o in ep if o["state"] == "observed"]
            dur = seen[-1]["t"] - seen[0]["t"]
            if dur < float(hc["min_duration_s"]):
                continue
            episodes.append({"kind": "episode", "id": f"H-{tid}-{n}", "rule": "hand_cash", "type_id": None,
                             "track_id": tid, "role": p.role, "status": "contact_candidate", "t_start": seen[0]["t"],
                             "t_end": seen[-1]["t"], "t_anchor": seen[0]["t"], "duration_s": round(dur, 3),
                             "counts": _counts(ep), "frames": ep})
    info["episodes"] = len(episodes)
    return episodes, [], info


# =============================================================== слияние повторов

def merge_events(events: list[Event], gap_s: float) -> list[Event]:
    """Повторы одного правила одного человека (или того же места телефона при смене ID) ближе gap_s — одно событие."""
    out: list[Event] = []
    for e in sorted(events, key=lambda x: (x.extra.get("rule", ""), x.t_start)):
        prev = out[-1] if out else None
        if prev is not None and prev.extra.get("rule") == e.extra.get("rule") and e.t_start - prev.t_end <= gap_s \
                and (set(prev.track_ids) & set(e.track_ids) or _same_place(prev, e)):
            prev.t_end = max(prev.t_end, e.t_end)
            prev.confidence = max(prev.confidence, e.confidence)
            prev.track_ids = sorted(set(prev.track_ids) | set(e.track_ids))
            prev.extra.setdefault("merged_episodes", [prev.extra.get("episode_id")]).append(e.extra.get("episode_id"))
            prev.extra["duration_s"] = round(prev.t_end - prev.t_start, 3)
            continue
        out.append(e)
    return sorted(out, key=lambda x: x.t_start)


def _same_place(a: Event, b: Event) -> bool:
    if a.extra.get("rule") != "A_phone":
        return False
    pa, pb = a.extra.get("last_box") or a.bboxes.get("phone"), b.bboxes.get("phone")
    if pa is None or pb is None:
        return False
    ca, cb = center(np.asarray(pa)), center(np.asarray(pb))
    return float(np.hypot(*(ca - cb))) <= 0.05
