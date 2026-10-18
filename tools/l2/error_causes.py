"""Причины ошибок профиля по сохранённым трассировкам (без нового GPU-прогона). Диагностика на знакомых видео.

Для каждой SHA-группы: эталон → вердикт контроля → вердикт профиля → категория → причина. Причина назначается
только по доказательствам из трассировок, в порядке проверки:
  frames      — эталонный момент вне обработанных кадров;
  perception  — в окне нет людей или суставов кассира;
  fusion      — событие правила было, но в ответ не попало;
  skipped     — проверка пропущена (лимит, бюджет, ошибка правила);
  rule_partial— реализованное правило видело эпизод в окне, но условия не выполнены (статус эпизода);
  no_rule     — в окне ни одно реализованное правило ничего не наблюдало (тип не подтверждён человеком).
FP: какое правило дало нарушение и что показала контрастная проверка.

Таблица с ID видео — локально (CSV); в stdout — только агрегаты.

    python tools/l2/error_causes.py --control runs/l2/compare/exp4_metrics.json --profile runs/l2/pose_rules/metrics.json \\
        --preds runs/l2/pose_rules/preds --jobs data/jobs --cache cache --out runs/l2/analysis/causes.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

WIN = 5.0          # окно вокруг эталонного момента для «наблюдало ли правило что-то», с
NEAR = 2.0         # окно для кадров, людей и суставов, с
OBJ = ("cell_phone", "cash", "cash_bag", "bag", "money_counter", "cup", "bottle", "food", "document", "keys", "door")


def load_pv(path: str) -> dict[str, dict]:
    return {e["video_id"]: e for e in json.loads(Path(path).read_text(encoding="utf-8"))["per_video"]}


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.open(encoding="utf-8")] if p.exists() else []


def evidence(sha_dir: Path, obs_items: list[dict], times: list[float], pred: dict) -> dict:
    dets = read_jsonl(sha_dir / "dets.jsonl")
    tracks = json.loads((sha_dir / "tracks.json").read_text(encoding="utf-8")) if (sha_dir / "tracks.json").exists() else []
    zones = [z["name"] for z in json.loads((sha_dir / "zones.json").read_text(encoding="utf-8"))] if (sha_dir / "zones.json").exists() else []
    poses = read_jsonl(sha_dir / "poses.jsonl")
    det_t = np.array(sorted({round(float(d["t"]), 3) for d in dets}))
    cashier = [t["track_id"] for t in tracks if t["label"] == "person" and t["role"] == "cashier"]
    ev: dict = {"zones": zones}
    near_frames, persons, wrist_ok, wrist_n, objs, starts_ends = 0, set(), 0, 0, Counter(), 0
    for t in times:
        if len(det_t):
            near_frames += int(((det_t >= t - NEAR) & (det_t <= t + NEAR)).sum())
        for d in dets:
            if abs(float(d["t"]) - t) <= NEAR and d["label"] == "person":
                persons.add(round(float(d["t"]), 3))
            if abs(float(d["t"]) - t) <= 3.0 and d["label"] in OBJ:
                objs[d["label"]] += 1
        for p in poses:
            if p.get("track_id") in cashier and abs(float(p["t"]) - t) <= NEAR:
                wrist_n += 1
                k = p["keypoints"]
                wrist_ok += int(max(k[9][2], k[10][2]) >= 0.3)
        for tr in tracks:
            if tr["label"] != "person" or not tr["boxes"]:
                continue
            a, b = float(tr["boxes"][0][0]), float(tr["boxes"][-1][0])
            starts_ends += int(abs(a - t) <= NEAR) + int(abs(b - t) <= NEAR)
    ev.update(frames_near=near_frames, person_frames_near=len(persons), cashier_pose_frames=wrist_n,
              cashier_wrist_visible=round(wrist_ok / wrist_n, 2) if wrist_n else None, objects_near=dict(objs),
              track_starts_ends_near=starts_ends)
    meta = obs_items[0] if obs_items and obs_items[0].get("kind") == "meta" else {}
    eps = [e for e in obs_items[1:] if e.get("kind") == "episode" and e.get("rule") != "storage"]
    win_eps = [e for e in eps if any(e["t_end"] >= t - WIN and e["t_start"] <= t + WIN for t in times)]
    ev["episodes_in_window"] = dict(Counter(f"{e['rule']}:{e['status']}" for e in win_eps))
    ev["rule_status"] = {k: v.get("status") for k, v in (meta.get("rules") or {}).items() if k != "storage"}
    ev["checks_skipped"] = sum((meta.get("checks") or {}).get("skipped", {}).values()) if meta else None
    ev["seq_events_in_window"] = sum(1 for e in win_eps if e["status"] == "event")
    inputs = ((pred.get("meta") or {}).get("scores_raw") or {}).get("inputs", {}).get("events", [])
    ev["fusion_inputs"] = len(inputs)
    ev["hints_in_window"] = sum(1 for h in ((pred.get("meta") or {}).get("ui") or {}).get("review_hints", [])
                                if any(abs(float(h.get("timecode_sec", -99)) - t) <= WIN for t in times))
    ev["violations"] = len(pred.get("violations") or [])
    return ev


def cause_fn(ev: dict) -> str:
    if ev["frames_near"] == 0:
        return "frames"
    if ev["person_frames_near"] == 0 or (ev["cashier_wrist_visible"] is not None and ev["cashier_wrist_visible"] < 0.2):
        return "perception"
    if ev["seq_events_in_window"] > 0 and ev["violations"] == 0:
        return "fusion"
    if (ev["checks_skipped"] or 0) > 0 or any(s == "error" for s in ev["rule_status"].values()):
        return "skipped"
    if ev["episodes_in_window"]:
        return "rule_partial"
    return "no_rule"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--jobs", default="data/jobs")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="runs/l2/analysis/causes.csv")
    a = ap.parse_args()
    ctl, prof = load_pv(a.control), load_pv(a.profile)
    man = {r["video_id"]: r for r in csv.DictReader(open(a.manifest, encoding="utf-8"))}
    rows, fn_c, fp_c, fn_partial, ctl_on_fn = [], Counter(), Counter(), Counter(), Counter()
    for g, e in sorted(prof.items(), key=lambda kv: (len(kv[0]), kv[0])):
        if e.get("gt_label") in ("conflict", "unknown"):
            continue
        sha = man.get(g, {}).get("sha256", "")
        pred_p = Path(a.preds) / f"{g}.json"
        pred = json.loads(pred_p.read_text(encoding="utf-8")) if pred_p.exists() else {}
        obs_p = Path(a.jobs) / g / "observations.json"
        obs = json.loads(obs_p.read_text(encoding="utf-8")).get("items", []) if obs_p.exists() else []
        times = [float(t) for t in (e.get("gt_times") or [])]
        cat = e.get("category")
        row = {"group": g, "gt": e.get("gt_label"), "control": (ctl.get(g) or {}).get("verdict"),
               "control_cat": (ctl.get(g) or {}).get("category"), "profile": e.get("verdict"), "profile_cat": cat,
               "cause": "", "detail": ""}
        if cat == "FN" or (cat in ("early", "late") and False):
            ev = evidence(Path(a.cache) / sha, obs, times or [0.0], pred)
            c = cause_fn(ev)
            fn_c[c] += 1
            ctl_on_fn[f"{c}:control_{row['control_cat']}"] += 1
            if c == "rule_partial":
                for k in ev["episodes_in_window"]:
                    fn_partial[k] += 1
            row.update(cause=c, detail=json.dumps(ev, ensure_ascii=False))
        elif cat == "FP":
            vio = pred.get("violations") or []
            eps = [x for x in obs[1:] if x.get("kind") == "episode" and x.get("status") == "event" and x.get("rule") != "storage"]
            contrast = Counter(c.get("state") for x in eps for c in (x.get("contrast") or []))
            srcs = Counter((f.get("object_source") or "").split(":")[0] for x in eps for f in (x.get("frames") or [])
                           if f.get("state") == "observed")
            kinds = Counter(v.get("type_id") for v in vio)
            c = ("V05:contrast_mobile" if contrast.get("mobile") else
                 "V05:contrast_not_confirmed" if contrast else "V05:no_contrast_check") if kinds.get("V05") else \
                f"{'+'.join(sorted(kinds))}"
            fp_c[c] += 1
            row.update(cause=c, detail=json.dumps({"types": dict(kinds), "contrast": dict(contrast), "object_sources": dict(srcs)},
                                                  ensure_ascii=False))
        rows.append(row)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("FN causes:", dict(fn_c))
    print("FN cause × control category:", dict(ctl_on_fn))
    print("FN rule_partial episode statuses in window:", dict(fn_partial))
    print("FP causes:", dict(fp_c))
    print("local table:", out)


if __name__ == "__main__":
    main()
