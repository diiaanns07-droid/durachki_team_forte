"""Наши собственные съёмки: data/extra/labels.csv, замороженный hold-out камеры C3, отчёты по планировкам и свету.

Схема data/extra/labels.csv (одна строка на ролик)::

    video_id,file,label,t_all,type_id,camera_id,scene_id,layout,lighting[,orientation]
    E-001,C1/e001.mov,violation,12;40,V02;V03,C1,S01,L1,normal
    E-002,C3/e002.mov,no_violation,,,C3,S01,L1,off

* label — violation / no_violation (пусто/иное → unknown, исключается: unknown ≠ false);
* t_all / type_id — начала событий и их типы через «;» (типы — настоящая GT: сцены поставлены нами);
* scene_id — группа: все ролики одной сцены/дубля всегда в одном split (и разные камеры одной сцены);
* camera_id C3 («другой угол») ЗАМОРАЖИВАЕТСЯ до любой настройки (`freeze`) и не участвует в подборе порогов;
* layout / lighting / orientation — для отчётов по стратам (НЕ признаки модели).

CLI::

    python -m eval.extra check  --labels data/extra/labels.csv --videos data/extra
    python -m eval.extra freeze --labels data/extra/labels.csv --camera C3 [--strict-scenes]
    python -m eval.run --labels data/extra/labels_eval.csv --types data/extra/labels_types.csv \\
        --splits data/extra/splits.csv --subset train,calib --videos data/extra --out runs/extra/tune1
    python -m eval.run ... --subset holdout --final --kind holdout_frozen --out runs/extra/final   # один раз
    python -m eval.extra report --run runs/extra/tune1 --labels data/extra/labels.csv
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import re
from pathlib import Path
from typing import Any, Optional

from .common import read_json, write_json
from .labels import LABEL_CLEAN, LABEL_UNKNOWN, LABEL_VIOLATION, _read_csv_dicts

REQUIRED = ("video_id", "label", "camera_id", "scene_id")
STRATA = ("camera_id", "layout", "lighting", "orientation")


def read_extra(path: str | Path) -> tuple[list[dict], list[str]]:
    rows, problems = [], []
    for i, d in enumerate(_read_csv_dicts(path), 2):
        miss = [k for k in REQUIRED if not (d.get(k) or "").strip()]
        if miss:
            problems.append(f"строка {i}: нет {miss}")
        lab = (d.get("label") or "").strip().lower()
        d["label"] = (LABEL_VIOLATION if lab in ("violation", "1", "нарушение") else
                      LABEL_CLEAN if lab in ("no_violation", "0", "без нарушения") else LABEL_UNKNOWN)
        if d["label"] == LABEL_UNKNOWN:
            problems.append(f"строка {i}: {d.get('video_id')}: неизвестная метка {lab!r} — исключена")
        d["times"] = [float(x) for x in re.split(r"[;,|]", d.get("t_all") or d.get("t_sec") or "") if x.strip()]
        d["types"] = [x.strip().upper() for x in re.split(r"[;,|]", d.get("type_id") or "") if x.strip()]
        if d["types"] and d["times"] and len(d["types"]) != len(d["times"]):
            problems.append(f"строка {i}: {d.get('video_id')}: число типов ≠ числу таймкодов")
        if d["label"] == LABEL_VIOLATION and not d["times"]:
            problems.append(f"строка {i}: {d.get('video_id')}: нарушение без таймкода")
        rows.append(d)
    return rows, problems


def to_eval_labels(rows: list[dict], out_dir: str | Path, videos_dir: Optional[str | Path] = None) -> tuple[Path, Path]:
    """→ labels.csv (схема eval.labels, group = scene_id) и labels_types.csv для eval.metrics."""
    from .labels import LabelRow, hash_file, write_labels_csv

    out = Path(out_dir)
    lrows = []
    for d in rows:
        r = LabelRow(video_id=d["video_id"], label=d["label"], times=d["times"], file=d.get("file", ""),
                     group=d.get("scene_id") or d["video_id"], raw=d.get("t_all", ""))
        if videos_dir and d.get("file") and (Path(videos_dir) / d["file"]).exists():
            r.md5, r.sha256 = hash_file(Path(videos_dir) / d["file"])
        lrows.append(r)
    lab = out / "labels_eval.csv"
    write_labels_csv(lrows, lab)
    types = out / "labels_types.csv"
    with open(types, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "t_sec", "type_id"])
        for d in rows:
            for t, ty in zip(d["times"], d["types"]):
                w.writerow([d["video_id"], t, ty])
    return lab, types


def freeze(rows: list[dict], manifest: str | Path, camera: str = "C3", strict_scenes: bool = False,
           force: bool = False, videos_dir: Optional[str | Path] = None) -> dict:
    """Заморозить hold-out камеры (ДО настройки). strict_scenes: заморозить и другие камеры тех же сцен."""
    from .labels import hash_file
    from .metrics import git_sha

    mp = Path(manifest)
    if mp.exists() and not force:
        raise FileExistsError(f"{mp} уже есть: hold-out заморожен {read_json(mp).get('frozen_at')}. "
                              "Перезапись (--force) лишит его независимости.")
    cam_rows = [d for d in rows if (d.get("camera_id") or "").upper() == camera.upper()]
    scenes = {d.get("scene_id") for d in cam_rows}
    frozen = {d["video_id"] for d in cam_rows}
    overlap = [d["video_id"] for d in rows if d["video_id"] not in frozen and d.get("scene_id") in scenes]
    if strict_scenes:
        frozen |= set(overlap)
    hashes = {}
    for d in rows:
        if d["video_id"] in frozen and videos_dir and d.get("file") and (Path(videos_dir) / d["file"]).exists():
            hashes[d["video_id"]] = hash_file(Path(videos_dir) / d["file"])[1][:16]
    man = {"frozen": True, "kind": "holdout_frozen", "frozen_at": _dt.datetime.now().isoformat(timespec="seconds"),
           "git_sha": git_sha(), "camera": camera, "strict_scenes": strict_scenes,
           "video_ids": sorted(frozen), "sha256": hashes, "scenes": sorted(s for s in scenes if s),
           "scene_overlap_not_frozen": [] if strict_scenes else sorted(overlap),
           "rule": "заморожено ДО настройки; не использовать для подбора порогов/моделей; открыть один раз с --final"}
    write_json(mp, man)
    return man


def make_extra_splits(rows: list[dict], man: dict, out: str | Path, calib: float = 0.25, seed: int = 2026) -> Path:
    """splits.csv в формате eval.split: замороженные → holdout, остальные — train/calib группами по scene_id."""
    import random

    frozen = set(man.get("video_ids") or [])
    tune = [d for d in rows if d["video_id"] not in frozen and d["label"] != LABEL_UNKNOWN]
    scenes: dict[str, list[dict]] = {}
    for d in tune:
        scenes.setdefault(d.get("scene_id") or d["video_id"], []).append(d)
    rng = random.Random(seed)
    order = sorted(scenes)
    rng.shuffle(order)
    n_cal = int(round(len(order) * calib))
    split_of = {s: ("calib" if i < n_cal else "train") for i, s in enumerate(order)}
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "group", "split_group", "split", "fold", "label"])
        for d in rows:
            sc = d.get("scene_id") or d["video_id"]
            sp = "holdout" if d["video_id"] in frozen else split_of.get(sc, "excluded")
            w.writerow([d["video_id"], sc, sc, sp, -1, d["label"]])
    write_json(p.with_name(p.stem + "_manifest.json"), {"frozen": True, "kind": "holdout_frozen",
                                                         "holdout_from": man.get("camera"),
                                                         "frozen_at": man.get("frozen_at"), "seed": seed})
    return p


def stratified_report(metrics: dict, rows: list[dict], keys: tuple[str, ...] = STRATA,
                      videos_dir: Optional[str | Path] = None) -> dict:
    """Метрики по стратам (камера / планировка / свет / ориентация) из per_video eval.metrics."""
    by_vid = {d["video_id"]: d for d in rows}
    if videos_dir:
        from .common import ffprobe_info

        for d in rows:
            if not d.get("orientation") and d.get("file") and (Path(videos_dir) / d["file"]).exists():
                info = ffprobe_info(Path(videos_dir) / d["file"])
                if info.get("width") and info.get("height"):
                    d["orientation"] = "landscape" if info["width"] > info["height"] else "portrait"
    out: dict[str, Any] = {}
    for key in keys:
        groups: dict[str, list[dict]] = {}
        for e in metrics["per_video"]:
            d = by_vid.get(e["video_id"]) or next((by_vid[m] for m in e.get("members") or [] if m in by_vid), None)
            if d is None:
                continue
            groups.setdefault(str(d.get(key) or "—"), []).append(e)
        res = {}
        for val, es in sorted(groups.items()):
            pos = [e for e in es if e["gt_label"] == LABEL_VIOLATION]
            neg = [e for e in es if e["gt_label"] == LABEL_CLEAN]
            tp = sum(e["category"] in ("TP", "early", "late", "wrong_type") for e in pos)
            fp = sum(e["category"] == "FP" for e in neg)
            ev = sum(e.get("events_total") or 0 for e in pos)
            h2 = sum((e.get("event_hits") or {}).get("2", 0) for e in pos)
            res[val] = {"n": len(es), "n_pos": len(pos), "n_neg": len(neg), "tp": tp, "fn": len(pos) - tp, "fp": fp,
                        "tn": sum(e["category"] == "TN" for e in neg),
                        "failed_or_partial": sum(e["category"] in ("failed", "partial") for e in es),
                        "recall": round(tp / len(pos), 4) if pos else None,
                        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                        "hit2_events": round(h2 / ev, 4) if ev else None}
        out[key] = res
    out["note"] = "Страты — для диагностики переносимости; планировка/свет/ориентация НЕ признаки модели."
    return out


def format_strata(rep: dict) -> str:
    lines = []
    for key, vals in rep.items():
        if key == "note":
            continue
        lines.append(f"{key}:")
        for val, m in vals.items():
            lines.append(f"  {val:12s} n={m['n']:3d} (+{m['n_pos']}/-{m['n_neg']}) FN={m['fn']} FP={m['fp']} "
                         f"failed/partial={m['failed_or_partial']} recall={m['recall']} precision={m['precision']} "
                         f"hit@2={m['hit2_events']}")
    return "\n".join(lines + [rep.get("note", "")])


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Наши съёмки: проверка разметки, заморозка C3, отчёты по стратам")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "freeze", "report"):
        p = sub.add_parser(name)
        p.add_argument("--labels", default="data/extra/labels.csv")
        p.add_argument("--videos", default=None)
    sub.choices["freeze"].add_argument("--camera", default="C3")
    sub.choices["freeze"].add_argument("--strict-scenes", action="store_true")
    sub.choices["freeze"].add_argument("--force", action="store_true")
    sub.choices["report"].add_argument("--run", required=True)
    args = ap.parse_args(argv)
    rows, problems = read_extra(args.labels)
    base = Path(args.labels).parent
    if args.cmd == "check":
        for p in problems:
            print("⚠", p)
        lab, types = to_eval_labels(rows, base, args.videos)
        for key in ("camera_id", "layout", "lighting", "label"):
            cnt: dict[str, int] = {}
            for d in rows:
                cnt[str(d.get(key) or "—")] = cnt.get(str(d.get(key) or "—"), 0) + 1
            print(f"{key}: {dict(sorted(cnt.items()))}")
        print(f"→ {lab}, {types}")
        return 1 if any("нет [" in p for p in problems) else 0
    if args.cmd == "freeze":
        man = freeze(rows, base / "holdout_manifest.json", args.camera, args.strict_scenes, args.force, args.videos)
        sp = make_extra_splits(rows, man, base / "splits.csv")
        print(f"заморожено {len(man['video_ids'])} роликов камеры {args.camera}"
              + (f"; сцены пересекаются с другими камерами у {len(man['scene_overlap_not_frozen'])} роликов "
                 "(они в tune — при сомнении --strict-scenes)" if man["scene_overlap_not_frozen"] else ""))
        print(f"→ {base / 'holdout_manifest.json'}, {sp}")
        return 0
    m = read_json(Path(args.run) / "metrics.json")
    rep = stratified_report(m, rows, videos_dir=args.videos)
    write_json(Path(args.run) / "strata.json", rep)
    print(format_strata(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
