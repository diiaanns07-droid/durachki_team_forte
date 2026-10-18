"""Отбор 500–800 кадров для ручной разметки + задачи Label Studio с пред-разметкой OWLv2.

    python datasets/det/build_tasks.py --target 700

Кадров после extract_frames.py больше цели (≈1 кадр/с на 24 мин + плотные окна), поэтому отбираем жадно:
1. почти одинаковые соседние кадры (статичная камера, никто не двигается) выкидываются (--dup-frac);
2. с каждого видео минимум --min-per-video кадров, равномерно по времени (иначе сплит по видео бессмыслен);
3. остальное — по очкам: плотное окно около таймкода (+1), редкие классы в пред-разметке
   (cash, cash_bag, safe_*, money_counter, door_open; +1 за класс, максимум +2), удалённость по времени
   от уже взятых кадров того же видео (до +1.5).
Уже отобранные раньше кадры (data/frames/_ls/selected.txt) сохраняются; новые задачи пишутся в
data/frames/_ls/tasks_new.json — его и импортирует ls_project.py (так добавляются видео из datasets/extra).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FRAMES_DIR, INDEX_CSV, LS_DIR, PRED_JSONL, RARE, read_index  # noqa: E402

LS_PREFIX = "/data/local-files/?d=frames/"       # LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = <repo>/data


def load_predictions(path: Path) -> tuple[dict[str, list[dict]], str]:
    if not path.is_file():
        return {}, ""
    preds = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            j = json.loads(line)
            preds[j["file"]] = j["boxes"]
    meta_p = path.parent / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else {}
    return preds, f"owlv2-base-ens@{(meta.get('revision') or 'na')[:8]}/{meta.get('prompts_hash', 'na')}"


def mark_near_dups(rows: list[dict], frames: Path, frac: float, delta: float = 20.0) -> None:
    """r["dup"]=True, если от последнего оставленного кадра того же видео заметно (> delta уровней яркости на
    кадре, уменьшенном в 8 раз — шум сжатия усредняется) изменилось меньше доли frac пикселей.
    Средняя разница не годится: человек занимает несколько % кадра, его движение среднее почти не меняет."""
    import numpy as np
    from PIL import Image

    last: dict[str, object] = {}
    for r in sorted(rows, key=lambda r: (r["video_id"], r["t_sec"])):
        im = Image.open(frames / r["file"]).convert("L")
        a = np.asarray(im.resize((max(8, im.width // 8), max(8, im.height // 8))), dtype=np.float32)
        prev = last.get(r["video_id"])
        r["dup"] = prev is not None and prev.shape == a.shape and float((np.abs(a - prev) > delta).mean()) < frac
        if not r["dup"]:
            last[r["video_id"]] = a


def select(rows: list[dict], preds: dict[str, list[dict]], target: int, min_per_video: int,
           previous: set[str]) -> list[dict]:
    """Жадный отбор (см. docstring модуля). rows — с полем dup. Возвращает отобранные строки (включая previous)."""
    for r in rows:
        rare = {b["label"] for b in preds.get(r["file"], []) if b["label"] in RARE}
        r["prio"] = 1.0 * r["dense"] + min(2, len(rare))
    by_vid: dict[str, list[dict]] = {}
    for r in sorted(rows, key=lambda r: (r["video_id"], r["t_sec"])):
        by_vid.setdefault(r["video_id"], []).append(r)
    chosen: dict[str, dict] = {r["file"]: r for r in rows if r["file"] in previous}

    for rs in by_vid.values():
        pool = [r for r in rs if not r["dup"]]
        need = min_per_video - sum(r["file"] in chosen for r in rs)
        if need <= 0 or not pool:
            continue
        idx = sorted({round(k * (len(pool) - 1) / max(1, need - 1)) for k in range(min(need, len(pool)))})
        for k in idx:
            chosen.setdefault(pool[k]["file"], pool[k])

    def gap(r: dict) -> float:
        ts = [c["t_sec"] for c in chosen.values() if c["video_id"] == r["video_id"]]
        return min((abs(r["t_sec"] - t) for t in ts), default=3.0)

    cand = {r["file"]: r for r in rows if not r["dup"] and r["file"] not in chosen}
    gaps = {f: gap(r) for f, r in cand.items()}
    while len(chosen) < target and cand:
        f = max(cand, key=lambda f: (cand[f]["prio"] + min(gaps[f], 3.0) / 2, f))
        pick = cand.pop(f)
        chosen[f] = pick
        for g, r in cand.items():
            if r["video_id"] == pick["video_id"]:
                gaps[g] = min(gaps[g], abs(r["t_sec"] - pick["t_sec"]))
    return sorted(chosen.values(), key=lambda r: (r["video_id"], r["t_sec"]))


def to_task(r: dict, boxes: list[dict], model_version: str) -> dict:
    W, H = r["width"], r["height"]
    result = []
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b["xyxy"]
        result.append({"id": f"owl{i}", "from_name": "label", "to_name": "image", "type": "rectanglelabels",
                       "original_width": W, "original_height": H, "image_rotation": 0, "score": b["score"],
                       "value": {"x": x1 / W * 100, "y": y1 / H * 100, "width": (x2 - x1) / W * 100,
                                 "height": (y2 - y1) / H * 100, "rotation": 0, "rectanglelabels": [b["label"]]}})
    task = {"data": {"image": LS_PREFIX + r["file"], "file": r["file"], "video_id": r["video_id"],
                     "group": r["group"], "t_sec": r["t_sec"], "dense": r["dense"], "qa": "", "qa_note": ""}}
    if model_version:
        task["predictions"] = [{"model_version": model_version, "result": result,
                                "score": round(sum(b["score"] for b in boxes) / len(boxes), 4) if boxes else 0.0}]
    return task


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path, default=INDEX_CSV)
    ap.add_argument("--frames", type=Path, default=FRAMES_DIR)
    ap.add_argument("--pred", type=Path, default=PRED_JSONL)
    ap.add_argument("--out-dir", type=Path, default=LS_DIR)
    ap.add_argument("--target", type=int, default=700, help="сколько кадров всего (вместе с уже отобранными)")
    ap.add_argument("--min-per-video", type=int, default=6)
    ap.add_argument("--dup-frac", type=float, default=0.002,
                    help="кадр — почти-дубль, если изменилось меньше этой доли пикселей (0 — не фильтровать)")
    args = ap.parse_args(argv)

    rows = read_index(args.index)
    preds, model_version = load_predictions(args.pred)
    if not preds:
        print("ВНИМАНИЕ: нет пред-разметки (prelabel_owlv2.py) — задачи без боксов, отбор без учёта классов")
    sel_file = args.out_dir / "selected.txt"
    previous = set(sel_file.read_text(encoding="utf-8").split()) if sel_file.is_file() else set()
    mark_near_dups(rows, args.frames, args.dup_frac)
    chosen = select(rows, preds, args.target, args.min_per_video, previous)
    new = [r for r in chosen if r["file"] not in previous]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [to_task(r, preds.get(r["file"], []), model_version) for r in new]
    (args.out_dir / "tasks_new.json").write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
    sel_file.write_text("\n".join(r["file"] for r in chosen) + "\n", encoding="utf-8")

    counts: dict[str, int] = {}
    for r in chosen:
        for b in preds.get(r["file"], []):
            counts[b["label"]] = counts.get(b["label"], 0) + 1
    print(f"кадров в пуле: {len(rows)}, почти-дублей: {sum(r['dup'] for r in rows)}, "
          f"отобрано: {len(chosen)} (новых {len(new)}, плотных {sum(r['dense'] for r in chosen)}, "
          f"видео {len({r['video_id'] for r in chosen})})")
    print("боксов пред-разметки в отборе:", dict(sorted(counts.items())))
    print(f"→ {args.out_dir / 'tasks_new.json'}")
    if not 500 <= len(chosen) <= 800:
        print(f"ВНИМАНИЕ: отобрано {len(chosen)} — вне цели 500–800 (мало кадров в пуле или --target)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
