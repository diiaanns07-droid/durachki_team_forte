"""Экспорт Label Studio (JSON) → COCO: datasets/det/annotations.json (+ _train/_val), сплит по видео, статистика в README.

    python datasets/det/ls_project.py export           # скачивает JSON из Label Studio и вызывает этот скрипт
    python datasets/det/export_coco.py data/frames/_ls/export.json

* file_name — относительно data/frames (info.images_root); сами кадры в git не попадают.
* category_id — порядок CLASSES в common.py (1..10), не алфавит Label Studio.
* В COCO идут только задачи с отправленной (Submit) и не пропущенной аннотацией; аннотация без боксов —
  законный негативный кадр. Предсказания OWLv2 без Submit не экспортируются.
* Сплит train/val — по видео (группы одинаковых по md5 файлов вместе) и фиксируется в datasets/det/split.json:
  при доразметке старые видео не переезжают, новые распределяются так, чтобы каждый класс, встречающийся
  хотя бы в 2 группах, был и в val, а доля кадров val была ≈ --val-frac.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CAT_ID, CLASSES, DET_DIR, FRAMES_DIR, INDEX_CSV, LS_DIR  # noqa: E402

STATS_BEGIN, STATS_END = "<!-- STATS:BEGIN -->", "<!-- STATS:END -->"
HUMAN = "проверено человеком в Label Studio (Submit)"


def task_file(task: dict) -> str:
    d = task.get("data", {})
    if d.get("file"):
        return d["file"]
    url = d.get("image", "")
    q = parse_qs(urlparse(url).query).get("d")
    path = unquote(q[0]) if q else unquote(url)
    return path.split("frames/", 1)[1] if "frames/" in path else path


def rect_to_xywh(v: dict, W: int, H: int) -> tuple[float, float, float, float]:
    """Прямоугольник LS (проценты, поворот вокруг левого верхнего угла по часовой) → bbox COCO в пикселях, обрезанный."""
    x, y, w, h = v["x"] / 100 * W, v["y"] / 100 * H, v["width"] / 100 * W, v["height"] / 100 * H
    a = math.radians(v.get("rotation", 0) or 0)
    if a:
        c, s = math.cos(a), math.sin(a)
        pts = [(x + px * c - py * s, y + px * s + py * c) for px, py in ((0, 0), (w, 0), (w, h), (0, h))]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        x, y, w, h = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(W), x + w), min(float(H), y + h)
    return round(x1, 2), round(y1, 2), round(max(0.0, x2 - x1), 2), round(max(0.0, y2 - y1), 2)


def pick_annotation(task: dict) -> dict | None:
    anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    return max(anns, key=lambda a: (a.get("updated_at") or a.get("created_at") or "", a.get("id", 0))) if anns else None


def convert(tasks: list[dict], index: dict[str, dict]) -> tuple[list[dict], list[dict], list[str]]:
    """→ (images, annotations, ошибки). images без id/split; annotations ссылаются на images через "file"."""
    images, anns, errors = [], [], []
    seen: set[str] = set()
    for t in tasks:
        a = pick_annotation(t)
        if a is None:
            continue
        f = task_file(t)
        if f in seen:
            errors.append(f"{f}: задача-дубликат, взята первая")
            continue
        seen.add(f)
        meta = index.get(f, {})
        d = t.get("data", {})
        rects = [r for r in a.get("result", []) if r.get("type") == "rectanglelabels"]
        W = int(meta.get("width") or next((r["original_width"] for r in rects), 0))
        H = int(meta.get("height") or next((r["original_height"] for r in rects), 0))
        if not (W and H):
            errors.append(f"{f}: неизвестен размер кадра (нет в index.csv и нет боксов)")
            continue
        vid = meta.get("video_id") or d.get("video_id") or f.split("/", 1)[0]
        images.append({"file_name": f, "width": W, "height": H, "video_id": vid,
                       "group": meta.get("group") or d.get("group") or vid,
                       "t_sec": float(meta.get("t_sec", d.get("t_sec", 0.0))),
                       "dense": int(meta.get("dense", d.get("dense", 0))), "ls_task_id": t.get("id")})
        for r in rects:
            labels = r["value"].get("rectanglelabels") or []
            if len(labels) != 1 or labels[0] not in CAT_ID:
                errors.append(f"{f}: неизвестная/множественная метка {labels}")
                continue
            x, y, w, h = rect_to_xywh(r["value"], W, H)
            if w < 1 or h < 1:
                errors.append(f"{f}: вырожденный бокс {labels[0]} {w}x{h} — пропущен")
                continue
            anns.append({"file": f, "category_id": CAT_ID[labels[0]], "bbox": [x, y, w, h],
                         "area": round(w * h, 2), "iscrowd": 0})
    return images, anns, errors


def assign_split(images: list[dict], anns: list[dict], fixed: dict[str, str], val_frac: float,
                 seed: str = "det-labels") -> dict[str, str]:
    """group → train|val. fixed (из split.json) не меняется; новые группы — покрытие классов в val + доля кадров."""
    split = dict(fixed)
    groups = sorted({im["group"] for im in images})
    n_img = {g: sum(im["group"] == g for im in images) for g in groups}
    g_of = {im["file_name"]: im["group"] for im in images}
    cls_groups: dict[int, set[str]] = {}
    for a in anns:
        cls_groups.setdefault(a["category_id"], set()).add(g_of[a["file"]])
    new = sorted((g for g in groups if g not in split), key=lambda g: hashlib.sha1(f"{seed}:{g}".encode()).hexdigest())
    for _cid, gs in sorted(cls_groups.items(), key=lambda kv: (len(kv[1]), kv[0])):     # самые редкие классы первыми
        if len(gs) < 2 or any(split.get(g) == "val" for g in gs):
            continue
        cand = [g for g in new if g in gs and g not in split]
        if cand and sum(split.get(g) != "val" for g in gs) >= 2:      # в train класс тоже останется
            split[min(cand, key=lambda g: (n_img[g], new.index(g)))] = "val"
    def last_train_for_some_class(g: str) -> bool:
        return any(g in gs and not any(x != g and split.get(x) != "val" for x in gs) for gs in cls_groups.values())

    total = sum(n_img.values())
    for g in new:
        if g not in split:
            val_n = sum(n_img[x] for x in split if split[x] == "val" and x in n_img)
            to_val = val_n + n_img[g] / 2 < val_frac * total and not last_train_for_some_class(g)
            split[g] = "val" if to_val else "train"
    return split


def build_coco(images: list[dict], anns: list[dict], split: dict[str, str], status: str = HUMAN,
               proposals: str = "") -> dict:
    """proposals — имя генератора, если разметка не проверена человеком: тогда info.kind=proposals и
    verified=false у каждой аннотации — по этим полям ml/detect (A05) отказывается на ней учиться и калиброваться."""
    images = sorted(images, key=lambda im: im["file_name"])
    for i, im in enumerate(images, 1):
        im["id"] = i
        im["split"] = split[im["group"]]
    id_of = {im["file_name"]: im["id"] for im in images}
    out_anns = []
    for i, a in enumerate(sorted(anns, key=lambda a: (id_of[a["file"]], a["category_id"], a["bbox"])), 1):
        out_anns.append({"id": i, "image_id": id_of[a["file"]], **{k: v for k, v in a.items() if k != "file"},
                         **({"source": proposals, "verified": False} if proposals else {})})
    return {
        "info": {"description": "Kassa Sentinel: детектор сверху (кассовый узел), ручная разметка поверх OWLv2",
                 "version": dt.date.today().isoformat(), "date_created": dt.datetime.now().isoformat(timespec="seconds"),
                 "images_root": "data/frames",
                 "file_name": "относительно images_root (от корня репо); кадры в git не хранятся — "
                              "воспроизводятся datasets/det/extract_frames.py",
                 "split": "по видео (группы одинаковых по md5 файлов вместе), datasets/det/split.json",
                 "status": status,
                 **({"kind": "proposals", "verified": False, "generator": proposals} if proposals else {})},
        "licenses": [],
        "categories": [{"id": CAT_ID[c["name"]], "name": c["name"], "supercategory": "object",
                        "contract_label": c["contract"]} for c in CLASSES],
        "images": images,
        "annotations": out_anns,
    }


def subset(coco: dict, part: str) -> dict:
    ids = {im["id"] for im in coco["images"] if im["split"] == part}
    return {**coco, "images": [im for im in coco["images"] if im["id"] in ids],
            "annotations": [a for a in coco["annotations"] if a["image_id"] in ids]}


def stats_md(coco: dict, dups: dict[str, str]) -> str:
    split_of = {im["id"]: im["split"] for im in coco["images"]}
    rows = ["| id | класс | контракт | боксов | train | val | кадров с классом |", "|---|---|---|---|---|---|---|"]
    for c in coco["categories"]:
        an = [a for a in coco["annotations"] if a["category_id"] == c["id"]]
        tr = sum(split_of[a["image_id"]] == "train" for a in an)
        rows.append(f"| {c['id']} | `{c['name']}` | `{c['contract_label']}` | {len(an)} | {tr} | {len(an) - tr} | "
                    f"{len({a['image_id'] for a in an})} |")
    n_ann = len(coco["annotations"])
    n_tr = sum(split_of[a["image_id"]] == "train" for a in coco["annotations"])
    rows.append(f"| | **всего** | | **{n_ann}** | {n_tr} | {n_ann - n_tr} | |")
    with_boxes = {a["image_id"] for a in coco["annotations"]}
    lines = [f"Экспорт: {coco['info']['date_created']}. Статус разметки: **{coco['info'].get('status', HUMAN)}**.",
             "", *rows, ""]
    for part in ("train", "val"):
        ims = [im for im in coco["images"] if im["split"] == part]
        vids = sorted({im["group"] for im in ims}, key=lambda v: (len(v), v))
        named = [f"{v} (≡{dups[v]})" if dups.get(v) else v for v in vids]
        lines.append(f"* **{part}**: {len(ims)} кадров (из них без боксов {sum(im['id'] not in with_boxes for im in ims)}, "
                     f"плотных около таймкодов {sum(im['dense'] for im in ims)}), {len(vids)} видео: {', '.join(named)}")
    return "\n".join(lines)


def update_readme(readme: Path, block: str) -> None:
    text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    new = f"{STATS_BEGIN}\n{block}\n{STATS_END}"
    if STATS_BEGIN in text and STATS_END in text:
        head, rest = text.split(STATS_BEGIN, 1)
        text = head + new + rest.split(STATS_END, 1)[1]
    else:
        text = text.rstrip() + "\n\n" + new + "\n"
    readme.write_text(text, encoding="utf-8", newline="\n")


def read_index_map(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        return {r["file"]: r for r in csv.DictReader(f)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", type=Path, nargs="?", default=LS_DIR / "export.json", help="JSON-экспорт Label Studio")
    ap.add_argument("--index", type=Path, default=INDEX_CSV)
    ap.add_argument("--out-dir", type=Path, default=DET_DIR)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--resplit", action="store_true", help="забыть split.json и распределить заново")
    ap.add_argument("--val", nargs="*", help="задать val вручную: группы видео (X-3 X-17 ...)")
    ap.add_argument("--status", default=HUMAN, help="кем проверена разметка (пишется в info.status и README)")
    ap.add_argument("--proposals", default="", help="генератор, если разметка НЕ проверена человеком "
                    "(info.kind=proposals, verified=false — так её распознаёт ml/detect)")
    args = ap.parse_args(argv)

    tasks = json.loads(args.export.read_text(encoding="utf-8"))
    images, anns, errors = convert(tasks, read_index_map(args.index))
    for e in errors:
        print("  !", e, file=sys.stderr)
    if not images:
        print("нет размеченных (Submit) задач — нечего экспортировать", file=sys.stderr)
        return 1

    split_p = args.out_dir / "split.json"
    fixed = {} if args.resplit or not split_p.is_file() else json.loads(split_p.read_text(encoding="utf-8"))["groups"]
    if args.val is not None:
        fixed = {im["group"]: ("val" if im["group"] in args.val else "train") for im in images}
    split = assign_split(images, anns, fixed, args.val_frac)
    coco = build_coco(images, anns, split, args.status, args.proposals)

    dump = {"indent": None, "ensure_ascii": False, "separators": (",", ":")}
    (args.out_dir / "annotations.json").write_text(json.dumps(coco, **dump), encoding="utf-8")
    for part in ("train", "val"):
        (args.out_dir / f"annotations_{part}.json").write_text(json.dumps(subset(coco, part), **dump), encoding="utf-8")
    split_p.write_text(json.dumps({"val_frac": args.val_frac, "groups": dict(sorted(split.items()))},
                                  ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    dups = {}
    vcsv = FRAMES_DIR / "videos.csv"
    if vcsv.is_file():
        with open(vcsv, encoding="utf-8", newline="") as f:
            dups = {r["group"]: r["dups"] for r in csv.DictReader(f) if r.get("dups")}
    update_readme(args.out_dir / "README.md", stats_md(coco, dups))

    n_val = sum(im["split"] == "val" for im in coco["images"])
    print(f"COCO: {len(coco['images'])} кадров (val {n_val}), {len(coco['annotations'])} боксов, "
          f"ошибок {len(errors)} → {args.out_dir / 'annotations.json'}")
    missing_val = [c["name"] for c in coco["categories"]
                   if not any(a["category_id"] == c["id"] for a in subset(coco, "val")["annotations"])]
    if missing_val:
        print("ВНИМАНИЕ: в val нет классов:", ", ".join(missing_val))
    return 0


if __name__ == "__main__":
    sys.exit(main())
