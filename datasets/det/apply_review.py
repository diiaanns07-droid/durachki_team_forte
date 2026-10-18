"""Решения проверки листов (review_sheets.py) → боксы → Label Studio и/или черновой COCO.

    python datasets/det/apply_review.py --ls      # залить как пред-разметку «owlv2+claude-review» в Label Studio
    python datasets/det/apply_review.py --coco    # черновой datasets/det/annotations.json (status = не человек)

Решение по кадру (data/frames/_review/decisions/*.jsonl, по строке на кадр):
    {"file": "X-1/X-1_t0022.000.jpg", "keep": {"3": "person"}, "add": [{"label": "cash", "xyxy": [x1, y1, x2, y2]}],
     "skip": false, "note": ""}
keep — номера рамок-кандидатов с листа и их класс (рамка берётся пиксельно из candidates.jsonl);
add — пропущенный объект, рамка оценена по сетке листа (грубее, помечается src=added).
В Label Studio это ложится как предсказание: человек проверяет и жмёт Submit — только это даёт статус
«проверено человеком» при `ls_project.py export`.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASS_NAMES, INDEX_CSV, read_index  # noqa: E402
from prelabel_owlv2 import iou  # noqa: E402
from review_sheets import REVIEW_DIR  # noqa: E402

MODEL_VERSION = "owlv2+claude-review"
# Люди в дальнем офисе за двумя стёклами (10–25 px) — шум для детектора кассы, а проверяющие ставили их
# непоследовательно: такие боксы снимаются единым правилом (в комнате человек ≥ 70 px хотя бы по одной стороне).
MIN_PERSON_SIDE = 40
# У кассира один стол и одна счётная машинка; плоский детектор купюр перед ней часть проверяющих отмечала
# как вторую машинку. Единое правило: оставить одну, самую крупную.
MAX_PER_FRAME = {"money_counter": 1}
STATUS = "черновик: пред-разметка OWLv2, классы и рамки выбраны Claude по листам проверки; человеком НЕ проверено"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _area(xyxy: list[float]) -> float:
    return (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])


def apply(cands: dict[str, dict], decisions: list[dict]) -> tuple[dict[str, dict], list[str]]:
    """→ ({file: {"boxes": [...], "skip": bool, "note": str}}, ошибки)."""
    out: dict[str, dict] = {}
    errors: list[str] = []
    for d in decisions:
        f = d.get("file")
        c = cands.get(f)
        if c is None:
            errors.append(f"{f}: нет в candidates.jsonl")
            continue
        W, H = c["width"], c["height"]
        by_n = {int(x["n"]): x for x in c["cands"]}
        boxes: list[dict] = []
        for n, label in (d.get("keep") or {}).items():
            if label not in CLASS_NAMES:
                errors.append(f"{f}: #{n} неизвестный класс {label!r}")
            elif int(n) not in by_n:
                errors.append(f"{f}: нет рамки #{n}")
            else:
                boxes.append({"label": label, "xyxy": by_n[int(n)]["xyxy"], "src": by_n[int(n)]["src"]})
        for a in d.get("add") or []:
            try:
                x1, y1, x2, y2 = (float(v) for v in a["xyxy"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{f}: плохой add {a}")
                continue
            x1, x2 = sorted((max(0.0, min(W, x1)), max(0.0, min(W, x2))))
            y1, y2 = sorted((max(0.0, min(H, y1)), max(0.0, min(H, y2))))
            if a.get("label") not in CLASS_NAMES or x2 - x1 < 3 or y2 - y1 < 3:
                errors.append(f"{f}: плохой add {a}")
                continue
            boxes.append({"label": a["label"], "xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                          "src": "added"})
        boxes = [b for b in boxes if b["label"] != "person" or
                 max(b["xyxy"][2] - b["xyxy"][0], b["xyxy"][3] - b["xyxy"][1]) >= MIN_PERSON_SIDE]
        for label, k in MAX_PER_FRAME.items():
            same = sorted((b for b in boxes if b["label"] == label), key=lambda b: -_area(b["xyxy"]))
            boxes = [b for b in boxes if b["label"] != label or any(b is x for x in same[:k])]
        dedup: list[dict] = []                    # одна рамка на объект: дубль того же класса выкидываем
        for b in boxes:
            if all(not (b["label"] == k["label"] and iou(b["xyxy"], k["xyxy"]) > 0.8) for k in dedup):
                dedup.append(b)
        out[f] = {"boxes": dedup, "skip": bool(d.get("skip")), "note": d.get("note", "")}
    return out, errors


def ls_result(boxes: list[dict], W: int, H: int) -> list[dict]:
    res = []
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b["xyxy"]
        res.append({"id": f"rv{i}", "from_name": "label", "to_name": "image", "type": "rectanglelabels",
                    "original_width": W, "original_height": H, "image_rotation": 0,
                    "value": {"x": x1 / W * 100, "y": y1 / H * 100, "width": (x2 - x1) / W * 100,
                              "height": (y2 - y1) / H * 100, "rotation": 0, "rectanglelabels": [b["label"]]}})
    return res


def push_ls(reviewed: dict[str, dict], index: dict[str, dict], url: str) -> None:
    import ls_project

    api = ls_project.Api(url, ls_project.creds()["token"])
    pid = ls_project.project_id(api)
    if pid is None:
        raise SystemExit("проект Label Studio не найден — сначала `ls_project.py setup`")
    page, tasks = 1, []
    while True:
        r = api("GET", f"/api/tasks?project={pid}&page={page}&page_size=500&fields=all")
        batch = r.get("tasks", []) if isinstance(r, dict) else r
        tasks += batch
        if len(batch) < 500:
            break
        page += 1
    n = 0
    for t in tasks:
        f = t["data"].get("file")
        if f not in reviewed or reviewed[f]["skip"]:
            continue
        for p in t.get("predictions", []):         # перезаливка: старая версия ревью заменяется
            if p.get("model_version") == MODEL_VERSION:
                api("DELETE", f"/api/predictions/{p['id']}")
        W, H = index[f]["width"], index[f]["height"]
        api("POST", "/api/predictions", {"task": t["id"], "model_version": MODEL_VERSION,
                                         "result": ls_result(reviewed[f]["boxes"], W, H), "score": 1.0})
        n += 1
    api("PATCH", f"/api/projects/{pid}", {"model_version": MODEL_VERSION})
    print(f"Label Studio: залито {n} предсказаний {MODEL_VERSION}, проект показывает их по умолчанию")


def draft_coco(reviewed: dict[str, dict], index: dict[str, dict], out_dir: Path | None) -> int:
    import export_coco

    tasks = []
    for f, r in reviewed.items():
        if r["skip"]:
            continue
        W, H = index[f]["width"], index[f]["height"]
        tasks.append({"id": None, "data": {"file": f},
                      "annotations": [{"was_cancelled": False, "result": ls_result(r["boxes"], W, H)}]})
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "review_export.json"
        p.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
        return export_coco.main([str(p), "--status", STATUS, "--proposals", MODEL_VERSION,
                                 *(["--out-dir", str(out_dir)] if out_dir else [])])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review-dir", type=Path, default=REVIEW_DIR)
    ap.add_argument("--ls", action="store_true", help="залить в Label Studio как предсказания")
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--coco", action="store_true", help="черновой COCO в datasets/det")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    cands = {c["file"]: c for c in load_jsonl(args.review_dir / "candidates.jsonl")}
    decisions = [d for p in sorted((args.review_dir / "decisions").glob("*.jsonl")) for d in load_jsonl(p)]
    reviewed, errors = apply(cands, decisions)
    for e in errors:
        print("  !", e, file=sys.stderr)
    missing = sorted(set(cands) - set(reviewed))
    with open(args.review_dir / "reviewed.jsonl", "w", encoding="utf-8") as f:
        for fn, r in sorted(reviewed.items()):
            f.write(json.dumps({"file": fn, **r}, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for r in reviewed.values():
        for b in r["boxes"]:
            counts[b["label"]] = counts.get(b["label"], 0) + 1
    src = {s: sum(b["src"] == s for r in reviewed.values() for b in r["boxes"]) for s in ("owl", "extra", "added")}
    print(f"решений: {len(reviewed)} из {len(cands)} (нет: {len(missing)}), пропущено (skip): "
          f"{sum(r['skip'] for r in reviewed.values())}, ошибок: {len(errors)}")
    print("боксов по классам:", dict(sorted(counts.items(), key=lambda kv: CLASS_NAMES.index(kv[0]))))
    print("происхождение рамок:", src)
    index = {r["file"]: r for r in read_index(INDEX_CSV)}
    if args.ls:
        push_ls(reviewed, index, args.url)
    if args.coco:
        return draft_coco(reviewed, index, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
