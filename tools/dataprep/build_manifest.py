"""data/labels.csv + data/raw/*.mov → data/manifest.csv (sha256 + ffprobe) и data/review_queue.csv.

needs_review=1 — файлы с одинаковым sha256 и разными метками/таймкодами (решает человек, скрипт не выбирает).
Печатает только агрегаты.

usage: python -I -X utf8 tools/dataprep/build_manifest.py data
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from pathlib import Path


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_probe(j: dict) -> dict:
    """JSON ffprobe → метаданные видеопотока (rotation из side_data, avg_fps — VFR-средний)."""
    s = j["streams"][0]
    num, den = map(int, s["avg_frame_rate"].split("/"))
    rot = 0
    for sd in s.get("side_data_list", []) or []:
        rot = int(sd.get("rotation", 0) or 0)
    return dict(width=s["width"], height=s["height"], codec=s["codec_name"], rotation=rot,
                duration_s=round(float(j["format"]["duration"]), 3), nb_frames=int(s.get("nb_frames", 0)),
                avg_fps=round(num / den, 3) if den else 0.0, r_fps=s["r_frame_rate"])


def probe(p: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name:stream_side_data=rotation"
         ":format=duration", "-of", "json", str(p)], capture_output=True, text=True, check=True).stdout
    return parse_probe(json.loads(out))


def mark_duplicates(man: list[dict]) -> list[dict]:
    """dup_of / needs_review по sha256; возвращает строки для review_queue."""
    by_sha: dict[str, list[dict]] = {}
    for m in man:
        by_sha.setdefault(m["sha256"], []).append(m)
    for m in man:
        same = by_sha[m["sha256"]]
        m["dup_of"] = ";".join(d["video_id"] for d in same if d is not m)
        m["needs_review"] = int(len({(d["label"], d["t_sec"]) for d in same}) > 1)
    return [m for m in man if m["needs_review"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    a = ap.parse_args()
    data = a.data
    nm = data / "_dl" / "name_map.json"  # создаётся при распаковке; без него orig_name = file
    name_map = json.loads(nm.read_text(encoding="utf-8")) if nm.exists() else {}
    with open(data / "labels.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    man = []
    for r in rows:
        p = data / "raw" / r["file"]
        man.append(dict(video_id=r["video_id"], file=r["file"],
                        orig_name=name_map.get(f"raw/{r['file']}", {}).get("orig", r["file"]),
                        label=r["label"], t_sec=r["t_sec"], group=r["group"], md5=r["md5"], sha256=sha256_file(p),
                        size_bytes=p.stat().st_size, **probe(p)))
    rq = mark_duplicates(man)
    with open(data / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(man[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(man)
    with open(data / "review_queue.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["video_id", "dup_of", "label", "t_sec", "reason"])
        for m in rq:
            w.writerow([m["video_id"], m["dup_of"], m["label"], m["t_sec"],
                        "same sha256 as dup_of, different timecode; keep both, verify by human"])
    d = [m["duration_s"] for m in man]
    print(f"videos={len(man)} unique_sha256={len({m['sha256'] for m in man})} "
          f"pos={sum(m['label'] == 'violation' for m in man)} neg={sum(m['label'] == 'no_violation' for m in man)}")
    print(f"duration min/median/max = {min(d):.1f}/{statistics.median(d):.1f}/{max(d):.1f}s total={sum(d) / 60:.1f}min")
    print("resolutions:", sorted({(m["width"], m["height"], m["rotation"]) for m in man}),
          "codecs:", sorted({m["codec"] for m in man}),
          f"avg_fps range {min(m['avg_fps'] for m in man)}..{max(m['avg_fps'] for m in man)}")
    print("needs_review:", [m["video_id"] for m in rq])


if __name__ == "__main__":
    main()
