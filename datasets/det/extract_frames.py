"""Видео → кадры для разметки: шаг 1 с по всему ролику + 4 кадра/с в окне ±3 с вокруг таймкодов нарушений.

    python datasets/det/extract_frames.py                 # data/raw/*.mov + datasets/extra/** → data/frames/
    python datasets/det/extract_frames.py --window 4 --workers 8

Время кадра — PTS (`ffmpeg -ss t` до `-i` ищет точно по времени), видео VFR, поэтому i/fps не используем.
Дубликаты по md5 (X-4≡X-47, X-5≡X-48) режутся один раз, таймкоды всех копий объединяются.
Выход: data/frames/<video_id>/<video_id>_t0012.250.jpg, data/frames/index.csv, data/frames/videos.csv.
Повторный запуск дорезает только недостающие кадры.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (EXTRA_DIR, FRAMES_DIR, INDEX_FIELDS, LABELS_CSV, RAW_DIR, ROOT, assign_groups,  # noqa: E402
                    discover_videos, find_ffmpeg, frame_name, jpeg_size, probe_duration, read_timecodes)


def schedule(duration: float, times: list[float], step: float = 1.0, dense_fps: float = 4.0,
             window: float = 3.0, tail: float = 0.05) -> list[tuple[float, int]]:
    """[(t, dense)] на сетке в мс: 0, step, 2·step… и t±window с шагом 1/dense_fps. Совпадения схлопываются."""
    end_ms = int((duration - tail) * 1000)
    if end_ms < 0:
        return []
    grid = {i * int(round(step * 1000)) for i in range(end_ms // int(round(step * 1000)) + 1)}
    dense: set[int] = set()
    d = int(round(1000 / dense_fps))
    for t in times:
        c = int(round(t * 1000 / d)) * d                 # выравниваем на сетку 1/dense_fps, чтобы стыковались с шагом 1 с
        k = int(window * 1000) // d
        dense.update(c + j * d for j in range(-k, k + 1) if 0 <= c + j * d <= end_ms)
    return [(ms / 1000, int(ms in dense)) for ms in sorted(grid | dense)]


def extract_one(ffmpeg: str, video: Path, t: float, out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part.jpg")
    r = subprocess.run([ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{t:.3f}",
                        "-i", str(video), "-frames:v", "1", "-q:v", "2", str(tmp)], capture_output=True)
    if r.returncode != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(out)
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW_DIR)
    ap.add_argument("--extra", type=Path, default=EXTRA_DIR)
    ap.add_argument("--labels", type=Path, action="append",
                    help="CSV с таймкодами (по умолчанию data/labels.csv + datasets/extra/**/labels.csv)")
    ap.add_argument("--out", type=Path, default=FRAMES_DIR)
    ap.add_argument("--step", type=float, default=1.0, help="шаг по всему ролику, с")
    ap.add_argument("--dense-fps", type=float, default=4.0, help="кадров/с около таймкодов")
    ap.add_argument("--window", type=float, default=3.0, help="полуширина окна около таймкода, с "
                    "(таймкоды в разметке целые секунды и неточные)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", help="нарезать только эти video_id (индекс — по всем уже нарезанным)")
    args = ap.parse_args(argv)

    videos = discover_videos(args.raw, args.extra)
    if not videos:
        print(f"видео не найдены в {args.raw} и {args.extra}", file=sys.stderr)
        return 1
    label_files = args.labels or [LABELS_CSV, *(sorted(args.extra.rglob("labels.csv")) if args.extra.is_dir() else [])]
    tc = read_timecodes(label_files)
    if not any(p.is_file() for p in label_files):
        print("ВНИМАНИЕ: нет CSV с таймкодами (data/labels.csv из tools/dataprep/build_labels.py) — "
              "только шаг 1 с, без плотных окон", file=sys.stderr)
    assign_groups(videos)
    for v in videos:
        v.times = tc.get(v.video_id, [])
    # таймкоды копий-дубликатов переносим на представителя группы
    by_group: dict[str, list] = {}
    for v in videos:
        by_group.setdefault(v.group, []).append(v)
    ffmpeg = find_ffmpeg()
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    vrows: list[dict] = []
    jobs = []
    for group, members in by_group.items():
        rep = next(v for v in members if v.video_id == group)
        times = sorted({t for v in members for t in v.times})
        dur = probe_duration(ffmpeg, rep.path)
        sched = schedule(dur, times, args.step, args.dense_fps, args.window)
        path = rep.path.resolve()
        vrows.append(dict(video_id=rep.video_id, group=group, src=rep.src,
                          path=(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path).as_posix(),
            md5=rep.md5, duration=round(dur, 3), times=";".join(f"{t:g}" for t in times),
            dups=";".join(v.video_id for v in members if v is not rep), n_frames=len(sched)))
        for t, dense in sched:
            rel = frame_name(rep.video_id, t)
            out = args.out / rel
            rows.append(dict(file=rel, video_id=rep.video_id, group=group, t_sec=f"{t:.3f}", dense=dense, src=rep.src))
            if not out.is_file() and (not args.only or rep.video_id in args.only):
                jobs.append((rep.path, t, out))

    print(f"видео: {len(videos)} (групп {len(by_group)}), с таймкодами: {sum(bool(v.times) for v in videos)}; "
          f"кадров в плане: {len(rows)}, плотных: {sum(r['dense'] for r in rows)}, нарезать: {len(jobs)}")
    failed = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for i, ok in enumerate(ex.map(lambda j: extract_one(ffmpeg, *j), jobs), 1):
            failed += not ok
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    kept = []
    for r in rows:
        p = args.out / r["file"]
        if p.is_file():
            r["width"], r["height"] = jpeg_size(p)
            kept.append(r)
    with open(args.out / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(kept)
    with open(args.out / "videos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(vrows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(vrows)
    print(f"готово: {len(kept)} кадров в {args.out} (не нарезалось: {failed})")
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
