"""Сравнение бэкендов позы на одинаковых фрагментах (L2): одни и те же треки людей и кадры для каждого бэкенда.

Печатает только агрегаты (время, доля кадров со скелетом, доступность суставов). Подробности — JSON в --out
(локально, вне git). Реальные кадры не сохраняются и не выводятся.

    python tools/l2/pose_compare.py data/raw/X-1.mov data/raw/X-3.mov --out runs/l2/pose_compare
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np

# одинаковый отбор людей для всех бэкендов (у A07 MediaPipe по умолчанию — только кассир, 1 кроп)
SAME_SELECTION = {"max_persons_per_frame": 3, "roles": None}
GROUPS = {"head": [0, 1, 2, 3, 4], "shoulders": [5, 6], "wrists": [9, 10], "elbows": [7, 8], "hips": [11, 12]}
BACKENDS = {
    "rtmpose_onnx": {"backend": {"order": ["rtmpose_onnx"]}},
    "mediapipe_video": {"backend": {"order": ["mediapipe"], "mediapipe": {"running_mode": "video", "selection": SAME_SELECTION}}},
    "mediapipe_image": {"backend": {"order": ["mediapipe"], "mediapipe": {"running_mode": "image", "selection": SAME_SELECTION}}},
}


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def run(videos: list[str], out_dir: Path, sample_fps: float, kp_thr: float) -> dict:
    from ml.detect import detect
    from ml.ingest import load_video
    from ml.pose.config import load_config
    from ml.pose.estimator import estimate_impl
    from ml.pose.status import status as get_status
    from ml.track import track

    out_dir.mkdir(parents=True, exist_ok=True)
    common = {"selection": {"max_persons_per_frame": 3, "budget_s": 600.0, "max_crops_total": 100000, "roles": None}}
    summary: dict = {name: {"elapsed_s": 0.0, "crops": 0, "poses": 0, "person_frames": 0, "posed_person_frames": 0,
                            "joints": {g: [0, 0] for g in GROUPS}} for name in BACKENDS}
    for v in videos:
        clip = load_video(v, sample_fps=sample_fps)
        dets = detect(clip)
        tracks, _zones = track(clip, dets)
        persons = [t for t in tracks if t.label == "person"]
        rec = {"video": os.path.basename(v), "n_frames": len(clip.times), "n_person_tracks": len(persons)}
        for name, over in BACKENDS.items():
            cfg = _merge(_merge(load_config(), common), over)
            t0 = time.perf_counter()
            poses = estimate_impl(clip, persons, cfg)
            dt = time.perf_counter() - t0
            st = get_status() if callable(get_status) else {}
            s = summary[name]
            s["elapsed_s"] += dt
            s["crops"] += int(st.get("n_crops", 0) or 0)
            s["poses"] += len(poses)
            s["person_frames"] += int(st.get("n_crops", 0) or 0)
            s["posed_person_frames"] += len(poses)
            for p in poses:
                sc = np.array([k[2] for k in p.keypoints])
                for g, idx in GROUPS.items():
                    s["joints"][g][0] += int((sc[idx] >= kp_thr).sum())
                    s["joints"][g][1] += len(idx)
            rec[name] = {"elapsed_s": round(dt, 2), "poses": len(poses), "crops": st.get("n_crops"),
                         "state": st.get("state"), "providers": st.get("providers"), "stride": st.get("stride")}
        (out_dir / f"{Path(v).stem}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, s in summary.items():
        s["pose_rate"] = round(s["posed_person_frames"] / max(1, s["person_frames"]), 3)
        s["ms_per_crop"] = round(1000 * s["elapsed_s"] / max(1, s["crops"]), 1)
        s["joint_availability"] = {g: round(a / max(1, b), 3) for g, (a, b) in s["joints"].items()}
        del s["joints"]
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="runs/l2/pose_compare")
    ap.add_argument("--sample-fps", type=float, default=5.0)
    ap.add_argument("--kp-thr", type=float, default=0.3)
    a = ap.parse_args()
    summary = run(a.videos, Path(a.out), a.sample_fps, a.kp_thr)
    for name, s in summary.items():
        print(f"{name:16s} elapsed={s['elapsed_s']:.1f}s crops={s['crops']} ms/crop={s['ms_per_crop']} "
              f"pose_rate={s['pose_rate']} joints={s['joint_availability']}")


if __name__ == "__main__":
    main()
