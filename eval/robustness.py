"""Стресс-тест «другого угла» и качества на СИНТЕТИЧЕСКИХ возмущениях (ffmpeg или ml.ingest.augment).

ВАЖНО: перспектива/поворот/кроп — синтетика; стабильность на ней НЕ доказывает работу на реальном новом
ракурсе камеры. Варианты наследуют группу/split исходного видео (split делается ДО аугментаций).

Метрики: доля смены вердикта относительно оригинала, сдвиг таймкода, точность относительно GT по вариантам.

CLI::

    python -m eval.robustness --videos data/videos --labels data/labels.csv --out runs/robust \\
        [--variants persp_left,rot_p10,mirror] [--subset calib --splits data/splits.csv] [--limit 5]
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

from .common import ffprobe_info, load_config, read_json, write_json

DISCLAIMER = ("Синтетические возмущения (ffmpeg): не доказательство качества на реальном новом ракурсе; "
              "mock-прогоны проверяют код, не качество CV.")

# {W} {H} — размер исходника после rotation; выход того же размера
VARIANTS: dict[str, str] = {
    "persp_left": "perspective=x0=0:y0=0:x1=W:y1=H*0.10:x2=0:y2=H:x3=W:y3=H*0.90:sense=destination",
    "persp_right": "perspective=x0=0:y0=H*0.10:x1=W:y1=0:x2=0:y2=H*0.90:x3=W:y3=H:sense=destination",
    "rot_p10": "rotate=10*PI/180:fillcolor=black",
    "rot_m10": "rotate=-10*PI/180:fillcolor=black",
    "rot_p20": "rotate=20*PI/180:fillcolor=black",
    "rot_m20": "rotate=-20*PI/180:fillcolor=black",
    "zoom80": "crop=trunc(iw*0.8/2)*2:trunc(ih*0.8/2)*2,scale={W}:{H}",
    "mirror": "hflip",
    "bright_p30": "lutyuv=y='clip(val*1.3,0,255)'",
    "bright_m30": "lutyuv=y='val*0.7'",
    "blur": "gblur=sigma=2.5",
    "h264_crf32": "null",
    "fps25": "fps=25",
    # «другой угол» может оказаться горизонтальным: пиксельный поворот на 90° и letterbox 16:9
    "landscape_rot90": "transpose=1",
    "landscape_pad": "pad=trunc(ih*16/9/2)*2:ih:(ow-iw)/2:0:black",
    # поворот только в метаданных (displaymatrix), пиксели не трогаем — проверка учёта rotation
    "rotmeta90": "@display_rotation=90",
    # переменный fps: выбрасываем кадры неравномерно, PTS сохраняются → время ДОЛЖНО совпасть с оригиналом
    "vfr_drop": "@vfr",
}
SIZE_CHANGING = {"landscape_rot90", "landscape_pad", "rotmeta90"}


def make_variant(src: str | Path, dst: str | Path, name: str, crf: int = 20) -> Path:
    """Вариант видео. Сначала ml.ingest.augment (A04), если умеет; иначе ffmpeg."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        from ml.ingest import augment as _aug  # type: ignore

        for fn_name in ("make_variant", "apply_variant", "augment_video"):
            fn = getattr(_aug, fn_name, None)
            if callable(fn):
                fn(str(src), str(dst), name)
                if dst.exists():
                    return dst
    except Exception:  # noqa: BLE001 — нет A04 / другой интерфейс → ffmpeg
        pass
    if name not in VARIANTS:
        raise KeyError(f"неизвестный вариант {name}; есть: {sorted(VARIANTS)}")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("нет ffmpeg")
    info = ffprobe_info(src)
    w, h = int(info.get("width") or 0) or 464, int(info.get("height") or 0) or 848
    spec = VARIANTS[name]
    q = 32 if name == "h264_crf32" else crf
    enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(q)]
    if spec == "@display_rotation=90":   # только метаданные: копия потока + displaymatrix
        cmd = ["ffmpeg", "-y", "-v", "error", "-display_rotation:v:0", "90", "-i", str(src), "-an", "-c", "copy", str(dst)]
    elif spec == "@vfr":
        sel = "select='lt(mod(n\\,3)\\,1)+between(t\\,1\\,2)'"   # 1/3 кадров, кроме 1–2 с — все
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-an", "-vf", f"{sel},format=yuv420p",
               "-fps_mode", "vfr"] + enc + [str(dst)]
    else:
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-an", "-vf", f"{spec.format(W=w, H=h)},format=yuv420p"] \
            + enc + [str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    if r.returncode != 0 and spec == "@display_rotation=90":  # старый ffmpeg: устаревший тег rotate
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-an", "-c", "copy", "-metadata:s:v:0", "rotate=90", str(dst)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    if r.returncode != 0 or not dst.exists():
        raise RuntimeError(f"ffmpeg {name}: {r.stderr[-400:]}")
    return dst


def _top_t(api: dict, gt_times: list[float]) -> Optional[float]:
    vios = [v for v in (api or {}).get("violations") or [] if isinstance(v, dict) and v.get("timecode_sec") is not None]
    if not vios:
        return None
    if gt_times:
        return float(min(vios, key=lambda v: min(abs(float(v["timecode_sec"]) - g) for g in gt_times))["timecode_sec"])
    return float(max(vios, key=lambda v: float(v.get("confidence") or 0))["timecode_sec"])


def compare(orig: dict[str, dict], variants: dict[str, dict[str, dict]], gt: Optional[dict] = None,
            shift_tol_s: float = 2.0) -> dict:
    """orig: {vid: api}; variants: {variant: {vid: api}}; gt: {vid: GroundTruth} (необязательно)."""
    per: list[dict] = []
    for var, preds in variants.items():
        for vid, api in preds.items():
            o = orig.get(vid)
            if o is None:
                continue
            g = (gt or {}).get(vid)
            times = list(getattr(g, "times", []) or [])
            ov, vv = o.get("verdict"), (api or {}).get("verdict") if (api or {}).get("status") == "done" else None
            to, tv = _top_t(o, times), _top_t(api, times)
            shift = abs(to - tv) if (to is not None and tv is not None) else None
            per.append({"variant": var, "video_id": vid, "orig_verdict": ov, "var_verdict": vv, "flip": ov != vv,
                        "shift_s": None if shift is None else round(shift, 3),
                        "correct": (vv == g.label) if g is not None else None})
    out: dict[str, Any] = {"per_variant": {}, "rows": per, "disclaimer": DISCLAIMER}
    for var in variants:
        rows = [r for r in per if r["variant"] == var]
        sh = [r["shift_s"] for r in rows if r["shift_s"] is not None]
        corr = [r["correct"] for r in rows if r["correct"] is not None]
        out["per_variant"][var] = {
            "n": len(rows), "flip_rate": round(sum(r["flip"] for r in rows) / len(rows), 4) if rows else None,
            "timecode_stable_rate": round(sum(s <= shift_tol_s for s in sh) / len(sh), 4) if sh else None,
            "mean_shift_s": round(sum(sh) / len(sh), 3) if sh else None,
            "accuracy_vs_gt": round(sum(corr) / len(corr), 4) if corr else None}
    sh_all = [r["shift_s"] for r in per if r["shift_s"] is not None]
    out["summary"] = {"n": len(per), "flip_rate": round(sum(r["flip"] for r in per) / len(per), 4) if per else None,
                      "timecode_stable_rate": round(sum(s <= shift_tol_s for s in sh_all) / len(sh_all), 4)
                      if sh_all else None, "shift_tol_s": shift_tol_s, "synthetic": True, "disclaimer": DISCLAIMER}
    return out


def main(argv: Optional[list[str]] = None) -> int:
    from .labels import load_ground_truth
    from .metrics import load_predictions
    from .run import run_folder, select_videos

    ap = argparse.ArgumentParser(description="Устойчивость к синтетическим возмущениям ракурса/качества")
    ap.add_argument("--videos", default="data/raw", help="папка видео (L1: data/raw)")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--out", default="runs/robustness")
    ap.add_argument("--variants", default=None, help="через запятую; по умолчанию config robustness.default_set")
    ap.add_argument("--mode", choices=["inproc", "cli", "api"], default="inproc")
    ap.add_argument("--api", default=None)
    ap.add_argument("--key", default="")
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--subset", default=None, help="напр. calib; holdout сюда не берите")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    rcfg = cfg.get("robustness") or {}
    names = args.variants.split(",") if args.variants else list(rcfg.get("default_set") or VARIANTS)
    subset = None
    split_of: dict[str, dict] = {}
    if Path(args.splits).exists():
        from .split import ids_for, load_splits

        split_of = load_splits(args.splits)
        if args.subset:
            subset = ids_for(split_of, args.subset)
    vids = select_videos(args.videos, args.labels, cfg, subset,
                         set(args.only.split(",")) if args.only else None, args.limit)
    out = Path(args.out)
    mode = "api" if args.api else args.mode
    run_folder(vids, out / "orig", cfg, mode, args.api or "", args.key, skip_existing=True)
    manifest = []
    var_preds: dict[str, dict[str, dict]] = {}
    for name in names:
        vv = []
        for vid, p in vids:
            dst = out / "variants" / name / f"{vid}.mov"
            if not dst.exists():
                make_variant(p, dst, name, int(rcfg.get("crf", 20)))
            vv.append((vid, dst))
            s = split_of.get(vid, {})
            manifest.append({"variant": name, "source_id": vid, "group": s.get("group", vid),
                             "split": s.get("split", ""), "file": str(dst)})
        run_folder(vv, out / name, cfg, mode, args.api or "", args.key, skip_existing=True)
        var_preds[name] = load_predictions(out / name / "preds")[0]
    with open(out / "variants_manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "source_id", "group", "split", "file"])
        w.writeheader()
        w.writerows(manifest)
    gt = load_ground_truth(args.labels) if Path(args.labels).exists() else None
    res = compare(load_predictions(out / "orig" / "preds")[0], var_preds, gt, float(rcfg.get("shift_tol_s", 2.0)))
    write_json(out / "robustness.json", res)
    print(DISCLAIMER)
    for var, d in res["per_variant"].items():
        print(f"{var:12s} n={d['n']:3d} смена вердикта={d['flip_rate']} стабильный таймкод={d['timecode_stable_rate']} "
              f"сдвиг={d['mean_shift_s']} acc={d['accuracy_vs_gt']}")
    print(f"ИТОГО: {res['summary']} → {out / 'robustness.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
