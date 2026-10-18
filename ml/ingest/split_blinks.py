"""Нарезка наших съёмок по синхро-меткам «мигание светом» + сопоставление со сценами журнала.

    python -m ml.ingest.split_blinks --videos data/extra/incoming --journal data/extra/journal.csv \\
        --out data/extra/clips [--labels data/extra/labels.csv] [--dry-run] [--cameras C1,C2] [--force]

Вход: ``incoming/<камера>/*.mov`` (подпапки C1..C4 — камеры, снимавшие одни и те же сцены),
журнал CSV: scene_id,block,kind,code,title,priority,done,t_onset,t_end,layout,light,cashier,notes,shot_at
(разделитель «,» или «;», десятичная запятая допустима).

Синхро-метка — короткий (``min_dip_s``..``max_dip_s``, ~1 с) провал средней яркости кадра
относительно скользящего опорного уровня (75-й перцентиль за ±``ref_window_s``) с восстановлением
света. Длинное затемнение (свет выключили по сценарию) меткой НЕ считается.
Клип = [включение света − ``pre_s`` (0.5 с), выключение света следующей метки) либо до конца файла.
Сегмент до первой метки — подготовка, отбрасывается.

Сопоставление: для каждой камеры сегменты (по файлам в естественном порядке имён) ↔ сцены журнала
с done=да в порядке ``shot_at`` (иначе — порядок строк). Число должно совпасть; иначе камера
пропускается (``--force`` — сопоставить по порядку первые min(N) и пометить). Метка видна во всех
камерах одновременно, поэтому длительности сегментов сравниваются между камерами (рассинхрон
> ``max_cross_cam_dev_s`` — предупреждение).

Выход: ``<out>/<камера>/<scene_id>-<камера>.mov`` (H.264, 30 fps CFR, короткая сторона 464, без звука)
и labels.csv: video_id,file,label,t_sec,type_id,camera,layout,light,scene_id; t_sec = t_onset + pre_s
(t_onset — секунды от включения света). Для no_violation t_sec и type_id пустые (как в eval.labels).

ПРИВАТНОСТЬ: в отчёт и stdout не попадают title/notes журнала и кадры — только id, коды и времена.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

VIDEO_EXT = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
LABEL_FIELDS = ["video_id", "file", "label", "t_sec", "type_id", "camera", "layout", "light", "scene_id"]
DEFAULTS: dict[str, Any] = {
    "pre_s": 0.5,             # клип начинается за 0.5 с до включения света
    "ratio": 0.5,             # кадр «тёмный», если яркость < ratio * опорный уровень
    "min_drop": 20.0,         # ... и ниже опорного хотя бы на столько уровней (0..255)
    "ref_window_s": 5.0,      # окно опорного уровня ±, сек
    "ref_percentile": 75.0,
    "min_dip_s": 0.3,         # длительность синхро-провала
    "max_dip_s": 3.0,
    "merge_gap_s": 0.5,       # провалы ближе — одна метка (мерцание)
    "recover_ratio": 0.7,     # после провала свет вернулся: уровень после >= recover_ratio * уровень до
    "min_clip_s": 2.0,        # короче — подозрительный сегмент (двойная метка)
    "max_cross_cam_dev_s": 1.5,
    "fps": 30.0,
    "short_side": 464,
    "crf": 18,
}


# ----------------------------------------------------------------------------- яркость и метки


@dataclass
class Blink:
    off_t: float   # первый тёмный кадр (PTS, сек от первого кадра файла)
    on_t: float    # первый кадр после восстановления света
    depth: float   # опорный уровень − минимум в провале
    pre_level: float
    post_level: float

    @property
    def duration(self) -> float:
        return self.on_t - self.off_t


def _luma_mean(frame: Any, step: int = 4) -> float:
    fmt = getattr(frame.format, "name", "")
    if fmt.startswith(("yuv", "yuvj", "nv12", "nv21", "gray")) and "p1" not in fmt:  # 8-битные планарные
        plane = frame.planes[0]
        arr = np.frombuffer(plane, dtype=np.uint8)
        arr = arr[: plane.line_size * frame.height].reshape(frame.height, plane.line_size)[:, : frame.width]
        return float(arr[::step, ::step].mean())
    gray = frame.to_ndarray(format="gray")
    return float(gray[::step, ::step].mean())


def scan_brightness(path: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """(t, яркость 0..255) для КАЖДОГО кадра; t — PTS относительно первого кадра. PyAV, фолбэк OpenCV."""
    try:
        import av
    except ImportError:  # pragma: no cover
        av = None
    if av is not None:
        err = getattr(av.error, "FFmpegError", Exception)
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            tb = s.time_base
            start = (c.start_time or 0) / 1_000_000
            ts: list[float] = []
            bs: list[float] = []
            for packet in c.demux(s):
                try:
                    frames = packet.decode()
                except err:
                    continue
                for f in frames:
                    if f.pts is None:
                        continue
                    ts.append(float(f.pts * tb))
                    bs.append(_luma_mean(f))
        if not ts:
            raise RuntimeError(f"нет кадров: {path}")
        order = np.argsort(ts, kind="stable")
        t = np.asarray(ts)[order]
        t0 = float(t[0])
        return t - t0, np.asarray(bs)[order], {"t0_abs": t0, "container_start": start, "backend": "pyav"}
    import cv2  # pragma: no cover

    cap = cv2.VideoCapture(str(path))  # pragma: no cover
    ts, bs = [], []  # pragma: no cover
    while cap.grab():  # pragma: no cover
        ok, img = cap.retrieve()
        if ok:
            ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            bs.append(float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[::4, ::4].mean()))
    cap.release()  # pragma: no cover
    t = np.asarray(ts)  # pragma: no cover
    return t - t[0], np.asarray(bs), {"t0_abs": 0.0, "container_start": 0.0, "backend": "opencv"}  # pragma: no cover


def detect_blinks(t: np.ndarray, b: np.ndarray, params: Optional[dict[str, Any]] = None) -> tuple[list[Blink], list[dict[str, float]]]:
    """Синхро-метки и (отдельно) длинные затемнения по ряду яркости."""
    p = {**DEFAULTS, **(params or {})}
    t = np.asarray(t, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(t) < 3:
        return [], []
    # опорный уровень по бинам 0.1 с (скользящий перцентиль), затем интерполяция на кадры
    bin_s = 0.1
    nb = int(math.floor(t[-1] / bin_s)) + 1
    idx = np.minimum((t / bin_s).astype(int), nb - 1)
    sums = np.bincount(idx, weights=b, minlength=nb)
    cnt = np.bincount(idx, minlength=nb)
    have = cnt > 0
    binned = np.interp(np.arange(nb), np.nonzero(have)[0], sums[have] / cnt[have])
    half = max(1, int(round(p["ref_window_s"] / bin_s)))
    padded = np.pad(binned, half, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    ref_bins = np.percentile(win, p["ref_percentile"], axis=1)
    ref = ref_bins[idx]
    dark = (b < p["ratio"] * ref) & (ref - b > p["min_drop"])
    # серии тёмных кадров
    runs: list[list[int]] = []
    for i in np.nonzero(dark)[0]:
        if runs and t[i] - t[runs[-1][1]] <= p["merge_gap_s"]:
            runs[-1][1] = int(i)
        else:
            runs.append([int(i), int(i)])
    blinks: list[Blink] = []
    long_dark: list[dict[str, float]] = []
    for i0, i1 in runs:
        off_t = float(t[i0])
        on_i = min(i1 + 1, len(t) - 1)
        on_t = float(t[on_i]) if i1 + 1 < len(t) else float(t[i1])
        dur = on_t - off_t
        pre = b[(t >= off_t - 2.5) & (t < off_t - 0.2)]
        post = b[(t > on_t + 0.2) & (t <= on_t + 2.5)]
        pre_l = float(np.median(pre)) if pre.size else float(ref[i0])
        post_l = float(np.median(post)) if post.size else 0.0
        info = {"off_t": off_t, "on_t": on_t, "duration": dur}
        if dur > p["max_dip_s"]:
            long_dark.append(info)  # свет выключали по сценарию — не метка
            continue
        if dur < p["min_dip_s"] or i1 + 1 >= len(t):
            continue
        # свет горел И до, И после (симметрично): иначе это край длинного затемнения по сценарию
        if min(pre_l, post_l) < p["recover_ratio"] * max(pre_l, post_l):
            long_dark.append({**info, "transition": 1.0})
            continue
        blinks.append(Blink(off_t, on_t, float(ref[i0] - b[i0 : i1 + 1].min()), pre_l, post_l))
    return blinks, long_dark


# ----------------------------------------------------------------------------- журнал


def _truthy(v: Any) -> bool:
    return str(v or "").strip().lower() in {"1", "true", "yes", "y", "да", "д", "x", "+", "✓", "done", "ok", "снято"}


_VIOL = {"violation", "нарушение", "наруш", "v", "viol", "positive", "pos", "1", "да"}
_CLEAN = {"no_violation", "clean", "норма", "без нарушения", "чисто", "negative", "neg", "0", "normal", "нет", "ok"}


def _label(kind: str, code: str) -> str:
    k = (kind or "").strip().lower()
    if k in _VIOL:
        return "violation"
    if k in _CLEAN:
        return "no_violation"
    if re.fullmatch(r"V\d{2}", (code or "").strip().upper()):
        return "violation"
    return "unknown"


def parse_seconds(v: Any) -> Optional[float]:
    """'12.5' | '12,5' | '0:12.5' | '1:02:03' | '12 с' -> секунды; пусто/мусор -> None."""
    s = str(v if v is not None else "").strip().lower()
    s = re.sub(r"\s*(сек|с|s|sec)\.?$", "", s).replace(",", ".")
    if not s:
        return None
    try:
        if ":" in s:
            parts = [float(x) for x in s.split(":")]
            return float(sum(x * 60 ** k for k, x in enumerate(reversed(parts))))
        return float(s)
    except ValueError:
        return None


def _order_key(v: Any) -> Optional[tuple[int, Any]]:
    s = str(v or "").strip()
    if not s:
        return None
    try:
        return (0, float(s.replace(",", ".")))
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M",
                "%H:%M:%S", "%H:%M"):
        try:
            return (1, (datetime.strptime(s, fmt) - datetime(1900, 1, 1)).total_seconds())
        except ValueError:
            continue
    return (2, s)


def read_journal(path: str) -> list[dict[str, str]]:
    raw = Path(path).read_text(encoding="utf-8-sig")
    first = raw.splitlines()[0] if raw else ""
    delim = ";" if first.count(";") > first.count(",") else ","
    rows = list(csv.DictReader(io.StringIO(raw), delimiter=delim))
    return [{(k or "").strip().lower(): (v or "").strip() for k, v in r.items()} for r in rows]


def expected_scenes(journal: Sequence[dict[str, str]], warnings: list[str]) -> list[dict[str, Any]]:
    """Сцены с done=да в порядке съёмки (shot_at), с метками и t_sec."""
    done = [dict(r, _row=i) for i, r in enumerate(journal) if _truthy(r.get("done"))]
    keys = [_order_key(r.get("shot_at")) for r in done]
    if done and all(k is not None for k in keys) and len({k[0] for k in keys}) == 1:  # type: ignore[index]
        done = [r for _, r in sorted(zip(keys, done), key=lambda kr: (kr[0], kr[1]["_row"]))]
    elif done:
        warnings.append("shot_at пуст или в разных форматах — порядок сцен = порядок строк журнала")
    out = []
    for r in done:
        label = _label(r.get("kind", ""), r.get("code", ""))
        t_on = parse_seconds(r.get("t_onset"))
        if label == "unknown":
            warnings.append(f"сцена {r.get('scene_id')}: не распознан kind={r.get('kind')!r} -> label=unknown")
        if label == "violation" and t_on is None:
            warnings.append(f"сцена {r.get('scene_id')}: нарушение без t_onset")
        out.append({"scene_id": r.get("scene_id", ""), "label": label, "t_onset": t_on,
                    "type_id": (r.get("code", "").strip().upper() if label == "violation" else ""),
                    "layout": r.get("layout", ""), "light": r.get("light", ""), "block": r.get("block", "")})
    return out


# ----------------------------------------------------------------------------- сегменты


@dataclass
class Segment:
    camera: str
    file: str
    start: float          # сек от первого кадра файла
    end: float
    on_t: float           # включение света (метка)
    file_duration: float
    flags: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


def _natural_key(p: Path) -> list[Any]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]


def list_cameras(videos: str, only: Optional[Sequence[str]] = None) -> dict[str, list[Path]]:
    root = Path(videos)
    cams: dict[str, list[Path]] = {}
    for d in sorted([p for p in root.iterdir() if p.is_dir()], key=_natural_key):
        if only and d.name not in only:
            continue
        files = sorted([f for f in d.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXT], key=_natural_key)
        if files:
            cams[d.name] = files
    return cams


def read_marks(path: Optional[str]) -> dict[tuple[str, str], list[Blink]]:
    """Ручные метки (если автодетект ошибся): CSV camera,file,on_t[,off_t] — сек от первого кадра файла."""
    marks: dict[tuple[str, str], list[Blink]] = {}
    if not path:
        return marks
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            on = parse_seconds(r.get("on_t"))
            if on is None:
                continue
            off = parse_seconds(r.get("off_t"))
            key = (str(r.get("camera", "")).strip(), Path(str(r.get("file", "")).strip()).name)
            marks.setdefault(key, []).append(Blink(off if off is not None else max(0.0, on - 1.0), on, 0.0, 0.0, 0.0))
    for v in marks.values():
        v.sort(key=lambda bl: bl.on_t)
    return marks


def segments_for_file(camera: str, path: Path, params: dict[str, Any],
                      manual: Optional[list[Blink]] = None) -> tuple[list[Segment], dict[str, Any]]:
    t, b, info = scan_brightness(str(path))
    if manual is not None:
        blinks, long_dark = list(manual), []
    else:
        blinks, long_dark = detect_blinks(t, b, params)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    file_dur = float(t[-1] + dt)
    segs = []
    for k, bl in enumerate(blinks):
        start = max(0.0, bl.on_t - params["pre_s"])
        end = blinks[k + 1].off_t if k + 1 < len(blinks) else file_dur
        seg = Segment(camera, str(path), start, end, bl.on_t, file_dur)
        if seg.duration < params["min_clip_s"]:
            seg.flags.append("short_segment")
        segs.append(seg)
    report = {"file": path.name, "duration": file_dur, "frames": int(len(t)),
              "blinks": [asdict(x) | {"duration": x.duration} for x in blinks], "long_dark": long_dark,
              "preamble_s": blinks[0].off_t if blinks else file_dur, "manual_marks": manual is not None,
              **{k: info[k] for k in ("t0_abs", "container_start")}}
    return segs, report


def cross_camera_check(per_cam: dict[str, list[Segment]], max_dev: float) -> list[str]:
    warns = []
    n = min((len(v) for v in per_cam.values()), default=0)
    for i in range(n):
        durs = {c: v[i].duration for c, v in per_cam.items()}
        if durs and max(durs.values()) - min(durs.values()) > max_dev:
            warns.append(f"сегмент #{i + 1}: длительности по камерам расходятся {json.dumps({c: round(d, 2) for c, d in durs.items()})}"
                         " — возможна пропущенная/лишняя метка")
    return warns


# ----------------------------------------------------------------------------- нарезка


def _ffmpeg() -> Optional[str]:
    from .probe import ffmpeg_bin

    return ffmpeg_bin()


def _scale_expr(short: int) -> str:
    return f"scale='if(gt(iw,ih),-2,{short})':'if(gt(iw,ih),{short},-2)':flags=bicubic"


def cut_ffmpeg(seg: Segment, dst: Path, params: dict[str, Any], offset: float) -> None:
    exe = _ffmpeg()
    if not exe:
        raise RuntimeError("ffmpeg не найден")
    dst.parent.mkdir(parents=True, exist_ok=True)
    last_err = ""
    for enc in ("libx264", "libopenh264"):
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{seg.start + offset:.6f}", "-i", seg.file,
               "-t", f"{seg.duration:.6f}", "-an", "-vf", f"{_scale_expr(int(params['short_side']))},fps={params['fps']:g}",
               "-c:v", enc, "-pix_fmt", "yuv420p"]
        cmd += ["-crf", str(params["crf"]), "-preset", "veryfast"] if enc == "libx264" else ["-b:v", "6M"]
        proc = subprocess.run(cmd + ["-movflags", "+faststart", str(dst)], capture_output=True)
        if proc.returncode == 0:
            return
        last_err = proc.stderr.decode("utf-8", "replace")[-400:]
    raise RuntimeError(f"ffmpeg: не удалось закодировать H.264: {last_err}")


def cut_pyav(seg: Segment, dst: Path, params: dict[str, Any]) -> None:
    """Фолбэк без ffmpeg CLI: PyAV (libx264 из колеса), CFR по ближайшему кадру, поворот учтён."""
    import av
    import cv2

    from .augment import resample_indices
    from .probe import pyav_frame_orientation

    fps = float(params["fps"])
    short = int(params["short_side"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    frames: list[np.ndarray] = []
    times: list[float] = []
    with av.open(seg.file) as c:
        s = c.streams.video[0]
        tb = s.time_base
        first = next((f for f in c.decode(s) if f.pts is not None), None)
        if first is None:
            raise RuntimeError(f"нет кадров: {seg.file}")
        t0, orient = float(first.pts * tb), pyav_frame_orientation(first)
        c.seek(int(max(0.0, seg.start - 2.0 + t0) / tb), stream=s, backward=True, any_frame=False)
        for f in c.decode(s):
            if f.pts is None:
                continue
            t = float(f.pts * tb) - t0
            if t < seg.start - 1e-6:
                continue
            if t >= seg.end - 1e-6:
                break
            frames.append(orient.apply(f.to_ndarray(format="bgr24")))
            times.append(t - seg.start)
    if not frames:
        raise RuntimeError(f"нет кадров в сегменте {seg.start:.2f}-{seg.end:.2f} с")
    h, w = frames[0].shape[:2]
    scale = short / float(min(h, w))
    ow, oh = int(round(w * scale / 2) * 2), int(round(h * scale / 2) * 2)
    grid_n = max(1, int(round(seg.duration * fps)))
    idx = resample_indices(times, fps)[:grid_n]
    from fractions import Fraction

    encoders = [e for e in ("libx264", "libopenh264", "h264_nvenc", "h264_qsv", "h264_amf") if e in av.codecs_available]
    if not encoders:
        raise RuntimeError("в PyAV нет H.264-кодировщика (libx264/libopenh264/...)")
    with av.open(str(dst), "w") as out:
        opts = {"crf": str(params["crf"]), "preset": "veryfast"} if encoders[0] == "libx264" else {}
        st = out.add_stream(encoders[0], rate=int(round(fps)), options=opts)
        st.width, st.height, st.pix_fmt = ow, oh, "yuv420p"
        st.codec_context.time_base = Fraction(1, int(round(fps)))
        for k, i in enumerate(idx):
            img = cv2.resize(frames[i], (ow, oh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
            vf = av.VideoFrame.from_ndarray(img, format="bgr24")
            vf.pts, vf.time_base = k, Fraction(1, int(round(fps)))
            for pkt in st.encode(vf):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)


def _merge_labels(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if path.is_file():
        with open(path, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                existing[r.get("video_id", "")] = r
    for r in rows:
        existing[r["video_id"]] = r
    return sorted(existing.values(), key=lambda r: (str(r.get("scene_id", "")), str(r.get("camera", ""))))


# ----------------------------------------------------------------------------- main


def split(videos: str, journal: str, out: str, *, labels: Optional[str] = None, dry_run: bool = False,
          cameras: Optional[Sequence[str]] = None, force: bool = False, engine: str = "auto",
          params: Optional[dict[str, Any]] = None, marks: Optional[str] = None) -> dict[str, Any]:
    from .names import normalize_video_id

    p = {**DEFAULTS, **(params or {})}
    warnings: list[str] = []
    scenes = expected_scenes(read_journal(journal), warnings)
    cams = list_cameras(videos, cameras)
    if not cams:
        warnings.append(f"нет подпапок-камер с видео в {videos}")
    out_dir = Path(out)
    labels_path = Path(labels) if labels else out_dir.parent / "labels.csv"
    report: dict[str, Any] = {"dry_run": dry_run, "params": p, "scenes_expected": [s["scene_id"] for s in scenes],
                              "cameras": {}, "warnings": warnings}
    per_cam: dict[str, list[Segment]] = {}
    offsets: dict[str, float] = {}
    manual = read_marks(marks)
    for cam, files in cams.items():
        segs: list[Segment] = []
        file_reports = []
        for f in files:
            s, rep = segments_for_file(cam, f, p, manual.get((cam, f.name)))
            segs += s
            file_reports.append(rep)
            offsets[str(f)] = float(rep["t0_abs"] - rep["container_start"])
        per_cam[cam] = segs
        status = "ok" if len(segs) == len(scenes) else "count_mismatch"
        report["cameras"][cam] = {"files": file_reports, "segments": len(segs), "expected": len(scenes), "status": status,
                                  "segments_detail": [{"file": Path(s.file).name, "start": s.start, "end": s.end,
                                                       "duration": s.duration, "flags": s.flags} for s in segs]}
        if status != "ok":
            warnings.append(f"{cam}: сегментов {len(segs)}, сцен в журнале {len(scenes)} — "
                            + ("сопоставлено по порядку (--force)" if force else "камера пропущена"))
    warnings += cross_camera_check(per_cam, float(p["max_cross_cam_dev_s"]))

    rows: list[dict[str, Any]] = []
    use_ffmpeg = engine == "ffmpeg" or (engine == "auto" and _ffmpeg() is not None)
    for cam, segs in per_cam.items():
        if len(segs) != len(scenes) and not force:
            continue
        matched = []
        for seg, sc in zip(segs, scenes):
            vid = normalize_video_id(f"{sc['scene_id']}-{cam}")
            dst = out_dir / cam / f"{vid}.mov"
            t_sec = (sc["t_onset"] + p["pre_s"]) if (sc["label"] == "violation" and sc["t_onset"] is not None) else None
            if t_sec is not None and t_sec >= seg.duration:
                seg.flags.append("t_onset_outside_clip")
                warnings.append(f"{vid}: t_sec={t_sec:.2f} за пределами клипа ({seg.duration:.2f} с)")
            matched.append({"video_id": vid, "scene_id": sc["scene_id"], "start": seg.start, "end": seg.end,
                            "t_sec": t_sec, "label": sc["label"], "flags": list(seg.flags)})
            if not dry_run:
                if use_ffmpeg:
                    cut_ffmpeg(seg, dst, p, offsets.get(seg.file, 0.0))
                else:
                    cut_pyav(seg, dst, p)
            try:
                rel = dst.relative_to(labels_path.parent).as_posix()
            except ValueError:
                rel = dst.as_posix()
            rows.append({"video_id": vid, "file": rel, "label": sc["label"],
                         "t_sec": "" if t_sec is None else f"{t_sec:.3f}", "type_id": sc["type_id"], "camera": cam,
                         "layout": sc["layout"], "light": sc["light"], "scene_id": sc["scene_id"]})
        report["cameras"][cam]["matched"] = matched
    report["labels_rows"] = len(rows)
    if not dry_run and rows:
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        merged = _merge_labels(labels_path, rows)
        with open(labels_path, "w", encoding="utf-8", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=LABEL_FIELDS, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(merged)
        report["labels_csv"] = str(labels_path)
    return report


def _print_summary(rep: dict[str, Any]) -> None:
    print(f"{'камера':<8}{'файлов':>7}{'меток':>7}{'сегм.':>7}{'сцен':>6}  статус")
    for cam, c in rep["cameras"].items():
        marks = sum(len(f["blinks"]) for f in c["files"])
        print(f"{cam:<8}{len(c['files']):>7}{marks:>7}{c['segments']:>7}{c['expected']:>6}  {c['status']}")
    for w in rep["warnings"]:
        print(f"! {w}")
    print(f"строк labels: {rep['labels_rows']}" + (" (dry-run, файлы не записаны)" if rep["dry_run"] else ""))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ml.ingest.split_blinks",
                                 description="Нарезка съёмок по синхро-меткам (выключение света) + labels.csv")
    ap.add_argument("--videos", required=True, help="папка incoming с подпапками камер C1..C4")
    ap.add_argument("--journal", required=True)
    ap.add_argument("--out", required=True, help="куда писать клипы (<out>/<камера>/<id>.mov)")
    ap.add_argument("--labels", default=None, help="по умолчанию <out>/../labels.csv")
    ap.add_argument("--report", default=None, help="JSON-отчёт (по умолчанию <out>/split_report.json)")
    ap.add_argument("--cameras", default=None, help="например C1,C3")
    ap.add_argument("--dry-run", action="store_true", help="только отчёт сопоставления, без нарезки")
    ap.add_argument("--force", action="store_true", help="сопоставлять по порядку при несовпадении числа сцен")
    ap.add_argument("--engine", default="auto", choices=["auto", "ffmpeg", "pyav"])
    ap.add_argument("--set", action="append", default=[], help="параметр детектора, напр. --set max_dip_s=2.5")
    ap.add_argument("--marks", default=None, help="CSV ручных меток camera,file,on_t[,off_t] (перекрывает автодетект)")
    args = ap.parse_args(argv)
    params: dict[str, Any] = {}
    for kv in args.set:
        k, _, v = kv.partition("=")
        if k not in DEFAULTS:
            ap.error(f"неизвестный параметр {k}")
        params[k] = type(DEFAULTS[k])(float(v)) if not isinstance(DEFAULTS[k], str) else v
    rep = split(args.videos, args.journal, args.out, labels=args.labels, dry_run=args.dry_run,
                cameras=args.cameras.split(",") if args.cameras else None, force=args.force, engine=args.engine,
                params=params, marks=args.marks)
    rpath = Path(args.report) if args.report else Path(args.out) / "split_report.json"
    rpath.parent.mkdir(parents=True, exist_ok=True)
    rpath.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _print_summary(rep)
    print(f"отчёт: {rpath}")
    return 0 if all(c["status"] == "ok" for c in rep["cameras"].values()) and rep["cameras"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
