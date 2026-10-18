"""Синтетические искажения ракурса и качества (аугментации) — для обучения и стресс-тестов.

ВАЖНО: это СИНТЕТИЧЕСКОЕ искажение (гомография/кроп/поворот/фотометрия одного и того же
кадра), а не съёмка с другой камеры. Оно не создаёт новых видимых сторон объектов,
окклюзий и параллакса, поэтому устойчивость на этих пресетах — необходимое, но НЕ
достаточное условие работы на реальном видео «под другим углом».

Две формы:
  (а) numpy: ``apply(img, aug)``, ``homography``, ``transform_points`` / ``transform_bbox``
      (геометрия общая для кадра и разметки), ``ClipAugmenter`` — одни параметры на весь клип;
  (б) CLI на файлах: ``python -m ml.ingest.augment in.mov out.mov --preset angle_left|...|all``
      (ffmpeg-фильтры; без ffmpeg — python-движок через cv2.VideoWriter).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

BBoxT = tuple[float, float, float, float]


@dataclass(frozen=True)
class Aug:
    """Параметры одного искажения. Все поля по умолчанию — тождественное преобразование."""

    name: str = "identity"
    persp: tuple[float, float] = (0.0, 0.0)  # (px, py): px>0 сжимает левый край («вид слева»), py>0 — верхний
    rotate_deg: float = 0.0                  # по часовой стрелке
    zoom: float = 1.0                        # доля кадра в кропе (0.7..1), затем растяжение до исходного размера
    center: tuple[float, float] = (0.5, 0.5)  # центр кропа (норм.)
    mirror: bool = False                     # горизонтальное зеркало
    brightness: float = 0.0                  # -1..1, аддитивно (как ffmpeg eq)
    contrast: float = 1.0
    gamma: float = 1.0
    blur_sigma: float = 0.0
    noise_sigma: float = 0.0                 # СКО гауссова шума, уровни 0..255
    jpeg_quality: int = 0                    # 0 = выкл; иначе сжать/разжать JPEG с таким качеством
    fps: Optional[float] = None              # только для видео: смена частоты кадров
    border: str = "replicate"                # replicate | constant (чёрный)
    out_orientation: Optional[str] = None    # None | landscape | portrait: выходной кадр с другим соотношением сторон
    kind: str = "perturbation"               # ВСЕ пресеты — синтетическое возмущение, не реальный новый ракурс

    @property
    def is_geometric(self) -> bool:
        return (self.persp != (0.0, 0.0) or self.rotate_deg != 0.0 or self.zoom != 1.0 or self.mirror
                or self.out_orientation is not None)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


PRESETS: dict[str, Aug] = {
    "angle_left": Aug("angle_left", persp=(0.22, 0.0), rotate_deg=-4.0, zoom=0.92),
    "angle_right": Aug("angle_right", persp=(-0.22, 0.0), rotate_deg=4.0, zoom=0.92),
    "high": Aug("high", persp=(0.0, 0.2), zoom=0.95),
    "low": Aug("low", persp=(0.0, -0.2), zoom=0.95),
    "rot_cw": Aug("rot_cw", rotate_deg=15.0, zoom=0.85),
    "rot_ccw": Aug("rot_ccw", rotate_deg=-15.0, zoom=0.85),
    "zoom": Aug("zoom", zoom=0.75, center=(0.5, 0.45)),
    "mirror": Aug("mirror", mirror=True),
    "dark": Aug("dark", brightness=-0.12, contrast=0.85, gamma=0.75),
    "bright": Aug("bright", brightness=0.08, contrast=1.1, gamma=1.3),
    "blur": Aug("blur", blur_sigma=2.0),
    "noise": Aug("noise", noise_sigma=8.0),
    "jpeg": Aug("jpeg", jpeg_quality=20),
    "fps25": Aug("fps25", fps=25.0),
    # «как будто другая камера жюри»: перспектива + поворот + кроп в ЛАНДШАФТ. Это perturbation
    # (то же изображение, искажённое), а не настоящий новый ракурс: нет параллакса и новых окклюзий.
    "angle_jury": Aug("angle_jury", persp=(0.18, 0.12), rotate_deg=8.0, zoom=0.9, center=(0.5, 0.45),
                      out_orientation="landscape"),
}
CLI_PRESETS = ["angle_left", "angle_right", "high", "low", "zoom", "mirror", "dark", "rot_cw", "rot_ccw",
               "bright", "blur", "noise", "jpeg", "fps25", "angle_jury"]
PERTURBATION_NOTE = "синтетическое искажение (perturbation), не реальный новый ракурс"


# ----------------------------------------------------------------------------- геометрия


def _persp_corners(px: float, py: float, w: float, h: float) -> np.ndarray:
    """Куда уходят углы TL, TR, BR, BL (в пикселях) для «трапеции» вида сбоку/сверху."""
    tl, tr, br, bl = [0.0, 0.0], [w, 0.0], [w, h], [0.0, h]
    if px > 0:
        tl[1] += px / 2 * h
        bl[1] -= px / 2 * h
    elif px < 0:
        tr[1] += -px / 2 * h
        br[1] -= -px / 2 * h
    if py > 0:
        tl[0] += py / 2 * w
        tr[0] -= py / 2 * w
    elif py < 0:
        bl[0] += -py / 2 * w
        br[0] -= -py / 2 * w
    return np.array([tl, tr, br, bl], dtype=np.float64)


def _homography_from_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    a, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b += [u, v]
    sol = np.linalg.solve(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    return np.append(sol, 1.0).reshape(3, 3)


def output_size(aug: Aug, width: int, height: int) -> tuple[int, int]:
    """(W', H') выходного кадра: тот же, либо с переставленными сторонами для out_orientation."""
    w, h = int(width), int(height)
    if aug.out_orientation == "landscape" and h > w or aug.out_orientation == "portrait" and w > h:
        return h, w
    return w, h


def _geometry(aug: Aug, width: int, height: int) -> tuple[np.ndarray, tuple[float, float, float, float], tuple[int, int]]:
    """(H_geo — зеркало/перспектива/поворот в исходном холсте, окно кропа (x0, y0, cw, ch), выходной размер)."""
    w, h = float(width), float(height)
    full = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    m = np.eye(3)
    if aug.mirror:
        m = np.array([[-1.0, 0.0, w], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) @ m
    if aug.persp != (0.0, 0.0):
        m = _homography_from_points(full, _persp_corners(aug.persp[0], aug.persp[1], w, h)) @ m
    if aug.rotate_deg:
        a = math.radians(aug.rotate_deg)  # по часовой (ось y вниз)
        cx, cy = w / 2, h / 2
        rot = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
        t1 = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
        t2 = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]])
        m = t2 @ rot @ t1 @ m
    out_w, out_h = output_size(aug, width, height)
    aspect = out_w / float(out_h)
    cw0 = min(w, h * aspect)  # наибольшее окно с соотношением выхода внутри холста
    z = min(max(float(aug.zoom), 0.05), 1.0)
    cw, ch = cw0 * z, cw0 / aspect * z
    x0 = min(max(aug.center[0] * w - cw / 2, 0.0), w - cw)
    y0 = min(max(aug.center[1] * h - ch / 2, 0.0), h - ch)
    return m, (x0, y0, cw, ch), (out_w, out_h)


def homography(aug: Aug, width: int, height: int) -> np.ndarray:
    """3x3 гомография: пиксели исходника -> пиксели выхода (:func:`output_size`), непрерывные координаты."""
    m, (x0, y0, cw, ch), (out_w, out_h) = _geometry(aug, width, height)
    sx, sy = out_w / cw, out_h / ch
    crop = np.array([[sx, 0.0, -x0 * sx], [0.0, sy, -y0 * sy], [0.0, 0.0, 1.0]])
    return crop @ m


def transform_points(points: Sequence[tuple[float, float]], aug: Aug, width: int, height: int) -> list[tuple[float, float]]:
    """Нормированные точки (x, y) исходника -> нормированные к ВЫХОДНОМУ кадру (могут выйти за 0..1)."""
    if not points:
        return []
    hm = homography(aug, width, height)
    out_w, out_h = output_size(aug, width, height)
    pts = np.asarray(points, dtype=np.float64) * [width, height]
    homog = np.c_[pts, np.ones(len(pts))] @ hm.T
    out = homog[:, :2] / homog[:, 2:3]
    return [(float(x) / out_w, float(y) / out_h) for x, y in out]


def transform_bbox(bbox: BBoxT, aug: Aug, width: int, height: int, min_visible: float = 0.25) -> Optional[BBoxT]:
    """BBox (норм.) -> описывающий bbox после искажения, обрезанный к кадру.

    None, если в кадре осталось меньше ``min_visible`` площади преобразованного bbox.
    """
    x1, y1, x2, y2 = bbox
    pts = transform_points([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], aug, width, height)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    bx1, by1, bx2, by2 = min(xs), min(ys), max(xs), max(ys)
    full = max((bx2 - bx1) * (by2 - by1), 1e-12)
    cx1, cy1, cx2, cy2 = max(bx1, 0.0), max(by1, 0.0), min(bx2, 1.0), min(by2, 1.0)
    if cx2 <= cx1 or cy2 <= cy1 or (cx2 - cx1) * (cy2 - cy1) / full < min_visible:
        return None
    return (cx1, cy1, cx2, cy2)


# ----------------------------------------------------------------------------- применение к кадру


def apply(img: np.ndarray, aug: Aug, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Применить искажение к BGR uint8 кадру. dtype сохраняется; размер — :func:`output_size`
    (тот же, кроме пресетов с ``out_orientation``)."""
    import cv2

    h, w = img.shape[:2]
    out = img
    if aug.is_geometric:
        border = cv2.BORDER_REPLICATE if aug.border == "replicate" else cv2.BORDER_CONSTANT
        out = cv2.warpPerspective(out, homography(aug, w, h), output_size(aug, w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=border)
    if aug.brightness or aug.contrast != 1.0 or aug.gamma != 1.0:
        lut = np.arange(256, dtype=np.float64) / 255.0
        lut = np.clip((lut - 0.5) * aug.contrast + 0.5 + aug.brightness, 0.0, 1.0)
        if aug.gamma != 1.0:
            lut = lut ** (1.0 / max(aug.gamma, 1e-3))
        out = cv2.LUT(out, np.clip(lut * 255.0 + 0.5, 0, 255).astype(np.uint8))
    if aug.blur_sigma > 0:
        out = cv2.GaussianBlur(out, (0, 0), aug.blur_sigma)
    if aug.noise_sigma > 0:
        rng = rng or np.random.default_rng(0)
        noise = rng.normal(0.0, aug.noise_sigma, out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if aug.jpeg_quality:
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), int(aug.jpeg_quality)])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return np.ascontiguousarray(out, dtype=np.uint8)


def resample_indices(times: Sequence[float], dst_fps: float) -> list[int]:
    """Смена fps для последовательности кадров: индексы исходных кадров на сетке 1/dst_fps."""
    if not times or dst_fps <= 0:
        return list(range(len(times)))
    t = np.asarray(times, dtype=np.float64)
    grid = np.arange(0.0, t[-1] + 0.5 / dst_fps, 1.0 / dst_fps)
    idx = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
    prev = np.clip(idx - 1, 0, len(t) - 1)
    choose_prev = np.abs(t[prev] - grid) <= np.abs(t[idx] - grid)
    return [int(i) for i in np.where(choose_prev, prev, idx)]


def random_aug(rng: np.random.Generator, strength: float = 1.0, p_landscape: float = 0.0) -> Aug:
    """Случайное искажение для обучения (A10/A05): ракурс + качество, сила 0..1.

    ``p_landscape`` — вероятность выхода в ландшафт (по умолчанию 0: поведение не меняется)."""
    s = float(min(max(strength, 0.0), 1.0))
    u = lambda lo, hi: float(rng.uniform(lo, hi))  # noqa: E731
    persp = (u(-0.25, 0.25) * s, u(-0.2, 0.2) * s) if rng.random() < 0.7 else (0.0, 0.0)
    rot = u(-20.0, 20.0) * s if rng.random() < 0.5 else 0.0
    zoom = 1.0 - u(0.05, 0.3) * s if (rng.random() < 0.6 or persp != (0.0, 0.0) or rot) else 1.0
    return Aug(
        name="random",
        persp=persp,
        rotate_deg=rot,
        zoom=zoom,
        center=(0.5 + u(-0.1, 0.1) * s, 0.5 + u(-0.1, 0.1) * s),
        mirror=bool(rng.random() < 0.5),
        brightness=u(-0.15, 0.12) * s if rng.random() < 0.6 else 0.0,
        contrast=1.0 + u(-0.2, 0.2) * s if rng.random() < 0.5 else 1.0,
        gamma=1.0 + u(-0.3, 0.4) * s if rng.random() < 0.4 else 1.0,
        blur_sigma=u(0.5, 2.0) * s if rng.random() < 0.3 else 0.0,
        noise_sigma=u(2.0, 8.0) * s if rng.random() < 0.3 else 0.0,
        jpeg_quality=int(u(20, 60)) if rng.random() < 0.3 * s else 0,
        out_orientation="landscape" if (p_landscape > 0 and rng.random() < p_landscape) else None,
    )


class ClipAugmenter:
    """Одни и те же параметры на все кадры клипа (важно для временных моделей)."""

    def __init__(self, seed: int = 0, strength: float = 1.0) -> None:
        self.rng = np.random.default_rng(seed)
        self.strength = strength

    def sample(self) -> Aug:
        return random_aug(self.rng, self.strength)

    def apply_frames(self, frames: Sequence[np.ndarray], aug: Optional[Aug] = None) -> tuple[list[np.ndarray], Aug]:
        aug = aug or self.sample()
        noise_rng = np.random.default_rng(int(self.rng.integers(0, 2**31)))
        return [apply(f, aug, noise_rng) for f in frames], aug


# ----------------------------------------------------------------------------- файлы (ffmpeg / python)


def ffmpeg_filter(aug: Aug, width: int, height: int) -> str:
    """Цепочка -vf для ffmpeg, эквивалентная :func:`apply` (геометрия — через perspective)."""
    parts: list[str] = []
    if aug.is_geometric and output_size(aug, width, height) != (width, height):
        m, (x0, y0, cw, ch), (out_w, out_h) = _geometry(aug, width, height)
        if not np.allclose(m, np.eye(3)):
            c = np.array([[0, 0, 1], [width, 0, 1], [0, height, 1], [width, height, 1]], dtype=np.float64) @ m.T
            xy = c[:, :2] / c[:, 2:3]
            coords = ":".join(f"x{i}={x:.3f}:y{i}={y:.3f}" for i, (x, y) in enumerate(xy))
            parts.append(f"perspective={coords}:sense=destination:interpolation=linear")
        parts.append(f"crop={cw:.3f}:{ch:.3f}:{x0:.3f}:{y0:.3f}:exact=1,scale={out_w}:{out_h}:flags=bilinear")
    elif aug.is_geometric:
        hm = homography(aug, width, height)
        corners = np.array([[0, 0, 1], [width, 0, 1], [0, height, 1], [width, height, 1]], dtype=np.float64) @ hm.T
        xy = corners[:, :2] / corners[:, 2:3]  # порядок ffmpeg: TL, TR, BL, BR
        coords = ":".join(f"x{i}={x:.3f}:y{i}={y:.3f}" for i, (x, y) in enumerate(xy))
        parts.append(f"perspective={coords}:sense=destination:interpolation=linear")
    if aug.brightness or aug.contrast != 1.0 or aug.gamma != 1.0:
        parts.append(f"eq=brightness={aug.brightness:.3f}:contrast={aug.contrast:.3f}:gamma={aug.gamma:.3f}")
    if aug.blur_sigma > 0:
        parts.append(f"gblur=sigma={aug.blur_sigma:.2f}")
    if aug.noise_sigma > 0:
        parts.append(f"noise=alls={int(round(aug.noise_sigma * 2))}:allf=t")
    if aug.fps:
        parts.append(f"fps={aug.fps:g}")
    parts.append("format=yuv420p")
    return ",".join(parts)


def _augment_ffmpeg(src: str, dst: str, aug: Aug, exe: str, width: int, height: int) -> None:
    cmd = [exe, "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-an", "-vf", ffmpeg_filter(aug, width, height)]
    if aug.jpeg_quality:  # «JPEG-артефакты» на видео = сильное сжатие
        cmd += ["-c:v", "mpeg4", "-q:v", str(int(np.interp(aug.jpeg_quality, [10, 90], [24, 3])))]
    else:
        cmd += ["-c:v", "mpeg4", "-q:v", "3"]
    cmd.append(dst)
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg: {proc.stderr.decode('utf-8', 'replace')[-500:]}")


def _augment_python(src: str, dst: str, aug: Aug, seed: int) -> None:
    import cv2

    from .video import load_video

    clip = load_video(src, sample_fps=0.0, config={"storage": {"max_side": 0}})
    try:
        out_fps = float(aug.fps or clip.fps or 30.0)
        indices = resample_indices(clip.times, out_fps) if aug.fps else list(range(len(clip)))
        out_w, out_h = output_size(aug, clip.frame(0).shape[1], clip.frame(0).shape[0])
        writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (out_w, out_h))
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter не открылся: {dst}")
        rng = np.random.default_rng(seed)
        for i in indices:
            writer.write(apply(clip.frame(i), aug, rng))
        writer.release()
    finally:
        clip.close()


def augment_file(src: str, dst: str, aug: Aug, engine: str = "auto", seed: int = 0) -> str:
    """Записать искажённую копию видео. engine: auto | ffmpeg | python."""
    from .probe import ffmpeg_bin, probe

    exe = ffmpeg_bin() if engine in ("auto", "ffmpeg") else None
    if engine == "ffmpeg" and not exe:
        raise RuntimeError("ffmpeg не найден (PATH / $SENTINEL_FFMPEG)")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    if exe:
        info = probe(src)
        _augment_ffmpeg(src, dst, aug, exe, int(info["display_width"]), int(info["display_height"]))
    else:
        _augment_python(src, dst, aug, seed)
    return dst


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ml.ingest.augment",
        description="Синтетическое искажение ракурса/качества видео (НЕ замена проверки на реальной другой камере).",
    )
    ap.add_argument("src")
    ap.add_argument("dst", help="выходной файл; для --preset all — шаблон, к имени добавится _<preset>")
    ap.add_argument("--preset", default="angle_left", help="|".join(CLI_PRESETS) + "|all")
    ap.add_argument("--engine", default="auto", choices=["auto", "ffmpeg", "python"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    names = CLI_PRESETS if args.preset == "all" else [args.preset]
    unknown = [n for n in names if n not in PRESETS]
    if unknown:
        ap.error(f"неизвестный пресет: {unknown[0]}")
    dst = Path(args.dst)
    outputs = []
    for name in names:
        out = dst if args.preset != "all" else dst.with_name(f"{dst.stem}_{name}{dst.suffix or '.mov'}")
        augment_file(args.src, str(out), PRESETS[name], engine=args.engine, seed=args.seed)
        outputs.append({"preset": name, "output": str(out), "kind": PRESETS[name].kind, "note": PERTURBATION_NOTE})
    print(json.dumps(outputs, ensure_ascii=False, indent=2))  # только пути, без содержимого кадров
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
