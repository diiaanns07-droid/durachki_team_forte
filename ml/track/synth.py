"""СИНТЕТИКА (только тесты, replay и демо — НЕ прод): сцены кассы, ряды детекций (+ кадры) и возмущения ракурса.

Никаких реальных данных: прямоугольники-«люди» с цветом одежды, статичные предметы.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ml.track.contract import Detection

FPS = 5.0
LAPTOP = (0.38, 0.50, 0.58, 0.60)
COUNTER = (0.60, 0.52, 0.70, 0.60)
SAFE = (0.74, 0.74, 0.95, 0.95)
DOOR = (0.02, 0.05, 0.22, 0.40)
SEAT = (0.50, 0.73)          # кассир
SEAT2 = (0.28, 0.75)         # второй сотрудник
CLIENT = (0.50, 0.33)        # клиент по ту сторону стола
DOOR_PT = (0.12, 0.22)
NEAR_SAFE = (0.68, 0.70)
PSIZE = (0.20, 0.14)


@dataclass
class Person:
    key: str
    path: list            # [(t, x, y)] ключевые точки
    visible: list         # [(t_in, t_out)]
    color: tuple = (200, 80, 40)  # BGR одежда
    size: tuple = PSIZE
    jitter: float = 0.003
    miss: list = field(default_factory=list)  # интервалы, где человек ЕСТЬ, но детектор его не видит

    def pos(self, t: float) -> Optional[tuple]:
        if not any(a <= t <= b for a, b in self.visible):
            return None
        ts = [p[0] for p in self.path]
        x = float(np.interp(t, ts, [p[1] for p in self.path]))
        y = float(np.interp(t, ts, [p[2] for p in self.path]))
        return x, y


@dataclass
class Obj:
    label: str
    box: tuple
    visible: list
    score: float = 0.8
    follow: Optional[str] = None      # ключ человека, рядом с которым предмет (сумка)
    offset: tuple = (0.09, 0.03)
    size: tuple = (0.07, 0.05)


@dataclass
class Scene:
    duration: float
    persons: list
    objects: list = field(default_factory=list)
    width: int = 464
    height: int = 848


class SynthClip:
    def __init__(self, scene: Scene, A: Optional[np.ndarray] = None, render: bool = False,
                 width: Optional[int] = None, height: Optional[int] = None) -> None:
        self.scene = scene
        self.A = A
        self.path = "synthetic"
        self.duration = scene.duration
        self.fps = 30.0
        self.width = width or scene.width
        self.height = height or scene.height
        self.times = [round(i / FPS, 4) for i in range(int(round(scene.duration * FPS)))]
        self.render = render
        self._boxes: dict[int, list] = {}

    def frame(self, i: int):
        if not self.render:
            raise RuntimeError("no frames in this synthetic clip")
        img = np.full((self.height, self.width, 3), 170, np.uint8)
        for (x1, y1, x2, y2), color in self._boxes.get(i, []):
            p = lambda x, y: (int(round(x * self.width)), int(round(y * self.height)))  # noqa: E731
            a, b = p(x1, y1), p(x2, y2)
            img[a[1]:b[1], a[0]:b[0]] = color
            hh = max(2, (b[1] - a[1]) // 5)
            img[a[1]:a[1] + hh, a[0] + (b[0] - a[0]) // 3:b[0] - (b[0] - a[0]) // 3] = (140, 170, 210)
        return img

    def frame_at(self, t: float):
        return self.frame(int(np.argmin(np.abs(np.asarray(self.times) - t))))


def affine(theta_deg: float = 0.0, scale: float = 1.0, tx: float = 0.0, ty: float = 0.0,
           flip_x: bool = False, aspect: float = 464 / 848) -> np.ndarray:
    """Аффинка в нормированных координатах (поворот считается в изотропных единицах)."""
    th = np.deg2rad(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) * scale
    if flip_x:
        R = R @ np.array([[-1.0, 0.0], [0.0, 1.0]])
    S = np.diag([aspect, 1.0])
    M = np.linalg.inv(S) @ R @ S
    c = np.array([0.5, 0.5])
    t = c - M @ c + np.array([tx, ty])
    return np.hstack([M, t[:, None]])


def apply_pt(A: Optional[np.ndarray], x: float, y: float) -> tuple:
    if A is None:
        return (x, y)
    v = A @ np.array([x, y, 1.0])
    return (float(v[0]), float(v[1]))


def apply_box(A: Optional[np.ndarray], b: Sequence[float]) -> tuple:
    if A is None:
        return tuple(float(v) for v in b)
    pts = [apply_pt(A, b[0], b[1]), apply_pt(A, b[2], b[1]), apply_pt(A, b[2], b[3]), apply_pt(A, b[0], b[3])]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def make(scene: Scene, A: Optional[np.ndarray] = None, seed: int = 0, drop: float = 0.0,
         render: bool = False, every_n: int = 1, width: Optional[int] = None,
         height: Optional[int] = None, blackout: Sequence[tuple] = (), jitter: Optional[float] = None,
         H: Optional[np.ndarray] = None, noise_dets: float = 0.0) -> tuple[SynthClip, list[Detection], dict]:
    """blackout — интервалы «свет выключили» (нет НИ ОДНОЙ детекции); H — гомография (perspective);
    noise_dets — вероятность ложной детекции person с низким score в случайном месте на кадр."""
    rng = np.random.default_rng(seed)
    clip = SynthClip(scene, A, render, width, height)
    dets: list[Detection] = []
    gt: dict[int, dict] = {}
    for i, t in enumerate(clip.times):
        if i % every_n:
            continue
        dark = any(a <= t <= b for a, b in blackout)
        pos = {}
        for p in scene.persons:
            xy = p.pos(t)
            if xy is None:
                continue
            jit = p.jitter if jitter is None else jitter
            x = xy[0] + rng.normal(0, jit)
            y = xy[1] + rng.normal(0, jit)
            pos[p.key] = (x, y)
            b = apply_box(A, (x - p.size[0] / 2, y - p.size[1] / 2, x + p.size[0] / 2, y + p.size[1] / 2))
            if H is not None:
                b = apply_box_h(H, b)
            b = tuple(float(np.clip(v, 0, 1)) for v in b)
            clip._boxes.setdefault(i, []).append((b, p.color))
            if dark or any(a <= t <= bb for a, bb in p.miss) or (drop and rng.random() < drop):
                continue
            d = Detection(t=t, label="person", bbox=b, score=float(np.clip(rng.normal(0.85, 0.05), 0.5, 0.99)))
            dets.append(d)
            gt[id(d)] = {"key": p.key}
        for o in scene.objects:
            if not any(a <= t <= bb for a, bb in o.visible):
                continue
            if o.follow is not None and o.follow in pos:
                x, y = pos[o.follow]
                box = (x + o.offset[0] - o.size[0] / 2, y + o.offset[1] - o.size[1] / 2,
                       x + o.offset[0] + o.size[0] / 2, y + o.offset[1] + o.size[1] / 2)
            elif o.follow is not None:
                continue
            else:
                box = o.box
            if dark:
                continue
            b = apply_box(A, box)
            if H is not None:
                b = apply_box_h(H, b)
            b = tuple(float(np.clip(v, 0, 1)) for v in b)
            dets.append(Detection(t=t, label=o.label, bbox=b, score=o.score))
        if noise_dets and rng.random() < noise_dets:
            cx, cy = rng.uniform(0.1, 0.9, 2)
            dets.append(Detection(t=t, label="person", bbox=(cx - 0.08, cy - 0.06, cx + 0.08, cy + 0.06),
                                  score=float(rng.uniform(0.15, 0.55))))
    return clip, dets, gt


def homography(k: float = 0.25) -> np.ndarray:
    """Перспектива «камера наклонена»: верх кадра сжат (нормированные координаты)."""
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, k, 1.0]]) @ np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def apply_box_h(H: np.ndarray, b: Sequence[float]) -> tuple:
    pts = []
    for x, y in ((b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])):
        # центрируем, чтобы перспектива была вокруг центра кадра
        v = H @ np.array([x - 0.5, y - 0.5, 1.0])
        pts.append((v[0] / v[2] + 0.5, v[1] / v[2] + 0.5))
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def base_objects(duration: float, safe_label: str = "safe") -> list:
    full = [(0.0, duration)]
    return [Obj("laptop", LAPTOP, full, 0.85), Obj("money_counter", COUNTER, full, 0.75),
            Obj(safe_label, SAFE, full, 0.8), Obj("door", DOOR, full, 0.7)]


def cashier(duration: float, **kw) -> Person:
    return Person("cashier", [(0.0, *SEAT), (duration, *SEAT)], [(0.0, duration)], color=(200, 80, 40), **kw)


def walker(key: str, keyframes: list, color: tuple) -> Person:
    return Person(key, keyframes, [(keyframes[0][0], keyframes[-1][0])], color=color)


def scene_clean(duration: float = 40.0) -> Scene:
    v = walker("visitor", [(8, *DOOR_PT), (12, *CLIENT), (28, *CLIENT), (32, *DOOR_PT)], (40, 40, 220))
    return Scene(duration, [cashier(duration), v], base_objects(duration))


def scene_v01(duration: float = 40.0) -> Scene:
    v = walker("outsider", [(8, *DOOR_PT), (12, *NEAR_SAFE), (24, *NEAR_SAFE), (28, *DOOR_PT)], (40, 40, 220))
    return Scene(duration, [cashier(duration), v], base_objects(duration))


def scene_v02(duration: float = 60.0) -> Scene:
    c = Person("cashier", [(0, *SEAT), (15, *SEAT), (19, *DOOR_PT), (40, *DOOR_PT), (44, *SEAT), (duration, *SEAT)],
               [(0.0, 19.0), (40.0, duration)], color=(200, 80, 40))
    objs = base_objects(duration) + [Obj("cash", (0.44, 0.56, 0.52, 0.60), [(0.0, duration)], 0.7)]
    return Scene(duration, [c], objs)


def scene_v02_outsider(duration: float = 60.0) -> Scene:
    """Кассир ушёл, на его место сел другой человек (другая одежда)."""
    c = Person("cashier", [(0, *SEAT), (15, *SEAT), (19, *DOOR_PT)], [(0.0, 19.0)], color=(200, 80, 40))
    o = Person("other", [(40, *DOOR_PT), (44, *SEAT), (55, *SEAT)], [(40.0, 55.0)], color=(30, 200, 240))
    return Scene(duration, [c, o], base_objects(duration))


def scene_v07(duration: float = 40.0) -> Scene:
    col = walker("collector", [(10, *DOOR_PT), (14, 0.62, 0.42), (20, 0.62, 0.42), (24, *DOOR_PT)], (40, 160, 40))
    bag_carry = Obj("cash_bag", (0, 0, 0, 0), [(10.0, 17.0)], 0.75, follow="collector")
    bag_left = Obj("cash_bag", (0.58, 0.47, 0.66, 0.52), [(17.2, duration)], 0.75)
    return Scene(duration, [cashier(duration), col], base_objects(duration) + [bag_carry, bag_left])


def scene_two_staff(duration: float = 40.0) -> Scene:
    s2 = Person("staff2", [(0, *SEAT2), (10, SEAT2[0] + 0.05, SEAT2[1]), (20, *SEAT2), (30, SEAT2[0] + 0.05, SEAT2[1]),
                           (duration, *SEAT2)], [(0.0, duration)], color=(120, 120, 30))
    return Scene(duration, [cashier(duration), s2], base_objects(duration))


def scene_no_desk(duration: float = 30.0) -> Scene:
    v = walker("visitor", [(5, *DOOR_PT), (9, *CLIENT), (20, *CLIENT), (24, *DOOR_PT)], (40, 40, 220))
    return Scene(duration, [cashier(duration), v], [])


def without(scene: Scene, label: str) -> Scene:
    """Та же сцена без предметов данного класса (например, без сейфа)."""
    return Scene(scene.duration, scene.persons, [o for o in scene.objects if o.label != label], scene.width, scene.height)


def scene_cashier_enters_client_seated(duration: float = 40.0) -> Scene:
    """Клиент уже сидит напротив; кассир входит через дверь в первые секунды и садится за кассу."""
    client = Person("client", [(0, *CLIENT), (duration, *CLIENT)], [(0.0, duration)], color=(40, 40, 220))
    cas = Person("cashier", [(1.0, *DOOR_PT), (4.0, *SEAT), (duration, *SEAT)], [(1.0, duration)], color=(200, 80, 40))
    return Scene(duration, [client, cas], base_objects(duration))


def scene_fidget(duration: float = 40.0) -> Scene:
    """Кассир дважды на ~1 с отклоняется от места (взять что-то) — это НЕ уход с рабочего места."""
    away = (SEAT[0] + 0.22, SEAT[1] + 0.06)
    path = [(0, *SEAT), (9.6, *SEAT), (10.0, *away), (11.0, *away), (11.4, *SEAT),
            (24.6, *SEAT), (25.0, *away), (26.0, *away), (26.4, *SEAT), (duration, *SEAT)]
    return Scene(duration, [Person("cashier", path, [(0.0, duration)], color=(200, 80, 40))], base_objects(duration))


def scene_v02_bounce(duration: float = 60.0) -> Scene:
    """Кассир ушёл (15 с), на 1 с мелькнул у края рабочего места (24 с) и снова ушёл: одно событие ухода."""
    mid = (0.18, 0.40)       # далеко от стола, по пути к двери
    edge = (0.47, 0.72)      # на 1 с вернулась к месту
    c = Person("cashier", [(0, *SEAT), (15, *SEAT), (17, *mid), (23.5, *mid), (24, *edge), (25, *edge), (25.5, *mid),
                           (27, *DOOR_PT)], [(0.0, 27.0)], color=(200, 80, 40))
    objs = base_objects(duration) + [Obj("cash", (0.44, 0.56, 0.52, 0.60), [(0.0, duration)], 0.7)]
    return Scene(duration, [c], objs)


def scene_crossing_front(duration: float = 30.0) -> Scene:
    """Посетитель проходит прямо перед сидящим кассиром; кассир перекрыт и не детектируется 1.4 с."""
    c = cashier(duration)
    c.miss = [(11.4, 12.8)]
    v = walker("visitor", [(8, 0.15, 0.70), (16, 0.85, 0.72)], (40, 40, 220))
    return Scene(duration, [c, v], base_objects(duration))
