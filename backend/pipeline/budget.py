"""Бюджет времени задачи: дедлайн, адаптивный sample_fps, таймауты и пропуск стадий при отставании."""
from __future__ import annotations

import math
import time
from typing import Callable, Mapping, Optional

from .config import cfg_get

# Чем дальше в списке, тем раньше стадия пропускается при нехватке времени.
SKIP_PRIORITY = ["vlm_discover", "vlm_verify", "pose"]


def choose_sample_fps(duration: Optional[float], deadline_s: float, device: str, cfg: Mapping) -> float:
    """fps сэмплирования: base_fps, но не больше max_frames кадров и с поправкой на жёсткий дедлайн."""
    base = float(cfg_get(cfg, "sampling.base_fps", 5.0))
    min_fps = float(cfg_get(cfg, "sampling.min_fps", 2.0))
    step = float(cfg_get(cfg, "sampling.fps_step", 0.5)) or 0.0
    max_frames = float(cfg_get(cfg, "sampling.max_frames_gpu" if device == "cuda" else "sampling.max_frames_cpu",
                               900 if device == "cuda" else 300))
    fps = base
    if duration and duration > 0 and duration * fps > max_frames:
        fps = max_frames / duration
    ref_deadline = float(cfg_get(cfg, "sampling.reference_deadline_s", 240))
    if deadline_s and ref_deadline > 0 and deadline_s < ref_deadline:
        scale = max(float(cfg_get(cfg, "sampling.min_deadline_scale", 0.4)), deadline_s / ref_deadline)
        fps *= scale
    if step > 0:
        fps = math.floor(fps / step + 1e-9) * step
    fps = max(min_fps, min(base, fps))
    return round(fps, 3)


class TimeBudget:
    """Учёт времени одной задачи. Все решения детерминированы относительно часов ``clock``."""

    def __init__(self, deadline_s: float, cfg: Mapping, *, n_frames: Optional[int] = None,
                 device: str = "cpu", clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.t0 = clock()
        self.deadline_s = float(deadline_s)
        self.cfg = cfg
        self.device = device
        self.slowness = 1.0           # EMA отношения факт/бюджет по завершённым стадиям
        self._n_frames = n_frames

    # -------------------------------------------------------------- базовое
    def elapsed(self) -> float:
        return self.clock() - self.t0

    def remaining(self) -> float:
        return self.deadline_s - self.elapsed()

    def set_frames(self, n_frames: Optional[int]) -> None:
        self._n_frames = n_frames

    # -------------------------------------------------------------- бюджеты
    def scale(self) -> float:
        ref = float(cfg_get(self.cfg, "reference_frames", 725)) or 725.0
        if not self._n_frames:
            return 1.0
        return max(0.05, self._n_frames / ref)

    def budget(self, stage: str) -> float:
        """Мягкий бюджет стадии с учётом длины видео, устройства и наблюдаемой «медленности»."""
        base = float(cfg_get(self.cfg, f"budgets_s.{stage}", 5.0))
        factor = self.scale() * self.slowness
        if self.device != "cuda" and stage in (cfg_get(self.cfg, "gpu_stages", []) or []):
            factor *= float(cfg_get(self.cfg, "timeouts.cpu_factor", 4.0))
        return base * factor

    def reserve_after(self, stage: str) -> float:
        """Сколько времени оставить после стадии на финальное слияние. Обязательным стадиям — минимум
        (без них результат всё равно неполный), дополнительным — полный резерв."""
        if stage == "fusion":
            return 0.0
        from .stages import is_required

        if is_required(stage, self.cfg):
            return float(cfg_get(self.cfg, "reserve_s.fusion_min", 3))
        return float(cfg_get(self.cfg, "reserve_s.fusion", 12))

    def timeout(self, stage: str) -> float:
        """Жёсткий таймаут: max(бюджет × hard_factor, min_s), но не дальше дедлайна минус резерв."""
        hard = max(self.budget(stage) * float(cfg_get(self.cfg, "timeouts.hard_factor", 4.0)),
                   float(cfg_get(self.cfg, "timeouts.min_s", 20.0)))
        avail = self.remaining() - self.reserve_after(stage)
        if stage == "fusion":
            avail = max(avail, 3.0)  # слиянию всегда даём шанс, даже за дедлайном
        return max(0.0, min(hard, avail))

    def can_start(self, stage: str) -> tuple[bool, str]:
        """Решение «запускать ли стадию сейчас» + причина пропуска (по-русски, для meta.warnings)."""
        avail = self.remaining() - self.reserve_after(stage)
        if stage == "fusion":
            return True, ""
        optional = stage in (cfg_get(self.cfg, "optional_stages", []) or [])
        if optional:
            need = self.budget(stage) * float(cfg_get(self.cfg, "min_start_fraction", 0.6))
            if stage.startswith("vlm"):
                need = max(need, float(cfg_get(self.cfg, "reserve_s.vlm_min_budget", 6)))
            if avail < need:
                return False, (f"{stage}: пропущена по бюджету времени (доступно {avail:.1f} с, "
                               f"нужно ≈{need:.1f} с)")
            return True, ""
        if avail <= float(cfg_get(self.cfg, "reserve_s.essential_min", 1.5)):
            return False, f"{stage}: пропущена — до дедлайна осталось {self.remaining():.1f} с"
        return True, ""

    def stage_budget_for_call(self, stage: str) -> float:
        """budget_s, передаваемый в ml.vlm.verify/discover: не больше остатка минус резерв."""
        avail = self.remaining() - self.reserve_after(stage)
        return max(0.0, min(self.budget(stage), avail))

    def observe(self, stage: str, seconds: float) -> None:
        """Обновляет оценку «медленности» по факту выполнения стадии (только тяжёлые стадии)."""
        if stage not in ("detect", "pose", "temporal") or seconds <= 0:
            return
        base = self.budget(stage) / max(self.slowness, 1e-6)
        if base <= 0:
            return
        ratio = min(4.0, max(0.25, seconds / base))
        self.slowness = 0.5 * self.slowness + 0.5 * ratio

    def near_deadline(self) -> bool:
        return self.remaining() < float(cfg_get(self.cfg, "partial.margin_s", 25))

    def snapshot(self) -> dict:
        return {"deadline_s": self.deadline_s, "elapsed_s": round(self.elapsed(), 3),
                "remaining_s": round(self.remaining(), 3), "slowness": round(self.slowness, 3),
                "n_frames": self._n_frames}
