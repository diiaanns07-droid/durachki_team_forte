"""Kassa Sentinel — замороженный контракт v1. НЕ РЕДАКТИРОВАТЬ."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 — нормировано 0..1 к кадру ПОСЛЕ учёта поворота

class VideoClip(Protocol):
    path: str
    duration: float          # сек
    fps: float               # исходный fps
    width: int               # после учёта rotation
    height: int
    times: list[float]       # таймстемпы сэмплированных кадров (по умолчанию 5 fps), сек от начала
    def frame(self, i: int) -> Any: ...        # np.ndarray BGR uint8, i — индекс в times
    def frame_at(self, t: float) -> Any: ...   # точный кадр, ближайший к t (для доказательств)

LABELS = ["person", "cell_phone", "cup", "bottle", "food", "cash", "money_counter", "laptop",
          "safe", "safe_open", "door", "door_open", "cash_bag", "bag", "document", "keys"]

SIGNALS = ["cashier_present", "cashier_in_workplace", "n_persons", "outsider_in_cashier_area",
           "collector_present", "safe_open", "door_open", "cash_visible", "cash_bag_visible",
           "phone_in_hand", "phone_to_ear", "eating_drinking", "hands_hidden", "cashier_facing_away",
           "workplace_unattended_with_cash", "handover_motion", "motion_energy"]

Role = Literal["cashier", "visitor", "collector", "unknown"]

@dataclass
class Detection:
    t: float
    label: str               # из LABELS
    bbox: BBox
    score: float
    track_id: Optional[int] = None
    source: str = ""

@dataclass
class Track:
    track_id: int
    label: str
    role: Role = "unknown"
    role_conf: float = 0.0
    boxes: list[tuple[float, BBox]] = field(default_factory=list)  # (t, bbox)

@dataclass
class Zone:
    name: str                # cashier_area | desk | safe | door | client_area | cash_tray
    polygon: list[tuple[float, float]]  # нормированные координаты
    conf: float = 1.0

@dataclass
class PoseFrame:
    t: float
    track_id: Optional[int]
    keypoints: list[tuple[float, float, float]]  # COCO-17: (x, y нормированные, score)

@dataclass
class SceneState:
    t: float
    signals: dict[str, float]  # 0..1 (n_persons — целое), имена из SIGNALS

@dataclass
class Event:
    type_id: str             # V01..V08, V99 — ключи norms/catalog.yaml
    t_start: float
    t_end: float
    t_peak: float            # таймкод, который отдаём жюри
    confidence: float        # 0..1
    source: str              # rules | temporal | vlm | fusion
    facts: list[str] = field(default_factory=list)          # наблюдаемые факты по-русски, без домыслов
    bboxes: dict[str, BBox] = field(default_factory=dict)    # что подсветить на кадре t_peak
    track_ids: list[int] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)      # trace, raw scores и т.п.

@dataclass
class EvidenceFrame:
    timestamp_sec: float
    image_base64: str        # JPEG

@dataclass
class RuleRef:
    document: str
    clause: Optional[str]
    url: str
    quote: str
    verified: bool = False

@dataclass
class Violation:
    type: str                # человекочитаемое русское название
    timecode_sec: float
    description: str
    evidence_frames: list[EvidenceFrame]
    type_id: str = ""
    confidence: float = 0.0
    t_start: float = 0.0
    t_end: float = 0.0
    rules: list[RuleRef] = field(default_factory=list)

@dataclass
class AnalysisResult:
    job_id: str
    status: Literal["processing", "done", "failed"]
    verdict: Optional[Literal["violation", "no_violation"]] = None
    violations: list[Violation] = field(default_factory=list)
    timeline: list[tuple[float, float]] = field(default_factory=list)  # (t, risk 0..1)
    meta: dict[str, Any] = field(default_factory=dict)  # timings, models, warnings, version

# Публичные функции модулей (сигнатуры заморожены):
# ml.ingest.load_video(path: str, sample_fps: float = 5.0) -> VideoClip
# ml.detect.detect(clip, every_n: int = 1) -> list[Detection]
# ml.track.track(clip, dets) -> tuple[list[Track], list[Zone]]
# ml.pose.estimate(clip, tracks) -> list[PoseFrame]
# ml.pose.pose_signals(poses, tracks, zones) -> list[SceneState]
# ml.scene.scene_states(clip, dets, tracks, zones, pose_states) -> list[SceneState]
# ml.rules.run_rules(states, tracks, zones, catalog: dict) -> list[Event]
# ml.temporal.score(clip) -> tuple[list[tuple[float, float]], list[Event]]   # (risk curve, events)
# ml.vlm.verify(clip, events, budget_s: float) -> list[Event]
# ml.vlm.discover(clip, budget_s: float) -> list[Event]
# ml.fusion.fuse(clip, events, risk, catalog: dict, job_id: str) -> AnalysisResult
# norms.load_catalog() -> dict          # norms/catalog.yaml
# backend.pipeline.run(video_path: str, job_id: str, progress: Callable[[str, float], None], deadline_s: float = 240) -> AnalysisResult
# backend.pipeline.warmup() -> None
# contracts.serialize.result_to_api(result: AnalysisResult) -> dict    # ровно схема ТЗ + доп. поля
