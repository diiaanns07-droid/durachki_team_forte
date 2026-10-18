"""JSON-(де)сериализация датаклассов контракта для кэша, partial.json и --dump-intermediate."""
from __future__ import annotations

import dataclasses
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

from .contract import (AnalysisResult, Detection, Event, EvidenceFrame, PoseFrame, RuleRef,
                       SceneState, Track, Violation, Zone)


def to_jsonable(obj: Any) -> Any:
    """Рекурсивно приводит объект к JSON-совместимому виду (numpy, кортежи, датаклассы, множества)."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        f = float(obj)
        return f if math.isfinite(f) else None
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    tolist = getattr(obj, "tolist", None)  # numpy-массивы и скаляры
    if callable(tolist):
        return to_jsonable(tolist())
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return to_jsonable(item())
        except Exception:  # noqa: BLE001
            pass
    return str(obj)


def dumps(obj: Any, **kw: Any) -> str:
    return json.dumps(to_jsonable(obj), ensure_ascii=False, **kw)


def _bbox(v: Any) -> tuple:
    return tuple(float(x) for x in v)


def _opt_int(v: Any):
    return None if v is None else int(v)


def load_detection(d: dict) -> Detection:
    return Detection(t=float(d["t"]), label=str(d["label"]), bbox=_bbox(d["bbox"]),
                     score=float(d["score"]), track_id=_opt_int(d.get("track_id")),
                     source=str(d.get("source", "")))


def load_track(d: dict) -> Track:
    return Track(track_id=int(d["track_id"]), label=str(d["label"]), role=d.get("role", "unknown"),
                 role_conf=float(d.get("role_conf", 0.0)),
                 boxes=[(float(t), _bbox(b)) for t, b in d.get("boxes", [])])


def load_zone(d: dict) -> Zone:
    return Zone(name=str(d["name"]), polygon=[(float(x), float(y)) for x, y in d.get("polygon", [])],
                conf=float(d.get("conf", 1.0)))


def load_pose(d: dict) -> PoseFrame:
    return PoseFrame(t=float(d["t"]), track_id=_opt_int(d.get("track_id")),
                     keypoints=[tuple(float(v) for v in kp) for kp in d.get("keypoints", [])])


def load_state(d: dict) -> SceneState:
    return SceneState(t=float(d["t"]), signals={str(k): float(v) for k, v in (d.get("signals") or {}).items()
                                                if v is not None})


def load_event(d: dict) -> Event:
    return Event(type_id=str(d["type_id"]), t_start=float(d["t_start"]), t_end=float(d["t_end"]),
                 t_peak=float(d["t_peak"]), confidence=float(d["confidence"]),
                 source=str(d.get("source", "")), facts=[str(f) for f in d.get("facts", [])],
                 bboxes={str(k): _bbox(v) for k, v in (d.get("bboxes") or {}).items()},
                 track_ids=[int(x) for x in d.get("track_ids", [])],
                 extra=dict(d.get("extra") or {}))


def load_risk(d: Any) -> list[tuple[float, float]]:
    return [(float(t), float(r)) for t, r in (d or []) if t is not None and r is not None]


def load_rule(d: dict) -> RuleRef:
    return RuleRef(document=str(d.get("document", "")), clause=d.get("clause"), url=str(d.get("url", "")),
                   quote=str(d.get("quote", "")), verified=bool(d.get("verified", False)))


def load_violation(d: dict) -> Violation:
    return Violation(type=str(d["type"]), timecode_sec=float(d["timecode_sec"]),
                     description=str(d.get("description", "")),
                     evidence_frames=[EvidenceFrame(float(e["timestamp_sec"]), str(e["image_base64"]))
                                      for e in d.get("evidence_frames", [])],
                     type_id=str(d.get("type_id", "")), confidence=float(d.get("confidence", 0.0)),
                     t_start=float(d.get("t_start", 0.0)), t_end=float(d.get("t_end", 0.0)),
                     rules=[load_rule(r) for r in d.get("rules", [])])


def load_result(d: dict) -> AnalysisResult:
    return AnalysisResult(job_id=str(d["job_id"]), status=d["status"], verdict=d.get("verdict"),
                          violations=[load_violation(v) for v in d.get("violations", [])],
                          timeline=load_risk(d.get("timeline")), meta=dict(d.get("meta") or {}))


LOADERS: dict[str, Callable[[dict], Any]] = {
    "detection": load_detection, "track": load_track, "zone": load_zone, "pose": load_pose,
    "state": load_state, "event": load_event,
    "obs": dict,   # наблюдения ml.seqrules — JSON-словари как есть (meta + эпизоды)
}


def roundtrip_list(items: Iterable[Any], kind: str) -> list:
    """JSON-roundtrip списка датаклассов: «живой» выход стадии ≡ выходу, прочитанному из кэша."""
    loader = LOADERS[kind]
    return [loader(json.loads(json.dumps(to_jsonable(x)))) for x in items]


def roundtrip_risk(risk: Iterable[Any]) -> list[tuple[float, float]]:
    return load_risk(json.loads(json.dumps(to_jsonable(list(risk)))))


# ---------------------------------------------------------------- атомарная запись файлов

def atomic_write_text(path: str | os.PathLike, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_json(path: str | os.PathLike, obj: Any, indent: int | None = None) -> None:
    atomic_write_text(path, dumps(obj, indent=indent))


def write_jsonl(path: str | os.PathLike, items: Iterable[Any]) -> None:
    atomic_write_text(path, "".join(dumps(x) + "\n" for x in items))


def read_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_jsonl(path: str | os.PathLike) -> list:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------- результат → API

def result_to_dict(result: AnalysisResult) -> dict:
    """Схема ТЗ + доп. поля. Если есть contracts.serialize.result_to_api (A01) — используется он."""
    try:
        from contracts.serialize import result_to_api  # type: ignore
    except ImportError:
        result_to_api = None
    if result_to_api is not None:
        try:
            return to_jsonable(result_to_api(result))
        except Exception:  # noqa: BLE001 — не даём чужому сериализатору уронить задачу
            pass
    return to_jsonable(result)
