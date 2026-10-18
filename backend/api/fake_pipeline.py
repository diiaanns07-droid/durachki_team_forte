"""Фейковый пайплайн для изолированных тестов и демо UI.
Включается ТОЛЬКО явно (SENTINEL_FAKE_PIPELINE=1). Никогда не подставляется автоматически:
API жюри без настоящего backend.pipeline отвечает readyz=503, а задачи — failed."""
from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Callable

# 64x36 серый JPEG — последний вариант, если нет ни cv2, ни Pillow
_PLACEHOLDER_JPEG = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAoHBwgHBgoICAgLCgoLDhgQDg0NDh0VFhEYIx8lJCIfIiEmKzcvJik0KSEi"
    "MEExNDk7Pj4+JS5ESUM8SDc9Pjv/2wBDAQoLCw4NDhwQEBw7KCIoOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7"
    "Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozv/wAARCAAkAEADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcI"
    "CQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRol"
    "JicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ip"
    "qrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAA"
    "AAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLR"
    "ChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaX"
    "mJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEA"
    "PwDlbW1SeKWSSbykixk7d3Wn/ZrH/oIf+QWotv8AkG3v/bP+dU6lKUpPX8uy8jpbhThD3E21fW/dro12Ln2ax/6CH/kF"
    "qPs1j/0EP/ILVToquSX8z/D/ACI9tD/n2v8Ayb/5IufZrH/oIf8AkFqPs1j/ANBD/wAgtVOijkl/M/w/yD20P+fa/wDJ"
    "v/ki59msf+gh/wCQWpl1apBFFJHN5qS5wdu3pVarlz/yDbL/ALafzqWpRktfy7PyLThUhP3EmlfS/dLq33C2/wCQbe/9"
    "s/51Tq/YxPPY3kca7nbZgZx3NM/sm+/54f8Aj6/40lOMZSTfX9EVOjVqU6bhFtW6L+8ynRVz+yb7/nh/4+v+NH9k33/P"
    "D/x9f8av2tP+ZfeZfVa/8j+5lOirn9k33/PD/wAfX/Gj+yb7/nh/4+v+NHtaf8y+8Pqtf+R/cynVy5/5Btl/20/nR/ZN"
    "9/zw/wDH1/xp99E8FjaRyLtdd+RnPcVDnGUopPr+jNYUatOnUc4tK3Vf3kUKKKK3OIKKKKACiiigAooooA//2Q==")


def _placeholder(t: float) -> str:
    try:
        from PIL import Image, ImageDraw  # lazy
        import io
        im = Image.new("RGB", (464, 848), (52, 56, 64))
        d = ImageDraw.Draw(im)
        d.rectangle([40, 300, 424, 640], outline=(230, 80, 60), width=6)
        d.text((60, 320), f"FAKE PIPELINE  t={t:.1f}s", fill=(240, 240, 240))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return _PLACEHOLDER_JPEG


def _catalog_rules(type_id: str) -> list[Any]:
    """Нормы только из norms/catalog.yaml (A13). Нет каталога — пустой список, ничего не выдумываем."""
    try:
        from norms import load_catalog  # type: ignore
        entry = (load_catalog().get("types") or {}).get(type_id) or {}
        name = entry.get("name")
        return [entry.get("rules") or [], name]  # type: ignore[list-item]
    except Exception:
        return [[], None]


def make_fake_runner(seconds: float = 3.0) -> Callable[..., Any]:
    def run(video_path: str, job_id: str, progress: Callable[[str, float], None],
            deadline_s: float = 240) -> Any:
        from .results import grab_frame_at
        from .uploads import probe_mp4
        try:
            from contracts.sentinel_contracts import (  # type: ignore
                AnalysisResult, EvidenceFrame, RuleRef, Violation)
        except ImportError:
            from contracts.sentinel_contracts import AnalysisResult, EvidenceFrame, RuleRef, Violation

        total = max(0.0, min(float(seconds), max(0.0, float(deadline_s)) * 0.5))
        stages = ["decode", "detect", "track", "rules", "fusion"]
        for i, st in enumerate(stages):
            progress(st, i / len(stages))
            time.sleep(total / len(stages))
        duration = float(probe_mp4(Path(video_path)).get("duration") or 10.0)
        t = round(max(0.0, duration * 0.4), 2)
        g = grab_frame_at(Path(video_path), t)
        if g.ok:  # фактический PTS кадра, а не запрошенное время
            t, img = float(g.pts if g.pts is not None else t), g.jpeg_b64 or ""
        else:
            img = _placeholder(t)  # фейк: явно синтетическая картинка с надписью FAKE
        rules_raw, name = _catalog_rules("V01")
        rules = []
        for r in rules_raw:
            if isinstance(r, dict) and r.get("url"):
                rules.append(RuleRef(document=str(r.get("document", "")), clause=r.get("clause"),
                                     url=str(r["url"]), quote=str(r.get("quote", "")),
                                     verified=bool(r.get("verified", False))))
        v = Violation(
            type=name or "Посторонний в кассовом помещении",
            timecode_sec=t,
            description=("ТЕСТОВЫЙ РЕЗУЛЬТАТ (фейковый пайплайн, не анализ видео). "
                         f"Пример формата: на {t:.1f} с признаки возможного нарушения."),
            evidence_frames=[EvidenceFrame(timestamp_sec=t, image_base64=img)],
            type_id="V01", confidence=0.5, t_start=max(0.0, t - 1.0), t_end=t + 1.0, rules=rules)
        progress("done", 1.0)
        timeline = [(round(x * duration / 20, 2), 0.9 if abs(x * duration / 20 - t) < 1.5 else 0.1)
                    for x in range(21)]
        return AnalysisResult(job_id=job_id, status="done", verdict="violation", violations=[v],
                              timeline=timeline,
                              meta={"fake_pipeline": True, "version": "fake-0",
                                    "warnings": ["FAKE PIPELINE: result is synthetic"]})
    return run
