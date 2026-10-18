"""Кадры-доказательства только по PTS с допуском; готовность (/readyz) по итогу прогрева пайплайна;
разные профили ответа жюри и UI без потери обязательных полей. Только синтетика."""
from __future__ import annotations

import base64
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from backend.api import results as R
from backend.api.results import enforce_evidence, fill_missing_evidence, grab_frame_at
from backend.api.worker import PipelineSpec, warmup_problem

from .conftest import (FTYP_ONLY, H, KEY, Violation, make_result, ok_run, spec, upload, wait_job,
                       wait_ready, write_mov, write_vfr_mov)

TZ_TOP = {"job_id", "status", "verdict", "violations"}
TZ_VIOLATION = {"type", "timecode_sec", "description", "evidence_frames"}


def _pixel(b64: str) -> float:
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    return float(img[img.shape[0] // 2, 10].mean())


# ------------------------------------------------------------------ реальные синтетические ролики
def test_valid_frame_has_actual_pts(tmp_path):
    p = write_mov(tmp_path / "v.mov", seconds=3.0)
    for t, k in ((0.0, 0), (1.0, 30), (1.51, 45), (2.94, 88)):
        g = grab_frame_at(p, t)
        assert g.ok and g.error is None
        assert g.pts == pytest.approx(k / 30, abs=1e-3), (t, g.pts)   # фактический PTS кадра
        raw = base64.b64decode(g.jpeg_b64)
        assert raw[:3] == b"\xff\xd8\xff" and raw[-2:] == b"\xff\xd9"


def test_vfr_frame_selected_by_pts_not_by_fps(tmp_path):
    """PTS: 0..0.967 шаг 1/30, затем 1.0, 1.25, ... 8.25. fps-предположение дало бы другой кадр."""
    p = write_vfr_mov(tmp_path / "vfr.mov")
    g = grab_frame_at(p, 3.0)                       # кадр 30 + 8 = 38 -> яркость 152
    assert g.ok and g.pts == pytest.approx(3.0, abs=1e-3)
    assert abs(_pixel(g.jpeg_b64) - (38 * 4) % 255) < 12
    g = grab_frame_at(p, 6.0)                       # ветка с seek: кадр 30 + 20 = 50
    assert g.ok and g.pts == pytest.approx(6.0, abs=1e-3)
    assert abs(_pixel(g.jpeg_b64) - (50 * 4) % 255) < 12
    g = grab_frame_at(p, 1.1)                       # ближайший кадр 1.0 в пределах допуска
    assert g.ok and g.pts == pytest.approx(1.0, abs=1e-3)
    g = grab_frame_at(p, 1.12, tolerance_s=0.05)    # ни одного кадра ближе 0.05 с
    assert not g.ok and g.error == "too_far" and g.jpeg_b64 is None


def test_unavailable_timecode_is_explicit_error(tmp_path, monkeypatch):
    p = write_mov(tmp_path / "two.mov", seconds=2.0)
    g = grab_frame_at(p, 60.0)                      # аудит: раньше — первый кадр с меткой 60 с
    assert not g.ok and g.error == "out_of_range" and g.jpeg_b64 is None and g.pts is None
    assert R.extract_jpeg(p, 60.0) is None
    for bad in (-1.0, float("nan"), float("inf")):
        assert grab_frame_at(p, bad).error == "out_of_range"
    monkeypatch.setattr(R, "_probe_duration", lambda path: None)  # длительность неизвестна
    g = grab_frame_at(p, 5.0)
    assert not g.ok and g.error == "out_of_range" and "ends at PTS" in g.detail


# ------------------------------------------------------------------ сбои декодера (заглушка)
class FakeCap:
    """Мини-VideoCapture: заданная последовательность PTS (с), управляемый seek."""

    def __init__(self, pts: list[Any], seek_ok: bool = True, seek_to: int | None = None,
                 opened: bool = True):
        self.pts, self.i, self.seek_ok, self.seek_to, self.opened = pts, -1, seek_ok, seek_to, opened

    def isOpened(self) -> bool:
        return self.opened

    def set(self, prop: int, value: float) -> bool:
        if not self.seek_ok:
            return False
        if self.seek_to is not None:
            self.i = self.seek_to - 1
        return True

    def grab(self) -> bool:
        if self.i + 1 >= len(self.pts):
            return False
        self.i += 1
        return True

    def get(self, prop: int) -> float:
        v = self.pts[self.i]
        return v * 1000.0 if isinstance(v, (int, float)) else v

    def retrieve(self) -> tuple[bool, Any]:
        import numpy as np
        return True, np.full((48, 32, 3), int(self.pts[self.i] * 10) % 255, np.uint8)

    def release(self) -> None:
        pass


def _factory(*caps: FakeCap):
    it = iter(caps)
    return lambda path: next(it)


CFR = [k / 30 for k in range(300)]  # 10 с


def test_bad_pts_is_explicit_error(tmp_path):
    pytest.importorskip("cv2")
    p = tmp_path / "x.mov"
    cases = {
        "nan": [0.0, 0.033, float("nan"), 0.1],
        "negative": [0.0, -2.0, 0.1],
        "backwards": [0.0, 0.5, 0.9, 0.2, 0.3, 1.0],   # скачок назад до нужного момента
        "stuck": [0.0] * 10,
    }
    for name, pts in cases.items():
        g = grab_frame_at(p, 1.0, duration=10.0, capture_factory=_factory(FakeCap(pts)))
        assert not g.ok and g.error == "bad_pts" and g.jpeg_b64 is None, name


def test_failed_seek_never_returns_wrong_frame(tmp_path):
    pytest.importorskip("cv2")
    p = tmp_path / "x.mov"
    # seek отклонён -> честный проход с начала по PTS -> кадр 6.0 с фактическим PTS
    g = grab_frame_at(p, 6.0, duration=10.0, capture_factory=_factory(FakeCap(CFR, seek_ok=False),
                                                                      FakeCap(CFR)))
    assert g.ok and g.pts == pytest.approx(6.0, abs=1e-3) and "seek rejected" in g.detail
    # seek «перескочил» за t (кадр 9 с), а переоткрыть файл нельзя -> ошибка, а не кадр 9 с
    g = grab_frame_at(p, 6.0, duration=10.0, capture_factory=_factory(FakeCap(CFR, seek_to=270),
                                                                      FakeCap(CFR, opened=False)))
    assert not g.ok and g.error == "seek_failed" and g.jpeg_b64 is None
    # seek отклонён и после него не декодируется ни один кадр
    g = grab_frame_at(p, 6.0, duration=10.0, capture_factory=_factory(FakeCap(CFR, seek_ok=False),
                                                                      FakeCap([])))
    assert not g.ok and g.error == "seek_failed"


def test_exhausted_budget_is_explicit_error(tmp_path):
    pytest.importorskip("cv2")
    p = tmp_path / "x.mov"
    ticks = iter(range(10_000))
    g = grab_frame_at(p, 8.0, duration=10.0, time_budget_s=5.0, pre_roll_s=100.0,
                      capture_factory=_factory(FakeCap(CFR)), clock=lambda: float(next(ticks)))
    assert not g.ok and g.error == "budget_exhausted" and g.jpeg_b64 is None
    # общий бюджет задачи уже потрачен: кадр не ищется, нарушение без кадра
    d = {"verdict": "violation", "violations": [{"type": "x", "timecode_sec": 1.0, "description": "",
                                                 "evidence_frames": []}]}
    w = fill_missing_evidence(d, p, time_budget_s=0.0)
    assert d["violations"][0]["evidence_frames"] == []
    assert d["violations"][0]["evidence_error"].startswith("budget_exhausted") and w


def test_enforce_evidence_never_turns_loss_into_no_violation():
    frame = {"timestamp_sec": 1.0, "image_base64": "x"}
    d = {"verdict": "violation", "violations": [
        {"type": "a", "timecode_sec": 1.0, "description": "", "evidence_frames": [frame]},
        {"type": "b", "timecode_sec": 99.0, "description": "", "evidence_frames": [],
         "evidence_error": "out_of_range: beyond duration"}]}
    warns, problem = enforce_evidence(d)
    assert problem is None and [v["type"] for v in d["violations"]] == ["a"]
    assert [c["type"] for c in d["candidates"]] == ["b"] and d["verdict"] == "violation" and warns
    d = {"verdict": "violation", "violations": [{"type": "b", "timecode_sec": 99.0, "description": "",
                                                 "evidence_frames": []}]}
    _, problem = enforce_evidence(d)
    assert problem and "not reported as no_violation" in problem and d["violations"] == []
    assert d["verdict"] == "violation"  # вердикт не подменяется; задачу закрывает воркер как failed


# ------------------------------------------------------------------ через API
def _run_with(violations_tc: list[float]):
    def run(video_path, job_id, progress, deadline_s=240):
        r = make_result(job_id)
        r.violations = [Violation(type=f"V{i}", timecode_sec=t, description="d", evidence_frames=[],
                                  type_id="V01", confidence=0.7) for i, t in enumerate(violations_tc)]
        return r
    return run


def test_api_evidence_from_video_with_actual_pts(client_factory, mov_bytes):
    with client_factory(spec(_run_with([1.01])), video_validation="required") as c:
        d = wait_job(c, upload(c, mov_bytes).json()["job_id"])
        assert d["status"] == "done"
        f = d["violations"][0]["evidence_frames"][0]
        assert f["timestamp_sec"] == pytest.approx(1.0, abs=1e-3)  # PTS кадра, не 1.01
        assert any("PTS 1.000s" in w for w in d["meta"]["warnings"])


def test_api_violation_without_obtainable_frame_is_failed_not_clean(client_factory, mov_bytes):
    with client_factory(spec(_run_with([60.0])), video_validation="required") as c:
        jid = upload(c, mov_bytes).json()["job_id"]
        d = wait_job(c, jid)
        assert d["status"] == "failed" and d["verdict"] is None and d["violations"] == []
        assert TZ_TOP <= set(d) and "evidence_missing" in d["error"] and "out_of_range" in d["error"]
        c.post("/ui/v1/session", json={"api_key": KEY})
        u = c.get(f"/ui/v1/jobs/{jid}").json()
        cand = u["partial"]["rejected_result"]["candidates"]
        assert cand[0]["timecode_sec"] == 60.0 and cand[0]["evidence_error"].startswith("out_of_range")


def test_api_mixed_evidence_keeps_only_backed_violations(client_factory, mov_bytes):
    with client_factory(spec(_run_with([1.0, 60.0])), video_validation="required") as c:
        jid = upload(c, mov_bytes).json()["job_id"]
        d = wait_job(c, jid)
        assert d["status"] == "done" and d["verdict"] == "violation"
        assert [v["timecode_sec"] for v in d["violations"]] == [1.0]
        assert all(v["evidence_frames"] for v in d["violations"]) and "candidates" not in d
        c.post("/ui/v1/session", json={"api_key": KEY})
        u = c.get(f"/ui/v1/jobs/{jid}").json()
        assert [x["timecode_sec"] for x in u["candidates"]] == [60.0]


# ------------------------------------------------------------------ готовность и прогрев A03
@pytest.mark.parametrize("ret,report,ready", [
    ({"hooks_error": "CUDA out of memory"}, None, False),
    ({"synthetic": {"status": "failed"}}, None, False),
    ({"ok": False, "error": "weights missing"}, None, False),
    (False, None, False),
    (None, {"synthetic_error": "RuntimeError: no weights"}, False),   # как хранит A03 на 2055fc4
    (None, {"hooks": {"ml.detect": "ok"}, "synthetic": {"status": "done"}, "seconds": 3.1}, True),
    ({"ok": True, "synthetic": {"status": "done"}}, None, True),
    (None, None, True),                                                  # отчёта нет — совместимость
])
def test_readyz_follows_warmup_outcome(client_factory, ret, report, ready):
    sp = PipelineSpec(run=ok_run, mode="real", warmup=lambda: ret, warmup_report=lambda: report)
    with client_factory(sp) as c:
        r = wait_ready(c)
        assert (r.status_code == 200) is ready, r.json()
        assert r.json()["state"] == ("ready" if ready else "warmup_failed")
        if not ready:
            assert r.json()["detail"]


def test_readyz_reads_a03_registry_report(client_factory, monkeypatch):
    """Модуль как A03 на 2055fc4: warmup() -> None, отчёт в ModelRegistry.instance().warmup_report."""
    mod = types.ModuleType("fake_a03_pipeline")
    registry = types.SimpleNamespace(warmup_report={})

    class ModelRegistry:
        @staticmethod
        def instance():
            return registry

    def warmup():
        registry.warmup_report = {"hooks_error": "ImportError: ml.detect", "seconds": 1.0}
    mod.run, mod.warmup, mod.ModelRegistry, mod.PIPELINE_VERSION = ok_run, warmup, ModelRegistry, "t"
    monkeypatch.setitem(sys.modules, "fake_a03_pipeline", mod)
    with client_factory(None, pipeline="fake_a03_pipeline") as c:
        r = wait_ready(c)
        assert r.status_code == 503 and r.json()["state"] == "warmup_failed"
        assert "ml.detect" in r.json()["detail"]

    def good_warmup():
        registry.warmup_report = {"synthetic": {"status": "done"}, "seconds": 1.0}
    mod.warmup = good_warmup
    with client_factory(None, pipeline="fake_a03_pipeline") as c:
        assert wait_ready(c).status_code == 200


def test_warmup_problem_unit():
    assert warmup_problem(None) is None and warmup_problem({}) is None
    assert warmup_problem({"errors": ["x"]}) and warmup_problem({"status": "error"})
    assert warmup_problem({"synthetic": {"status": "done"}}) is None


# ------------------------------------------------------------------ профили жюри и UI
def test_jury_and_ui_profiles_differ_and_keep_required_fields(client_factory, mov_bytes):
    run = _run_with([1.0, 60.0])  # одно нарушение с кадром, одно без -> candidates в UI
    for prof in ("extended", "strict"):
        with client_factory(spec(run), video_validation="required", jury_profile=prof) as c:
            jid = upload(c, mov_bytes).json()["job_id"]
            jury = wait_job(c, jid)
            c.post("/ui/v1/session", json={"api_key": KEY})
            ui = c.get(f"/ui/v1/jobs/{jid}").json()
            for d in (jury, ui):
                assert TZ_TOP <= set(d) and d["status"] == "done" and d["verdict"] == "violation"
                for v in d["violations"]:
                    assert TZ_VIOLATION <= set(v) and 1 <= len(v["evidence_frames"]) <= 3
            assert "candidates" in ui and "candidates" not in jury
            assert not ({"verdict_state", "analysis_complete", "partial", "warnings"} & set(jury))
            if prof == "strict":
                assert set(jury) == TZ_TOP and set(jury["violations"][0]) == TZ_VIOLATION
            else:
                assert "type_id" in jury["violations"][0] and "meta" in jury
            assert jury != ui
