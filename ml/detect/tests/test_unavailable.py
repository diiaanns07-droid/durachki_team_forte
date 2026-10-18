"""Нет весов → DetectorUnavailable (не mock), без обращений к сети."""
from __future__ import annotations

import pytest

from ml.detect import DetectorUnavailable, available, detect, last_stats, warmup
from ml.detect.synth import synth_clip


def test_missing_weights_raise_unavailable_not_mock(no_network):
    clip, _ = synth_clip(duration_s=1.0)
    with pytest.raises(DetectorUnavailable):
        detect(clip)
    st = last_stats()
    assert st["status"] == "unavailable" and st["mock"] is False
    assert any("not found locally" in w for w in st["warnings"])


def test_missing_weights_empty_mode(no_network):
    clip, _ = synth_clip(duration_s=1.0)
    assert detect(clip, config={"runtime": {"on_unavailable": "empty"}}) == []
    assert last_stats()["status"] == "unavailable"


def test_warmup_and_available_report_unavailable(no_network):
    w = warmup()
    assert w["status"] == "unavailable" and w["backends"] == []
    a = available()
    assert a["ready"] is False and a["mock"] is False


def test_import_has_no_side_effects():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    code = ("import sys, ml.detect; heavy = [m for m in ('torch', 'transformers', 'onnxruntime', 'cv2') "
            "if m in sys.modules]; assert not heavy, heavy; assert 'ml.detect.pipeline' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)
