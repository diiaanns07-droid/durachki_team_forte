"""read_types_csv: подтверждённые человеком строки — эталон, предложения VLM — нет; явная политика для старого CSV."""
import pytest

from eval.labels import read_types_csv


def _write(tmp_path, text):
    p = tmp_path / "labels_types.csv"
    p.write_text(text, encoding="utf-8")
    return p


def test_only_confirmed_rows_become_ground_truth(tmp_path):
    p = _write(tmp_path, "video_id,t_sec,type_id,confirmed\nX-1,5,V01,1\nX-1,9,V02,0\nX-2,3,V05,\n")
    assert read_types_csv(p) == {"X-1": [(5.0, "V01")]}


def test_legacy_csv_policy_is_explicit(tmp_path):
    p = _write(tmp_path, "video_id,t_sec,type_id\nX-1,5,V01\n")
    assert read_types_csv(p) == {"X-1": [(5.0, "V01")]}
    assert read_types_csv(p, legacy="reject") == {}
    with pytest.raises(ValueError):
        read_types_csv(p, legacy="maybe")
