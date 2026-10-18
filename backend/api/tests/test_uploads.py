"""Приём файла: стриминг, лимиты, сигнатура, настоящий декодер."""
from __future__ import annotations

import io

import pytest

from backend.api import uploads
from backend.api.uploads import UploadError, normalize_filename, sniff_isobmff, validate_video

from .conftest import FTYP_ONLY, H, ok_run, spec, upload, wait_job


def test_415_on_txt_and_non_video(client_factory):
    with client_factory(spec(ok_run)) as c:
        assert upload(c, b"hello world", name="notes.txt").status_code == 415
        assert upload(c, b"hello world, not a movie at all", name="fake.mov").status_code == 415
        assert upload(c, b"\x89PNG\r\n\x1a\n" + b"\x00" * 100, name="x.mp4").status_code == 415
        assert upload(c, FTYP_ONLY, name="clip.avi").status_code == 415
        r = c.post("/api/v1/analyze", headers={**H, "Content-Type": "text/plain"}, content=b"abc")
        assert r.status_code == 415
        assert c.get("/api/v1/jobs", headers=H).json()["jobs"] == []  # ничего не создано


def test_400_empty_and_missing_field(client_factory):
    with client_factory(spec(ok_run)) as c:
        assert upload(c, b"", name="empty.mov").status_code == 400
        r = c.post("/api/v1/analyze", headers=H, data={"note": "x"},
                   files={"other": ("", b"", "text/plain")})
        assert r.status_code in (400, 415)
        r = c.post("/api/v1/analyze", headers=H, data={"note": "x"})
        assert r.status_code in (400, 415)


def test_413_too_large(client_factory):
    with client_factory(spec(ok_run), max_upload_mb=0.01) as c:  # ~10 КБ
        big = FTYP_ONLY + b"\x00" * 20000
        assert upload(c, big).status_code == 413
        assert list((c.app.state.ctx.s.jobs_dir).iterdir()) == []  # мусор убран


def test_streaming_large_file_written_to_disk(client_factory):
    payload = FTYP_ONLY + b"\x00" * (3 * 1024 * 1024 + 123)  # > нескольких чанков flush
    with client_factory(spec(ok_run)) as c:
        r = upload(c, payload)
        assert r.status_code == 200
        row = c.app.state.ctx.store.get(r.json()["job_id"])
        assert row["size_bytes"] == len(payload)
        import hashlib
        assert row["sha256"] == hashlib.sha256(payload).hexdigest()


def test_raw_body_upload(client_factory):
    with client_factory(spec(ok_run)) as c:
        r = c.post("/api/v1/analyze", headers={**H, "Content-Type": "video/quicktime",
                                               "X-Filename": "X-9.mov"}, content=FTYP_ONLY)
        assert r.status_code == 200
        assert wait_job(c, r.json()["job_id"])["status"] == "done"


def test_valid_mov_is_decoded(client_factory, mov_bytes):
    """Настоящий MOV (cv2.VideoWriter) проходит декодер; метаданные — от декодера."""
    with client_factory(spec(ok_run), video_validation="required") as c:
        r = upload(c, mov_bytes, name="X-1.mov")
        assert r.status_code == 200, r.text
        d = wait_job(c, r.json()["job_id"])
        v = d["meta"]["api"]["video"]
        assert v["validated"] is True and v["decoder"] in ("cv2", "pyav", "ffprobe")
        assert abs(v["duration"] - 2.0) < 0.2 and v["decoded_width"] == 232


def test_ftyp_only_is_rejected_by_decoder(client_factory):
    pytest.importorskip("cv2")
    with client_factory(spec(ok_run), video_validation="required") as c:
        r = upload(c, FTYP_ONLY)
        assert r.status_code == 415 and "decodable" in r.json()["detail"]


def test_no_decoder_required_is_503(client_factory, monkeypatch):
    monkeypatch.setattr(uploads, "DEFAULT_PROBES", ())
    with client_factory(spec(ok_run), video_validation="required") as c:
        assert upload(c, FTYP_ONLY).status_code == 503
    with client_factory(spec(ok_run), video_validation="best_effort") as c:
        assert upload(c, FTYP_ONLY).status_code == 200


def test_sniff_and_probe(tmp_path, mov_bytes):
    assert sniff_isobmff(FTYP_ONLY)
    assert sniff_isobmff(b"\x00\x00\x00\x08wide\x00\x00\x10\x00mdat")  # QuickTime без ftyp
    assert not sniff_isobmff(b"hello world....")
    assert not sniff_isobmff(b"\x00\x00\x00")
    p = tmp_path / "a.mov"
    p.write_bytes(mov_bytes)
    info = uploads.probe_mp4(p)
    assert abs(info["duration"] - 2.0) < 0.2 and info["width"] == 232 and info["height"] == 424
    with pytest.raises(UploadError) as e:
        q = tmp_path / "b.mov"
        q.write_bytes(FTYP_ONLY)
        validate_video(q, "required")
    assert e.value.status == 415


def test_filename_normalization():
    assert normalize_filename("Х-3.mov") == "X-3.mov"           # кириллица -> латиница
    assert normalize_filename("C:\\fakepath\\X-1.MOV") == "X-1.MOV"
    assert normalize_filename("../../etc/passwd") == "passwd"
