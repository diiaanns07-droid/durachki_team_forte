import stat
import sys
from pathlib import Path

from deploy import merge_requirements as mr


def _tree(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return tmp_path


def test_intersection_without_picking_versions(tmp_path):
    root = _tree(tmp_path, {
        "ml/a/requirements.txt": "numpy>=1.24  # comment\nPillow\n",
        "ml/b/requirements.txt": "numpy<2\npillow>=10\n",
    })
    code, report, text = mr.run(root, "gpu", tmp_path / "out.in")
    assert code == 0 and not report.conflicts
    assert "numpy<2,>=1.24  # ml/a/requirements.txt, ml/b/requirements.txt" in text
    assert "pillow>=10" in text
    assert (tmp_path / "out.in").read_text(encoding="utf-8") == text


def test_conflict_is_reported_not_resolved(tmp_path):
    root = _tree(tmp_path, {
        "ml/a/requirements.txt": "numpy==1.26.4\n",
        "ml/b/requirements.txt": "numpy>=2.0\n",
    })
    out = tmp_path / "out.in"
    code, report, _ = mr.run(root, "gpu", out)
    assert code == 1
    assert len(report.conflicts) == 1 and "numpy" in report.conflicts[0]
    assert "ml/a/requirements.txt:1" in report.conflicts[0]
    assert not out.exists()  # без --allow-conflicts ничего не пишем


def test_pins_conflict_and_platform_torch(tmp_path):
    root = _tree(tmp_path, {
        "ml/a/requirements.txt": "torch>=2.7\n",
        "deploy/platform.in": "torch==2.6.0\n",
    })
    code, report, _ = mr.run(root, "gpu", None)
    assert code == 1 and "torch" in report.conflicts[0]


def test_compatible_pins(tmp_path):
    root = _tree(tmp_path, {"a/requirements.txt": "transformers>=4.45,<5\n", "b/requirements.txt": "transformers==4.49.0\n"})
    code, report, text = mr.run(root, "gpu", None)
    assert code == 0, report.conflicts
    assert "transformers<5,==4.49.0,>=4.45" in text


def test_exclusive_distributions_profiles(tmp_path):
    root = _tree(tmp_path, {
        "a/requirements.txt": "opencv-python>=4.8\nonnxruntime>=1.18\n",
        "b/requirements.txt": "opencv-contrib-python-headless\nonnxruntime-gpu>=1.19\n",
    })
    code, report, text = mr.run(root, "gpu", None)
    assert code == 0
    assert "opencv-contrib-python-headless>=4.8" in text and "opencv-python>" not in text
    assert "onnxruntime-gpu>=1.18,>=1.19" in text
    code, report, text = mr.run(root, "cpu", None)
    assert "\nonnxruntime>=1.18,>=1.19" in text and "onnxruntime-gpu" not in text.split("\n\n", 1)[1]


def test_banned_license_package(tmp_path):
    root = _tree(tmp_path, {"ml/detect/requirements.txt": "ultralytics==8.3.0\n"})
    code, report, _ = mr.run(root, "gpu", tmp_path / "x.in")
    assert code == 2 and "ultralytics" in report.banned[0]


def test_includes_markers_extras_dev_and_options(tmp_path):
    root = _tree(tmp_path, {
        "a/requirements.txt": "-r base.txt\npytest>=8\npywin32>=306 ; sys_platform == 'win32'\n"
                              "--extra-index-url https://download.pytorch.org/whl/cu124\n"
                              "transformers[torch]>=4.45 \\\n    --hash=sha256:abc\n",
        "a/base.txt": "Transformers[sentencepiece]\n",
    })
    code, report, text = mr.run(root, "gpu", None)
    assert code == 0
    assert "transformers[sentencepiece,torch]>=4.45" in text
    assert 'pywin32>=306 ; sys_platform == "win32"' in text
    assert "pytest" not in text.split("\n\n", 1)[1]
    assert any("pytest" in d for d in report.dropped)
    assert any("download.pytorch.org" in d for d in report.dropped)


def test_satisfiable_helpers():
    assert mr.satisfiable([">=1.0", "<2"])
    assert mr.satisfiable([">1.2.3", "<1.2.5"])
    assert not mr.satisfiable([">=2", "<2"])
    assert not mr.satisfiable(["==1.0", "!=1.0"])
    assert mr.satisfiable([])


def test_lock_calls_real_resolver_with_platform(tmp_path):
    fake_uv = tmp_path / "uv"
    log = tmp_path / "args.txt"
    fake_uv.write_text(f"#!{sys.executable}\nimport sys, pathlib\n"
                       f"pathlib.Path({str(log)!r}).write_text(' '.join(sys.argv[1:]))\n"
                       "out = sys.argv[sys.argv.index('-o') + 1]\npathlib.Path(out).write_text('x==1\\n')\n")
    fake_uv.chmod(fake_uv.stat().st_mode | stat.S_IEXEC)
    req_in = tmp_path / "r.in"
    req_in.write_text("x\n")
    out = mr.lock(req_in, "windows-cu124", tmp_path / "locks", [str(tmp_path / "c.txt")], uv=str(fake_uv))
    args = log.read_text()
    assert out.name == "windows-cu124.lock.txt" and out.is_file()
    for needle in ("pip compile", "--python-platform x86_64-pc-windows-msvc", "--torch-backend cu124",
                   "--python-version 3.11", "--generate-hashes", "-c "):
        assert needle in args


def test_lock_without_uv_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(mr.shutil, "which", lambda _: None)
    req_in = tmp_path / "r.in"
    req_in.write_text("x\n")
    try:
        mr.lock(req_in, "linux-cpu", tmp_path, [])
    except RuntimeError as exc:
        assert "uv" in str(exc)
    else:
        raise AssertionError("должна быть ошибка")


def test_repo_merge_is_clean():
    code, report, _ = mr.run(mr.REPO_ROOT, "gpu", None, check_only=True)
    assert code == 0, report.conflicts + report.banned


def _fake_uv(tmp_path, body: str, code: int = 0):
    uv = tmp_path / "uv"
    uv.write_text(f"#!{sys.executable}\nimport sys, pathlib\n"
                  "out = sys.argv[sys.argv.index('-o') + 1]\n"
                  f"pathlib.Path(out).write_text({body!r})\n"
                  f"pathlib.Path(out + '.args').write_text(' '.join(sys.argv[1:]))\n"
                  f"sys.exit({code})\n")
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
    return str(uv)


def test_resolve_check_detects_drift_and_extras(tmp_path):
    lock = tmp_path / "l1.lock.txt"
    lock.write_text("torch==2.11.0+cu128\npillow==13.0.0\nnumpy==2.4.6\n")
    req = tmp_path / "r.in"
    req.write_text("pillow>=10\nfastapi\n")
    uv = _fake_uv(tmp_path, "torch==2.11.0\npillow==13.0.0\nnumpy==2.4.6\nfastapi==0.120.0\n")
    res = mr.resolve_check(req, lock, "windows-cu128", tmp_path / "w", no_torch_index=True, uv=uv)
    assert res["ok"] and res["drift"] == {} and res["extra"] == ["fastapi"]
    args = (tmp_path / "w" / "resolve_windows-cu128.txt.args").read_text()
    assert "--python-platform x86_64-pc-windows-msvc" in args and "--torch-backend" not in args
    assert "torch==2.11.0\n" in (tmp_path / "w" / "constraints_pypi.txt").read_text()  # +cu128 снят
    uv = _fake_uv(tmp_path, "torch==2.11.0+cu128\npillow==12.2.0\nnumpy==2.4.6\n")
    res = mr.resolve_check(req, lock, "windows-cu128", tmp_path / "w2", uv=uv)
    assert not res["ok"] and res["drift"]["pillow"] == {"lock": "13.0.0", "resolved": "12.2.0"}
    assert "--torch-backend cu128" in (tmp_path / "w2" / "resolve_windows-cu128.txt.args").read_text()
    uv = _fake_uv(tmp_path, "", code=1)
    assert not mr.resolve_check(req, lock, "windows-cu128", tmp_path / "w3", uv=uv)["ok"]


def test_parse_pins():
    assert mr.parse_pins("a==1.0\n    # via x\nB_c[extra]==2.0 ; python_version>'3'\ntorch==2.11.0+cu128 \\\n") == {
        "a": "1.0", "b-c": "2.0", "torch": "2.11.0+cu128"}
