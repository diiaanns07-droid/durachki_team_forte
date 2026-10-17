import json
import os
import shutil
import socket
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NETGUARD = ROOT / "deploy" / "netguard"


def _run(code: str, env_extra: dict, pythonpath: list[Path]) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SENTINEL_")}
    env.update(env_extra)
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in pythonpath)
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)


@pytest.fixture()
def loopback_server():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    threading.Thread(target=lambda: srv.accept(), daemon=True).start()
    yield srv.getsockname()[1]
    srv.close()


def test_netguard_blocks_egress_allows_loopback(tmp_path, loopback_server):
    log = tmp_path / "ng.jsonl"
    code = textwrap.dedent(f"""
        import socket, sys
        assert "sitecustomize" in sys.modules
        s = socket.create_connection(("127.0.0.1", {loopback_server}), timeout=2); s.close()
        for fn in (lambda: socket.create_connection(("203.0.113.7", 443), timeout=1),
                   lambda: socket.getaddrinfo("huggingface.co", 443)):
            try:
                fn()
            except PermissionError:
                print("blocked")
    """)
    r = _run(code, {"SENTINEL_NETGUARD": "block", "SENTINEL_NETGUARD_LOG": str(log)}, [NETGUARD])
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["blocked", "blocked"]
    recs = [json.loads(x) for x in log.read_text().splitlines()]
    assert {x["target"] for x in recs} == {"203.0.113.7", "huggingface.co"}


def test_netguard_allowlist_and_log_mode(tmp_path):
    log = tmp_path / "ng.jsonl"
    code = textwrap.dedent("""
        import sentinel_netguard as ng
        assert ng.is_local("llm") and ng.is_local("10.1.2.3") and ng.is_local("::1")
        assert not ng.is_local("8.8.8.8") and not ng.is_local("example.com")
        import socket
        try:
            socket.gethostbyname("example.invalid")
        except OSError:
            pass
        print(ng.status()["attempts"])
    """)
    r = _run(code, {"SENTINEL_NETGUARD": "log", "SENTINEL_NETGUARD_LOG": str(log),
                    "SENTINEL_NETGUARD_ALLOW": "llm,10.0.0.0/8"}, [NETGUARD])
    assert r.returncode == 0, r.stderr
    assert int(r.stdout.strip()) >= 1 and "logged" in log.read_text()


def test_netguard_off_by_default():
    r = _run("import sys; print('sentinel_netguard' in sys.modules)", {}, [NETGUARD])
    assert r.stdout.strip() == "False"


FAKE_PIPELINE = '''
import os, subprocess, sys, urllib.request
CHILD = "import urllib.request\\ntry:\\n    urllib.request.urlopen('https://huggingface.co', timeout=2)\\nexcept Exception:\\n    pass\\n"
def warmup():
    pass
def run(video_path, job_id, progress, deadline_s=240):
    progress("detect", 0.5)
    if os.environ.get("FAKE_PHONE_HOME"):
        try:
            urllib.request.urlopen("https://huggingface.co/api/models", timeout=2)
        except Exception:
            pass  # «graceful fallback» — smoke всё равно должен упасть
    if os.environ.get("FAKE_WORKER_PHONE_HOME"):  # как spawn-воркер A03: отдельный процесс
        subprocess.run([sys.executable, "-c", CHILD])
    meta = {"complete": os.environ.get("FAKE_INCOMPLETE") != "1", "mock": os.environ.get("FAKE_MOCK") == "1",
            "executor": {"mode": "process", "pid": 1},
            "stages": {"detect": {"status": "ok", "origin": "hf", "vram_peak_mb": 900}}}
    return {"job_id": job_id, "status": "done", "verdict": "no_violation", "violations": [], "meta": meta}
'''


@pytest.fixture()
def fake_backend(tmp_path):
    pkg = tmp_path / "fake" / "backend"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "pipeline.py").write_text(FAKE_PIPELINE)
    video = tmp_path / "v.mov"
    video.write_bytes(b"not-a-real-video")
    return tmp_path / "fake", video


def _smoke(tmp_path, fake_dir, video, extra_env=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SENTINEL_", "FAKE_"))}
    env.update({"PYTHONPATH": os.pathsep.join([str(fake_dir), str(ROOT)]), "SENTINEL_LOG_DIR": str(tmp_path / "logs"),
                "SENTINEL_WEIGHTS": str(tmp_path / "w")})
    env.update(extra_env or {})
    report = tmp_path / "report.json"
    r = subprocess.run([sys.executable, str(ROOT / "deploy" / "offline_worker_smoke.py"), "--video", str(video),
                        "--allow-empty-lock", "--report", str(report)], env=env, capture_output=True, text=True,
                       timeout=120)
    return r, (json.loads(report.read_text()) if report.exists() else None)


def test_offline_smoke_passes_with_local_worker(tmp_path, fake_backend):
    r, rep = _smoke(tmp_path, *fake_backend)
    assert r.returncode == 0, r.stdout + r.stderr
    assert rep["ok"] and rep["network_attempts"] == 0 and rep["result"]["verdict"] == "no_violation"
    assert rep["env"]["HF_HUB_OFFLINE"] == "1" and rep["stages"] == ["detect"]


def test_offline_smoke_fails_on_swallowed_network_attempt(tmp_path, fake_backend):
    r, rep = _smoke(tmp_path, *fake_backend, {"FAKE_PHONE_HOME": "1"})
    assert r.returncode == 1 and rep["network_attempts"] >= 1
    assert any("сеть" in p for p in rep["problems"])


def test_offline_smoke_refuses_mock(tmp_path, fake_backend):
    r, rep = _smoke(tmp_path, *fake_backend, {"SENTINEL_MOCK": "1"})
    assert r.returncode == 2 and rep is None


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="нужен ffmpeg")
def test_synth_video_target_size(tmp_path):
    from deploy.synth_video import make_video
    p = make_video(tmp_path / "s.mov", seconds=4, fps=10, target_mb=2)
    assert 1.0 < p.stat().st_size / 2**20 < 3.5


def test_netguard_blocks_loopback_proxy(tmp_path, loopback_server):
    code = textwrap.dedent(f"""
        import os, socket
        assert "HTTPS_PROXY" not in os.environ and os.environ["NO_PROXY"] == "*"
        try:
            socket.create_connection(("127.0.0.1", {loopback_server}), timeout=2)
        except PermissionError:
            print("proxy-blocked")
    """)
    r = _run(code, {"SENTINEL_NETGUARD": "block", "SENTINEL_NETGUARD_LOG": str(tmp_path / "l"),
                    "HTTPS_PROXY": f"http://127.0.0.1:{loopback_server}"}, [NETGUARD])
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "proxy-blocked"


def test_scripts_portable():
    for p in (ROOT / "deploy").rglob("*.ps1"):
        data = p.read_bytes()
        assert all(b < 128 for b in data), f"{p}: не-ASCII (PowerShell 5.1 прочтёт как ANSI)"
    for p in (ROOT / "deploy").rglob("*.sh"):
        data = p.read_bytes()
        assert data.startswith(b"#!") and b"\r\n" not in data, p


def test_ps1_parse_with_pwsh(tmp_path):
    """Синтаксис .ps1 парсером PowerShell (если pwsh есть: PATH или SENTINEL_TEST_PWSH)."""
    pwsh = os.environ.get("SENTINEL_TEST_PWSH") or shutil.which("pwsh")
    if not pwsh:
        pytest.skip("pwsh не установлен")
    files = [str(p) for p in sorted((ROOT / "deploy").rglob("*.ps1"))]
    checker = tmp_path / "parse.ps1"
    checker.write_text("$bad=0; foreach($f in $args){$t=$null;$e=$null;"
                       "[void][System.Management.Automation.Language.Parser]::ParseFile($f,[ref]$t,[ref]$e);"
                       "foreach($x in $e){$bad++; Write-Output ($f+':'+$x.Extent.StartLineNumber+': '+$x.Message)}}"
                       "; exit $bad\n", encoding="ascii")
    r = subprocess.run([pwsh, "-NoLogo", "-NoProfile", "-File", str(checker), *files], capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


def test_offline_smoke_counts_attempts_from_spawned_worker(tmp_path, fake_backend):
    r, rep = _smoke(tmp_path, *fake_backend, {"FAKE_WORKER_PHONE_HOME": "1"})
    assert r.returncode == 1, r.stdout + r.stderr
    assert rep["network_attempts"] >= 1 and rep["network_attempts_sample"][0]["action"] == "BLOCKED"


def test_offline_smoke_rejects_mock_meta_and_incomplete_no_violation(tmp_path, fake_backend):
    r, rep = _smoke(tmp_path, *fake_backend, {"FAKE_MOCK": "1"})
    assert r.returncode == 1 and any("mock" in p for p in rep["problems"])
    r, rep = _smoke(tmp_path, *fake_backend, {"FAKE_INCOMPLETE": "1"})
    assert r.returncode == 1 and any("неполн" in p for p in rep["problems"])


def test_offline_smoke_pitch_log_and_evidence(tmp_path, fake_backend):
    r, rep = _smoke(tmp_path, *fake_backend)
    assert r.returncode == 0, r.stdout + r.stderr
    assert rep["result"]["stages"]["detect"]["vram_peak_mb"] == 900
    logs = list((tmp_path / "logs").glob("offline_proof_*.md"))
    assert len(logs) == 1
    text = logs[0].read_text(encoding="utf-8")
    assert "попыток выхода: **0**" in text and "**OK**" in text


def test_egress_selftest_parsing(monkeypatch):
    from deploy import offline_worker_smoke as oss
    monkeypatch.setattr(oss, "EGRESS_PROBE", "import json; print(json.dumps({'tcp': 'blocked: X', 'dns': 'blocked: Y'}))")
    assert oss.egress_selftest()["all_blocked"] is True
    monkeypatch.setattr(oss, "EGRESS_PROBE", "import json; print(json.dumps({'tcp': 'OPEN'}))")
    assert oss.egress_selftest()["all_blocked"] is False
