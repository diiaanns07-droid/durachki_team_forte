import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ml.vlm.backends import BackendUnavailable, create_backend, resolve_backend
from ml.vlm.backends.llamacpp import LlamaCppPythonBackend, LlamaCppServerBackend, is_local_url
from ml.vlm.backends.mock import MockBackend
from ml.vlm.backends.transformers_backend import TransformersBackend, check_license
from ml.vlm.config import load_config


class _Fake(BaseHTTPRequestHandler):
    """Фейковый llama-server на 127.0.0.1: /health и /v1/chat/completions."""
    requests: list = []
    reject_schema = False

    def log_message(self, *a):  # тишина в выводе тестов
        pass

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode())
        type(self).requests.append(body)
        rf = body.get("response_format") or {}
        if type(self).reject_schema and rf.get("type") == "json_schema":
            self._send(400, {"error": "unsupported"})
            return
        content = json.dumps({"visible": True, "type_id": "V05", "observed_facts": [], "confidence": 0.7,
                              "best_frame": 1, "visible_frames": [1], "uncertain_because": ""})
        self._send(200, {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 900, "completion_tokens": 40}})

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def fake_server():
    _Fake.requests = []
    _Fake.reject_schema = False
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _imgs(n=2):
    from PIL import Image
    return [Image.new("RGB", (245, 448), (10 * i, 0, 0)) for i in range(n)]


def test_is_local_url():
    assert is_local_url("http://127.0.0.1:8081") and is_local_url("http://localhost:8081")
    assert is_local_url("http://192.168.1.5:8081") and is_local_url("http://[::1]:8081")
    assert is_local_url("http://vlm:8081") and is_local_url("http://host.docker.internal:8081")
    assert not is_local_url("https://api.example.com/v1") and not is_local_url("http://8.8.8.8:80")


def test_llamacpp_server_refuses_cloud_url():
    be = LlamaCppServerBackend(url="https://api.openai.com")
    ok, why = be.available()
    assert not ok and "non-local" in why
    with pytest.raises(BackendUnavailable):
        be.generate(_imgs(), "s", "p")


def test_llamacpp_server_roundtrip(fake_server):
    be = LlamaCppServerBackend(url=fake_server)
    assert be.available() == (True, "")
    res = be.generate(_imgs(3), "SYS", "PROMPT", max_new_tokens=50, timeout_s=5,
                      json_schema={"type": "object"})
    assert json.loads(res.text)["type_id"] == "V05" and res.completion_tokens == 40
    req = _Fake.requests[-1]
    user = req["messages"][1]["content"]
    assert [c["type"] for c in user] == ["image_url"] * 3 + ["text"]
    assert user[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert req["response_format"]["type"] == "json_schema" and req["temperature"] == 0.0
    assert req["max_tokens"] == 50


def test_llamacpp_server_schema_fallback(fake_server):
    _Fake.reject_schema = True
    be = LlamaCppServerBackend(url=fake_server)
    res = be.generate(_imgs(), "s", "p", json_schema={"type": "object"}, timeout_s=5)
    assert res.text and be.info()["schema_mode"] == "json_object"
    assert _Fake.requests[-1]["response_format"] == {"type": "json_object"}


def test_llamacpp_server_unreachable():
    be = LlamaCppServerBackend(url="http://127.0.0.1:9", health_timeout_s=0.3)
    ok, why = be.available()
    assert not ok and "not reachable" in why


def test_llamacpp_python_unavailable_without_files(tmp_path):
    be = LlamaCppPythonBackend(models_dir=str(tmp_path), model_file="x.gguf", mmproj_file="m.gguf")
    ok, why = be.available()
    assert not ok


def test_transformers_unavailable_reason_without_cuda_or_torch():
    be = TransformersBackend(model="Qwen/Qwen3-VL-4B-Instruct", quant="bnb4")
    ok, why = be.available()
    try:
        import torch  # noqa: F401
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    if not has_cuda:
        assert not ok and why


def test_license_guard(tmp_path):
    (tmp_path / "README.md").write_text("---\nlicense: apache-2.0\n---\n", encoding="utf-8")
    assert check_license(str(tmp_path), "strict")[0]
    (tmp_path / "README.md").write_text("---\nlicense: other\nlicense_name: qwen-research\n---\n", encoding="utf-8")
    assert not check_license(str(tmp_path), "warn")[0]  # запрещённая — блок даже в warn
    assert check_license(str(tmp_path / "nope"), "warn")[0] and not check_license(str(tmp_path / "nope"), "strict")[0]


def test_resolve_backend_modes(fake_server):
    cfg = load_config(use_env=False)
    cfg["backend"] = "off"
    assert resolve_backend(cfg)[0] is None
    cfg["backend"] = "mock"
    assert isinstance(resolve_backend(cfg)[0], MockBackend)
    cfg["backend"] = "nonsense"
    be, why = resolve_backend(cfg)
    assert be is None and "unknown" in why
    cfg["backend"] = "auto"
    cfg["auto_order"] = ["llamacpp_server"]
    cfg["llamacpp_server"]["url"] = fake_server
    be, _ = resolve_backend(cfg)
    assert isinstance(be, LlamaCppServerBackend)
    with pytest.raises(BackendUnavailable):
        create_backend("nonsense", cfg)


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("SENTINEL_MOCK", "0")
    monkeypatch.setenv("SENTINEL_VLM_BACKEND", "llamacpp_server")
    monkeypatch.setenv("SENTINEL_VLM_URL", "http://127.0.0.1:9999")
    monkeypatch.setenv("SENTINEL_VLM_MODEL", "/models/qwen3vl")
    cfg = load_config()
    assert cfg["backend"] == "llamacpp_server" and cfg["llamacpp_server"]["url"] == "http://127.0.0.1:9999"
    assert cfg["presets"][cfg["preset"]]["transformers"]["model"] == "/models/qwen3vl"
    monkeypatch.setenv("SENTINEL_MOCK", "1")
    assert load_config()["backend"] == "mock"  # глобальный SENTINEL_MOCK=1 сильнее: прогон без весов


@pytest.mark.gpu
@pytest.mark.slow
def test_real_transformers_backend_smoke():
    """Только локально: CUDA + скачанные веса (HF_HUB_OFFLINE=1). В облаке/CI — skip."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    cfg = load_config(use_env=False, overrides={"backend": "transformers"})
    be = create_backend("transformers", cfg)
    ok, why = be.available()
    if not ok:
        pytest.skip(why)
    try:
        be.load()
    except BackendUnavailable as e:
        pytest.skip(str(e))
    res = be.generate(_imgs(4), "Answer with JSON.", 'Return {"visible": false, "type_id": "V99", '
                      '"observed_facts": [], "confidence": 0.5, "best_frame": 1, "visible_frames": [], '
                      '"uncertain_because": ""}', max_new_tokens=80, timeout_s=30)
    assert res.text and not res.error
