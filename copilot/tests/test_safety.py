"""Безопасность и локальность: нет облачных вызовов, нет загрузки моделей при импорте, только локальный LLM-сервер."""
from __future__ import annotations

import ast
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from copilot.llm import LlamaServerBackend, check_local_url, make_llm

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent

FORBIDDEN_IMPORTS = {"openai", "anthropic", "google.generativeai", "google.genai", "cohere", "mistralai",
                     "groq", "langchain_openai", "together", "replicate"}


def _py_files() -> list[Path]:
    return [p for p in PKG.rglob("*.py") if "tests" not in p.parts]


def test_no_cloud_ai_sdk_imports() -> None:
    for path in _py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                assert not any(n == f or n.startswith(f + ".") for f in FORBIDDEN_IMPORTS), f"{path}: {n}"


def test_no_cloud_endpoints_in_code() -> None:
    bad = ("api.openai.com", "api.anthropic.com", "generativelanguage.googleapis.com", "api.cohere",
           "api.mistral.ai", "api.groq.com", "openrouter.ai")
    for path in _py_files():
        text = path.read_text(encoding="utf-8")
        for b in bad:
            assert b not in text, f"{path} mentions {b}"


def test_import_is_light_and_loads_no_models() -> None:
    """Импорт пакета и подмодулей не тянет torch/transformers/llama_cpp/sentence_transformers."""
    code = (
        "import sys, copilot, copilot.service, copilot.router, copilot.rag, copilot.llm, copilot.report;"
        "heavy=[m for m in ('torch','transformers','llama_cpp','sentence_transformers','faiss') if m in sys.modules];"
        "print(','.join(heavy))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"heavy modules imported at import time: {out.stdout}"


def test_service_creation_does_not_touch_llm_or_corpus() -> None:
    from copilot.service import CopilotService

    svc = CopilotService()
    assert svc._llm is None and svc._corpus is None and svc._retriever is None and not svc._llm_resolved


@pytest.mark.parametrize("url", ["http://127.0.0.1:8081", "http://localhost:8080/", "http://[::1]:9000",
                                 "http://192.168.1.10:8081", "http://llm:8080"])
def test_local_urls_allowed(url: str) -> None:
    assert check_local_url(url).startswith("http")


@pytest.mark.parametrize("url", ["https://api.openai.com/v1", "http://8.8.8.8:80", "https://example.com",
                                 "ftp://127.0.0.1", "http://llm.example.org:8080"])
def test_remote_urls_rejected(url: str) -> None:
    with pytest.raises(ValueError):
        check_local_url(url)


def test_private_hosts_can_be_disallowed() -> None:
    with pytest.raises(ValueError):
        check_local_url("http://192.168.1.10:8081", allow_private=False)
    with pytest.raises(ValueError):
        check_local_url("http://llm:8080", allow_private=False)


def test_remote_server_url_never_selected(monkeypatch) -> None:
    from copilot.config import load_config

    warnings: list[str] = []
    cfg = load_config({"enabled": True, "llm": {"backend": "auto", "server_url": "https://api.openai.com",
                                                "model_candidates": []}})
    assert make_llm(cfg, warnings) is None
    assert any("local" in w for w in warnings)


def test_no_network_during_mock_answer(cfg, corpus, job_violation, monkeypatch) -> None:
    """Ответ в mock-режиме не открывает ни одного сокета."""
    from copilot.service import CopilotService

    def _boom(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", _boom)
    monkeypatch.setattr(socket.socket, "connect", _boom)
    r = CopilotService(cfg=cfg, corpus=corpus).answer(job_violation, "Почему это нарушение?", [])
    assert r["error"] is None


def test_server_backend_disables_proxies() -> None:
    import urllib.request

    srv = LlamaServerBackend("http://127.0.0.1:1")
    opener = srv._opener()
    # пустой ProxyHandler подавляет системный (HTTPS_PROXY и т.п.): трафик к LLM не уходит через прокси
    proxy_handlers = [h for h in opener.handlers if isinstance(h, urllib.request.ProxyHandler)]
    assert all(not h.proxies for h in proxy_handlers)
    assert srv.healthy(timeout=0.2) is False


def test_fixture_not_used_in_prod_mode(monkeypatch, tmp_path) -> None:
    """Без SENTINEL_MOCK/COPILOT_ALLOW_FIXTURE тестовые нормы-фикстуры не подставляются."""
    from copilot.config import load_config
    from copilot.corpus import load_corpus

    cfg = load_config({"corpus": {"catalog_path": str(tmp_path / "missing.yaml"),
                                  "sources_glob": str(tmp_path / "none" / "*.md")}})
    corpus = load_corpus(cfg)
    assert corpus.chunks == [] and not corpus.is_fixture
    monkeypatch.setenv("SENTINEL_MOCK", "1")
    assert load_corpus(cfg).is_fixture
