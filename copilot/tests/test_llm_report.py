from __future__ import annotations

import sys
import types

import pytest

from copilot.config import load_config
from copilot.llm import LLMTimeout, LlamaCppBackend, plan_gpu_layers, strip_think
from copilot.report import report


def test_plan_gpu_layers() -> None:
    assert plan_gpu_layers(None, 2500, 36) == 0                     # нет GPU -> CPU
    assert plan_gpu_layers(8000, 2500, 36, reserve_mb=1500) == -1   # хватает -> всё на GPU
    assert plan_gpu_layers(1000, 2500, 36, reserve_mb=1500) == 0    # VRAM занята VLM -> CPU
    part = plan_gpu_layers(3400, 2500, 36, reserve_mb=1500, kv_mb=400)
    assert 0 < part < 36                                            # частичный offload


def test_strip_think() -> None:
    assert strip_think("<think>a</think>Ответ") == "Ответ"
    assert strip_think("Ответ<think>незакрыт") == "Ответ"


class _FakeLlama:
    instances = 0

    def __init__(self, **kwargs):
        _FakeLlama.instances += 1
        self.kwargs = kwargs

    def create_chat_completion(self, messages, stream, **kw):
        for piece in ["Телефон ", "запрещён ", "[N1]."]:
            yield {"choices": [{"delta": {"content": piece}}]}


@pytest.fixture()
def fake_llama_cpp(monkeypatch):
    mod = types.ModuleType("llama_cpp")
    mod.Llama = _FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    _FakeLlama.instances = 0
    return mod


def test_llama_cpp_lazy_load_and_release(fake_llama_cpp, tmp_path) -> None:
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"\0" * 16)
    cfg = load_config({"llm": {"gpu_layers": "0", "idle_unload_s": 0}})
    be = LlamaCppBackend(gguf, cfg)
    assert _FakeLlama.instances == 0                       # конструктор не грузит модель
    text = "".join(be.stream_chat([{"role": "user", "content": "x"}]))
    assert text == "Телефон запрещён [N1]." and _FakeLlama.instances == 1
    assert be._llm.kwargs["n_gpu_layers"] == 0
    "".join(be.stream_chat([{"role": "user", "content": "x"}]))
    assert _FakeLlama.instances == 1                       # модель тёплая
    be.release()
    assert be._llm is None


def test_llama_cpp_deadline(fake_llama_cpp, tmp_path) -> None:
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"\0")
    be = LlamaCppBackend(gguf, load_config({"llm": {"gpu_layers": "0", "idle_unload_s": 0}}))
    with pytest.raises(LLMTimeout):
        "".join(be.stream_chat([{"role": "user", "content": "x"}], deadline=0.0))


def test_make_llm_picks_llama_cpp_when_gguf_exists(fake_llama_cpp, tmp_path, monkeypatch) -> None:
    import importlib.util

    from copilot.llm import make_llm

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"\0")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name == "llama_cpp" else None)
    be = make_llm(load_config({"llm": {"backend": "llama_cpp", "model_candidates": [str(gguf)]}}))
    assert isinstance(be, LlamaCppBackend) and be._llm is None
    # auto больше не грузит отдельную модель: на 8 ГБ рядом с VLM она не помещается
    assert make_llm(load_config({"llm": {"backend": "auto", "model_candidates": [str(gguf)]}})) is None


def test_report_contents(cfg, corpus, job_violation) -> None:
    md = report(job_violation, corpus=corpus, cfg=cfg)
    assert md.startswith("# Заключение для проверяющего")
    assert "не юридическое заключение" in md
    assert "00:22" in md and "00:48" in md or "00:47" in md
    assert "п. 9" in md and "https://example.invalid/fixture/reglament#p9" in md
    assert "Выдуманный закон" not in md and "999" not in md          # выдуманная норма отброшена
    assert "Отброшено ссылок, не найденных в каталоге норм: 1" in md
    assert "image_base64" not in md and "QUFBQQ" not in md
    assert "Что проверить вручную" in md


def test_report_clean_and_failed(cfg, corpus, job_clean) -> None:
    md = report(job_clean, corpus=corpus, cfg=cfg)
    assert "Признаков нарушения не обнаружено" in md
    md2 = report({"job_id": "f", "status": "failed"}, corpus=corpus, cfg=cfg)
    assert "Нет данных" in md2
