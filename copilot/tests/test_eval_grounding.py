"""Синтетическая проверка groundedness: скрипт работает без GPU; безопасность 100% даже у «плохой модели»."""
from __future__ import annotations

import json

import pytest

from copilot import eval_grounding as eg


@pytest.fixture(autouse=True)
def _fixture_norms(monkeypatch):
    monkeypatch.setenv("COPILOT_ALLOW_FIXTURE", "1")
    monkeypatch.setenv("COPILOT_EMBED_BACKEND", "hashing")


def test_cases_cover_required_categories() -> None:
    cats = {c.category for c in eg.build_cases()}
    assert {"frame", "norm", "unobservable", "fake_citation", "incomplete", "injection", "explain"} <= cats


@pytest.mark.parametrize("backend", ["mock", "retrieval"])
def test_deterministic_modes_pass_everything(backend) -> None:
    rep = eg.run(backend)
    s = rep["summary"]
    failed = [(c["id"], c["checks"], c["answer"][:120]) for c in rep["cases"] if not c["pass"]]
    assert s["pass_rate"] == 1.0, failed
    assert s["safety_pass_rate"] == 1.0 and s["error_rate"] == 0.0
    assert s["guard_cut_rate"] == 0.0  # шаблонные ответы ничего не выдумывают
    assert s["modes"] == [backend]


def test_adversarial_model_is_fully_neutralized() -> None:
    """«Плохая модель» выдумывает пункт, сумму, ссылку и отрицание — guard убирает всё."""
    s = eg.run("adversarial")["summary"]
    assert s["safety_pass_rate"] == 1.0
    assert s["guard_cut_rate"] >= 0.8


def test_cli_writes_json_and_markdown_and_compares(tmp_path, capsys) -> None:
    out = tmp_path / "g.json"
    assert eg.main(["--backend", "retrieval", "--out", str(out), "--quiet"]) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["summary"]["n"] == len(eg.build_cases())
    md = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "Copilot groundedness" in md and "безопасность" in md
    assert eg.main(["--compare", str(out), str(out)]) == 0
    assert "safety_pass_rate" in capsys.readouterr().out
