"""Опциональный LoRA (вне критического пути): датасет из guard-чистых ответов, dry-run без torch."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from copilot.train import make_dataset as md

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _fixture_norms(monkeypatch):
    monkeypatch.setenv("COPILOT_ALLOW_FIXTURE", "1")
    monkeypatch.setenv("COPILOT_EMBED_BACKEND", "hashing")


def test_dataset_examples_are_guard_clean_and_runtime_format() -> None:
    rows = md.build(n=60, seed=3)
    assert len(rows) == 60
    cats = {r["meta"]["category"] for r in rows}
    assert {"unobservable", "norm", "why"} & cats and len(cats) >= 6
    for r in rows:
        assert r["messages"][0]["role"] == "system" and "<<<ДАННЫЕ" in r["messages"][0]["content"]
        assert r["messages"][-1]["content"].startswith("Вопрос проверяющего:")
        assert r["completion"].strip() and "<|" not in r["completion"]
    refusals = [r for r in rows if r["meta"]["category"] in {"unobservable", "identity"}]
    assert refusals and all(x["completion"].startswith("По имеющимся данным") for x in refusals)


def test_dataset_is_deterministic() -> None:
    a = md.build(n=20, seed=5)
    b = md.build(n=20, seed=5)
    assert [x["completion"] for x in a] == [x["completion"] for x in b]


def test_cli_writes_split_and_lora_dry_run_without_torch(tmp_path) -> None:
    out = tmp_path / "data"
    assert md.main(["--n", "40", "--out", str(out)]) == 0
    train_rows = (out / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    val_rows = (out / "val.jsonl").read_text(encoding="utf-8").strip().splitlines() if (out / "val.jsonl").exists() else []
    assert len(train_rows) + len(val_rows) == 40 and json.loads(train_rows[0])["completion"]
    code = ("import sys, json; from copilot.train import lora; "
            f"rc = lora.main(['--data', r'{out}', '--dry-run']); "
            "print(json.dumps({'rc': rc, 'torch': 'torch' in sys.modules, 'peft': 'peft' in sys.modules}))")
    res = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    last = json.loads(res.stdout.strip().splitlines()[-1])
    plan = json.loads(res.stdout.strip().splitlines()[-2])
    assert last == {"rc": 0, "torch": False, "peft": False}
    assert plan["train"] == len(train_rows) and plan["lora"]["target"] == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_lora_rejects_malformed_data(tmp_path) -> None:
    from copilot.train import lora

    (tmp_path / "train.jsonl").write_text(json.dumps({"messages": [{"role": "user", "content": "x"}],
                                                      "completion": "y"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        lora.main(["--data", str(tmp_path), "--dry-run"])
