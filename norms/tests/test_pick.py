"""Тесты norms.pick / norms.texts / импорта norms.fetch на СИНТЕТИЧЕСКОМ тексте (не норма, сеть не нужна)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

import norms  # noqa: E402
from norms import fetch, pick, texts, validate as v  # noqa: E402

SYNTH = """СИНТЕТИЧЕСКИЙ ТЕКСТ ДЛЯ ТЕСТОВ — НЕ НОРМА, НЕ ЦИТИРОВАТЬ
Глава 1. Общие положения
1. Настоящие тестовые правила определяют порядок проверки.
Глава 2. Порядок работы
12. Хранение личных вещей, наличных денег и ценостей работников банка осуществляется вне кассового узла в порядке, установленном внутренними документами банка.
13. Банк заключает договоры о материальной ответственности.
20. Наличные деньги принимаются и пересчитываются кассовым работником в присутствии клиента.
21. Сейф закрывается кассовым работником; ключ хранится у него.
"""


def _html(text: str) -> str:
    body = "".join(f"<p>{line}</p>" for line in text.splitlines())
    return f"<html><head><script>x=1</script></head><body><nav>меню</nav>{body}</body></html>"


@pytest.fixture()
def fetched_dir(tmp_path: Path) -> Path:
    d = tmp_path / "norms_fetched"
    d.mkdir()
    (d / "V1900019680.html").write_text(_html(SYNTH), encoding="utf-8")
    return d


def test_segment_numbers_and_headings(fetched_dir: Path):
    docs = texts.load_dir(fetched_dir, norms.load_catalog())
    assert [d.doc_id for d in docs] == ["nbrk_231_2019"]
    clauses = texts.segment(docs[0])
    nums = [c.number for c in clauses if c.number]
    assert nums == ["1", "12", "13", "20", "21"]
    c12 = next(c for c in clauses if c.number == "12")
    assert c12.heading.startswith("Глава 2") and "личных вещей" in c12.text


def test_pick_finds_v09_clause_and_nothing_for_phone(fetched_dir: Path):
    docs = texts.load_dir(fetched_dir, norms.load_catalog())
    res = pick.pick(docs, pick.load_keywords(), types=["V09", "V05", "V03"], top=3, min_score=2)
    assert res["V09"][0].clause == "п. 12"
    assert res["V09"][0].quote.startswith("Хранение личных вещей")
    assert res["V05"] == []  # про телефон в тексте ничего — тип остаётся без пункта
    assert res["V03"][0].clause == "п. 21"


def test_check_catalog_locates_quotes(fetched_dir: Path):
    cat = norms.load_catalog()
    checks = {(q.type_id, q.index): q for q in pick.check_catalog(cat, texts.load_dir(fetched_dir, cat))}
    assert checks[("V08", 0)].status == "FOUND"
    assert checks[("V09", 0)].status == "FOUND"
    assert checks[("V08", 1)].status == "NOT_FOUND"  # в синтетическом тексте этой фразы нет


def test_emit_yaml_is_ruleref_shaped(fetched_dir: Path):
    docs = texts.load_dir(fetched_dir, norms.load_catalog())
    c = pick.pick(docs, pick.load_keywords(), types=["V09"], min_score=2)["V09"][0]
    rule = yaml.safe_load(pick.emit_yaml(c, "Документ"))[0]
    assert tuple(rule) == norms.RULE_KEYS and rule["verified"] is False and rule["clause"] == "п. 12"


def test_pick_cli_writes_report(fetched_dir: Path, tmp_path: Path):
    out = tmp_path / "PICK.md"
    rc = pick.main(["--dir", str(fetched_dir), "--no-sources", "--check-catalog", "--emit-yaml",
                    "--min-score", "2", "--out", str(out)])
    text = out.read_text(encoding="utf-8")
    assert rc == 0 and "## V09" in text and "п. 12" in text and "FOUND" in text


def test_pick_without_texts_returns_2(tmp_path: Path):
    assert pick.main(["--dir", str(tmp_path / "empty"), "--no-sources"]) == 2


def test_git_ref_ingestion(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "norms_fetched").mkdir(parents=True)
    (repo / "norms_fetched" / "V1900019680.html").write_text(_html(SYNTH), encoding="utf-8")
    env = ["-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q", "-b", "data/norms-sources"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", *env, "commit", "-q", "-m", "synthetic"], cwd=repo, check=True)
    docs = texts.load_git_ref("data/norms-sources", catalog=norms.load_catalog(), repo=repo)
    assert len(docs) == 1 and docs[0].doc_id == "nbrk_231_2019"
    with pytest.raises(FileNotFoundError):
        texts.load_git_ref("no/such-ref", repo=repo)


def test_import_then_validate_flags_missing_quotes(fetched_dir: Path, tmp_path: Path):
    base = tmp_path / "norms_copy"
    shutil.copytree(norms.SOURCES_DIR, base / "sources")
    rc = fetch.main(["--from-dir", str(fetched_dir), "--out-dir", str(base)])
    assert rc == 0
    meta, _ = v.parse_front_matter((base / "sources" / "nbrk_231_2019_cash_operations.md").read_text(encoding="utf-8"))
    assert meta["retrieval"] == "fulltext"
    rep = v.validate_catalog(norms.load_catalog(), base)
    status = {(q.type_id, q.index): q.status for q in rep.quotes}
    assert status[("V08", 0)] == "OK" and status[("V09", 0)] == "OK"
    assert status[("V08", 1)] == "NOT_FOUND" and not rep.ok  # полный текст выявляет неподтверждённую цитату


@pytest.mark.parametrize("name,code", [
    ("V1900019680.html", "V1900019680"),
    ("V1900019680_full.html", "V1900019680_"),
    ("bizup_v1900019680.txt", "V1900019680"),
    ("Z950002444_.md", "Z950002444_"),
    ("no_code_here.md", None),
])
def test_adilet_code_detection(name: str, code):
    assert texts.adilet_code(name) == code


def test_suffixed_filename_maps_to_catalog_doc(tmp_path: Path):
    d = tmp_path / "f"
    d.mkdir()
    (d / "V1900019680_full.html").write_text(_html(SYNTH), encoding="utf-8")
    assert texts.load_dir(d, norms.load_catalog())[0].doc_id == "nbrk_231_2019"
