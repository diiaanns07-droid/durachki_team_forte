"""Тесты валидатора: мини-фикстура, анти-галлюцинационная проверка цитат, правила статусов."""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from norms import validate as v  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "mini"


def _mini() -> dict:
    return yaml.safe_load((FIXTURE / "catalog.yaml").read_text(encoding="utf-8"))


def _run(cat: dict, base: Path = FIXTURE) -> v.Report:
    return v.validate_catalog(cat, base, required_types=("V01", "V99"))


def test_mini_fixture_is_valid():
    rep = _run(_mini())
    assert rep.ok, rep.errors
    assert [q.status for q in rep.quotes] == ["OK"]


def test_altered_quote_is_rejected():
    cat = _mini()
    cat["types"]["V01"]["rules"][0]["quote"] = "Кассовый работник выполняет тестовое требование без клиента."
    rep = _run(cat)
    assert not rep.ok
    assert any("НЕ найдена" in e for e in rep.errors)
    assert rep.quotes[0].status == "NOT_FOUND"


def test_not_quotable_block_is_excluded():
    cat = _mini()
    cat["types"]["V01"]["rules"][0]["quote"] = "кассир всегда улыбается клиенту"
    rep = _run(cat)
    assert rep.quotes[0].status == "NOT_FOUND"


def test_typography_normalization_only():
    assert v.normalize("«Касса» – это  помещение") == v.normalize('"касса" - это помещение')
    assert v.normalize("пересчитывает") != v.normalize("пересчитал")


def test_verified_true_requires_fulltext(tmp_path: Path):
    cat = _mini()
    cat["types"]["V01"]["rules"][0]["verified"] = True
    rep = _run(cat)
    assert any("полным текстом" in e for e in rep.errors)

    base = tmp_path / "mini"
    shutil.copytree(FIXTURE, base)
    src = base / "sources" / "doc_a.md"
    src.write_text(src.read_text(encoding="utf-8").replace("retrieval: search_snippets", "retrieval: fulltext"),
                   encoding="utf-8")
    cat["types"]["V01"]["rule_checks"][0]["quote_level"] = "fulltext"
    rep = _run(cat, base)
    assert any("quote_checked_by_human" in e for e in rep.errors)  # цитату должен сверить человек
    cat["types"]["V01"]["rule_checks"][0]["quote_checked_by_human"] = True
    rep = _run(cat, base)
    assert rep.ok, rep.errors


def test_extra_rule_key_breaks_ruleref_and_is_rejected():
    cat = _mini()
    cat["types"]["V01"]["rules"][0]["note"] = "комментарий"
    rep = _run(cat)
    assert any("лишние ключи" in e for e in rep.errors)


def test_missing_required_type():
    cat = _mini()
    del cat["types"]["V99"]
    rep = _run(cat)
    assert any("V99" in e and "отсутствует" in e for e in rep.errors)


def test_template_placeholders():
    cat = _mini()
    cat["types"]["V01"]["description_template"] = "{t}: без фактов"
    assert any("{facts}" in e for e in _run(cat).errors)
    cat["types"]["V01"]["description_template"] = "{t}: {facts} {unknown}"
    assert any("плейсхолдер" in e for e in _run(cat).errors)


def test_disabled_type_must_not_carry_rules():
    cat = _mini()
    cat["types"]["V99"]["rules"] = copy.deepcopy(cat["types"]["V01"]["rules"])
    rep = _run(cat)
    assert any("disabled" in e for e in rep.errors)


def test_candidate_requires_numbered_clause():
    cat = _mini()
    cat["types"]["V01"]["rules"][0]["clause"] = None
    rep = _run(cat)
    assert any("candidate" in e for e in rep.errors)


def test_exclusions_required():
    cat = _mini()
    cat["types"]["V01"]["exclusions"] = []
    assert any("exclusions" in e for e in _run(cat).errors)


def test_rule_checks_alignment_and_level():
    cat = _mini()
    cat["types"]["V01"]["rule_checks"] = []
    assert any("той же длины" in e for e in _run(cat).errors)
    cat = _mini()
    cat["types"]["V01"]["rule_checks"][0]["quote_level"] = "none"
    assert any("не согласован" in e for e in _run(cat).errors)


def test_unknown_document_url():
    cat = _mini()
    cat["types"]["V01"]["rules"][0]["url"] = "https://adilet.zan.kz/rus/docs/NOPE"
    rep = _run(cat)
    assert any("не найден в разделе documents" in e for e in rep.errors)


def test_strict_turns_warnings_into_errors():
    rep = v.validate_catalog(_mini(), FIXTURE, strict=True, required_types=("V01", "V99"))
    assert not rep.ok
    assert all(e.startswith("[strict]") for e in rep.errors)


def test_cli_on_mini_fixture(capsys):
    rc = v.main(["--catalog", str(FIXTURE / "catalog.yaml"), "--json"])
    out = capsys.readouterr().out
    # мини-фикстура не содержит V02..V08 → CLI (полный набор типов) должен упасть
    assert rc == 1 and '"ok": false' in out


def test_agent_notes_in_snippet_file_are_not_quotable():
    cat = _mini()
    cat["types"]["V01"]["rules"][0]["quote"] = "Строка выше разбита переносом"
    assert _run(cat).quotes[0].status == "NOT_FOUND"


def test_quote_cannot_span_two_blocks():
    text = "> первый фрагмент окончание\n\n> второй фрагмент\n"
    q = v.quotable_text(text, "search_snippets")
    assert v.normalize("первый фрагмент окончание") in q
    assert v.normalize("окончание второй фрагмент") not in q
    assert v.normalize("окончание второй") in v.quotable_text(text.replace(">", ""), "fulltext")


def test_real_sources_header_notes_are_not_quotable():
    import norms

    body = (norms.SOURCES_DIR / "nbrk_231_2019_cash_operations.md").read_text(encoding="utf-8")
    meta, rest = v.parse_front_matter(body)
    q = v.quotable_text(rest, meta["retrieval"])
    assert v.normalize("Статус на самой странице adilet НЕ проверен") not in q
    assert v.normalize("Наличные деньги принимаются и пересчитываются кассовым работником в присутствии клиента.") in q


def test_confirmed_requires_human_applicability_check(tmp_path: Path):
    base = tmp_path / "mini"
    shutil.copytree(FIXTURE, base)
    src = base / "sources" / "doc_a.md"
    src.write_text(src.read_text(encoding="utf-8").replace("retrieval: search_snippets", "retrieval: fulltext"),
                   encoding="utf-8")
    cat = _mini()
    t = cat["types"]["V01"]
    t["legal_status"] = "confirmed"
    t["rules"][0]["verified"] = True
    t["rule_checks"][0].update(quote_level="fulltext", quote_checked_by_human=True)
    assert any("applicability_checked_by_human" in e for e in _run(cat, base).errors)
    t["rule_checks"][0]["applicability_checked_by_human"] = True
    assert _run(cat, base).ok


def test_examples_schema():
    cat = _mini()
    ex = cat["types"]["V01"]["examples"]
    ex[1]["signals"] = {"cash_visible": 0.0}  # unknown без null
    assert any("null" in e for e in _run(cat).errors)
    cat = _mini()
    cat["types"]["V01"]["examples"][0]["signals"] = {"no_such_signal": 1.0}
    assert any("SIGNALS" in e for e in _run(cat).errors)
    cat = _mini()
    cat["types"]["V01"]["examples"][0]["expected"] = "no_violation"
    assert any("expected" in e for e in _run(cat).errors)
    cat = _mini()
    cat["types"]["V01"]["examples"] = [cat["types"]["V01"]["examples"][1]]
    assert any("positive" in e for e in _run(cat).errors)


def test_report_counts_synthetic_examples():
    rep = _run(_mini())
    assert rep.examples == {"positive": 1, "unknown": 1}
    assert rep.to_dict()["examples"] == {"positive": 1, "unknown": 1}
