"""Тесты реального каталога и load_catalog(): схема, fallback, хелперы, совместимость с RuleRef."""
from __future__ import annotations

import builtins
from pathlib import Path

import pytest

import norms
from norms import validate as v

try:
    from contracts.sentinel_contracts import RuleRef  # noqa: F401
except ImportError:  # пока ветка A01 не смерджена
    from contracts.sentinel_contracts import RuleRef  # noqa: F401


@pytest.fixture(autouse=True)
def _fresh_cache():
    norms.clear_cache()
    yield
    norms.clear_cache()


def test_load_catalog_has_all_types():
    cat = norms.load_catalog()
    assert cat["version"] == 1
    for tid in norms.TYPE_IDS:
        t = cat["types"][tid]
        assert t["name"] and t["observable"] and t["not_observable"] and t["exclusions"]
        assert "{t}" in t["description_template"] and "{facts}" in t["description_template"]
        assert t["legal_status"] in ("confirmed", "candidate", "disabled")


def test_real_catalog_passes_validation():
    yaml = pytest.importorskip("yaml")
    cat = yaml.safe_load(norms.CATALOG_PATH.read_text(encoding="utf-8"))
    rep = v.validate_catalog(cat)
    v.check_fallback(cat, rep)
    assert rep.ok, rep.errors
    assert all(q.status == "OK" for q in rep.quotes if q.status != "EMPTY")


def test_rules_are_ruleref_compatible():
    cat = norms.load_catalog()
    for tid, t in cat["types"].items():
        for r in t["rules"]:
            assert tuple(r) == norms.RULE_KEYS
            RuleRef(**r)


def test_no_verified_without_fulltext_and_no_invented_clause():
    """Пока источники — поисковые фрагменты, verified=true недопустим (честность каталога)."""
    cat = norms.load_catalog()
    for t in cat["types"].values():
        for r in t["rules"]:
            assert r["verified"] is False
            assert r["url"].startswith("https://adilet.zan.kz/rus/docs/")


def test_disabled_types_return_no_rule_refs():
    cat = norms.load_catalog()
    for tid, t in cat["types"].items():
        if t["legal_status"] == "disabled":
            assert norms.rule_refs(tid, cat) == []
    assert norms.rule_refs("V08", cat)  # единственный candidate на сейчас


def test_exclusions_cover_phone_food_door_second_person():
    text = " ".join(norms.exclusions()).lower()
    for word in ("телефон", "еда", "дверь", "второй человек"):
        assert word in text


def test_fallback_when_file_missing(tmp_path: Path):
    cat = norms.load_catalog(tmp_path / "nope.yaml", use_cache=False)
    assert norms.catalog_source() == "fallback"
    assert set(norms.TYPE_IDS) <= set(cat["types"])


def test_fallback_when_yaml_not_installed(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("no yaml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    cat = norms.load_catalog(use_cache=False)
    assert norms.catalog_source() == "fallback"
    assert cat["types"]["V08"]["rules"]


def test_fallback_matches_yaml():
    yaml = pytest.importorskip("yaml")
    from norms._fallback import CATALOG

    assert CATALOG == yaml.safe_load(norms.CATALOG_PATH.read_text(encoding="utf-8"))


def test_broken_yaml_falls_back(tmp_path: Path):
    pytest.importorskip("yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("types: [unclosed", encoding="utf-8")
    cat = norms.load_catalog(bad, use_cache=False)
    assert norms.catalog_source() == "fallback" and "V01" in cat["types"]


def test_env_override(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(norms.ENV_CATALOG, str(tmp_path / "missing.yaml"))
    norms.load_catalog(use_cache=False)
    assert norms.catalog_source() == "fallback"


def test_load_catalog_returns_independent_copies():
    a = norms.load_catalog()
    a["types"]["V01"]["name"] = "испорчено"
    assert norms.load_catalog()["types"]["V01"]["name"] != "испорчено"


def test_render_description_and_unknown_type():
    text = norms.render_description("V08", 22.4, ["клиента у окна нет", "кассир пересчитывает наличные."])
    assert text.startswith("00:22:") and "клиента у окна нет; кассир пересчитывает наличные" in text
    assert norms.get_type("V42")["name"] == norms.load_catalog()["types"]["V99"]["name"]
    assert norms.format_timecode(125) == "02:05"
    assert "см. кадры-доказательства" in norms.render_description("V99", 1, [])


def test_import_has_no_side_effects():
    import subprocess
    import sys

    code = "import sys, norms; print('yaml' in sys.modules, norms.catalog_source())"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=Path(norms.NORMS_DIR).parent, check=True).stdout.split()
    assert out == ["False", "none"]  # импорт не читает каталог и не тянет PyYAML


def test_legal_basis_block():
    lb = norms.legal_basis("V08")
    assert lb["status"] == "candidate" and lb["rules"] and len(lb["checks"]) == len(lb["rules"])
    assert lb["checks"][0]["quote_level"] == "snippet"
    off = norms.legal_basis("V05")
    assert off["status"] == "disabled" and off["rules"] == [] and off["checks"] == []
    assert any("телефон" in x.lower() for x in off["exclusions"])
    assert norms.legal_basis("V42")["type_id"] == "V99"


def test_validate_report_exposes_statuses():
    yaml = pytest.importorskip("yaml")
    rep = v.validate_catalog(yaml.safe_load(norms.CATALOG_PATH.read_text(encoding="utf-8")))
    d = rep.to_dict()
    assert d["legal_status"]["V08"] == "candidate" and "not a legal verification" in d["check"]


def test_phase2_types_and_examples():
    cat = norms.load_catalog()
    assert "клиент" in " ".join(cat["types"]["V01"]["exclusions"])  # клиент напротив — норма
    assert cat["types"]["V01"]["name"].startswith("Посторонний на стороне кассира")
    v09 = cat["types"]["V09"]
    assert v09["legal_status"] == "candidate" and v09["rules"][0]["clause"] == "п. 12"
    for tid, t in cat["types"].items():
        kinds = {e["kind"] for e in t["examples"]}
        assert "positive" in kinds and kinds & {"negative", "unknown"}, tid
        for chk in t.get("rule_checks", []):
            assert chk["quote_checked_by_human"] is False and chk["applicability_checked_by_human"] is False


def test_review_matrix_in_sync_and_separates_statuses():
    yaml = pytest.importorskip("yaml")
    cat = yaml.safe_load(norms.CATALOG_PATH.read_text(encoding="utf-8"))
    rep = v.Report()
    v.check_matrix(cat, rep)
    assert rep.ok, rep.errors
    text = v.render_matrix(cat)
    assert "цитату сверил человек" in text and "применимость проверил человек" in text
    assert "V09" in text and "без пункта, без правового вывода" in text


def test_examples_helper():
    all_ex = norms.examples()
    assert {e["type_id"] for e in all_ex} >= set(norms.TYPE_IDS)
    assert all(e["expected"] in ("candidate", "no_violation", "unknown") for e in all_ex)
    unk = [e for e in norms.examples("V02") if e["kind"] == "unknown"][0]
    assert any(val is None for val in unk["signals"].values())  # unknown ≠ false


def test_v02_candidate_with_clause_22_and_client_condition():
    t = norms.load_catalog()["types"]["V02"]
    assert t["legal_status"] == "candidate"
    assert t["rules"][0]["clause"] == "п. 22" and t["rules"][0]["verified"] is False
    assert "до окончания обслуживания клиента" in t["rules"][0]["quote"]
    assert "во время обслуживания клиента" in t["rule_checks"][0]["applicability"]
    assert any("клиент" in c for c in t["rule_checks"][0]["conditions_visible"])
    assert t["rules"][1]["clause"] is None  # номер пункта не установлен — честно null
    assert norms.rule_refs("V02")  # candidate → норма выдаётся с verified=false


def test_v08_client_presence_is_the_condition():
    t = norms.load_catalog()["types"]["V08"]
    assert "отсутствие клиента" in t["legal_note"] and "не по внешности" in t["legal_note"]
    neg = [e for e in t["examples"] if e["kind"] == "negative"]
    assert any(e["signals"].get("hands_hidden") == 1.0 for e in neg)  # руки скрыты при клиенте — не V08
    assert all(e["expected"] == "no_violation" for e in neg)


def test_v09_applicability_is_cash_unit_not_appearance():
    t = norms.load_catalog()["types"]["V09"]
    assert "кассовом узле" in t["legal_note"]
    hidden = " ".join(t["rule_checks"][0]["conditions_not_visible"])
    assert "принадлежность вещи" in hidden and "кассовый узел" in hidden


def test_no_human_flags_set_by_agent():
    for tid, t in norms.load_catalog()["types"].items():
        for chk in t.get("rule_checks", []):
            assert chk["quote_checked_by_human"] is False, tid
            assert chk["applicability_checked_by_human"] is False, tid
        assert t["legal_status"] != "confirmed", tid
