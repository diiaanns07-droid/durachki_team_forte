import re

import pytest

from ml.vlm.catalog import catalog_types, describe_types
from ml.vlm.prompting import CHECKLIST_TEXT, get_prompt, list_prompts, render_discover, render_verify, retry_suffix
from ml.vlm.config import load_config

PH = re.compile(r"\{\{\s*[a-z_]+\s*\}\}")


def test_library_has_all_prompts_in_both_languages():
    keys = set(list_prompts())
    for pid in ("system", "verify", "discover", "retry_suffix"):
        for lang in ("ru", "en"):
            assert f"{pid}.v1.{lang}" in keys


@pytest.mark.parametrize("lang", ["ru", "en"])
def test_render_verify_substitutes_everything(lang):
    system, user = render_verify("V05", ["Телефон у уха"], [21.0, 21.5, 22.0, 22.5, 23.0], None, lang)
    assert not PH.search(system) and not PH.search(user)
    # имя — из АКТИВНОГО каталога (norms A13 в общей сборке, иначе запасной), строка кандидата проверяется целиком
    name = catalog_types()["V05"]["name"]
    line = f"- тип: V05 «{name}»" if lang == "ru" else f'- type: V05 "{name}"'
    assert line in user.splitlines(), f"нет строки кандидата {line!r}"
    assert "Телефон у уха" in user
    for tid in catalog_types():
        assert tid in user  # описания всех типов каталога подставлены
    assert '"visible"' in user and '"visible_frames"' in user and "5: 23.0" in user


@pytest.mark.parametrize("lang", ["ru", "en"])
def test_render_discover_checklist(lang):
    checklist = load_config(use_env=False)["discover"]["checklist"]
    system, user = render_discover(checklist, [1.0, 2.0, 3.0, 4.0], None, lang)
    assert not PH.search(user)
    for key, tid in checklist.items():
        assert f'"{key}" ({tid})' in user
    assert set(checklist) <= set(CHECKLIST_TEXT)
    assert '"findings": []' in user


def test_boxes_note_depends_on_boxes():
    _, with_boxes = render_verify("V05", [], [1.0], None, "ru", has_boxes=True)
    _, without = render_verify("V05", [], [1.0], None, "ru", has_boxes=False)
    assert "Красными рамками" in with_boxes and "Рамок на кадрах нет" in without


def test_unknown_candidate_type_falls_back_to_v99():
    _, user = render_verify("V42", [], [1.0], None, "ru")
    assert catalog_types()["V99"]["name"] in user


def test_missing_placeholder_raises():
    with pytest.raises(KeyError):
        get_prompt("verify", "v1", "ru").render(n_frames=1)


def test_catalog_from_norms_shape_is_used():
    cat = {"version": 1, "types": {"V05": {"name": "Телефон X", "observable": ["держит телефон"],
                                            "not_observable": ["собеседник"],
                                            "rules": [{"document": "d", "clause": "п. 1", "url": "u", "quote": "q"}]}}}
    text = describe_types(cat)
    assert "Телефон X" in text and "собеседник" in text and "V99" in text
    assert "п. 1" not in text  # нормы в промпт не попадают — VLM их не цитирует


def test_retry_suffix_and_no_dataset_fewshot():
    assert "JSON" in retry_suffix("ru") and "JSON" in retry_suffix("en")
    for key in list_prompts():
        pid, ver, lang = key.split(".")
        body = get_prompt(pid, ver, lang).body
        assert "base64" not in body and ".mov" not in body.lower()


def test_candidate_line_uses_explicit_catalog():
    """Явный синтетический каталог побеждает и norms A13, и запасной (изоляция от окружения)."""
    synth = {"version": 1, "types": {"V05": {"name": "Синтетическое имя V05", "observable": ["x"]},
                                     "V99": {"name": "Иное (синт.)", "observable": ["y"]}}}
    _, user = render_verify("V05", [], [1.0], synth, "ru")
    assert "- тип: V05 «Синтетическое имя V05»" in user.splitlines()
