import pytest

from ml.vlm.facts_filter import check_fact, filter_facts


@pytest.mark.parametrize("fact,cat", [
    ("Кассир пересчитывает сумму 50 000 тенге", "amount"),
    ("На столе лежит 20000 тг", "amount"),
    ("Операция проведена без проводки в системе", "accounting"),
    ("Кассира зовут Анна", "identity"),
    ("Кассир Иванова держит телефон", "identity"),
    ("Купюры фальшивые", "authenticity"),
    ("Кассир умышленно прячет деньги", "intent"),
    ("Клиент пытается украсть пачку денег", "intent"),
    ("Посетитель не имеет права находиться за столом", "norm_or_authority"),
    ("Это нарушение п. 15 Правил", "norm_or_authority"),
    ("Инкассатор без допуска стоит у сейфа", "norm_or_authority"),
    ("Кассир нарушает порядок работы с ценностями", "conclusion"),
    ("Кассир нервничает", "mental_state"),
    ("В сумке лежат деньги", "hidden_content"),
    ("Вероятно, кассир отошёл к сейфу", "hedge"),
    ("Что-то странное происходит", "no_anchor"),
])
def test_bad_facts_are_flagged(fact, cat):
    assert check_fact(fact) == cat


@pytest.mark.parametrize("fact", [
    "Кассир держит у уха предмет, похожий на мобильный телефон",
    "Дверца сейфа открыта, рядом с сейфом никого нет",
    "За кассовым столом находятся два человека",
    "На столе рядом с наличными стоит чашка",
    "Руки кассира с купюрами находятся под столом и не видны",
    "Инкассатор с зелёной сумкой стоит у стола",
])
def test_observable_facts_kept(fact):
    assert check_fact(fact) == ""


def test_filter_dedupe_trim_and_language():
    r = filter_facts(["  кассир держит телефон  ", "Кассир держит телефон", "The cashier holds a phone",
                      "1. Дверь открыта", "x"], max_len=220)
    assert r.kept == ["Кассир держит телефон", "Дверь открыта"]
    assert ("The cashier holds a phone", "not_russian") in r.dropped


def test_filter_max_len_and_items():
    long = "Кассир " + "очень " * 80 + "держит телефон"
    r = filter_facts([long, "Дверь открыта", "Сейф открыт", "Телефон в руке", "Чашка на столе"], max_len=50,
                     max_items=3)
    assert len(r.kept) == 3
    assert len(r.kept[0]) <= 51 and r.kept[0].endswith("…")


def test_filter_does_not_guarantee_truth():
    # фильтр проверяет форму, а не истинность: правдоподобная галлюцинация проходит (см. README, ограничения)
    assert check_fact("Кассир держит телефон") == ""
