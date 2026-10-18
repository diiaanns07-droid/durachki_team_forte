"""Офлайн-тесты norms.fetch: разбор HTML и формат файла-источника (сеть не используется)."""
from __future__ import annotations

from pathlib import Path

from norms import fetch, validate as v

HTML = """<html><head><title>x</title><script>var a=1;</script></head><body>
<nav>Меню сайта</nav><div class="text"><p>20. Наличные деньги принимаются&nbsp;и пересчитываются
кассовым работником в присутствии клиента.</p><p>21. Следующий пункт.</p></div></body></html>"""


def test_html_to_text_strips_scripts_and_nav():
    text = fetch.html_to_text(HTML)
    assert "var a" not in text and "Меню сайта" not in text
    assert "20. Наличные деньги принимаются и пересчитываются" in text


def test_rendered_source_is_fulltext_and_quotable(tmp_path: Path):
    doc = {"title": "Тест", "act": "Акт", "url": "https://adilet.zan.kz/rus/docs/TEST", "source_file": "s.md"}
    md = fetch.render_source_md("t", doc, fetch.html_to_text(HTML), accessed="2026-10-17")
    meta, body = v.parse_front_matter(md)
    assert meta["retrieval"] == "fulltext" and meta["url"] == doc["url"]
    q = v.quotable_text(body, meta["retrieval"])
    assert v.normalize("Наличные деньги принимаются и пересчитываются кассовым работником в присутствии клиента.") in q
