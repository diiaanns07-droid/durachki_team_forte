"""Отчёт различает наблюдение системы, подтверждённый факт и непроверенную норму (синтетические задачи)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from copilot.config import load_config
from copilot.corpus import load_corpus
from copilot.report import report

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "norms"


def _block(md: str, header: str) -> str:
    """Текст блока от строки-заголовка до пустой строки (пустая строка сразу после «## …» не считается)."""
    start = md.index(header)
    body = start + len(header)
    if md.startswith("\n\n", body):
        body += 2
    end = md.find("\n\n", body)
    return md[start:end if end >= 0 else None]


def _corpus_with_v05(tmp_path: Path, *, verified: bool, legal_status: str | None):
    cat = yaml.safe_load((FIXTURE / "catalog.yaml").read_text(encoding="utf-8"))
    cat["types"]["V05"]["rules"][0]["verified"] = verified
    if legal_status:
        cat["types"]["V05"]["legal_status"] = legal_status
    path = tmp_path / f"catalog-{verified}-{legal_status}.yaml"
    path.write_text(yaml.safe_dump(cat, allow_unicode=True), encoding="utf-8")
    cfg = load_config({"enabled": True, "embed": {"backend": "hashing"},
                       "llm": {"backend": "mock", "model_candidates": []},
                       "corpus": {"catalog_path": str(path), "sources_glob": str(FIXTURE / "sources" / "*.md"),
                                  "fixture_fallback": False}})
    return cfg, load_corpus(cfg)


def _with_vlm_supports(job: dict) -> dict:
    job = copy.deepcopy(job)
    job["meta"]["ui"] = {"evidence_overlays": [{"violation_index": 0, "facts": ["телефон в руке кассира"],
                                                "sources": {"rules": 0.7, "vlm": 0.6}, "vlm": "supports"}]}
    return job


def test_legend_and_three_kinds_are_separated(cfg, corpus, job_violation) -> None:
    md = report(_with_vlm_supports(job_violation), corpus=corpus, cfg=cfg)
    legend = _block(md, "- **Наблюдение системы**")
    assert "не установленный факт" in legend
    assert "**Подтверждённый факт**" in md and "**Непроверенная норма**" in md
    assert "Само нарушение система не подтверждает" in md

    obs = md.split("### 1.", 1)[1]
    observation = _block(obs, "**Наблюдение системы**")
    facts = _block(obs, "**Подтверждённые факты:**")
    unverified = _block(obs, "**Непроверенные нормы**")
    # описание детектора и согласие VLM — наблюдение, не факт
    assert "телефон у уха" in observation and "VLM по кадрам согласна с детектором" in observation
    assert "телефон у уха" not in facts and "VLM" not in facts
    # факт обработки: кадры сохранены; решения человека в данных нет
    assert "00:21, 00:22" in facts and "Решение проверяющего:** в данных нет" in facts
    # нормы фикстуры не сверены -> только в непроверенных, никакой «подтверждённой нормы»
    assert "п. 9" in unverified and "цитата требует сверки с первоисточником" in unverified
    assert "Подтверждённая норма" not in md


def test_verdict_is_labelled_as_observation_not_decision(cfg, corpus, job_violation) -> None:
    md = report(job_violation, corpus=corpus, cfg=cfg)
    processing = _block(md, "## Данные обработки (подтверждённые факты)")
    assert "Статус анализа" in processing and "Вердикт" not in processing and "нарушени" not in processing
    assert "**Вывод системы (наблюдение, не решение):** обнаружены признаки возможного нарушения" in md


def test_confirmed_norm_requires_confirmed_status_and_verified_quote(tmp_path, job_violation) -> None:
    job = {**job_violation, "violations": job_violation["violations"][:1]}
    cfg, corpus = _corpus_with_v05(tmp_path, verified=True, legal_status="confirmed")
    md = report(job, corpus=corpus, cfg=cfg)
    facts = _block(md.split("### 1.", 1)[1], "**Подтверждённые факты:**")
    assert "**Подтверждённая норма:** [ФИКСТУРА: Тестовый регламент кассы, п. 9]" in facts
    assert "**Непроверенные нормы**" not in md

    for verified, status, tag in ((True, "candidate", "норма-кандидат"),
                                  (False, "confirmed", "цитата требует сверки с первоисточником")):
        cfg, corpus = _corpus_with_v05(tmp_path, verified=verified, legal_status=status)
        md = report(job, corpus=corpus, cfg=cfg)
        assert "Подтверждённая норма" not in md, (verified, status)
        assert tag in _block(md.split("### 1.", 1)[1], "**Непроверенные нормы**")


def test_type_without_clause_is_unverified_norm(tmp_path) -> None:
    cfg, corpus = _corpus_with_v05(tmp_path, verified=True, legal_status="confirmed")
    job = {"job_id": "nc", "status": "done", "verdict": "violation", "meta": {},
           "violations": [{"type": "Иное", "type_id": "V99", "timecode_sec": 3.0, "description": "синтетика",
                           "evidence_frames": []}]}
    md = report(job, corpus=corpus, cfg=cfg)
    block = _block(md.split("### 1.", 1)[1], "**Непроверенные нормы**")
    assert "без пункта" in block and "Подтверждённая норма" not in md
    assert "**Кадры-доказательства:** не сохранены" in md


def test_html_keeps_labels(cfg, corpus, job_violation) -> None:
    h = report(job_violation, corpus=corpus, cfg=cfg, fmt="html")
    assert "<strong>Наблюдение системы</strong>" in h and "<strong>Подтверждённые факты:</strong>" in h
    assert "<strong>Непроверенные нормы</strong>" in h and "Как читать заключение" in h
