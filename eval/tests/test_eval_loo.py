"""LOGO / nested CV: оба тюнера (встроенный scalar и адаптер к настоящему ml.fusion.tune A12), без утечки групп."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.loo import (FusionTuneAdapter, ScalarThresholdTuner, TunerContractError, assert_no_leak, logo, main,
                      nested_cv, prepare_items, video_score)


def _cand(conf: float, t: float = 5.0, type_id: str = "V01", **kw) -> dict:
    return {"type_id": type_id, "confidence": conf, "t_peak": t, "t_start": t - 1, "t_end": t + 1,
            "gate": kw.get("gate"), "suppressed_by": kw.get("suppressed_by")}


def _rows() -> list[dict]:
    rows = []
    for i in range(10):
        pos = i % 2 == 0
        conf = 0.8 + i * 0.01 if pos else 0.2 + i * 0.01
        rows.append({"video_id": f"X-{i}", "group": f"X-{i}", "label": "violation" if pos else "no_violation",
                     "gt_times": [5.0] if pos else [], "scores_raw": {"candidates": [_cand(conf)]}})
    # дубликат X-0 (та же группа, другой таймкод — конфликт решает человек)
    rows.append({"video_id": "X-47", "group": "X-0", "label": "violation", "gt_times": [6.0],
                 "scores_raw": {"candidates": [_cand(0.81)]}})
    rows.append({"video_id": "X-90", "group": "X-90", "label": "no_violation", "gt_times": []})          # нет скоров
    rows.append({"video_id": "X-91", "group": "X-91", "label": None, "scores_raw": {"candidates": []}})  # нет метки
    return rows


class Recording:
    """Обёртка тюнера: запоминает, какие группы видел fit, чтобы проверить отсутствие утечки."""

    def __init__(self, inner):
        self.inner, self.name, self.fit_calls = inner, inner.name, []

    def prepare(self, row, item):
        return self.inner.prepare(row, item)

    def fit(self, train):
        self.fit_calls.append({it.key for it in train} | {m for it in train for m in it.members})
        return self.inner.fit(train)

    def predict(self, params, item):
        return self.inner.predict(params, item)


def _check_no_leak(rec: Recording, res: dict, test_key: str) -> None:
    tests = [set(f[test_key]) for f in res["per_fold"]]
    assert len(tests) == len(rec.fit_calls)
    for seen, test in zip(rec.fit_calls, tests):
        assert not (seen & test), f"fit видел тестовую группу: {seen & test}"


def test_video_score_semantics():
    assert video_score({"scores_raw": {"candidates": [_cand(0.9, gate="risk_only"), _cand(0.4)]}}) == 0.4
    assert video_score({"scores_raw": {"candidates": [_cand(0.9, suppressed_by="V02")]}}) == 0.0
    assert video_score({"scores_raw": {"candidates": []}}) == 0.0         # кандидатов нет — честный 0
    assert video_score({"scores_raw": {"max_conf": 0.7}}) == 0.7
    assert video_score({"violations": [{"confidence": 0.9}]}) is None      # нет scores_raw — скора нет, не 0
    assert video_score({}) is None


@pytest.mark.parametrize("proc", ["logo", "nested"])
def test_scalar_tuner_no_leak(proc):
    rec = Recording(ScalarThresholdTuner())
    res = logo(_rows(), rec) if proc == "logo" else nested_cv(_rows(), rec, folds=3)
    _check_no_leak(rec, res, "test")
    assert res["n"] == 10 and res["recall"] == 1.0 and res["fp"] == 0
    reasons = {e["key"]: e["reason"] for e in res["excluded"]}
    assert "X-90" in reasons and "скалярного скора" in reasons["X-90"] and "X-91" in reasons
    assert res["tuner"] == "scalar-threshold" and res["leakage_check"] == "ok"
    if proc == "logo":
        assert res["status"] == "exploratory" and "не независимый" in res["note"]
    else:
        assert res["status"] == "nested_cv" and "outer-train" in res["note"]


def test_duplicate_group_is_one_unit():
    items, _ = prepare_items(_rows(), ScalarThresholdTuner())
    x0 = next(it for it in items if it.key == "X-0")
    assert x0.members == ["X-0", "X-47"] and x0.gt_times == [5.0, 6.0]
    with pytest.raises(AssertionError):
        assert_no_leak([x0], [x0])


def test_label_conflict_group_excluded():
    rows = _rows() + [{"video_id": "X-48", "group": "X-1", "label": "violation", "gt_times": [3.0],
                       "scores_raw": {"candidates": [_cand(0.9)]}}]
    _, excluded = prepare_items(rows, ScalarThresholdTuner())
    assert any(e["key"] == "X-1" and "конфликт меток" in e["reason"] for e in excluded)


def test_tuner_exceptions_not_swallowed():
    class Boom(ScalarThresholdTuner):
        def fit(self, train):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        logo(_rows(), Boom())
    with pytest.raises(RuntimeError, match="boom"):
        nested_cv(_rows(), Boom())


def test_fusion_adapter_rejects_wrong_contract():
    """Чужую функцию не выбираем по имени: fit(rows) без start/default → TunerContractError, не fallback."""
    import types as _t
    from dataclasses import dataclass

    @dataclass
    class Sample:
        video: str
        positive: bool
        seconds: list
        cands: list
        group: str

    @dataclass
    class VideoPred:
        video: str
        raw: dict

    old = _t.SimpleNamespace(__name__="fake_tune", fit=lambda rows: {}, predict=lambda params, row: True,
                             candidates_for=lambda pred, cfg, types: ([], 0.0), Sample=Sample, VideoPred=VideoPred)
    with pytest.raises(TunerContractError, match="samples"):
        FusionTuneAdapter(module=old, config={}, thresholds={}, types={})

    def fit(samples, start, default, **kw):
        return {"V01": 7.0}                                           # не порог 0..1

    def predict(s, thr, default):
        return []

    def candidates_for(pred, cfg, types):
        return [("V01", 0.9, 5.0, 4.0)], 0.0

    bad = _t.SimpleNamespace(__name__="fake_tune", fit=fit, predict=predict, candidates_for=candidates_for,
                             Sample=Sample, VideoPred=VideoPred)
    ad = FusionTuneAdapter(module=bad, config={}, thresholds={"default": 0.45}, types={})
    with pytest.raises(TypeError):        # Sample без conflict/type_id/members — несовпадение не маскируется
        prepare_items(_rows(), ad)


def test_fusion_adapter_real_a12_no_leak(monkeypatch):
    """Настоящий ml.fusion.tune (A12): fit(samples, start, default, …) / predict(s, thr, default)."""
    tune = pytest.importorskip("ml.fusion.tune", reason="ml.fusion (A12) нет в этой ветке — прогон в интеграции")
    adapter = FusionTuneAdapter()
    seen: list[set] = []
    real_fit = tune.fit

    def recording_fit(samples, start, default, **kw):
        seen.append({s.group for s in samples} | {m for s in samples for m in (s.members or [s.video])})
        return real_fit(samples, start, default, **kw)

    monkeypatch.setattr(adapter.t, "fit", recording_fit)
    for proc in ("logo", "nested"):
        seen.clear()
        res = logo(_rows(), adapter) if proc == "logo" else nested_cv(_rows(), adapter, folds=3)
        tests = [set(f["test"]) for f in res["per_fold"]]
        assert len(seen) == len(tests)
        for s, t in zip(seen, tests):
            assert not (s & t)
        assert res["tuner"] == "ml.fusion.tune" and res["n"] == 10
        assert res["recall"] == 1.0 and res["fp"] == 0
        assert all(isinstance(f["params"], dict) for f in res["per_fold"])


def test_cli_fusion_contract_error_exit_code(tmp_path: Path, monkeypatch, capsys):
    import eval.loo as loo

    p = tmp_path / "scores_raw.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in _rows()), encoding="utf-8")

    def boom(name, **kw):
        raise TunerContractError("контракт не совпал")

    monkeypatch.setattr(loo, "make_tuner", boom)
    assert main(["--scores", str(p), "--tuner", "fusion"]) == 2
    assert "контракт не совпал" in capsys.readouterr().err
    monkeypatch.undo()
    assert main(["--scores", str(p), "--tuner", "scalar"]) == 0
    out = json.loads((tmp_path / "loo_scalar.json").read_text(encoding="utf-8"))
    assert out["logo"]["status"] == "exploratory" and out["nested_cv"]["leakage_check"] == "ok"


def test_too_few_groups_is_explicit_error(tmp_path: Path, capsys):
    rows = _rows()[:2]                                   # 1 позитив + 1 негатив: после LOGO в train один класс
    with pytest.raises(ValueError, match="обоих классов"):
        logo(rows, ScalarThresholdTuner())
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert main(["--scores", str(p)]) == 2 and "не выполнены" in capsys.readouterr().err
    assert not (tmp_path / "loo_scalar.json").exists()   # никакого подменного результата
