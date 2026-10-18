"""Безопасный парсер выражений: белый список, нечёткая и интервальная семантика, окна."""
from __future__ import annotations

import numpy as np
import pytest

from ml.rules.expr import EvalContext, ExprError, compile_expr, evaluate_scalar


def _ctx(**signals: list[float]) -> EvalContext:
    n = len(next(iter(signals.values())))
    return EvalContext(times=np.arange(n) * 0.2, signals={k: np.asarray(v, float) for k, v in signals.items()})


@pytest.mark.parametrize("bad", [
    "__import__('os').system('echo pwned')",
    "a.__class__",
    "a[0]",
    "lambda: 1",
    "'text'",
    "open('x')",
    "eval('1')",
    "[x for x in a]",
    "a if b else 1",
    "a ** 2",
    "a % 2",
    "_private",
    "min(a, key=b)",
    "was(a, b)",          # окно должно быть константой
    "ramp(a, b, 1)",      # границы ramp — константы
    "",
    "a +",
])
def test_rejects_unsafe_or_invalid(bad: str) -> None:
    with pytest.raises(ExprError):
        compile_expr(bad)


def test_rejects_unknown_names_when_whitelisted() -> None:
    with pytest.raises(ExprError):
        compile_expr("foo > 1", allowed_names={"bar"})


def test_fuzzy_semantics_known_values() -> None:
    ctx = _ctx(a=[0.0, 0.7, 1.0], b=[0.3, 0.3, 0.3])
    assert np.allclose(compile_expr("a and b")(ctx), [0.0, 0.3, 0.3])
    assert np.allclose(compile_expr("a or b")(ctx), [0.3, 0.7, 1.0])
    assert np.allclose(compile_expr("not a")(ctx), [1.0, 0.3, 0.0])
    assert np.allclose(compile_expr("a > 0.5 and b < 0.5")(ctx), [0.0, 1.0, 1.0])
    assert np.allclose(compile_expr("0.2 < b < 0.8")(ctx), [1.0, 1.0, 1.0])
    assert np.allclose(compile_expr("ramp(a, 0.5, 1.0)")(ctx), [0.0, 0.4, 1.0])
    assert np.allclose(compile_expr("min(a, b) / 0")(ctx), [0.0, 0.0, 0.0])
    assert np.allclose(compile_expr("clip(a * 2, 0, 1)")(ctx), [0.0, 1.0, 1.0])
    assert np.allclose(compile_expr("step(a, 0.7)")(ctx), [0.0, 1.0, 1.0])
    assert np.allclose(compile_expr("abs(a - 1)")(ctx), [1.0, 0.3, 0.0])


def test_unknown_is_not_negation() -> None:
    nan = float("nan")
    ctx = _ctx(safe=[1.0, 1.0, 0.0], cashier=[nan, 0.0, nan])
    lo, hi = compile_expr("min(safe, not cashier)").bounds(ctx)
    # кадр 0: сейф открыт, кассир неизвестен → не доказано (lo=0), но возможно (hi=1)
    assert lo[0] == 0.0 and hi[0] == 1.0
    # кадр 1: кассир доказанно отсутствует → доказано
    assert lo[1] == 1.0 and hi[1] == 1.0
    # кадр 2: сейф доказанно закрыт → AND доказанно ложен, неизвестный кассир не важен
    assert lo[2] == 0.0 and hi[2] == 0.0


def test_unknown_in_comparisons_and_or() -> None:
    nan = float("nan")
    ctx = _ctx(a=[nan, 1.0], b=[1.0, nan])
    lo, hi = compile_expr("a > 0.5 or b > 0.5").bounds(ctx)
    assert list(lo) == [1.0, 1.0]  # OR: известная истина достаточна
    lo, hi = compile_expr("a > 0.5 and b > 0.5").bounds(ctx)
    assert list(lo) == [0.0, 0.0] and list(hi) == [1.0, 1.0]
    lo, hi = compile_expr("not (a > 0.5)").bounds(ctx)
    assert lo[0] == 0.0 and hi[0] == 1.0


def test_missing_signal_entirely_unknown_with_range() -> None:
    ctx = EvalContext(times=np.arange(3) * 0.2, signals={}, ranges={"n_persons": (0.0, 10.0)})
    lo, hi = compile_expr("n_persons >= 3").bounds(ctx)
    assert list(lo) == [0.0, 0.0, 0.0] and list(hi) == [1.0, 1.0, 1.0]


def test_temporal_functions() -> None:
    t = np.arange(0, 10, 0.2)
    ctx = EvalContext(times=t, signals={"a": (t >= 3.0).astype(float)})
    i3 = int(np.argmin(np.abs(t - 3.0)))
    assert compile_expr("was(a, 2)")(ctx)[i3] == 1.0
    assert compile_expr("was(a, 2)")(ctx)[i3 - 1] == 0.0
    held = compile_expr("held(a, 2)")(ctx)
    assert held[i3] == 0.0 and held[int(np.argmin(np.abs(t - 5.0)))] == 1.0
    assert compile_expr("ahead(a, 1)")(ctx)[i3 - 5] == 1.0
    assert compile_expr("ever(a)")(ctx)[-1] == 1.0
    avg = compile_expr("avg(a, 1)")(ctx)
    assert 0.0 < avg[i3 + 2] < 1.0
    # held на начале ролика: истории меньше окна → не доказано
    ctx2 = EvalContext(times=t, signals={"a": np.ones_like(t)})
    assert compile_expr("held(a, 2)")(ctx2)[0] == 0.0


def test_comparisons_recorded_for_trace() -> None:
    e = compile_expr("safe_open > 0.5 and 0.3 >= cashier_present")
    pairs = {(c.name, c.op, c.threshold) for c in e.comparisons}
    assert ("safe_open", ">", 0.5) in pairs and ("cashier_present", "<=", 0.3) in pairs
    c = next(c for c in e.comparisons if c.name == "safe_open")
    assert c.check(0.9) is True and c.check(0.1) is False and c.check(float("nan")) is None
    assert e.names == frozenset({"safe_open", "cashier_present"})


def test_evaluate_scalar() -> None:
    assert evaluate_scalar("base + w * s", {"base": 0.3, "w": 0.5, "s": 0.8}) == pytest.approx(0.7)
    with pytest.raises(ExprError):
        evaluate_scalar("base + x", {"base": 1.0})
