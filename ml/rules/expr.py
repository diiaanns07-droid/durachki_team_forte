"""Безопасный векторный парсер выражений правил (никакого ``eval``) с интервальной логикой.

Выражение из ``rules.yaml`` разбирается через :mod:`ast`, проверяется по белому списку
узлов и компилируется в дерево замыканий, которое считает значение сразу для всего
временного ряда (numpy-массивы формы ``(N,)``).

**Отсутствие сигнала ≠ отрицание.** Неизвестное значение (сигнал не пришёл, окклюзия,
пропуск дольше forward-fill) — это интервал ``[lo, hi]`` допустимого диапазона сигнала
(для 0..1 — ``[0, 1]``). Каждый узел возвращает пару ``(lo, hi)``:

* ``lo`` — что *доказано* наблюдениями (используется для событий);
* ``hi`` — что *возможно* при любых значениях неизвестных (используется для
  ``insufficient_context``: «условие могло выполняться, но не наблюдалось»).

Семантика (нечёткая логика, значения сигналов 0..1):

* ``a and b`` → ``min``, ``a or b`` → ``max``, ``not a`` → ``1 - a`` (операнды обрезаются в 0..1;
  для интервалов ``not [lo, hi] = [1-hi, 1-lo]``);
* сравнения → 0/1 (``lo`` = выполняется при любых неизвестных, ``hi`` = может выполняться);
* арифметика ``+ - * /`` (деление на 0 → 0), унарный минус;
* точечные функции: ``min``, ``max``, ``abs``, ``clip(x, lo, hi)``, ``ramp(x, lo, hi)``,
  ``step(x, thr)``;
* временные функции (окно — константа в секундах):
  ``was(x, w)`` — максимум за последние ``w`` с, ``held(x, w)`` — минимум за ``w`` с
  (при истории короче ``w`` нижняя граница = 0: «держится» не доказано), ``avg(x, w)`` —
  среднее, ``ahead(x, w)`` — максимум на ``w`` с вперёд, ``ever(x)`` — максимум с начала,
  ``prev_min(x, w)`` / ``prev_max(x, w)`` — минимум/максимум строго в прошлом ``[t-w, t)``
  (без прошлого — неизвестно); например вход: ``n_persons >= prev_min(n_persons, 3) + 1``.

Имена — сигналы из ``SIGNALS`` (или любые другие ключи ``SceneState.signals``) и
константы ``true``/``false``. Атрибуты, индексы, строки, лямбды, comprehension-ы,
вызовы чего-либо кроме белого списка — запрещены (``ExprError`` на этапе компиляции).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np

__all__ = [
    "ExprError",
    "EvalContext",
    "CompiledExpr",
    "Comparison",
    "compile_expr",
    "evaluate_scalar",
    "POINTWISE_FUNCS",
    "TEMPORAL_FUNCS",
]

MAX_EXPR_LEN = 2000
MAX_NODES = 300

POINTWISE_FUNCS = frozenset({"min", "max", "abs", "clip", "ramp", "step"})
TEMPORAL_FUNCS = frozenset({"was", "held", "avg", "ahead", "ever", "prev_min", "prev_max"})
_UNBOUNDED = 1e9  # «неизвестно» для выражений без диапазона (нет прошлого у prev_*)
_CONSTANT_NAMES = {"true": 1.0, "false": 0.0, "True": 1.0, "False": 0.0}

_CMP_SYMBOL = {
    ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!=",
}
_ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Compare, ast.Lt, ast.LtE, ast.Gt,
    ast.GtE, ast.Eq, ast.NotEq, ast.Name, ast.Load, ast.Constant, ast.Call,
)

Interval = tuple[Any, Any]  # (lo, hi): float или np.ndarray


class ExprError(ValueError):
    """Ошибка разбора/проверки выражения правила."""


@dataclass
class EvalContext:
    """Контекст вычисления: общая временная ось и ряды сигналов (NaN = неизвестно)."""

    times: np.ndarray
    signals: Mapping[str, np.ndarray]
    ranges: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    default_range: tuple[float, float] = (0.0, 1.0)
    _window_cache: dict = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return int(self.times.shape[0])

    def get(self, name: str) -> Interval:
        lo_r, hi_r = self.ranges.get(name, self.default_range)
        arr = self.signals.get(name)
        if arr is None:
            return np.full(self.n, float(lo_r)), np.full(self.n, float(hi_r))
        unknown = ~np.isfinite(arr)
        if not unknown.any():
            return arr, arr
        return np.where(unknown, lo_r, arr), np.where(unknown, hi_r, arr)

    def left_index(self, window_s: float) -> np.ndarray:
        key = ("L", float(window_s))
        if key not in self._window_cache:
            self._window_cache[key] = np.searchsorted(self.times, self.times - window_s - 1e-9, side="left")
        return self._window_cache[key]

    def right_index(self, window_s: float) -> np.ndarray:
        key = ("R", float(window_s))
        if key not in self._window_cache:
            self._window_cache[key] = np.searchsorted(self.times, self.times + window_s + 1e-9, side="right")
        return self._window_cache[key]


@dataclass(frozen=True)
class Comparison:
    """Сравнение вида ``signal <op> константа`` — для объяснимости (trace)."""

    name: str
    op: str
    threshold: float

    def check(self, value: float) -> Optional[bool]:
        """True/False по наблюдённому значению; ``None`` — значение неизвестно."""
        if value is None or not np.isfinite(value):
            return None
        lo, _ = _cmp(self.op, (float(value), float(value)), (self.threshold, self.threshold))
        return bool(np.asarray(lo) > 0.5)


def _finite(x: Any, n: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    return np.nan_to_num(arr, nan=0.0, posinf=1e9, neginf=-1e9)


@dataclass
class CompiledExpr:
    source: str
    names: frozenset[str]
    funcs: frozenset[str]
    comparisons: tuple[Comparison, ...]
    max_window_s: float
    _fn: Callable[[EvalContext], Interval] = field(repr=False)

    def bounds(self, ctx: EvalContext) -> tuple[np.ndarray, np.ndarray]:
        """(доказано, возможно) для всего ряда."""
        lo, hi = self._fn(ctx)
        lo, hi = _finite(lo, ctx.n), _finite(hi, ctx.n)
        return np.minimum(lo, hi), np.maximum(lo, hi)

    def __call__(self, ctx: EvalContext) -> np.ndarray:
        """Нижняя (доказанная наблюдениями) граница."""
        return self.bounds(ctx)[0]


# --------------------------------------------------------------------------- interval ops

def _truth(iv: Interval) -> Interval:
    return np.clip(iv[0], 0.0, 1.0), np.clip(iv[1], 0.0, 1.0)


def _not(iv: Interval) -> Interval:
    lo, hi = _truth(iv)
    return 1.0 - hi, 1.0 - lo


def _cmp(op: str, a: Interval, b: Interval) -> Interval:
    a0, a1 = a
    b0, b1 = b
    if op == "<":
        lo, hi = np.less(a1, b0), np.less(a0, b1)
    elif op == "<=":
        lo, hi = np.less_equal(a1, b0), np.less_equal(a0, b1)
    elif op == ">":
        lo, hi = np.greater(a0, b1), np.greater(a1, b0)
    elif op == ">=":
        lo, hi = np.greater_equal(a0, b1), np.greater_equal(a1, b0)
    elif op in ("==", "!="):
        definite_eq = np.isclose(a0, a1) & np.isclose(b0, b1) & np.isclose(a0, b0)
        overlap = np.less_equal(a0, np.add(b1, 1e-9)) & np.less_equal(b0, np.add(a1, 1e-9))
        lo, hi = (definite_eq, overlap) if op == "==" else (~overlap, ~definite_eq)
    else:  # pragma: no cover - защищено белым списком
        raise ExprError(f"Недопустимый оператор сравнения: {op}")
    return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)


def _safe_div_scalar(a: Any, b: Any) -> Any:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    shape = np.broadcast(a, b).shape
    out = np.zeros(shape, dtype=float)
    np.divide(np.broadcast_to(a, shape), np.broadcast_to(b, shape), out=out, where=np.broadcast_to(b != 0, shape))
    return out


def _mul(a: Interval, b: Interval) -> Interval:
    prods = [np.multiply(x, y) for x in a for y in b]
    return np.minimum.reduce(np.broadcast_arrays(*prods)), np.maximum.reduce(np.broadcast_arrays(*prods))


def _div(a: Interval, b: Interval) -> Interval:
    qs = [_safe_div_scalar(x, y) for x in a for y in b]
    lo = np.minimum.reduce(np.broadcast_arrays(*qs))
    hi = np.maximum.reduce(np.broadcast_arrays(*qs))
    touches_zero = np.not_equal(b[0], b[1]) & (np.equal(b[0], 0) | np.equal(b[1], 0))
    straddle = (np.less(b[0], 0) & np.greater(b[1], 0)) | touches_zero
    if np.any(straddle):
        lo = np.where(straddle, -np.inf, lo)
        hi = np.where(straddle, np.inf, hi)
    return lo, hi


def _abs(a: Interval) -> Interval:
    lo, hi = np.asarray(a[0], float), np.asarray(a[1], float)
    out_lo = np.where(lo >= 0, lo, np.where(hi <= 0, -hi, 0.0))
    out_hi = np.maximum(np.abs(lo), np.abs(hi))
    return out_lo, out_hi


def _ramp(x: Any, a: Any, b: Any) -> Any:
    return np.clip(_safe_div_scalar(np.subtract(x, a), np.subtract(b, a)), 0.0, 1.0)


def _as_array(x: Any, n: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    return arr


def _window_max(ctx: EvalContext, arr: np.ndarray, window_s: float, forward: bool = False) -> np.ndarray:
    n = ctx.n
    out = np.empty(n)
    if forward:
        right = ctx.right_index(window_s)
        for i in range(n):
            out[i] = arr[i:max(i + 1, int(right[i]))].max()
    else:
        left = ctx.left_index(window_s)
        for i in range(n):
            out[i] = arr[int(left[i]):i + 1].max()
    return out


def _window_min(ctx: EvalContext, arr: np.ndarray, window_s: float) -> np.ndarray:
    left = ctx.left_index(window_s)
    out = np.empty(ctx.n)
    for i in range(ctx.n):
        out[i] = arr[int(left[i]):i + 1].min()
    return out


def _window_mean(ctx: EvalContext, arr: np.ndarray, window_s: float) -> np.ndarray:
    left = ctx.left_index(window_s)
    cs = np.concatenate([[0.0], np.cumsum(arr)])
    idx = np.arange(ctx.n)
    cnt = idx + 1 - left
    return (cs[idx + 1] - cs[left]) / np.maximum(cnt, 1)


def _temporal(ctx: EvalContext, iv: Interval, window_s: float, kind: str) -> Interval:
    n = ctx.n
    lo, hi = _as_array(iv[0], n), _as_array(iv[1], n)
    if n == 0:
        return lo, hi
    if kind == "ever":
        return np.maximum.accumulate(lo), np.maximum.accumulate(hi)
    if kind == "was":
        return _window_max(ctx, lo, window_s), _window_max(ctx, hi, window_s)
    if kind == "ahead":
        return _window_max(ctx, lo, window_s, True), _window_max(ctx, hi, window_s, True)
    if kind == "avg":
        return _window_mean(ctx, lo, window_s), _window_mean(ctx, hi, window_s)
    if kind in ("prev_min", "prev_max"):
        left = ctx.left_index(window_s)
        red = np.min if kind == "prev_min" else np.max
        out_lo = np.full(n, -_UNBOUNDED)
        out_hi = np.full(n, _UNBOUNDED)
        for i in range(n):
            j0 = int(left[i])
            if i > j0:
                out_lo[i] = red(lo[j0:i])
                out_hi[i] = red(hi[j0:i])
        return out_lo, out_hi
    if kind == "held":
        short = (ctx.times - float(ctx.times[0])) < window_s - 1e-6
        out_lo = np.where(short, 0.0, _window_min(ctx, lo, window_s))  # истории меньше окна — не доказано
        return out_lo, _window_min(ctx, hi, window_s)
    raise ExprError(f"Неизвестная временная функция: {kind}")  # pragma: no cover


# --------------------------------------------------------------------------- compiler

class _Compiler:
    def __init__(self, allowed_names: Optional[set[str]]) -> None:
        self.allowed_names = allowed_names
        self.names: set[str] = set()
        self.funcs: set[str] = set()
        self.comparisons: list[Comparison] = []
        self.max_window = 0.0

    @staticmethod
    def _const_number(node: ast.AST, what: str) -> float:
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_Compiler._const_number(node.operand, what)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        raise ExprError(f"{what}: ожидалась числовая константа")

    def compile(self, node: ast.AST) -> Callable[[EvalContext], Interval]:
        if isinstance(node, ast.Expression):
            return self.compile(node.body)
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool):
                fv = 1.0 if v else 0.0
            elif isinstance(v, (int, float)):
                fv = float(v)
            else:
                raise ExprError(f"Недопустимая константа: {v!r}")
            return lambda ctx, fv=fv: (fv, fv)
        if isinstance(node, ast.Name):
            name = node.id
            if name in _CONSTANT_NAMES:
                fv = _CONSTANT_NAMES[name]
                return lambda ctx, fv=fv: (fv, fv)
            if name.startswith("_"):
                raise ExprError(f"Недопустимое имя: {name}")
            if self.allowed_names is not None and name not in self.allowed_names:
                raise ExprError(f"Неизвестный сигнал: {name}")
            self.names.add(name)
            return lambda ctx, name=name: ctx.get(name)
        if isinstance(node, ast.BoolOp):
            parts = [self.compile(v) for v in node.values]
            red = np.minimum if isinstance(node.op, ast.And) else np.maximum

            def _bool(ctx: EvalContext, parts=parts, red=red) -> Interval:
                lo, hi = _truth(parts[0](ctx))
                for p in parts[1:]:
                    plo, phi = _truth(p(ctx))
                    lo, hi = red(lo, plo), red(hi, phi)
                return lo, hi
            return _bool
        if isinstance(node, ast.UnaryOp):
            inner = self.compile(node.operand)
            if isinstance(node.op, ast.Not):
                return lambda ctx, inner=inner: _not(inner(ctx))
            if isinstance(node.op, ast.USub):
                def _neg(ctx: EvalContext, inner=inner) -> Interval:
                    lo, hi = inner(ctx)
                    return np.negative(hi), np.negative(lo)
                return _neg
            if isinstance(node.op, ast.UAdd):
                return inner
            raise ExprError("Недопустимый унарный оператор")
        if isinstance(node, ast.BinOp):
            a, b = self.compile(node.left), self.compile(node.right)
            if isinstance(node.op, ast.Add):
                def _add(ctx: EvalContext, a=a, b=b) -> Interval:
                    (a0, a1), (b0, b1) = a(ctx), b(ctx)
                    return np.add(a0, b0), np.add(a1, b1)
                return _add
            if isinstance(node.op, ast.Sub):
                def _sub(ctx: EvalContext, a=a, b=b) -> Interval:
                    (a0, a1), (b0, b1) = a(ctx), b(ctx)
                    return np.subtract(a0, b1), np.subtract(a1, b0)
                return _sub
            if isinstance(node.op, ast.Mult):
                return lambda ctx, a=a, b=b: _mul(a(ctx), b(ctx))
            if isinstance(node.op, ast.Div):
                return lambda ctx, a=a, b=b: _div(a(ctx), b(ctx))
            raise ExprError("Недопустимый арифметический оператор")
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            fns = [self.compile(o) for o in operands]
            ops = []
            for op_node, left, right in zip(node.ops, operands[:-1], operands[1:]):
                sym = _CMP_SYMBOL.get(type(op_node))
                if sym is None:
                    raise ExprError("Недопустимый оператор сравнения")
                ops.append(sym)
                self._record_comparison(left, sym, right)

            def _compare(ctx: EvalContext, fns=fns, ops=ops) -> Interval:
                vals = [f(ctx) for f in fns]
                lo: Any = 1.0
                hi: Any = 1.0
                for i, sym in enumerate(ops):
                    clo, chi = _cmp(sym, vals[i], vals[i + 1])
                    lo, hi = np.minimum(lo, clo), np.minimum(hi, chi)
                return lo, hi
            return _compare
        if isinstance(node, ast.Call):
            return self._compile_call(node)
        raise ExprError(f"Недопустимая конструкция: {type(node).__name__}")

    def _record_comparison(self, left: ast.AST, sym: str, right: ast.AST) -> None:
        flip = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}
        try:
            if isinstance(left, ast.Name) and left.id not in _CONSTANT_NAMES:
                self.comparisons.append(Comparison(left.id, sym, self._const_number(right, "")))
            elif isinstance(right, ast.Name) and right.id not in _CONSTANT_NAMES:
                self.comparisons.append(Comparison(right.id, flip[sym], self._const_number(left, "")))
        except ExprError:
            pass  # сравнение двух сигналов — в trace не выносим

    def _compile_call(self, node: ast.Call) -> Callable[[EvalContext], Interval]:
        if not isinstance(node.func, ast.Name):
            raise ExprError("Вызывать можно только функции из белого списка")
        fname = node.func.id
        if node.keywords:
            raise ExprError(f"{fname}(): именованные аргументы не поддерживаются")
        args = node.args
        if fname in POINTWISE_FUNCS:
            self.funcs.add(fname)
            fns = [self.compile(a) for a in args]
            if fname in ("min", "max"):
                if not fns:
                    raise ExprError(f"{fname}(): нужен хотя бы один аргумент")
                red = np.minimum if fname == "min" else np.maximum

                def _mm(ctx: EvalContext, fns=fns, red=red) -> Interval:
                    lo, hi = fns[0](ctx)
                    for f in fns[1:]:
                        flo, fhi = f(ctx)
                        lo, hi = red(lo, flo), red(hi, fhi)
                    return lo, hi
                return _mm
            if fname == "abs":
                if len(fns) != 1:
                    raise ExprError("abs(x): ровно один аргумент")
                return lambda ctx, f=fns[0]: _abs(f(ctx))
            if fname == "clip":
                if len(fns) != 3:
                    raise ExprError("clip(x, lo, hi): ровно три аргумента")

                def _clip(ctx: EvalContext, f=fns) -> Interval:
                    (x0, x1), (l0, l1), (h0, h1) = f[0](ctx), f[1](ctx), f[2](ctx)
                    return np.clip(x0, l0, h0), np.clip(x1, l1, h1)
                return _clip
            if fname == "ramp":
                if len(fns) != 3:
                    raise ExprError("ramp(x, lo, hi): ровно три аргумента")
                if not all(isinstance(a, (ast.Constant, ast.UnaryOp)) for a in args[1:]):
                    raise ExprError("ramp(x, lo, hi): границы должны быть константами")
                lo_c = self._const_number(args[1], "ramp(): lo")
                hi_c = self._const_number(args[2], "ramp(): hi")
                if hi_c <= lo_c:
                    raise ExprError("ramp(x, lo, hi): нужно hi > lo")
                return lambda ctx, f=fns[0], a=lo_c, b=hi_c: tuple(_ramp(v, a, b) for v in f(ctx))  # type: ignore[return-value]
            if fname == "step":
                if len(fns) != 2:
                    raise ExprError("step(x, thr): ровно два аргумента")
                return lambda ctx, f=fns: _cmp(">=", f[0](ctx), f[1](ctx))
        if fname in TEMPORAL_FUNCS:
            self.funcs.add(fname)
            if fname == "ever":
                if len(args) != 1:
                    raise ExprError("ever(x): ровно один аргумент")
                inner = self.compile(args[0])
                return lambda ctx, inner=inner: _temporal(ctx, inner(ctx), 0.0, "ever")
            if len(args) != 2:
                raise ExprError(f"{fname}(x, окно_с): ровно два аргумента")
            inner = self.compile(args[0])
            window = self._const_number(args[1], f"{fname}(): окно")
            if not (0.0 <= window <= 3600.0):
                raise ExprError(f"{fname}(): окно должно быть в диапазоне 0..3600 с")
            self.max_window = max(self.max_window, window)
            return lambda ctx, inner=inner, w=window, k=fname: _temporal(ctx, inner(ctx), w, k)
        raise ExprError(f"Функция не разрешена: {fname}")


def _check_tree(tree: ast.AST) -> None:
    count = 0
    for node in ast.walk(tree):
        count += 1
        if count > MAX_NODES:
            raise ExprError("Выражение слишком сложное")
        if not isinstance(node, _ALLOWED_NODES):
            raise ExprError(f"Недопустимая конструкция: {type(node).__name__}")


def compile_expr(source: Any, allowed_names: Optional[Iterable[str]] = None) -> CompiledExpr:
    """Скомпилировать выражение правила. Число/bool допускаются как константное выражение."""
    if isinstance(source, bool):
        source = "true" if source else "false"
    elif isinstance(source, (int, float)):
        source = repr(float(source))
    if not isinstance(source, str):
        raise ExprError(f"Выражение должно быть строкой, получено {type(source).__name__}")
    text = " ".join(source.split())
    if not text:
        raise ExprError("Пустое выражение")
    if len(text) > MAX_EXPR_LEN:
        raise ExprError("Выражение слишком длинное")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ExprError(f"Синтаксическая ошибка в выражении {text!r}: {exc.msg}") from None
    _check_tree(tree)
    comp = _Compiler(set(allowed_names) if allowed_names is not None else None)
    fn = comp.compile(tree)
    return CompiledExpr(
        source=text,
        names=frozenset(comp.names),
        funcs=frozenset(comp.funcs),
        comparisons=tuple(comp.comparisons),
        max_window_s=comp.max_window,
        _fn=fn,
    )


def evaluate_scalar(expr: CompiledExpr | str, variables: Mapping[str, float]) -> float:
    """Вычислить выражение для одного набора известных скалярных переменных (формулы confidence)."""
    compiled = compile_expr(expr) if isinstance(expr, str) else expr
    missing = sorted(n for n in compiled.names if n not in variables)
    if missing:
        raise ExprError(f"Нет переменных для формулы: {', '.join(missing)}")
    ctx = EvalContext(
        times=np.zeros(1),
        signals={k: np.asarray([float(v)], dtype=float) for k, v in variables.items()},
    )
    return float(compiled(ctx)[0])
