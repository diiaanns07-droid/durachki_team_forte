"""Кросс-валидация порогов вердикта по сырым скорам (scores_raw.jsonl из eval.run --raw): LOGO и nested CV.

* LOGO по данным, на ошибках которых много раз меняли пайплайн, — **exploratory**, НЕ независимый тест;
* nested CV: внешние групповые стратифицированные фолды, подбор порога только на outer-train → оценка
  процедуры подбора; итоговые цифры — только замороженный hold-out (eval.extra) с --final.

Тюнер выбирается ЯВНО (``--tuner``), чужая функция не подхватывается по имени:

* ``scalar`` (по умолчанию) — встроенный порог по скалярному скору видео: max ``confidence`` кандидатов
  ``scores_raw.candidates`` без gate/suppressed (как у A12), либо явный ``max_conf``/``max``/``score``;
  видео без scores_raw — исключается с причиной (нет скора ≠ скор 0);
* ``fusion`` — адаптер к ``ml.fusion.tune`` (A12): строки → ``tune.Sample`` через ``tune.candidates_for``,
  пороги по типам ``tune.fit(samples, start, default, **kw)``, вердикт ``bool(tune.predict(sample, thr, default))``,
  стартовые пороги — ``ml.fusion.config.load_thresholds``. Сигнатуры и форма результатов проверяются;
  несовпадение → ``TunerContractError`` (без подмены встроенным тюнером). Исключения тюнера не глушатся.

Единица оценки — группа (md5-дубликаты Х-4≡Х-47 — одна группа, всегда целиком в train или в test).

CLI::

    python -m eval.loo --scores runs/eval/dev1/scores_raw.jsonl [--tuner scalar|fusion] [--folds 5]
"""
from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

from .common import write_json

EXPLORATORY = "exploratory: LOGO на данных, по ошибкам которых настраивали пайплайн, — не независимый тест"
NESTED_NOTE = "nested CV: порог подбирается только на outer-train; оценка процедуры подбора, не hold-out"
LABELS = ("violation", "no_violation")


class TunerContractError(RuntimeError):
    """Контракт тюнера (A12) не совпал с ожидаемым — нужна правка адаптера/владельца, не тихий fallback."""


@dataclass
class Item:
    key: str                      # группа — единица оценки
    video_id: str                 # представитель группы
    members: list[str]
    positive: bool
    gt_times: list[float]
    payload: Any = None           # скаляр (scalar) или tune.Sample (fusion)


class Tuner(Protocol):
    name: str

    def prepare(self, row: dict, item: Item) -> tuple[Any, Optional[str]]: ...   # (payload, причина исключения)

    def fit(self, train: list[Item]) -> Any: ...

    def predict(self, params: Any, item: Item) -> bool: ...


# ----------------------------------------------------------------------------- скаляр
def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def video_score(row: dict) -> Optional[float]:
    """Скалярный скор видео из scores_raw. None — скора нет (видео исключается, а не получает 0)."""
    sr = row.get("scores_raw")
    if isinstance(sr, list):
        sr = {"candidates": sr}
    if not isinstance(sr, dict):
        return None
    cands = sr.get("candidates")
    if isinstance(cands, list):
        vals = []
        for c in cands:
            if not isinstance(c, dict) or c.get("gate") or c.get("suppressed_by"):
                continue
            v = _finite(c.get("confidence", c.get("score")))
            if v is not None and v > 0:
                vals.append(v)
        return max(vals) if vals else 0.0     # список кандидатов есть, но пуст/всё отсечено → честный 0
    for k in ("max_conf", "max", "score"):
        v = _finite(sr.get(k))
        if v is not None:
            return v
    return None


def fit_threshold(scores: list[float], labels: list[bool], fn_cost: float = 2.0, fp_cost: float = 1.0) -> float:
    """Порог с минимальной ценой ошибок (пропуск дороже), середина между соседними скорами."""
    pts = sorted(set(scores))
    if not pts:
        raise ValueError("нет скоров для подбора порога")
    cands = [pts[0] - 1e-6] + [(a + b) / 2 for a, b in zip(pts, pts[1:])] + [pts[-1] + 1e-6]

    def cost(th: float) -> float:
        return sum(fn_cost * (y and s < th) + fp_cost * ((not y) and s >= th) for s, y in zip(scores, labels))

    return min(cands, key=lambda th: (cost(th), th))


@dataclass
class ScalarThresholdTuner:
    fn_cost: float = 2.0
    fp_cost: float = 1.0
    name: str = "scalar-threshold"

    def prepare(self, row: dict, item: Item) -> tuple[Any, Optional[str]]:
        s = video_score(row)
        return (s, None) if s is not None else (None, "нет скалярного скора (scores_raw отсутствует/без кандидатов)")

    def fit(self, train: list[Item]) -> dict:
        return {"threshold": fit_threshold([it.payload for it in train], [it.positive for it in train],
                                           self.fn_cost, self.fp_cost)}

    def predict(self, params: dict, item: Item) -> bool:
        return float(item.payload) >= float(params["threshold"])


# ----------------------------------------------------------------------------- адаптер A12
def _check_params(fn: Any, required: tuple[str, ...]) -> None:
    try:
        params = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError) as e:
        raise TunerContractError(f"{fn!r}: сигнатура не читается: {e}") from e
    if params[:len(required)] != list(required):
        raise TunerContractError(f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__name__', '?')}{inspect.signature(fn)}:"
                                 f" ожидались первые параметры {required}")


def _check_fields(cls: Any, required: set[str]) -> None:
    if not dataclasses.is_dataclass(cls):
        raise TunerContractError(f"{cls!r} — не dataclass")
    have = {f.name for f in dataclasses.fields(cls)}
    if not required <= have:
        raise TunerContractError(f"{cls.__name__}: нет полей {sorted(required - have)}")


class FusionTuneAdapter:
    """Явный адаптер к ml.fusion.tune (A12): fit(samples, start, default, …) → {type_id: τ}; predict(s, thr, default)."""

    name = "ml.fusion.tune"
    FIT = ("samples", "start", "default")
    PREDICT = ("s", "thr", "default")
    CANDIDATES_FOR = ("pred", "cfg", "types")

    def __init__(self, fit_kw: Optional[dict] = None, module: Any = None, config: Optional[dict] = None,
                 thresholds: Optional[dict] = None, types: Optional[dict] = None) -> None:
        """module/config/thresholds/types — только для тестов контракта; по умолчанию — настоящий A12."""
        if module is None:
            try:
                from ml.fusion import tune as module  # type: ignore
            except ImportError as e:
                raise TunerContractError(f"ml.fusion.tune (A12) недоступен: {e}") from e
        t = module
        for attr in ("fit", "predict", "candidates_for", "Sample", "VideoPred"):
            if not hasattr(t, attr):
                raise TunerContractError(f"{getattr(t, '__name__', t)}: нет {attr}")
        _check_params(t.fit, self.FIT)
        _check_params(t.predict, self.PREDICT)
        _check_params(t.candidates_for, self.CANDIDATES_FOR)
        _check_fields(t.Sample, {"video", "positive", "seconds", "cands", "group"})
        _check_fields(t.VideoPred, {"video", "raw"})
        self.fit_kw = dict(fit_kw or {})
        unknown = set(self.fit_kw) - set(inspect.signature(t.fit).parameters)
        if unknown:
            raise TunerContractError(f"tune.fit не принимает {sorted(unknown)}")
        self.t = t
        if config is None or thresholds is None or types is None:
            try:
                from ml.fusion.catalog import resolve_catalog  # type: ignore
                from ml.fusion.config import load_config, load_thresholds  # type: ignore
            except ImportError as e:
                raise TunerContractError(f"ml.fusion.config/catalog (A12) недоступны: {e}") from e
            config = load_config() if config is None else config
            thresholds = load_thresholds() if thresholds is None else thresholds
            types = resolve_catalog(None)[0] if types is None else types
        self.cfg, self.types = config, types
        self.default = float(thresholds.get("default", 0.45))
        per_type = thresholds.get("per_type") or {}
        self.start = {k: float(v if v is not None else self.default) for k, v in per_type.items()}

    def prepare(self, row: dict, item: Item) -> tuple[Any, Optional[str]]:
        sr = row.get("scores_raw")
        if not isinstance(sr, dict) or not (isinstance(sr.get("candidates"), list) or isinstance(sr.get("inputs"), dict)):
            return None, "scores_raw не в формате ml.fusion (нужен candidates[] или inputs{})"
        cands, _ = self.t.candidates_for(self.t.VideoPred(video=item.video_id, raw=sr), self.cfg, self.types)
        if not isinstance(cands, list):
            raise TunerContractError(f"candidates_for вернул {type(cands).__name__}, ожидался list")
        for c in cands:
            if not (isinstance(c, tuple) and len(c) >= 3 and isinstance(c[0], str) and _finite(c[1]) is not None
                    and 0.0 <= float(c[1]) <= 1.0 and _finite(c[2]) is not None):
                raise TunerContractError(f"кандидат не (type_id, score∈[0,1], t_peak, …): {c!r}")
        secs = sorted(item.gt_times) if item.positive else []
        conflict = len(secs) > 1 and max(secs) - min(secs) > 1.0       # как у A12: решает человек
        sample = self.t.Sample(video=item.video_id, positive=item.positive, seconds=secs, cands=cands, group=item.key,
                               conflict=conflict, type_id=None, members=list(item.members))
        return sample, None

    def fit(self, train: list[Item]) -> dict:
        thr = self.t.fit([it.payload for it in train], dict(self.start), self.default, **self.fit_kw)
        if not isinstance(thr, dict) or not all(isinstance(k, str) and _finite(v) is not None and 0.0 <= float(v) <= 1.0
                                                for k, v in thr.items()):
            raise TunerContractError(f"tune.fit вернул не {{type_id: порог 0..1}}: {thr!r}")
        return thr

    def predict(self, params: dict, item: Item) -> bool:
        hits = self.t.predict(item.payload, params, self.default)
        if not isinstance(hits, list):
            raise TunerContractError(f"tune.predict вернул {type(hits).__name__}, ожидался list кандидатов")
        return bool(hits)


def make_tuner(name: str, **kw: Any) -> Tuner:
    if name == "scalar":
        return ScalarThresholdTuner(**kw)
    if name == "fusion":
        return FusionTuneAdapter(**kw)
    raise ValueError(f"неизвестный тюнер {name!r}: scalar | fusion")


# ----------------------------------------------------------------------------- группы и фолды
def group_rows(rows: list[dict]) -> tuple[list[tuple[Item, dict]], list[dict]]:
    """Строки → по одной на группу (представитель — видео с id группы). Неизвестная метка / конфликт меток — исключение."""
    excluded: list[dict] = []
    groups: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("label") not in LABELS:
            excluded.append({"key": r.get("video_id"), "reason": f"метка {r.get('label')!r} — не violation/no_violation"})
            continue
        groups.setdefault(str(r.get("group") or r["video_id"]), []).append(r)
    out = []
    for key in sorted(groups):
        members = groups[key]
        labels = {m["label"] for m in members}
        if len(labels) > 1:
            excluded.append({"key": key, "reason": "конфликт меток в группе дубликатов — решает человек"})
            continue
        rep = next((m for m in members if m["video_id"] == key), sorted(members, key=lambda m: m["video_id"])[0])
        times = sorted({float(t) for m in members for t in (m.get("gt_times") or []) if _finite(t) is not None})
        item = Item(key=key, video_id=rep["video_id"], members=sorted(m["video_id"] for m in members),
                    positive=labels.pop() == "violation", gt_times=times)
        out.append((item, rep))
    return out, excluded


def prepare_items(rows: list[dict], tuner: Tuner) -> tuple[list[Item], list[dict]]:
    grouped, excluded = group_rows(rows)
    items = []
    for item, rep in grouped:
        payload, reason = tuner.prepare(rep, item)
        if reason:
            excluded.append({"key": item.key, "reason": reason})
            continue
        item.payload = payload
        items.append(item)
    return items, excluded


def assert_no_leak(train: list[Item], test: list[Item]) -> None:
    """Группы и видео train/test не пересекаются (иначе — исключение, а не тихо завышенная метрика)."""
    keys = {it.key for it in train} & {it.key for it in test}
    vids = {m for it in train for m in it.members} & {m for it in test for m in it.members}
    if keys or vids:
        raise AssertionError(f"утечка train/test: группы {sorted(keys)}, видео {sorted(vids)}")


def _metrics(pairs: list[tuple[bool, bool]]) -> dict:
    tp = sum(g and p for g, p in pairs)
    fn = sum(g and not p for g, p in pairs)
    fp = sum((not g) and p for g, p in pairs)
    tn = sum((not g) and (not p) for g, p in pairs)
    return {"n": len(pairs), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None}


def _jsonable_params(p: Any) -> Any:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()} if isinstance(p, dict) else str(p)


def logo(rows: list[dict], tuner: Optional[Tuner] = None) -> dict:
    tuner = tuner or ScalarThresholdTuner()
    items, excluded = prepare_items(rows, tuner)
    pairs, folds = [], []
    for test in items:
        train = [it for it in items if it.key != test.key]
        assert_no_leak(train, [test])
        if not any(it.positive for it in train) or all(it.positive for it in train):
            raise ValueError(f"в train без группы {test.key} нет обоих классов — порог не подобрать")
        params = tuner.fit(train)
        pairs.append((test.positive, tuner.predict(params, test)))
        folds.append({"test": [test.key], "n_train": len(train), "params": _jsonable_params(params)})
    return {"status": "exploratory", "note": EXPLORATORY, "tuner": tuner.name, **_metrics(pairs),
            "excluded": excluded, "leakage_check": "ok", "per_fold": folds}


def grouped_folds(items: list[Item], k: int, seed: int) -> list[list[Item]]:
    rng = random.Random(seed)
    folds: list[list[Item]] = [[] for _ in range(k)]
    for positive in (True, False):
        part = sorted((it for it in items if it.positive == positive), key=lambda it: it.key)
        rng.shuffle(part)
        for i, it in enumerate(part):
            folds[i % k].append(it)
    return folds


def nested_cv(rows: list[dict], tuner: Optional[Tuner] = None, folds: int = 5, seed: int = 2026) -> dict:
    tuner = tuner or ScalarThresholdTuner()
    items, excluded = prepare_items(rows, tuner)
    k = max(2, min(folds, len(items)))
    parts = grouped_folds(items, k, seed)
    pairs, per_fold = [], []
    for i, test in enumerate(parts):
        train = [it for j, p in enumerate(parts) if j != i for it in p]
        if not test or not train:
            continue
        assert_no_leak(train, test)
        if not any(it.positive for it in train) or all(it.positive for it in train):
            raise ValueError(f"в outer-train фолда {i} нет обоих классов — порог не подобрать (мало групп?)")
        params = tuner.fit(train)                       # подбор только на outer-train
        fp = [(it.positive, tuner.predict(params, it)) for it in test]
        pairs += fp
        per_fold.append({"fold": i, **_metrics(fp), "test": sorted(it.key for it in test),
                         "params": _jsonable_params(params)})
    return {"status": "nested_cv", "note": NESTED_NOTE, "tuner": tuner.name, "folds": k, **_metrics(pairs),
            "excluded": excluded, "leakage_check": "ok", "per_fold": per_fold}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="LOGO (exploratory) и nested CV порога вердикта")
    ap.add_argument("--scores", required=True)
    ap.add_argument("--tuner", choices=["scalar", "fusion"], default="scalar",
                    help="scalar — встроенный скалярный порог; fusion — адаптер ml.fusion.tune (A12)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    rows: list[dict[str, Any]] = [json.loads(x) for x in Path(args.scores).read_text(encoding="utf-8").splitlines()
                                  if x.strip()]
    try:
        tuner = make_tuner(args.tuner)
    except TunerContractError as e:
        print(f"тюнер {args.tuner}: контракт не совпал — {e}", file=sys.stderr)
        return 2
    try:
        res = {"logo": logo(rows, tuner), "nested_cv": nested_cv(rows, tuner, args.folds, args.seed)}
    except (ValueError, TunerContractError) as e:   # явный отказ (данных/контракта), без подменного результата
        print(f"LOGO/nested CV не выполнены: {e}", file=sys.stderr)
        return 2
    write_json(args.out or Path(args.scores).with_name(f"loo_{args.tuner}.json"), res)
    print(EXPLORATORY)
    for k, v in res.items():
        print(f"{k}: n={v['n']} recall={v['recall']} precision={v['precision']} FN={v['fn']} FP={v['fp']} "
              f"({v['tuner']}; исключено {len(v['excluded'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
