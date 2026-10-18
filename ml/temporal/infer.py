"""Инференс: score(clip) -> (risk curve 0.5 с, события).

Temporal — генератор риска/кандидатов, НЕ самостоятельное доказательство нарушения. Без обученного
чекпоинта (и при любой невозможности анализа) score() по умолчанию бросает TemporalUnavailable: пайплайн
записывает стадию как error (fusion: degraded), а пустой результат не выдаётся за «видео чистое».
Тип события — только из головы, обученной на проверенных type_id; иначе V99 + zero-shot подсказка в extra."""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import TYPE_IDS, feature_signature, is_mock, load_config, stable_hash, weights_dir
from .errors import TemporalUnavailable
from .features import FeatureSet, cache_path, extract_clip, get_backend, type_scores

try:
    from contracts.sentinel_contracts import Event  # noqa
except ImportError:  # пока ветка A01 не смерджена
    from contracts.sentinel_contracts import Event  # noqa

log = logging.getLogger("sentinel.temporal")
_CK_LOCK = threading.Lock()
_CK: dict[str, Any] = {}
_LAST: dict[str, Any] = {}
CHECKPOINT_FILES = ("meta.json", "spec.npz", "baseline.npz", "temporal.pt", "temporal.json")


def checkpoint_files(root: Path) -> list[Path]:
    """Файлы, определяющие поведение чекпоинта: selected.json + файлы всех видов (для штампа и model_info)."""
    from .predictors import KINDS

    out = [root / "selected.json"]
    for k in KINDS:
        out += [root / k / f for f in CHECKPOINT_FILES]
    return [p for p in out if p.exists()]


def _stamp(root: Path) -> tuple:
    sig = []
    for p in checkpoint_files(root):
        try:
            st = p.stat()
            sig.append((str(p.relative_to(root)), st.st_size, st.st_mtime_ns))
        except OSError:
            continue
    return (str(root), tuple(sig))


def _checkpoint(cfg: dict, root: Optional[Path] = None) -> Optional[tuple[Any, Any, dict]]:
    """Загруженный чекпоинт; перечитывается при изменении ЛЮБОГО его файла (размер/mtime), а не только selected.json."""
    from .predictors import load_checkpoint

    root = Path(root) if root is not None else weights_dir(cfg)
    stamp = _stamp(root)
    with _CK_LOCK:
        if _CK.get(str(root), (None,))[0] != stamp:
            try:
                ck = load_checkpoint(root, cfg)
            except Exception as e:  # битый/несовместимый чекпоинт — честный статус, не падение пайплайна
                log.warning("temporal: не удалось загрузить чекпоинт из %s: %s", root, e)
                ck = None
            if len(_CK) > 16:  # OOF-режим: по чекпоинту на фолд
                _CK.clear()
            _CK[str(root)] = (stamp, ck)
        return _CK[str(root)][1]


def _oof_enabled() -> bool:
    import os

    return os.environ.get("SENTINEL_TEMPORAL_OOF", "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_root(cfg: dict, clip: Any) -> tuple[Optional[Path], Optional[str]]:
    """Каталог чекпоинта для клипа. Обычный режим — weights/temporal. OOF-режим (только оценка,
    SENTINEL_TEMPORAL_OOF=1) — чекпоинт фолда, НЕ видевшего группу этого видео (по md5 содержимого)."""
    root = weights_dir(cfg)
    if not _oof_enabled():
        return root, None
    import json

    from .labels import file_md5

    idx_path = root / "oof" / "index.json"
    if not idx_path.exists():
        return None, f"OOF-режим: нет {idx_path} (сначала python -m ml.temporal.oof)"
    p = Path(str(getattr(clip, "path", "") or ""))
    if not p.is_file():
        return None, "OOF-режим: у клипа нет файла для md5"
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    hit = (idx.get("videos") or {}).get(file_md5(p))
    if not hit:
        return None, "OOF-режим: видео не участвовало в cross-fit обучении (нет OOF-чекпоинта)"
    return root / "oof" / str(hit["dir"]), None


def model_info() -> dict:
    """Хук идентичности для кэша пайплайна (backend.pipeline.stages.module_identity): подпись файлов
    чекпоинта, эффективного конфига и признаков. Без него переобучение не меняет ключ кэша стадии temporal."""
    from .features import FEATURE_VERSION

    cfg = load_config()
    root = weights_dir(cfg)
    oof = _oof_enabled()
    files = checkpoint_files(root)
    if oof:
        oroot = root / "oof"
        files = ([oroot / "index.json"] if (oroot / "index.json").exists() else []) + \
            [p for d in sorted(oroot.glob("fold_*")) for p in checkpoint_files(d)]
    ck = None if oof else _checkpoint(cfg)
    info: dict[str, Any] = {
        "checkpoint": [str(p) for p in files],
        "mode": "oof" if oof else "normal",
        "status": ("oof" if files else "no_checkpoint") if oof else ("trained" if ck is not None else "no_checkpoint"),
        "feature_version": FEATURE_VERSION,
        "config_hash": stable_hash(cfg, 16),
        "mock": is_mock(),
    }
    if ck is not None:
        info.update(kind=ck[0].kind, threshold=ck[2].get("threshold"), created=ck[2].get("created"),
                    feature_signature=ck[2].get("feature_signature"))
    return info


def status() -> dict:
    """Диагностика последнего вызова score() (для meta/warnings пайплайна)."""
    return dict(_LAST)


def warmup() -> None:
    """Загрузить чекпоинт и бэкенд признаков заранее (держим тёплыми)."""
    cfg = load_config()
    ck = _checkpoint(cfg)
    if ck is None:
        return
    try:
        from .data_index import bank_from_meta

        backend = get_backend(cfg, force=ck[1].backend)
        bank_from_meta(cfg, ck[2], backend)
        backend.encode_images(np.zeros((1, 224, 224, 3), np.uint8))
    except Exception as e:
        log.warning("temporal: warmup не удался: %s", e)


def _clip_key(clip: Any, cfg: dict, backend: str, fps: float, views: list, zones: Any) -> Optional[dict]:
    p = Path(str(getattr(clip, "path", "") or ""))
    if not p.is_file():
        return None
    from .labels import file_md5

    zkey = stable_hash([(getattr(z, "name", ""), getattr(z, "polygon", [])) for z in (zones or [])])
    return {"md5": file_md5(p), "sig": feature_signature(cfg, backend), "fps": fps, "views": views, "z": zkey,
            "aug": 0}


def _features(clip: Any, cfg: dict, backend: Any, spec: Any, zones: Any, budget_s: Optional[float],
              t0: float) -> FeatureSet:
    """Признаки клипа с кэшем по md5 содержимого (имя файла в ключ не входит и признаком не является)."""
    inf = cfg["infer"]
    on_cuda = str(getattr(backend, "device", "cpu")).startswith("cuda") or backend.name == "mock"
    f = cfg["features"]
    if spec is not None:
        fps = float(spec.fps if on_cuda else f.get("cpu_fps", 1.0))
        views = list(spec.views if on_cuda else f.get("cpu_views", ["full"]))
    else:
        fps = float(f["fps"] if on_cuda else f.get("cpu_fps", 1.0))
        views = list(f["views"] if on_cuda else f.get("cpu_views", ["full"]))
    budget = float(budget_s if budget_s is not None else inf.get("budget_s", 30.0))
    key = _clip_key(clip, cfg, backend.name, fps, views, zones)
    if key is not None and cache_path(cfg, key).exists():
        return FeatureSet.load(cache_path(cfg, key))
    fs = extract_clip(clip, backend, cfg, zones=zones, fps=fps, views=views, deadline=t0 + 0.85 * budget)
    if key is not None and "truncated_at" not in fs.meta:
        try:
            fs.save(cache_path(cfg, key))
        except OSError:
            pass
    return fs


def score(clip: Any, zones: Any = None, budget_s: Optional[float] = None
          ) -> tuple[list[tuple[float, float]], list[Any]]:
    """Публичная функция (сигнатура контракта: score(clip)); zones/budget_s — необязательные расширения."""
    from .postproc import find_peaks, pool_score, risk_curve

    t0 = time.monotonic()
    cfg = load_config()
    inf = cfg["infer"]
    _LAST.clear()
    root, why = _resolve_root(cfg, clip)
    if root is None:
        return _unavailable(cfg, "oof_unavailable", why or "OOF-чекпоинт недоступен")
    ck = _checkpoint(cfg, root)
    if ck is None:
        return _unavailable(cfg, "no_checkpoint", f"нет обученной модели в {root}")
    model, spec, meta = ck
    if (spec.backend == "mock") != is_mock():
        return _unavailable(cfg, "backend_mismatch",
                            f"чекпоинт обучен на признаках '{spec.backend}', а режим mock={is_mock()}")
    want = meta.get("feature_signature")
    have = stable_hash(feature_signature(cfg, spec.backend), 16)
    if want and want != have:
        return _unavailable(cfg, "checkpoint_mismatch",
                            "чекпоинт обучен на других признаках (модель/ревизия/версия извлечения/настройки) — "
                            "нужно переобучить: python -m ml.temporal.train")
    try:
        from .data_index import bank_from_meta

        backend = get_backend(cfg, force=spec.backend)
        bank = bank_from_meta(cfg, meta, backend)
        fs = _features(clip, cfg, backend, spec, zones, budget_s, t0)
        if len(fs) < int(inf.get("min_frames", 4)):
            return _unavailable(cfg, "too_short", f"слишком мало кадров для временной модели: {len(fs)}")
        from .dataset import build_inputs

        X, grid = build_inputs(fs, bank, spec)
        probs, tprobs = model.predict(X, spec.fps)
    except Exception as e:  # пайплайн ловит исключение стадии сам; здесь — честный статус и причина
        log.exception("temporal: ошибка инференса")
        return _unavailable(cfg, "error", f"{type(e).__name__}: {e}", cause=e)

    duration = float(getattr(clip, "duration", 0.0) or grid[-1])
    # пулинг/шаг — как при калибровке чекпоинта (null-CDF и порог подобраны именно для них)
    pcfg = {**cfg, "infer": {**inf, **{k: v for k, v in (meta.get("pooling") or {}).items() if v is not None}}}
    t_curve, r_curve = risk_curve(probs, grid, pcfg, duration)
    thr = float(meta.get("threshold", inf["event_threshold"]))
    vscore = pool_score(r_curve, pcfg, np.asarray(meta.get("null") or [], np.float64), duration)
    events = []
    if vscore >= thr and len(r_curve):  # видео помечено калиброванным скором -> события из пиков кривой
        rmax = float(r_curve.max())
        peaks = find_peaks(t_curve, r_curve, float(inf.get("peak_rel", 0.6)) * rmax,
                           float(inf["min_peak_distance_s"]), int(inf["max_events"]), float(inf["extent_ratio"]))
        zs, zs_keys = type_scores(fs, bank)
        for pk in peaks:
            pk["confidence"] = vscore * pk["peak"] / max(rmax, 1e-9)
            pk["t_peak"] = _apply_policy(t_curve, r_curve, pk, pcfg)
            events.append(_make_event(pk, model, meta, grid, tprobs, fs, zs, zs_keys, bank))
    _LAST.update(coverage=_coverage(fs, duration), analyzed=True, checkpoint_dir=str(root))
    _LAST.update(status="ok", kind=model.kind, threshold=thr, n_events=len(events), video_score=round(vscore, 4),
                 pooling=pcfg["infer"].get("pooling"), max_risk=float(r_curve.max()) if len(r_curve) else 0.0,
                 elapsed_s=round(time.monotonic() - t0, 3), n_frames=len(fs), fps=fs.meta.get("fps"),
                 views=fs.views, truncated_at=fs.meta.get("truncated_at"))
    return [(round(float(t), 2), round(float(r), 4)) for t, r in zip(t_curve, r_curve)], events


def _unavailable(cfg: dict, reason: str, message: str, cause: Optional[BaseException] = None
                 ) -> tuple[list, list]:
    """Честный статус «не проанализировано». По умолчанию (infer.on_unavailable=raise) — исключение:
    стадия temporal в пайплайне получает status=error с причиной; empty — старое поведение ([], [])."""
    msg = f"ml.temporal: {message} — временная модель не выполнялась; пустой результат не означает «нарушений нет»"
    _LAST.update(status=reason, analyzed=False, warning=msg,
                 coverage={"analyzed_until_s": 0.0, "fraction": 0.0, "complete": False, "reason": reason})
    log.warning("temporal: %s", msg)
    if str(cfg["infer"].get("on_unavailable", "raise")) == "raise":
        raise TemporalUnavailable(reason, msg) from cause
    return [], []


def _coverage(fs: FeatureSet, duration: float) -> dict:
    """Полнота анализа: неполный анализ — не «нарушений нет». complete=False при обрезке по бюджету."""
    until = float(fs.meta.get("truncated_at") or (fs.times[-1] if len(fs.times) else 0.0))
    step = 1.0 / max(float(fs.meta.get("fps", 4.0)), 1e-6)
    frac = min(1.0, (until + step) / max(duration, 1e-6)) if duration else 0.0
    return {"analyzed_until_s": round(until, 2), "duration_s": round(float(duration), 2),
            "fraction": round(frac, 3), "complete": bool(frac >= 0.98 and not fs.meta.get("truncated_at"))}


def _apply_policy(t: np.ndarray, r: np.ndarray, pk: dict, cfg: dict) -> float:
    from .postproc import onset_index

    inf = cfg["infer"]
    if str(inf.get("timecode_policy", "peak")) == "onset":
        a = onset_index(t, r, int(pk["i"]), float(inf.get("onset_ratio", 0.5)), float(inf.get("onset_max_s", 6.0)))
        return float(t[a])
    return float(pk["t_peak"])


def _window(times: np.ndarray, t: float, half: float) -> np.ndarray:
    m = np.abs(times - t) <= half
    if not m.any():
        m = np.zeros(len(times), bool)
        m[int(np.argmin(np.abs(times - t)))] = True
    return m


def _make_event(pk: dict, model: Any, meta: dict, grid: np.ndarray, tprobs: Optional[np.ndarray],
                fs: FeatureSet, zs: np.ndarray, zs_keys: list[str], bank: Any) -> Any:
    tp, a, b = pk["t_peak"], pk["t_start"], pk["t_end"]
    extra: dict[str, Any] = {"kind": model.kind, "peak_risk": round(pk["peak"], 4),
                             "threshold": meta.get("threshold"), "model_created": meta.get("created"),
                             "timecode_policy": cfg_policy(meta),
                             "role": "risk_support", "standalone_evidence": False}
    type_id = "V99"
    if tprobs is not None and getattr(model, "type_trained", False):
        tp_mean = tprobs[_window(grid, tp, 1.5)].mean(axis=0)
        k = int(np.argmax(tp_mean))
        type_id = TYPE_IDS[k] if k < len(TYPE_IDS) else "V99"
        extra["type_source"] = ("pseudo_type_head (предварительно: обучено на псевдометках VLM)"
                                if meta.get("type_head_source") == "pseudo" else "trained_type_head")
        extra["type_probs"] = {TYPE_IDS[i]: round(float(v), 4) for i, v in enumerate(tp_mean) if i < len(TYPE_IDS)}
    else:
        extra["type_source"] = "none"  # тип должен прийти от rules/vlm; ниже — только подсказка
    if len(zs_keys):
        m = _window(fs.times, tp, 1.5)
        zmean = zs[m].mean(axis=0)
        order = np.argsort(-zmean)[:3]
        extra["zero_shot_hint"] = {
            "trained": False, "source": f"zero-shot {fs.meta.get('backend', '')}",
            "note": "сходство кадра с текстовыми подсказками, не обученная классификация",
            "top": [{"type_id": zs_keys[i], "margin_vs_normal": round(float(zmean[i]), 4)} for i in order]}
    facts = [f"Временная модель ({model.kind}) отметила повышенный риск в интервале {a:.1f}–{b:.1f} с "
             f"(максимум {pk['peak']:.2f} на {tp:.1f} с)."]
    bboxes = {}
    if "focus" in fs.views and len(fs.boxes):
        i = int(np.argmin(np.abs(fs.times - tp)))
        box = tuple(float(round(v, 4)) for v in fs.boxes[i])
        if box != (0.0, 0.0, 1.0, 1.0):
            bboxes["temporal_focus"] = box
            extra["temporal_focus_mode"] = fs.meta.get("focus_mode")
    return Event(type_id=type_id, t_start=round(min(a, tp), 2), t_end=round(max(b, tp), 2), t_peak=round(tp, 2),
                 confidence=round(float(pk.get("confidence", pk["peak"])), 4), source="temporal", facts=facts,
                 bboxes=bboxes, extra=extra)


def cfg_policy(meta: dict) -> str:
    return str(load_config()["infer"].get("timecode_policy", "peak"))


# ----------------------------------------------------------------------------- генератор кандидатов
_LAST_PROPOSE: dict[str, Any] = {}


def propose_status() -> dict:
    return dict(_LAST_PROPOSE)


def propose(clip: Any, zones: Any = None, budget_s: Optional[float] = None, n_max: Optional[int] = None,
            mode: Optional[str] = None) -> list[Any]:
    """3–8 моментов-кандидатов для VLM/правил (Event с extra.role="candidate", type_id="V99").

    С чекпоинтом — пики риск-кривой + кривая изменений (режим propose.mode); без чекпоинта — только кривая
    изменений (без обучения, детерминированная; уверенность ≤ change_salience_cap). Это НЕ вердикт:
    слабый скор не отсекает видео от остального анализа."""
    from .candidates import change_curve, resample_to, select_candidates
    from .postproc import risk_curve

    t0 = time.monotonic()
    cfg = load_config()
    inf = cfg["infer"]
    _LAST_PROPOSE.clear()
    root, why = _resolve_root(cfg, clip)
    ck = _checkpoint(cfg, root) if root is not None else None
    warn = why
    if ck is not None and (ck[1].backend == "mock") != is_mock():
        warn, ck = f"чекпоинт на признаках '{ck[1].backend}' не подходит к режиму mock={is_mock()}", None
    if ck is not None and ck[2].get("feature_signature") and \
            ck[2]["feature_signature"] != stable_hash(feature_signature(cfg, ck[1].backend), 16):
        warn, ck = "чекпоинт обучен на других признаках — риск не используется, только кривая изменений", None
    model, spec, meta = ck if ck is not None else (None, None, {})
    try:
        backend = get_backend(cfg, force=spec.backend if spec is not None else None)
        fs = _features(clip, cfg, backend, spec, zones, budget_s, t0)
        if len(fs) < int(inf.get("min_frames", 4)):
            _LAST_PROPOSE.update(status="too_short", n=len(fs), analyzed=False,
                                 coverage={"fraction": 0.0, "complete": False, "reason": "too_short"})
            return []
        duration = float(getattr(clip, "duration", 0.0) or fs.times[-1])
        step = float(inf.get("risk_step_s", 0.5))
        t_grid = np.arange(0.0, duration + 1e-6, step, dtype=np.float32)
        risk = None
        bank = None
        if model is not None:
            from .data_index import bank_from_meta
            from .dataset import build_inputs

            bank = bank_from_meta(cfg, meta, backend)
            X, grid = build_inputs(fs, bank, spec)
            probs, _ = model.predict(X, spec.fps)
            pcfg = {**cfg, "infer": {**inf, **{k: v for k, v in (meta.get("pooling") or {}).items() if v is not None}}}
            t_grid, risk = risk_curve(probs, grid, pcfg, duration)
        change = resample_to(fs.times, change_curve(fs.times, fs.emb, fs.dyn, cfg), t_grid)
        cands = select_candidates(t_grid, risk, change, cfg, mode=mode, n_max=n_max)
        if bank is None:
            from .features import get_text_bank

            bank = get_text_bank(cfg, backend)
        zs, zs_keys = type_scores(fs, bank)
    except Exception as e:  # graceful: генератор кандидатов не должен ронять пайплайн
        _LAST_PROPOSE.update(status="error", warning=f"temporal.propose: {type(e).__name__}: {e}", analyzed=False,
                             coverage={"fraction": 0.0, "complete": False, "reason": "error"})
        log.exception("temporal: ошибка propose")
        return []
    pw = float(cfg.get("propose", {}).get("window_s", 2.0))
    events = []
    for c in cands:
        t = c["t"]
        extra: dict[str, Any] = {"role": "candidate", "candidate_source": c["source"], "rank": c["rank"],
                                 "risk": c["risk"], "change": c["change"], "type_source": "none",
                                 "checkpoint": model is not None,
                                 "mode": mode or cfg.get("propose", {}).get("mode", "hybrid")}
        if len(zs_keys):
            m = _window(fs.times, t, 1.5)
            zmean = zs[m].mean(axis=0)
            order = np.argsort(-zmean)[:3]
            extra["zero_shot_hint"] = {"trained": False, "note": "не обученная классификация",
                                       "top": [{"type_id": zs_keys[i], "margin_vs_normal": round(float(zmean[i]), 4)}
                                               for i in order]}
        if c["source"] == "risk":
            fact = f"Кандидат для проверки: временная модель отметила повышенный риск около {t:.1f} с."
        else:
            fact = f"Кандидат для проверки: заметное изменение кадра около {t:.1f} с (без классификации)."
        bboxes = {}
        if "focus" in fs.views and len(fs.boxes):
            i = int(np.argmin(np.abs(fs.times - t)))
            box = tuple(float(round(v, 4)) for v in fs.boxes[i])
            if box != (0.0, 0.0, 1.0, 1.0):
                bboxes["temporal_focus"] = box
        events.append(Event(type_id="V99", t_start=round(max(0.0, t - pw), 2), t_end=round(min(duration, t + pw), 2),
                            t_peak=round(t, 2), confidence=round(float(c["salience"]), 4), source="temporal",
                            facts=[fact], bboxes=bboxes, extra=extra))
    _LAST_PROPOSE.update(coverage=_coverage(fs, duration), analyzed=True,
                         risk_used=model is not None)  # без чекпоинта — только кривая изменений, не риск
    _LAST_PROPOSE.update(status="ok", n=len(events), checkpoint=model is not None, warning=warn,
                         elapsed_s=round(time.monotonic() - t0, 3), n_frames=len(fs), views=fs.views)
    return events
