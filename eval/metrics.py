"""Метрики «как у жюри» по предсказаниям (preds/*.json) и разметке (data/labels.csv).

* по видео: confusion, precision / recall / F1, FP / FN / failed;
* локализация: |Δt| ближайшего нарушения до GT, hit@0.5/1/2/3 с, знак сдвига (рано/поздно);
* кадр-доказательство в ±evidence_tol_s от GT (и валидный JPEG);
* типы (если есть data/labels_types.csv): точность, confusion; иначе — распределение;
* нормы: у каждого нарушения ≥1 rule, доля verified, URL adilet, наличие в norms/catalog.yaml;
* схема ТЗ; язык описаний (без домыслов); латентность; согласованность md5-дубликатов.

CLI::

    python -m eval.metrics --labels data/labels.csv --preds runs/eval/preds [--types data/labels_types.csv]
"""
from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .common import load_config, normalize_video_id, read_json, video_sort_key, write_json
from .labels import LABEL_VIOLATION, GroundTruth, load_ground_truth
from .schema import check_api_response, coverage_problem, coverage_ratio, decode_jpeg_b64

CATEGORIES = ("TP", "TN", "FP", "FN", "early", "late", "wrong_type", "failed", "partial", "label_conflict",
              "label_unknown")
EVAL_VERSION = "a16-eval-2.0"
KIND_NOTES = {
    "exploratory": "РАЗВЕДОЧНАЯ оценка: эти видео уже использовались при разработке/разведке — не независимый тест.",
    "holdout_frozen": "Замороженный hold-out (заморожен до настройки, не использовался для подбора порогов).",
}
LOCALIZATION_CAVEAT = ("GT размечена одной секундой на видео (неполная event-разметка): лишние кандидаты "
                       "(extra_candidates) не обязательно ложные — показываются отдельно и не входят в FP по видео.")
TYPES_NOTE = "Точность типа считается только по настоящей разметке типов (data/labels_types.csv); без неё — n/a."
NORMS_NOTE = ("Проверяется полнота и формат ссылок (rule есть, из каталога, флаг verified). Наличие quote/URL "
              "≠ правильная трактовка нормы — корректность только по ручной экспертизе (data/norms_review.csv).")
_TIMEOUT_RE = ("timeout", "time out", "timed out", "deadline", "таймаут", "тайм-аут", "превышен")
_DEGRADED_RE = ("deadline", "timeout", "budget", "таймаут", "бюджет", "дедлайн", "fallback")
_DEGRADED_STAGE = ("timeout", "crashed", "cancelled")


# ----------------------------------------------------------------------------- загрузка
def unwrap_prediction(d: Any) -> tuple[dict, dict]:
    """preds/*.json бывают «голым» ответом API или обёрткой {"response": ..., "latency_s": ...}."""
    if isinstance(d, dict) and isinstance(d.get("response"), dict):
        info = {k: v for k, v in d.items() if k != "response"}
        return d["response"], info
    return (d if isinstance(d, dict) else {}), {}


def load_predictions(preds_dir: str | Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """→ ({video_id: api_dict}, {video_id: runinfo}). Файлы, начинающиеся с «_», — служебные."""
    preds: dict[str, dict] = {}
    info: dict[str, dict] = {}
    pdir = Path(preds_dir)
    runinfo_path = pdir / "_runinfo.json"
    if runinfo_path.exists():
        for k, v in (read_json(runinfo_path) or {}).items():
            info[normalize_video_id(k) or k] = v
    for p in sorted(pdir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            api, extra = unwrap_prediction(read_json(p))
        except Exception as e:  # noqa: BLE001
            api, extra = {}, {"error": f"битый JSON: {e}"}
        vid = normalize_video_id(extra.get("video_id") or p.stem) or p.stem
        preds[vid] = api
        if extra:
            info.setdefault(vid, {}).update(extra)
    return preds, info


def load_catalog(cfg: Optional[dict] = None) -> Optional[dict]:
    """norms.load_catalog() (A13) → иначе YAML по cfg.norms.catalog_path → иначе None."""
    try:
        from norms import load_catalog as _lc  # type: ignore

        cat = _lc()
        if isinstance(cat, dict) and cat:
            return cat
    except Exception:  # noqa: BLE001
        pass
    from .common import REPO_ROOT

    path = Path(((cfg or {}).get("norms") or {}).get("catalog_path") or "norms/catalog.yaml")
    if not path.is_absolute():
        path = REPO_ROOT / path
    if path.exists():
        try:
            import yaml

            return yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
    return None


# ----------------------------------------------------------------------------- per-video
@dataclass
class VideoEval:
    video_id: str
    gt_label: str
    gt_times: list[float]
    gt_types: list[tuple[float, str]]
    members: list[str]
    conflict: bool
    pred_id: Optional[str] = None          # чьё предсказание взяли (член md5-группы)
    status: str = "missing"                # done | failed | processing | missing
    verdict: Optional[str] = None
    n_violations: int = 0
    category: str = "failed"
    pred_t: Optional[float] = None         # таймкод ближайшего к GT нарушения (или топового)
    pred_type: Optional[str] = None
    pred_conf: Optional[float] = None
    matched_gt_t: Optional[float] = None
    dt_signed: Optional[float] = None      # pred - gt
    dt_abs: Optional[float] = None
    hits: dict[str, bool] = field(default_factory=dict)
    evidence_ok: Optional[bool] = None
    evidence_dt: Optional[float] = None
    type_results: list[dict] = field(default_factory=list)
    norms: list[dict] = field(default_factory=list)
    schema_errors: list[str] = field(default_factory=list)
    schema_warnings: list[str] = field(default_factory=list)
    frames_total: int = 0
    frames_valid: int = 0
    language: list[dict] = field(default_factory=list)
    latency_s: Optional[float] = None
    duration: Optional[float] = None
    error: Optional[str] = None
    violations_brief: list[dict] = field(default_factory=list)
    # one-to-one сопоставление предсказанных нарушений с GT-событиями
    events_total: int = 0
    events_matched: int = 0
    event_hits: dict[str, int] = field(default_factory=dict)
    extra_candidates: int = 0
    nearest_dt_signed: Optional[float] = None   # без окна — справочно (насколько далеко промах)
    match_pairs: list[dict] = field(default_factory=list)
    timeout: bool = False
    degraded: bool = False
    pipeline_version: Optional[str] = None
    coverage: Optional[float] = None            # meta.coverage: доля проанализированного видео (None — не сообщено)
    partial: bool = False                       # no_violation при неполном анализе — НЕ «чисто»
    invalid_verdict: bool = False               # status=done, но verdict не из схемы — как failed
    vram_process_peak_mb: Optional[float] = None    # процесс/воркер пайплайна
    vram_process_source: Optional[str] = None
    vram_device_peak_mb: Optional[float] = None     # устройство целиком (все процессы)
    vram_device_baseline_mb: Optional[float] = None
    coverage_problem: Optional[str] = None
    mock: bool = False
    policy_dt: dict[str, Optional[float]] = field(default_factory=dict)   # |Δt| при разных политиках таймкода

    @property
    def gt_positive(self) -> bool:
        return self.gt_label == LABEL_VIOLATION

    @property
    def pred_positive(self) -> bool:
        return self.status == "done" and self.verdict == "violation"


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _latency(api: dict, info: dict) -> Optional[float]:
    for v in (info.get("latency_s"), info.get("elapsed_s")):
        if _num(v) is not None:
            return _num(v)
    meta = api.get("meta") or {}
    t = meta.get("timings") or {}
    for k in ("total_s", "total", "wall_s", "elapsed_s"):
        if isinstance(t, dict) and _num(t.get(k)) is not None:
            return _num(t.get(k))
    for k in ("elapsed_s", "latency_s", "processing_s"):
        if _num(meta.get(k)) is not None:
            return _num(meta.get(k))
    return None


def _duration(api: dict, info: dict) -> Optional[float]:
    meta = api.get("meta") or {}
    for v in (info.get("duration"), meta.get("duration"), meta.get("duration_s"), (meta.get("video") or {}).get("duration")
              if isinstance(meta.get("video"), dict) else None):
        if _num(v) is not None and _num(v) > 0:
            return _num(v)
    return None


def _vram_split(api: dict, info: dict) -> dict:
    """Две разные величины, НЕ смешиваются и НЕ подменяются нулём:

    * процесс/воркер пайплайна — A03 meta.stages[*].vram_peak_mb (torch.max_memory_allocated воркера),
      иначе meta.vram_peak_mb / meta.resources|gpu.vram_peak_mb, иначе пик процесса eval (inproc --inline);
    * устройство целиком (все процессы на карте, включая чужие) — сэмплер nvidia-smi из runinfo, с фоном.
    """
    meta = api.get("meta") or {}
    out: dict[str, Any] = {"process_peak_mb": None, "process_source": None,
                           "device_peak_mb": _num(info.get("vram_device_peak_mb", info.get("vram_peak_mb"))),
                           "device_baseline_mb": _num(info.get("vram_device_baseline_mb"))}
    stages = meta.get("stages") if isinstance(meta.get("stages"), dict) else {}
    st = [_num(v.get("vram_peak_mb")) for v in stages.values() if isinstance(v, dict)]
    st = [v for v in st if v is not None]
    if st:
        out.update(process_peak_mb=max(st), process_source="pipeline_stages")
        return out
    cands = [meta.get("vram_peak_mb")] + [(meta.get(k) or {}).get("vram_peak_mb") for k in ("resources", "gpu")
                                          if isinstance(meta.get(k), dict)]
    cands = [_num(v) for v in cands if _num(v) is not None]
    if cands:
        out.update(process_peak_mb=max(cands), process_source="pipeline_meta")
        return out
    ev = _num(info.get("vram_eval_process_peak_mb", info.get("vram_torch_peak_mb")))
    if ev is not None:
        out.update(process_peak_mb=ev, process_source="eval_process")
    return out


def _is_mock(api: dict) -> bool:
    meta = api.get("meta") or {}
    try:
        from contracts.extensions import is_mock_result  # type: ignore
    except ImportError:
        is_mock_result = None
    if is_mock_result is not None:
        return bool(is_mock_result(meta))
    return meta.get("mock") is True or (isinstance(meta.get("provenance"), dict) and meta["provenance"].get("mock") is True)


def _degraded(api: dict, texts: list[str]) -> bool:
    """Деградация анализа: A03 meta.degraded, стадия timeout/crashed/cancelled, пропуск по дедлайну/бюджету,
    fallback. «skipped — нет кандидатов» — штатный пропуск, не деградация."""
    meta = api.get("meta") or {}
    if meta.get("degraded") is True:
        return True
    stages = meta.get("stages") if isinstance(meta.get("stages"), dict) else {}
    for st in stages.values():
        if not isinstance(st, dict):
            continue
        err = str(st.get("error") or "").lower()
        if st.get("status") in _DEGRADED_STAGE or st.get("origin") == "fallback":
            return True
        if st.get("status") == "skipped" and any(k in err for k in ("дедлайн", "deadline", "бюджет", "budget", "отмена")):
            return True
    if stages:   # структурированные статусы есть — свободный текст предупреждений не переинтерпретируем
        return False
    return any(k in t for t in texts for k in _DEGRADED_RE)


def _texts(api: dict, info: dict) -> list[str]:
    meta = api.get("meta") or {}
    out = [str(info.get("error") or ""), str(api.get("error") or ""), str(meta.get("error") or "")]
    w = meta.get("warnings")
    if isinstance(w, list):
        out += [str(x) for x in w]
    elif w:
        out.append(str(w))
    return [x.lower() for x in out if x]


def match_one_to_one(events: list[list[float]], timed: list[tuple[int, float]], window_s: float) -> list[dict]:
    """Жадное one-to-one сопоставление по |Δt| (каждое GT-событие ↔ не более одного нарушения и наоборот).
    Событие = список альтернативных таймкодов (конфликт дубликатов) — берётся ближайшая альтернатива."""
    cands = []
    for ei, alts in enumerate(events):
        if not alts:
            continue
        for i, t in timed:
            a = min(alts, key=lambda x: abs(t - x))
            d = abs(t - a)
            if d <= window_s + 1e-9:
                cands.append((d, ei, i, t - a, a))
    cands.sort()
    used_e: set[int] = set()
    used_p: set[int] = set()
    pairs = []
    for d, ei, i, sd, a in cands:
        if ei in used_e or i in used_p:
            continue
        used_e.add(ei)
        used_p.add(i)
        pairs.append({"event": ei, "pred": i, "gt_t": a, "dt_signed": round(sd, 3), "dt_abs": round(d, 3)})
    return pairs


def _rule_key(r: dict) -> tuple:
    return (str(r.get("url") or "").strip().rstrip("/"), str(r.get("clause") or "").strip())


def _catalog_rules(catalog: Optional[dict], type_id: str) -> Optional[set]:
    if not catalog:
        return None
    t = ((catalog.get("types") or {}).get(type_id) or {})
    return {_rule_key(r) for r in (t.get("rules") or []) if isinstance(r, dict)}


def evaluate_video(gt: GroundTruth, api: Optional[dict], info: dict, cfg: dict,
                   catalog: Optional[dict] = None, pred_id: Optional[str] = None) -> VideoEval:
    ve = VideoEval(video_id=gt.video_id, gt_label=gt.label, gt_times=list(gt.times), gt_types=list(gt.types),
                   members=list(gt.members or [gt.video_id]), conflict=gt.conflict, pred_id=pred_id)
    ve.latency_s = _latency(api or {}, info or {})
    vr = _vram_split(api or {}, info or {})
    ve.vram_process_peak_mb, ve.vram_process_source = vr["process_peak_mb"], vr["process_source"]
    ve.vram_device_peak_mb, ve.vram_device_baseline_mb = vr["device_peak_mb"], vr["device_baseline_mb"]
    ve.mock = _is_mock(api or {})
    ve.error = (info or {}).get("error")
    ve.events_total = len(gt.events or [[t] for t in gt.times]) if gt.positive else 0
    texts = _texts(api or {}, info or {})
    if not api:
        ve.status, ve.category = "missing", "failed"
        ve.timeout = bool(info.get("timeout")) or any(k in t for t in texts for k in _TIMEOUT_RE)
        return ve
    ve.status = api.get("status") or "missing"
    ve.pipeline_version = str((api.get("meta") or {}).get("version") or "") or None
    if ve.status != "done":
        ve.timeout = bool(info.get("timeout")) or any(k in t for t in texts for k in _TIMEOUT_RE)
    else:
        ve.degraded = _degraded(api, texts)
    ve.duration = _duration(api, info or {})
    rep = check_api_response(api, duration=ve.duration, max_frames=int(cfg.get("max_evidence_frames", 3)))
    ve.schema_errors, ve.schema_warnings = rep.errors, rep.warnings
    ve.frames_total, ve.frames_valid = rep.frames_total, rep.frames_valid_jpeg
    if ve.status != "done":
        ve.category = "failed"
        return ve
    ve.verdict = api.get("verdict")
    if ve.verdict not in ("violation", "no_violation"):   # unknown ≠ false: невалидный вердикт — не «чисто»
        ve.invalid_verdict = True
        ve.category = "failed"
        return ve
    ve.coverage = coverage_ratio(api.get("meta"))
    if ve.verdict == "no_violation":
        ve.coverage_problem = coverage_problem(api.get("meta"), cfg)
        ve.partial = ve.coverage_problem is not None      # неполный анализ ≠ «нарушений нет»
    vios = [v for v in (api.get("violations") or []) if isinstance(v, dict)]
    ve.n_violations = len(vios)
    ve.violations_brief = [{"t": _num(v.get("timecode_sec")), "type_id": v.get("type_id") or "",
                            "type": v.get("type") or "", "conf": _num(v.get("confidence")),
                            "t_start": _num(v.get("t_start")), "t_end": _num(v.get("t_end")),
                            "n_frames": len(v.get("evidence_frames") or []), "n_rules": len(v.get("rules") or [])}
                           for v in vios]
    lang = cfg.get("language") or {}
    hedges = [h.lower() for h in lang.get("hedge_markers", [])]
    forbidden = [w.lower() for w in lang.get("forbidden", [])]
    allowed = tuple((cfg.get("norms") or {}).get("allowed_url_prefixes") or ())
    for v in vios:
        desc = str(v.get("description") or "").lower()
        ve.language.append({"hedged": any(h in desc for h in hedges),
                            "forbidden": [w for w in forbidden if w in desc]})
        rules = [r for r in (v.get("rules") or []) if isinstance(r, dict)]
        cat_rules = _catalog_rules(catalog, str(v.get("type_id") or ""))
        ve.norms.append({
            "type_id": v.get("type_id") or "",
            "n_rules": len(rules),
            "verified_share": (sum(bool(r.get("verified")) for r in rules) / len(rules)) if rules else 0.0,
            "url_ok": bool(rules) and all(str(r.get("url") or "").startswith(allowed) for r in rules) if allowed
            else bool(rules),
            "in_catalog": (bool(rules) and all(_rule_key(r) in cat_rules for r in rules)) if cat_rules is not None
            else None,
            "type_in_catalog": (str(v.get("type_id") or "") in (catalog.get("types") or {})) if catalog else None,
        })

    # --- вердикт
    pred_pos = ve.verdict == "violation"
    if gt.label == "conflict":
        ve.category = "label_conflict"
    elif gt.label == "unknown":
        ve.category = "label_unknown"
    elif ve.partial:
        ve.category = "partial"
    elif not gt.positive:
        ve.category = "FP" if pred_pos else "TN"
    elif not pred_pos:
        ve.category = "FN"
    else:
        ve.category = "TP"

    # --- локализация: one-to-one сопоставление нарушений с GT-событиями (альтернативы — для конфликтов)
    timed = [(i, t) for i, t in ((i, _num(v.get("timecode_sec"))) for i, v in enumerate(vios)) if t is not None]
    events = (gt.events or [[t] for t in gt.times]) if gt.positive else []
    windows = [float(k) for k in cfg.get("hit_windows_s", [0.5, 1, 2, 3])]
    win = max([float(cfg.get("match_window_s", 5.0))] + windows)
    pairs = match_one_to_one(events, timed, win) if pred_pos else []
    ve.match_pairs = pairs
    ve.events_matched = len(pairs)
    ve.extra_candidates = (len(timed) - len(pairs)) if (pred_pos and gt.positive) else 0
    for k in windows:
        ve.event_hits[f"{k:g}"] = sum(p["dt_abs"] <= k + 1e-9 for p in pairs)
    nearest = min(((abs(t - a), i, t - a, a) for i, t in timed for alts in events for a in alts), default=None)
    if nearest is not None:
        ve.nearest_dt_signed = round(nearest[2], 3)
    v: Optional[dict] = None
    if pairs:
        best = min(pairs, key=lambda p: p["dt_abs"])
        v = vios[best["pred"]]
        ve.pred_t, ve.matched_gt_t = _num(v.get("timecode_sec")), best["gt_t"]
        ve.dt_signed, ve.dt_abs = best["dt_signed"], best["dt_abs"]
    elif nearest is not None:
        v = vios[nearest[1]]
        ve.pred_t = _num(v.get("timecode_sec"))
    elif timed:  # GT без таймкода или GT-негатив: показываем самое уверенное нарушение
        i_best = max((i for i, _ in timed), key=lambda i: _num(vios[i].get("confidence")) or 0.0)
        v = vios[i_best]
        ve.pred_t = _num(v.get("timecode_sec"))
    if v is not None:
        ve.pred_type = v.get("type_id") or None
        ve.pred_conf = _num(v.get("confidence"))
    if gt.positive and pred_pos and events:
        # «что если»: какой момент отдавать жюри — timecode_sec, t_start (начало) или середину интервала
        alts = [a for ev in events for a in ev]
        for pol in ("timecode_sec", "t_start", "t_mid"):
            ts = []
            for x in vios:
                if pol == "t_mid":
                    a_, b_ = _num(x.get("t_start")), _num(x.get("t_end"))
                    t = (a_ + b_) / 2 if a_ is not None and b_ is not None else None
                else:
                    t = _num(x.get(pol))
                if t is not None:
                    ts.append(t)
            ve.policy_dt[pol] = round(min(abs(t - a) for t in ts for a in alts), 3) if ts and alts else None
        for k in windows:
            ve.hits[f"{k:g}"] = ve.dt_abs is not None and ve.dt_abs <= k + 1e-9
        if v is not None and ve.matched_gt_t is not None:
            tol = float(cfg.get("evidence_tol_s", 1.0))
            frames = [f for f in (v.get("evidence_frames") or []) if isinstance(f, dict)]
            dts = [abs(_num(f.get("timestamp_sec")) - ve.matched_gt_t) for f in frames
                   if _num(f.get("timestamp_sec")) is not None and decode_jpeg_b64(f.get("image_base64")) is not None]
            ve.evidence_dt = round(min(dts), 3) if dts else None
            ve.evidence_ok = bool(dts) and min(dts) <= tol + 1e-9
        else:
            ve.evidence_ok = False
        el = float(cfg.get("early_late_tol_s", 2.0))
        sd = ve.dt_signed if ve.dt_signed is not None else ve.nearest_dt_signed
        if sd is not None and abs(sd) > el:
            ve.category = "early" if sd < 0 else "late"

    # --- типы
    if gt.types:
        win = float(cfg.get("type_match_window_s", 3.0))
        for gt_t, gt_type in gt.types:
            cands = [(abs(t - gt_t), i) for i, t in timed] if gt_t == gt_t else []
            cands = [c for c in cands if c[0] <= win]
            if not pred_pos or not cands:
                ve.type_results.append({"gt_t": gt_t, "gt_type": gt_type, "pred_type": None, "ok": False,
                                        "reason": "missed" if not pred_pos else "not_localized"})
                continue
            _, i = min(cands)
            pt = vios[i].get("type_id") or None
            ve.type_results.append({"gt_t": gt_t, "gt_type": gt_type, "pred_type": pt, "ok": pt == gt_type,
                                    "reason": "ok" if pt == gt_type else "wrong_type"})
        if ve.category == "TP" and any(r["reason"] == "wrong_type" for r in ve.type_results):
            ve.category = "wrong_type"
    return ve


# ----------------------------------------------------------------------------- агрегаты
def _rate(a: float, b: float) -> Optional[float]:
    return round(a / b, 4) if b else None


def wilson(k: int, n: int, z: float = 1.96) -> Optional[list[float]]:
    """95% доверительный интервал Вильсона — при n≈20–40 он широкий, это честно."""
    if not n:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "mean": None, "median": None, "p90": None, "max": None}
    s = sorted(xs)
    p90 = s[min(len(s) - 1, int(math.ceil(0.9 * len(s))) - 1)]
    return {"n": len(s), "mean": round(statistics.fmean(s), 3), "median": round(statistics.median(s), 3),
            "p90": round(p90, 3), "max": round(s[-1], 3)}


def _localization(evals: list[VideoEval], cfg: dict) -> dict:
    windows = [f"{float(k):g}" for k in cfg.get("hit_windows_s", [0.5, 1, 2, 3])]
    pos = [e for e in evals if e.gt_positive and e.events_total]
    tp = [e for e in pos if e.pred_positive]
    ev_all = sum(e.events_total for e in pos)
    ev_tp = sum(e.events_total for e in tp)
    matched = [p for e in tp for p in e.match_pairs]
    return {
        "n_videos_pos_with_time": len(pos), "events_total": ev_all, "events_in_tp_videos": ev_tp,
        "events_matched": len(matched),
        "abs_dt_matched": _stats([p["dt_abs"] for p in matched]),
        "mean_signed_dt_matched": round(statistics.fmean([p["dt_signed"] for p in matched]), 3) if matched else None,
        "nearest_abs_dt_unconstrained": _stats([abs(e.nearest_dt_signed) for e in tp if e.nearest_dt_signed is not None]),
        "hit_events_all": {k: _rate(sum(e.event_hits.get(k, 0) for e in pos), ev_all) for k in windows},
        "hit_events_tp": {k: _rate(sum(e.event_hits.get(k, 0) for e in tp), ev_tp) for k in windows},
        "extra_candidates_total": sum(e.extra_candidates for e in tp),
        "extra_candidates_mean_per_tp_video": round(statistics.fmean([e.extra_candidates for e in tp]), 3) if tp else None,
        "early_ids": [e.video_id for e in evals if e.category == "early"],
        "late_ids": [e.video_id for e in evals if e.category == "late"],
        "match_window_s": max([float(cfg.get("match_window_s", 5.0))] + [float(k) for k in windows]),
        "caveat": LOCALIZATION_CAVEAT,
    }


def _norms_review(cfg: dict) -> dict[tuple, bool]:
    """data/norms_review.csv: type_id,url,clause,correct(1/0) — ручная экспертиза трактовки норм."""
    from .common import REPO_ROOT

    p = Path(((cfg.get("norms") or {}).get("review_path")) or "data/norms_review.csv")
    if not p.is_absolute():
        p = REPO_ROOT / p if not p.exists() else p
    if not p.exists():
        return {}
    from .labels import _read_csv_dicts

    out = {}
    for d in _read_csv_dicts(p):
        key = (str(d.get("type_id") or "").upper(), str(d.get("url") or "").strip().rstrip("/"),
               str(d.get("clause") or "").strip())
        out[key] = str(d.get("correct") or "").strip().lower() in ("1", "true", "yes", "да", "верно")
    return out


def _conflict_block(evals: list[VideoEval]) -> list[dict]:
    """Группы дубликатов с конфликтом таймкодов (57 / 68 с) — отдельно; решает человек."""
    out = []
    for e in evals:
        if not e.conflict:
            continue
        out.append({"group": e.video_id, "members": e.members, "gt_alternatives": e.gt_times, "verdict": e.verdict,
                    "pred_t": e.pred_t,
                    "dt_to_alternatives": ({f"{g:g}": round(e.pred_t - g, 3) for g in e.gt_times}
                                           if e.pred_t is not None else None)})
    return out


def summarize(evals_all: list[VideoEval], cfg: dict, dup_check: Optional[list[dict]] = None) -> dict:
    conflicts = [e for e in evals_all if e.gt_label == "conflict"]
    unknown = [e for e in evals_all if e.gt_label == "unknown"]
    evals = [e for e in evals_all if e.gt_label not in ("conflict", "unknown")]
    pos = [e for e in evals if e.gt_positive]
    neg = [e for e in evals if not e.gt_positive]
    failed = [e for e in evals if e.status != "done" or e.invalid_verdict]
    partial = [e for e in evals if e.partial]
    tp = [e for e in pos if e.pred_positive]
    fn = [e for e in pos if not e.pred_positive]                  # включая failed/partial: ответа «есть» нет
    fp = [e for e in neg if e.pred_positive]
    tn = [e for e in neg if e.status == "done" and e.verdict == "no_violation" and not e.partial]
    unresolved_neg = [e for e in neg if not e.pred_positive and e not in tn]  # failed/partial ≠ «нарушений нет»
    prec = _rate(len(tp), len(tp) + len(fp))
    rec = _rate(len(tp), len(pos))
    f1 = round(2 * prec * rec / (prec + rec), 4) if prec and rec else (0.0 if (prec is not None and rec is not None) else None)
    confusion: dict[str, dict[str, int]] = {}
    for e in evals:
        pred = ("failed" if (e.status != "done" or e.invalid_verdict) else
                "partial" if e.partial else e.verdict)
        confusion.setdefault(e.gt_label, {}).setdefault(str(pred), 0)
        confusion[e.gt_label][str(pred)] += 1
    kind = str((cfg.get("dataset") or {}).get("evaluation_kind") or "exploratory")
    video = {
        "n": len(evals), "n_pos": len(pos), "n_neg": len(neg),
        "tp": len(tp), "tn": len(tn), "fp": len(fp), "fn": len(fn), "failed": len(failed),
        "partial": len(partial), "unresolved_neg": len(unresolved_neg),
        "fn_failed_or_partial": sum(1 for e in fn if e in failed or e.partial),
        "timeouts": sum(e.timeout for e in evals), "degraded": sum(e.degraded for e in evals),
        "coverage_unknown": sum(1 for e in evals if e.status == "done" and not e.invalid_verdict and e.coverage is None),
        "precision": prec, "recall": rec, "f1": f1,
        "precision_ci95": wilson(len(tp), len(tp) + len(fp)), "recall_ci95": wilson(len(tp), len(pos)),
        "accuracy": _rate(len(tp) + len(tn), len(evals)),
        "specificity": _rate(len(tn), len(neg)), "specificity_ci95": wilson(len(tn), len(neg)),
        "fp_ids": [e.video_id for e in fp], "fn_ids": [e.video_id for e in fn],
        "failed_ids": [e.video_id for e in failed], "partial_ids": [e.video_id for e in partial],
        "timeout_ids": [e.video_id for e in evals if e.timeout],
        "label_conflict_ids": [e.video_id for e in conflicts], "label_unknown_ids": [e.video_id for e in unknown],
        "confusion": confusion, "evaluation_kind": kind, "kind_note": KIND_NOTES.get(kind, kind),
        "note": "failed/partial/невалидный вердикт: на позитиве — пропуск (FN), на негативе — НЕ TN (unresolved_neg); "
                "группы с конфликтом или неизвестной меткой исключены.",
    }
    policy = str(cfg.get("conflict_policy") or "separate")
    no_conf = [e for e in evals if not e.conflict]
    if policy == "alternatives":
        localization = _localization(evals, cfg)
        localization_alt = _localization(no_conf, cfg) if len(no_conf) != len(evals) else None
    else:  # separate: конфликтные группы не входят в основные метрики локализации
        localization = _localization(no_conf, cfg)
        localization_alt = _localization(evals, cfg) if len(no_conf) != len(evals) else None
    localization["conflict_policy"] = policy
    in_loc = (lambda e: True) if policy == "alternatives" else (lambda e: not e.conflict)  # noqa: E731
    loc_tp = [e for e in tp if e.events_total and e.dt_abs is not None and in_loc(e)]
    timed_pos = [e for e in pos if e.events_total and in_loc(e)]
    frames_total = sum(e.frames_total for e in evals)
    n_vios = sum(e.n_violations for e in evals)
    frames_per_vio_ok = sum(1 for e in evals for v in e.violations_brief
                            if 1 <= v["n_frames"] <= int(cfg.get("max_evidence_frames", 3)))
    evidence = {
        "ok_tp": _rate(sum(bool(e.evidence_ok) for e in loc_tp), len(loc_tp)),
        "ok_all": _rate(sum(bool(e.evidence_ok) for e in timed_pos), len(timed_pos)),
        "valid_jpeg_rate": _rate(sum(e.frames_valid for e in evals), frames_total),
        "frames_per_violation_ok": _rate(frames_per_vio_ok, n_vios),
        "evidence_dt": _stats([e.evidence_dt for e in loc_tp if e.evidence_dt is not None]),
        "tol_s": float(cfg.get("evidence_tol_s", 1.0)),
    }
    # типы — только по настоящей разметке
    trs = [r for e in evals for r in e.type_results]
    tconf: dict[str, dict[str, int]] = {}
    per_type: dict[str, dict[str, int]] = {}
    for r in trs:
        tconf.setdefault(r["gt_type"], {}).setdefault(str(r["pred_type"] or "—"), 0)
        tconf[r["gt_type"]][str(r["pred_type"] or "—")] += 1
        pt = per_type.setdefault(r["gt_type"], {"n": 0, "correct": 0})
        pt["n"] += 1
        pt["correct"] += int(r["ok"])
    pred_dist: dict[str, int] = {}
    for e in evals:
        if e.pred_positive:
            for v in e.violations_brief:
                pred_dist[v["type_id"] or "?"] = pred_dist.get(v["type_id"] or "?", 0) + 1
    types = {"available": bool(trs), "n_events": len(trs),
             "accuracy": _rate(sum(r["ok"] for r in trs), len(trs)) if trs else None,
             "accuracy_ci95": wilson(sum(r["ok"] for r in trs), len(trs)) if trs else None,
             "accuracy_localized": _rate(sum(r["ok"] for r in trs), sum(r["reason"] in ("ok", "wrong_type") for r in trs)),
             "confusion": tconf, "per_type": per_type, "pred_distribution": dict(sorted(pred_dist.items())),
             "note": TYPES_NOTE}
    # нормы — полнота ссылок, не корректность трактовки
    nrm = [n for e in evals for n in e.norms]
    nrm_tp = [n for e in tp for n in e.norms]
    cat_known = [n for n in nrm if n["in_catalog"] is not None]
    review = _norms_review(cfg)
    reviewed = [ok for e in evals for n in e.norms for ok in n.get("_review", [])] if review else []
    norms = {"n_violations": len(nrm),
             "with_rule_rate": _rate(sum(n["n_rules"] > 0 for n in nrm), len(nrm)),
             "with_rule_rate_tp": _rate(sum(n["n_rules"] > 0 for n in nrm_tp), len(nrm_tp)),
             "verified_share": round(statistics.fmean([n["verified_share"] for n in nrm]), 4) if nrm else None,
             "url_ok_rate": _rate(sum(bool(n["url_ok"]) for n in nrm), len(nrm)),
             "in_catalog_rate": _rate(sum(bool(n["in_catalog"]) for n in cat_known), len(cat_known)) if cat_known else None,
             "type_in_catalog_rate": _rate(sum(bool(n["type_in_catalog"]) for n in nrm if n["type_in_catalog"] is not None),
                                           sum(n["type_in_catalog"] is not None for n in nrm)),
             "review_available": bool(review),
             "reviewed_correct_rate": _rate(sum(reviewed), len(reviewed)) if reviewed else None,
             "note": NORMS_NOTE}
    present = [e for e in evals_all if e.status != "missing"]
    schema = {"n": len(present),
              "valid_rate": _rate(sum(not e.schema_errors for e in present), len(present)),
              "errors": {e.video_id: e.schema_errors for e in evals_all if e.schema_errors},
              "warnings": {e.video_id: e.schema_warnings for e in evals_all if e.schema_warnings}}
    lng = [x for e in evals for x in e.language]
    language = {"n": len(lng), "hedged_rate": _rate(sum(x["hedged"] for x in lng), len(lng)),
                "forbidden_rate": _rate(sum(bool(x["forbidden"]) for x in lng), len(lng)),
                "forbidden_examples": sorted({w for x in lng for w in x["forbidden"]})}
    lat_cfg = cfg.get("latency") or {}
    sla = float(lat_cfg.get("sla_s", 60.0))
    ref = float(lat_cfg.get("ref_duration_s", 145.0))

    def _sla(e: VideoEval) -> float:
        if lat_cfg.get("scale_with_duration") and e.duration:
            return sla * max(1.0, e.duration / ref)
        return sla

    lats = [e for e in evals_all if e.latency_s is not None]
    latency = {**_stats([e.latency_s for e in lats]), "sla_s": sla,
               "over_sla_ids": [e.video_id for e in lats if e.latency_s > _sla(e)],
               "within_sla_rate": _rate(sum(e.latency_s <= _sla(e) for e in lats), len(lats)),
               "rt_factor": _stats([e.latency_s / e.duration for e in lats if e.duration])}
    cats = {c: [e.video_id for e in evals_all if e.category == c] for c in CATEGORIES}
    pol_es = [e for e in tp if e.policy_dt and in_loc(e)]
    timecode_policy = {}
    for pol in ("timecode_sec", "t_start", "t_mid"):
        ds = [e.policy_dt.get(pol) for e in pol_es if e.policy_dt.get(pol) is not None]
        timecode_policy[pol] = {"n": len(ds), "median_abs_dt": round(statistics.median(ds), 3) if ds else None,
                                "hit": {f"{float(k):g}": _rate(sum(d <= float(k) + 1e-9 for d in ds), len(ds))
                                        for k in cfg.get("hit_windows_s", [0.5, 1, 2, 3])}}
    timecode_policy["note"] = ("«что если» на тех же предсказаниях (TP, ближайшее нарушение, без one-to-one): "
                               "какую точку отдавать как timecode_sec — решать замером, не на глаз")
    vp = [e.vram_process_peak_mb for e in evals_all if e.vram_process_peak_mb is not None]
    vd = [e.vram_device_peak_mb for e in evals_all if e.vram_device_peak_mb is not None]
    vdelta = [e.vram_device_peak_mb - e.vram_device_baseline_mb for e in evals_all
              if e.vram_device_peak_mb is not None and e.vram_device_baseline_mb is not None]
    resources = {
        "vram_process_peak_mb": _stats(vp),
        "vram_process_sources": sorted({e.vram_process_source for e in evals_all if e.vram_process_source}),
        "vram_process_not_measured": sum(e.vram_process_peak_mb is None for e in evals_all if e.status != "missing"),
        "vram_device_peak_mb": _stats(vd), "vram_device_delta_mb": _stats(vdelta),
        "vram_device_not_measured": sum(e.vram_device_peak_mb is None for e in evals_all if e.status != "missing"),
        "note": "процесс/воркер — память пайплайна (A03 meta.stages); устройство — memory.used всей карты "
                "(все процессы, включая чужие), delta = пик − фон до старта; None = не измерено (не 0)",
    }
    mock_ids = [e.video_id for e in evals_all if e.mock]
    video["mock_results"] = len(mock_ids)
    video["mock_ids"] = mock_ids
    out = {"video": video, "localization": localization, "localization_alt": localization_alt,
           "conflict_groups": _conflict_block(evals), "resources": resources, "timecode_policy": timecode_policy,
           "evidence": evidence, "types": types, "norms": norms, "schema": schema, "language": language,
           "latency": latency, "categories": cats}
    if dup_check is not None:
        out["duplicates"] = {"groups": dup_check,
                             "consistent_rate": _rate(sum(d["consistent"] for d in dup_check), len(dup_check))}
    return out


def git_sha(short: bool = True) -> Optional[str]:
    import subprocess

    from .common import REPO_ROOT

    try:
        cmd = ["git", "-C", str(REPO_ROOT), "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
        sha = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10, check=False).stdout.strip()
        return (sha + ("-dirty" if dirty else "")) if sha else None
    except Exception:  # noqa: BLE001
        return None


def collect_versions(preds: dict[str, dict], labels_csv: Optional[str | Path] = None,
                     extra: Optional[dict] = None) -> dict:
    import hashlib

    pv = sorted({str((p.get("meta") or {}).get("version")) for p in preds.values()
                 if isinstance(p, dict) and (p.get("meta") or {}).get("version")})
    out = {"eval_version": EVAL_VERSION, "git_sha": git_sha(), "pipeline_versions": pv}
    if labels_csv and Path(labels_csv).exists():
        out["labels_sha256"] = hashlib.sha256(Path(labels_csv).read_bytes()).hexdigest()[:16]
    out.update(extra or {})
    return out


def scorecard(summary: dict, versions: Optional[dict] = None) -> dict:
    """Главная сводка: реальные P/R/FP/FN/failed/timeouts/latency/VRAM с числом примеров, видом оценки и версиями."""
    v, lat, loc = summary["video"], summary["latency"], summary["localization"]
    res = summary.get("resources") or {}
    return {
        "evaluation_kind": v.get("evaluation_kind"), "kind_note": v.get("kind_note"),
        "n_videos": v["n"], "n_pos": v["n_pos"], "n_neg": v["n_neg"],
        "precision": v["precision"], "precision_ci95": v["precision_ci95"],
        "recall": v["recall"], "recall_ci95": v["recall_ci95"],
        "specificity": v["specificity"], "specificity_ci95": v["specificity_ci95"],
        "fp": v["fp"], "fn": v["fn"], "failed": v["failed"], "partial": v.get("partial", 0),
        "unresolved_neg": v.get("unresolved_neg", 0), "timeouts": v["timeouts"], "degraded": v["degraded"],
        "coverage_unknown": v.get("coverage_unknown", 0),
        "hit_events_all": loc["hit_events_all"], "events_total": loc["events_total"],
        "extra_candidates_total": loc["extra_candidates_total"],
        "evidence_ok_all": (summary.get("evidence") or {}).get("ok_all"),
        "types_available": (summary.get("types") or {}).get("available", False),
        "type_accuracy": (summary.get("types") or {}).get("accuracy"),
        "latency_s": {k: lat.get(k) for k in ("n", "median", "p90", "max")}, "over_sla": len(lat["over_sla_ids"]),
        "vram_process_peak_mb_max": (res.get("vram_process_peak_mb") or {}).get("max"),
        "vram_process_sources": res.get("vram_process_sources") or [],
        "vram_device_peak_mb_max": (res.get("vram_device_peak_mb") or {}).get("max"),
        "vram_device_delta_mb_max": (res.get("vram_device_delta_mb") or {}).get("max"),
        "mock_results": v.get("mock_results", 0),
        "label_conflicts": len(v["label_conflict_ids"]), "conflict_groups": len(summary.get("conflict_groups") or []),
        "label_unknown": len(v.get("label_unknown_ids") or []),
        "versions": versions or {},
    }


def evaluate(gt: dict[str, GroundTruth], preds: dict[str, dict], runinfo: Optional[dict[str, dict]] = None,
             cfg: Optional[dict] = None, catalog: Optional[dict] = None, versions: Optional[dict] = None,
             subset: Optional[set[str]] = None) -> dict:
    """Главная функция: → {"scorecard", "summary", "per_video"}. subset — ID групп (например, holdout)."""
    cfg = cfg or load_config()
    runinfo = runinfo or {}
    review = _norms_review(cfg)
    evals: list[VideoEval] = []
    dup_check: list[dict] = []
    for key, g in gt.items():
        members = g.members or [key]
        if subset is not None and key not in subset and not (set(members) & subset):
            continue
        order = [key] + [m for m in members if m != key]
        pid = next((m for m in order if m in preds), None)
        ve = evaluate_video(g, preds.get(pid) if pid else None, runinfo.get(pid or key, {}), cfg, catalog, pid)
        if review and pid:
            vios = [x for x in (preds[pid].get("violations") or []) if isinstance(x, dict)]
            for n, x in zip(ve.norms, vios):
                n["_review"] = [review[k] for k in ((str(x.get("type_id") or "").upper(),) + _rule_key(r)
                                                    for r in (x.get("rules") or []) if isinstance(r, dict))
                                if k in review]
        evals.append(ve)
        have = [m for m in members if m in preds]
        if len(have) > 1:
            verdicts = {m: (preds[m] or {}).get("verdict") for m in have}
            tcs = {m: sorted(_num(x.get("timecode_sec")) or 0.0 for x in (preds[m] or {}).get("violations") or []
                             if isinstance(x, dict)) for m in have}
            dup_check.append({"group": key, "members": have, "verdicts": verdicts, "timecodes": tcs,
                              "consistent": len(set(verdicts.values())) == 1})
    extra = sorted(set(preds) - {m for g in gt.values() for m in (g.members or [g.video_id])}, key=video_sort_key)
    summary = summarize(evals, cfg, dup_check)
    summary["unlabeled_predictions"] = extra
    versions = versions if versions is not None else collect_versions(preds)
    return {"scorecard": scorecard(summary, versions), "summary": summary, "per_video": [asdict(e) for e in evals]}


def evaluate_dirs(labels_csv: str | Path, preds_dir: str | Path, types_csv: Optional[str | Path] = None,
                  cfg: Optional[dict] = None, subset: Optional[set[str]] = None,
                  version_extra: Optional[dict] = None) -> dict:
    cfg = cfg or load_config()
    gt = load_ground_truth(labels_csv, types_csv, dedupe=bool(cfg.get("dedupe_groups", True)))
    preds, info = load_predictions(preds_dir)
    versions = collect_versions(preds, labels_csv, version_extra)
    return evaluate(gt, preds, info, cfg, load_catalog(cfg), versions=versions, subset=subset)


def public_summary(res: dict, proxy: Optional[dict] = None) -> dict:
    """Агрегированная сводка БЕЗ кадров, описаний и ID видео — единственное, что можно отдавать
    вне локальной машины (в т.ч. облачным агентам/LLM). Полные preds/HTML-отчёты — только локально."""
    s, v = res["summary"], res["summary"]["video"]
    loc, lat = s["localization"], s["latency"]
    keep = ("n", "n_pos", "n_neg", "tp", "tn", "fp", "fn", "failed", "partial", "unresolved_neg", "timeouts", "degraded",
            "coverage_unknown", "precision", "recall", "f1", "accuracy", "specificity", "precision_ci95", "recall_ci95",
            "specificity_ci95", "evaluation_kind", "kind_note")
    out = {
        "note": "Агрегаты без кадров/описаний/ID видео. Результат — подсказка, не юридическое заключение.",
        "scorecard": res.get("scorecard"),
        "video": {k: v.get(k) for k in keep} | {"label_conflicts": len(v.get("label_conflict_ids") or [])},
        "localization": {k: loc.get(k) for k in ("events_total", "events_matched", "abs_dt_matched",
                                                  "mean_signed_dt_matched", "hit_events_all", "hit_events_tp",
                                                  "extra_candidates_total", "match_window_s", "caveat")},
        "evidence": s["evidence"],
        "types": {k: s["types"].get(k) for k in ("available", "n_events", "accuracy", "accuracy_ci95", "per_type",
                                                  "pred_distribution", "note")},
        "norms": s["norms"],
        "schema": {"n": s["schema"]["n"], "valid_rate": s["schema"]["valid_rate"],
                   "n_with_errors": len(s["schema"]["errors"])},
        "language": {k: s["language"].get(k) for k in ("n", "hedged_rate", "forbidden_rate")},
        "latency": {k: lat.get(k) for k in ("n", "mean", "median", "p90", "max", "sla_s", "within_sla_rate")}
        | {"n_over_sla": len(lat.get("over_sla_ids") or [])},
        "categories": {c: len(ids) for c, ids in s["categories"].items()},
        "resources": {k: v for k, v in (s.get("resources") or {}).items()},
        "conflict_groups": len(s.get("conflict_groups") or []),
    }
    if proxy:
        out["proxy_score"] = {"value": proxy.get("proxy_score"), "disclaimer": proxy.get("disclaimer"),
                              "components": {k: {x: c.get(x) for x in ("points", "weight", "source")}
                                             for k, c in (proxy.get("components") or {}).items()}}
    return out


def _f(x: Any) -> str:
    return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def _mb(x: Any) -> str:
    return "не измерено" if x is None else f"{x:.0f} МБ"


def format_scorecard(sc: dict) -> str:
    ci = lambda c: f" [{c[0]:.2f}–{c[1]:.2f}]" if c else ""  # noqa: E731
    lat = sc["latency_s"]
    ver = sc.get("versions") or {}
    return "\n".join([
        f"SCORECARD [{sc.get('evaluation_kind')}] {sc.get('kind_note') or ''}",
        f"  n={sc['n_videos']} (+{sc['n_pos']} / -{sc['n_neg']})  eval={ver.get('eval_version')} git={ver.get('git_sha')} "
        f"pipeline={','.join(ver.get('pipeline_versions') or []) or '—'} config={ver.get('pipeline_config_sha', '—')} "
        f"labels={ver.get('labels_sha256', '—')} subset={ver.get('subset', 'all')}",
        f"  recall={_f(sc['recall'])}{ci(sc['recall_ci95'])}  precision={_f(sc['precision'])}{ci(sc['precision_ci95'])}  "
        f"specificity={_f(sc['specificity'])}{ci(sc['specificity_ci95'])}",
        f"  FN={sc['fn']}  FP={sc['fp']}  failed={sc['failed']}  partial={sc.get('partial', 0)}  "
        f"негативы без ответа={sc.get('unresolved_neg', 0)}  timeouts={sc['timeouts']}  degraded={sc['degraded']}  "
        f"coverage не сообщён={sc.get('coverage_unknown', 0)}",
        f"  hit событий (из {sc['events_total']}, без конфликтных групп): "
        + " ".join(f"@{k}={_f(x)}" for k, x in sc["hit_events_all"].items())
        + f"  кадр ±1 с={_f(sc.get('evidence_ok_all'))}  лишних кандидатов={sc['extra_candidates_total']}  "
        f"конфликтных групп={sc.get('conflict_groups', 0)}",
        f"  тип: {('acc=' + _f(sc.get('type_accuracy'))) if sc.get('types_available') else 'n/a (нет GT типов)'}",
        f"  latency: n={lat['n']} median={_f(lat['median'])} p90={_f(lat['p90'])} max={_f(lat['max'])} с; "
        f"> SLA: {sc['over_sla']}",
        f"  VRAM процесс/воркер пайплайна: peak={_mb(sc.get('vram_process_peak_mb_max'))} "
        f"({', '.join(sc.get('vram_process_sources') or []) or 'источник —'});  устройство (все процессы на карте): "
        f"peak={_mb(sc.get('vram_device_peak_mb_max'))}, прирост над фоном={_mb(sc.get('vram_device_delta_mb_max'))}",
    ] + ([f"  ⚠ {sc['mock_results']} результат(ов) получены на MOCK — это проверка кода, не измерение качества"]
         if sc.get("mock_results") else []))


def format_summary(s: dict) -> str:
    v, loc, ev, ty, nr, lat = s["video"], s["localization"], s["evidence"], s["types"], s["norms"], s["latency"]
    f = _f
    lines = [
        f"Видео: TP={v['tp']} TN={v['tn']} FP={v['fp']} FN={v['fn']} failed={v['failed']} F1={f(v['f1'])} "
        f"accuracy={f(v['accuracy'])}",
        f"  FN: {', '.join(v['fn_ids']) or '—'}   FP: {', '.join(v['fp_ids']) or '—'}   "
        f"failed: {', '.join(v['failed_ids']) or '—'}",
        f"Локализация (one-to-one, окно {loc['match_window_s']:g} с): |Δt| median={f(loc['abs_dt_matched']['median'])} "
        f"mean={f(loc['abs_dt_matched']['mean'])} signed={f(loc['mean_signed_dt_matched'])}; "
        f"без окна median={f(loc['nearest_abs_dt_unconstrained']['median'])}",
        "  hit(события в TP): " + " ".join(f"@{k}={f(x)}" for k, x in loc["hit_events_tp"].items())
        + f"; лишних кандидатов {loc['extra_candidates_total']} — {loc['caveat']}",
        f"Кадр ±{ev['tol_s']:g} с: TP={f(ev['ok_tp'])} все={f(ev['ok_all'])}  валидный JPEG={f(ev['valid_jpeg_rate'])} "
        f"1–3 кадра={f(ev['frames_per_violation_ok'])}",
        (f"Типы: acc={f(ty['accuracy'])} по {ty['n_events']} GT-событиям" if ty["available"]
         else f"Типы: n/a (нет labels_types); предсказанные: {ty['pred_distribution']}"),
        f"Нормы (полнота ссылок, не корректность): ≥1 rule={f(nr['with_rule_rate'])} verified={f(nr['verified_share'])} "
        f"adilet={f(nr['url_ok_rate'])} в каталоге={f(nr['in_catalog_rate'])} экспертиза={f(nr['reviewed_correct_rate'])}",
        f"Схема ТЗ: valid={f(s['schema']['valid_rate'])}  ошибок у {len(s['schema']['errors'])} видео; "
        f"язык: hedged={f(s['language']['hedged_rate'])} forbidden={f(s['language']['forbidden_rate'])}",
    ]
    if s.get("duplicates") and s["duplicates"]["groups"]:
        lines.append(f"Дубликаты: согласованность вердиктов={f(s['duplicates']['consistent_rate'])}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Метрики Kassa Sentinel по preds/*.json")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--types", default="data/labels_types.csv")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="metrics.json (по умолчанию <preds>/../metrics.json)")
    ap.add_argument("--no-dedupe", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, {"dedupe_groups": False} if args.no_dedupe else None)
    res = evaluate_dirs(args.labels, args.preds, args.types, cfg)
    out = Path(args.out) if args.out else Path(args.preds).resolve().parent / "metrics.json"
    write_json(out, res)
    print(format_scorecard(res["scorecard"]))
    print(format_summary(res["summary"]))
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
