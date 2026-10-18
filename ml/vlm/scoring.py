"""Смешивание уверенности события с ответом VLM (логит-пулинг) и классификация согласия.

logit(p_post) = logit(p_prior) + w * logit(p_vlm), где p_vlm = c при visible=true, иначе 1 - c
(c — заявленная уверенность VLM, обрезанная в p_vlm_clip; w < 1 — VLM переуверена и коррелирует с правилами).
«Пола» для сильных правил нет: противоречие кадров понижает уверенность и помечается флагом для fusion.
"""
from __future__ import annotations

import math
from typing import Any, Optional

AGREEMENTS = ("supports", "contradicts", "uncertain", "type_mismatch", "not_checked")


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def logit(p: float) -> float:
    p = clamp(float(p), 1e-4, 1 - 1e-4)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def p_vlm(visible: bool, conf: float, clip: tuple[float, float] = (0.05, 0.95)) -> float:
    p = conf if visible else 1.0 - conf
    return clamp(p, clip[0], clip[1])


def blend(p_prior: float, pv: float, w: float) -> float:
    return sigmoid(logit(p_prior) + float(w) * logit(pv))


def agreement(pv: float, thr: float = 0.6) -> str:
    if pv >= thr:
        return "supports"
    if pv <= 1.0 - thr:
        return "contradicts"
    return "uncertain"


def update_confidence(p_prior: float, visible: bool, conf: float, candidate_type: str,
                      vlm_type: Optional[str], cfg: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """-> (новая уверенность, трейс). Тип VLM ≠ кандидат при visible=true -> type_mismatch (без изменения p)."""
    vcfg = cfg.get("verify") or {}
    w = float(vcfg.get("vlm_weight", 0.6))
    clip = tuple(vcfg.get("p_vlm_clip", (0.05, 0.95)))
    thr = float(vcfg.get("agree_threshold", 0.6))
    max_raise = float(vcfg.get("max_raise", 0.25))
    p0 = float(p_prior)
    p0 = 0.5 if not math.isfinite(p0) else clamp(p0, 0.0, 1.0)  # NaN/inf от источника -> нейтральный приор
    conf = float(conf) if math.isfinite(float(conf)) else 0.5
    if visible and vlm_type and vlm_type != candidate_type:
        return p0, {"p_prior": round(p0, 4), "p_vlm": None, "p_post": round(p0, 4), "agreement": "type_mismatch"}
    pv = p_vlm(bool(visible), float(conf), (float(clip[0]), float(clip[1])))
    p = blend(p0, pv, w)
    p = min(p, p0 + max_raise)
    agr = agreement(pv, thr)
    return p, {"p_prior": round(p0, 4), "p_vlm": round(pv, 4), "p_post": round(p, 4), "agreement": agr}


# Мнение VLM в конвенции ml.fusion (A12): extra.vlm = {"verdict": confirm|reject|unsure, "score": 0..1}.
_AGREEMENT_TO_VERDICT = {"supports": "confirm", "contradicts": "reject", "uncertain": "unsure",
                         "type_mismatch": "unsure"}


def fusion_opinion(agreement: str, pv: Optional[float]) -> dict[str, Any]:
    """{"verdict", "score"} для fusion; для not_checked — {} (fusion не видит мнения и ничего не меняет).

    score — уверенность VLM в СВОЁМ вердикте: confirm -> p_vlm, reject -> 1 - p_vlm, unsure -> 0.5.
    """
    verdict = _AGREEMENT_TO_VERDICT.get(agreement)
    if verdict is None:
        return {}
    if verdict == "confirm" and pv is not None:
        score = pv
    elif verdict == "reject" and pv is not None:
        score = 1.0 - pv
    else:
        score = 0.5
    return {"verdict": verdict, "score": round(float(score), 4)}
