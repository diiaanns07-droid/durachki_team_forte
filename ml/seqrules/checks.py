"""Ограниченные проверки по кадрам для правил (OWLv2 из ml.detect, контрастные пары запросов A05).

- phone_contrast: кроп вокруг телефона → «mobile phone» против похожих предметов стола (стационарный телефон,
  калькулятор, платёжный терминал…); боксы OWLv2 переводятся из координат кропа обратно в координаты кадра
  и сопоставляются с исходным боксом по IoU.
- safe_state: ml.detect.inventory.safe_door_state (open / closed / unknown; closed — только по свидетельству).

Каждая проверка стоит времени GPU, поэтому число и общий бюджет ограничены конфигом. Невыполненная проверка
возвращает state="skipped" и никогда не считается «проверено, всё в порядке».
"""
from __future__ import annotations

import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .observe import iou


class Checker:
    def __init__(self, clip: Any, cfg: Mapping[str, Any]):
        self.clip = clip
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", True)) and clip is not None
        self.budget_s = float(cfg.get("budget_s", 25.0))
        self.spent_s = 0.0
        self.counts = {"phone_contrast": 0, "safe_state": 0, "site_state": 0, "site_appearance": 0}
        self.skipped = {"phone_contrast": 0, "safe_state": 0, "site_state": 0, "site_appearance": 0}
        self.warnings: list[str] = []
        self._scorer: Any = None
        self._scorer_err: Optional[str] = None
        self._det_cfg: Optional[dict] = None

    # ---------------------------------------------------------------- служебное
    def _allowed(self, kind: str) -> Optional[str]:
        sub = self.cfg.get(kind, {}) or {}
        if not self.enabled:
            return "проверки выключены или нет доступа к кадрам"
        if not sub.get("enabled", True):
            return f"{kind}: выключено в конфиге"
        if self.counts[kind] >= int(sub.get("max_checks", 0)):
            return f"{kind}: исчерпан лимит {sub.get('max_checks')} проверок на видео"
        if self.spent_s >= self.budget_s:
            return f"проверки: исчерпан бюджет {self.budget_s:.0f} с"
        return None

    def _override(self) -> dict:
        """Один и тот же набор запросов для телефона и сейфа → один экземпляр OWLv2 на процесс."""
        pc = self.cfg.get("phone_contrast", {}) or {}
        ss = self.cfg.get("site_state", {}) or {}
        return {"inventory": {"mode": "rare_open_vocab", "rare_open_vocab": {"queries": {
            "seq_phone": list(pc.get("phone_queries", ["mobile phone", "smartphone"])),
            "seq_not_phone": list(pc.get("not_phone_queries", ["desk telephone", "landline phone"])),
            "seq_open": list(ss.get("open_queries", ["open safe", "open drawer"])),
            "seq_closed": list(ss.get("closed_queries", ["closed safe", "closed drawer"]))}}}}

    def _detect_cfg(self) -> dict:
        if self._det_cfg is None:
            from ml.detect.config import load_config

            self._det_cfg = load_config(self._override())
        return self._det_cfg

    def _get_scorer(self) -> Any:
        if self._scorer is None and self._scorer_err is None:
            try:
                from ml.detect.backends import get_inventory_scorer

                warns: list[str] = []
                self._scorer = get_inventory_scorer(self._detect_cfg(), warns)
                if self._scorer is None:
                    self._scorer_err = "; ".join(warns) or "OWLv2 недоступен"
                self.warnings.extend(warns)
            except Exception as exc:  # noqa: BLE001
                self._scorer_err = f"{type(exc).__name__}: {exc}"
        return self._scorer

    def summary(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "spent_s": round(self.spent_s, 2), "budget_s": self.budget_s,
                "done": dict(self.counts), "skipped": dict(self.skipped),
                "scorer_error": self._scorer_err, "warnings": self.warnings[:10]}

    # ---------------------------------------------------------------- телефон: мобильный или стационарный
    def phone_contrast(self, t: float, box: Sequence[float]) -> dict[str, Any]:
        why = self._allowed("phone_contrast")
        if why:
            self.skipped["phone_contrast"] += 1
            return {"t": round(float(t), 4), "state": "skipped", "reason": why}
        scorer = self._get_scorer()
        if scorer is None:
            self.skipped["phone_contrast"] += 1
            return {"t": round(float(t), 4), "state": "unavailable", "reason": self._scorer_err}
        from ml.detect.inventory import get_frame

        t0 = time.perf_counter()
        sub = self.cfg.get("phone_contrast", {}) or {}
        try:
            frame = get_frame(self.clip, float(t))
            if frame is None:
                return {"t": round(float(t), 4), "state": "unknown", "reason": "кадр не декодирован"}
            H, W = frame.shape[:2]
            b = np.asarray(box, np.float32) * np.array([W, H, W, H], np.float32)
            side = max(float(sub.get("crop_scale", 6.0)) * max(b[2] - b[0], b[3] - b[1]),
                       float(sub.get("crop_min_side_rel", 0.25)) * min(W, H))
            side = min(side, float(min(W, H)))
            cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            x0 = int(round(min(max(cx - side / 2.0, 0.0), W - side)))
            y0 = int(round(min(max(cy - side / 2.0, 0.0), H - side)))
            s = int(round(side))
            crop = np.ascontiguousarray(frame[y0:y0 + s, x0:x0 + s])
            boxes, probs = scorer.score([crop])[0]
            self.counts["phone_contrast"] += 1
            if len(boxes) == 0:
                return {"t": round(float(t), 4), "state": "unknown", "reason": "OWLv2 не дал боксов"}
            # кроп → кадр: x = (x0 + bx·s) / W
            fb = np.stack([(x0 + boxes[:, 0] * s) / W, (y0 + boxes[:, 1] * s) / H,
                           (x0 + boxes[:, 2] * s) / W, (y0 + boxes[:, 3] * s) / H], axis=1)
            nb = np.asarray(box, np.float32)
            ious = np.array([iou(fb[i], nb) for i in range(len(fb))])
            q = self._override()["inventory"]["rare_open_vocab"]["queries"]
            col = {tx: i for i, tx in enumerate(scorer.texts)}
            ph_cols = [col[x] for x in q["seq_phone"] if x in col]
            np_cols = [col[x] for x in q["seq_not_phone"] if x in col]
            match = ious >= float(sub.get("match_iou", 0.15))
            if not match.any() or not ph_cols or not np_cols:
                return {"t": round(float(t), 4), "state": "unknown", "reason": "нет бокса OWLv2 на месте телефона",
                        "max_iou": round(float(ious.max()), 3)}
            ph = probs[:, ph_cols].max(axis=1)
            other = probs[:, np_cols]
            oth = other.max(axis=1)
            j = int(np.argmax(np.where(match, np.maximum(ph, oth), -1.0)))
            margin = float(sub.get("margin", 0.05))
            thr = float(sub.get("min_score", 0.2))
            state = "unsure"
            if ph[j] >= thr and ph[j] - oth[j] >= margin:
                state = "mobile"
            elif oth[j] >= thr and oth[j] - ph[j] >= margin:
                state = "not_mobile"
            alt = q["seq_not_phone"][int(np.argmax(other[j]))] if len(np_cols) == len(q["seq_not_phone"]) else None
            return {"t": round(float(t), 4), "state": state, "mobile": round(float(ph[j]), 4),
                    "other": round(float(oth[j]), 4), "other_best": alt, "iou": round(float(ious[j]), 3),
                    "owl_box": [round(float(v), 5) for v in fb[j]], "source": getattr(scorer, "name", "owlv2")}
        except Exception as exc:  # noqa: BLE001 — сбой проверки = нет свидетельства
            return {"t": round(float(t), 4), "state": "error", "reason": f"{type(exc).__name__}: {exc}"}
        finally:
            self.spent_s += time.perf_counter() - t0

    # ---------------------------------------------------------------- место хранения (найдено по рукам)
    @staticmethod
    def _site_crop(frame: np.ndarray, box: Sequence[float], min_side_rel: float) -> np.ndarray:
        H, W = frame.shape[:2]
        b = np.asarray(box, np.float32) * np.array([W, H, W, H], np.float32)
        side = min(max(float(b[2] - b[0]), float(b[3] - b[1]), min_side_rel * min(W, H)), float(min(W, H)))
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        x0 = int(round(min(max(cx - side / 2.0, 0.0), W - side)))
        y0 = int(round(min(max(cy - side / 2.0, 0.0), H - side)))
        s = int(round(side))
        return np.ascontiguousarray(frame[y0:y0 + s, x0:x0 + s])

    def site_state(self, t: float, box: Sequence[float]) -> dict[str, Any]:
        """Увеличенная вырезка места: «открыто» против «закрыто» (OWLv2, боксы в центральной части вырезки)."""
        why = self._allowed("site_state")
        if why:
            self.skipped["site_state"] += 1
            return {"t": round(float(t), 4), "state": "skipped", "reason": why}
        scorer = self._get_scorer()
        if scorer is None:
            self.skipped["site_state"] += 1
            return {"t": round(float(t), 4), "state": "unavailable", "reason": self._scorer_err}
        from ml.detect.inventory import get_frame

        t0 = time.perf_counter()
        sub = self.cfg.get("site_state", {}) or {}
        try:
            frame = get_frame(self.clip, float(t))
            if frame is None:
                return {"t": round(float(t), 4), "state": "unknown", "reason": "кадр не декодирован"}
            crop = self._site_crop(frame, box, float(sub.get("crop_min_side_rel", 0.18)))
            boxes, probs = scorer.score([crop])[0]
            self.counts["site_state"] += 1
            q = self._override()["inventory"]["rare_open_vocab"]["queries"]
            col = {tx: i for i, tx in enumerate(scorer.texts)}
            oc = [col[x] for x in q["seq_open"] if x in col]
            cc = [col[x] for x in q["seq_closed"] if x in col]
            if len(boxes) == 0 or not oc or not cc:
                return {"t": round(float(t), 4), "state": "unknown", "reason": "OWLv2 не дал боксов"}
            cxy = (boxes[:, :2] + boxes[:, 2:]) / 2.0
            central = (np.abs(cxy - 0.5) <= float(sub.get("central", 0.35))).all(axis=1)
            if not central.any():
                return {"t": round(float(t), 4), "state": "unknown", "reason": "в центре вырезки нет предмета"}
            op = np.where(central, probs[:, oc].max(axis=1), -1.0)
            cl = np.where(central, probs[:, cc].max(axis=1), -1.0)
            so, scl = float(op.max()), float(cl.max())
            margin, thr = float(sub.get("margin", 0.05)), float(sub.get("min_score", 0.2))
            state = "unsure"
            if so >= thr and so - scl >= margin:
                state = "open"
            elif scl >= thr and scl - so >= margin:
                state = "closed"
            return {"t": round(float(t), 4), "state": state, "open": round(so, 4), "closed": round(scl, 4),
                    "bbox": [round(float(v), 5) for v in box], "source": getattr(scorer, "name", "owlv2")}
        except Exception as exc:  # noqa: BLE001
            return {"t": round(float(t), 4), "state": "error", "reason": f"{type(exc).__name__}: {exc}"}
        finally:
            self.spent_s += time.perf_counter() - t0

    def site_appearance(self, t: float, box: Sequence[float]) -> Optional[np.ndarray]:
        """Вид места для сравнения «до / после»: 32×32 серый, нормированный (без цвета и абсолютной яркости).
        Возвращается только в правило, в наблюдения не пишется."""
        if self._allowed("site_appearance"):
            self.skipped["site_appearance"] += 1
            return None
        import cv2
        from ml.detect.inventory import get_frame

        t0 = time.perf_counter()
        try:
            frame = get_frame(self.clip, float(t))
            if frame is None:
                return None
            crop = self._site_crop(frame, box, float((self.cfg.get("site_state", {}) or {}).get("crop_min_side_rel", 0.18)))
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            v = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
            self.counts["site_appearance"] += 1
            sd = float(v.std())
            return (v - float(v.mean())) / sd if sd > 1e-6 else None
        except Exception:  # noqa: BLE001
            return None
        finally:
            self.spent_s += time.perf_counter() - t0

    # ---------------------------------------------------------------- сейф
    def safe_state(self, t: float, zones: Any) -> dict[str, Any]:
        why = self._allowed("safe_state")
        if why:
            self.skipped["safe_state"] += 1
            return {"t": round(float(t), 4), "state": "skipped", "reason": why}
        t0 = time.perf_counter()
        try:
            from ml.detect.inventory import safe_door_state

            st = safe_door_state(self.clip, float(t), zones, config=self._override())
            self.counts["safe_state"] += 1
            return {"t": round(float(t), 4), "state": st.state, "conf": st.conf, "mode": st.mode, "zone": st.zone,
                    "bbox": list(st.bbox) if st.bbox else None, "votes": st.votes, "warnings": st.warnings[:3]}
        except Exception as exc:  # noqa: BLE001
            return {"t": round(float(t), 4), "state": "error", "reason": f"{type(exc).__name__}: {exc}"}
        finally:
            self.spent_s += time.perf_counter() - t0
