"""Kassa Sentinel — eval (владелец A16).

Инструменты оценки: разметка (labels), групповой split (split), протокол прогона (protocol), прогон (run),
метрики и scorecard (metrics), санити-полы (sanity), абляции (ablate), журнал итераций (runs), black-box
приёмка API (jury_sim), наши съёмки с замороженной камерой (extra), proxy-балл (score_like_jury),
синтетические возмущения (robustness), HTML-отчёт (report), LOGO/nested CV (loo).

Импорт пакета не тянет тяжёлых зависимостей и не имеет побочных эффектов.
"""
from __future__ import annotations

__all__ = ["labels", "split", "protocol", "run", "metrics", "sanity", "ablate", "runs", "jury_sim", "extra", "schema",
           "score_like_jury", "robustness", "report", "loo"]
