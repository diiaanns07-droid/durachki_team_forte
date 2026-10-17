"""Самодостаточный HTML-отчёт по прогону (только локально: содержит кадры из датасета!).

Таблица по видео (GT vs pred, Δt, тип, вердикт, категория ошибки, латентность), миниатюры кадров-доказательств,
полоса времени (GT, нарушения, кривая риска), сводка (scorecard), hit@k. Без внешних скриптов и шрифтов.

CLI::

    python -m eval.report --run runs/eval/dev1 [--no-frames] [--robustness runs/robust/robustness.json]
"""
from __future__ import annotations

import argparse
import base64
import html
from pathlib import Path
from typing import Any, Optional

from .common import read_json
from .metrics import format_scorecard

CAT = {  # категория → (статус-цвет, значок, подпись) — цвет никогда не единственный носитель смысла
    "TP": ("var(--good)", "✓", "верно"), "TN": ("var(--good)", "✓", "верно, без нарушения"),
    "FN": ("var(--critical)", "✗", "ПРОПУСК"), "FP": ("var(--serious)", "!", "ложное срабатывание"),
    "early": ("var(--warning)", "←", "рано"), "late": ("var(--warning)", "→", "поздно"),
    "wrong_type": ("var(--warning)", "≠", "не тот тип"), "failed": ("var(--critical)", "✗", "ошибка/таймаут"),
    "label_conflict": ("var(--text-secondary)", "?", "конфликт меток"),
    "partial": ("var(--critical)", "◐", "неполный анализ (не «чисто»)"),
    "label_unknown": ("var(--text-secondary)", "?", "метка неизвестна"),
}

CSS = """
:root{--surface:#fcfcfb;--text:#0b0b0b;--text2:#52514e;--grid:#e4e3df;--series:#2a78d6;--good:#0ca30c;
--warning:#fab219;--serious:#ec835a;--critical:#d03b3b;--text-secondary:#52514e}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--surface:#1a1a19;--text:#f0efec;--text2:#c3c2b7;
--grid:#383835;--series:#3987e5;--text-secondary:#c3c2b7}}
:root[data-theme=dark]{--surface:#1a1a19;--text:#f0efec;--text2:#c3c2b7;--grid:#383835;--series:#3987e5;--text-secondary:#c3c2b7}
body{background:var(--surface);color:var(--text);font:14px/1.45 system-ui,sans-serif;margin:0;padding:16px;max-width:1280px}
.banner{border:2px solid var(--critical);padding:8px 12px;border-radius:6px;margin-bottom:12px}
pre{background:transparent;border:1px solid var(--grid);padding:8px;overflow-x:auto;font-size:12px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid var(--grid);padding:4px 6px;
text-align:left;vertical-align:middle}th{color:var(--text2);font-weight:600}
.chip{display:inline-flex;gap:4px;align-items:center;padding:1px 6px;border-radius:10px;border:1.5px solid}
.bar{height:10px;background:var(--series);border-radius:0 4px 4px 0}.muted{color:var(--text2)}
img.thumb{height:72px;border-radius:4px;margin-right:4px}.wrap{overflow-x:auto}
"""


def _chip(cat: str) -> str:
    color, icon, label = CAT.get(cat, ("var(--text2)", "·", cat))
    return f'<span class="chip" style="border-color:{color}"><b style="color:{color}">{icon}</b>{html.escape(label)}</span>'


def _thumb(b64: str, height: int = 144) -> str:
    """Уменьшенная копия (cv2, если есть) — отчёт легче; иначе исходный JPEG."""
    try:
        import cv2
        import numpy as np

        img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
        if img is not None and img.shape[0] > height:
            w = int(img.shape[1] * height / img.shape[0])
            ok, buf = cv2.imencode(".jpg", cv2.resize(img, (w, height)), [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                return base64.b64encode(buf.tobytes()).decode()
    except Exception:  # noqa: BLE001
        pass
    return b64


def _strip(e: dict, api: dict, width: int = 260, height: int = 26) -> str:
    """Полоса времени: кривая риска (серая), GT (вертикаль), нарушения (точки). Подсказки — <title>."""
    tl = [p for p in (api.get("timeline") or []) if isinstance(p, (list, tuple)) and len(p) == 2]
    dur = e.get("duration") or max([p[0] for p in tl] + [t for t in e.get("gt_times") or []] +
                                   [v["t"] or 0 for v in e.get("violations_brief") or []] + [1.0])
    x = lambda t: 2 + (width - 4) * max(0.0, min(1.0, float(t) / dur))  # noqa: E731
    parts = [f'<svg width="{width}" height="{height}" role="img" aria-label="полоса времени">',
             f'<line x1="2" y1="{height - 4}" x2="{width - 2}" y2="{height - 4}" stroke="var(--grid)"/>']
    if tl:
        pts = " ".join(f"{x(t):.1f},{height - 4 - (height - 8) * max(0, min(1, float(r))):.1f}" for t, r in tl)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="var(--text2)" stroke-width="1.5"><title>риск</title></polyline>')
    for g in e.get("gt_times") or []:
        parts.append(f'<line x1="{x(g):.1f}" y1="2" x2="{x(g):.1f}" y2="{height - 2}" stroke="var(--text)" '
                     f'stroke-width="2"><title>GT {g:g} с</title></line>')
    for v in e.get("violations_brief") or []:
        if v.get("t") is not None:
            parts.append(f'<circle cx="{x(v["t"]):.1f}" cy="{height / 2:.1f}" r="4.5" fill="var(--series)" '
                         f'stroke="var(--surface)" stroke-width="2"><title>{html.escape(str(v.get("type_id") or ""))} '
                         f'{v["t"]:g} с conf={v.get("conf")}</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


def build_report(metrics: dict, preds: Optional[dict[str, dict]] = None, robustness: Optional[dict] = None,
                 proxy: Optional[dict] = None, with_frames: bool = True, title: str = "Kassa Sentinel — eval",
                 sanity: Optional[dict] = None) -> str:
    preds = preds or {}
    s, sc = metrics["summary"], metrics.get("scorecard")
    out = [f"<!doctype html><html lang=ru><head><meta charset=utf-8><meta name=viewport content='width=device-width,"
           f"initial-scale=1'><title>{html.escape(title)}</title><style>{CSS}</style></head><body>",
           f"<h1>{html.escape(title)}</h1>",
           '<div class="banner"><b>ТОЛЬКО ЛОКАЛЬНО.</b> Отчёт содержит кадры из датасета: не коммитить, не загружать '
           'в облако/LLM. Наружу — summary_public.json. Результаты — подсказка, не юридическое заключение.</div>']
    if sc:
        out.append(f"<h2>Scorecard</h2><pre>{html.escape(format_scorecard(sc))}</pre>")
    if sanity:
        from .sanity import format_sanity

        out.append(f"<h2>Санити-полы и утечки</h2><pre>{html.escape(format_sanity(sanity))}</pre>")
    tp = s.get("timecode_policy") or {}
    if tp:
        out.append("<h2>Политика таймкода («что если»)</h2><table><tr><th>точка</th><th>n</th><th>median |Δt|</th>"
                   "<th>hit</th></tr>" + "".join(
                       f"<tr><td>{html.escape(k)}</td><td>{v['n']}</td><td>{v['median_abs_dt']}</td>"
                       f"<td>{html.escape(str(v['hit']))}</td></tr>" for k, v in tp.items() if isinstance(v, dict))
                   + f"</table><p class=muted>{html.escape(tp.get('note', ''))}</p>")
    if proxy:
        out.append(f"<p class=muted>Proxy-балл (наш, формула жюри неизвестна): <b>{proxy.get('proxy_score')}</b> / 100; "
                   f"n/a: {html.escape(', '.join(proxy.get('missing') or []) or '—')}</p>")
    loc = s["localization"]
    out.append("<h2>Попадание по времени (события)</h2><table><tr><th>окно</th><th>доля событий</th><th></th></tr>")
    for k, val in (loc.get("hit_events_all") or {}).items():
        w = int(200 * (val or 0))
        out.append(f"<tr><td>±{k} с</td><td>{'—' if val is None else f'{val:.2f}'}</td>"
                   f"<td><div class=bar style='width:{w}px' title='{val}'></div></td></tr>")
    out.append(f"</table><p class=muted>{html.escape(loc.get('caveat', ''))} Лишних кандидатов: "
               f"{loc.get('extra_candidates_total')}.</p>")
    cats = s.get("categories") or {}
    out.append("<h2>Категории</h2><p>" + " ".join(f"{_chip(c)} {len(ids)}" for c, ids in cats.items() if ids) + "</p>")
    if robustness:
        out.append("<h2>Устойчивость (синтетические возмущения)</h2><p class=muted>"
                   f"{html.escape(robustness.get('disclaimer', ''))}</p><table><tr><th>вариант</th><th>n</th>"
                   "<th>смена вердикта</th><th>стабильный таймкод</th><th>сдвиг, с</th><th>acc vs GT</th></tr>")
        for var, d in (robustness.get("per_variant") or {}).items():
            out.append(f"<tr><td>{html.escape(var)}</td><td>{d['n']}</td><td>{d['flip_rate']}</td>"
                       f"<td>{d['timecode_stable_rate']}</td><td>{d['mean_shift_s']}</td><td>{d['accuracy_vs_gt']}</td></tr>")
        out.append("</table>")
    out.append("<h2>По видео</h2><div class=wrap><table><tr><th>видео</th><th>GT</th><th>предсказание</th><th>Δt, с</th>"
               "<th>тип</th><th>категория</th><th>время</th><th>лишние</th><th>полоса</th>"
               + ("<th>кадры</th>" if with_frames else "") + "</tr>")
    for e in metrics["per_video"]:
        api = preds.get(e.get("pred_id") or e["video_id"]) or {}
        gt = "нарушение " + ", ".join(f"{t:g}" for t in e["gt_times"]) if e["gt_label"] == "violation" else e["gt_label"]
        pred = (e.get("verdict") or e.get("status") or "—") + (f" @ {e['pred_t']:g}" if e.get("pred_t") is not None else "")
        dt = e.get("dt_signed") if e.get("dt_signed") is not None else e.get("nearest_dt_signed")
        gt_types = ", ".join(t for _, t in e.get("gt_types") or [])
        lat_s = "—" if e.get("latency_s") is None else f"{e['latency_s']:.1f}"
        row = (f"<tr><td>{html.escape(e['video_id'])}{' <span class=muted>(дубль/конфликт)</span>' if e.get('conflict') else ''}"
               f"</td><td>{html.escape(gt)}</td><td>{html.escape(pred)}</td>"
               f"<td>{'—' if dt is None else f'{dt:+.1f}'}</td>"
               f"<td>{html.escape(str(e.get('pred_type') or '—'))}{(' / GT ' + html.escape(gt_types)) if gt_types else ''}</td>"
               f"<td>{_chip(e['category'])}</td><td>{lat_s}</td>"
               f"<td>{e.get('extra_candidates', 0)}</td><td>{_strip(e, api)}</td>")
        if with_frames:
            imgs = []
            for v in (api.get("violations") or [])[:2]:
                for f in (v.get("evidence_frames") or [])[:1]:
                    if isinstance(f, dict) and f.get("image_base64"):
                        imgs.append(f'<img class=thumb alt="кадр {f.get("timestamp_sec")} с" title="{f.get("timestamp_sec")} с" '
                                    f'src="data:image/jpeg;base64,{_thumb(f["image_base64"])}">')
            row += f"<td>{''.join(imgs) or '—'}</td>"
        out.append(row + "</tr>")
    out.append("</table></div></body></html>")
    return "\n".join(out)


def main(argv: Optional[list[str]] = None) -> int:
    from .metrics import load_predictions

    ap = argparse.ArgumentParser(description="HTML-отчёт по прогону (локально)")
    ap.add_argument("--run", required=True, help="папка прогона eval.run (metrics.json, preds/)")
    ap.add_argument("--robustness", default=None)
    ap.add_argument("--no-frames", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    run = Path(args.run)
    metrics = read_json(run / "metrics.json")
    preds = load_predictions(run / "preds")[0] if (run / "preds").exists() else {}
    proxy = read_json(run / "proxy_score.json") if (run / "proxy_score.json").exists() else None
    san = read_json(run / "sanity.json") if (run / "sanity.json").exists() else None
    rob = read_json(args.robustness) if args.robustness else None
    out = Path(args.out) if args.out else run / "report.html"
    out.write_text(build_report(metrics, preds, rob, proxy, not args.no_frames, sanity=san), encoding="utf-8")
    print(f"→ {out} (только локально: содержит кадры)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
