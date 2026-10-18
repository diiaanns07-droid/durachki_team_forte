"""Журнал итераций: reports/runs.jsonl (только агрегаты — можно отдавать наружу) + HTML-график.

Каждая запись — один прогон одной процедурой (protocol.json): гипотеза, SHA, хэши конфигов/входов,
FP/FN, recall/precision, hit@1/2/3, кадр ±1 с, failed/partial/timeouts, latency, VRAM, proxy-балл.

CLI::

    python -m eval.runs add --run runs/eval/dev1 --label "baseline"     # обычно делает eval.run сам
    python -m eval.runs list
    python -m eval.runs html --out reports/runs.html
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Optional

from .common import read_json

DEFAULT_LOG = Path("reports/runs.jsonl")


def record_from_run(run_dir: str | Path, label: Optional[str] = None, role: str = "single",
                    ablation_id: Optional[str] = None) -> dict:
    run = Path(run_dir)
    m = read_json(run / "metrics.json")
    prot = read_json(run / "protocol.json") if (run / "protocol.json").exists() else {}
    proxy = read_json(run / "proxy_score.json") if (run / "proxy_score.json").exists() else {}
    sc, s = m.get("scorecard") or {}, m["summary"]
    lat = sc.get("latency_s") or {}
    return {
        "run_id": prot.get("run_id") or run.name, "at": prot.get("created_at"), "label": label or prot.get("label"),
        "role": role, "ablation_id": ablation_id, "run_dir": str(run),
        "protocol": {k: prot.get(k) for k in ("git_sha", "eval_version", "eval_config_sha", "pipeline_config_sha",
                                              "labels_sha256", "mode", "subset", "evaluation_kind")}
        | {"inputs_sha": (prot.get("inputs") or {}).get("inputs_sha"), "pipeline_env": prot.get("pipeline_env")},
        "metrics": {
            "n": sc.get("n_videos"), "n_pos": sc.get("n_pos"), "n_neg": sc.get("n_neg"),
            "fp": sc.get("fp"), "fn": sc.get("fn"), "recall": sc.get("recall"), "precision": sc.get("precision"),
            "specificity": sc.get("specificity"), "failed": sc.get("failed"), "partial": sc.get("partial"),
            "unresolved_neg": sc.get("unresolved_neg"), "timeouts": sc.get("timeouts"),
            "hit_1": (sc.get("hit_events_all") or {}).get("1"), "hit_2": (sc.get("hit_events_all") or {}).get("2"),
            "hit_3": (sc.get("hit_events_all") or {}).get("3"), "evidence_ok": sc.get("evidence_ok_all"),
            "extra_candidates": sc.get("extra_candidates_total"), "type_accuracy": sc.get("type_accuracy"),
            "latency_p50": lat.get("median"), "latency_p90": lat.get("p90"), "latency_max": lat.get("max"),
            "vram_process_peak_mb": sc.get("vram_process_peak_mb_max"),
            "vram_device_peak_mb": sc.get("vram_device_peak_mb_max"), "mock_results": sc.get("mock_results"),
            "proxy_score": proxy.get("proxy_score"),
            "median_abs_dt": ((s.get("localization") or {}).get("abs_dt_matched") or {}).get("median"),
        },
    }


def append(log: str | Path, rec: dict) -> None:
    p = Path(log)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def load(log: str | Path) -> list[dict]:
    p = Path(log)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


# ----------------------------------------------------------------------------- HTML (маленькие мультипликаторы)
_CSS = """
:root{--surface:#fcfcfb;--text:#0b0b0b;--text2:#52514e;--grid:#e4e3df;--s1:#2a78d6;--s2:#eb6834}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--surface:#1a1a19;--text:#f0efec;--text2:#c3c2b7;
--grid:#383835;--s1:#3987e5;--s2:#d95926}}
:root[data-theme=dark]{--surface:#1a1a19;--text:#f0efec;--text2:#c3c2b7;--grid:#383835;--s1:#3987e5;--s2:#d95926}
body{background:var(--surface);color:var(--text);font:14px/1.45 system-ui,sans-serif;margin:0;padding:16px;max-width:1100px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.card h3{margin:0 0 4px;font-size:14px}.muted{color:var(--text2);font-size:12px}
.legend span{display:inline-flex;align-items:center;gap:4px;margin-right:12px;font-size:12px;color:var(--text2)}
.sw{width:10px;height:10px;border-radius:50%;display:inline-block}
table{border-collapse:collapse;width:100%;font-size:12px;margin-top:16px}td,th{border-bottom:1px solid var(--grid);
padding:3px 6px;text-align:left}th{color:var(--text2)}.wrap{overflow-x:auto}
svg text{fill:var(--text2);font-size:11px}
"""


def _chart(runs: list[dict], series: list[tuple[str, str, str]], title: str, unit: str, w: int = 340, h: int = 170) -> str:
    """Линия + маркеры по итерациям; одна ось Y на график; подсказка на каждой точке (<title>)."""
    pad_l, pad_r, pad_t, pad_b = 36, 10, 10, 22
    vals = [r["metrics"].get(k) for r in runs for k, _, _ in series if isinstance(r["metrics"].get(k), (int, float))]
    if not vals:
        return f'<div class=card><h3>{html.escape(title)}</h3><p class=muted>нет данных</p></div>'
    vmax = max(vals) if max(vals) > 0 else 1.0
    if unit == "доля":
        vmax = 1.0
    n = len(runs)
    x = lambda i: pad_l + (w - pad_l - pad_r) * (i / max(1, n - 1) if n > 1 else 0.5)  # noqa: E731
    y = lambda v: pad_t + (h - pad_t - pad_b) * (1 - v / vmax)  # noqa: E731
    out = [f'<svg width="100%" viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(title)}">']
    for frac in (0, 0.5, 1):
        yy = y(vmax * frac)
        out.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="var(--grid)"/>'
                   f'<text x="{pad_l - 4}" y="{yy + 4:.1f}" text-anchor="end">{vmax * frac:.3g}</text>')
    for i, r in enumerate(runs):
        if n <= 12 or i % max(1, n // 12) == 0:
            out.append(f'<text x="{x(i):.1f}" y="{h - 6}" text-anchor="middle">{i + 1}</text>')
    for key, color, name in series:
        pts = [(i, r["metrics"].get(key)) for i, r in enumerate(runs) if isinstance(r["metrics"].get(key), (int, float))]
        if len(pts) > 1:
            out.append('<polyline fill="none" stroke-width="2" stroke="' + color + '" points="'
                       + " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in pts) + '"/>')
        for i, v in pts:
            r = runs[i]
            tip = f"#{i + 1} {r.get('label') or r.get('run_id')} · {name}: {v:.4g} · git {r['protocol'].get('git_sha')}"
            out.append(f'<g><circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="10" fill="transparent"><title>{html.escape(tip)}'
                       f'</title></circle><circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="4" fill="{color}" '
                       f'stroke="var(--surface)" stroke-width="2" pointer-events="none"/></g>')
        if pts:  # прямая подпись последнего значения
            i, v = pts[-1]
            out.append(f'<text x="{min(x(i) + 6, w - 2):.1f}" y="{y(v) - 6:.1f}" text-anchor="end">{v:.3g}</text>')
    out.append("</svg>")
    legend = ""
    if len(series) > 1:
        legend = '<div class=legend>' + "".join(f'<span><i class=sw style="background:{c}"></i>{html.escape(nm)}</span>'
                                               for _, c, nm in series) + "</div>"
    return f'<div class=card><h3>{html.escape(title)}</h3><div class=muted>{html.escape(unit)}</div>{legend}{"".join(out)}</div>'


def render_html(runs_all: list[dict], title: str = "Kassa Sentinel — журнал итераций") -> str:
    runs = [r for r in runs_all if isinstance(r.get("metrics"), dict)]
    verdicts = [r for r in runs_all if r.get("role") == "ablation"]
    charts = [
        _chart(runs, [("fn", "var(--s1)", "FN (пропуски)"), ("fp", "var(--s2)", "FP (ложные)")], "Ошибки по видео", "шт."),
        _chart(runs, [("hit_2", "var(--s1)", "hit@2")], "Попадание таймкода ±2 с", "доля"),
        _chart(runs, [("latency_p90", "var(--s1)", "p90")], "Латентность p90", "с"),
        _chart(runs, [("vram_process_peak_mb", "var(--s1)", "процесс/воркер пайплайна"),
                      ("vram_device_peak_mb", "var(--s2)", "устройство (все процессы)")], "Пик VRAM", "МБ"),
    ]
    cols = ["label", "role", "n", "fn", "fp", "recall", "precision", "hit_1", "hit_2", "hit_3", "evidence_ok", "failed",
            "partial", "timeouts", "latency_p90", "vram_process_peak_mb", "vram_device_peak_mb", "mock_results",
            "proxy_score"]
    rows = []
    for i, r in enumerate(runs, 1):
        m = r["metrics"]
        cells = [r.get("label") or r.get("run_id"), r.get("role")] + [m.get(c) for c in cols[2:]]
        rows.append(f"<tr><td>{i}</td>" + "".join(f"<td>{html.escape('—' if c is None else str(c))}</td>" for c in cells)
                    + f"<td>{html.escape(str(r['protocol'].get('git_sha')))}</td>"
                    f"<td>{html.escape(str(r['protocol'].get('evaluation_kind')))}</td></tr>")
    table = ("<div class=wrap><table><tr><th>#</th>" + "".join(f"<th>{c}</th>" for c in cols)
             + "<th>git</th><th>вид</th></tr>" + "".join(rows) + "</table></div>")
    if verdicts:
        table += ("<h2>Абляции</h2><div class=wrap><table><tr><th>гипотеза</th><th>вердикт</th><th>изменено</th>"
                  "<th>ΔFN</th><th>ΔFP</th><th>Δhit@2</th><th>Δp90, с</th><th>исправлено/сломано</th></tr>"
                  + "".join(f"<tr><td>{html.escape(str(v.get('ablation_id')))}</td><td><b>{html.escape(str(v.get('verdict')))}"
                            f"</b></td><td>{html.escape(', '.join(v.get('changed_params') or []) or '—')}</td>"
                            f"<td>{(v.get('deltas') or {}).get('fn')}</td><td>{(v.get('deltas') or {}).get('fp')}</td>"
                            f"<td>{((v.get('deltas') or {}).get('hit') or {}).get('2')}</td>"
                            f"<td>{(v.get('deltas') or {}).get('latency_p90_s')}</td>"
                            f"<td>{(v.get('paired') or {}).get('fixed')}/{(v.get('paired') or {}).get('broken')}</td></tr>"
                            for v in verdicts) + "</table></div>")
    return (f"<!doctype html><html lang=ru><head><meta charset=utf-8><meta name=viewport content='width=device-width,"
            f"initial-scale=1'><title>Журнал итераций</title><style>{_CSS}</style></head><body><h1>{html.escape(title)}</h1>"
            "<p class=muted>Только агрегаты. Все прогоны на 42 видео — разведочные (не независимый тест). Proxy-балл — "
            "наша аппроксимация, не формула жюри.</p>"
            f"<div class=grid>{''.join(charts)}</div>{table}</body></html>")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Журнал итераций (reports/runs.jsonl)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("--run", required=True)
    a.add_argument("--label", default=None)
    a.add_argument("--role", default="single")
    a.add_argument("--log", default=str(DEFAULT_LOG))
    lst = sub.add_parser("list")
    lst.add_argument("--log", default=str(DEFAULT_LOG))
    h = sub.add_parser("html")
    h.add_argument("--log", default=str(DEFAULT_LOG))
    h.add_argument("--out", default="reports/runs.html")
    args = ap.parse_args(argv)
    if args.cmd == "add":
        rec = record_from_run(args.run, args.label, args.role)
        append(args.log, rec)
        print(f"+ {rec['run_id']} → {args.log}")
    elif args.cmd == "list":
        for i, r in enumerate(load(args.log), 1):
            if r.get("role") == "ablation":
                print(f"{i:3d} АБЛЯЦИЯ {r.get('ablation_id')}: {r.get('verdict')} Δ={r.get('deltas', {}).get('fn')}FN/"
                      f"{r.get('deltas', {}).get('fp')}FP")
                continue
            m: dict[str, Any] = r["metrics"]
            print(f"{i:3d} {str(r.get('label'))[:40]:40s} FN={m.get('fn')} FP={m.get('fp')} hit@2={m.get('hit_2')} "
                  f"p90={m.get('latency_p90')} VRAM процесс={m.get('vram_process_peak_mb')} "
                  f"устройство={m.get('vram_device_peak_mb')} git={r['protocol'].get('git_sha')}")
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(render_html(load(args.log)), encoding="utf-8")
        print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
