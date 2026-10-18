"""Локальная страница разбора ошибок прогона (FN + FP) для человека — CPU, без отправки данных куда-либо.

Для каждого ошибочного видео: плеер с переходом к метке и к таймкоду сервиса, кривая риска сервиса с отметками,
контакт-листы (вокруг метки — для пропусков; вокруг таймкода сервиса — для ложных тревог), кадры-доказательства
и описание сервиса, статусы стадий, форма «причина ошибки / какой тип на самом деле» с выгрузкой CSV.
Результат — cache/error_review/<run>/index.html (локально; НЕ публиковать). В stdout — только счётчики.

usage: python -I -X utf8 tools/dataprep/error_review.py <run_dir> [--root .] [--catalog norms/catalog.yaml]
"""
from __future__ import annotations

import argparse
import base64
import binascii
import html
import importlib.util
import json
from pathlib import Path


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_er, _cs, _tc = _load("eval_run"), _load("contact_sheets"), _load("type_catalog")

FP_CAUSES = ["", "детектор: ложный объект", "объект есть, но не у кассира / не в зоне кассира",
             "нормальная процедура принята за нарушение", "событие есть — разметка спорная", "другое"]
FN_CAUSES = ["", "нарушение видно, правила его не ловят (тип не покрыт)", "сигнал был, но отсечён порогом / VLM",
             "плохо видно: мелко, перекрыто, вне кадра", "разметка спорная", "другое"]

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--fg:#1d2330;--mut:#5d6677;--line:#dfe3ea;--acc:#2457d6;--bad:#c23b3b;--ok:#1d8a4e;--chip:#eef2fb}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#14171d;--card:#1c2029;--fg:#e6e9ef;--mut:#9aa3b2;
--line:#2c3240;--acc:#7aa2ff;--bad:#ff7b7b;--ok:#4cc38a;--chip:#242b3a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,Arial}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;
display:flex;gap:12px;align-items:center;flex-wrap:wrap}h1{font-size:17px;margin:0}main{max-width:1500px;margin:0 auto;padding:12px 16px 80px}
button{font:inherit;padding:6px 12px;border-radius:6px;border:1px solid var(--line);background:var(--chip);color:var(--fg);cursor:pointer}
button.pri{background:var(--acc);color:#fff;border-color:var(--acc)}.card{background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:14px;margin:14px 0}.card.done{border-left:5px solid var(--ok)}
.row{display:grid;grid-template-columns:minmax(260px,330px) 1fr minmax(280px,380px);gap:14px}@media (max-width:1100px){.row{grid-template-columns:1fr}}
h2{font-size:16px;margin:0 0 8px}.chip{display:inline-block;background:var(--chip);border-radius:999px;padding:2px 9px;margin:0 6px 4px 0;font-size:12.5px}
.bad{color:var(--bad);font-weight:600}.note{color:var(--mut)}video{width:100%;max-height:480px;background:#000;border-radius:6px}
.imgs img{width:100%;border-radius:6px;border:1px solid var(--line);margin-bottom:8px}.ev{display:flex;gap:6px;flex-wrap:wrap}
.ev img{width:31%;min-width:90px}label{display:block;font-size:12.5px;color:var(--mut);margin-top:8px}
select,textarea{width:100%;font:inherit;padding:6px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--fg)}
textarea{min-height:60px}svg{width:100%;height:70px;background:var(--bg);border-radius:6px}pre{white-space:pre-wrap;font-size:12px}
"""

JS = r"""
const RUN = document.body.dataset.run, KEY = 'kassa_error_review_' + RUN;
let S = {}; try { S = JSON.parse(localStorage.getItem(KEY) || '{}') || {}; } catch (e) { S = {}; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(S)); } catch (e) {} prog(); }
function prog() { const c = document.querySelectorAll('.card'); let n = 0;
  c.forEach(x => { const v = S[x.dataset.vid] || {}; const d = !!v.cause; x.classList.toggle('done', d); n += d; });
  document.getElementById('prog').textContent = 'разобрано ' + n + ' / ' + c.length; }
document.querySelectorAll('[data-f]').forEach(inp => {
  const vid = inp.closest('.card').dataset.vid, f = inp.dataset.f; S[vid] = S[vid] || {};
  if (S[vid][f] !== undefined) inp.value = S[vid][f];
  inp.addEventListener('input', () => { S[vid][f] = inp.value; save(); });
  inp.addEventListener('change', () => { S[vid][f] = inp.value; save(); });
});
document.querySelectorAll('[data-seek]').forEach(b => b.addEventListener('click', () => {
  const v = b.closest('.card').querySelector('video'); v.currentTime = Math.max(0, +b.dataset.seek); v.play(); }));
function esc(v) { v = v === undefined || v === null ? '' : String(v); return /[",\n;]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }
document.getElementById('exp').onclick = () => {
  const cols = ['video_id', 'outcome', 'label_t', 'service_type', 'service_t', 'cause', 'true_type', 'notes'];
  const lines = [cols.join(',')];
  document.querySelectorAll('.card').forEach(c => { const v = S[c.dataset.vid] || {};
    lines.push([c.dataset.vid, c.dataset.outcome, c.dataset.lt, c.dataset.st, c.dataset.stt, v.cause, v.true_type, v.notes].map(esc).join(',')); });
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob(['﻿' + lines.join('\n') + '\n'], {type: 'text/csv'}));
  a.download = 'error_review_' + RUN + '.csv'; document.body.append(a); a.click(); a.remove(); };
prog();
"""


def sparkline(timeline, dur: float, labels: list[float], preds: list[float]) -> str:
    w, h = 300, 60
    pts = [(float(t), float(r)) for t, r in (timeline or []) if r is not None]
    x = lambda t: 4 + (w - 8) * min(max(t / max(dur, 1e-6), 0), 1)  # noqa: E731
    poly = " ".join(f"{x(t):.1f},{h - 4 - (h - 8) * min(max(r, 0), 1):.1f}" for t, r in pts)
    marks = "".join(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="0" y2="{h}" stroke="#d33" stroke-width="2"/>' for t in labels)
    marks += "".join(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="0" y2="{h}" stroke="#36f" stroke-dasharray="3,2"/>'
                     for t in preds)
    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none">{marks}'
            f'<polyline fill="none" stroke="currentColor" stroke-width="1.5" points="{poly}"/></svg>'
            '<div class="note">кривая риска сервиса; <span style="color:#d33">красное</span> — метка, '
            '<span style="color:#36f">синий пунктир</span> — таймкод(ы) сервиса</div>')


def save_evidence(frames, out: Path, stem: str) -> list[str]:
    names = []
    for i, fr in enumerate(frames or []):
        b64 = (fr or {}).get("image_base64") if isinstance(fr, dict) else None
        if not b64:
            continue
        try:
            data = base64.b64decode(b64.split(",", 1)[-1], validate=False)
        except (binascii.Error, ValueError):
            continue
        name = f"{stem}_e{i}.jpg"
        (out / name).write_bytes(data)
        names.append(name)
    return names


def sheet_safe(video: Path, out: Path, times: list[float], scale: float) -> str | None:
    try:
        _cs.make_sheet(str(video), str(out), ",".join(f"{t:g}" for t in times), scale)
        return out.name
    except (SystemExit, Exception):  # noqa: BLE001 — страница строится и без отдельного листа
        return None


def build(run_dir: Path, root: Path, types: dict[str, str]) -> tuple[Path, int, int]:
    rows = _er.per_video(run_dir, root)
    man = {m["video_id"]: m for m in _er.load_labels(root)[0].values()}
    errs = [x for x in rows if x["outcome"] in ("FN", "FP")]
    out = root / "cache/error_review" / run_dir.name
    img = out / "img"
    img.mkdir(parents=True, exist_ok=True)
    e = html.escape
    cards = []
    for x in errs:
        vid, m = x["video_id"], man[x["video_id"]]
        video = root / "data/raw" / m["file"]
        dur = float(m["duration_s"])
        full = run_dir / f"{vid}.json"
        j = json.loads(full.read_text(encoding="utf-8")) if full.exists() else {}
        viols = j.get("violations") or []
        labels = [float(t) for t in x["t_label"].split(";") if t]
        preds = [float(v["timecode_sec"]) for v in viols if v.get("timecode_sec") is not None]
        sheets = []
        if x["outcome"] == "FN":
            for t in labels:
                n = sheet_safe(video, img / f"{vid}_label_{t:g}.jpg", [t + d for d in range(-5, 7)], 0.6)
                sheets.append((f"вокруг метки t={t:g} с", n))
        for i, t in enumerate(preds[:2]):
            n = sheet_safe(video, img / f"{vid}_pred{i}.jpg", [t + d for d in range(-5, 7)], 0.6)
            sheets.append((f"вокруг таймкода сервиса t={t:g} с", n))
        ov = sheet_safe(video, img / f"{vid}_overview.jpg", [dur * (k + 0.5) / 12 for k in range(12)], 0.45)
        viol_html = []
        for i, v in enumerate(viols):
            ev = save_evidence(v.get("evidence_frames"), img, f"{vid}_v{i}")
            viol_html.append(
                f'<div><span class="chip bad">{e(str(v.get("type_id", "")))} · t={e(str(v.get("timecode_sec")))} с · '
                f'conf {e(str(v.get("confidence")))}</span><div>{e(str(v.get("type", "")))}</div>'
                f'<div class="note">{e(str(v.get("description", "")))}</div>'
                f'<div class="ev">{"".join(f"<a href=img/{n} target=_blank><img src=img/{n}></a>" for n in ev)}</div></div>')
        stages = (j.get("coverage") or {}).get("stages") or {}
        st_html = " ".join(f'<span class="chip{" bad" if str(s) in ("error", "skipped") else ""}">{e(k)}: {e(str(s))}</span>'
                           for k, s in stages.items())
        warn = "".join(f"<li>{e(str(w))}</li>" for w in (j.get("warnings") or []))
        causes = FN_CAUSES if x["outcome"] == "FN" else FP_CAUSES
        opts = "".join(f'<option value="{e(c)}">{e(c) or "— выберите —"}</option>' for c in causes)
        topts = '<option value="">—</option>' + "".join(f'<option value="{k}">{k} — {e(v)}</option>' for k, v in types.items())
        seek = "".join(f'<button data-seek="{t - 3:g}">▶ метка {t:g}</button> ' for t in labels)
        seek += "".join(f'<button data-seek="{t - 3:g}">▶ сервис {t:g}</button> ' for t in preds[:3])
        head = ("ПРОПУСК: нарушение по разметке, сервис — нет" if x["outcome"] == "FN"
                else "ЛОЖНАЯ ТРЕВОГА: по разметке чисто, сервис — нарушение")
        lt = x["t_label"] or "—"
        cards.append(f"""
<div class="card" data-vid="{vid}" data-outcome="{x['outcome']}" data-lt="{e(x['t_label'])}" data-st="{e(x['type_id'])}"
 data-stt="{e(str(x['timecode'] if x['timecode'] is not None else ''))}">
<h2>{vid} · {head}</h2>
<div><span class="chip">разметка: {e(x['label'])} @ {e(lt)} с</span><span class="chip">сервис: {e(x['verdict'])}
 {e(x['type_id'])} @ {e(str(x['timecode']))} с</span><span class="chip">risk_max {e(str(x['risk_max']))}</span>
 <span class="chip">нарушений в ответе {x['n_violations']}</span></div>
<div class="row">
 <div><video controls preload="none" src="../../../data/raw/{e(m['file'])}"></video><div>{seek}</div>
  {sparkline(j.get('timeline'), dur, labels, preds)}<div>{st_html}</div>{f'<ul class=note>{warn}</ul>' if warn else ''}</div>
 <div class="imgs">{''.join(f'<div class=note>{e(c)}</div><a href=img/{n} target=_blank><img src=img/{n}></a>' for c, n in sheets if n)}
  {f'<div class=note>обзор ролика</div><a href=img/{ov} target=_blank><img src=img/{ov}></a>' if ov else ''}</div>
 <div><label>Причина ошибки<select data-f="cause">{opts}</select></label>
  <label>{'Какой тип нарушения на самом деле' if x['outcome'] == 'FN' else 'Если сервис прав — какой тип'}
  <select data-f="true_type">{topts}</select></label>
  <label>Заметки<textarea data-f="notes"></textarea></label>
  <div style="margin-top:10px">{''.join(viol_html) or '<span class=note>сервис нарушений не вернул</span>'}</div></div>
</div></div>""")
    page = (f"<!doctype html><html lang='ru'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>Разбор ошибок</title><style>{CSS}</style></head>"
            f"<body data-run='{e(run_dir.name)}'><header><h1>Разбор ошибок прогона {e(run_dir.name)}</h1>"
            f"<span id='prog' class='chip'></span><button class='pri' id='exp'>error_review.csv</button>"
            f"<span class='note'>локальный файл — не публиковать</span></header><main>"
            f"<p class='note'>Пропусков {sum(x['outcome'] == 'FN' for x in errs)}, ложных тревог "
            f"{sum(x['outcome'] == 'FP' for x in errs)}. Для каждого видео выберите причину (и тип для пропусков), "
            f"правки сохраняются в браузере; в конце — «error_review.csv».</p>{''.join(cards)}</main>"
            f"<script>{JS}</script></body></html>")
    (out / "index.html").write_text(page, encoding="utf-8")
    return out / "index.html", sum(x["outcome"] == "FN" for x in errs), sum(x["outcome"] == "FP" for x in errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--catalog", type=Path, default=None)
    a = ap.parse_args()
    root = a.root.resolve()
    types, src = _tc.load_types(a.catalog, root)
    page, fn, fp = build(_er.run_paths(a.run)[0], root, types)
    print(f"страница: {page} | пропусков {fn}, ложных тревог {fp} | типы ← {src}")


if __name__ == "__main__":
    main()
