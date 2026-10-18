"""Локальная VLM-разметка ТИПОВ нарушений (Qwen3-VL-4B-Instruct, nf4) — ПРЕДЛОЖЕНИЯ для подтверждения человеком.

Всё локально: кадры, ответы VLM и контакт-листы пишутся только на диск (cache/types_review/), в stdout —
только технический статус и агрегаты (без описаний сцен). Подтверждение — tools/dataprep/types_review.py → HTML.

Проходы:
  win  — 4 обзорных кадра ролика + кадры t-5..t+5 (шаг 1 с) вокруг размеченного таймкода;
  wide — если confidence < --doubt или тип NEW/не распознан: t-10..t+10 (шаг 2 с);
  pair — для md5-дубликатов с разными таймкодами (Х-4≡Х-47): одно событие или два;
  clean — 6–8 чистых видео: как выглядит КОРРЕКТНАЯ процедура и что может дать ложное срабатывание.

usage: python -I -X utf8 tools/dataprep/vlm_types.py --root . [--clean 8] [--doubt 0.6] [--limit N]
Повторный запуск дописывает только недостающие записи (resume по vlm_raw.jsonl).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from pathlib import Path

import cv2
import torch
from PIL import Image, ImageDraw, ImageFont

CATALOG = {
    "V01": "посторонний в кассовом помещении / за столом кассира",
    "V02": "кассир покинул рабочее место, ценности (деньги, сейф, компьютер) без присмотра",
    "V03": "сейф открыт без присмотра или при посторонних",
    "V04": "дверь кассы открыта / вход постороннего",
    "V05": "мобильный телефон в руках у кассира",
    "V06": "еда или напитки на рабочем месте кассира",
    "V07": "нарушение приёма-передачи инкассаторской сумки",
    "V08": "пересчёт или передача денег вне обзора камеры (руки/деньги закрыты телом, под столом и т.п.)",
}
CATALOG_TXT = "\n".join(f"{k} — {v};" for k, v in CATALOG.items()) + "\nNEW — другое нарушение (опиши)."
ROLE_TXT = ("Кассир — сотрудник банка за кассовым столом (счётная машинка, ноутбук, сейф рядом). Остальные люди — "
            "клиенты, инкассаторы или посторонние. Лица не описывай, людей не идентифицируй, только роли и действия. "
            "Описывай только то, что видно на кадрах, без домыслов.")

PROMPT_VIOL = """Ты — аудитор кассовой дисциплины банка. Кадры с потолочной камеры кассового помещения.
Сначала {n_ov} обзорных кадров всего ролика, затем {n_win} кадров подряд вокруг момента t={t} с, который проверяющий \
отметил как нарушение. На каждом кадре слева сверху подписано время в секундах.
{roles}
Определи тип нарушения в момент t={t} с по списку:
{catalog}
Ответь ОДНИМ JSON без пояснений и без markdown:
{{"type_id": "V01..V08 или NEW", "alt_type_id": "второй по вероятности тип или пусто", \
"new_type": "если NEW — короткое название и суть, иначе пусто", "confidence": число 0..1, \
"event_time_sec": число — секунда, где ключевое действие видно лучше всего, \
"onset_or_peak": "onset — если в t={t} действие начинается; peak — если в t={t} оно в разгаре/наиболее явно; \
before — если t={t} раньше действия; after — если позже; unclear", \
"what_is_visible": "1–2 предложения: что происходит в t={t}", \
"actors": "кто в кадре: роли и количество", "objects": "значимые предметы: деньги, сейф, сумка, телефон, дверь и т.п."}}"""

PROMPT_PAIR = """Ты — аудитор кассовой дисциплины банка. Кадры с потолочной камеры, время подписано на кадрах.
В этом ролике проверяющий отметил нарушение в двух моментах: t={t1} с и t={t2} с.
{roles}
Типы нарушений:
{catalog}
Ответь ОДНИМ JSON без пояснений: {{"same_event": true/false — это одно и то же действие или два разных события, \
"type_t1": "тип в t={t1}", "type_t2": "тип в t={t2}", "confidence": число 0..1, \
"event_t1": "1 предложение: что происходит в t={t1}", "event_t2": "1 предложение: что происходит в t={t2}"}}"""

PROMPT_CLEAN = """Ты — аудитор кассовой дисциплины банка. Это ролик БЕЗ нарушений — эталон корректной работы кассы.
{n} кадров по порядку, время подписано слева сверху. {roles}
Опиши корректную процедуру. Ответь ОДНИМ JSON без пояснений:
{{"procedure_steps": ["t≈… с — шаг", "..."], "actors": "роли и количество людей", \
"collectors_present": true/false, "safe_used": "да/нет/неясно и как", \
"cash_handling": "как обращаются с деньгами: счётная машинка, стол, передача клиенту", \
"door_events": "входы/выходы, дверь", \
"fp_risks": ["что могло бы ложно сработать как нарушение V01–V08 и почему здесь это норма"], \
"suspected_violation": "пусто, если нарушений не видно; иначе тип V01–V08 и секунда"}}"""


# ----------------------------------------------------------------------------- кадры
def grab_frames(path: Path, times: list[float]) -> dict[float, tuple[float, object]]:
    """{t: (pts, bgr)} — ближайший по PTS кадр (последовательное декодирование: видео VFR)."""
    targets = sorted({round(t, 3) for t in times})
    best = {t: (math.inf, None, None) for t in targets}
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        for t in targets:
            d = abs(pts - t)
            if d < best[t][0]:
                best[t] = (d, pts, fr)
    cap.release()
    return {t: (b[1], b[2]) for t, b in best.items() if b[2] is not None}


_FONT = None


def font(size=22):
    global _FONT
    if _FONT is None or _FONT.size != size:
        try:
            _FONT = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", size)
        except OSError:
            _FONT = ImageFont.load_default()
    return _FONT


def labeled(bgr, pts: float, scale: float, mark=False) -> Image.Image:
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    f = font(max(14, int(30 * scale)))
    txt = f"t={pts:.0f}s"
    w = d.textlength(txt, font=f)
    d.rectangle([0, 0, w + 10, f.size + 8], fill=(0, 0, 0))
    d.text((5, 3), txt, fill=(255, 255, 0), font=f)
    if mark:
        d.rectangle([0, 0, img.width - 1, img.height - 1], outline=(255, 0, 0), width=4)
    return img


def sheet(tiles: list[Image.Image], out: Path, cols=4, rows=3):
    tw, th = tiles[0].size
    s = Image.new("RGB", (cols * tw + (cols - 1) * 4, rows * th + (rows - 1) * 4), (40, 40, 40))
    for i, t in enumerate(tiles[: cols * rows]):
        s.paste(t, ((i % cols) * (tw + 4), (i // cols) * (th + 4)))
    out.parent.mkdir(parents=True, exist_ok=True)
    s.save(out, quality=85)


# ----------------------------------------------------------------------------- VLM
class VLM:
    def __init__(self, model_id: str, cache: str):
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        hf = dict(cache_dir=cache, local_files_only=True)
        self.proc = AutoProcessor.from_pretrained(model_id, **hf)
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["visual", "lm_head"])
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda:0", quantization_config=q, **hf).eval()

    @torch.inference_mode()
    def ask(self, content: list[dict], max_new=450, max_time=None) -> tuple[str, int, float]:
        inp = self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                            add_generation_prompt=True, return_dict=True,
                                            return_tensors="pt").to(self.model.device)
        t0 = time.time()
        out = self.model.generate(**inp, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.05,
                                  max_time=max_time)
        n_in = inp["input_ids"].shape[1]
        return self.proc.decode(out[0, n_in:], skip_special_tokens=True).strip(), n_in, time.time() - t0

    def ask_safe(self, groups: list[tuple[str, list[Image.Image]]], prompt: str, max_new=450, max_time=90.0):
        """groups: [(подпись, [кадры])]. При OOM или упоре в max_time (спилл WDDM в RAM) — повтор с уменьшенными
        кадрами; фактический масштаб возвращается как shrink."""
        last = ("", 0, 0.0, 0.0)
        for shrink in (1.0, 0.75, 0.55):
            content = []
            for caption, imgs in groups:
                content.append({"type": "text", "text": caption})
                for im in imgs:
                    if shrink < 1:
                        im = im.resize((int(im.width * shrink), int(im.height * shrink)), Image.LANCZOS)
                    content.append({"type": "image", "image": im})
            content.append({"type": "text", "text": prompt})
            try:
                txt, n_in, dt = self.ask(content, max_new, max_time)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            last = (txt, n_in, dt, shrink)
            if max_time is None or dt < max_time * 0.97:
                return last
            torch.cuda.empty_cache()
        return last


def parse_json(txt: str) -> tuple[dict, bool]:
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        s = m.group(0)
        for cand in (s, re.sub(r",\s*([}\]])", r"\1", s)):
            try:
                return json.loads(cand), True
            except json.JSONDecodeError:
                pass
    d = {}
    for k in ("type_id", "alt_type_id", "onset_or_peak"):
        mm = re.search(rf'"{k}"\s*:\s*"([^"]*)"', txt)
        if mm:
            d[k] = mm.group(1)
    mm = re.search(r'"confidence"\s*:\s*([0-9.]+)', txt)
    if mm:
        d["confidence"] = mm.group(1)
    return d, False


def norm_type(v) -> str:
    s = str(v or "").upper()
    m = re.search(r"V0?([1-8])\b", s)
    if m:
        return f"V0{m.group(1)}"
    return "NEW" if "NEW" in s or "НОВ" in s else "UNK"


def norm_oop(v) -> str:
    s = str(v or "").strip().lower()
    for k in ("onset", "peak", "before", "after", "unclear"):
        if s.startswith(k):
            return k
    return "other"


def fnum(v, default=float("nan")) -> float:
    try:
        return float(str(v).replace(",", "."))
    except (TypeError, ValueError):
        return default


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--hf-cache", default="weights/hf/hub")
    ap.add_argument("--doubt", type=float, default=0.6)
    ap.add_argument("--clean", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="только первые N нарушений (отладка)")
    ap.add_argument("--win-scale", type=float, default=0.6, help="масштаб кадров окна для VLM (464x848 → ×)")
    ap.add_argument("--ov-frames", type=int, default=4, help="обзорных кадров ролика в запросе VLM")
    ap.add_argument("--ov-scale", type=float, default=0.45)
    ap.add_argument("--vram-margin", type=float, default=2.0, help="ГиБ VRAM вне лимита torch: рабочий стол + CUDA-контекст")
    a = ap.parse_args()
    root = Path(a.root)
    out_dir = root / "cache/types_review"
    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "vlm_raw.jsonl"
    done = {}
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["key"]] = r
    rows = list(csv.DictReader(open(root / "data/labels.csv", encoding="utf-8")))
    man = {r["video_id"]: r for r in csv.DictReader(open(root / "data/manifest.csv", encoding="utf-8"))}
    dur = {k: float(v["duration_s"]) for k, v in man.items()}
    raw_f = open(raw_path, "a", encoding="utf-8")

    t_start = time.time()
    free0, total = torch.cuda.mem_get_info()
    # потолок torch = свободная VRAM − запас: лучше OOM (→ ask_safe уменьшит кадры), чем спилл WDDM в системную RAM
    frac = max(0.3, min(0.95, (free0 - a.vram_margin * 2**30) / total))
    torch.cuda.set_per_process_memory_fraction(frac)
    vlm = VLM(a.model, str(root / a.hf_cache))
    torch.cuda.reset_peak_memory_stats()
    print(f"VLM loaded: allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB, torch cap {frac * total / 2**30:.2f} GiB",
          flush=True)

    def save(rec):
        done[rec["key"]] = rec
        raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        raw_f.flush()
        print(f"[{len(done)}] {rec['key']} tokens={rec['n_tokens']} {rec['latency_s']}s shrink={rec['shrink']} "
              f"parse_ok={rec['parse_ok']}", flush=True)

    def frames_for(vid, times):  # одно декодирование на все нужные моменты
        return grab_frames(root / "data/raw" / man[vid]["file"], times)

    def ov_times(vid, n):
        return [dur[vid] * (i + 0.5) / n for i in range(n)]

    def tiles(fr, times, scale, marks=()):
        keys = sorted({round(x, 3) for x in times} & set(fr))
        return [labeled(fr[x][1], fr[x][0], scale, mark=any(abs(x - m) < 0.01 for m in marks)) for x in keys]

    # --- нарушения: win (+ wide при сомнении)
    viol = [r for r in rows if r["label"] == "violation"]
    if a.limit:
        viol = viol[: a.limit]
    for r in viol:
        vid, t = r["video_id"], float(r["t_sec"])
        for pss, offs in (("win", [t + d for d in range(-5, 6)]), ("wide", [t + d for d in range(-10, 11, 2)])):
            key = f"violation|{vid}|{t:g}|{pss}"
            if key in done:
                continue
            if pss == "wide":
                w = done.get(f"violation|{vid}|{t:g}|win", {}).get("parsed", {})
                if fnum(w.get("confidence"), 0) >= a.doubt and norm_type(w.get("type_id")) not in ("NEW", "UNK"):
                    continue
            ts = [x for x in offs if 0 <= x <= dur[vid]]
            o_vlm, o_sheet = ov_times(vid, a.ov_frames), ov_times(vid, 12)
            fr = frames_for(vid, ts + o_vlm + (o_sheet if pss == "win" else []))
            sheet_name = f"{vid}_t{t:g}_{pss}.jpg"
            sheet(tiles(fr, ts, 0.75, [t]), img_dir / sheet_name)  # для человека — крупнее
            if pss == "win":
                sheet(tiles(fr, o_sheet, 0.5), img_dir / f"{vid}_overview.jpg")
            win, ov = tiles(fr, ts, a.win_scale, [t]), tiles(fr, o_vlm, a.ov_scale)
            prompt = PROMPT_VIOL.format(n_ov=len(ov), n_win=len(win), t=f"{t:g}", roles=ROLE_TXT, catalog=CATALOG_TXT)
            txt, n_in, dt, shrink = vlm.ask_safe([("Обзорные кадры всего ролика:", ov),
                                                  (f"Кадры вокруг t={t:g} с (шаг {1 if pss == 'win' else 2} с):", win)],
                                                 prompt)
            parsed, ok = parse_json(txt)
            save(dict(key=key, kind="violation", video_id=vid, t_sec=t, pass_=pss, frames=[round(x, 2) for x in ts],
                      sheet=f"img/{sheet_name}", overview=f"img/{vid}_overview.jpg", n_tokens=n_in,
                      latency_s=round(dt, 1), shrink=shrink, raw=txt, parsed=parsed, parse_ok=ok))

    # --- пары md5-дубликатов с разными таймкодами
    groups: dict[str, list[dict]] = {}
    for r in rows:
        if r["label"] == "violation":
            groups.setdefault(r["md5"], []).append(r)
    for g in groups.values():
        ts = sorted({float(r["t_sec"]) for r in g})
        if len(g) < 2 or len(ts) < 2:
            continue
        vids = sorted({r["video_id"] for r in g}, key=lambda v: int(v.split("-")[1]))
        key = f"pair|{'+'.join(vids)}|{ts[0]:g}+{ts[-1]:g}"
        if key in done:
            continue
        vid = vids[0]
        span = [x for x in range(int(ts[0]) - 5, int(ts[-1]) + 6, 2) if 0 <= x <= dur[vid]]
        span = sorted(set(span) | {int(ts[0]), int(ts[-1])})
        fr = frames_for(vid, span)
        human = tiles(fr, span, 0.6, ts)
        sheet_name = f"{'+'.join(vids)}_pair.jpg"
        sheet(human, img_dir / sheet_name, cols=6, rows=math.ceil(len(human) / 6))
        txt, n_in, dt, shrink = vlm.ask_safe([(f"Кадры {span[0]}–{span[-1]} с:", tiles(fr, span, a.ov_scale + 0.05, ts))],
                                             PROMPT_PAIR.format(t1=f"{ts[0]:g}", t2=f"{ts[-1]:g}", roles=ROLE_TXT,
                                                                catalog=CATALOG_TXT))
        parsed, ok = parse_json(txt)
        save(dict(key=key, kind="pair", video_ids=vids, t=ts, sheet=f"img/{sheet_name}", n_tokens=n_in,
                  latency_s=round(dt, 1), shrink=shrink, raw=txt, parsed=parsed, parse_ok=ok))

    # --- чистые: 6–8 уникальных, равномерно по длительности
    clean = sorted([r for r in rows if r["label"] == "no_violation" and r["group"] == r["video_id"]],
                   key=lambda r: dur[r["video_id"]])
    k = min(a.clean, len(clean))
    pick = [clean[round(i * (len(clean) - 1) / max(k - 1, 1))] for i in range(k)] if k else []
    for r in pick:
        vid = r["video_id"]
        key = f"clean|{vid}"
        if key in done:
            continue
        o12 = ov_times(vid, 12)
        fr = frames_for(vid, o12)
        ov = tiles(fr, o12, 0.5)
        sheet(ov, img_dir / f"{vid}_overview.jpg")
        txt, n_in, dt, shrink = vlm.ask_safe([("Кадры ролика по порядку:", ov)],
                                             PROMPT_CLEAN.format(n=len(ov), roles=ROLE_TXT), max_new=600)
        parsed, ok = parse_json(txt)
        save(dict(key=key, kind="clean", video_id=vid, overview=f"img/{vid}_overview.jpg", n_tokens=n_in,
                  latency_s=round(dt, 1), shrink=shrink, raw=txt, parsed=parsed, parse_ok=ok))
    raw_f.close()

    # --- агрегаты (без описаний сцен)
    recs = list(done.values())
    v = [x for x in recs if x["kind"] == "violation"]
    final = {}
    for x in v:
        kk = (x["video_id"], x["t_sec"])
        c = fnum(x["parsed"].get("confidence"), 0)
        if kk not in final or c > fnum(final[kk]["parsed"].get("confidence"), 0):
            final[kk] = x
    types = [norm_type(x["parsed"].get("type_id")) for x in final.values()]
    confs = [fnum(x["parsed"].get("confidence")) for x in final.values()]
    oop = [norm_oop(x["parsed"].get("onset_or_peak")) for x in final.values()]
    offs = [fnum(x["parsed"].get("event_time_sec")) - x["t_sec"] for x in final.values()]
    offs = [o for o in offs if not math.isnan(o)]
    lat = [x["latency_s"] for x in recs]
    print(f"records={len(recs)} (violation passes={len(v)}, wide={sum(x['pass_'] == 'wide' for x in v)}) "
          f"parse_ok={sum(x['parse_ok'] for x in recs)}/{len(recs)} oom_shrunk={sum(x['shrink'] < 1 for x in recs)}")
    print("VLM type proposals (final per labeled event):",
          {t: types.count(t) for t in sorted(set(types))})
    if confs:
        print(f"confidence mean {statistics.mean([c for c in confs if not math.isnan(c)]):.2f}, "
              f"<{a.doubt}: {sum(c < a.doubt for c in confs if not math.isnan(c))}")
    print("onset_or_peak:", {o: oop.count(o) for o in sorted(set(oop))})
    if offs:
        print(f"event_time - t_label: median {statistics.median(offs):+.1f}s, |.|<=1s {sum(abs(o) <= 1 for o in offs)}"
              f"/{len(offs)}")
    for x in recs:
        if x["kind"] == "pair":
            p = x["parsed"]
            print(f"pair {'+'.join(x['video_ids'])} t={x['t']}: same_event={p.get('same_event')} "
                  f"type_t1={norm_type(p.get('type_t1'))} type_t2={norm_type(p.get('type_t2'))} conf={p.get('confidence')}")
    cl = [x for x in recs if x["kind"] == "clean"]
    if cl:
        sus = sum(bool(str(x["parsed"].get("suspected_violation", "")).strip()) for x in cl)
        print(f"clean videos described: {len(cl)}, VLM suspected a violation in {sus} (false-positive signal)")
    print(f"latency per call: median {statistics.median(lat):.1f}s max {max(lat):.1f}s | total {time.time() - t_start:.0f}s "
          f"| peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
