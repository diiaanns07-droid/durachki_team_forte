"""Нейтральный VLM-опросник наблюдаемых признаков (yes/no/unclear) — для правил и для проверки типов человеком.

В отличие от vlm_types.py, модели НЕ сообщают, что в окне отмечено нарушение (нет якоря на t и на «надо выбрать
тип»). Окна: t-5..t+5 вокруг каждой метки нарушения и, для сравнения, окна чистых видео (1/3 и 2/3 длительности).
Разница частот «yes» (нарушения vs чистые) показывает, какие сигналы реально отличают нарушение.
Пишет cache/types_review/vlm_probe.jsonl; в stdout — только агрегаты.

usage: python -I -X utf8 tools/dataprep/vlm_probe.py --root . [--clean 8]
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location("vlm_types", Path(__file__).with_name("vlm_types.py"))
_vt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vt)

# признак → (вопрос, сигнал контракта / тип)
ATTRS = {
    "cashier_at_desk": ("кассир находится на своём рабочем месте за кассовым столом", "cashier_in_workplace"),
    "cashier_absent": ("кассира нет в кадре или он вышел из кассового помещения", "V02"),
    "outsider_behind_desk": ("посторонний (не кассир) находится за кассовым столом или в зоне кассира", "V01"),
    "person_enters": ("кто-то входит в помещение через дверь", "V04"),
    "door_open": ("дверь помещения открыта", "door_open"),
    "safe_open": ("дверца сейфа открыта", "V03"),
    "phone_in_cashier_hand": ("у кассира в руке мобильный телефон или он говорит по телефону", "V05"),
    "food_or_drink": ("у кассира еда или напиток, он ест или пьёт", "V06"),
    "cash_visible": ("видны наличные деньги", "cash_visible"),
    "cash_counting": ("кто-то пересчитывает деньги (руками или счётной машинкой)", "cash_counting"),
    "cash_hidden_from_camera": ("деньги или руки с деньгами закрыты от камеры телом, столом и т.п.", "V08"),
    "cash_handover": ("деньги передаются из рук в руки или через стол", "handover_motion"),
    "cash_bag": ("видна инкассаторская сумка или мешок", "V07"),
    "collector_present": ("в кадре инкассатор (сотрудник инкассации с сумкой, в форме или жилете)", "collector_present"),
}

PROMPT = """Кадры подряд с потолочной камеры кассового помещения банка, время подписано слева сверху.
{roles}
Для каждого признака ответь, виден ли он хотя бы на одном кадре: "yes", "no" или "unclear", и первую секунду,
когда он виден (или null). Признаки:
{attrs}
Ответь ОДНИМ JSON без пояснений, ключи — как в списке: {{"cashier_at_desk": ["yes", 12], "safe_open": ["no", null], ...}}"""


def norm_ans(v) -> tuple[str, float | None]:
    if isinstance(v, (list, tuple)) and v:
        a, t = str(v[0]).lower(), (v[1] if len(v) > 1 else None)
    elif isinstance(v, dict):
        a, t = str(v.get("answer", v.get("a", ""))).lower(), v.get("t")
    else:
        a, t = str(v or "").lower(), None
    a = "yes" if a.startswith(("yes", "да")) else "no" if a.startswith(("no", "нет")) else "unclear"
    try:
        t = float(t) if t is not None else None
    except (TypeError, ValueError):
        t = None
    return a, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--hf-cache", default="weights/hf/hub")
    ap.add_argument("--clean", type=int, default=8)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--vram-margin", type=float, default=2.0, help="ГиБ VRAM вне лимита torch: рабочий стол + CUDA-контекст")
    a = ap.parse_args()
    root = Path(a.root)
    out = root / "cache/types_review/vlm_probe.jsonl"
    done = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["key"]] = r
    rows = list(csv.DictReader(open(root / "data/labels.csv", encoding="utf-8")))
    man = {r["video_id"]: r for r in csv.DictReader(open(root / "data/manifest.csv", encoding="utf-8"))}
    dur = {k: float(v["duration_s"]) for k, v in man.items()}

    jobs = [(r["video_id"], float(r["t_sec"]), "violation") for r in rows if r["label"] == "violation"]
    clean = sorted([r for r in rows if r["label"] == "no_violation" and r["group"] == r["video_id"]],
                   key=lambda r: dur[r["video_id"]])
    k = min(a.clean, len(clean))
    for r in [clean[round(i * (len(clean) - 1) / max(k - 1, 1))] for i in range(k)] if k else []:
        for frac in (1 / 3, 2 / 3):
            jobs.append((r["video_id"], round(dur[r["video_id"]] * frac), "no_violation"))

    t0 = time.time()
    free0, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(max(0.3, min(0.95, (free0 - a.vram_margin * 2**30) / total)))
    vlm = _vt.VLM(a.model, str(root / a.hf_cache))
    torch.cuda.reset_peak_memory_stats()
    attrs_txt = "\n".join(f"- {k2}: {q}" for k2, (q, _) in ATTRS.items())
    prompt = PROMPT.format(roles=_vt.ROLE_TXT, attrs=attrs_txt)
    f = open(out, "a", encoding="utf-8")
    for vid, t, label in jobs:
        key = f"probe|{vid}|{t:g}"
        if key in done:
            continue
        ts = [x for x in (t + d for d in range(-5, 6)) if 0 <= x <= dur[vid]]
        fr = _vt.grab_frames(root / "data/raw" / man[vid]["file"], ts)
        tiles = [_vt.labeled(fr[x][1], fr[x][0], a.scale) for x in sorted(fr)]
        txt, n_in, dt, shrink = vlm.ask_safe([("Кадры:", tiles)], prompt, max_new=500)
        parsed, ok = _vt.parse_json(txt)
        ans = {k2: norm_ans(parsed.get(k2)) for k2 in ATTRS}
        rec = dict(key=key, kind="probe", video_id=vid, t_center=t, label=label, n_tokens=n_in, latency_s=round(dt, 1),
                   shrink=shrink, parse_ok=ok, answers={k2: list(v) for k2, v in ans.items()}, raw=txt)
        done[key] = rec
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        print(f"[{len(done)}/{len(jobs)}] {key} {label} tokens={n_in} {dt:.1f}s parse_ok={ok}", flush=True)
    f.close()

    recs = list(done.values())
    pos = [r for r in recs if r["label"] == "violation"]
    neg = [r for r in recs if r["label"] == "no_violation"]
    print(f"probe windows: violation={len(pos)} clean={len(neg)} parse_ok={sum(r['parse_ok'] for r in recs)}/{len(recs)}")
    print(f"{'attribute':26s} {'viol yes':>9s} {'clean yes':>9s}  (unclear viol/clean)")
    for k2 in ATTRS:
        py = sum(r["answers"][k2][0] == "yes" for r in pos) / max(len(pos), 1)
        ny = sum(r["answers"][k2][0] == "yes" for r in neg) / max(len(neg), 1)
        pu = sum(r["answers"][k2][0] == "unclear" for r in pos)
        nu = sum(r["answers"][k2][0] == "unclear" for r in neg)
        print(f"{k2:26s} {py:9.0%} {ny:9.0%}  ({pu}/{nu})")
    print(f"total {time.time() - t0:.0f}s, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
