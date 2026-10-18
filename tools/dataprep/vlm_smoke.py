"""Smoke-тест локальной VLM (Qwen3-VL-4B-Instruct, 4-bit nf4 через bitsandbytes) на RTX 4060 8 ГБ.

Работает на СИНТЕТИЧЕСКИХ кадрах реального разрешения датасета (464x848), чтобы замерить
VRAM/латентность без реальных сцен. Печатает только технический статус.

usage: python -I tools/dataprep/vlm_smoke.py [--model Qwen/Qwen3-VL-4B-Instruct] [--hf-cache weights/hf/hub] [--frames 1,3]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


def synth_frame(i: int, w: int = 464, h: int = 848) -> Image.Image:
    rng = np.random.default_rng(i)
    img = Image.fromarray((rng.random((h, w, 3)) * 40 + 150).astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.rectangle([60, 380, 400, 520], fill=(150, 110, 70))          # «стол»
    d.ellipse([180 + 10 * i, 200, 280 + 10 * i, 300], fill=(220, 30, 30))  # красный круг
    d.rectangle([330, 250, 420, 360], fill=(120, 120, 130))        # серый ящик
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--hf-cache", default="weights/hf/hub", help="HF hub-кэш (раскладка deploy/weights.py)")
    ap.add_argument("--frames", default="1,3")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--bf16", action="store_true", help="без квантизации (на 8 ГБ, скорее всего, не влезет)")
    a = ap.parse_args()

    free0, total = torch.cuda.mem_get_info()
    print(f"GPU {torch.cuda.get_device_name(0)}: free {free0 / 2**30:.2f} / {total / 2**30:.2f} GiB before load")
    t0 = time.time()
    hf = dict(cache_dir=a.hf_cache, local_files_only=True)  # без сети
    proc = AutoProcessor.from_pretrained(a.model, **hf)
    kw = dict(dtype=torch.bfloat16, device_map="cuda:0", **hf)
    if not a.bf16:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["visual", "lm_head"])
    model = AutoModelForImageTextToText.from_pretrained(a.model, **kw).eval()
    torch.cuda.synchronize()
    print(f"loaded ({'bf16' if a.bf16 else 'nf4 LM + bf16 vision'}) in {time.time() - t0:.1f}s, "
          f"allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB, free {torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB")

    for n in [int(x) for x in a.frames.split(",")]:
        content = [{"type": "image", "image": synth_frame(i)} for i in range(n)]
        content.append({"type": "text", "text": "How many red circles are in each image? Answer briefly."})
        msgs = [{"role": "user", "content": content}]
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                       return_tensors="pt").to(model.device)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t1 = time.time()
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=a.max_new, do_sample=False)
        torch.cuda.synchronize()
        dt = time.time() - t1
        new = out[0, inp["input_ids"].shape[1]:]
        txt = proc.decode(new, skip_special_tokens=True).strip().replace("\n", " ")
        print(f"frames={n} input_tokens={inp['input_ids'].shape[1]} new_tokens={len(new)} latency={dt:.1f}s "
              f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB | synthetic answer: {txt[:80]!r}")


if __name__ == "__main__":
    main()
