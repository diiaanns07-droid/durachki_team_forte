"""ОПЦИОНАЛЬНОЕ LoRA-дообучение языковой части общей Qwen3-VL-4B (вне критического пути, НЕ запускалось).

Цель — закрепить формат ответа ([Nk]/[Vk]), честные отказы и «без пункта» (датасет: make_dataset.py).
Обучаются только q/k/v/o-проекции языковой модели (визуальная башня не трогается), потери — только на
ответе. Без внешних тренеров (TRL не нужен), PEFT — отдельной зависимостью (requirements-lora.txt),
основной lock L1 не меняется. Применять адаптер в A11 — только после замера eval_grounding до/после.

    python -m copilot.train.lora --data outputs/copilot/lora_data --dry-run      # проверка данных, без torch
    python -m copilot.train.lora --data outputs/copilot/lora_data --base weights/hf/hub/<Qwen3-VL-4B snapshot> \
        --out outputs/copilot/lora_adapter --load-4bit --epochs 1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            msgs = r.get("messages")
            if not isinstance(msgs, list) or not msgs or msgs[0].get("role") != "system":
                raise ValueError(f"{path}:{i}: ожидается messages с первым system")
            if any(m.get("role") not in {"system", "user", "assistant"} for m in msgs):
                raise ValueError(f"{path}:{i}: недопустимая роль")
            if not str(r.get("completion") or "").strip():
                raise ValueError(f"{path}:{i}: пустой completion")
            rows.append(r)
    return rows


def plan(data_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    train = load_rows(data_dir / "train.jsonl")
    val_path = data_dir / "val.jsonl"
    val = load_rows(val_path) if val_path.is_file() else []
    chars = [sum(len(m["content"]) for m in r["messages"]) + len(r["completion"]) for r in train]
    steps = math.ceil(len(train) * args.epochs / max(1, args.batch * args.grad_accum))
    return {"train": len(train), "val": len(val), "max_chars": max(chars) if chars else 0,
            "mean_chars": round(sum(chars) / len(chars)) if chars else 0, "optimizer_steps": steps,
            "lora": {"r": args.r, "alpha": args.alpha, "dropout": args.dropout,
                     "target": ["q_proj", "k_proj", "v_proj", "o_proj"]},
            "load_4bit": args.load_4bit, "max_len": args.max_len, "lr": args.lr}


def _encode(tok: Any, row: dict[str, Any], max_len: int) -> Optional[dict[str, Any]]:
    """input_ids/labels: потери только на ответе ассистента."""
    prompt = tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
    full = prompt + row["completion"] + (tok.eos_token or "")
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
    if len(f_ids) <= len(p_ids):
        return None  # ответ не влез в max_len
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return {"input_ids": f_ids, "labels": labels[:len(f_ids)]}


def train(args: argparse.Namespace) -> int:  # pragma: no cover - требует GPU/весов, в CI не запускается
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, AutoTokenizer

    data = load_rows(Path(args.data) / "train.jsonl")
    try:
        tok = AutoProcessor.from_pretrained(args.base, local_files_only=True).tokenizer
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    quant = None
    if args.load_4bit:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = None
    import transformers

    for cls_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(args.base, local_files_only=True, quantization_config=quant,
                                        torch_dtype=torch.bfloat16, device_map="auto")
            break
        except Exception as exc:  # другой класс модели в этой версии transformers
            print(f"{cls_name}: {type(exc).__name__}", file=sys.stderr)
    if model is None:
        raise SystemExit("не удалось загрузить базовую модель локально")
    if quant is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none", task_type="CAUSAL_LM",
        # только языковая модель: в визуальной башне Qwen-VL проекции называются иначе (qkv/proj)
        target_modules=r".*(language_model|model\.layers).*\.(q_proj|k_proj|v_proj|o_proj)$"))
    model.print_trainable_parameters()
    enc = [e for e in (_encode(tok, r, args.max_len) for r in data) if e is not None]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    step, t0 = 0, time.time()
    dev = next(model.parameters()).device
    for epoch in range(args.epochs):
        for i, ex in enumerate(enc):
            ids = torch.tensor([ex["input_ids"]], device=dev)
            labels = torch.tensor([ex["labels"]], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                loss = model(input_ids=ids, labels=labels).loss / args.grad_accum
            loss.backward()
            if (i + 1) % args.grad_accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    print(json.dumps({"epoch": epoch, "step": step, "loss": round(loss.item() * args.grad_accum, 4),
                                      "elapsed_s": round(time.time() - t0)}), flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    print(json.dumps({"saved": args.out, "steps": step, "examples": len(enc)}))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Опциональный LoRA для Sentinel Copilot (вне критического пути)")
    ap.add_argument("--data", default="outputs/copilot/lora_data")
    ap.add_argument("--base", default="Qwen/Qwen3-VL-4B-Instruct", help="HF id в локальном кэше или папка снапшота")
    ap.add_argument("--out", default="outputs/copilot/lora_adapter")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", dest="grad_accum", type=int, default=8)
    ap.add_argument("--max-len", dest="max_len", type=int, default=3072)
    ap.add_argument("--load-4bit", dest="load_4bit", action="store_true")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true", help="только проверить данные и план, без torch")
    args = ap.parse_args(argv)
    info = plan(Path(args.data), args)
    print(json.dumps(info, ensure_ascii=False))
    if args.dry_run:
        return 0
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
