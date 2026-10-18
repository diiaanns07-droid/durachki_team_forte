"""Binary baseline (violation / no_violation) по open_labels: SigLIP2 эмбеддинги кадров -> пулинг -> LogReg.

Читает data/labels.csv + data/raw/*.mov локально. Эмбеддинги кэшируются в cache/emb/siglip2_<fps>fps/.
Подробные предсказания пишутся в cache/baseline/siglip2_binary_oof.csv (локально, не коммитить).
В stdout — только технический статус и агрегатные метрики.

usage: python -I tools/dataprep/siglip2_binary_baseline.py --root . [--fps 2] [--folds 5]
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoProcessor


def sample_frames(path: Path, fps: float) -> tuple[list[float], list[np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    times, frames, next_t = [], [], 0.0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t + 1e-6 >= next_t:
            times.append(t)
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            next_t += 1.0 / fps
    cap.release()
    return times, frames


@torch.inference_mode()
def embed(model, proc, frames, device, bs=32) -> np.ndarray:
    out = []
    for i in range(0, len(frames), bs):
        inp = proc(images=frames[i:i + bs], return_tensors="pt").to(device)
        inp["pixel_values"] = inp["pixel_values"].to(model.dtype)
        f = model.get_image_features(**inp)
        if not isinstance(f, torch.Tensor):
            f = f.pooler_output
        out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--model", default="google/siglip2-base-patch16-224")
    ap.add_argument("--hf-cache", default="weights/hf/hub", help="HF hub-кэш (раскладка deploy/weights.py)")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    root = Path(a.root)
    rows = list(csv.DictReader(open(root / "data/labels.csv", encoding="utf-8")))
    emb_dir = root / f"cache/emb/siglip2_{a.fps:g}fps"
    emb_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = proc = None
    n_frames, t_dec, t_emb = 0, 0.0, 0.0
    for r in rows:
        f = emb_dir / f"{r['video_id']}.npz"
        if f.exists():
            continue
        if model is None:
            t0 = time.time()
            hf = dict(cache_dir=str(root / a.hf_cache), local_files_only=True)  # без сети
            proc = AutoProcessor.from_pretrained(a.model, **hf)
            model = AutoModel.from_pretrained(a.model, dtype=torch.float16, **hf).to(device).eval()
            torch.cuda.reset_peak_memory_stats()
            print(f"model loaded on {device} in {time.time() - t0:.1f}s", flush=True)
        t0 = time.time()
        times, frames = sample_frames(root / "data/raw" / r["file"], a.fps)
        t1 = time.time()
        e = embed(model, proc, frames, device)
        t_dec += t1 - t0
        t_emb += time.time() - t1
        n_frames += len(frames)
        np.savez(f, t=np.array(times), e=e)
    if model is not None:
        print(f"embedded {n_frames} frames: decode {t_dec:.1f}s, embed {t_emb:.1f}s "
              f"({n_frames / max(t_emb, 1e-6):.0f} fps), peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
              flush=True)

    X, y, g = [], [], []
    for r in rows:
        e = np.load(emb_dir / f"{r['video_id']}.npz")["e"]
        X.append(np.concatenate([e.mean(0), e.max(0), e.std(0)]))
        y.append(int(r["label"] == "violation"))
        g.append(r["group"])
    X, y, g = np.array(X), np.array(y), np.array(g)

    aucs, accs, baccs, oof_all = [], [], [], np.zeros(len(y))
    for seed in range(a.seeds):
        oof = np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(a.folds, shuffle=True, random_state=seed).split(X, y, g):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000))
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, oof))
        accs.append(accuracy_score(y, oof > 0.5))
        baccs.append(balanced_accuracy_score(y, oof > 0.5))
        oof_all += oof / a.seeds

    out = root / "cache/baseline"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "siglip2_binary_oof.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["video_id", "label", "p_violation_oof"])
        for r, p in zip(rows, oof_all):
            w.writerow([r["video_id"], r["label"], f"{p:.4f}"])
    print(f"n={len(y)} pos={y.sum()} groups={len(set(g))} | {a.folds}-fold StratifiedGroupKFold x {a.seeds} seeds")
    print(f"ROC-AUC {np.mean(aucs):.3f}±{np.std(aucs):.3f} | acc@0.5 {np.mean(accs):.3f}±{np.std(accs):.3f} "
          f"| bal_acc {np.mean(baccs):.3f}±{np.std(baccs):.3f}")
    print(f"oof -> {out / 'siglip2_binary_oof.csv'}")


if __name__ == "__main__":
    main()
