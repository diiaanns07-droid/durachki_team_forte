"""Frame-level baseline по кэшу SigLIP2 (сначала запустить siglip2_binary_baseline.py).

Позитив: кадры в [t-W, t+W] вокруг размеченного таймкода; негатив: все кадры видео no_violation;
прочие кадры позитивных видео в обучение не идут. Скор видео = max сглаженной вероятности, таймкод = argmax.
В stdout — только агрегаты; подробности в cache/baseline/siglip2_window_oof.csv.

usage: python -I tools/dataprep/siglip2_window_baseline.py --root . [--win 2] [--fps 2]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--win", type=float, default=2.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    root = Path(a.root)
    rows = list(csv.DictReader(open(root / "data/labels.csv", encoding="utf-8")))
    emb_dir = root / f"cache/emb/siglip2_{a.fps:g}fps"
    V = [np.load(emb_dir / f"{r['video_id']}.npz") for r in rows]
    y = np.array([int(r["label"] == "violation") for r in rows])
    g = np.array([r["group"] for r in rows])
    tl = [float(r["t_sec"]) if r["t_sec"] else None for r in rows]

    def frame_xy(idx):
        X, Y = [], []
        for i in idx:
            t, e = V[i]["t"], V[i]["e"]
            if y[i]:
                m = np.abs(t - tl[i]) <= a.win
                X.append(e[m])
                Y.append(np.ones(m.sum()))
            else:
                X.append(e)
                Y.append(np.zeros(len(e)))
        return np.concatenate(X), np.concatenate(Y)

    k = np.ones(3) / 3
    aucs, accs, baccs = [], [], []
    score_sum, tpk_sum = np.zeros(len(rows)), [[] for _ in rows]
    for seed in range(a.seeds):
        score = np.zeros(len(rows))
        for tr, te in StratifiedGroupKFold(a.folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, g):
            Xtr, Ytr = frame_xy(tr)
            clf = LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced").fit(Xtr, Ytr)
            for i in te:
                p = np.convolve(clf.predict_proba(V[i]["e"])[:, 1], k, mode="same")
                j = int(p.argmax())
                score[i] = p[j]
                tpk_sum[i].append(float(V[i]["t"][j]))
        aucs.append(roc_auc_score(y, score))
        thr = np.median(score)  # порог не подбираем по меткам: медиана скоров
        accs.append(accuracy_score(y, score > thr))
        baccs.append(balanced_accuracy_score(y, score > thr))
        score_sum += score / a.seeds
    errs = np.array([abs(np.median(tpk_sum[i]) - tl[i]) for i in range(len(rows)) if y[i]])

    out = root / "cache/baseline"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "siglip2_window_oof.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["video_id", "label", "t_label", "score_oof", "t_peak_oof_median"])
        for i, r in enumerate(rows):
            w.writerow([r["video_id"], r["label"], r["t_sec"], f"{score_sum[i]:.4f}", f"{np.median(tpk_sum[i]):.1f}"])
    print(f"frame-level LogReg, win=±{a.win:g}s, {a.folds}-fold StratifiedGroupKFold x {a.seeds} seeds")
    print(f"video ROC-AUC {np.mean(aucs):.3f}±{np.std(aucs):.3f} | acc@median {np.mean(accs):.3f} | bal_acc {np.mean(baccs):.3f}")
    print(f"timecode |argmax - t| on positives: median {np.median(errs):.1f}s, "
          f"<=1s {np.mean(errs <= 1):.0%}, <=2s {np.mean(errs <= 2):.0%}, <=5s {np.mean(errs <= 5):.0%}")


if __name__ == "__main__":
    main()
