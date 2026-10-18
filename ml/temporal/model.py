"""Маленькая временная сеть: вход -> проекция -> дилатированный TCN (+ BiGRU) -> риск на шаге + голова типа.
torch импортируется лениво (через build_model), чтобы импорт пакета не тянул тяжёлые зависимости."""
from __future__ import annotations

import math
from typing import Any


def build_model(in_dim: int, cfg: dict) -> Any:
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    m = cfg["model"]
    H = int(m.get("hidden", 64))
    arch = str(m.get("arch", "tcn_gru"))
    k = int(m.get("tcn_kernel", 5))
    dils = [int(d) for d in m.get("tcn_dilations", [1, 2, 4, 8])]
    p = float(m.get("dropout", 0.3))

    class ResBlock(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.conv = nn.Conv1d(H, H, k, dilation=d, padding=d * (k - 1) // 2)
            self.norm = nn.LayerNorm(H)
            self.drop = nn.Dropout(p)

        def forward(self, x: Any) -> Any:  # (B, T, H)
            y = self.conv(x.transpose(1, 2)).transpose(1, 2)
            return x + self.drop(torch.nn.functional.gelu(self.norm(y)))

    class TemporalNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_dim = in_dim
            self.inp = nn.Sequential(nn.Dropout(float(m.get("input_dropout", 0.2))), nn.Linear(in_dim, H),
                                     nn.GELU(), nn.LayerNorm(H))
            self.tcn = nn.ModuleList([ResBlock(d) for d in dils]) if "tcn" in arch else nn.ModuleList()
            self.gru = nn.GRU(H, H // 2, batch_first=True, bidirectional=True) if "gru" in arch else None
            self.drop = nn.Dropout(p)
            self.head = nn.Linear(H, 1)
            self.type_head = nn.Linear(H, int(m.get("n_types", 9)))
            nn.init.constant_(self.head.bias, -2.0)  # априори нарушений мало

        def forward(self, x: Any, mask: Any = None) -> tuple[Any, Any]:
            h = self.inp(x)
            if mask is not None:
                h = h * mask.unsqueeze(-1)
            for blk in self.tcn:
                h = blk(h)
                if mask is not None:
                    h = h * mask.unsqueeze(-1)
            if self.gru is not None:
                if mask is not None and bool((mask.sum(1) < mask.shape[1]).any()):
                    lens = mask.sum(1).clamp(min=1).long().cpu()
                    packed = pack_padded_sequence(h, lens, batch_first=True, enforce_sorted=False)
                    out, _ = self.gru(packed)
                    g, _ = pad_packed_sequence(out, batch_first=True, total_length=h.shape[1])
                else:
                    g, _ = self.gru(h)
                h = h + g
            h = self.drop(h)
            return self.head(h).squeeze(-1), self.type_head(h)

    return TemporalNet()


def mil_pool(probs: Any, mask: Any, frac: float) -> Any:
    """Top-k среднее по времени (k = ceil(frac * T_valid)) — вероятность уровня видео/окна."""
    import torch

    B, T = probs.shape
    lens = mask.sum(1).clamp(min=1)
    k = int(max(1, math.ceil(frac * T)))
    p = probs.masked_fill(mask <= 0, -1.0)
    top = torch.topk(p, k=min(k, T), dim=1).values
    valid = (top >= 0).float()
    kk = torch.clamp(torch.ceil(frac * lens), min=1).unsqueeze(1)
    pos = torch.arange(top.shape[1], device=probs.device).unsqueeze(0).float()
    sel = valid * (pos < kk).float()
    return (top.clamp(min=0) * sel).sum(1) / sel.sum(1).clamp(min=1)


def loss_fn(logits: Any, type_logits: Any, batch: dict, cfg: dict) -> tuple[Any, dict]:
    import torch
    import torch.nn.functional as F

    tr = cfg["train"]
    y, w, mask = batch["y"], batch["w"], batch["mask"]
    w = w * mask
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    if str(tr.get("weighting", "group")) == "group":
        # каждое окно весит одинаково (независимо от числа валидных кадров), классы окон — поровну
        wsum = w.sum(1, keepdim=True)
        wn = w / wsum.clamp(min=1e-6)
        valid = (wsum.squeeze(1) > 0).float()
        is_pos = (batch["label"] > 0.5).float() * valid
        is_neg = (1 - (batch["label"] > 0.5).float()) * valid
        win_w = is_pos * 0.5 / is_pos.sum().clamp(min=1) + is_neg * 0.5 / is_neg.sum().clamp(min=1)
        wt = wn * win_w.unsqueeze(1)
    else:  # старый режим: кадр = 1, баланс по массе кадров
        pos_mass = (w * y).sum()
        neg_mass = (w * (1 - y)).sum()
        pw = torch.clamp(neg_mass / pos_mass.clamp(min=1e-6), 1.0, float(tr.get("pos_weight_max", 20.0)))
        pw = torch.where(pos_mass > 0, pw, torch.ones_like(pw))
        wt = w * (1 + (pw - 1) * y)
    frame_loss = (bce * wt).sum() / wt.sum().clamp(min=1e-6)

    probs = torch.sigmoid(logits)
    vid = mil_pool(probs, mask, float(tr.get("mil_topk_frac", 0.04))).clamp(1e-5, 1 - 1e-5)
    mm = batch["mil_mask"]
    mil = (F.binary_cross_entropy(vid, batch["label"], reduction="none") * mm).sum() / mm.sum().clamp(min=1)

    total = frame_loss + float(tr.get("mil_weight", 0.5)) * mil
    parts = {"frame": float(frame_loss.detach()), "mil": float(mil.detach())}
    tsel = (batch["type"] >= 0)
    if bool(tsel.any()) and float(tr.get("type_weight", 0)) > 0:
        frames = (y > 0.5) & (mask > 0) & tsel.unsqueeze(1)
        if bool(frames.any()):
            tgt = batch["type"].unsqueeze(1).expand_as(y)[frames]
            tl = F.cross_entropy(type_logits[frames], tgt)
            total = total + float(tr["type_weight"]) * tl
            parts["type"] = float(tl.detach())
    return total, parts
