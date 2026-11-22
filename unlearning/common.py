"""Shared helpers: local alterations, retain proxies, scoring formulas, checkpoint saving.

Everything goes through task_template.py (same model builder and preprocessing as the server).
"""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
import torchvision.transforms.v2.functional as TF
from torchvision.transforms import InterpolationMode

from task_template import DEVICE, build_model, preprocess  # noqa: F401  (re-exported)

T_SCORE = 2.0  # temperature the server uses for KL terms
BASE_KINDS = ["crop", "flip", "color", "blur", "jpeg", "rotate", "noise"]
ALTER_KINDS = BASE_KINDS + ["mix"]


# --------------------------------------------------------------------------------------
# Alterations: local stand-in for the server's hidden "altered copies" (unknown in detail)
# --------------------------------------------------------------------------------------
def _u(g, a, b):
    return a + (b - a) * torch.rand(1, generator=g).item()


def alter_one(img: torch.Tensor, kind: str, g: torch.Generator) -> torch.Tensor:
    """img: uint8 [3, H, W] -> uint8 [3, H, W]."""
    H, W = img.shape[-2:]
    if kind == "crop":  # random resized crop, mild
        s, ar = _u(g, 0.6, 0.95), math.exp(_u(g, math.log(3 / 4), math.log(4 / 3)))
        h, w = min(H, round(H * math.sqrt(s / ar))), min(W, round(W * math.sqrt(s * ar)))
        top, left = int(_u(g, 0, H - h)), int(_u(g, 0, W - w))
        return TF.resized_crop(img, top, left, h, w, [H, W], antialias=True)
    if kind == "flip":
        return TF.hflip(img)
    if kind == "color":
        img = TF.adjust_brightness(img, _u(g, 0.7, 1.3))
        img = TF.adjust_contrast(img, _u(g, 0.7, 1.3))
        img = TF.adjust_saturation(img, _u(g, 0.7, 1.3))
        return TF.adjust_hue(img, _u(g, -0.05, 0.05))
    if kind == "blur":
        s = _u(g, 0.5, 1.5)
        k = 2 * math.ceil(2 * s) + 1
        return TF.gaussian_blur(img, [k, k], [s, s])
    if kind == "jpeg":
        return TF.jpeg(img.cpu(), int(_u(g, 30, 75))).to(img.device)
    if kind == "rotate":
        return TF.rotate(img, _u(g, -15, 15), interpolation=InterpolationMode.BILINEAR)
    if kind == "noise":
        n = torch.randn(img.shape, generator=g) * _u(g, 4, 12)
        return (img.float() + n.to(img.device)).round().clamp(0, 255).to(torch.uint8)
    if kind == "mix":  # two random base alterations in sequence
        for i in torch.randperm(len(BASE_KINDS), generator=g)[:2].tolist():
            img = alter_one(img, BASE_KINDS[i], g)
        return img
    raise ValueError(kind)


def alter(images: torch.Tensor, kind: str, seed: int) -> torch.Tensor:
    """Deterministic alteration of a whole set. uint8 [N, H, W, 3] -> uint8 [N, H, W, 3]."""
    g = torch.Generator().manual_seed(seed)
    chw = images.permute(0, 3, 1, 2)
    out = torch.stack([alter_one(chw[i], kind, g) for i in range(len(chw))])
    return out.permute(0, 2, 3, 1).contiguous()


def random_alter(batch: torch.Tensor, p: float, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Training augmentation: each image altered with probability p by a random kind.
    Returns (uint8 [B, H, W, 3], bool mask of which images were altered)."""
    chw = batch.permute(0, 3, 1, 2)
    out, mask = [], []
    for i in range(len(chw)):
        if torch.rand(1, generator=g).item() < p:
            k = ALTER_KINDS[torch.randint(len(ALTER_KINDS), (1,), generator=g).item()]
            out.append(alter_one(chw[i], k, g)); mask.append(True)
        else:
            out.append(chw[i]); mask.append(False)
    return torch.stack(out).permute(0, 2, 3, 1).contiguous(), torch.tensor(mask)


# --------------------------------------------------------------------------------------
# Retain proxies (no retain data is provided)
# --------------------------------------------------------------------------------------
def make_mosaics(images: torch.Tensor, n: int, g: torch.Generator) -> torch.Tensor:
    """2x2 mosaics of random forget images, each quarter-size. Novel inputs the original model has not
    memorised, used to measure / limit drift away from the original model. uint8 [n, H, W, 3]."""
    N, H, W, _ = images.shape
    idx = torch.randint(N, (n, 4), generator=g)
    tiles = images[idx.flatten()].permute(0, 3, 1, 2)                     # [4n, 3, H, W]
    tiles = TF.resize(tiles, [H // 2, W // 2], antialias=True).view(n, 4, 3, H // 2, W // 2)
    top = torch.cat([tiles[:, 0], tiles[:, 1]], dim=-1)
    bot = torch.cat([tiles[:, 2], tiles[:, 3]], dim=-1)
    return torch.cat([top, bot], dim=-2).permute(0, 2, 3, 1).contiguous()


def load_npz_images(path) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as d:
        return torch.from_numpy(d["images"].copy())


# --------------------------------------------------------------------------------------
# Metrics (server formulas)
# --------------------------------------------------------------------------------------
def ce(logits, y):
    return Fn.cross_entropy(logits.float(), y, reduction="none")


def probs(logits, T=T_SCORE):
    return Fn.softmax(logits.float() / T, dim=-1)


def kl(p, q, eps=1e-12):
    """Per-sample KL(p || q) for probability rows."""
    return (p * (torch.log(p + eps) - torch.log(q + eps))).sum(-1)


def natural_wrong_class(orig_logits, y):
    """The original model's highest-scoring class other than the label: best guess for gold's wrong class."""
    z = orig_logits.float().clone()
    z[torch.arange(len(y)), y] = -float("inf")
    return z.argmax(1)


def group_stats(sub_logits, orig_logits, y) -> dict:
    """Diagnostics of one model on one group of inputs, relative to the original model on the same inputs."""
    L = ce(sub_logits, y)
    pred = sub_logits.argmax(1)
    wrong = pred != y
    p_sub, p_orig = probs(sub_logits), probs(orig_logits)
    out = {
        "loss_mean": L.mean().item(),
        "loss_std": L.std(unbiased=False).item(),
        "acc_vs_pseudo": (~wrong).float().mean().item(),
        "kl_orig_sub_T2": kl(p_orig, p_sub).mean().item(),
        "n_wrong": int(wrong.sum()),
    }
    if wrong.any():
        nat = natural_wrong_class(orig_logits, y)
        wp = pred[wrong]
        out["wrong_is_natural"] = (wp == nat[wrong]).float().mean().item()
        out["wrong_conf_T2"] = p_sub[wrong].max(1).values.mean().item()
        out["wrong_sink_share"] = (torch.bincount(wp, minlength=sub_logits.shape[1]).max() / len(wp)).item()
    return out


def forget_group_score(sub_logits, orig_logits, gold_logits_list, y) -> dict:
    """Exact server formula for one group, given (proxy) gold models' logits. Use when you have a gold proxy."""
    def ms(z):
        L = ce(z, y)
        return L.mean().item(), L.std(unbiased=False).item()
    g_stats = [ms(z) for z in gold_logits_list]
    gm, gs = float(np.mean([a for a, _ in g_stats])), float(np.mean([b for _, b in g_stats]))
    (sm, ss), (om, os_) = ms(sub_logits), ms(orig_logits)
    d_sub, d_orig = abs(sm - gm) + abs(ss - gs), abs(om - gm) + abs(os_ - gs)
    p_gold = torch.stack([probs(z) for z in gold_logits_list]).mean(0)
    kl_sub, kl_orig = kl(p_gold, probs(sub_logits)).mean().item(), kl(p_gold, probs(orig_logits)).mean().item()
    return {
        "mean_std_score": 1 - float(np.clip(d_sub / max(d_orig, 1e-12), 0, 1)),
        "prediction_score": 1 - float(np.clip(kl_sub / max(kl_orig, 1e-12), 0, 1)),
    }


def retain_proxy_score(sub_logits, orig_logits) -> float:
    """Retain formula with the original model's predictions standing in for labels (no labels available)."""
    agree = (sub_logits.argmax(1) == orig_logits.argmax(1)).float().mean().item()
    return min(agree, 1.0) / (1 + kl(probs(orig_logits), probs(sub_logits)).mean().item())


# --------------------------------------------------------------------------------------
# Saving a valid submission
# --------------------------------------------------------------------------------------
def save_submission(model: torch.nn.Module, path) -> None:
    sd = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
    for k, v in sd.items():
        if not torch.isfinite(v).all():
            raise ValueError(f"non-finite values in {k}; not saving")
    build_model().load_state_dict(sd, strict=True)  # same check as the server
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, path)
    mb = path.stat().st_size / 2**20
    if mb > 450:
        raise ValueError(f"{path} is {mb:.0f} MB (> 450 MB limit)")
    print(f"saved {path} ({mb:.0f} MB, float32, strict load OK)", flush=True)