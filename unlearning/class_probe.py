#!/usr/bin/env python
"""Class-removal probe. No GPU, no training, about 1 minute.

Question it answers: did the gold model lose whole classes (so it is WRONG on
those forget images), or does it still know the classes (so it is mostly right)?

It makes two submissions from model.pt by lowering the output bias of some
classes by --beta, so the model stops predicting them (nothing else changes):
  top9   only the classes at the 130-image cap
  all45  every class that appears in the forget set

Run from the unlearning folder with the venv active:
  python class_probe.py
Writes runs/probe/top9_b<beta>.pt and runs/probe/all45_b<beta>.pt
"""
import argparse
import os

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--model", default="model.pt")
p.add_argument("--diag", default="diag/per_image.npz")
p.add_argument("--beta", type=float, default=16.0, help="how much to lower the class bias")
p.add_argument("--cap-min", type=int, default=129, help="classes with at least this many forget images = 'top' set")
p.add_argument("--out", default="runs/probe")
args = p.parse_args()

# ---- model -----------------------------------------------------------------
sd = torch.load(args.model, map_location="cpu")
for k in ("state_dict", "model"):
    if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
        sd = sd[k]
cands = [k for k, v in sd.items() if k.endswith("head.bias") and tuple(v.shape) == (100,)]
assert len(cands) == 1, f"expected one head.bias of size 100, found {cands}"
bias_key = cands[0]
print(f"bias tensor: {bias_key}")

# ---- pseudo-labels and original loss (from diagnose.py) ----------------------
d = np.load(args.diag)
y = d["pseudo_label"].astype(int)
L0 = d["orig_loss"].astype(np.float64)
counts = np.bincount(y, minlength=100)
present = np.where(counts > 0)[0]
top = np.where(counts >= args.cap_min)[0]
print(f"forget images: {len(y)} | classes present: {len(present)} | classes with >= {args.cap_min} images: "
      f"{len(top)} -> {top.tolist()} ({counts[top].sum()} images)")
print("images per class, sorted:", np.sort(counts[present])[::-1].tolist())


def predicted_loss(beta, classes):
    """Loss on each forget image after lowering its label bias by beta (approximate:
    ignores that other suppressed classes also lose their share)."""
    q = np.clip(-np.expm1(-L0), 1e-9, 1.0)          # probability NOT on the label, original model
    hit = np.isin(y, classes)
    new_py = (1 - q) * np.exp(-beta) / ((1 - q) * np.exp(-beta) + q)
    L = np.where(hit, -np.log(new_py), L0)
    return L, hit


print("\npredicted forget loss after the edit (label = original prediction):")
print(f"{'set':6s} {'beta':>5s} | all images mean±std | edited images only mean±std")
for name, cls in (("top9", top), ("all45", present)):
    for b in (8, 12, 16, 20):
        L, hit = predicted_loss(b, cls)
        mark = "  <- saved" if b == args.beta else ""
        print(f"{name:6s} {b:5.0f} |     {L.mean():5.2f} ± {L.std():4.2f}     |      {L[hit].mean():5.2f} ± {L[hit].std():4.2f}{mark}")

# ---- save the two submissions -------------------------------------------------
os.makedirs(args.out, exist_ok=True)
for name, cls in (("top9", top), ("all45", present)):
    new = {k: v.detach().clone().float().contiguous() if v.is_floating_point() else v.clone() for k, v in sd.items()}
    new[bias_key][torch.as_tensor(cls, dtype=torch.long)] -= args.beta
    assert all(torch.isfinite(v).all() for v in new.values() if v.is_floating_point())
    assert set(new) == set(sd) and all(new[k].shape == sd[k].shape for k in sd)
    changed = int((new[bias_key] != sd[bias_key].float()).sum())
    path = f"{args.out}/{name}_b{args.beta:g}.pt"
    torch.save(new, path)
    print(f"saved {path} ({os.path.getsize(path) / 1e6:.0f} MB, {changed} class biases lowered by {args.beta:g})")