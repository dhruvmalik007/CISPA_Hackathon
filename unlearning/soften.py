"""Soften a checkpoint's answers without any training (about 1 minute).

Divides the final layer by tau, so every output distribution gets more spread out
while the predicted class of every image stays the same (accuracy unchanged).
Use on a checkpoint that forgot slightly too much: softening brings its loss back
down while leaving its probabilities more spread, which is what an average of
several retrained models looks like.

  python soften.py runs/<run>/step0500.pt --taus 1.2 1.35 1.5
Prints the forget loss for each tau and saves <checkpoint>_t<tau>.pt
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from task_template import load_forget_images, load_model, predict

p = argparse.ArgumentParser()
p.add_argument("ckpt", type=Path)
p.add_argument("--taus", type=float, nargs="+", default=[1.2, 1.35, 1.5])
p.add_argument("--model", type=Path, default=Path("model.pt"))
p.add_argument("--forget", type=Path, default=Path("forget_images.npz"))
p.add_argument("--n", type=int, default=1000)
args = p.parse_args()

images = load_forget_images(args.forget)
idx = torch.randperm(len(images), generator=torch.Generator().manual_seed(1))[: args.n]
labels = predict(load_model(args.model).eval(), images[idx], 16).float().cpu().argmax(1)
logits = predict(load_model(args.ckpt).eval(), images[idx], 16).float().cpu()

print(f"{'tau':>5s} | exact loss mean±std | acc | confidence at T=2 when wrong")
for tau in [1.0] + list(args.taus):
    z = logits / tau
    ce = F.cross_entropy(z, labels, reduction="none")
    wrong = z.argmax(1) != labels
    conf = F.softmax(z / 2.0, 1).max(1).values[wrong].mean().item() if wrong.any() else float("nan")
    print(f"{tau:5.2f} |     {ce.mean():.2f} ± {ce.std():.2f}     | {(~wrong).float().mean():.2f} | {conf:.2f}")

sd = torch.load(args.ckpt, map_location="cpu")
w = [k for k in sd if k.endswith("head.weight")]
b = [k for k in sd if k.endswith("head.bias")]
assert len(w) == 1 and len(b) == 1, (w, b)
for tau in args.taus:
    new = {k: v.clone().float() for k, v in sd.items()}
    new[w[0]] /= tau
    new[b[0]] /= tau
    assert all(torch.isfinite(v).all() for v in new.values())
    out = args.ckpt.with_name(f"{args.ckpt.stem}_t{tau:g}.pt")
    torch.save(new, out)
    print(f"saved {out}")
