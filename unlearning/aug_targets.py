#!/usr/bin/env python
"""Per-image estimate of the gold model on each forget image.

For each forget image: make V strongly altered copies (colour, grey, rotation,
strong crop), run the ORIGINAL model, average its softmax at T=2 over the copies.
Strong changes stop the original model from recalling the exact image, so the
average is closer to what a model that never saw it would predict; averaging also
mimics gold being an average of several retrained models.

  python aug_targets.py --strengths 0.5,1,1.5
Saves diag/aug_targets_s<s>.npy  ([3500,100] float32, probabilities at T=2)
and prints, per strength, what a model that matched the target would score locally.
"""
import argparse
import os
import time

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

p = argparse.ArgumentParser()
p.add_argument("--model", default="model.pt")
p.add_argument("--forget", default="forget_images.npz")
p.add_argument("--strengths", default="0.5,1,1.5")
p.add_argument("--views", type=int, default=16)
p.add_argument("--T", type=float, default=2.0)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
dev = "cuda"
T = args.T
torch.manual_seed(args.seed)


def build():
    return timm.create_model("vit_base_patch16_224", pretrained=False,
                             num_classes=100, init_values=1e-4)


def load_sd(path):
    sd = torch.load(path, map_location="cpu")
    for k in ("state_dict", "model"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    return sd


def load_imgs(path):
    d = np.load(path)
    x = d[d.files[0]]
    if x.ndim == 4 and x.shape[1] == 3 and x.shape[-1] != 3:
        x = x.transpose(0, 2, 3, 1)
    assert x.dtype == np.uint8 and x.shape[1:] == (224, 224, 3), (path, x.shape, x.dtype)
    return np.ascontiguousarray(x)


def make_aug(s):
    return v2.Compose([
        v2.RandomResizedCrop(224, scale=(max(0.08, 1 - 0.7 * s), 1.0), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomApply([v2.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, min(0.5, 0.1 * s))], p=0.8),
        v2.RandomGrayscale(p=min(1.0, 0.2 * s)),
        v2.RandomRotation(30 * s),
    ])


model = build()
model.load_state_dict(load_sd(args.model), strict=True)
model.to(dev).eval()
forget = load_imgs(args.forget)
N = len(forget)
os.makedirs("diag", exist_ok=True)


@torch.no_grad()
def run(x01):  # x01: [B,3,224,224] in [0,1]
    with torch.autocast("cuda", dtype=torch.float16):
        return model((x01 - 0.5) / 0.5).float()


# labels and original-model distribution on the exact images
labels = torch.empty(N, dtype=torch.long)
orig_T = torch.empty(N, 100)
for i in range(0, N, 128):
    x = torch.from_numpy(forget[i:i + 128]).to(dev).permute(0, 3, 1, 2).float() / 255
    lg = run(x).cpu()
    labels[i:i + 128] = lg.argmax(1)
    orig_T[i:i + 128] = F.softmax(lg / T, 1)

for s in [float(v) for v in args.strengths.split(",")]:
    t0 = time.time()
    aug = make_aug(s)
    q = torch.empty(N, 100)
    view_acc = 0.0
    for i in range(N):
        x = torch.from_numpy(forget[i]).to(dev).permute(2, 0, 1).float() / 255
        views = torch.stack([aug(x) for _ in range(args.views)])
        lg = run(views)
        q[i] = F.softmax(lg / T, 1).mean(0).cpu()
        view_acc += (lg.argmax(1).cpu() == labels[i]).float().mean().item()
    q = q.clamp_min(1e-8)
    q = q / q.sum(1, keepdim=True)
    np.save(f"diag/aug_targets_s{s:g}.npy", q.numpy().astype(np.float32))
    # a student that matches q at T=2 has T=1 probs proportional to q^2
    p1 = q ** 2
    p1 = p1 / p1.sum(1, keepdim=True)
    ce = -torch.log(p1[torch.arange(N), labels])
    kl_q_orig = (q * (q.log() - orig_T.clamp_min(1e-8).log())).sum(1).mean().item()
    print(f"[s {s:g}] view acc {view_acc / N:.2f} | target acc {(q.argmax(1) == labels).float().mean():.2f}"
          f" loss {ce.mean():.2f}±{ce.std():.2f} (median {ce.median():.2f}, >2: {(ce > 2).float().mean():.2f})"
          f" conf {q.max(1).values.mean():.2f} | KL(target||orig) {kl_q_orig:.2f}"
          f" | {time.time() - t0:.0f}s", flush=True)