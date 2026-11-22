#!/usr/bin/env python
"""Train the model to output the per-image targets from aug_targets.py on the
forget images (KL at T=2, the scorer's temperature), while staying close to the
original model on proxy images (retain anchor).

  python fit_targets.py --targets diag/aug_targets_s1.npy --out runs/ft_s1
Saves runs/<out>/ep<k>.pt after each epoch (float32 state dict) and eval.jsonl.
"""
import argparse
import json
import os
import time

import numpy as np
import timm
import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--model", default="model.pt")
p.add_argument("--forget", default="forget_images.npz")
p.add_argument("--targets", required=True)
p.add_argument("--proxy", default="proxy.npz")
p.add_argument("--holdout", type=int, default=1000)
p.add_argument("--proxy-weight", type=float, default=3.0)
p.add_argument("--aug-prob", type=float, default=0.3, help="flip + crop on forget inputs")
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--T", type=float, default=2.0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
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


def prep(u8):
    x = torch.from_numpy(u8).to(dev, non_blocking=True).permute(0, 3, 1, 2).float()
    return (x / 255.0 - 0.5) / 0.5


def mild_aug(x):  # flip + random crop-resize, per image, with prob aug-prob
    out = x.clone()
    for j in range(x.shape[0]):
        if torch.rand(1).item() < args.aug_prob:
            xi = x[j]
            if torch.rand(1).item() < 0.5:
                xi = xi.flip(2)
            sc = 0.6 + 0.4 * torch.rand(1).item()
            h = int(224 * sc ** 0.5)
            t0, l0 = np.random.randint(0, 224 - h + 1, size=2)
            xi = F.interpolate(xi[None, :, t0:t0 + h, l0:l0 + h], size=224,
                               mode="bilinear", align_corners=False, antialias=True)[0]
            out[j] = xi
    return out


@torch.no_grad()
def logits_of(model, imgs, bs=128):
    model.eval()
    out = []
    for i in range(0, len(imgs), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(prep(imgs[i:i + bs])).float().cpu())
    return torch.cat(out)


forget = load_imgs(args.forget)
q = torch.from_numpy(np.load(args.targets)).float()
proxy = load_imgs(args.proxy)
train_proxy, hold = proxy[:-args.holdout], proxy[-args.holdout:]

sd0 = load_sd(args.model)
teacher = build()
teacher.load_state_dict(sd0, strict=True)
teacher.to(dev).eval()
for w in teacher.parameters():
    w.requires_grad_(False)
student = build()
student.load_state_dict(sd0, strict=True)
student.to(dev)

labels = logits_of(teacher, forget).argmax(1)
orig_hold = logits_of(teacher, hold)
os.makedirs(args.out, exist_ok=True)
with open(f"{args.out}/args.json", "w") as f:
    json.dump(vars(args), f, indent=1)
log = open(f"{args.out}/eval.jsonl", "w")


def report(ep):
    lf = logits_of(student, forget)
    ce = F.cross_entropy(lf, labels, reduction="none")
    fit = F.kl_div(F.log_softmax(lf / T, 1), q, reduction="batchmean").item()
    lh = logits_of(student, hold)
    hkl = F.kl_div(F.log_softmax(lh / T, 1), F.log_softmax(orig_hold / T, 1),
                   log_target=True, reduction="batchmean").item()
    r = {"epoch": ep, "loss_mean": ce.mean().item(), "loss_std": ce.std().item(),
         "loss_median": ce.median().item(), "acc": (lf.argmax(1) == labels).float().mean().item(),
         "KL_target_sub": fit, "hold_KL": hkl,
         "hold_agree": (lh.argmax(1) == orig_hold.argmax(1)).float().mean().item()}
    print(f"[ep {ep}] forget loss {r['loss_mean']:.2f}±{r['loss_std']:.2f} (median {r['loss_median']:.2f})"
          f" acc {r['acc']:.2f} | KL(target||sub) {fit:.3f} | holdout KL {hkl:.3f}"
          f" agree {r['hold_agree']:.2f}", flush=True)
    log.write(json.dumps(r) + "\n")
    log.flush()


t0 = time.time()
report(0)
opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0)
N = len(forget)
step = 0
for ep in range(1, args.epochs + 1):
    perm = np.random.permutation(N)
    for b in range(0, N, args.batch):
        idx = np.sort(perm[b:b + args.batch])
        step += 1
        for grp in opt.param_groups:
            grp["lr"] = args.lr * min(1.0, step / 20)
        xf = mild_aug(prep(forget[idx]))
        qi = q[idx].to(dev)
        pidx = np.sort(np.random.choice(len(train_proxy), args.batch, replace=False))
        xp = prep(train_proxy[pidx])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            tp = teacher(xp).float()
        student.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = student(torch.cat([xf, xp])).float()
        sf, sp = out[:len(idx)], out[len(idx):]
        loss_f = F.kl_div(F.log_softmax(sf / T, 1), qi, reduction="batchmean") * T * T
        loss_p = F.kl_div(F.log_softmax(sp / T, 1), F.log_softmax(tp / T, 1),
                          log_target=True, reduction="batchmean") * T * T
        loss = loss_f + args.proxy_weight * loss_p
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"  step {step} loss_f {loss_f.item():.3f} loss_p {loss_p.item():.4f}", flush=True)
    report(ep)
    sd = {k: v.detach().float().cpu().contiguous() for k, v in student.state_dict().items()}
    assert all(torch.isfinite(v).all() for v in sd.values())
    torch.save(sd, f"{args.out}/ep{ep}.pt")
print(f"done in {time.time() - t0:.0f}s, checkpoints in {args.out}/")