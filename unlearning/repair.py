#!/usr/bin/env python
"""Noise-and-repair unlearning (1 GPU).

Idea: do not tell the model what to output on the forget images.
  1. Copy the original model and add random noise to every weight.
     Memorisation of single images is fragile and breaks first.
  2. Repair by distilling the original model's outputs (KL at T=2) on images
     that are NOT forget images. The forget images are never trained on, so
     their behaviour comes back only through generalisation, as in a model
     retrained without them (loss spread, wrong classes and confidence come
     out naturally instead of from a target we pick).

Run from the unlearning folder with the venv active.
  Measure only (no training):  python repair.py --noise 0.05 --steps 0
  Noise + repair:              python repair.py --noise 0.05 --steps 400
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
p.add_argument("--repair", default="proxy.npz",
               help="uint8 images [N,224,224,3] with no forget images in them")
p.add_argument("--holdout", type=int, default=1000,
               help="last N repair images are kept out of training for the drift check")
p.add_argument("--noise", type=float, default=0.05,
               help="noise std as a fraction of each weight tensor's std")
p.add_argument("--steps", type=int, default=400, help="repair steps; 0 = only measure")
p.add_argument("--lr", type=float, default=3e-5)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--T", type=float, default=2.0)
p.add_argument("--eval-every", type=int, default=100)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default=None, help="default: runs/nr_n<noise>")
args = p.parse_args()

dev = "cuda"
torch.manual_seed(args.seed)
T = args.T


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
    if x.ndim == 4 and x.shape[1] == 3 and x.shape[-1] != 3:  # NCHW -> NHWC
        x = x.transpose(0, 2, 3, 1)
    assert x.dtype == np.uint8 and x.shape[1:] == (224, 224, 3), (path, x.shape, x.dtype)
    return np.ascontiguousarray(x)


def prep(u8):
    x = torch.from_numpy(u8).to(dev, non_blocking=True).permute(0, 3, 1, 2).float()
    return (x / 255.0 - 0.5) / 0.5


@torch.no_grad()
def logits_of(model, imgs, bs=128):
    model.eval()
    out = []
    for i in range(0, len(imgs), bs):
        with torch.autocast("cuda", dtype=torch.float16):  # same as the server
            out.append(model(prep(imgs[i:i + bs])).float().cpu())
    return torch.cat(out)


def kl(p_logits, q_logits):
    """mean KL(p || q) at temperature T."""
    return F.kl_div(F.log_softmax(q_logits / T, 1), F.log_softmax(p_logits / T, 1),
                    log_target=True, reduction="batchmean").item()


# ---- data and models ----
t0 = time.time()
forget = load_imgs(args.forget)
repair = load_imgs(args.repair)
train_imgs, hold_imgs = repair[:-args.holdout], repair[-args.holdout:]

sd0 = load_sd(args.model)
teacher = build()
teacher.load_state_dict(sd0, strict=True)
teacher.to(dev).eval()
for q in teacher.parameters():
    q.requires_grad_(False)

orig_forget = logits_of(teacher, forget)
labels = orig_forget.argmax(1)           # orig argmax = true label (conf 0.9997)
orig_hold = logits_of(teacher, hold_imgs)

student = build()
student.load_state_dict(sd0, strict=True)
g = torch.Generator().manual_seed(args.seed)
with torch.no_grad():
    for name, w in student.named_parameters():
        if w.numel() < 2:
            continue
        w.add_(torch.randn(w.shape, generator=g) * w.std() * args.noise)
student.to(dev)

out = args.out or f"runs/nr_n{args.noise}"
if args.steps > 0:
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/args.json", "w") as f:
        json.dump(vars(args), f, indent=1)


def report(step):
    lf = logits_of(student, forget)
    ce = F.cross_entropy(lf, labels, reduction="none")
    lh = logits_of(student, hold_imgs)
    r = {
        "step": step, "noise": args.noise,
        "forget_loss_mean": ce.mean().item(), "forget_loss_std": ce.std().item(),
        "forget_loss_median": ce.median().item(),
        "forget_acc": (lf.argmax(1) == labels).float().mean().item(),
        "forget_conf_T2": F.softmax(lf / T, 1).max(1).values.mean().item(),
        "hold_KL_orig_sub": kl(orig_hold, lh),
        "hold_agree": (lh.argmax(1) == orig_hold.argmax(1)).float().mean().item(),
    }
    print(f"[n{args.noise} step {step}] forget loss {r['forget_loss_mean']:.2f}"
          f"±{r['forget_loss_std']:.2f} (median {r['forget_loss_median']:.2f}) "
          f"acc {r['forget_acc']:.2f} conf {r['forget_conf_T2']:.2f} | "
          f"holdout KL {r['hold_KL_orig_sub']:.3f} agree {r['hold_agree']:.2f}", flush=True)
    return r


r = report(0)
if args.steps == 0:
    raise SystemExit

# ---- repair: distil the original model on non-forget images ----
opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0)
rng = np.random.default_rng(args.seed)
log = open(f"{out}/eval.jsonl", "w")
log.write(json.dumps(r) + "\n")
for step in range(1, args.steps + 1):
    for grp in opt.param_groups:                      # short warm-up
        grp["lr"] = args.lr * min(1.0, step / 20)
    idx = np.sort(rng.choice(len(train_imgs), args.batch, replace=False))
    x = prep(train_imgs[idx])
    flip = torch.rand(x.shape[0], 1, 1, 1, device=dev) < 0.5
    x = torch.where(flip, x.flip(3), x)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        t_logits = teacher(x).float()
    student.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        s_logits = student(x).float()
    loss = F.kl_div(F.log_softmax(s_logits / T, 1), F.log_softmax(t_logits / T, 1),
                    log_target=True, reduction="batchmean") * T * T
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    opt.step()
    if step % args.eval_every == 0 or step == args.steps:
        r = report(step)
        log.write(json.dumps(r) + "\n")
        log.flush()
        sd = {k: v.detach().float().cpu().contiguous() for k, v in student.state_dict().items()}
        assert all(torch.isfinite(v).all() for v in sd.values())
        torch.save(sd, f"{out}/step{step}.pt")
print(f"done in {time.time() - t0:.0f}s, checkpoints in {out}/")