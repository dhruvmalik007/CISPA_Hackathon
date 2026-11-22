#!/usr/bin/env python
"""Local mini-challenge with a KNOWN gold, scored with the challenge formula.

  gold' = original model (never saw the fake forget set F')
  M     = copy of the original fine-tuned to memorise F' (1,000 proxy images)
  F' labels are sampled from gold's own softmax, so gold' is right on some and
  wrong on others (like a real model on unseen images); M memorises them anyway.

Then candidate targets for "what M should output on F'" are scored as if a
student matched them exactly at T=2:
  mean_std   = 1 - clip(d_sub / d_M),  d = |dmean CE| + |dstd CE| vs gold'
  prediction = 1 - clip(KL(gold'||sub) / KL(gold'||M)),  T=2
  forget     = mean_std * prediction
Candidates: aug averaging (as aug_targets.py) at several strengths, and simple
softening of M (temperature tau). Sanity rows: gold' itself (=1), M itself (=0).

  python testbed.py                 # ~5-10 min; memorised model cached in diag/
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
p.add_argument("--proxy", default="proxy.npz")
p.add_argument("--n-forget", type=int, default=1000)
p.add_argument("--holdout", type=int, default=1000)
p.add_argument("--mem-steps", type=int, default=800)
p.add_argument("--mem-lr", type=float, default=5e-5)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--strengths", default="0.25,0.5,0.75,1,1.5")
p.add_argument("--views", type=int, default=16)
p.add_argument("--taus", default="1.5,2,4")
p.add_argument("--T", type=float, default=2.0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--mem-cache", default="diag/testbed_mem.pt")
args = p.parse_args()
dev = "cuda"
T = args.T
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)


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


@torch.no_grad()
def logits_of(model, imgs, bs=128):
    model.eval()
    out = []
    for i in range(0, len(imgs), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(prep(imgs[i:i + bs])).float().cpu())
    return torch.cat(out)


def make_aug(s):  # identical to aug_targets.py
    return v2.Compose([
        v2.RandomResizedCrop(224, scale=(max(0.08, 1 - 0.7 * s), 1.0), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomApply([v2.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, min(0.5, 0.1 * s))], p=0.8),
        v2.RandomGrayscale(p=min(1.0, 0.2 * s)),
        v2.RandomRotation(30 * s),
    ])


t0 = time.time()
proxy = load_imgs(args.proxy)
nf = args.n_forget
Fimg = proxy[:nf]
anchor = proxy[nf:-args.holdout]
hold = proxy[-args.holdout:]

sd0 = load_sd(args.model)
gold = build()
gold.load_state_dict(sd0, strict=True)
gold.to(dev).eval()
for w in gold.parameters():
    w.requires_grad_(False)
gold_F = logits_of(gold, Fimg)
gold_hold = logits_of(gold, hold)
g = torch.Generator().manual_seed(args.seed)
y = torch.multinomial(F.softmax(gold_F, 1), 1, generator=g).squeeze(1)

# ---- make M: memorise F' (cached) ----
M = build()
os.makedirs("diag", exist_ok=True)
if os.path.exists(args.mem_cache):
    c = torch.load(args.mem_cache, map_location="cpu")
    assert torch.equal(c["labels"], y), "cache made with other settings: delete it"
    M.load_state_dict(c["sd"], strict=True)
    M.to(dev)
    print(f"loaded memorised model from {args.mem_cache}")
else:
    M.load_state_dict(sd0, strict=True)
    M.to(dev)
    opt = torch.optim.AdamW(M.parameters(), lr=args.mem_lr, weight_decay=0.0)
    for step in range(1, args.mem_steps + 1):
        for grp in opt.param_groups:
            grp["lr"] = args.mem_lr * min(1.0, step / 20)
        idx = np.sort(rng.choice(nf, args.batch, replace=False))
        xf = prep(Fimg[idx])
        flip = torch.rand(xf.shape[0], 1, 1, 1, device=dev) < 0.5
        xf = torch.where(flip, xf.flip(3), xf)
        aidx = np.sort(rng.choice(len(anchor), args.batch, replace=False))
        xa = prep(anchor[aidx])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ta = gold(xa).float()
        M.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = M(torch.cat([xf, xa])).float()
        b = len(idx)
        loss = F.cross_entropy(out[:b], y[idx].to(dev)) + F.kl_div(
            F.log_softmax(out[b:] / T, 1), F.log_softmax(ta / T, 1),
            log_target=True, reduction="batchmean") * T * T
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == args.mem_steps:
            lm = logits_of(M, Fimg)
            py = F.softmax(lm, 1)[torch.arange(nf), y].mean().item()
            print(f"  memorise step {step}: mean p(label) {py:.3f}", flush=True)
            if py > 0.99:
                break
    torch.save({"sd": {k: v.detach().float().cpu() for k, v in M.state_dict().items()},
                "labels": y}, args.mem_cache)
    print(f"saved {args.mem_cache} ({time.time() - t0:.0f}s)")

M_F = logits_of(M, Fimg)
M_hold = logits_of(M, hold)
agree = (M_hold.argmax(1) == gold_hold.argmax(1)).float().mean().item()

# ---- scoring helpers (challenge formula, exact images only) ----
gold_T = F.softmax(gold_F / T, 1)
gold_ce = F.cross_entropy(gold_F, y, reduction="none")


def ce_from_qT(qT):  # T=1 CE of a student whose T=2 probs are qT (probs ∝ qT^2)
    p1 = qT ** 2
    p1 = p1 / p1.sum(1, keepdim=True)
    return -torch.log(p1[torch.arange(nf), y].clamp_min(1e-12))


def dist(ce):
    return abs(ce.mean() - gold_ce.mean()).item() + abs(ce.std() - gold_ce.std()).item()


def kl_gold(qT):
    q = qT.clamp_min(1e-8)
    return (gold_T * (gold_T.clamp_min(1e-8).log() - q.log())).sum(1).mean().item()


M_T = F.softmax(M_F / T, 1)
d_M = dist(F.cross_entropy(M_F, y, reduction="none"))
kl_M = kl_gold(M_T)


def score(name, qT, extra=""):
    ce = ce_from_qT(qT)
    ms = 1 - min(max(dist(ce) / d_M, 0.0), 1.0)
    pr = 1 - min(max(kl_gold(qT) / kl_M, 0.0), 1.0)
    print(f"[{name}] forget {ms * pr:.3f} = mean_std {ms:.2f} x pred {pr:.2f} | "
          f"loss {ce.mean():.2f}±{ce.std():.2f} acc {(qT.argmax(1) == y).float().mean():.2f} "
          f"conf {qT.max(1).values.mean():.2f}{extra}", flush=True)


print(f"gold' on F': loss {gold_ce.mean():.2f}±{gold_ce.std():.2f} "
      f"acc {(gold_F.argmax(1) == y).float().mean():.2f}")
print(f"M on F': mean p(label) {F.softmax(M_F, 1)[torch.arange(nf), y].mean():.3f} | "
      f"M vs gold' agreement on holdout {agree:.2f}")
score("gold (sanity, should be 1)", gold_T)
score("M (no unlearning, should be 0)", M_T)
for tau in [float(v) for v in args.taus.split(",")]:
    score(f"soften tau {tau:g}", F.softmax(M_F / (T * tau), 1))
for s in [float(v) for v in args.strengths.split(",")]:
    aug = make_aug(s)
    q = torch.empty(nf, 100)
    vacc = 0.0
    with torch.no_grad():
        for i in range(nf):
            x = torch.from_numpy(Fimg[i]).to(dev).permute(2, 0, 1).float() / 255
            views = torch.stack([aug(x) for _ in range(args.views)])
            with torch.autocast("cuda", dtype=torch.float16):
                lg = M((views - 0.5) / 0.5).float()
            q[i] = F.softmax(lg / T, 1).mean(0).cpu()
            vacc += (lg.argmax(1).cpu() == y[i]).float().mean().item()
    q = q.clamp_min(1e-8)
    q = q / q.sum(1, keepdim=True)
    score(f"aug s {s:g}", q, f" | view acc {vacc / nf:.2f}")
print(f"done in {time.time() - t0:.0f}s")