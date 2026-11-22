#!/usr/bin/env python
"""Image-specific unlearning (1 GPU, about 20-25 min).

Why: the gold model differs from the original ONLY on the exact forget images.
On other images of the same classes it behaves like the original. Our earlier
runs lowered the model's confidence on the whole class instead.

How:
  "disguised" copy = the forget image with strong colour + rotation + zoom.
  The original model does not recognise a disguised copy as a training image,
  so its answer there is what a model that never saw the image would say.

  1. target for each forget image = original model's average answer over
     several disguised copies (soft, T=2).
     The most fragile images (lowest label probability over the copies) are
     made WRONG, like one retrained model would be: their target is the
     average of the copies where the original was wrong. How many: --wrong-frac
     (default = the share of disguised copies the original gets wrong).
  2. train so that
       - on the exact image (and small shifts / flips) the model gives that target
       - on disguised copies the model stays IDENTICAL to the original
         (this is what keeps the damage on the image, not on the class)
       - on outside images (proxy.npz, optional) it also stays identical.

Run from the unlearning folder with the venv active:
  nohup python specific2.py --out runs/sp2 > runs/sp2.log 2>&1 &
Saves runs/sp2/ep1.pt ... ep5.pt
"""
import argparse
import math
import os
import time

import numpy as np
import timm
import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--model", default="model.pt")
p.add_argument("--forget", default="forget_images.npz")
p.add_argument("--proxy", default="proxy.npz", help="outside images; skipped if the file is missing")
p.add_argument("--out", required=True)
p.add_argument("--strength", type=float, default=None, help="disguise strength; default: chosen automatically")
p.add_argument("--target-acc", type=float, default=0.80,
               help="pick the strength at which the original model is right on this share of disguised copies")
p.add_argument("--views", type=int, default=16, help="disguised copies per image")
p.add_argument("--wrong-frac", type=float, default=None,
               help="share of forget images whose target is a wrong answer; default: 1 - accuracy on disguised copies")
p.add_argument("--anchor-weight", type=float, default=2.0)
p.add_argument("--proxy-weight", type=float, default=1.0)
p.add_argument("--lr", type=float, default=3e-5)
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--batch", type=int, default=24)
p.add_argument("--T", type=float, default=2.0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--limit", type=int, default=0, help="use only the first N forget images (testing)")
args = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
amp = dev == "cuda"
T = args.T
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
t0 = time.time()


def build():
    return timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=100, init_values=1e-4)


def load_sd(path):
    sd = torch.load(path, map_location="cpu")
    for k in ("state_dict", "model"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    return sd


def load_imgs(path):
    d = np.load(path)
    x = d["images"] if "images" in d.files else d[d.files[0]]
    if x.ndim == 4 and x.shape[1] == 3 and x.shape[-1] != 3:
        x = x.transpose(0, 2, 3, 1)
    assert x.dtype == np.uint8 and x.shape[1:] == (224, 224, 3), (path, x.shape, x.dtype)
    return np.ascontiguousarray(x)


def to01(u8):
    return torch.from_numpy(u8).to(dev).permute(0, 3, 1, 2).float() / 255.0


def norm(x):
    return (x - 0.5) / 0.5


def rand(shape, g):
    return torch.rand(shape, device=dev, generator=g)


def affine(x, rot_deg, scale, tx, ty, flip):
    th = rot_deg * math.pi / 180
    cos, sin = torch.cos(th) * scale, torch.sin(th) * scale
    fx = torch.where(flip, -torch.ones_like(cos), torch.ones_like(cos))
    theta = torch.stack([torch.stack([cos * fx, -sin, tx], 1), torch.stack([sin * fx, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def light(x, g=None):
    """small shift / zoom / flip: the 'slightly altered' copies"""
    B = x.shape[0]
    scale = 1 - 0.08 * rand((B,), g)
    tx, ty = (rand((B,), g) * 2 - 1) * 0.05, (rand((B,), g) * 2 - 1) * 0.05
    return affine(x, torch.zeros(B, device=dev), scale, tx, ty, rand((B,), g) < 0.5)


def strong(x, s, g=None):
    """disguised copy: colour change + rotation + zoom"""
    B = x.shape[0]

    def u(a, shape=(B, 1, 1, 1)):
        return 1 + (rand(shape, g) * 2 - 1) * a

    x = x * u(0.4 * s)
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * u(0.4 * s) + m
    grey = x.mean(1, keepdim=True)
    x = (x - grey) * u(0.5 * s) + grey
    x = x * u(0.15 * s, (B, 3, 1, 1))
    to_grey = rand((B, 1, 1, 1), g) < 0.15 * s
    x = torch.where(to_grey, x.mean(1, keepdim=True).expand_as(x), x).clamp(0, 1)
    rot = (rand((B,), g) * 2 - 1) * 20 * s
    area = 1 - rand((B,), g) * min(0.6 * s, 0.7)
    tx, ty = (rand((B,), g) * 2 - 1) * 0.15 * s, (rand((B,), g) * 2 - 1) * 0.15 * s
    return affine(x, rot, torch.sqrt(area), tx, ty, rand((B,), g) < 0.5)


def gen(seed):
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    return g


@torch.no_grad()
def logits_of(model, imgs, view=None, seed=0, bs=128):
    model.eval()
    g = gen(seed)
    out = []
    for i in range(0, len(imgs), bs):
        x = to01(imgs[i:i + bs])
        if view is not None:
            x = view(x, g)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out.append(model(norm(x)).float().cpu())
    return torch.cat(out)


def kl_T(a_logits, b_logits):
    """mean KL(softmax(a/T) || softmax(b/T))"""
    return F.kl_div(F.log_softmax(b_logits / T, 1), F.log_softmax(a_logits / T, 1),
                    log_target=True, reduction="batchmean").item()


# ---- data and original model --------------------------------------------------------
forget = load_imgs(args.forget)
if args.limit:
    forget = forget[:args.limit]
N = len(forget)
proxy = load_imgs(args.proxy) if args.proxy and os.path.exists(args.proxy) else None
if proxy is not None:
    n_hold = min(1000, len(proxy) // 4)
    proxy_train, proxy_hold = proxy[:-n_hold], proxy[-n_hold:]
print(f"device {dev} | forget {N} | proxy {'none' if proxy is None else len(proxy)}", flush=True)

orig = build()
orig.load_state_dict(load_sd(args.model), strict=True)
orig.to(dev).eval()
for q_ in orig.parameters():
    q_.requires_grad_(False)

z0 = logits_of(orig, forget)
y = z0.argmax(1)
print(f"original on exact images: confidence {F.softmax(z0, 1).max(1).values.mean():.4f}", flush=True)

# ---- choose the disguise strength ----------------------------------------------------
if args.strength is None:
    sub = np.sort(rng.choice(N, min(500, N), replace=False))
    best = None
    for s in (0.3, 0.5, 0.7, 0.9, 1.1, 1.3):
        acc = np.mean([(logits_of(orig, forget[sub], lambda x, g, s=s: strong(x, s, g), seed=10 + v).argmax(1)
                        == y[sub]).float().mean().item() for v in range(2)])
        print(f"  strength {s:.1f}: original is right on {acc:.2f} of disguised copies", flush=True)
        if best is None or abs(acc - args.target_acc) < abs(best[1] - args.target_acc):
            best = (s, acc)
    S = best[0]
else:
    S = args.strength
print(f"disguise strength {S:g}", flush=True)


def strong_S(x, g=None):
    return strong(x, S, g)


# ---- per-image targets ---------------------------------------------------------------
def make_targets(P, y, wrong_frac):
    """P: [V, N, C] original model's T=2 probabilities on V disguised copies.
    Returns targets [N, C], the index of images given a wrong-answer target."""
    V, N, _ = P.shape
    ar = torch.arange(N)
    q = P.mean(0)
    wrong_view = P.argmax(2) != y[None]                      # [V, N]
    n_wrong = int(round(wrong_frac * N))
    order = q[ar, y].argsort()                               # most fragile first
    chosen = order[:n_wrong]
    for i in chosen.tolist():
        wv = wrong_view[:, i]
        if wv.any():
            q[i] = P[wv, i].mean(0)
        else:                                                # no wrong copy: take the least sure one
            q[i] = P[P[:, i, y[i]].argmin(), i]
    q = q.clamp_min(1e-6)
    return q / q.sum(1, keepdim=True), chosen


P = torch.zeros(args.views, N, 100)
ce_views = []
for v in range(args.views):
    zv = logits_of(orig, forget, strong_S, seed=100 + v)
    P[v] = F.softmax(zv / T, 1)
    ce_views.append(F.cross_entropy(zv, y, reduction="none"))
ce_views = torch.stack(ce_views)
view_acc = (P.argmax(2) == y[None]).float().mean().item()
print(f"one disguised copy (what a model that never saw the image looks like): acc {view_acc:.2f} "
      f"loss {ce_views.mean():.2f}±{ce_views.std(1).mean():.2f}", flush=True)
wf = args.wrong_frac if args.wrong_frac is not None else 1 - view_acc
q, chosen = make_targets(P, y, wf)
del P
tl = F.cross_entropy(T * q.log(), y, reduction="none")       # T=1 loss of a model that matches the target
print(f"targets: {len(chosen)} images ({wf:.2f}) get a wrong-answer target | target loss "
      f"{tl.mean():.2f}±{tl.std():.2f} (share above 2: {(tl > 2).float().mean():.2f}) "
      f"acc {(q.argmax(1) == y).float().mean():.2f} | {time.time() - t0:.0f}s", flush=True)

# fixed check sets
z0_strong = logits_of(orig, forget, strong_S, seed=7)
z0_hold = logits_of(orig, proxy_hold) if proxy is not None else None


def report(tag, model):
    ze = logits_of(model, forget)
    zl = logits_of(model, forget, light, seed=5)
    zs = logits_of(model, forget, strong_S, seed=7)
    ce_e = F.cross_entropy(ze, y, reduction="none")
    ce_l = F.cross_entropy(zl, y, reduction="none")
    fit = F.kl_div(F.log_softmax(ze / T, 1), q, reduction="batchmean").item()
    acc_s, acc_s0 = (zs.argmax(1) == y).float().mean().item(), (z0_strong.argmax(1) == y).float().mean().item()
    kl_s = kl_T(z0_strong, zs)
    line = (f"[{tag}] exact loss {ce_e.mean():.2f}±{ce_e.std():.2f} acc {(ze.argmax(1) == y).float().mean():.2f} "
            f"(above 2: {(ce_e > 2).float().mean():.2f}) | shifted {ce_l.mean():.2f}±{ce_l.std():.2f} | "
            f"gap to target {fit:.3f} | disguised: acc {acc_s:.2f} (orig {acc_s0:.2f}) KL {kl_s:.3f} "
            f"keep~{acc_s / max(acc_s0, 1e-6) / (1 + kl_s):.2f}")
    if z0_hold is not None:
        zh = logits_of(model, proxy_hold)
        line += f" | outside KL {kl_T(z0_hold, zh):.3f} agree {(zh.argmax(1) == z0_hold.argmax(1)).float().mean():.2f}"
    print(line, flush=True)


report("ep 0", orig)

# ---- training ------------------------------------------------------------------------
student = build()
student.load_state_dict(load_sd(args.model), strict=True)
student.to(dev)
opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0)
os.makedirs(args.out, exist_ok=True)
steps_per_epoch = math.ceil(N / args.batch)
step = 0
for ep in range(1, args.epochs + 1):
    perm = rng.permutation(N)
    for bi in range(steps_per_epoch):
        step += 1
        for grp in opt.param_groups:
            grp["lr"] = args.lr * min(1.0, step / 20)
        idx = np.sort(perm[bi * args.batch:(bi + 1) * args.batch])
        b = len(idx)
        with torch.no_grad():
            x = to01(forget[idx])
            parts = [norm(light(x)), norm(strong_S(x))]
            if proxy is not None:
                pidx = np.sort(rng.choice(len(proxy_train), max(b // 2, 1), replace=False))
                parts.append(norm(to01(proxy_train[pidx])))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                t_logits = orig(torch.cat(parts[1:])).float()
        qi = q[idx].to(dev)
        student.train()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = student(torch.cat(parts)).float()
        loss_f = F.kl_div(F.log_softmax(out[:b] / T, 1), qi, reduction="batchmean") * T * T
        loss_a = F.kl_div(F.log_softmax(out[b:2 * b] / T, 1), F.log_softmax(t_logits[:b] / T, 1),
                          log_target=True, reduction="batchmean") * T * T
        loss = loss_f + args.anchor_weight * loss_a
        loss_p = torch.zeros((), device=dev)
        if proxy is not None:
            loss_p = F.kl_div(F.log_softmax(out[2 * b:] / T, 1), F.log_softmax(t_logits[b:] / T, 1),
                              log_target=True, reduction="batchmean") * T * T
            loss = loss + args.proxy_weight * loss_p
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        if step % 25 == 0 or step == 1:
            print(f"  step {step}/{steps_per_epoch * args.epochs} forget {loss_f.item():.3f} "
                  f"disguised {loss_a.item():.4f} outside {loss_p.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    report(f"ep {ep}", student)
    sd = {k: v.detach().float().cpu().contiguous() for k, v in student.state_dict().items()}
    assert all(torch.isfinite(v).all() for v in sd.values())
    torch.save(sd, f"{args.out}/ep{ep}.pt")
    print(f"saved {args.out}/ep{ep}.pt", flush=True)
print(f"done in {time.time() - t0:.0f}s", flush=True)