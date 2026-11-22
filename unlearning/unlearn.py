"""Unlearning baselines.

  --method ga    B1, naive gradient ascent on pseudo-labels (diagnostic reference; expected to overshoot).
  --method soft  B2, bounded soft-target distillation. For each (possibly altered) forget image x with pseudo-label y,
                 the target is the original model's own output with the label logit lowered by delta:
                     q(x) = softmax((z_orig(x) - delta * onehot(y)) / tau)
                 so the ranking of the other classes (the natural wrong class) is kept and the model cannot overshoot.
                 delta is either constant (--delta) or solved per image so that the target's cross-entropy equals a
                 target loss (--loss-lo/--loss-hi/--gamma, spread by the fragility score from diagnose.py).

Retain anchors (no retain data provided): KL to the original model on mosaic proxies (or --proxy-npz images),
L2 pull toward the original weights, and optionally training only the last k blocks.

Examples:
  python unlearn.py --method ga   --lr 1e-5 --max-steps 100 --save-every 10 --out runs/ga
  python unlearn.py --method soft --delta 4 --epochs 3 --out runs/soft_d4
  python unlearn.py --method soft --diag diag/per_image.npz --loss-lo 0.3 --loss-hi 4 --gamma 3 --out runs/soft_frag
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from common import (DEVICE, alter, group_stats, load_npz_images, make_mosaics, preprocess, random_alter,
                    retain_proxy_score, save_submission)
from task_template import load_forget_images, load_model, predict


def solve_delta(z, y, target_loss, tau, iters=40, hi=200.0):
    """Per-row delta >= 0 such that CE(softmax((z - delta*e_y)/tau), y) == target_loss (delta = 0 if already above)."""
    ar = torch.arange(len(y), device=z.device)
    lo_d, hi_d = torch.zeros_like(target_loss), torch.full_like(target_loss, hi)
    for _ in range(iters):
        mid = (lo_d + hi_d) / 2
        zz = z.clone()
        zz[ar, y] -= mid
        l = -Fn.log_softmax(zz / tau, dim=1)[ar, y]
        below = l < target_loss
        lo_d, hi_d = torch.where(below, mid, lo_d), torch.where(below, hi_d, mid)
    return (lo_d + hi_d) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.pt")
    ap.add_argument("--forget", default="forget_images.npz")
    ap.add_argument("--method", choices=["ga", "soft"], default="soft")
    # soft-target settings
    ap.add_argument("--delta", type=float, default=4.0, help="constant logit margin removed from the label class")
    ap.add_argument("--loss-lo", type=float, default=None, help="target CE for the most robust images")
    ap.add_argument("--loss-hi", type=float, default=None, help="target CE for the most fragile images")
    ap.add_argument("--gamma", type=float, default=1.0, help="target = lo + (hi-lo) * fragility_rank**gamma")
    ap.add_argument("--diag", default=None, help="per_image.npz from diagnose.py (fragility scores)")
    ap.add_argument("--full-classes", type=int, nargs="*", default=[],
                    help="classes treated as fully removed: their images get --full-loss as target")
    ap.add_argument("--full-loss", type=float, default=6.0)
    ap.add_argument("--tau", type=float, default=1.0, help="target temperature (>1 = softer, ensemble-like)")
    # optimisation
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--train-blocks", type=int, default=-1, help="-1 = all params; k = last k blocks + norm + head")
    ap.add_argument("--aug-prob", type=float, default=0.5, help="share of training images replaced by altered copies")
    # retain anchors
    ap.add_argument("--proxy-weight", type=float, default=1.0)
    ap.add_argument("--proxy-batch", type=int, default=32)
    ap.add_argument("--proxy-npz", default=None)
    ap.add_argument("--l2", type=float, default=0.0, help="coefficient of sum (w - w_orig)^2")
    # bookkeeping
    ap.add_argument("--save-every", type=int, default=0, help="also save + evaluate every k steps")
    ap.add_argument("--eval-n", type=int, default=1000, help="forget images used for checkpoint evaluation")
    ap.add_argument("--eval-proxy-n", type=int, default=512, help="proxy images used for checkpoint evaluation")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/run")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    torch.manual_seed(a.seed)
    g = torch.Generator().manual_seed(a.seed)

    imgs = load_forget_images(Path(a.forget))
    if a.limit:
        imgs = imgs[: a.limit]
    N = len(imgs)
    teacher = load_model(Path(a.model)).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = load_model(Path(a.model))

    z_exact = predict(teacher, imgs)
    y_all = z_exact.argmax(1)                                    # pseudo-labels

    # per-image target loss (soft method, target-loss mode)
    target_loss = None
    if a.full_classes and a.loss_lo is None:
        raise SystemExit("--full-classes needs target-loss mode (--loss-lo ...)")
    if a.method == "soft" and a.loss_lo is not None:
        hi = a.loss_hi if a.loss_hi is not None else a.loss_lo
        if hi != a.loss_lo:
            if not a.diag:
                raise SystemExit("--loss-hi != --loss-lo needs --diag per_image.npz for fragility")
            frag = np.load(a.diag)["fragility"][:N]
            rank = torch.from_numpy(frag.argsort().argsort() / max(N - 1, 1)).float()
        else:
            rank = torch.zeros(N)
        target_loss = a.loss_lo + (hi - a.loss_lo) * rank ** a.gamma
        if a.full_classes:
            full = torch.isin(y_all, torch.tensor(a.full_classes))
            target_loss[full] = a.full_loss
            print(f"{int(full.sum())} images in --full-classes get target loss {a.full_loss}")
        print(f"target loss per image: mean {target_loss.mean():.3f}, std {target_loss.std(unbiased=False):.3f}, "
              f"min {target_loss.min():.3f}, max {target_loss.max():.3f}")

    # trainable parameters
    if a.train_blocks >= 0:
        for p in student.parameters():
            p.requires_grad_(False)
        mods = list(student.blocks[len(student.blocks) - a.train_blocks:]) + [student.norm, student.head]
        if hasattr(student, "fc_norm"):
            mods.append(student.fc_norm)
        for m in mods:
            for p in m.parameters():
                p.requires_grad_(True)
    params = [p for p in student.parameters() if p.requires_grad]
    anchors = [p.detach().clone() for p in params] if a.l2 > 0 else None
    print(f"trainable params: {sum(p.numel() for p in params)/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)

    use_bf16 = DEVICE == "cuda" and torch.cuda.is_bf16_supported()
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16)  # noqa: E731

    proxy_pool = load_npz_images(a.proxy_npz) if a.proxy_npz else None
    def proxy_batch(n):
        if proxy_pool is not None:
            return proxy_pool[torch.randint(len(proxy_pool), (n,), generator=g)]
        return make_mosaics(imgs, n, g)

    # fixed evaluation sets
    ev_idx = torch.arange(min(a.eval_n, N))
    ev_imgs, ev_y = imgs[ev_idx], y_all[ev_idx]
    ev_alt = alter(ev_imgs, "mix", seed=a.seed + 99)
    ev_proxy = proxy_batch(a.eval_proxy_n) if proxy_pool is None else proxy_pool[:a.eval_proxy_n]
    ev_ref = {"exact": z_exact[ev_idx], "mix": predict(teacher, ev_alt), "proxy": predict(teacher, ev_proxy)}

    def evaluate(step):
        r = {"step": step,
             "exact": group_stats(predict(student, ev_imgs), ev_ref["exact"], ev_y),
             "mix": group_stats(predict(student, ev_alt), ev_ref["mix"], ev_y)}
        zp = predict(student, ev_proxy)
        r["proxy"] = {"kl_orig_sub_T2": group_stats(zp, ev_ref["proxy"], ev_ref["proxy"].argmax(1))["kl_orig_sub_T2"],
                      "retain_proxy_score": retain_proxy_score(zp, ev_ref["proxy"])}
        e, m = r["exact"], r["mix"]
        print(f"[eval {step}] exact loss {e['loss_mean']:.3f}±{e['loss_std']:.3f} acc {e['acc_vs_pseudo']:.3f} "
              f"natural {e.get('wrong_is_natural', float('nan')):.2f} | mix loss {m['loss_mean']:.3f}±"
              f"{m['loss_std']:.3f} acc {m['acc_vs_pseudo']:.3f} | proxy KL {r['proxy']['kl_orig_sub_T2']:.4f} "
              f"retain~{r['proxy']['retain_proxy_score']:.3f}", flush=True)
        with open(out / "eval.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
        student.train()

    steps_per_epoch = math.ceil(N / a.batch)
    total = a.max_steps or int(a.epochs * steps_per_epoch)
    evaluate(0)
    student.train()
    step, t0 = 0, time.time()
    while step < total:
        perm = torch.randperm(N, generator=g)
        for i in range(0, N, a.batch):
            if step >= total:
                break
            idx = perm[i:i + a.batch]
            xb, _ = random_alter(imgs[idx], a.aug_prob, g)
            x, y = preprocess(xb), y_all[idx].to(DEVICE)

            with amp():
                zs = student(x).float()
            if a.method == "ga":
                loss_f = -Fn.cross_entropy(zs, y)
            else:
                with torch.no_grad(), amp():
                    zt = teacher(x).float()
                if target_loss is not None:
                    delta = solve_delta(zt, y, target_loss[idx].to(DEVICE), a.tau)
                else:
                    delta = torch.full((len(y),), a.delta, device=DEVICE)
                zt = zt.clone()
                zt[torch.arange(len(y), device=DEVICE), y] -= delta
                q = Fn.softmax(zt / a.tau, dim=1)
                loss_f = (q * (torch.log(q + 1e-12) - Fn.log_softmax(zs, dim=1))).sum(1).mean()

            loss = loss_f
            loss_p = torch.zeros((), device=DEVICE)
            if a.proxy_weight > 0:
                xp = preprocess(proxy_batch(a.proxy_batch))
                with torch.no_grad(), amp():
                    zpt = teacher(xp).float()
                with amp():
                    zps = student(xp).float()
                pt = Fn.softmax(zpt / 2, 1)
                loss_p = (pt * (torch.log(pt + 1e-12) - Fn.log_softmax(zps / 2, 1))).sum(1).mean()
                loss = loss + a.proxy_weight * loss_p
            if anchors is not None:
                loss = loss + a.l2 * sum(((p - p0) ** 2).sum() for p, p0 in zip(params, anchors))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            if step % 10 == 0 or step == 1:
                print(f"step {step}/{total} loss_f {loss_f.item():.4f} loss_proxy {loss_p.item():.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if a.save_every and step % a.save_every == 0 and step < total:
                evaluate(step)
                save_submission(student, out / f"step{step}.pt")

    evaluate(step)
    save_submission(student, out / "final.pt")


if __name__ == "__main__":
    main()