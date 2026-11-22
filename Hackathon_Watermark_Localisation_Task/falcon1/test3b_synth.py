"""
Test 3b: seamless holes (own clean pixels), REAL hole shapes (other methods' masks),
synthetic pre-training, then 5-fold fine-tuning on the real images. No submission.
  PYTHONPATH=.. python -u test3b_synth.py --data ../release
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi

import wmloc.watermarks as W
from wmloc.data import LocalisationDataset
from step2_shapes import kind_thresholds, mask128, bits_of
from step3_train import loss_fn, miou_batch
import step6_stega as S6
from test3_synth import load_covers, build_pool, u8, to_input

BATCH = 8
GRID = np.arange(0.05, 0.96, 0.05)
REF = {"trustmark": 0.486, "pixelseal": 0.509}


def tf_np(a, f):
    if f & 1:
        a = a[:, ::-1]
    if f & 2:
        a = a[::-1]
    if f & 4:
        a = a.T
    return np.ascontiguousarray(a)


def sample(i, C, Wm, rng, holes):
    H = tf_np(holes[int(rng.integers(len(holes)))], int(rng.integers(8)))
    al = ndi.gaussian_filter(H.astype(np.float32), rng.uniform(0.7, 1.5))[..., None]
    img = Wm[i].astype(np.float32) * (1 - al) + C[i].astype(np.float32) * al
    img = np.clip(np.round(img), 0, 255).astype(np.uint8)
    y = H.reshape(128, 4, 128, 4).mean((1, 3)) < 0.5
    return img, y


def best_score(P, Y):
    return max((miou_batch(P > t, Y.bool()).mean().item(), float(t)) for t in GRID)


def predict(net, I, idx, dev):
    return S6.infer(net, I, torch.zeros(len(I), 1, 128, 128), idx, dev)


def train_loop(net, batches, steps, lr, dev, log_every=0, evalf=None):
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    extra = torch.zeros(BATCH, 1, 128, 128, device=dev)
    tot, t0 = 0.0, time.time()
    for s in range(steps):
        net.train()
        img, y = batches()
        f = random.randrange(8)
        img, y = S6.tf(img.to(dev).float() / 255, f), S6.tf(y.to(dev).float(), f)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
            logit = net(img, extra[:len(img)])
        loss = loss_fn(logit.float(), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        tot += loss.item()
        if log_every and (s + 1) % log_every == 0:
            msg = evalf() if evalf else ""
            print(f"  epoch {(s + 1) // log_every:2d}/{steps // log_every}  loss {tot / log_every:.3f}   {msg}"
                  f"({time.time() - t0:.0f}s)", flush=True)
            tot, t0 = 0.0, time.time()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../release")
    ap.add_argument("--method", default="trustmark")
    ap.add_argument("--covers", default="")
    ap.add_argument("--pool", type=int, default=3000)
    ap.add_argument("--input", default="hp", choices=["rgb", "hp", "both"])
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ft_epochs", type=int, default=20)
    ap.add_argument("--ft_lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev, meth = a.device, a.method
    S6.ENCODER, S6.IN_CH = a.encoder, (6 if a.input == "both" else 3)
    random.seed(0)
    torch.manual_seed(0)

    accs = dict(np.load("accs.npz", allow_pickle=True)["accs"].item())
    all_train = list(LocalisationDataset(a.data, "train"))
    thr = kind_thresholds(all_train, accs)
    mostly_wm = lambda it: accs[it["image_name"]] > thr[str(it["method"]).lower()]
    real = [it for it in all_train if str(it["method"]).lower() == meth and mostly_wm(it)]
    R_y = np.stack([mask128(it["mask128"]).astype(bool) for it in real])
    print(f"real group: {meth} mostly watermarked, n={len(real)}, watermarked share {R_y.mean():.2f}", flush=True)
    RI = torch.stack([to_input(u8(it["image"]), a.input) for it in real])
    RY = torch.from_numpy(R_y).float()

    holes = []
    for it in all_train:
        if str(it["method"]).lower() != meth and mostly_wm(it):
            m = mask128(it["mask128"]).astype(bool)
            holes.append(np.repeat(np.repeat(~m, 4, 0), 4, 1))
    print(f"hole shapes: {len(holes)} real masks (other methods), hole share {np.mean([h.mean() for h in holes]):.2f}",
          flush=True)

    cached = (Path(f"cache_synth_{meth}") / "done.txt").exists()
    if cached:
        wm, covers = None, []
    else:
        wm, covers = W._CLASSES[meth](Path("models").resolve(), dev), load_covers(a, meth)
    C, Wm, msgs, tr, val = build_pool(a, wm, covers, len(bits_of(real[0]["message"])))
    del covers, wm

    rng = np.random.default_rng(11)
    vs = [sample(int(i), C, Wm, rng, holes) for i in rng.choice(val, 300, replace=len(val) < 300)]
    VI = torch.stack([to_input(s[0], a.input) for s in vs])
    VY = torch.from_numpy(np.stack([s[1] for s in vs])).float()
    del vs
    real_all = torch.arange(len(RI))

    rng = np.random.default_rng(3)

    def syn_batch():
        bs = [sample(int(i), C, Wm, rng, holes) for i in rng.choice(tr, BATCH)]
        return (torch.stack([to_input(s[0], a.input) for s in bs]),
                torch.from_numpy(np.stack([s[1] for s in bs])))

    def evalf():
        ss = best_score(predict(net, VI, torch.arange(len(VI)), dev), VY)[0]
        rs = best_score(predict(net, RI, real_all, dev), RY)[0]
        return f"synthetic {ss:.3f}   REAL (no fine-tuning) {rs:.3f}   "

    print(f"\n=== PHASE 1: synthetic pre-training ({a.epochs} epochs x {a.steps * BATCH} images, input {a.input}) ===")
    print(f"  0.448 = 'everything watermarked'; score to beat: {REF.get(meth, float('nan')):.3f}", flush=True)
    import os
    ckpt = f"test3b_{meth}_pretrained.pt"
    if os.path.exists(ckpt):
        print(f"  reusing {ckpt} (skipping phase 1)", flush=True)
        pre = torch.load(ckpt, map_location="cpu")
    else:
        net = S6.Net(1).to(dev)
        net = train_loop(net, syn_batch, a.epochs * a.steps, a.lr, dev, log_every=a.steps, evalf=evalf)
        pre = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        torch.save(pre, ckpt)

    print(f"\n=== PHASE 2: fine-tuning on real images, 5-fold ({a.ft_epochs} epochs, lr {a.ft_lr}) ===", flush=True)
    folds = torch.randperm(len(RI), generator=torch.Generator().manual_seed(0)) % 5
    oof = torch.zeros_like(RY)
    for f in range(5):
        tr_i, va_i = torch.where(folds != f)[0], torch.where(folds == f)[0]
        net = S6.Net(1).to(dev)
        net.load_state_dict(pre)
        g = np.random.default_rng(100 + f)

        def real_batch():
            b = torch.from_numpy(g.choice(tr_i.numpy(), BATCH))
            return RI[b], RY[b]

        steps = a.ft_epochs * max(1, len(tr_i) // BATCH)
        net = train_loop(net, real_batch, steps, a.ft_lr, dev)
        oof[va_i] = predict(net, RI, va_i, dev)
        try:
            torch.save(net.state_dict(), f"test3b_{meth}_fold{f}.pt")
        except RuntimeError as e:
            print(f"  (fold {f} checkpoint not saved: {e})", flush=True)
        print(f"  fold {f + 1}/5 done", flush=True)
    sc, t = best_score(oof, RY)
    ref = REF.get(meth, float("nan"))
    print(f"\n=== RESULT ({meth} mostly watermarked, n={len(RI)}) ===")
    print(f"  current CV {ref:.3f}   ->   synthetic pre-training + real fine-tuning: {sc:.3f}  (threshold {t:.2f})")
    print(f"  {'BETTER: worth submitting' if sc > ref + 0.01 else 'not better than current'}. Paste me this.")


if __name__ == "__main__":
    main()
