"""
Test 3c: like test 3b, but the encoder is the official TrustMark DECODER (ResNet-50),
which already sees the watermark. FPN head on its layer1-4 maps -> 128x128 mask.
Phase 1: synthetic pre-training. Phase 2: 5-fold fine-tuning on real. No submission.
  PYTHONPATH=.. python -u test3c_dec.py --data ../release
"""
import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import wmloc.watermarks as W
from wmloc.data import LocalisationDataset
from step2_shapes import kind_thresholds, mask128, bits_of, to_tensor
from step3_train import loss_fn
import step6_stega as S6
from test3_synth import build_pool, u8, to_input
from test3b_synth import sample, best_score

BATCH = 8
REF = {"trustmark": 0.486}


def raw_logits(wm, x):
    fn = getattr(type(wm).logits, "__wrapped__", None)
    return fn(wm, x) if fn is not None else wm.logits(x)


def measure_preproc(wm, img_u8, dev):
    cap = {}
    h = wm.net.conv1.register_forward_pre_hook(lambda m, inp: cap.__setitem__("x", inp[0].detach().float()))
    x = to_tensor(img_u8)[None].to(dev).float()
    with torch.no_grad():
        raw_logits(wm, x)
    h.remove()
    xin = cap["x"][:1]
    size = tuple(xin.shape[-2:])
    best = None
    for perm in ([0, 1, 2], [2, 1, 0]):
        for mode in ("bilinear", "bicubic", "area", "nearest"):
            for aa in (False, True):
                if aa and mode not in ("bilinear", "bicubic"):
                    continue
                kw = {} if mode in ("area", "nearest") else {"align_corners": False, "antialias": aa}
                r = F.interpolate(x[:, perm], size=size, mode=mode, **kw)
                a, b, res = [], [], 0.0
                for c in range(3):
                    A = torch.stack([r[0, c].flatten(), torch.ones(r[0, c].numel(), device=dev)], 1)
                    sol = torch.linalg.lstsq(A, xin[0, c].flatten()[:, None]).solution[:, 0]
                    a.append(sol[0].item())
                    b.append(sol[1].item())
                    res += (A @ sol - xin[0, c].flatten()).abs().mean().item() / 3
                if best is None or res < best[0]:
                    best = (res, perm, mode, aa, a, b)
    res, perm, mode, aa, a, b = best
    print(f"decoder input: {size}, resize '{mode}' antialias={aa}, channels {perm}, "
          f"scale {[round(v, 3) for v in a]}, offset {[round(v, 3) for v in b]}, fit error {res:.4f}", flush=True)
    if res > 0.02:
        print("  [!] fit error is high: the preprocessing guess may be off", flush=True)
    return size, perm, mode, aa, a, b


class DecNet(nn.Module):
    def __init__(self, backbone, pre):
        super().__init__()
        size, perm, mode, aa, a, b = pre
        self.size, self.perm, self.mode, self.aa = size, perm, mode, aa
        self.register_buffer("a", torch.tensor(a).view(1, 3, 1, 1))
        self.register_buffer("b", torch.tensor(b).view(1, 3, 1, 1))
        self.bb = copy.deepcopy(backbone).float()
        for p in self.bb.parameters():
            p.requires_grad_(True)
        with torch.no_grad():
            dev0 = next(self.bb.parameters()).device
            ch = [f.shape[1] for f in self.feats(torch.zeros(1, 3, *size, device=dev0))]
        self.lat = nn.ModuleList([nn.Conv2d(c, 128, 1) for c in ch])
        self.fuse = nn.Sequential(S6.conv(128, 96), S6.conv(96, 64), S6.conv(64, 64))
        self.out = nn.Conv2d(64, 1, 1)

    def feats(self, x):
        bb = self.bb
        h = bb.maxpool(bb.relu(bb.bn1(bb.conv1(x))))
        f1 = bb.layer1(h)
        f2 = bb.layer2(f1)
        f3 = bb.layer3(f2)
        return [f1, f2, f3, bb.layer4(f3)]

    def train(self, mode=True):
        super().train(mode)
        for m in self.bb.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, img, extra):
        kw = {} if self.mode in ("area", "nearest") else {"align_corners": False, "antialias": self.aa}
        x = F.interpolate(img[:, self.perm].float(), size=self.size, mode=self.mode, **kw)
        fs = self.feats(x * self.a + self.b)
        p = self.lat[3](fs[3])
        for k in (2, 1, 0):
            p = F.interpolate(p, size=fs[k].shape[-2:], mode="bilinear", align_corners=False) + self.lat[k](fs[k])
        p = F.interpolate(p, size=(128, 128), mode="bilinear", align_corners=False)
        return self.out(self.fuse(p))[:, 0]


def predict(net, I, idx, dev):
    return S6.infer(net, I, torch.zeros(len(I), 1, 128, 128), idx, dev)


def train_loop(net, batches, steps, lr, bb_mult, dev, log_every=0, evalf=None):
    bb = [p for n, p in net.named_parameters() if n.startswith("bb.")]
    head = [p for n, p in net.named_parameters() if not n.startswith("bb.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": lr * bb_mult}, {"params": head, "lr": lr}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr * bb_mult, lr], total_steps=steps, pct_start=0.1)
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
    ap.add_argument("--pool", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bb_mult", type=float, default=0.3)
    ap.add_argument("--ft_epochs", type=int, default=20)
    ap.add_argument("--ft_lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev, meth = a.device, a.method
    random.seed(0)
    torch.manual_seed(0)

    accs = dict(np.load("accs.npz", allow_pickle=True)["accs"].item())
    all_train = list(LocalisationDataset(a.data, "train"))
    thr = kind_thresholds(all_train, accs)
    mostly_wm = lambda it: accs[it["image_name"]] > thr[str(it["method"]).lower()]
    real = [it for it in all_train if str(it["method"]).lower() == meth and mostly_wm(it)]
    R_img = [u8(it["image"]) for it in real]
    RY = torch.from_numpy(np.stack([mask128(it["mask128"]).astype(bool) for it in real])).float()
    RI = torch.stack([to_input(im, "rgb") for im in R_img])
    print(f"real group: {meth} mostly watermarked, n={len(real)}", flush=True)

    holes = []
    for it in all_train:
        if str(it["method"]).lower() != meth and mostly_wm(it):
            m = mask128(it["mask128"]).astype(bool)
            holes.append(np.repeat(np.repeat(~m, 4, 0), 4, 1))
    print(f"hole shapes: {len(holes)} real masks (other methods)", flush=True)

    wm = W._CLASSES[meth](Path("models").resolve(), dev)
    pre = measure_preproc(wm, R_img[0], dev)
    if not (Path(f"cache_synth_{meth}") / "done.txt").exists():
        raise SystemExit("synthetic pool missing: run test3_synth.py --check first")
    C, Wm, msgs, tr, val = build_pool(a, None, [], len(bits_of(real[0]["message"])))

    rng = np.random.default_rng(11)
    vs = [sample(int(i), C, Wm, rng, holes) for i in rng.choice(val, 300, replace=len(val) < 300)]
    VI = torch.stack([to_input(s[0], "rgb") for s in vs])
    VY = torch.from_numpy(np.stack([s[1] for s in vs])).float()
    del vs

    rng = np.random.default_rng(3)

    def syn_batch():
        bs = [sample(int(i), C, Wm, rng, holes) for i in rng.choice(tr, BATCH)]
        return (torch.stack([to_input(s[0], "rgb") for s in bs]),
                torch.from_numpy(np.stack([s[1] for s in bs])))

    net = DecNet(wm.net, pre).to(dev)
    del wm

    def evalf():
        ss = best_score(predict(net, VI, torch.arange(len(VI)), dev), VY)[0]
        rs = best_score(predict(net, RI, torch.arange(len(RI)), dev), RY)[0]
        return f"synthetic {ss:.3f}   REAL (no fine-tuning) {rs:.3f}   "

    print(f"\n=== PHASE 1: synthetic pre-training, encoder = TrustMark decoder ({a.epochs} epochs) ===")
    print(f"  0.448 = 'everything watermarked'; test 3b reached 0.494; score to beat: {REF.get(meth, 0):.3f}", flush=True)
    net = train_loop(net, syn_batch, a.epochs * a.steps, a.lr, a.bb_mult, dev, log_every=a.steps, evalf=evalf)
    pre_sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    torch.save(pre_sd, f"test3c_{meth}_pretrained.pt")

    print(f"\n=== PHASE 2: fine-tuning on real images, 5-fold ({a.ft_epochs} epochs) ===", flush=True)
    folds = torch.randperm(len(RI), generator=torch.Generator().manual_seed(0)) % 5
    oof = torch.zeros_like(RY)
    for f in range(5):
        tr_i, va_i = torch.where(folds != f)[0], torch.where(folds == f)[0]
        net.load_state_dict(pre_sd)
        g = np.random.default_rng(100 + f)

        def real_batch():
            b = torch.from_numpy(g.choice(tr_i.numpy(), BATCH))
            return RI[b], RY[b]

        steps = a.ft_epochs * max(1, len(tr_i) // BATCH)
        train_loop(net, real_batch, steps, a.ft_lr, a.bb_mult, dev)
        oof[va_i] = predict(net, RI, va_i, dev)
        torch.save(net.state_dict(), f"test3c_{meth}_fold{f}.pt")
        print(f"  fold {f + 1}/5 done", flush=True)
    sc, t = best_score(oof, RY)
    ref = REF.get(meth, 0)
    print(f"\n=== RESULT ({meth} mostly watermarked, n={len(RI)}) ===")
    print(f"  current CV {ref:.3f}   ->   decoder-initialised, synthetic + real fine-tuning: {sc:.3f}  (threshold {t:.2f})")
    print(f"  {'BETTER: worth submitting' if sc > ref + 0.01 else 'not better than current'}. Paste me this.")


if __name__ == "__main__":
    main()
