"""
Step 3: TRAIN a small neural network (U-Net) that draws the 128x128 mask from the image + step-2 maps.
    PYTHONPATH=.. python -u step3_train.py --data ../release --submit
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wmloc.data import LocalisationDataset
from step2_shapes import METHODS, kind_thresholds, mask128, znorm

EPOCHS = 60
BATCH = 16
LR = 2e-3
WIDTH = 32
FOLDS = 5
SEEDS_FINAL = 3


def load_maps(split):
    maps = {}
    for m in METHODS:
        p = os.path.join("cache_step2", f"{split}_{m}.pt")
        if os.path.exists(p):
            maps.update(torch.load(p))
    return maps


def features(it, maps, accs, thr):
    name, meth = it["image_name"], str(it["method"]).lower()
    d = maps[name]
    img = torch.from_numpy(np.asarray(it["image"]).astype(np.float32) / 255).permute(2, 0, 1)
    img = F.interpolate(img[None], size=(128, 128), mode="area")[0]
    rise = znorm(d["rise"].float())
    has_feat = d["feat"] is not None
    feat = znorm(d["feat"].float()) if has_feat else torch.zeros(128, 128)
    acc = accs[name]
    kind = float(acc > thr[meth])
    const = lambda v: torch.full((1, 128, 128), float(v))
    onehot = torch.zeros(3, 128, 128)
    onehot[METHODS.index(meth)] = 1
    return torch.cat([img, feat[None], rise[None], const(has_feat), const(kind), const((acc - 0.5) * 2), onehot])


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, cin=11, w=WIDTH):
        super().__init__()
        self.e1, self.e2, self.e3, self.b = block(cin, w), block(w, 2 * w), block(2 * w, 4 * w), block(4 * w, 4 * w)
        self.d3, self.d2, self.d1 = block(8 * w, 2 * w), block(4 * w, w), block(2 * w, w)
        self.out = nn.Conv2d(w, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.b(F.max_pool2d(e3, 2))
        up = lambda t: F.interpolate(t, scale_factor=2, mode="nearest")
        d3 = self.d3(torch.cat([up(b), e3], 1))
        d2 = self.d2(torch.cat([up(d3), e2], 1))
        d1 = self.d1(torch.cat([up(d2), e1], 1))
        return self.out(d1)[:, 0]


def loss_fn(logit, y):
    p = torch.sigmoid(logit)
    iou1 = (p * y).sum((1, 2)) / (p + y - p * y).sum((1, 2)).clamp(min=1e-6)
    q, z = 1 - p, 1 - y
    iou0 = (q * z).sum((1, 2)) / (q + z - q * z).sum((1, 2)).clamp(min=1e-6)
    return F.binary_cross_entropy_with_logits(logit, y) + (1 - (iou0 + iou1) / 2).mean()


def augment(x, y):
    if random.random() < 0.5:
        x, y = x.flip(-1), y.flip(-1)
    if random.random() < 0.5:
        x, y = x.flip(-2), y.flip(-2)
    if random.random() < 0.5:
        x, y = x.transpose(-1, -2), y.transpose(-1, -2)
    return x, y


def train(X, Y, dev, epochs):
    net = UNet(cin=X.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    steps = epochs * ((len(X) + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = augment(X[idx].to(dev), Y[idx].to(dev))
            loss = loss_fn(net(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return net


@torch.no_grad()
def infer(net, X, dev):
    net.eval()
    out = []
    for i in range(0, len(X), 32):
        xb = X[i:i + 32].to(dev)
        p = torch.sigmoid(net(xb)) + torch.sigmoid(net(xb.flip(-1))).flip(-1) + torch.sigmoid(net(xb.flip(-2))).flip(-2)
        out.append((p / 3).cpu())
    return torch.cat(out)


def miou_batch(P, G):
    i1, u1 = (P & G).sum((1, 2)).float(), (P | G).sum((1, 2)).float()
    i0, u0 = (~P & ~G).sum((1, 2)).float(), (~P | ~G).sum((1, 2)).float()
    iou1 = torch.where(u1 > 0, i1 / u1.clamp(min=1), torch.ones_like(u1))
    iou0 = torch.where(u0 > 0, i0 / u0.clamp(min=1), torch.ones_like(u0))
    return (iou0 + iou1) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="release")
    ap.add_argument("--out", default="submission_step3.pt")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)

    accs = dict(np.load("accs.npz", allow_pickle=True)["accs"].item())
    train_items = [it for it in LocalisationDataset(a.data, "train")]
    thr = kind_thresholds(train_items, accs)
    tmaps = load_maps("train")
    train_items = [it for it in train_items if it["image_name"] in tmaps]
    print(f"train images with maps: {len(train_items)}")
    X = torch.stack([features(it, tmaps, accs, thr) for it in train_items])
    Y = torch.stack([torch.from_numpy(mask128(it["mask128"])).float() for it in train_items])
    meth = [str(it["method"]).lower() for it in train_items]
    kind = [int(accs[it["image_name"]] > thr[m]) for it, m in zip(train_items, meth)]

    folds = (torch.randperm(len(X)) % FOLDS)
    oof = torch.zeros_like(Y)
    for f in range(FOLDS):
        tr, va = folds != f, folds == f
        net = train(X[tr], Y[tr], a.device, a.epochs)
        oof[va] = infer(net, X[va], a.device)
        print(f"  fold {f + 1}/{FOLDS} done", flush=True)

    G = Y.bool()
    cut = {}
    print("\n=== CROSS-VALIDATION (images the model never saw) ===")
    for m in METHODS:
        for k in (1, 0):
            ids = [i for i in range(len(X)) if meth[i] == m and kind[i] == k]
            if not ids:
                continue
            best = max((miou_batch(oof[ids] > t, G[ids]).mean().item(), t) for t in np.arange(0.1, 0.91, 0.05))
            cut[(m, k)] = float(best[1])
            label = "mostly watermarked" if k else "mostly clean      "
            print(f"  {m:<10} {label} ({len(ids):>3} imgs): {best[0]:.3f}  (cut-off {best[1]:.2f})")
    P = torch.stack([oof[i] > cut[(meth[i], kind[i])] for i in range(len(X))])
    print(f"\nCV score: {miou_batch(P, G).mean().item():.3f}   (step 2 was 0.474 on train, 0.458 on the leaderboard)")

    smaps = load_maps("test")
    test_items = [it for it in LocalisationDataset(a.data, "test")]
    missing = [it["image_name"] for it in test_items if it["image_name"] not in smaps]
    if missing:
        print(f"\n{len(missing)} test images have no maps yet. No submission written.")
        return
    Xt = torch.stack([features(it, smaps, accs, thr) for it in test_items])
    Pt = torch.zeros(len(test_items), 128, 128)
    for s in range(SEEDS_FINAL):
        torch.manual_seed(100 + s)
        Pt += infer(train(X, Y, a.device, a.epochs), Xt, a.device) / SEEDS_FINAL
        print(f"  final model {s + 1}/{SEEDS_FINAL} done", flush=True)
    names, masks = [], []
    for i, it in enumerate(test_items):
        m = str(it["method"]).lower()
        k = int(accs[it["image_name"]] > thr[m])
        names.append(it["image_name"])
        masks.append(Pt[i] > cut[(m, k)])
    torch.save({"image_names": names, "masks": torch.stack(masks).bool()}, a.out)
    print(f"wrote {a.out}: {len(names)} masks")
    if a.submit:
        import submission_template as st
        st.BASE_URL = "http://34.147.229.55"
        st.submit(a.out)


if __name__ == "__main__":
    main()
