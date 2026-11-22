"""
Step 5: pretrained ResNet-34 on the FULL-RESOLUTION image + our 11 step-3 channels.
Trained per method; used only for groups where it beats step 3 in 5-fold CV.
Other images keep their mask from submission_step4.pt.
    PYTHONPATH=.. python -u step5_fullres.py --data ../release --methods stegastamp --submit
"""
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wmloc.data import LocalisationDataset
from step2_shapes import METHODS, kind_thresholds, mask128
from step3_train import features, load_maps, loss_fn, miou_batch

ENCODER = "resnet34"
EPOCHS = 40
BATCH = 8
LR = 3e-4
FOLDS = 5
USE_RECT = ("pixelseal", "trustmark")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def conv(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Net(nn.Module):
    def __init__(self, cextra=11, pretrained=True):
        super().__init__()
        import timm
        try:
            self.enc = timm.create_model(ENCODER, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        except Exception as e:
            print(f"  [!] could not load pretrained weights ({e.__class__.__name__}); using random init")
            self.enc = timm.create_model(ENCODER, pretrained=False, features_only=True, out_indices=(1, 2, 3, 4))
        ch = self.enc.feature_info.channels()
        self.lat = nn.ModuleList([nn.Conv2d(c, 128, 1) for c in ch])
        self.fuse = nn.Sequential(conv(128 + cextra, 96), conv(96, 64), conv(64, 64))
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, img, extra):
        fs = self.enc((img - MEAN.to(img.device)) / STD.to(img.device))
        p = self.lat[3](fs[3])
        for k in (2, 1, 0):
            p = F.interpolate(p, size=fs[k].shape[-2:], mode="bilinear", align_corners=False) + self.lat[k](fs[k])
        return self.out(self.fuse(torch.cat([p, extra], 1)))[:, 0]


def tf(t, f):
    if f & 1:
        t = t.flip(-1)
    if f & 2:
        t = t.flip(-2)
    if f & 4:
        t = t.transpose(-1, -2)
    return t


def train(I, X, Y, idx, dev, epochs, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    net = Net(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    steps = epochs * ((len(idx) + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    for ep in range(epochs):
        net.train()
        perm = idx[torch.randperm(len(idx))]
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            f = random.randrange(8)
            img = tf(I[b].to(dev).float() / 255, f)
            xb, yb = tf(X[b].to(dev), f), tf(Y[b].to(dev), f)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                logit = net(img, xb)
            loss = loss_fn(logit.float(), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        if (ep + 1) % 10 == 0:
            print(f"    epoch {ep + 1}/{epochs}  loss {tot / len(idx):.3f}", flush=True)
    return net


@torch.no_grad()
def infer(net, I, X, idx, dev):
    net.eval()
    out = []
    for i in range(0, len(idx), 16):
        b = idx[i:i + 16]
        img, xb = I[b].to(dev).float() / 255, X[b].to(dev)
        p = 0
        for f in (0, 1, 2, 3):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                p = p + tf(torch.sigmoid(net(tf(img, f), tf(xb, f)).float()), f)
        out.append((p / 4).cpu())
    return torch.cat(out)


def is_box(m):
    if not m.any():
        return False
    ys, xs = torch.where(m)
    return m.sum().item() == (ys.max() - ys.min() + 1).item() * (xs.max() - xs.min() + 1).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="release")
    ap.add_argument("--methods", default="stegastamp")
    ap.add_argument("--base", default="submission_step4.pt")
    ap.add_argument("--out", default="submission_step5.pt")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(0)
    use = [m.strip().lower() for m in a.methods.split(",")]

    accs = dict(np.load("accs.npz", allow_pickle=True)["accs"].item())
    all_train = list(LocalisationDataset(a.data, "train"))
    thr = kind_thresholds(all_train, accs)
    tmaps = load_maps("train")
    train_items = [it for it in all_train if it["image_name"] in tmaps and str(it["method"]).lower() in use]
    meth = [str(it["method"]).lower() for it in train_items]
    kind = [int(accs[it["image_name"]] > thr[m]) for it, m in zip(train_items, meth)]
    to_img = lambda it: torch.from_numpy(np.array(it["image"])).permute(2, 0, 1)
    I = torch.stack([to_img(it) for it in train_items])
    X = torch.stack([features(it, tmaps, accs, thr) for it in train_items])
    Y = torch.stack([torch.from_numpy(mask128(it["mask128"])).float() for it in train_items])
    G = Y.bool()
    print(f"methods: {use}   train images: {len(train_items)}", flush=True)

    folds = torch.randperm(len(X)) % FOLDS
    oof = torch.full_like(Y, float("nan"))
    nets = []
    for f in range(1 if a.quick else FOLDS):
        tr, va = torch.where(folds != f)[0], torch.where(folds == f)[0]
        print(f"  fold {f + 1}/{FOLDS}: training on {len(tr)} images", flush=True)
        net = train(I, X, Y, tr, a.device, a.epochs, seed=f)
        oof[va] = infer(net, I, X, va, a.device)
        nets.append(net.cpu())
    done = [i for i in range(len(X)) if not torch.isnan(oof[i]).any()]

    d = torch.load("analysis_oof.pt")
    s3 = dict(zip(d["names"], d["oof"].float()))
    grid = np.arange(0.05, 0.96, 0.05)
    cut, better, gain = {}, set(), 0.0
    print("\n=== CROSS-VALIDATION (images the model never saw) ===")
    print(f"  {'group':<32}{'n':>4}{'step 3':>9}{'step 5':>9}   use")
    for m in use:
        for k in (1, 0):
            ids = [i for i in done if meth[i] == m and kind[i] == k]
            if not ids:
                continue
            sc5, t5 = max((miou_batch(oof[ids] > t, G[ids]).mean().item(), float(t)) for t in grid)
            P3 = torch.stack([s3[train_items[i]["image_name"]] for i in ids])
            sc3 = max(miou_batch(P3 > t, G[ids]).mean().item() for t in grid)
            cut[(m, k)] = t5
            ok = sc5 > sc3 + 0.01
            if ok:
                better.add((m, k))
                gain += (sc5 - sc3) * len(ids)
            label = f"{m} {'mostly wm' if k else 'mostly clean'}"
            print(f"  {label:<32}{len(ids):>4}{sc3:>9.3f}{sc5:>9.3f}   {'step 5' if ok else 'keep step 4'}", flush=True)
    gain = gain / len(done) * len(train_items) / len(all_train)
    print(f"\nexpected change on the overall score: +{gain:.3f}", flush=True)
    if a.quick:
        print("Quick check done (1 fold).")
        return

    sub = torch.load(a.base)
    names, masks = list(sub["image_names"]), sub["masks"].clone().bool()
    pos = {n: i for i, n in enumerate(names)}
    test_items = [it for it in LocalisationDataset(a.data, "test") if str(it["method"]).lower() in use]
    test_items = [it for it in test_items
                  if (str(it["method"]).lower(), int(accs[it["image_name"]] > thr[str(it["method"]).lower()])) in better]
    replaced = 0
    if test_items:
        smaps = load_maps("test")
        It = torch.stack([to_img(it) for it in test_items])
        Xt = torch.stack([features(it, smaps, accs, thr) for it in test_items])
        allidx = torch.arange(len(test_items))
        Pt = 0
        for n, net in enumerate(nets):
            Pt = Pt + infer(net.to(a.device), It, Xt, allidx, a.device) / len(nets)
            net.cpu()
            print(f"  test predictions {n + 1}/{len(nets)}", flush=True)
        for i, it in enumerate(test_items):
            m = str(it["method"]).lower()
            k = int(accs[it["image_name"]] > thr[m])
            j = pos[it["image_name"]]
            if is_box(masks[j]) and m in USE_RECT and k == 0:
                continue
            masks[j] = Pt[i] > cut[(m, k)]
            replaced += 1
    assert masks.shape == (1020, 128, 128) and len(set(names)) == len(names)
    torch.save({"image_names": names, "masks": masks.bool()}, a.out)
    print(f"wrote {a.out}: {len(names)} masks, {replaced} replaced by step 5 (rest from {a.base})")
    if a.submit:
        if gain > 0.005:
            import submission_template as st
            st.BASE_URL = "http://34.147.229.55"
            st.submit(a.out)
        else:
            print("NOT submitted: not enough gain on train. Your best stays on the leaderboard.")


if __name__ == "__main__":
    main()
