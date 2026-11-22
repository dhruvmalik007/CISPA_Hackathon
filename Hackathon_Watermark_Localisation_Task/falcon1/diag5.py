"""
Diagnostics round 5 (no submission): what does the full-resolution network use,
and can it work on PixelSeal / TrustMark? 2-fold CV per method and input variant.
    PYTHONPATH=.. python -u diag5.py --data ../release --methods pixelseal,trustmark,stegastamp
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from wmloc.data import LocalisationDataset
from step2_shapes import kind_thresholds, mask128
from step3_train import features, load_maps, miou_batch
from step5_fullres import train, infer

EPOCHS = 25
FOLDS = 2


def highpass_u8(img_u8):
    x = img_u8.float()[None] / 255
    b = F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), 5, 1)
    return ((0.5 + 8 * (x - b)).clamp(0, 1) * 255).round().to(torch.uint8)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="release")
    ap.add_argument("--methods", default="pixelseal,trustmark")
    ap.add_argument("--variants", default="rgb,hp,rgb_nomap")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    accs = dict(np.load("accs.npz", allow_pickle=True)["accs"].item())
    all_train = list(LocalisationDataset(a.data, "train"))
    thr = kind_thresholds(all_train, accs)
    tmaps = load_maps("train")
    d = torch.load("analysis_oof.pt")
    s3 = dict(zip(d["names"], d["oof"].float()))
    grid = np.arange(0.05, 0.96, 0.05)

    def best(P, G):
        return max(miou_batch(P > t, G).mean().item() for t in grid)

    rows = []
    for m in [x.strip().lower() for x in a.methods.split(",")]:
        items = [it for it in all_train if it["image_name"] in tmaps and str(it["method"]).lower() == m]
        kind = torch.tensor([int(accs[it["image_name"]] > thr[m]) for it in items])
        RGB = torch.stack([torch.from_numpy(np.array(it["image"])).permute(2, 0, 1) for it in items])
        X = torch.stack([features(it, tmaps, accs, thr) for it in items])
        Y = torch.stack([torch.from_numpy(mask128(it["mask128"])).float() for it in items])
        G = Y.bool()
        P3 = torch.stack([s3[it["image_name"]] for it in items])
        torch.manual_seed(0)
        folds = torch.randperm(len(items)) % FOLDS
        for v in [x.strip() for x in a.variants.split(",")]:
            t0 = time.time()
            I = torch.stack([highpass_u8(im) for im in RGB]) if v == "hp" else RGB
            Xv = X.clone()
            if v == "rgb_nomap":
                Xv[:, 3:6] = 0
            oof = torch.zeros_like(Y)
            for f in range(FOLDS):
                tr, va = torch.where(folds != f)[0], torch.where(folds == f)[0]
                net = train(I, Xv, Y, tr, a.device, a.epochs, seed=f)
                oof[va] = infer(net, I, Xv, va, a.device)
                del net
                torch.cuda.empty_cache()
            for k, kname in ((1, "mostly wm"), (0, "mostly clean")):
                ids = torch.where(kind == k)[0]
                rows.append((m, kname, v, len(ids), best(P3[ids], G[ids]), best(oof[ids], G[ids])))
                print(f"  {m:<10} {kname:<13} {v:<10} n={len(ids):>3}  step3 {rows[-1][4]:.3f}  "
                      f"new {rows[-1][5]:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    print("\n=== SUMMARY (2-fold CV, every image scored by a model that never saw it) ===")
    print(f"{'method':<11}{'group':<14}{'variant':<11}{'n':>4}{'step 3':>9}{'new':>8}{'gain':>8}")
    for m, kname, v, n, s3v, snew in rows:
        print(f"{m:<11}{kname:<14}{v:<11}{n:>4}{s3v:>9.3f}{snew:>8.3f}{snew - s3v:>+8.3f}")


if __name__ == "__main__":
    main()
