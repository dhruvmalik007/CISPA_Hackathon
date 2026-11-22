"""
Step 1: decide per image "mostly watermarked" (predict all ones) vs "mostly clean" (predict all zeros)
from the bit accuracy of decoding the FULL image with the known message.

    PYTHONPATH=.. python step1_kind.py --data ../release
Writes: accs.npz (cached bit accuracies) and submission_step1.pt
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from wmloc.data import LocalisationDataset
from wmloc.utils import score
import wmloc.watermarks as W

METHODS = ["pixelseal", "trustmark", "stegastamp"]
BATCH = 16          # lower to 4 if you run out of memory


def bits_of(msg):
    if isinstance(msg, str):
        return np.array([int(c) for c in msg if c in "01"])
    if torch.is_tensor(msg):
        msg = msg.cpu().numpy()
    return np.asarray(msg).flatten().astype(int)


def to_tensor(img):
    x = img.cpu().numpy() if torch.is_tensor(img) else np.asarray(img)
    x = torch.from_numpy(x.astype(np.float32))
    if x.shape[-1] == 3:
        x = x.permute(2, 0, 1)
    if x.max() > 1.5:
        x = x / 255.0
    return x


def mask128(m):
    m = m.cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
    return (m.squeeze() > 0.5).astype(np.uint8)


def decode_accs(items, device):
    """Full-image bit accuracy for every item. Returns dict image_name -> acc."""
    out = {}
    for meth in METHODS:
        its = [it for it in items if str(it["method"]).lower() == meth]
        if not its:
            continue
        wm = W._CLASSES[meth](Path("models").resolve(), device)
        for i in range(0, len(its), BATCH):
            chunk = its[i:i + BATCH]
            x = torch.stack([to_tensor(it["image"]) for it in chunk]).to(device)
            with torch.no_grad():
                d = wm.decode(x)
            d = (d.cpu().numpy() if torch.is_tensor(d) else np.asarray(d)).reshape(len(chunk), -1)
            for it, row in zip(chunk, d):
                b = bits_of(it["message"])
                L = min(len(b), len(row))
                out[it["image_name"]] = float(((row[:L] > 0.5).astype(int) == b[:L]).mean())
            print(f"  {meth}: {min(i + BATCH, len(its))}/{len(its)}", end="\r", flush=True)
        print()
        del wm
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="release")
    ap.add_argument("--out", default="submission_step1.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    train = [it for it in LocalisationDataset(a.data, "train")]
    test = [it for it in LocalisationDataset(a.data, "test")]

    cache = "accs.npz"
    if os.path.exists(cache):
        accs = dict(np.load(cache, allow_pickle=True)["accs"].item())
        print(f"loaded cached accuracies for {len(accs)} images")
    else:
        accs = {}
    todo = [it for it in train + test if it["image_name"] not in accs]
    if todo:
        print(f"decoding {len(todo)} images on {a.device} ...")
        accs.update(decode_accs(todo, a.device))
        np.savez(cache, accs=np.array(accs, dtype=object))

    print("\n=== TRAIN: full-image bit accuracy by type ===")
    thr = {}
    for meth in METHODS:
        its = [it for it in train if str(it["method"]).lower() == meth]
        acc = np.array([accs[it["image_name"]] for it in its])
        flag = np.array([bool(np.asarray(it["full_watermark"])) for it in its])
        gts = [mask128(it["mask128"]) for it in its]
        print(f"\n--- {meth}")
        print(f"  True  (mostly watermarked): mean {acc[flag].mean():.3f}  min {acc[flag].min():.3f}  max {acc[flag].max():.3f}")
        print(f"  False (mostly clean)      : mean {acc[~flag].mean():.3f}  min {acc[~flag].min():.3f}  max {acc[~flag].max():.3f}")
        best = (-1, None)
        for t in np.arange(0.50, 1.001, 0.01):
            preds = [np.ones_like(g) if a_ > t else np.zeros_like(g) for a_, g in zip(acc, gts)]
            s = score(preds, gts)
            if s > best[0]:
                best = (s, t)
        oracle = score([np.ones_like(g) if f else np.zeros_like(g) for f, g in zip(flag, gts)], gts)
        correct = ((acc > best[1]) == flag).mean()
        thr[meth] = best[1]
        print(f"  best threshold {best[1]:.2f}: classifies {correct:.0%} correctly, train score {best[0]:.3f}"
              f"   (if classification were perfect: {oracle:.3f})")

    allp = [np.ones_like(mask128(it["mask128"])) if accs[it["image_name"]] > thr[str(it["method"]).lower()]
            else np.zeros_like(mask128(it["mask128"])) for it in train]
    print(f"\nTRAIN overall score with this rule: {score(allp, [mask128(it['mask128']) for it in train]):.3f}  (baseline was 0.252)")

    names, masks = [], []
    for it in test:
        on = accs[it["image_name"]] > thr[str(it["method"]).lower()]
        names.append(it["image_name"])
        masks.append(np.full((128, 128), 1 if on else 0, np.uint8))
    torch.save({"image_names": names, "masks": torch.from_numpy(np.stack(masks)).bool()}, a.out)
    print(f"wrote {a.out}: {len(names)} masks, {sum(m[0, 0] for m in masks)} predicted 'mostly watermarked'")


if __name__ == "__main__":
    main()