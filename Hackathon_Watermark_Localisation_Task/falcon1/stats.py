"""
Explore the TRAIN split before building anything.

    PYTHONPATH=.. python stats.py --data ../release            # numbers only
    PYTHONPATH=.. python stats.py --data ../release --viz 4    # + 4 pictures per method in stats_viz/
    PYTHONPATH=.. python stats.py --data ../release --decode   # + bit accuracy tests (better on a GPU node)

Part A  how big is the watermarked area, how often it is empty/full, and simple baseline scores
Part B  does decoding the right crop give the message back? (the key idea of the method)
Part C  pictures: image | true mask | full_watermark (original rectangle)
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from wmloc.data import LocalisationDataset
from wmloc.utils import score
import wmloc.watermarks as W

METHODS = ["pixelseal", "trustmark", "stegastamp"]


def arr(x):
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    return np.asarray(x)


def mask128(x):
    """Any mask (512 or 128) -> 128x128 uint8."""
    m = arr(x).squeeze().astype(np.float32)
    if m.ndim == 3:
        m = m[..., 0] if m.shape[-1] in (1, 3) else m[0]
    if m.shape[0] == 512:
        m = m.reshape(128, 4, 128, 4).mean((1, 3))
    return (m >= 0.5).astype(np.uint8)


def bbox_mask(m):
    out = np.zeros_like(m)
    ys, xs = np.nonzero(m)
    if len(ys):
        out[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 1
    return out


def bits_of(msg):
    if isinstance(msg, str):
        return np.array([int(c) for c in msg if c in "01"])
    return arr(msg).flatten().astype(int)


def to_tensor(img):
    x = torch.from_numpy(arr(img).astype(np.float32))
    if x.shape[-1] == 3:
        x = x.permute(2, 0, 1)
    if x.max() > 1.5:
        x = x / 255.0
    return x


def pct(v):
    v = np.asarray(v)
    return f"mean {v.mean():.2f} | median {np.median(v):.2f} | min {v.min():.2f} | max {v.max():.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="release")
    ap.add_argument("--viz", type=int, default=0)
    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    train = LocalisationDataset(a.data, "train")
    items = [train[i] for i in range(len(train))]
    print(f"train images: {len(items)}   keys: {list(items[0].keys())}\n")

    # ------------------------------------------------------------------ Part A
    print("=" * 30, "PART A: masks and simple baselines", "=" * 30)
    for meth in METHODS:
        its = [it for it in items if str(it["method"]).lower() == meth]
        if not its:
            continue
        gts = [mask128(it["mask128"]) for it in its]
        frac = [g.mean() for g in gts]
        fw = [arr(it["full_watermark"]).item() for it in its]
        for v in sorted(set(fw)):
            sh = [f for f, x in zip(frac, fw) if x == v]
            print(f"  full_watermark = {v!s:<6}: {len(sh)} images, watermarked share mean {np.mean(sh):.2f}")
        has_full = False
        print(f"\n--- {meth} ({len(its)} images)")
        print(f"  watermarked share of image : {pct(frac)}")
        print(f"  images with NO watermark   : {sum(f == 0 for f in frac)}   fully watermarked: {sum(f == 1 for f in frac)}")
        sc = {
            "all zeros": score([np.zeros_like(g) for g in gts], gts),
            "all ones": score([np.ones_like(g) for g in gts], gts),
            "ORACLE bbox of true mask": score([bbox_mask(g) for g in gts], gts),
        }
        if has_full:
            fulls = [mask128(it["full_watermark"]) for it in its]
            ffrac = [f.mean() for f in fulls]
            removed = [1 - (g & f).sum() / max(f.sum(), 1) for g, f in zip(gts, fulls)]
            outside = [(g & (1 - f)).sum() / max(g.sum(), 1) for g, f in zip(gts, fulls)]
            ys_xs = [np.nonzero(f) for f in fulls]
            asp = [max((y.max() - y.min() + 1) / (x.max() - x.min() + 1), (x.max() - x.min() + 1) / (y.max() - y.min() + 1))
                   for y, x in ys_xs if len(y)]
            print(f"  full_watermark share       : {pct(ffrac)}")
            print(f"  share of it later removed  : {pct(removed)}   (holes made by edits)")
            print(f"  true mask outside it       : {pct(outside)}   (should be ~0)")
            if asp:
                print(f"  aspect ratio of rectangle  : {pct(asp)}")
            sc["ORACLE full_watermark"] = score(fulls, gts)
        for k, v in sc.items():
            print(f"  score if we predict {k:<26}: {v:.3f}")
    print("\nHow to read the ORACLE lines: the score we would get if we found the rectangle PERFECTLY.")
    print("The gap between that and 1.0 is what hole/object detection has to win.")

    # ------------------------------------------------------------------ Part B
    if a.decode:
        print("\n" + "=" * 30, "PART B: does decoding the right crop work?", "=" * 30)
        for meth in METHODS:
            its = [it for it in items if str(it["method"]).lower() == meth][:40]
            if not its:
                continue
            wm = W._CLASSES[meth](__import__("pathlib").Path("models").resolve(), a.device)
            accs = {"full image": [], "crop = bbox of true mask": [], "crop = full_watermark": [], "random crop": []}
            for it in its:
                x = to_tensor(it["image"])[None].to(a.device)
                b = bits_of(it["message"])
                crops = {"full image": x}
                g = mask128(it["mask128"])
                for key, m in [("crop = bbox of true mask", g),
                               ("crop = full_watermark", mask128(it["full_watermark"]) if arr(it["full_watermark"]).ndim >= 2 else None)]:
                    if m is None or m.sum() == 0:
                        continue
                    ys, xs = np.nonzero(m)
                    y0, y1, x0, x1 = ys.min() * 4, (ys.max() + 1) * 4, xs.min() * 4, (xs.max() + 1) * 4
                    crops[key] = F.interpolate(x[:, :, y0:y1, x0:x1], size=(512, 512), mode="bilinear", align_corners=False)
                y0, x0 = np.random.randint(0, 256, 2)
                crops["random crop"] = F.interpolate(x[:, :, y0:y0 + 256, x0:x0 + 256], size=(512, 512), mode="bilinear", align_corners=False)
                for key, c in crops.items():
                    with torch.no_grad():
                        d = arr(wm.decode(c)).flatten()[: len(b)]
                    accs[key].append(((d > 0.5).astype(int) == b[: len(d)]).mean())
            print(f"\n--- {meth} (first {len(its)} images): mean bit accuracy (0.5 = chance, 1.0 = perfect)")
            for k, v in accs.items():
                if v:
                    print(f"  {k:<28}: {np.mean(v):.3f}")
            del wm

    # ------------------------------------------------------------------ Part C
    if a.viz:
        from PIL import Image
        os.makedirs("stats_viz", exist_ok=True)

        def up(m):
            return np.repeat(np.repeat(mask128(m), 4, 0), 4, 1)[..., None].repeat(3, 2) * 255

        for meth in METHODS:
            its = [it for it in items if str(it["method"]).lower() == meth][: a.viz]
            for k, it in enumerate(its):
                img = arr(it["image"]).astype(np.uint8)
                if img.shape[0] == 3:
                    img = img.transpose(1, 2, 0)
                panels = [img, up(it["mask128"])]
                if arr(it["full_watermark"]).ndim >= 2:
                    panels.append(up(it["full_watermark"]))
                sep = np.full((512, 8, 3), 128, np.uint8)
                row = np.concatenate(sum([[p.astype(np.uint8), sep] for p in panels], [])[:-1], 1)
                Image.fromarray(row).save(f"stats_viz/{meth}_{k}_{it['image_name']}.png")
        print("\nSaved pictures to stats_viz/  (left: image | middle: true mask, white = watermarked | right: original rectangle)")


if __name__ == "__main__":
     main()
