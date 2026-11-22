"""
Step 2: find the actual shapes (holes in mostly-watermarked images, the small piece in mostly-clean ones).

For every image we build two 128x128 "watermark maps" (high = watermark here):
  feat : the decoder's own per-position guesses (PixelSeal, TrustMark) compared with the known message
  rise : hide random patches ~200 times, decode each version, measure which patches the decoding needs (all 3)
Then, on TRAIN, for each method and each kind (from step 1), we pick the map / smoothing / threshold that gives
the best score, keeping step 1's all-ones / all-zeros answer if nothing beats it. Finally the test masks are written.

Needs accs.npz from step 1 (same folder).
    PYTHONPATH=.. python step2_shapes.py --data ../release --limit 20   # quick try: 20 train images per method
    PYTHONPATH=.. python step2_shapes.py --data ../release              # full run -> submission_step2.pt
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(4)

from wmloc.data import LocalisationDataset
from wmloc.utils import score
import wmloc.watermarks as W

METHODS = ["pixelseal", "trustmark", "stegastamp"]
RISE_N = 192        # random hide-patterns per image (more = cleaner map, slower)
RISE_GRID = 8       # patches per side (8 x 8 = 64 patches of 64 x 64 pixels)
RISE_BATCH = 32
THRESHOLDS = np.linspace(-3, 4, 29)
SMOOTH = [0, 5, 11]  # box blur size on the 128 grid (0 = none)


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


@torch.no_grad()
def feat_map(wm, x, sgn):
    f = wm.features(x)
    if "per_position_logits" not in f:
        return None
    p = f["per_position_logits"].float().to(sgn.device)          # 1 x nbits x h x w
    L = min(p.shape[1], len(sgn))
    m = (p[:, :L] * sgn[:L, None, None]).mean(1, keepdim=True)    # agreement with the message at each position
    return F.interpolate(m, size=(128, 128), mode="bilinear", align_corners=False)[0, 0].cpu()


@torch.no_grad()
def rise_map(wm, x, sgn, gen):
    dev = x.device
    zf = wm.logits(x).float().to(dev)
    L = min(zf.shape[1], len(sgn))
    scale = zf[:, :L].abs().median() + 1e-6
    fill = F.interpolate(F.interpolate(x, size=(16, 16), mode="area"), size=x.shape[-2:], mode="bilinear", align_corners=False)
    cells = (torch.rand(RISE_N, 1, RISE_GRID, RISE_GRID, generator=gen) < 0.5).float().to(dev)
    up = F.interpolate(cells, size=x.shape[-2:], mode="nearest")
    s = []
    for i in range(0, RISE_N, RISE_BATCH):
        u = up[i:i + RISE_BATCH]
        z = wm.logits(u * x + (1 - u) * fill).float().to(dev)
        s.append(torch.tanh(z[:, :L] * sgn[:L] / scale).mean(1))
    s = torch.cat(s)
    X = cells.view(RISE_N, -1)
    Xc, yc = X - X.mean(0), s - s.mean()
    w = torch.linalg.solve(Xc.T @ Xc + 1.0 * torch.eye(X.shape[1], device=dev), Xc.T @ yc)
    return F.interpolate(w.view(1, 1, RISE_GRID, RISE_GRID), size=(128, 128), mode="bilinear", align_corners=False)[0, 0].cpu()


def compute_maps(items, split, device, cache_dir="cache_step2"):
    os.makedirs(cache_dir, exist_ok=True)
    out = {}
    for meth in METHODS:
        its = [it for it in items if str(it["method"]).lower() == meth]
        if not its:
            continue
        path = os.path.join(cache_dir, f"{split}_{meth}.pt")
        cache = torch.load(path) if os.path.exists(path) else {}
        todo = [it for it in its if it["image_name"] not in cache]
        if todo:
            wm = W._CLASSES[meth](Path("models").resolve(), device)
            gen = torch.Generator().manual_seed(0)
            t0 = time.time()
            for n, it in enumerate(todo):
                x = to_tensor(it["image"])[None].to(device)
                sgn = torch.tensor(bits_of(it["message"]) * 2 - 1, dtype=torch.float32, device=device)
                fm = feat_map(wm, x, sgn)
                rm = rise_map(wm, x, sgn, gen)
                cache[it["image_name"]] = {"feat": None if fm is None else fm.half(), "rise": rm.half()}
                if (n + 1) % 20 == 0 or n + 1 == len(todo):
                    torch.save(cache, path)
                    print(f"  {split} {meth}: {n + 1}/{len(todo)}  ({(time.time() - t0) / (n + 1):.2f}s per image)", flush=True)
            del wm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        out.update({k: v for k, v in cache.items() if k in {it["image_name"] for it in its}})
    return out


def znorm(m):
    m = m.float()
    return (m - m.median()) / (m.std() + 1e-6)


def smooth(m, k):
    if k == 0:
        return m
    return F.avg_pool2d(m[None, None], k, 1, k // 2, count_include_pad=False)[0, 0]


def get_map(maps, name, which):
    d = maps[name]
    if which == "feat":
        return None if d["feat"] is None else znorm(d["feat"])
    if which == "rise":
        return znorm(d["rise"])
    if d["feat"] is None:
        return None
    return znorm(znorm(d["feat"]) + znorm(d["rise"]))


def predict(maps, name, kind, rule):
    if rule[0] == "const":
        return np.full((128, 128), kind, np.uint8)
    which, k, t = rule
    m = smooth(get_map(maps, name, which), k)
    return (m > t).numpy().astype(np.uint8)


def kind_thresholds(train, accs):
    thr = {}
    for meth in METHODS:
        its = [it for it in train if str(it["method"]).lower() == meth]
        acc = np.array([accs[it["image_name"]] for it in its])
        gts = [mask128(it["mask128"]) for it in its]
        best = (-1, 0.75)
        for t in np.arange(0.50, 1.001, 0.01):
            s = score([np.ones_like(g) if a_ > t else np.zeros_like(g) for a_, g in zip(acc, gts)], gts)
            if s > best[0]:
                best = (s, t)
        thr[meth] = best[1]
    return thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="release")
    ap.add_argument("--out", default="submission_step2.pt")
    ap.add_argument("--limit", type=int, default=0, help="only N train images per method, no submission")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    accs = dict(np.load("accs.npz", allow_pickle=True)["accs"].item())
    train = [it for it in LocalisationDataset(a.data, "train")]
    thr = kind_thresholds(train, accs)
    if a.limit:
        train = sum([[it for it in train if str(it["method"]).lower() == m][: a.limit] for m in METHODS], [])

    print(f"computing maps for {len(train)} train images on {a.device} ...")
    tmaps = compute_maps(train, "train", a.device)

    print("\n=== TRAIN: best rule per method and kind ===")
    rules, total, n_total = {}, 0.0, 0
    for meth in METHODS:
        for kind in (1, 0):
            its = [it for it in train if str(it["method"]).lower() == meth and (accs[it["image_name"]] > thr[meth]) == bool(kind)]
            if not its:
                continue
            gts = [mask128(it["mask128"]) for it in its]
            names = [it["image_name"] for it in its]
            cands = [("const",)]
            for which in ("feat", "rise", "both"):
                if get_map(tmaps, names[0], which) is None:
                    continue
                cands += [(which, k, float(t)) for k in SMOOTH for t in THRESHOLDS]
            res = [(score([np.full((128, 128), kind, np.uint8) for _ in names], gts), ("const",))]
            G = torch.from_numpy(np.stack(gts)).bool()[None]
            thr_t = torch.tensor(THRESHOLDS, dtype=torch.float32)[:, None, None, None]
            for which in ("feat", "rise", "both"):
                if get_map(tmaps, names[0], which) is None:
                    continue
                base = torch.stack([get_map(tmaps, n, which) for n in names])
                for k in SMOOTH:
                    m = base if k == 0 else F.avg_pool2d(base[:, None], k, 1, k // 2, count_include_pad=False)[:, 0]
                    P = m[None] > thr_t
                    i1, u1 = (P & G).sum((2, 3)).float(), (P | G).sum((2, 3)).float()
                    i0, u0 = (~P & ~G).sum((2, 3)).float(), (~P | ~G).sum((2, 3)).float()
                    iou1 = torch.where(u1 > 0, i1 / u1.clamp(min=1), torch.ones_like(u1))
                    iou0 = torch.where(u0 > 0, i0 / u0.clamp(min=1), torch.ones_like(u0))
                    for t, s_ in zip(THRESHOLDS, ((iou0 + iou1) / 2).mean(1).tolist()):
                        res.append((s_, (which, k, float(t))))
            res.sort(key=lambda z: -z[0])
            const = [s for s, r in res if r[0] == "const"][0]
            rules[(meth, kind)] = res[0][1]
            total += res[0][0] * len(its)
            n_total += len(its)
            label = "mostly watermarked" if kind else "mostly clean      "
            print(f"  {meth:<10} {label} ({len(its):>3} imgs): step1 {const:.3f} -> step2 {res[0][0]:.3f}   rule {res[0][1]}")
    print(f"\nTRAIN overall score: {total / max(n_total, 1):.3f}   (step 1 was 0.453)")
    print("Note: rules are tuned on these same train images, so expect the leaderboard to be a bit lower.")

    if a.limit:
        print("\n--limit used: no test maps, no submission. Run without --limit for the full version.")
        return

    test = [it for it in LocalisationDataset(a.data, "test")]
    print(f"\ncomputing maps for {len(test)} test images ...")
    smaps = compute_maps(test, "test", a.device)
    names, masks = [], []
    for it in test:
        meth = str(it["method"]).lower()
        kind = int(accs[it["image_name"]] > thr[meth])
        names.append(it["image_name"])
        masks.append(predict(smaps, it["image_name"], kind, rules[(meth, kind)]))
    torch.save({"image_names": names, "masks": torch.from_numpy(np.stack(masks)).bool()}, a.out)
    print(f"wrote {a.out}: {len(names)} masks")


if __name__ == "__main__":
    main()