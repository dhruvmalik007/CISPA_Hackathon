"""
Step 4: use the EMBEDDERS to find the watermark pixel by pixel.
    PYTHONPATH=.. python -u step4_embed.py --data ../release --check 20    # quick check on 20 train imgs/method
    PYTHONPATH=.. python -u step4_embed.py --data ../release --submit      # full: maps, training, submission
"""
import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from step2_shapes import METHODS, kind_thresholds, mask128, znorm, bits_of, get_map, smooth
from step3_train import features as base_features, load_maps, train, infer, miou_batch
from wmloc.data import LocalisationDataset
import wmloc.watermarks as W

EPOCHS = 80
FOLDS = 5
SEEDS_FINAL = 3
WINDOWS = (8, 16, 32)
STEP3_CV = 0.486
CACHE = "cache_step4"
CHANNELS = ["ncc8", "ncc16", "ncc32", "beta16", "beta32", "psdet"]   # psdet = PixelSeal detection channel


def box(x, k):
    return F.avg_pool2d(x, k, 1, k // 2, count_include_pad=False)[..., : x.shape[-2], : x.shape[-1]]


def highpass(x):
    return x - box(x, 5)


@torch.no_grad()
def match_maps(img_u8, wm_u8, device):
    x = torch.from_numpy(img_u8.astype(np.float32)).permute(2, 0, 1)[None].to(device)
    r = torch.from_numpy(wm_u8.astype(np.float32)).permute(2, 0, 1)[None].to(device) - x     # re-created pattern
    d, r = highpass(x), highpass(r)
    dr, dd, rr = (d * r).sum(1, keepdim=True), (d * d).sum(1, keepdim=True), (r * r).sum(1, keepdim=True)
    out = {}
    for k in WINDOWS:
        a, b, c = box(dr, k), box(dd, k), box(rr, k)
        out[f"ncc{k}"] = a / (b * c).clamp(min=1e-6).sqrt()
        out[f"beta{k}"] = a / c.clamp(min=1e-6)
    maps = torch.cat([out[c] for c in CHANNELS[:5]], 1)
    maps = F.interpolate(maps, size=(128, 128), mode="area")[0]
    maps[3:] = maps[3:].clamp(-2, 3)
    return maps.cpu().half()


@torch.no_grad()
def pixelseal_detection(wm, img_u8, device):
    """Pixel Seal predicts per pixel a detection channel (0) + the bits; the kit drops channel 0, we recover it."""
    try:
        x = torch.from_numpy(img_u8.astype(np.float32) / 255).permute(2, 0, 1)[None].to(device)
        emb = wm.features(x)["embedding"].float()
        lin = wm.det.pixel_decoder.linear
        d = torch.einsum("bchw,c->bhw", emb, lin.weight[0].float()) + lin.bias[0].float()
        return F.interpolate(d[:, None], size=(128, 128), mode="bilinear", align_corners=False)[0].cpu().half()
    except Exception as e:  # noqa
        return torch.zeros(1, 128, 128).half()


def compute_match(items, device, tag):
    os.makedirs(CACHE, exist_ok=True)
    out = {}
    for meth in METHODS:
        its = [it for it in items if str(it["method"]).lower() == meth]
        path = os.path.join(CACHE, f"{tag}_{meth}.pt")
        cache = torch.load(path) if os.path.exists(path) else {}
        todo = [it for it in its if it["image_name"] not in cache]
        if todo:
            wm = W._CLASSES[meth](Path("models").resolve(), device)
            t0 = time.time()
            for n, it in enumerate(todo):
                img = np.asarray(it["image"]).astype(np.uint8)
                bits = [int(b) for b in bits_of(it["message"])]
                wimg = np.asarray(wm.embed(img, bits)).astype(np.uint8)
                mm = match_maps(img, wimg, device)
                det = pixelseal_detection(wm, img, device) if meth == "pixelseal" else torch.zeros(1, 128, 128).half()
                cache[it["image_name"]] = torch.cat([mm, det])
                if (n + 1) % 20 == 0 or n + 1 == len(todo):
                    torch.save(cache, path)
                    left = (len(todo) - n - 1) * (time.time() - t0) / (n + 1) / 60
                    print(f"  {tag} {meth}: {n + 1}/{len(todo)}  (~{left:.0f} min left for this part)", flush=True)
            del wm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        out.update(cache)
    return out


def best_cut_score(maps, gts, kind, smooth_k=(0, 5)):
    G = gts[None]
    best = 0.0
    thr = torch.linspace(-3, 4, 29)[:, None, None, None]
    for k in smooth_k:
        m = maps if k == 0 else F.avg_pool2d(maps[:, None], k, 1, k // 2, count_include_pad=False)[:, 0]
        P = m[None] > thr
        i1, u1 = (P & G).sum((2, 3)).float(), (P | G).sum((2, 3)).float()
        i0, u0 = (~P & ~G).sum((2, 3)).float(), (~P | ~G).sum((2, 3)).float()
        iou1 = torch.where(u1 > 0, i1 / u1.clamp(min=1), torch.ones_like(u1))
        iou0 = torch.where(u0 > 0, i0 / u0.clamp(min=1), torch.ones_like(u0))
        best = max(best, ((iou0 + iou1) / 2).mean(1).max().item())
    return best


def check(train_items, match, maps2, accs, thr):
    print("\n=== QUICK CHECK: best simple cut-off score per map (higher = sharper evidence) ===")
    print(f"{'group':<32}{'step1':>7}{'feat':>7}{'rise':>7}" + "".join(f"{c:>8}" for c in CHANNELS))
    for m in METHODS:
        for k in (1, 0):
            its = [it for it in train_items if str(it["method"]).lower() == m
                   and int(accs[it["image_name"]] > thr[m]) == k and it["image_name"] in match]
            if not its:
                continue
            names = [it["image_name"] for it in its]
            gts = torch.stack([torch.from_numpy(mask128(it["mask128"])).bool() for it in its])
            const = miou_batch(torch.full_like(gts, bool(k)), gts).mean().item()
            row = f"{m + (' mostly wm' if k else ' mostly clean'):<26}({len(its):>3}) {const:7.3f}"
            for which in ("feat", "rise"):
                if get_map(maps2, names[0], which) is None:
                    row += f"{'-':>7}"
                else:
                    row += f"{best_cut_score(torch.stack([get_map(maps2, n, which) for n in names]), gts, k):7.3f}"
            for ci, cname in enumerate(CHANNELS):
                mm = torch.stack([znorm(match[n][ci].float()) for n in names])
                sc = best_cut_score(mm, gts, k)
                if cname == "psdet":
                    sc = max(sc, best_cut_score(-mm, gts, k)) if m == "pixelseal" else float("nan")
                row += f"{sc:8.3f}"
            print(row)
    print("If the ncc/beta columns are clearly above feat/rise, the embedder maps are sharper.")


def feats(it, maps2, match, accs, thr):
    base = base_features(it, maps2, accs, thr)
    mm = match[it["image_name"]].float()
    raw_beta = mm[4:5].clamp(-1, 2)
    mm = mm[:5] if mm[5].abs().max() == 0 else mm
    z = torch.stack([znorm(c) for c in mm])
    if z.shape[0] == 5:
        z = torch.cat([z, torch.zeros(1, 128, 128)])
    return torch.cat([base, z, raw_beta])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="release")
    ap.add_argument("--out", default="submission_step4.pt")
    ap.add_argument("--check", type=int, default=0, help="quick check on N train images per method, then stop")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--min-cv", type=float, default=STEP3_CV)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)

    accs = dict(np.load("accs.npz", allow_pickle=True)["accs"].item())
    train_items = [it for it in LocalisationDataset(a.data, "train")]
    thr = kind_thresholds(train_items, accs)
    maps2 = load_maps("train")

    if a.check:
        sub = sum([[it for it in train_items if str(it["method"]).lower() == m][: a.check] for m in METHODS], [])
        print(f"=== computing embedder maps for {len(sub)} train images ===", flush=True)
        match = compute_match(sub, a.device, "train")
        check(sub, match, maps2, accs, thr)
        return

    test_items = [it for it in LocalisationDataset(a.data, "test")]
    print("=== 1/3 embedder maps for all images (cached, only done once) ===", flush=True)
    match = compute_match(train_items, a.device, "train")
    match.update(compute_match(test_items, a.device, "test"))
    check(train_items, match, maps2, accs, thr)

    smaps = load_maps("test")
    X = torch.stack([feats(it, maps2, match, accs, thr) for it in train_items])
    Y = torch.stack([torch.from_numpy(mask128(it["mask128"])).float() for it in train_items])
    Xt = torch.stack([feats(it, smaps, match, accs, thr) for it in test_items])
    meth = [str(it["method"]).lower() for it in train_items]
    kind = [int(accs[it["image_name"]] > thr[m]) for it, m in zip(train_items, meth)]
    print(f"\ninputs per pixel: {X.shape[1]} (step 3 had 11)", flush=True)

    print("\n=== 2/3 training with 5-fold cross-validation ===", flush=True)
    folds = torch.randperm(len(X)) % FOLDS
    oof = torch.zeros_like(Y)
    Pt = torch.zeros(len(test_items), 128, 128)
    for f in range(FOLDS):
        tr, va = folds != f, folds == f
        net = train(X[tr], Y[tr], a.device, a.epochs)
        oof[va] = infer(net, X[va], a.device)
        Pt += infer(net, Xt, a.device)
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
    cv = miou_batch(P, G).mean().item()
    print(f"\nCV score: {cv:.3f}   (step 3 was {STEP3_CV:.3f} CV / 0.473 leaderboard)", flush=True)

    print("\n=== 3/3 final models on all 510 images ===", flush=True)
    for s in range(SEEDS_FINAL):
        torch.manual_seed(100 + s)
        Pt += infer(train(X, Y, a.device, a.epochs), Xt, a.device)
        print(f"  final model {s + 1}/{SEEDS_FINAL} done", flush=True)
    Pt /= FOLDS + SEEDS_FINAL
    names, masks = [], []
    for i, it in enumerate(test_items):
        m = str(it["method"]).lower()
        k = int(accs[it["image_name"]] > thr[m])
        names.append(it["image_name"])
        masks.append(Pt[i] > cut[(m, k)])
    torch.save({"image_names": names, "masks": torch.stack(masks).bool()}, a.out)
    print(f"wrote {a.out}: {len(names)} masks")

    if a.submit:
        if cv > a.min_cv:
            import submission_template as st
            st.BASE_URL = "http://34.147.229.55"
            st.submit(a.out)
        else:
            print(f"NOT submitted: CV {cv:.3f} is not above {a.min_cv:.3f} (step 3). Your best stays on the leaderboard.")


if __name__ == "__main__":
    main()
