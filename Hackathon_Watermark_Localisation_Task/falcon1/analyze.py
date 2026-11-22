"""
Error analysis of the step-3 model (no submission).
    PYTHONPATH=.. python -u analyze.py --data ../release
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from wmloc.data import LocalisationDataset
from step2_shapes import METHODS, kind_thresholds, mask128, znorm, get_map
from step3_train import features, load_maps, train, infer, miou_batch

FOLDS = 5
CUTS = torch.arange(0.05, 0.96, 0.05)


def ious(P, G):
    i1, u1 = (P & G).sum((-2, -1)).float(), (P | G).sum((-2, -1)).float()
    i0, u0 = (~P & ~G).sum((-2, -1)).float(), (~P | ~G).sum((-2, -1)).float()
    iou1 = torch.where(u1 > 0, i1 / u1.clamp(min=1), torch.ones_like(u1))
    iou0 = torch.where(u0 > 0, i0 / u0.clamp(min=1), torch.ones_like(u0))
    return iou0, iou1


def pixel_auc(score, gt):
    s, g = score.flatten().float(), gt.flatten().bool()
    n1, n0 = g.sum().item(), (~g).sum().item()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = torch.empty_like(s)
    r[s.argsort()] = torch.arange(1, len(s) + 1, dtype=torch.float32)
    return (r[g].sum().item() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="release")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)

    accs = dict(np.load("accs.npz", allow_pickle=True)["accs"].item())
    items = [it for it in LocalisationDataset(a.data, "train")]
    thr = kind_thresholds(items, accs)
    maps = load_maps("train")
    items = [it for it in items if it["image_name"] in maps]
    X = torch.stack([features(it, maps, accs, thr) for it in items])
    Y = torch.stack([torch.from_numpy(mask128(it["mask128"])).float() for it in items])
    G = Y.bool()
    meth = [str(it["method"]).lower() for it in items]
    kind = [int(accs[it["image_name"]] > thr[m]) for it, m in zip(items, meth)]
    truek = [int(bool(np.asarray(it["full_watermark"]))) for it in items]

    print("re-running step 3 cross-validation to get honest predictions ...", flush=True)
    folds = torch.randperm(len(X)) % FOLDS
    oof = torch.zeros_like(Y)
    for f in range(FOLDS):
        tr, va = folds != f, folds == f
        oof[va] = infer(train(X[tr], Y[tr], a.device, a.epochs), X[va], a.device)
        print(f"  fold {f + 1}/{FOLDS} done", flush=True)
    torch.save({"oof": oof.half(), "names": [it["image_name"] for it in items]}, "analysis_oof.pt")

    cut = {}
    for m in METHODS:
        for k in (0, 1):
            ids = [i for i in range(len(X)) if meth[i] == m and kind[i] == k]
            if ids:
                cut[(m, k)] = max((miou_batch(oof[ids] > t, G[ids]).mean().item(), t.item()) for t in CUTS)[1]
    P = torch.stack([oof[i] > cut[(meth[i], kind[i])] for i in range(len(X))])
    i0, i1 = ious(P, G)
    model = (i0 + i1) / 2
    const = torch.stack([miou_batch(torch.full_like(G[i:i + 1], bool(kind[i])), G[i:i + 1])[0] for i in range(len(X))])
    oracle_cut = torch.stack([miou_batch(oof[i][None] > CUTS[:, None, None], G[i].expand(len(CUTS), -1, -1)).max()
                              for i in range(len(X))])
    frac = Y.mean((1, 2))

    def grp(m, k):
        return [i for i in range(len(X)) if meth[i] == m and kind[i] == k]

    print("\n=== 1. WHERE ARE THE POINTS LOST? (per group, mean over images) ===")
    print("minority = holes (mostly wm) or the small piece (mostly clean); majority = the other part")
    print(f"{'group':<28}{'n':>4}{'step1':>7}{'model':>7}{'best cut/img':>13}{'IoU minority':>14}{'IoU majority':>14}")
    for m in METHODS:
        for k in (1, 0):
            ids = grp(m, k)
            mi = (i0 if k else i1)[ids].mean().item()
            ma = (i1 if k else i0)[ids].mean().item()
            name = f"{m} {'mostly wm' if k else 'mostly clean'}"
            print(f"{name:<28}{len(ids):>4}{const[ids].mean():7.3f}{model[ids].mean():7.3f}"
                  f"{oracle_cut[ids].mean():13.3f}{mi:14.3f}{ma:14.3f}")
    print(f"{'ALL':<28}{len(X):>4}{const.mean():7.3f}{model.mean():7.3f}{oracle_cut.mean():13.3f}")

    print("\n=== 2. KIND (step 1) ERRORS ===")
    wrong = [i for i in range(len(X)) if kind[i] != truek[i]]
    print(f"misclassified: {len(wrong)}/{len(X)} images; their mean score {model[wrong].mean():.3f}" if wrong else "none")

    print("\n=== 3. HOW MUCH SIGNAL IS IN EACH INPUT? pixel AUC (0.5 = none, 1.0 = perfect ranking) ===")
    print(f"{'group':<28}{'feat':>7}{'rise':>7}{'model':>7}")
    for m in METHODS:
        for k in (1, 0):
            ids = grp(m, k)
            row = f"{m + (' mostly wm' if k else ' mostly clean'):<28}"
            for which in ("feat", "rise"):
                v = [pixel_auc(get_map(maps, items[i]["image_name"], which), G[i]) for i in ids
                     if get_map(maps, items[i]["image_name"], which) is not None]
                row += f"{np.nanmean(v):7.3f}" if v else f"{'-':>7}"
            row += f"{np.nanmean([pixel_auc(oof[i], G[i]) for i in ids]):7.3f}"
            print(row)

    print("\n=== 4. SCORE BY SIZE OF THE MINORITY PART (holes / small piece), all methods ===")
    minority = torch.tensor([(1 - frac[i]).item() if kind[i] else frac[i].item() for i in range(len(X))])
    bins = [(0, 0.02, "tiny  <2%"), (0.02, 0.05, "2-5%"), (0.05, 0.10, "5-10%"), (0.10, 0.20, "10-20%"), (0.20, 1.01, "large >20%")]
    print(f"{'minority size':<14}{'n':>5}{'step1':>8}{'model':>8}{'IoU minority':>14}")
    for lo, hi, lab in bins:
        ids = [i for i in range(len(X)) if lo <= minority[i] < hi]
        if ids:
            mi = torch.stack([(i0[i] if kind[i] else i1[i]) for i in ids]).mean()
            print(f"{lab:<14}{len(ids):>5}{const[ids].mean():8.3f}{model[ids].mean():8.3f}{mi:14.3f}")

    print("\n=== 5. MOSTLY-CLEAN IMAGES: rectangle pieces vs object pieces ===")
    for m in METHODS:
        ids = grp(m, 0)
        rect, obj = [], []
        for i in ids:
            ys, xs = torch.nonzero(G[i], as_tuple=True)
            if len(ys) == 0:
                continue
            box = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
            (rect if len(ys) / box > 0.9 else obj).append(i)
        for lab, s in (("rectangle", rect), ("object", obj)):
            if s:
                print(f"  {m:<10} {lab:<10} n={len(s):>3}  step1 {const[s].mean():.3f}  model {model[s].mean():.3f}"
                      f"  IoU piece {i1[s].mean():.3f}")

    print("\n=== 6. MODEL vs STEP 1 per image ===")
    d = model - const
    print(f"better than step 1: {(d > 0.01).float().mean():.0%}   about equal: {(d.abs() <= 0.01).float().mean():.0%}"
          f"   worse: {(d < -0.01).float().mean():.0%}")

    os.makedirs("analysis_viz", exist_ok=True)
    for m in METHODS:
        ids = sorted([i for i in range(len(X)) if meth[i] == m], key=lambda i: model[i].item())[:4]
        for r, i in enumerate(ids):
            it = items[i]
            img = np.asarray(it["image"]).astype(np.uint8)
            def up(a_):
                a_ = a_.float().numpy()
                a_ = (a_ - a_.min()) / (a_.max() - a_.min() + 1e-6)
                return np.repeat(np.repeat((a_ * 255).astype(np.uint8), 4, 0), 4, 1)[..., None].repeat(3, 2)
            fm = get_map(maps, it["image_name"], "feat")
            panels = [img, up(Y[i]), up(oof[i]), up(P[i].float()), up(fm if fm is not None else get_map(maps, it["image_name"], "rise"))]
            sep = np.full((512, 8, 3), 128, np.uint8)
            row = np.concatenate(sum([[p_, sep] for p_ in panels], [])[:-1], 1)
            Image.fromarray(row).save(f"analysis_viz/worst_{m}_{r}_score{model[i]:.2f}.png")
    print("\nSaved worst cases to analysis_viz/: image | truth | model probability | model mask | feat (or rise) map")


if __name__ == "__main__":
    main()
