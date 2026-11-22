"""
Do GRADIENT maps find the holes better than feat / rise? (no submission)
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from wmloc.data import LocalisationDataset
import wmloc.watermarks as W
from step2_shapes import bits_of, to_tensor, mask128, kind_thresholds, get_map, smooth
from step3_train import load_maps
from analyze import pixel_auc

SMOOTH_K = [0, 5, 11]


def blur(x, k=5):
    return F.avg_pool2d(F.pad(x, (k // 2,) * 4, mode="reflect"), k, 1)


def raw_logits(wm, x):
    fn = getattr(type(wm).logits, "__wrapped__", None)
    return fn(wm, x) if fn is not None else wm.logits(x)


def score_fn(wm, x, sgn, scale):
    z = raw_logits(wm, x).float()
    L = min(z.shape[1], len(sgn))
    return torch.tanh(z[:, :L] * sgn[:L].to(z.device) / scale).mean()


def grad_maps(wm, x, sgn):
    with torch.no_grad():
        scale = wm.logits(x).float().abs().median() + 1e-6
    xb = blur(x)
    gs = []
    for i in range(5):
        xi = (x + (0.01 * torch.randn_like(x) if i else 0)).clamp(0, 1).detach().requires_grad_(True)
        with torch.enable_grad():
            s = score_fn(wm, xi, sgn, scale)
            if not s.requires_grad:
                return None
            (g,) = torch.autograd.grad(s, xi)
        gs.append(g)
    pool = lambda m: F.avg_pool2d(m[None, None], 4)[0, 0].cpu()
    return {"absgrad": pool(gs[0].abs().sum(1)[0]),
            "drop": pool((gs[0] * (x - xb)).sum(1)[0]),
            "sgrad": pool((torch.stack(gs[1:]).mean(0) * (x - xb)).sum(1)[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="release")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    accs = dict(np.load("accs.npz", allow_pickle=True)["accs"].item())
    items = list(LocalisationDataset(a.data, "train"))
    thr = kind_thresholds(items, accs)
    maps = load_maps("train")
    cols = ["feat", "rise", "absgrad", "drop", "sgrad"]
    print("AUC of each map against the true mask (mean over images; 0.5 = chance)")
    print(f"{'method':<11}{'kind':<14}{'n':>4}" + "".join(f"{c:>10}" for c in cols) + "   best smoothing")
    for meth in ["pixelseal", "trustmark"]:
        wm = W._CLASSES[meth](Path("models").resolve(), a.device)
        for kind, kname in [(1, "mostly wm"), (0, "mostly clean")]:
            its = [it for it in items if it["method"] == meth and int(accs[it["image_name"]] > thr[meth]) == kind][: a.n]
            res = {c: {k: [] for k in SMOOTH_K} for c in cols}
            for it in its:
                gt = torch.from_numpy(mask128(it["mask128"])).bool()
                if gt.all() or (~gt).all():
                    continue
                x = to_tensor(it["image"])[None].to(a.device)
                sgn = torch.tensor(bits_of(it["message"]) * 2 - 1, dtype=torch.float32)
                gm = grad_maps(wm, x, sgn)
                if gm is None:
                    import inspect
                    print(f"  {meth}: logits() gives no gradient. Its source, paste it to me:")
                    print(inspect.getsource(type(wm).logits))
                    return
                cur = {"feat": get_map(maps, it["image_name"], "feat") if it["image_name"] in maps else None,
                       "rise": get_map(maps, it["image_name"], "rise") if it["image_name"] in maps else None}
                cur.update(gm)
                for c, m in cur.items():
                    if m is None:
                        continue
                    for k in SMOOTH_K:
                        res[c][k].append(pixel_auc(smooth(m.float(), k), gt))
            row, bestk = [], []
            for c in cols:
                vals = {k: np.nanmean(v) if v else float("nan") for k, v in res[c].items()}
                k = max(vals, key=lambda kk: -1 if np.isnan(vals[kk]) else vals[kk])
                row.append(vals[k])
                bestk.append(k)
            print(f"{meth:<11}{kname:<14}{len(res['drop'][0]):>4}" + "".join(f"{v:>10.3f}" for v in row)
                  + "   " + "/".join(str(k) for k in bestk), flush=True)


if __name__ == "__main__":
    main()
