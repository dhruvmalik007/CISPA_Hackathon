"""
Diagnostics round 3 (no submission).
    PYTHONPATH=.. python -u diag3.py --data ../release
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import binom

from wmloc.data import LocalisationDataset
from step2_shapes import METHODS, kind_thresholds, mask128, bits_of, to_tensor
from step3_train import miou_batch
from rect_search import box_mask
import wmloc.watermarks as W


def true_box(g):
    ys, xs = torch.nonzero(g, as_tuple=True)
    if len(ys) == 0:
        return None, 0.0
    b = (ys.min().item() * 4, xs.min().item() * 4, (ys.max().item() + 1) * 4, (xs.max().item() + 1) * 4)
    fill = len(ys) / (((b[2] - b[0]) / 4) * ((b[3] - b[1]) / 4))
    return b, fill


class Dec:
    def __init__(self, wm, it, device):
        self.wm, self.x = wm, to_tensor(it["image"])[None].to(device)
        self.sgn = torch.tensor(bits_of(it["message"]) * 2 - 1, dtype=torch.float32)
        self.R = int(getattr(wm, "resolution", 256))
        z = wm.logits(self.x).float().cpu()[0]
        self.L = min(len(z), len(self.sgn))
        self.scale = z[: self.L].abs().median().item() + 1e-6

    @torch.no_grad()
    def __call__(self, box):
        y0, x0, y1, x1 = [int(v) for v in box]
        y0, x0, y1, x1 = max(0, y0), max(0, x0), min(512, y1), min(512, x1)
        if y1 - y0 < 8 or x1 - x0 < 8:
            return 0.5, 0.0, 0.0
        c = F.interpolate(self.x[:, :, y0:y1, x0:x1], size=(self.R, self.R), mode="bilinear", align_corners=False, antialias=True)
        a = self.wm.logits(c).float().cpu()[0][: self.L] * self.sgn[: self.L]
        k = int((a > 0).sum())
        logp = -binom.logsf(k - 1, self.L, 0.5) / np.log(10)
        return k / self.L, (a / self.scale).mean().item(), logp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="release")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    accs = dict(np.load("accs.npz", allow_pickle=True)["accs"].item())
    items = [it for it in LocalisationDataset(a.data, "train")]
    thr = kind_thresholds(items, accs)
    meth_of = lambda it: str(it["method"]).lower()
    kind_of = lambda it: int(accs[it["image_name"]] > thr[meth_of(it)])
    gt = {it["image_name"]: torch.from_numpy(mask128(it["mask128"])).bool() for it in items}

    d = torch.load("analysis_oof.pt")
    oof = dict(zip(d["names"], d["oof"].float()))
    rect = {}
    for m in ("pixelseal", "trustmark"):
        p = os.path.join("cache_rect", f"train_{m}.pt")
        if os.path.exists(p):
            rect.update(torch.load(p))
    cut, T = {}, {}
    for m in METHODS:
        for k in (0, 1):
            its = [it for it in items if meth_of(it) == m and kind_of(it) == k]
            P = torch.stack([oof[it["image_name"]] for it in its])
            G = torch.stack([gt[it["image_name"]] for it in its])
            cut[(m, k)] = max((miou_batch(P > t, G).mean().item(), t) for t in np.arange(0.05, 0.96, 0.05))[1]
    for m in ("pixelseal", "trustmark"):
        its = [it for it in items if meth_of(it) == m and kind_of(it) == 0 and rect.get(it["image_name"])]
        rows = [(miou_batch((oof[it["image_name"]] > cut[(m, 0)])[None], gt[it["image_name"]][None]).item(),
                 miou_batch(box_mask(rect[it["image_name"]][0])[None], gt[it["image_name"]][None]).item(),
                 rect[it["image_name"]][2]) for it in its]
        base = np.mean([r[0] for r in rows]) if rows else 0
        best = max(((np.mean([b if ac >= t else s for s, b, ac in rows]), t) for t in np.arange(0.6, 1.001, 0.025)),
                   key=lambda z: z[0]) if rows else (0, 2.0)
        T[m] = best[1] if best[0] > base else 2.0

    cats = {}
    for it in items:
        n, m, k = it["image_name"], meth_of(it), kind_of(it)
        g = gt[n]
        mask = oof[n] > cut[(m, k)]
        r = rect.get(n)
        if k == 0 and m in T and r and r[2] >= T[m]:
            mask = box_mask(r[0])
        s = miou_batch(mask[None], g[None]).item()
        box, fill = true_box(g if k == 0 else ~g)
        minority = (g.float().mean() if k == 0 else 1 - g.float().mean()).item()
        if k == 1:
            gb, gfill = true_box(g)
            name = "mostly wm: watermark = rectangle (outpainting?)" if gfill > 0.9 and minority > 0.05 else \
                   ("mostly wm: holes < 5%" if minority < 0.05 else "mostly wm: holes >= 5%")
        else:
            name = ("mostly clean: rectangle" if fill > 0.9 else "mostly clean: object") + f" ({m})"
        cats.setdefault(name, []).append(s)
    print("=== A. WHERE ARE THE REMAINING POINTS? (step 4 applied to train; points left = (1 - score) x share) ===")
    print(f"{'category':<52}{'n':>5}{'score':>8}{'points left':>13}")
    for name, v in sorted(cats.items(), key=lambda kv: -(1 - np.mean(kv[1])) * len(kv[1])):
        print(f"{name:<52}{len(v):>5}{np.mean(v):8.3f}{(1 - np.mean(v)) * len(v) / len(items):13.3f}")
    print(f"{'ALL (step 4 on train)':<52}{len(items):>5}{np.mean(sum(cats.values(), [])):8.3f}")

    print("\n=== B. TOLERANCE: bit accuracy of the TRUE rectangle shifted by d px (mostly clean, rectangles) ===")
    print(f"{'method':<11}" + "".join(f"{'d=' + str(dd):>9}" for dd in (0, 4, 8, 16, 32)) + f"{'grow 8':>9}{'shrink 8':>9}")
    outp = {}
    for m in METHODS:
        wm = W._CLASSES[m](Path("models").resolve(), a.device)
        rects = [it for it in items if meth_of(it) == m and kind_of(it) == 0 and true_box(gt[it["image_name"]])[1] > 0.9][: a.n]
        res = {dd: [] for dd in (0, 4, 8, 16, 32, "g", "s")}
        for it in rects:
            dec = Dec(wm, it, a.device)
            b, _ = true_box(gt[it["image_name"]])
            for dd in (0, 4, 8, 16, 32):
                res[dd].append(dec((b[0] + dd, b[1] + dd, b[2] + dd, b[3] + dd))[0])
            res["g"].append(dec((b[0] - 8, b[1] - 8, b[2] + 8, b[3] + 8))[0])
            res["s"].append(dec((b[0] + 8, b[1] + 8, b[2] - 8, b[3] - 8))[0])
        if rects:
            print(f"{m:<11}" + "".join(f"{np.mean(res[dd]):9.3f}" for dd in (0, 4, 8, 16, 32))
                  + f"{np.mean(res['g']):9.3f}{np.mean(res['s']):9.3f}   (n={len(rects)})")
        ops = [it for it in items if meth_of(it) == m and kind_of(it) == 1
               and true_box(gt[it["image_name"]])[1] > 0.9 and (~gt[it["image_name"]]).float().mean() > 0.05][: a.n]
        rows = []
        for it in ops:
            dec = Dec(wm, it, a.device)
            b, _ = true_box(gt[it["image_name"]])
            rows.append([dec((0, 0, 512, 512)), dec(b), dec((b[0] - 16, b[1] - 16, b[2] + 16, b[3] + 16)),
                         dec((b[0] + 16, b[1] + 16, b[2] - 16, b[3] - 16))])
        outp[m] = rows
        del wm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=== C. OUTPAINTING: does the TRUE box win? (mostly wm, watermarked part = rectangle) ===")
    print(f"{'method':<11}{'n':>4}  {'score':<12}{'full image':>11}{'true box':>10}{'+16 px':>9}{'-16 px':>9}{'true box best':>15}")
    for m in METHODS:
        rows = outp[m]
        if not rows:
            print(f"{m:<11}   0  (no such images)")
            continue
        for si, sname in enumerate(("bit acc", "margin", "-log10 p")):
            vals = [np.mean([r[j][si] for r in rows]) for j in range(4)]
            win = np.mean([r[1][si] >= max(r[0][si], r[2][si], r[3][si]) for r in rows])
            print(f"{m:<11}{len(rows):>4}  {sname:<12}" + "".join(f"{v:>10.3f}" for v in vals) + f"{win:>14.0%}")


if __name__ == "__main__":
    main()
