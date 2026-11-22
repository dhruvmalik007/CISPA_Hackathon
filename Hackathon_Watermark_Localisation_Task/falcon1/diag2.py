"""
Diagnostics round 2 (no submission).
    PYTHONPATH=.. python -u diag2.py --data ../release
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wmloc.data import LocalisationDataset
from step2_shapes import METHODS, kind_thresholds, mask128, bits_of, to_tensor
from step3_train import miou_batch
import wmloc.watermarks as W


def roc_auc(score, label):
    s, y = np.asarray(score, float), np.asarray(label, bool)
    order = s.argsort()
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="release")
    ap.add_argument("--n", type=int, default=40, help="images per method for the oracle-crop test")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    accs = dict(np.load("accs.npz", allow_pickle=True)["accs"].item())
    items = [it for it in LocalisationDataset(a.data, "train")]
    thr = kind_thresholds(items, accs)
    meth_of = lambda it: str(it["method"]).lower()
    kind_of = lambda it: int(accs[it["image_name"]] > thr[meth_of(it)])

    def shape_of(g):
        ys, xs = torch.nonzero(g, as_tuple=True)
        if len(ys) == 0:
            return "empty", None
        box = (ys.min().item(), xs.min().item(), ys.max().item() + 1, xs.max().item() + 1)
        fill = len(ys) / ((box[2] - box[0]) * (box[3] - box[1]))
        return ("RECT" if fill > 0.9 else "object"), box

    print("=== 1. ORACLE CROP: bit accuracy when decoding the TRUE box of the watermarked part (mostly clean) ===")
    print(f"{'method':<11}{'shape':<8}{'n':>4}{'true box crop':>15}{'full image':>12}")
    for m in METHODS:
        its = [it for it in items if meth_of(it) == m and kind_of(it) == 0][: a.n]
        wm = W._CLASSES[m](Path("models").resolve(), a.device)
        R = int(getattr(wm, "resolution", 256))
        res = {"RECT": [], "object": []}
        for it in its:
            g = torch.from_numpy(mask128(it["mask128"])).bool()
            sh, box = shape_of(g)
            if box is None:
                continue
            y0, x0, y1, x1 = [v * 4 for v in box]
            x = to_tensor(it["image"])[None].to(a.device)
            crop = F.interpolate(x[:, :, y0:y1, x0:x1], size=(R, R), mode="bilinear", align_corners=False, antialias=True)
            sgn = torch.tensor(bits_of(it["message"]) * 2 - 1, dtype=torch.float32)
            with torch.no_grad():
                z = wm.logits(crop).float().cpu()[0]
            L = min(len(z), len(sgn))
            res[sh].append((((z[:L] * sgn[:L]) > 0).float().mean().item(), accs[it["image_name"]]))
        for sh in ("RECT", "object"):
            if res[sh]:
                print(f"{m:<11}{sh:<8}{len(res[sh]):>4}{np.mean([r[0] for r in res[sh]]):15.3f}"
                      f"{np.mean([r[1] for r in res[sh]]):12.3f}"
                      f"   (crops >= 0.9: {np.mean([r[0] >= 0.9 for r in res[sh]]):.0%})")
        del wm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=== 2. STEP 1 CLASSIFIER ('mostly watermarked' vs 'mostly clean'): ROC / TPR / FPR ===")
    for m in METHODS:
        its = [it for it in items if meth_of(it) == m]
        s = [accs[it["image_name"]] for it in its]
        y = [bool(np.asarray(it["full_watermark"])) for it in its]
        pred = [v > thr[m] for v in s]
        tp = sum(p and t for p, t in zip(pred, y)); fp = sum(p and not t for p, t in zip(pred, y))
        fn = sum((not p) and t for p, t in zip(pred, y)); tn = sum((not p) and (not t) for p, t in zip(pred, y))
        print(f"  {m:<10} AUC {roc_auc(s, y):.3f}   threshold {thr[m]:.2f}   TPR {tp / (tp + fn):.3f}"
              f"   FPR {fp / (fp + tn):.3f}   errors {fp + fn}/{len(its)}")

    d = torch.load("analysis_oof.pt")
    oof = dict(zip(d["names"], d["oof"].float()))
    print("\n=== 3+4. GATING and PER-IMAGE CUT-OFF rules on step-3 honest predictions ===")
    print(f"{'group':<26}{'fixed cut':>10}{'+gating':>9}{'area-match':>11}{'oracle/img':>11}")
    cuts = torch.arange(0.05, 0.96, 0.05)
    tot = {"fixed": [], "gate": [], "area": [], "oracle": []}
    for m in METHODS:
        for k in (1, 0):
            its = [it for it in items if meth_of(it) == m and kind_of(it) == k and it["image_name"] in oof]
            P = torch.stack([oof[it["image_name"]] for it in its])
            G = torch.stack([torch.from_numpy(mask128(it["mask128"])).bool() for it in its])
            per_cut = torch.stack([miou_batch(P > c, G) for c in cuts])
            ci = per_cut.mean(1).argmax()
            fixed = per_cut[ci]
            const = miou_batch(torch.full_like(G, bool(k)), G)
            minority = ((P > cuts[ci]) != bool(k)).float().mean((1, 2))
            gate_best = max(((torch.where(minority < g, const, fixed).mean().item(), g) for g in
                             [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08]), key=lambda z: z[0])
            gated = torch.where(minority < gate_best[1], const, fixed)
            area_scores = []
            for q in [0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]:
                pr = P if k == 0 else 1 - P
                t = torch.quantile(pr.flatten(1), 1 - q, dim=1)
                pm = pr > t[:, None, None]
                area_scores.append((miou_batch(pm if k == 0 else ~pm, G).mean().item(), q))
            area_best = max(area_scores)
            oracle = per_cut.max(0).values
            lab = f"{m} {'mostly wm' if k else 'mostly clean'}"
            print(f"{lab:<26}{fixed.mean():10.3f}{gated.mean():9.3f}{area_best[0]:11.3f}{oracle.mean():11.3f}"
                  f"   (gate<{gate_best[1]}, area q={area_best[1]})")
            for key, v in (("fixed", fixed.mean()), ("gate", gated.mean()), ("area", area_best[0]), ("oracle", oracle.mean())):
                tot[key].append((float(v), len(its)))
    w = lambda key: sum(v * n for v, n in tot[key]) / sum(n for _, n in tot[key])
    print(f"{'ALL':<26}{w('fixed'):10.3f}{w('gate'):9.3f}{w('area'):11.3f}{w('oracle'):11.3f}")


if __name__ == "__main__":
    main()
