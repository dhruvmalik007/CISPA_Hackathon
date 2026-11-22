"""
Step 4: step-3 model + rectangle search (mostly-clean PixelSeal and TrustMark images).
    PYTHONPATH=.. python -u step4_rect.py --data ../release --submit
"""
import argparse

import numpy as np
import torch

from wmloc.data import LocalisationDataset
from step2_shapes import kind_thresholds, mask128
from step3_train import miou_batch
from rect_search import run, box_mask

USE = ("pixelseal", "trustmark")      # StegaStamp: rectangle search did not help in the check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="release")
    ap.add_argument("--out", default="submission_step4.pt")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    accs = dict(np.load("accs.npz", allow_pickle=True)["accs"].item())
    train = [it for it in LocalisationDataset(a.data, "train")]
    test = [it for it in LocalisationDataset(a.data, "test")]
    thr = kind_thresholds(train, accs)
    meth_of = lambda it: str(it["method"]).lower()
    kind_of = lambda it: int(accs[it["image_name"]] > thr[meth_of(it)])
    target = lambda it: meth_of(it) in USE and kind_of(it) == 0

    d = torch.load("analysis_oof.pt")
    oof = dict(zip(d["names"], d["oof"].float()))
    cut = {}
    for m in ("pixelseal", "trustmark", "stegastamp"):
        for k in (0, 1):
            its = [it for it in train if meth_of(it) == m and kind_of(it) == k and it["image_name"] in oof]
            P = torch.stack([oof[it["image_name"]] for it in its])
            G = torch.stack([torch.from_numpy(mask128(it["mask128"])).bool() for it in its])
            cut[(m, k)] = max((miou_batch(P > t, G).mean().item(), t) for t in np.arange(0.05, 0.96, 0.05))[1]

    tr_t = [it for it in train if target(it)]
    print(f"=== 1/3 rectangle search on {len(tr_t)} train images (mostly clean, PixelSeal + TrustMark) ===", flush=True)
    rtr = run(tr_t, accs, thr, a.device, "train")
    T = {}
    print("\n=== TRAIN: model vs model + rectangles (mostly clean) ===")
    gain_sum = 0.0
    for m in USE:
        its = [it for it in tr_t if meth_of(it) == m]
        rows = []
        for it in its:
            g = torch.from_numpy(mask128(it["mask128"])).bool()
            mm = oof[it["image_name"]] > cut[(m, 0)]
            r = rtr[it["image_name"]]
            rows.append((miou_batch(mm[None], g[None]).item(),
                         miou_batch(box_mask(r[0])[None], g[None]).item() if r else 0.0,
                         r[2] if r else 0.0))
        base = np.mean([x[0] for x in rows])
        best = max(((np.mean([b if acc >= t else s for s, b, acc in rows]), t) for t in np.arange(0.6, 1.001, 0.025)),
                   key=lambda z: z[0])
        T[m] = best[1] if best[0] > base else 2.0
        used = np.mean([acc >= T[m] for _, _, acc in rows])
        gain_sum += (best[0] - base) * len(rows) if best[0] > base else 0.0
        print(f"  {m:<10} ({len(rows)} imgs): model {base:.3f} -> with rectangles {max(best[0], base):.3f}"
              f"   T={T[m]:.3f}, rectangle used on {used:.0%}")
    gain = gain_sum / len(train)
    print(f"\nexpected change on the overall score: +{gain:.3f}  (step 3 leaderboard 0.473 -> ~{0.473 + gain:.3f})",
          flush=True)

    te_t = [it for it in test if target(it)]
    print(f"\n=== 2/3 rectangle search on {len(te_t)} test images ===", flush=True)
    rte = run(te_t, accs, thr, a.device, "test")

    sub = torch.load("submission_step3.pt")
    names, masks = list(sub["image_names"]), sub["masks"].clone().bool()
    idx = {n: i for i, n in enumerate(names)}
    replaced = 0
    for it in te_t:
        r = rte.get(it["image_name"])
        if r is not None and r[2] >= T[meth_of(it)]:
            masks[idx[it["image_name"]]] = box_mask(r[0])
            replaced += 1
    assert masks.shape == (len(test), 128, 128) and len(set(names)) == len(names)
    torch.save({"image_names": names, "masks": masks.bool()}, a.out)
    print(f"\n=== 3/3 wrote {a.out}: {len(names)} masks, {replaced} replaced by a rectangle ===")

    if a.submit:
        if gain > 0.002:
            import submission_template as st
            st.BASE_URL = "http://34.147.229.55"
            st.submit(a.out)
        else:
            print("NOT submitted: rectangles did not help on train. Your best stays on the leaderboard.")


if __name__ == "__main__":
    main()
