"""
Step 7b: like step 7, but choose which saved models to use.
  --models folds       : average of the 5 fine-tuned models  <prefix>_fold{0..4}.pt
  --models pretrained  : the single synthetic-only model      <prefix>_pretrained.pt
Without --thr (pretrained only): threshold picked on the real TRAIN images (never trained on).
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import wmloc.watermarks as W
from wmloc.data import LocalisationDataset
from step2_shapes import kind_thresholds, mask128
from test3_synth import u8, to_input
from test3b_synth import best_score
from test3c_dec import DecNet, measure_preproc, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../release")
    ap.add_argument("--method", default="trustmark")
    ap.add_argument("--kind", default="wm", choices=["wm", "clean"])
    ap.add_argument("--prefix", default="test3c_trustmark")
    ap.add_argument("--models", default="folds", choices=["folds", "pretrained"])
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--base", default="submission_step7.pt")
    ap.add_argument("--out", default="submission_step7b.pt")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev, meth = a.device, a.method

    accs = dict(np.load("accs.npz", allow_pickle=True)["accs"].item())
    all_train = list(LocalisationDataset(a.data, "train"))
    thr = kind_thresholds(all_train, accs)
    want = (a.kind == "wm")
    of_kind = lambda it: str(it["method"]).lower() == meth and (accs[it["image_name"]] > thr[meth]) == want
    real = [it for it in all_train if of_kind(it)]
    test = [it for it in LocalisationDataset(a.data, "test") if of_kind(it)]
    print(f"group: {meth} mostly {'watermarked' if want else 'clean'}: {len(real)} train, {len(test)} test", flush=True)

    wm = W._CLASSES[meth](Path("models").resolve(), dev)
    pre = measure_preproc(wm, u8(real[0]["image"]), dev)
    net = DecNet(wm.net, pre).to(dev)
    del wm
    files = [f"{a.prefix}_pretrained.pt"] if a.models == "pretrained" else [f"{a.prefix}_fold{f}.pt" for f in range(5)]

    def run(items):
        X = torch.stack([to_input(u8(it["image"]), "rgb") for it in items])
        P = 0
        for fn in files:
            net.load_state_dict(torch.load(fn, map_location=dev))
            P = P + predict(net, X, torch.arange(len(X)), dev) / len(files)
        return P

    t = a.thr
    if t is None:
        if a.models != "pretrained":
            raise SystemExit("give --thr for --models folds (use the threshold printed on the RESULT line)")
        RY = torch.from_numpy(np.stack([mask128(it["mask128"]).astype(bool) for it in real])).float()
        sc, t = best_score(run(real), RY)
        print(f"score on the {len(real)} real train images (never trained on): {sc:.3f} at threshold {t:.2f}", flush=True)

    M = run(test) > t
    print(f"predicted watermarked share on test: {M.float().mean():.2f}", flush=True)
    sub = torch.load(a.base)
    names, masks = list(sub["image_names"]), sub["masks"].clone().bool()
    pos = {n: i for i, n in enumerate(names)}
    for i, it in enumerate(test):
        masks[pos[it["image_name"]]] = M[i]
    assert masks.shape == (1020, 128, 128) and len(set(names)) == len(names)
    torch.save({"image_names": names, "masks": masks}, a.out)
    print(f"wrote {a.out}: {len(test)} masks replaced, rest from {a.base}", flush=True)
    if a.submit:
        import submission_template as st
        st.BASE_URL = "http://34.147.229.55"
        st.submit(a.out)


if __name__ == "__main__":
    main()
