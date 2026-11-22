"""
Step 7: average the 5 fine-tuned test-3c models on the TEST images of TrustMark mostly-watermarked,
put those masks into our best submission, keep everything else.
  PYTHONPATH=.. python -u step7_submit.py --data ../release --thr 0.55            # writes file only
  PYTHONPATH=.. python -u step7_submit.py --data ../release --thr 0.55 --submit   # writes + submits
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import wmloc.watermarks as W
from wmloc.data import LocalisationDataset
from step2_shapes import kind_thresholds
from test3_synth import u8, to_input
from test3c_dec import DecNet, measure_preproc, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../release")
    ap.add_argument("--method", default="trustmark")
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--base", default="submission_step6.pt")
    ap.add_argument("--out", default="submission_step7.pt")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev, meth = a.device, a.method

    accs = dict(np.load("accs.npz", allow_pickle=True)["accs"].item())
    all_train = list(LocalisationDataset(a.data, "train"))
    thr = kind_thresholds(all_train, accs)
    test = [it for it in LocalisationDataset(a.data, "test")
            if str(it["method"]).lower() == meth and accs[it["image_name"]] > thr[meth]]
    print(f"test images to replace: {len(test)} ({meth} mostly watermarked)", flush=True)

    wm = W._CLASSES[meth](Path("models").resolve(), dev)
    ref_img = next(u8(it["image"]) for it in all_train if str(it["method"]).lower() == meth)
    pre = measure_preproc(wm, ref_img, dev)
    net = DecNet(wm.net, pre).to(dev)
    del wm

    TI = torch.stack([to_input(u8(it["image"]), "rgb") for it in test])
    idx = torch.arange(len(TI))
    P = 0
    for f in range(5):
        net.load_state_dict(torch.load(f"test3c_{meth}_fold{f}.pt", map_location=dev))
        P = P + predict(net, TI, idx, dev) / 5
        print(f"  model {f + 1}/5 done", flush=True)
    M = P > a.thr
    print(f"predicted watermarked share on test: {M.float().mean():.2f} (train group is ~0.90)", flush=True)

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
