"""Weight interpolation / extrapolation between the original model and an unlearned one: no training needed.

    theta(alpha) = theta_orig + alpha * (theta_unlearned - theta_orig)

alpha = 1 is the unlearned model itself; alpha > 1 forgets more, alpha < 1 forgets less.
For each alpha it saves a valid submission and prints a quick check (forget images + external proxy).

    python interpolate.py --sub runs/r3_pw3/final.pt --alphas 0.5 1 1.5 2 3 --proxy-npz proxy.npz
    -> runs/r3_pw3/alpha1.5.pt, runs/r3_pw3/alpha2.pt, ...
Takes about a minute per alpha on a GPU (mostly the forward passes for the check).
"""
import argparse
from pathlib import Path

import torch

from common import alter, group_stats, load_npz_images, retain_proxy_score, save_submission
from task_template import build_model, load_forget_images, predict, DEVICE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.pt")
    ap.add_argument("--forget", default="forget_images.npz")
    ap.add_argument("--sub", required=True, help="unlearned state dict (the direction)")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.5, 2, 3])
    ap.add_argument("--proxy-npz", default=None, help="external images for the drift check")
    ap.add_argument("--check-n", type=int, default=500, help="forget images used in the check")
    ap.add_argument("--proxy-n", type=int, default=1000)
    ap.add_argument("--no-save", action="store_true", help="only print the checks")
    a = ap.parse_args()

    sd0 = torch.load(a.model, map_location="cpu", weights_only=True)
    sd1 = torch.load(a.sub, map_location="cpu", weights_only=True)
    model = build_model().to(DEVICE)

    imgs = load_forget_images(Path(a.forget))[: a.check_n]
    flip = alter(imgs, "flip", seed=1)
    proxy = load_npz_images(a.proxy_npz)[: a.proxy_n] if a.proxy_npz else None

    model.load_state_dict(sd0)
    ref = {"exact": predict(model, imgs), "flip": predict(model, flip)}
    if proxy is not None:
        ref["proxy"] = predict(model, proxy)
    y = ref["exact"].argmax(1)

    print(f"{'alpha':>6s} | {'exact loss':>12s} {'acc':>5s} {'natural':>7s} | {'flip loss':>12s} {'acc':>5s} | "
          f"{'proxy KL':>8s} {'agree':>5s}")
    for alpha in a.alphas:
        sd = {k: (sd0[k].float() + alpha * (sd1[k].float() - sd0[k].float())) if sd0[k].is_floating_point()
              else sd1[k] for k in sd0}
        model.load_state_dict(sd)
        e = group_stats(predict(model, imgs), ref["exact"], y)
        f = group_stats(predict(model, flip), ref["flip"], y)
        line = (f"{alpha:6.2f} | {e['loss_mean']:5.2f}±{e['loss_std']:5.2f} {e['acc_vs_pseudo']:5.2f} "
                f"{e.get('wrong_is_natural', float('nan')):7.2f} | {f['loss_mean']:5.2f}±{f['loss_std']:5.2f} "
                f"{f['acc_vs_pseudo']:5.2f} | ")
        if proxy is not None:
            zp = predict(model, proxy)
            kl = group_stats(zp, ref["proxy"], ref["proxy"].argmax(1))["kl_orig_sub_T2"]
            agree = (zp.argmax(1) == ref["proxy"].argmax(1)).float().mean().item()
            line += f"{kl:8.3f} {agree:5.2f}"
        print(line, flush=True)
        if not a.no_save and alpha != 1:
            save_submission(model, Path(a.sub).parent / f"alpha{alpha:g}.pt")


if __name__ == "__main__":
    main()