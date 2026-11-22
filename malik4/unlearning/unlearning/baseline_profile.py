#!/usr/bin/env python
"""Baseline profile of the provided model, using task_template.py's own functions.

Runs the *unchanged* template helpers (build_model / load_model / load_forget_images /
preprocess / predict) on a small slice of the forget set, to establish the numbers you need
before any unlearning, as described in the baseline steps:

  1. Pseudo-label and class-concentration profile on the forget images.
     Question answered: is this a *class*-unlearning regime (few classes) or an
     *instance*-unlearning regime (spread across all 100)?
  2. Baseline top-1 confidence of `orig`.
  3. Baseline self-loss L_orig(x, y_orig), mean and std.

The challenge provides no true labels, so `y_orig` is the model's own argmax (a pseudo-label).
That is the same convention the runbook uses, and it is stated in the output.

Why the baseline scores 0 by construction (Unlearning.pdf):

    with sub == orig:  d_sub == d_orig            -> mean_std_score   = 0
                       KL(p_gold||p_sub) == KL(p_gold||p_orig) -> prediction_score = 0
    forget_score = 0 * 0 = 0,  final_score = forget * retain = 0
    retain stays 1.0, which is why the unchanged file clears the 70% utility gate but still scores 0.

Usage:
    python baseline_profile.py --limit 100
    python baseline_profile.py --limit 100 --altered 1     # also profile one altered copy
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".", help="directory holding model.pt and forget_images.npz")
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--limit", type=int, default=100, help="how many forget images to profile")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--altered", type=int, default=0, help="number of mild altered copies to profile")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="", help="write the JSON report here")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    # The official template may sit beside the data or in a copy directory; search both.
    for candidate in (data_dir, data_dir / "orig_files", Path(__file__).resolve().parent.parent / "orig_files"):
        if (candidate / "task_template.py").is_file():
            sys.path.insert(0, str(candidate))
            break
    else:
        raise SystemExit(
            "task_template.py not found; looked in "
            + ", ".join(str(c) for c in (data_dir, data_dir / "orig_files"))
        )
    # Import the official template so we use ITS loader, preprocessing and predict().
    import task_template as tt  # noqa: E402

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    print("=" * 70)
    print("Baseline profile — using task_template.py helpers unchanged")
    print("=" * 70)
    print(f"data dir : {data_dir}")
    print(f"device   : {device}")

    # --- load exactly as the template does ------------------------------------------------
    started = time.perf_counter()
    model = tt.load_model(data_dir / args.model)
    images = tt.load_forget_images(data_dir / "forget_images.npz")
    print(f"model    : loaded ({(time.perf_counter() - started):.1f}s)")
    print(f"forget   : {len(images)} images total, profiling first {min(args.limit, len(images))}")

    subset = images[: args.limit]
    n = len(subset)

    # --- 1. forward pass on the clean forget subset --------------------------------------
    t0 = time.perf_counter()
    logits = tt.predict(model, subset, batch_size=args.batch_size)
    seconds = time.perf_counter() - t0
    print(f"clean    : {n} images in {seconds:.1f}s ({n / max(seconds, 1e-9):.1f} img/s)")

    probs = torch.softmax(logits, dim=-1)
    preds = logits.argmax(dim=-1)
    confidence = probs.max(dim=-1).values

    # self-loss: CE(model(x), argmax(model(x))) — the pseudo-label convention
    self_loss = torch.nn.functional.cross_entropy(logits, preds, reduction="none")
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)

    # --- 2. class concentration ----------------------------------------------------------
    counts = torch.bincount(preds, minlength=100)
    present = counts[counts > 0]
    class_share = (present.float() / n).sort(descending=True).values
    # Normalised entropy of the predicted-class histogram: 0 = one class, 1 = uniform over 100.
    p = present.float() / n
    hist_entropy = float(-(p * torch.log(p)).sum() / math.log(100))
    top_class = int(counts.argmax())
    top_count = int(counts.max())

    # --- 3. optional altered-copy profile ------------------------------------------------
    altered_report = None
    if args.altered > 0:
        try:
            from unlearning import augment

            # Real variant names from augment.VARIANTS = (crop, flip, color, blur_jpeg, occlude).
            chosen = list(augment.DEFAULT_VARIANTS[: args.altered])
            built, _sources = augment.build_altered(subset, variants=chosen, seed=args.seed)
            t1 = time.perf_counter()
            alt_logits = tt.predict(model, built, batch_size=args.batch_size)
            alt_preds = alt_logits.argmax(dim=-1)
            alt_probs = torch.softmax(alt_logits, dim=-1)
            alt_self_loss = torch.nn.functional.cross_entropy(alt_logits, alt_preds, reduction="none")
            # Does each altered copy keep the clean image's predicted class?
            per_variant = len(chosen)
            same = (alt_preds.view(per_variant, len(subset)) == preds.unsqueeze(0)).float().mean()
            altered_report = {
                "n": int(len(built)),
                "variants": chosen,
                "top1_confidence_mean": round(float(alt_probs.max(dim=-1).values.mean()), 6),
                "self_loss_mean": round(float(alt_self_loss.mean()), 6),
                "self_loss_std": round(float(alt_self_loss.std()), 6),
                "prediction_preserved_vs_clean": round(float(same), 6),
                "seconds": round(time.perf_counter() - t1, 2),
            }
        except Exception as exc:  # noqa: BLE001 - profile is optional
            altered_report = {"error": f"{type(exc).__name__}: {exc}"}

    # --- report --------------------------------------------------------------------------
    report: dict[str, Any] = {
        "config": {
            "data_dir": str(data_dir),
            "device": device,
            "limit": args.limit,
            "n_profiled": n,
            "batch_size": args.batch_size,
            "datasets_total": int(len(images)),
        },
        "pseudo_labels": {
            "note": "no true labels are provided; y_orig = argmax(model(x))",
            "distinct_classes": int((counts > 0).sum()),
            "of_total_classes": 100,
            "regime": (
                "class-unlearning (few classes)"
                if int((counts > 0).sum()) <= 10
                else "instance-unlearning (spread across many classes)"
            ),
            "top_class": top_class,
            "top_class_count": top_count,
            "top_class_share": round(top_count / n, 4),
            "top5_class_share": round(float(class_share[:5].sum()), 4),
            "histogram_entropy_normalised": round(hist_entropy, 4),
            "counts_by_class": {int(i): int(c) for i, c in enumerate(counts) if c > 0},
        },
        "confidence": {
            "top1_prob_mean": round(float(confidence.mean()), 6),
            "top1_prob_std": round(float(confidence.std()), 6),
            "top1_prob_min": round(float(confidence.min()), 6),
            "top1_prob_median": round(float(confidence.median()), 6),
            "entropy_mean": round(float(entropy.mean()), 6),
            "entropy_of_uniform_100": round(math.log(100), 4),
        },
        "self_loss": {
            "note": "CE(model(x), argmax(model(x))); near 0 means memorised",
            "mean": round(float(self_loss.mean()), 6),
            "std": round(float(self_loss.std()), 6),
            "min": round(float(self_loss.min()), 6),
            "max": round(float(self_loss.max()), 6),
            "median": round(float(self_loss.median()), 6),
        },
        "altered": altered_report,
        "timing": {"clean_seconds": round(seconds, 2), "img_per_s": round(n / max(seconds, 1e-9), 1)},
        "expected_scores_for_this_baseline": {
            "mean_std_score": 0.0,
            "prediction_score": 0.0,
            "forget_score": 0.0,
            "retain_score": 1.0,
            "final_score": 0.0,
            "reason": "sub == orig makes both forget terms identically 0 (Unlearning.pdf)",
        },
    }

    print()
    print("--- 1. pseudo-label / class concentration ---")
    pl = report["pseudo_labels"]
    print(f"  distinct classes present : {pl['distinct_classes']}/100  -> {pl['regime']}")
    print(f"  most common class        : {pl['top_class']} ({pl['top_class_count']} imgs, {pl['top_class_share']:.1%})")
    print(f"  top-5 classes cover      : {pl['top5_class_share']:.1%}")
    print(f"  histogram entropy (norm) : {pl['histogram_entropy_normalised']}")
    print()
    print("--- 2. baseline top-1 confidence ---")
    cf = report["confidence"]
    print(f"  mean {cf['top1_prob_mean']:.6f}   std {cf['top1_prob_std']:.6f}   min {cf['top1_prob_min']:.6f}")
    print(f"  entropy mean {cf['entropy_mean']:.6f}   (uniform over 100 would be {cf['entropy_of_uniform_100']})")
    print()
    print("--- 3. baseline self-loss L_orig(x, y_orig) ---")
    sl = report["self_loss"]
    print(f"  mean {sl['mean']:.6f}   std {sl['std']:.6f}   median {sl['median']:.6f}")
    print(f"  min {sl['min']:.6f}   max {sl['max']:.6f}")
    if altered_report:
        print()
        print("--- altered copy ---")
        print(json.dumps(altered_report, indent=2))
    print()
    print("--- expected score for sub == orig ---")
    print("  forget_score = 0, final_score = 0, retain_score = 1.0 (Unlearning.pdf)")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
