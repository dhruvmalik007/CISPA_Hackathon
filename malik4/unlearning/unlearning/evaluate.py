#!/usr/bin/env python
"""Evaluate a submission against `Unlearning.pdf`, on a small number of forget images.

This implements the PDF's definitions literally, with one unavoidable substitution that is
labelled in the output:

    orig  : the provided model.pt                       (available)
    sub   : the candidate state dict being evaluated    (available)
    gold  : models retrained without the forget set      (NOT available to participants)

Because `gold` is hidden, the PDF's `gold` terms are replaced by the package's strong-alteration
target: the original model's distribution on a heavily altered copy of each image. That is a
heuristic for "a model that never saw the clean image", and the report says so rather than
pretending the gold model was used.

PDF formulas implemented exactly (for a model m, image x, temperature 2):

    L_m(x) = CE(m(x), true_label)
    p_m(x) = softmax(m(x) / 2)
    mean_std_score = 1 - clip(d_sub / d_orig, 0, 1)
        d_m = |mean(L_m) - mean(L_gold)| + |std(L_m) - std(L_gold)|
    prediction_score = 1 - clip(mean KL(p_gold ‖ p_sub) / mean KL(p_gold ‖ p_orig), 0, 1)
    forget_score = mean_std_score * prediction_score        (averaged: clean + altered, equally)
    retain_accuracy_score = clip(acc_sub / acc_orig, 0, 1)
    retain_kl_similarity  = 1 / (1 + mean KL(p_orig ‖ p_sub))
    retain_score = retain_accuracy_score * retain_kl_similarity
    final_score  = forget_score * retain_score

Usage (on the cluster, from the task directory):

    python -m unlearning.evaluate --limit 100 --orig model.pt --sub output/stage6-match/submission.pt

Labels are not provided by the challenge, so the PDF's `true label` is taken from the original
model's argmax — the same pseudo-label convention the runbook already uses. That is stated in the
output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from unlearning import augment, scoring
from unlearning.task import build_model, preprocess

TEMPERATURE = 2.0


def load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if hasattr(obj, "state_dict"):
        obj = obj.state_dict()
    return {k: v.float() if torch.is_tensor(v) else v for k, v in obj.items()}


def load_model_from(path: str | Path, device: str) -> torch.nn.Module:
    model = build_model()
    model.load_state_dict(load_state_dict(path), strict=True)
    return model.to(device).eval()


def load_forget_images(path: str | Path, limit: int | None) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as data:
        images = torch.from_numpy(data["images"].copy())
    if images.dtype != torch.uint8:
        raise ValueError(f"expected uint8 pixels, got {images.dtype}")
    if images.ndim != 4 or tuple(images.shape[1:]) != (224, 224, 3):
        raise ValueError(f"expected [N,224,224,3], got {tuple(images.shape)}")
    if limit and 0 < limit < len(images):
        images = images[:limit]
    return images


@torch.no_grad()
def logits_for(model: torch.nn.Module, images: torch.Tensor, device: str, batch_size: int = 64) -> torch.Tensor:
    """Server-equivalent forward pass: eval mode, fp16 autocast on CUDA."""
    out = []
    for start in range(0, len(images), batch_size):
        batch = preprocess(images[start : start + batch_size]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            out.append(model(batch).float().cpu())
    return torch.cat(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--orig", default="model.pt")
    parser.add_argument("--sub", required=True, help="candidate .pt state dict")
    parser.add_argument("--limit", type=int, default=100, help="number of forget images to score")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--altered-variants", type=int, default=1, help="mild altered copies per image")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    images = load_forget_images(data_dir / "forget_images.npz", args.limit)
    n = len(images)
    print(f"Scoring {n} forget image(s) on {device}")

    orig = load_model_from(args.orig, device)
    sub = load_model_from(args.sub, device)

    # 1) Clean pass; pseudo-labels from the original model (no true labels are provided).
    orig_clean = logits_for(orig, images, device, args.batch_size)
    sub_clean = logits_for(sub, images, device, args.batch_size)
    labels = orig_clean.argmax(dim=-1)

    # 2) The PDF's hidden "gold" stand-in: original model on a strong alteration.
    target_images = augment.build_target_copy(images, seed=args.seed)
    gold_logits = logits_for(orig, target_images, device, args.batch_size)

    # 3) Mild altered copies, as the server scores both groups equally.
    #    build_altered(images, variants=..., seed=...) -> (altered [N*len(variants),224,224,3], sources)
    orig_altered, sub_altered = [], []
    if args.altered_variants > 0:
        chosen = list(augment.DEFAULT_VARIANTS[: args.altered_variants])
        for variant in chosen:
            built, _sources = augment.build_altered(images, variants=[variant], seed=args.seed)
            orig_altered.append(logits_for(orig, built, device, args.batch_size))
            sub_altered.append(logits_for(sub, built, device, args.batch_size))

    forget = scoring.score_forget(
        orig_clean=orig_clean,
        sub_clean=sub_clean,
        labels=labels,
        orig_alterations=orig_altered,
        sub_alterations=sub_altered,
        orig_target=gold_logits,
    )

    # 4) Retain proxy on the same images: measures drift orig -> sub only.
    retain = scoring.score_retain(orig_clean, sub_clean, labels=None)

    pdf_final = forget["combined_score"] * (retain["kl_similarity"])

    report: dict[str, Any] = {
        "n_images": n,
        "device": device,
        "orig": str(args.orig),
        "sub": str(args.sub),
        "temperature": TEMPERATURE,
        "gold_substitute": "original model on a strong alteration (hidden gold models unavailable)",
        "labels": "pseudo-labels from the original model's argmax (no true labels provided)",
        "forget": forget,
        "retain_proxy": retain,
        "scores": {
            "forget_score": forget["combined_score"],
            "mean_std_score_clean": forget["clean"]["mean_std_score"],
            "mean_std_score_altered": (forget["altered"] or {}).get("mean_std_score"),
            "prediction_score_clean": forget["clean"]["prediction_score"],
            "prediction_score_altered": (forget["altered"] or {}).get("prediction_score"),
            "retain_kl_similarity": retain["kl_similarity"],
            "retain_top1_agreement": retain["top1_agreement_orig_sub"],
            "final_score_proxy": pdf_final,
        },
        "caveats": [
            "gold is replaced by a strong-alteration target; true gold models are hidden",
            "retain_accuracy_score needs a hidden control set; only KL similarity is measurable here",
            "the 70% utility gate is unverifiable locally",
            "labels are pseudo-labels, so per-image CE uses the original model's own prediction",
        ],
    }

    text = json.dumps(report, indent=2)
    print(text)
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "eval_pdf.json").write_text(text)
        print(f"\nwrote {out / 'eval_pdf.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
