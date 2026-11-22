"""Log-odds NPO with alteration consistency loss and hybrid retain anchor
(mosaics/natural images + non-target KL on original forget images).

Put this script next to common.py and task_template.py.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from common import alter, group_stats, make_mosaics, retain_proxy_score, save_submission
from task_template import DEVICE, load_forget_images, load_model, predict, preprocess


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("model.pt"))
    parser.add_argument("--forget", type=Path, default=Path("forget_images.npz"))
    parser.add_argument("--out", type=Path, required=True, help="New directory for checkpoints and metrics")
    parser.add_argument("--objective", choices=("log-odds", "standard"), default="log-odds")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3.5e-5)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--save-steps", type=int, nargs="*", default=[500, 600, 700, 800])
    parser.add_argument("--train-blocks", type=int, default=-1, help="-1 = all parameters")
    parser.add_argument("--aug-prob", type=float, default=0.5)
    parser.add_argument("--proxy-weight", type=float, default=4.5)
    parser.add_argument("--proxy-batch", type=int, default=64)
    parser.add_argument("--nt-weight", type=float, default=0.5, help="Relative weight of non-target KL inside proxy loss")
    parser.add_argument("--cons-weight", type=float, default=0.25, help="Weight for clean-vs-altered consistency KL")
    parser.add_argument("--retain-dir", type=Path, default=None, help="Optional folder of natural images or .npz file")
    parser.add_argument("--retain-max-n", type=int, default=5000, help="Max natural images to cache in RAM")
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--eval-n", type=int, default=1000)
    parser.add_argument("--eval-proxy-n", type=int, default=512)
    parser.add_argument("--class-weight-pow", type=float, default=0.0, help="Exponent for (class_count/max_count)^p NPO weighting")
    parser.add_argument("--freeze-zero-head", action="store_true", help="Freeze head rows for classes with 0 forget images")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for name in ("temperature", "beta", "lr"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name} must be positive and finite")
    for name in ("proxy_weight", "nt_weight", "cons_weight"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative and finite")
    for name in ("batch", "max_steps", "proxy_batch", "eval_batch", "eval_n", "eval_proxy_n", "retain_max_n"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if any(step < 1 for step in args.save_steps):
        parser.error("--save-steps must contain positive step numbers")
    if args.train_blocks < -1:
        parser.error("--train-blocks must be -1 or nonnegative")
    if not math.isfinite(args.aug_prob) or not (0.0 <= args.aug_prob <= 1.0):
        parser.error("--aug-prob must be between 0 and 1")
    return args


def load_retain_images(path: Path, max_n: int, seed: int):
    if path is None:
        return None
    if path.is_file() and path.suffix == ".npz":
        data = np.load(path)
        key = "images" if "images" in data else data.files[0]
        arr = data[key]
        if len(arr) > max_n:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(arr), size=max_n, replace=False)
            arr = arr[idx]
        tensor = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.uint8))
        print(f"Loaded {len(tensor)} natural retain images from {path}")
        return tensor

    if not path.is_dir():
        raise FileNotFoundError(f"--retain-dir not found: {path}")

    files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise ValueError(f"No images found in {path}")

    rng = np.random.default_rng(seed)
    if len(files) > max_n:
        idx = rng.choice(len(files), size=max_n, replace=False)
        files = [files[i] for i in idx]

    imgs = []
    for p in files:
        try:
            with Image.open(p) as im:
                im = im.convert("RGB").resize((224, 224), Image.BICUBIC)
                imgs.append(np.asarray(im, dtype=np.uint8))
        except Exception:
            continue
    if not imgs:
        raise ValueError(f"Failed to decode any images from {path}")
    tensor = torch.from_numpy(np.stack(imgs, axis=0))
    print(f"Loaded {len(tensor)} natural retain images from {path}")
    return tensor


def trainable_parameters(model, train_blocks):
    if train_blocks >= 0:
        if train_blocks > len(model.blocks):
            raise ValueError(f"The model has only {len(model.blocks)} transformer blocks")
        model.requires_grad_(False)
        for block in model.blocks[len(model.blocks) - train_blocks :]:
            block.requires_grad_(True)
        model.norm.requires_grad_(True)
        model.head.requires_grad_(True)
        if getattr(model, "fc_norm", None) is not None:
            model.fc_norm.requires_grad_(True)
    return [param for param in model.parameters() if param.requires_grad]


def target_log_odds(scaled_logits, labels):
    target_indices = labels.unsqueeze(1)
    target_logits = scaled_logits.gather(1, target_indices).squeeze(1)
    other_logits = scaled_logits.scatter(1, target_indices, -torch.inf)
    return target_logits - other_logits.logsumexp(dim=1)


def target_score(logits, labels, temperature, objective):
    scaled_logits = logits.float() / temperature
    if objective == "log-odds":
        return target_log_odds(scaled_logits, labels)
    return -F.cross_entropy(scaled_logits, labels, reduction="none")


def npo_loss(student_score, reference_score, beta, weights=None):
    per_sample = (2.0 / beta) * F.softplus(beta * (student_score - reference_score))
    if weights is not None:
        return (per_sample * weights).mean()
    return per_sample.mean()


@torch.no_grad()
def cache_reference_scores(reference, images, labels, args, amp):
    result = []
    for start in range(0, len(images), args.batch):
        stop = start + args.batch
        with amp():
            logits = reference(preprocess(images[start:stop]))
        score = target_score(logits, labels[start:stop].to(DEVICE), args.temperature, args.objective)
        result.append(score.cpu())
    return torch.cat(result)


def forget_metrics(logits, reference_logits, labels):
    metrics = group_stats(logits, reference_logits, labels)
    student_nll = F.cross_entropy(logits, labels, reduction="none")
    metrics["ce_mean"] = student_nll.mean().item()
    metrics["ce_std"] = student_nll.std().item()
    reference_nll = F.cross_entropy(reference_logits, labels, reduction="none")
    odds_drop = target_log_odds(reference_logits, labels) - target_log_odds(logits, labels)
    quantile_levels = torch.tensor([0.50, 0.90, 0.99])
    for name, values in (("loss_change", student_nll - reference_nll), ("log_odds_drop", odds_drop)):
        for percentile, value in zip((50, 90, 99), torch.quantile(values, quantile_levels).tolist()):
            metrics[f"{name}_p{percentile}"] = value
    return metrics


def save_checkpoint(model, path):
    pending = path.with_suffix(".tmp")
    try:
        save_submission(model, pending)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def main():
    args = parse_args()
    images = load_forget_images(args.forget)
    if not len(images):
        raise ValueError("The forget set is empty")
    retain_images = load_retain_images(args.retain_dir, args.retain_max_n, args.seed)

    checkpoint_steps = {step for step in args.save_steps if step <= args.max_steps}
    checkpoint_steps.add(args.max_steps)
    args.out.mkdir(parents=True, exist_ok=False)
    config = vars(args) | {"save_steps": sorted(checkpoint_steps)}
    (args.out / "args.json").write_text(json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8")

    torch.manual_seed(args.seed)
    train_rng = torch.Generator().manual_seed(args.seed)
    eval_rng = torch.Generator().manual_seed(args.seed + 1)
    reference = load_model(args.model).eval().requires_grad_(False)
    student = load_model(args.model)
    params = trainable_parameters(student, args.train_blocks)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    use_bf16 = DEVICE == "cuda" and torch.cuda.is_bf16_supported()

    def amp():
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16)

    print(f"{len(images)} images, {sum(p.numel() for p in params):,} trainable parameters on {DEVICE}")
    print(
        f"Caching reference scores for {args.objective} NPO at T={args.temperature:g}, "
        f"beta={args.beta:g}, cons_weight={args.cons_weight:g}, nt_weight={args.nt_weight:g}, "
        f"retain={'natural' if retain_images is not None else 'mosaics+nt'}"
    )
    original_logits = predict(reference, images, args.eval_batch)
    labels = original_logits.argmax(dim=1)
    class_counts = torch.bincount(labels, minlength=original_logits.shape[1]).float()
    zero_class_mask = (class_counts == 0).to(DEVICE)
    class_weights = (class_counts / class_counts.max().clamp(min=1.0)).pow(args.class_weight_pow).to(DEVICE)
    if args.freeze_zero_head:
        class_weights[zero_class_mask] = 0.0
    print(
        f"Class-aware setup: pow={args.class_weight_pow}, "
        f"frozen_zero_classes={int(zero_class_mask.sum().item()) if args.freeze_zero_head else 0}"
    )
    reference_scores = cache_reference_scores(reference, images, labels, args, amp)

    eval_indices = torch.randperm(len(images), generator=eval_rng)[: args.eval_n]
    exact = images[eval_indices]
    mixed = alter(exact, "mix", seed=args.seed + 8000)
    proxy = make_mosaics(images, args.eval_proxy_n, eval_rng)
    eval_labels = labels[eval_indices]
    original_exact = original_logits[eval_indices]
    original_mixed = predict(reference, mixed, args.eval_batch)
    original_proxy = predict(reference, proxy, args.eval_batch)
    proxy_labels = original_proxy.argmax(dim=1)

    def evaluate(step):
        exact_logits = predict(student, exact, args.eval_batch) if step else original_exact
        mixed_logits = predict(student, mixed, args.eval_batch) if step else original_mixed
        proxy_logits = predict(student, proxy, args.eval_batch) if step else original_proxy
        result = {
            "step": step,
            "exact": forget_metrics(exact_logits, original_exact, eval_labels),
            "mix": forget_metrics(mixed_logits, original_mixed, eval_labels),
            "proxy": {
                "agreement": (proxy_logits.argmax(dim=1) == proxy_labels).float().mean().item(),
                "kl_orig_sub_T2": group_stats(proxy_logits, original_proxy, proxy_labels)["kl_orig_sub_T2"],
                "retain_proxy_score": retain_proxy_score(proxy_logits, original_proxy),
            },
        }
        with (args.out / "eval.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(result, allow_nan=False) + "\n")
        for group in ("exact", "mix"):
            row = result[group]
            wrong_conf = row.get("wrong_conf_T2")
            wrong_conf_text = f"{wrong_conf:.3f}" if wrong_conf is not None else "n/a"
            print(
                f"[eval {step:04d}] {group:5s} agreement={row['acc_vs_pseudo']:.3f} "
                f"CE mean={row['ce_mean']:.3f} std={row['ce_std']:.3f} "
                f"log-odds drop p50={row['log_odds_drop_p50']:+.4f} "
                f"p90={row['log_odds_drop_p90']:+.4f} p99={row['log_odds_drop_p99']:+.4f} "
                f"wrong_conf_T2={wrong_conf_text}",
                flush=True,
            )
        print(f"[eval {step:04d}] retain proxy={result['proxy']['retain_proxy_score']:.4f}", flush=True)

    evaluate(0)
    student.train()
    step = 0
    while step < args.max_steps:
        order = torch.randperm(len(images), generator=train_rng)
        for indices in order.split(args.batch):
            clean_batch_imgs = images[indices]
            batch_imgs = clean_batch_imgs
            aug_mask = None
            if args.aug_prob > 0:
                aug_mask = torch.rand(len(indices), generator=train_rng) < args.aug_prob
                if aug_mask.any():
                    batch_imgs = clean_batch_imgs.clone()
                    batch_imgs[aug_mask] = alter(
                        clean_batch_imgs[aug_mask], "mix", seed=args.seed + 10000 + step
                    )

            inputs = preprocess(batch_imgs)
            targets = labels[indices].to(DEVICE)
            with amp():
                logits = student(inputs)

            student_score = target_score(logits, targets, args.temperature, args.objective)
            ref_scores = reference_scores[indices].to(DEVICE)
            has_aug = aug_mask is not None and aug_mask.any()
            if has_aug:
                aug_mask_dev = aug_mask.to(DEVICE)
                with torch.no_grad(), amp():
                    aug_ref_logits = reference(inputs[aug_mask_dev])
                ref_scores = ref_scores.clone()
                ref_scores[aug_mask_dev] = target_score(
                    aug_ref_logits, targets[aug_mask_dev], args.temperature, args.objective
                )

            sample_weights = class_weights[targets] if args.class_weight_pow > 0 else None
            loss_forget = npo_loss(student_score, ref_scores, args.beta, weights=sample_weights)

            # Consistency loss between clean (stop_grad) and altered forget copies at T=2
            loss_cons = loss_forget.new_zeros(())
            if args.cons_weight > 0 and has_aug:
                clean_inputs_aug = preprocess(clean_batch_imgs[aug_mask])
                with torch.no_grad(), amp():
                    clean_logits_aug = student(clean_inputs_aug)
                loss_cons = F.kl_div(
                    F.log_softmax(logits[aug_mask_dev].float() / 2.0, dim=1),
                    F.log_softmax(clean_logits_aug.float() / 2.0, dim=1),
                    reduction="batchmean",
                    log_target=True,
                )

            # Hybrid retain anchor: mosaics (or natural images) + non-target KL on clean original images at T=2
            loss_mosaic = loss_forget.new_zeros(())
            loss_nt = loss_forget.new_zeros(())
            loss_proxy = loss_forget.new_zeros(())
            if args.proxy_weight > 0:
                with torch.no_grad():
                    if retain_images is not None:
                        r_idx = torch.randint(0, len(retain_images), (args.proxy_batch,), generator=train_rng)
                        proxy_inputs = preprocess(retain_images[r_idx])
                    else:
                        proxy_inputs = preprocess(make_mosaics(images, args.proxy_batch, train_rng))
                with torch.no_grad(), amp():
                    proxy_ref_logits = reference(proxy_inputs)
                with amp():
                    proxy_student_logits = student(proxy_inputs)
                loss_mosaic = F.kl_div(
                    F.log_softmax(proxy_student_logits.float() / 2.0, dim=1),
                    F.log_softmax(proxy_ref_logits.float() / 2.0, dim=1),
                    reduction="batchmean",
                    log_target=True,
                )

                if args.nt_weight > 0:
                    clean_inputs = preprocess(clean_batch_imgs)
                    with torch.no_grad(), amp():
                        nt_ref_logits = reference(clean_inputs).float()
                    with amp():
                        nt_sub_logits = student(clean_inputs).float()
                    mask = F.one_hot(targets, num_classes=nt_ref_logits.shape[1]).bool()
                    nt_ref_logits = nt_ref_logits.masked_fill(mask, -1e9)
                    nt_sub_logits = nt_sub_logits.masked_fill(mask, -1e9)
                    loss_nt = F.kl_div(
                        F.log_softmax(nt_sub_logits / 2.0, dim=1),
                        F.log_softmax(nt_ref_logits / 2.0, dim=1),
                        reduction="batchmean",
                        log_target=True,
                    )

                loss_proxy = loss_mosaic + args.nt_weight * loss_nt

            loss = loss_forget + args.proxy_weight * loss_proxy + args.cons_weight * loss_cons
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            if args.freeze_zero_head and student.head.weight.grad is not None:
                student.head.weight.grad[zero_class_mask] = 0.0
                if student.head.bias is not None and student.head.bias.grad is not None:
                    student.head.bias.grad[zero_class_mask] = 0.0
            optimizer.step()
            step += 1

            if step == 1 or step % 10 == 0:
                print(
                    f"step {step}/{args.max_steps} npo={loss_forget.item():.5f} "
                    f"mosaic_kl={loss_mosaic.item():.5f} nt_kl={loss_nt.item():.5f} "
                    f"cons_kl={loss_cons.item():.5f}",
                    flush=True,
                )
            if step in checkpoint_steps:
                checkpoint = args.out / f"step{step:04d}.pt"
                save_checkpoint(student, checkpoint)
                print(f"Checkpoint ready: {checkpoint}", flush=True)
                evaluate(step)
                student.train()
            if step == args.max_steps:
                break


if __name__ == "__main__":
    main()