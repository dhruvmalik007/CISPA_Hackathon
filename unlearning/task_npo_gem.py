"""Compare log-odds NPO and standard NPO for EAH image unlearning.

Put this script next to common.py and task_template.py. Run it with the
original model.pt and forget_images.npz from the challenge.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from common import alter, group_stats, make_mosaics, retain_proxy_score, save_submission
from task_template import DEVICE, load_forget_images, load_model, predict, preprocess


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("model.pt"))
    parser.add_argument("--forget", type=Path, default=Path("forget_images.npz"))
    parser.add_argument("--out", type=Path, required=True, help="New directory for checkpoints and metrics")
    parser.add_argument("--objective", choices=("log-odds", "standard"), default="log-odds")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, nargs="*", default=[10, 25, 50, 100, 200])
    parser.add_argument("--train-blocks", type=int, default=4, help="-1 = all parameters")
    parser.add_argument("--proxy-weight", type=float, default=0.25)
    parser.add_argument("--proxy-batch", type=int, default=32)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--eval-n", type=int, default=1000)
    parser.add_argument("--eval-proxy-n", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--aug-prob", type=float, default=0.0)   
    args = parser.parse_args()

    for name in ("temperature", "beta", "lr"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name} must be positive and finite")
    if not math.isfinite(args.proxy_weight) or args.proxy_weight < 0:
        parser.error("--proxy-weight must be nonnegative and finite")
    for name in ("batch", "max_steps", "proxy_batch", "eval_batch", "eval_n", "eval_proxy_n"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if any(step < 1 for step in args.save_steps):
        parser.error("--save-steps must contain positive step numbers")
    if args.train_blocks < -1:
        parser.error("--train-blocks must be -1 or nonnegative")
    if not math.isfinite(args.aug_prob) or not (0.0 <= args.aug_prob <= 1.0):
        parser.error("--aug-prob must be between 0 and 1")
    return args


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


def npo_loss(student_score, reference_score, beta):
    return (2.0 / beta) * F.softplus(beta * (student_score - reference_score)).mean()


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
    # Evaluate at T=1 regardless of the temperature used for training.
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
    print(f"Caching reference scores for {args.objective} NPO at T={args.temperature:g}, beta={args.beta:g}")
    original_logits = predict(reference, images, args.eval_batch)
    labels = original_logits.argmax(dim=1)
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
            batch_imgs = images[indices]
            aug_mask = None
            if args.aug_prob > 0:
                aug_mask = torch.rand(len(indices), generator=train_rng) < args.aug_prob
                if aug_mask.any():
                    batch_imgs = batch_imgs.clone()
                    batch_imgs[aug_mask] = alter(
                        batch_imgs[aug_mask], "mix", seed=args.seed + 10000 + step
                    )

            inputs = preprocess(batch_imgs)
            targets = labels[indices].to(DEVICE)
            with amp():
                logits = student(inputs)
            student_score = target_score(logits, targets, args.temperature, args.objective)

            ref_scores = reference_scores[indices].to(DEVICE)
            if aug_mask is not None and aug_mask.any():
                aug_mask_dev = aug_mask.to(DEVICE)
                with torch.no_grad(), amp():
                    aug_ref_logits = reference(inputs[aug_mask_dev])
                ref_scores = ref_scores.clone()
                ref_scores[aug_mask_dev] = target_score(
                    aug_ref_logits, targets[aug_mask_dev], args.temperature, args.objective
                )

            loss_forget = npo_loss(student_score, ref_scores, args.beta)

            loss_proxy = loss_forget.new_zeros(())
            if args.proxy_weight > 0:
                with torch.no_grad():
                    proxy_inputs = preprocess(make_mosaics(images, args.proxy_batch, train_rng))
                with torch.no_grad(), amp():
                    proxy_ref_logits = reference(proxy_inputs)
                with amp():
                    proxy_student_logits = student(proxy_inputs)
                loss_proxy = F.kl_div(
                    F.log_softmax(proxy_student_logits.float() / 2.0, dim=1),
                    F.log_softmax(proxy_ref_logits.float() / 2.0, dim=1),
                    reduction="batchmean",
                    log_target=True,
                )

            loss = loss_forget + args.proxy_weight * loss_proxy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1

            if step == 1 or step % 10 == 0:
                print(f"step {step}/{args.max_steps} npo={loss_forget.item():.5f} proxy_kl={loss_proxy.item():.5f}", flush=True)
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
