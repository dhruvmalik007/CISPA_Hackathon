"""Compare random and class-matched retain mosaics with standard NPO.

Put this script next to common.py and task_template.py. Run it with the
original model.pt and forget_images.npz from the challenge.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as TF

from common import alter, group_stats, kl, probs, save_submission
from task_template import DEVICE, load_forget_images, load_model, predict, preprocess


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("model.pt"))
    parser.add_argument("--forget", type=Path, default=Path("forget_images.npz"))
    parser.add_argument("--out", type=Path, required=True, help="New directory for checkpoints and metrics")
    parser.add_argument("--proxy-sampling", choices=("random", "class-matched"), required=True)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, nargs="*", default=[50, 100, 150, 200])
    parser.add_argument("--train-blocks", type=int, default=4, help="-1 = all parameters")
    parser.add_argument("--proxy-weight", type=float, default=0.25)
    parser.add_argument("--proxy-batch", type=int, default=32)
    parser.add_argument("--proxy-pool", type=int, default=4096, help="Number of training mosaics")
    parser.add_argument("--min-proxies-per-class", type=int, default=8, help="Minimum pool size for a class match")
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--eval-n", type=int, default=1000)
    parser.add_argument("--eval-proxy-n", type=int, default=1024, help="Number of separate evaluation mosaics")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--aug-prob", type=float, default=0.0)
    args = parser.parse_args()

    for name in ("temperature", "beta", "lr"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name} must be positive and finite")
    if not math.isfinite(args.proxy_weight) or args.proxy_weight <= 0:
        parser.error("--proxy-weight must be positive and finite")
    for name in ("batch", "max_steps", "proxy_batch", "proxy_pool", "min_proxies_per_class",
                 "eval_batch", "eval_n", "eval_proxy_n"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if any(step < 1 for step in args.save_steps):
        parser.error("--save-steps must contain positive step numbers")
    if args.train_blocks < -1:
        parser.error("--train-blocks must be -1 or nonnegative")
    if args.min_proxies_per_class > args.proxy_pool:
        parser.error("--min-proxies-per-class must be at most --proxy-pool")
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


def target_score(logits, labels, temperature):
    return -F.cross_entropy(logits.float() / temperature, labels, reduction="none")


def npo_loss(student_score, reference_score, beta):
    return (2.0 / beta) * F.softplus(beta * (student_score - reference_score)).mean()


@torch.no_grad()
def cache_reference_scores(reference, images, labels, args, amp):
    result = []
    for start in range(0, len(images), args.batch):
        stop = start + args.batch
        with amp():
            logits = reference(preprocess(images[start:stop]))
        score = target_score(logits, labels[start:stop].to(DEVICE), args.temperature)
        result.append(score.cpu())
    return torch.cat(result)


def forget_metrics(logits, reference_logits, labels):
    metrics = group_stats(logits, reference_logits, labels)
    # Evaluate at T=1 regardless of the temperature used for training.
    student_nll = F.cross_entropy(logits, labels, reduction="none")
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


def mosaic_pool_indices(num_images, train_size, eval_size, rng):
    """Sample unique ordered tile tuples, then split them into disjoint pools."""
    total = train_size + eval_size
    if total > num_images ** 4:
        raise ValueError("Too few source images for the requested disjoint mosaic pools")
    seen = {}
    while len(seen) < total:
        draws = torch.randint(num_images, (total - len(seen), 4), generator=rng)
        seen.update(dict.fromkeys(map(tuple, draws.tolist())))
    return torch.tensor(list(seen), dtype=torch.long).split((train_size, eval_size))


def render_mosaics(tiles, tile_indices):
    """Assemble cached quarter-size tiles using common.make_mosaics' layout."""
    panels = tiles[tile_indices]
    top = torch.cat((panels[:, 0], panels[:, 1]), dim=-1)
    bottom = torch.cat((panels[:, 2], panels[:, 3]), dim=-1)
    return torch.cat((top, bottom), dim=-2).permute(0, 2, 3, 1).contiguous()


def predict_mosaics(model, tiles, tile_indices, batch_size):
    return torch.cat([
        predict(model, render_mosaics(tiles, batch), batch_size)
        for batch in tile_indices.split(batch_size)
    ])


def sample_proxy_rows(labels, proxy_batch, train_labels, counts, offsets, sorted_rows,
                      min_count, mode, rng):
    # Cycle through shuffled forget labels if the retain batch is larger.
    desired = labels[torch.arange(proxy_batch) % len(labels)]
    draws = torch.rand(proxy_batch, dtype=torch.float64, generator=rng)
    rows = (draws * len(train_labels)).long()
    fallback = 0
    if mode == "class-matched":
        eligible = counts[desired] >= min_count
        fallback = (~eligible).sum().item()
        classes = desired[eligible]
        class_slots = (draws[eligible] * counts[classes]).long()
        rows[eligible] = sorted_rows[offsets[classes] + class_slots]
    matched = (train_labels[rows] == desired).sum().item()
    return rows, matched, fallback


def proxy_metrics(logits, reference_logits, proxy_labels, forget_class_mask):
    agreement = (logits.argmax(dim=1) == proxy_labels).float()
    drift = kl(probs(reference_logits), probs(logits))
    class_count = len(forget_class_mask)
    counts = torch.bincount(proxy_labels, minlength=class_count)
    represented = (counts > 0) & forget_class_mask

    def summary(avg_agreement, avg_kl):
        return {"agreement": avg_agreement, "kl_orig_sub_T2": avg_kl,
                "retain_proxy_score": avg_agreement / (1 + avg_kl)}

    result = {"proxy_random_eval": summary(agreement.mean().item(), drift.mean().item()),
              "proxy_class_matched_eval": None}
    if represented.any():
        weights = counts[represented]
        macro_agreement = (torch.bincount(proxy_labels, weights=agreement, minlength=class_count)[represented]
                           / weights).mean().item()
        macro_kl = (torch.bincount(proxy_labels, weights=drift, minlength=class_count)[represented]
                    / weights).mean().item()
        result["proxy_class_matched_eval"] = summary(macro_agreement, macro_kl) | {
            "n_classes": int(represented.sum()), "n_images": int(weights.sum())}
    return result


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
    pool_rng = torch.Generator().manual_seed(args.seed + 2)
    proxy_rng = torch.Generator().manual_seed(args.seed + 3)
    reference = load_model(args.model).eval().requires_grad_(False)
    student = load_model(args.model)
    params = trainable_parameters(student, args.train_blocks)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    use_bf16 = DEVICE == "cuda" and torch.cuda.is_bf16_supported()

    def amp():
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16)

    print(f"{len(images)} images, {sum(p.numel() for p in params):,} trainable parameters on {DEVICE}")
    print(f"Caching standard NPO reference scores at T={args.temperature:g}, beta={args.beta:g}", flush=True)
    original_logits = predict(reference, images, args.eval_batch)
    labels = original_logits.argmax(dim=1)
    reference_scores = cache_reference_scores(reference, images, labels, args, amp)

    train_tiles, eval_tiles = mosaic_pool_indices(len(images), args.proxy_pool, args.eval_proxy_n, pool_rng)
    height, width = images.shape[1:3]
    tiles = torch.cat([
        TF.resize(batch.permute(0, 3, 1, 2), [height // 2, width // 2], antialias=True)
        for batch in images.split(args.eval_batch)
    ])
    proxy_targets = []
    print(f"Labeling {len(train_tiles)} train and {len(eval_tiles)} held-out mosaics", flush=True)
    with torch.no_grad():
        for batch in train_tiles.split(args.eval_batch):
            with amp():
                logits = reference(preprocess(render_mosaics(tiles, batch)))
            proxy_targets.append(F.log_softmax(logits.float() / 2.0, dim=1).cpu())
    proxy_targets = torch.cat(proxy_targets)
    train_proxy_labels = proxy_targets.argmax(dim=1)
    class_count = original_logits.shape[1]
    forget_counts = torch.bincount(labels, minlength=class_count)
    train_counts = torch.bincount(train_proxy_labels, minlength=class_count)
    train_offsets = train_counts.cumsum(0) - train_counts
    sorted_rows = train_proxy_labels.argsort()
    forget_class_mask = forget_counts > 0
    eligible_classes = train_counts >= args.min_proxies_per_class
    covered_classes = int((eligible_classes & forget_class_mask).sum())
    covered_images = int(forget_counts[eligible_classes].sum())

    eval_indices = torch.randperm(len(images), generator=eval_rng)[: args.eval_n]
    exact = images[eval_indices]
    mixed = alter(exact, "mix", seed=args.seed + 8000)
    eval_labels = labels[eval_indices]
    original_exact = original_logits[eval_indices]
    original_mixed = predict(reference, mixed, args.eval_batch)
    original_proxy = predict_mosaics(reference, tiles, eval_tiles, args.eval_batch)
    proxy_labels = original_proxy.argmax(dim=1)
    eval_counts = torch.bincount(proxy_labels, minlength=class_count)
    coverage = {"forget_classes": int(forget_class_mask.sum()), "train_covered_classes": covered_classes,
                "train_covered_forget_fraction": covered_images / len(images),
                "eval_covered_classes": int(((eval_counts > 0) & forget_class_mask).sum()),
                "class_counts": [{"class": i, "forget": int(forget_counts[i]),
                                  "train_mosaics": int(train_counts[i]), "eval_mosaics": int(eval_counts[i])}
                                 for i in range(class_count)]}
    (args.out / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    print(f"Eligible matches for {covered_classes}/{coverage['forget_classes']} forget classes and "
          f"{covered_images}/{len(images)} forget images. Held-out mosaics cover "
          f"{coverage['eval_covered_classes']} forget classes.", flush=True)
    if args.proxy_sampling == "class-matched" and covered_classes == 0:
        raise ValueError("No forget class has enough matching train mosaics; see coverage.json")

    matched_total = 0
    fallback_total = 0

    def evaluate(step):
        exact_logits = predict(student, exact, args.eval_batch) if step else original_exact
        mixed_logits = predict(student, mixed, args.eval_batch) if step else original_mixed
        proxy_logits = predict_mosaics(student, tiles, eval_tiles, args.eval_batch) if step else original_proxy
        result = {
            "step": step,
            "exact": forget_metrics(exact_logits, original_exact, eval_labels),
            "mix": forget_metrics(mixed_logits, original_mixed, eval_labels),
            "train_proxy_match_fraction": matched_total / (step * args.proxy_batch) if step else None,
            "train_proxy_fallback_fraction": (fallback_total / (step * args.proxy_batch)
                                              if step and args.proxy_sampling == "class-matched" else None),
            **proxy_metrics(proxy_logits, original_proxy, proxy_labels, forget_class_mask),
        }
        with (args.out / "eval.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(result, allow_nan=False) + "\n")
        for group in ("exact", "mix"):
            row = result[group]
            wrong_conf = row.get("wrong_conf_T2")
            wrong_conf_text = f"{wrong_conf:.3f}" if wrong_conf is not None else "n/a"
            print(
                f"[eval {step:04d}] {group:5s} agreement={row['acc_vs_pseudo']:.3f} "
                f"log-odds drop p50={row['log_odds_drop_p50']:+.4f} "
                f"p90={row['log_odds_drop_p90']:+.4f} p99={row['log_odds_drop_p99']:+.4f} "
                f"wrong_conf_T2={wrong_conf_text}",
                flush=True,
            )
        matched_eval = result["proxy_class_matched_eval"]
        matched_score = f"{matched_eval['retain_proxy_score']:.4f}" if matched_eval else "n/a"
        match_fraction = result["train_proxy_match_fraction"]
        match_text = f"{match_fraction:.1%}" if match_fraction is not None else "n/a"
        print(f"[eval {step:04d}] retain proxy random={result['proxy_random_eval']['retain_proxy_score']:.4f} "
              f"matched macro={matched_score} train match rate={match_text}", flush=True)

    evaluate(0)
    student.train()
    step = 0
    while step < args.max_steps:
        order = torch.randperm(len(images), generator=train_rng)
        for indices in order.split(args.batch):
            inputs = preprocess(images[indices])
            batch_labels = labels[indices]
            with amp():
                logits = student(inputs)
            student_score = target_score(logits, batch_labels.to(DEVICE), args.temperature)
            loss_forget = npo_loss(student_score, reference_scores[indices].to(DEVICE), args.beta)

            rows, matched, fallback = sample_proxy_rows(batch_labels, args.proxy_batch, train_proxy_labels,
                                                        train_counts, train_offsets, sorted_rows,
                                                        args.min_proxies_per_class, args.proxy_sampling, proxy_rng)
            matched_total += matched
            fallback_total += fallback
            proxy_inputs = preprocess(render_mosaics(tiles, train_tiles[rows]))
            with amp():
                proxy_student_logits = student(proxy_inputs)
            loss_proxy = F.kl_div(
                F.log_softmax(proxy_student_logits.float() / 2.0, dim=1), proxy_targets[rows].to(DEVICE),
                reduction="batchmean", log_target=True,
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
