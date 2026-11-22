"""SalUn for the 100-class ViT-B/16 challenge: weight saliency + soft runner-up distillation.

Paper: "SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image
Classification and Generation" (arXiv:2310.12508, ICLR 2024 Spotlight).

The paper specifies:

    weight saliency   m_S = 1( |grad_theta l_f(theta; D_f)|_{theta=theta_o} >= gamma )
                      gamma = median of the gradient (=> 50% sparsity)
    unlearned model   theta_u = m_S * (dtheta + theta_o) + (1 - m_S) * theta_o
    objective         min_dtheta  E_{(x,y)~D_f, y'!=y}[ CE(theta_u; x, y') ]
                                + alpha * E_{(x,y)~D_r}[ CE(theta_u; x, y) ]
    update            theta_u <- theta_u - eta * ( m_S * g )        (one SGD step per batch)

Four deliberate deviations, each configurable and recorded in `summary.json`:

1. **Soft runner-up distillation instead of uniform random labels.**
   The paper draws y' uniformly at random (y' != y). Here the target is deterministic and derived
   from the original model, so only the top-1 logit is suppressed and the runner-up structure
   survives:

       delta(c)  = delta_min + (count(c) - c_min) * (delta_max - delta_min) / (c_max - c_min)
       z_target  = z_orig;  z_target[top1] -= delta(count(top1))
       p_target  = softmax(z_target / T)
       l_f       = KL(p_target || softmax(z_student / T)) * T^2

   Rationale: the task scores KL(p_gold || p_sub) at T=2, so a T=2 soft target is metric-aligned.
   delta scales with how many forget images share that class, so heavily populated classes are
   pushed harder.

2. **Mask restricted to blocks.8..11 + norm + head, with a tunable keep ratio.**
   The paper thresholds the median over all parameters. Here patch_embed/pos_embed/cls_token and
   blocks.0..7 stay frozen, and gamma is the (1 - keep_ratio) quantile of |grad| inside the
   candidate set. Default keep_ratio 0.20 (80th percentile); sweep [0.15, 0.25, 0.50].

3. **Head gradient gating to the active classes.** head.weight/head.bias gradients are zeroed for
   the classes that have zero forget-set samples, so only classes actually present can move.

4. **Retain term = L2-SP anchor (+ optional CutMix KL).** D_r is not provided by the challenge, so
   the alpha * E_{D_r}[CE] term is replaced by ||theta - theta_o||^2 on masked parameters. An
   additional CutMix term anchors the student to the teacher on mixed forget pairs, which guards
   the inactive classes without needing D_r.

Mask caveat (tutorial page 9 / this cluster): JURECA does not cgroup GPU devices, so nothing here
assumes exclusive access to a device.

Usage:
    python -m unlearning.salun --compute-mask
    python -m unlearning.salun --keep-ratio 0.20 --lr 5e-3 --epochs 3 \
        --output-dir output/salun/k20
"""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
import torch.nn.functional as F

from unlearning.augment import DEFAULT_VARIANTS, build_altered_by_variant, build_target_copy
from unlearning.scoring import evaluate_logits, flatten_scores
from unlearning.task import (
    FORGET_FILENAME,
    MODEL_FILENAME,
    NUM_CLASSES,
    load_forget_images,
    load_model,
    predict_logits,
    preprocess,
    validate_submission,
)

# Candidate parameters for the saliency mask. Frozen: patch_embed, pos_embed, cls_token,
# blocks.0..7. Everything here is eligible, subject to the keep_ratio quantile.
MASK_BLOCK_START = 8
MASK_PREFIXES = ("blocks.", "norm.", "head.")

# Measured on all 3,500 forget images (reports/profile_all.json): active-class counts span 46..130,
# and the top-1 penalty delta interpolates across exactly that range.
COUNT_LOW = 46
COUNT_HIGH = 130

DEFAULTS: dict[str, Any] = {
    "data_dir": ".",
    "output_dir": "output/salun",
    "epochs": 3,
    "batch_size": 16,
    "eval_batch_size": 64,
    "lr": 5e-3,
    "weight_decay": 0.0,
    "anchor_weight": 0.5,
    "cutmix_weight": 0.5,
    "cutmix_prob": 1.0,
    "temperature": 2.0,
    "delta_min": 3.0,
    "delta_max": 7.0,
    "keep_ratio": 0.20,
    "class_gate": True,
    "forget_objective": "soft-runner-up",
    "projection": "fixed",
    "projection_ratio": 0.5,
    "mask_path": "",
    "compute_mask": False,
    "mask_batch_size": 8,
    "eval_limit": 512,
    "eval_steps": 0,
    "log_every": 20,
    "forget_loss_cap": 0.0,
    "min_retain_score": 0.0,
    "retain_min_agreement": 0.0,
    "aug_variants": list(DEFAULT_VARIANTS),
    "max_steps_per_epoch": 0,
    "grad_clip": 1.0,
    "seed": 0,
    "train_limit": 0,
    "scheduler": "none",
}


def _config(raw: dict[str, Any]) -> dict[str, Any]:
    merged = dict(DEFAULTS)
    merged.update({key: value for key, value in raw.items() if value is not None})
    return merged


# --------------------------------------------------------------------------------------
# masked parameter set
# --------------------------------------------------------------------------------------


def maskable_names(model: torch.nn.Module, block_start: int = MASK_BLOCK_START) -> list[str]:
    """Parameter names eligible for the saliency mask: blocks[block_start:] + norm + head."""
    names: list[str] = []
    for name, _parameter in model.named_parameters():
        if not name.startswith(MASK_PREFIXES):
            continue
        if name.startswith("blocks."):
            try:
                index = int(name.split(".")[1])
            except (IndexError, ValueError):
                continue
            if index < block_start:
                continue
        names.append(name)
    return names


def build_class_counts(pseudo_labels: torch.Tensor) -> torch.Tensor:
    """Per-class sample counts over the forget set, length NUM_CLASSES."""
    return torch.bincount(pseudo_labels.long(), minlength=NUM_CLASSES)


def delta_per_class(
    counts: torch.Tensor,
    delta_min: float = 3.0,
    delta_max: float = 7.0,
    count_low: int | None = COUNT_LOW,
    count_high: int | None = COUNT_HIGH,
) -> torch.Tensor:
    """Map class counts to the top-1 penalty delta.

    `delta` interpolates linearly from `delta_min` at `count_low` to `delta_max` at `count_high`.
    Both default to the measured active-class range over all 3,500 forget images (46..130), NOT to
    whatever subset happens to be in hand: a 100-image profile spans 1..8, and normalising to that
    would make the same class receive a different penalty at every subset size. Pass explicit
    bounds to change the mapping, or None to fall back to the observed range.
    """
    counts = counts.to(torch.float64)
    present = counts > 0
    if not bool(present.any()):
        return torch.full_like(counts, float(delta_min))

    observed = counts[present]
    low = float(observed.min()) if count_low is None else float(count_low)
    high = float(observed.max()) if count_high is None else float(count_high)
    if high <= low:
        scaled = torch.zeros_like(counts)
    else:
        scaled = ((counts - low) / (high - low)).clamp(0.0, 1.0)
    delta = float(delta_min) + scaled * (float(delta_max) - float(delta_min))
    delta[~present] = float(delta_min)
    return delta


def runner_up_target(
    teacher_logits: torch.Tensor,
    counts: torch.Tensor,
    temperature: float = 2.0,
    delta_min: float = 3.0,
    delta_max: float = 7.0,
    count_low: int | None = COUNT_LOW,
    count_high: int | None = COUNT_HIGH,
) -> torch.Tensor:
    """p_target = softmax(z_target / T) where only the top-1 logit is penalised by delta(class)."""
    deltas = delta_per_class(counts, delta_min, delta_max, count_low, count_high)
    top1 = teacher_logits.argmax(dim=-1)
    target = teacher_logits.clone()
    penalty = deltas[top1].to(target.dtype)
    # Subtract only at the top-1 position; every other logit keeps its runner-up structure.
    target.scatter_add_(
        -1, top1.unsqueeze(-1), (-penalty).unsqueeze(-1).to(target.dtype)
    )
    return F.softmax(target / float(temperature), dim=-1)


def compute_saliency_mask(
    model: torch.nn.Module,
    images: torch.Tensor,
    pseudo_labels: torch.Tensor,
    keep_ratio: float,
    device: torch.device,
    batch_size: int = 8,
    block_start: int = MASK_BLOCK_START,
) -> dict[str, Any]:
    """One backward pass at theta_o -> per-parameter boolean mask at a keep_ratio quantile."""
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")

    names = maskable_names(model, block_start)
    if not names:
        raise RuntimeError(f"no maskable parameters found under {MASK_PREFIXES}")

    # The teacher is frozen for inference, but the saliency mask is |grad| at theta_o, so
    # gradients must flow. Enable them here and restore the caller's state afterwards.
    previous = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    model.eval()
    model.zero_grad(set_to_none=True)

    used = 0
    for start in range(0, len(images), batch_size):
        chunk = preprocess(images[start : start + batch_size], device)
        labels = pseudo_labels[start : start + batch_size].to(device)
        logits = model(chunk)
        # The paper's forgetting loss for the mask is CE (Eq. 5, classification branch).
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        used += len(chunk)

    magnitudes: list[torch.Tensor] = []
    for name in names:
        parameter = dict(model.named_parameters())[name]
        grad = parameter.grad
        magnitudes.append(
            torch.zeros_like(parameter).abs().flatten()
            if grad is None
            else grad.detach().abs().flatten()
        )

    flat = torch.cat(magnitudes) if magnitudes else torch.zeros(1)
    gamma = float(torch.quantile(flat, 1.0 - float(keep_ratio)))

    masks: dict[str, torch.Tensor] = {}
    for name in names:
        parameter = dict(model.named_parameters())[name]
        grad = parameter.grad
        if grad is None:
            masks[name] = torch.zeros_like(parameter, dtype=torch.bool)
        else:
            masks[name] = (grad.detach().abs() >= gamma)

    total = sum(mask.numel() for mask in masks.values())
    kept = sum(int(mask.sum()) for mask in masks.values())
    model.zero_grad(set_to_none=True)

    # Restore the caller's requires_grad state so the teacher stays frozen for inference.
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(previous.get(name, False))

    return {
        "masks": masks,
        "gamma": gamma,
        "keep_ratio": float(keep_ratio),
        "block_start": int(block_start),
        "candidate_tensors": len(names),
        "candidate_elements": int(total),
        "masked_elements": int(kept),
        "effective_ratio": float(kept / total) if total else 0.0,
        "images_used": int(used),
        "gradient_quantile": float(1.0 - float(keep_ratio)),
    }


def class_gate_head(
    masks: dict[str, torch.Tensor],
    active_classes: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Zero head.weight rows / head.bias entries for classes absent from the forget set."""
    gated = {name: mask.clone() for name, mask in masks.items()}
    active = active_classes.to(torch.bool)
    for name in ("head.weight", "head.bias"):
        if name not in gated:
            continue
        mask = gated[name]
        if mask.ndim == 2:
            mask[~active, :] = False
        else:
            mask[~active] = False
        gated[name] = mask
    return gated


# --------------------------------------------------------------------------------------
# CutMix
# --------------------------------------------------------------------------------------


def cutmix_batch(
    images: torch.Tensor,
    generator: torch.Generator,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, float]:
    """CutMix two shuffled copies; returns (mixed uint8 images, lambda).

    `torch.distributions.Beta.sample()` draws from the global RNG and ignores a Generator, which
    would make runs non-reproducible. Beta is therefore sampled via two Gamma draws that do accept
    a generator.
    """
    index = torch.randperm(len(images), generator=generator)
    shuffled = images[index]

    if alpha > 0:
        gamma_a = torch._standard_gamma(torch.tensor(float(alpha)), generator=generator)
        gamma_b = torch._standard_gamma(torch.tensor(float(alpha)), generator=generator)
        lam = float(gamma_a / (gamma_a + gamma_b))
    else:
        lam = 1.0

    height, width = images.shape[1], images.shape[2]
    cut_ratio = math.sqrt(max(0.0, 1.0 - lam))
    cut_h, cut_w = int(height * cut_ratio), int(width * cut_ratio)
    if cut_h < 1 or cut_w < 1:
        return images, 1.0
    cy = int(torch.randint(0, height, (1,), generator=generator).item())
    cx = int(torch.randint(0, width, (1,), generator=generator).item())
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, height)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, width)

    mixed = images.clone()
    mixed[:, y1:y2, x1:x2, :] = shuffled[:, y1:y2, x1:x2, :]
    # Actual area replaced, which is the only correct lambda for the KL weight.
    lam = 1.0 - ((y2 - y1) * (x2 - x1) / (height * width))
    return mixed, float(lam)


# --------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------


def _batches(
    images: torch.Tensor,
    batch_size: int,
    generator: torch.Generator | None,
    max_steps: int = 0,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    order = torch.randperm(len(images), generator=generator)
    steps = 0
    for start in range(0, len(images), batch_size):
        index = order[start : start + batch_size]
        yield index, images[index]
        steps += 1
        if max_steps and steps >= max_steps:
            break


def _apply_mask_grads(model: torch.nn.Module, masks: dict[str, torch.Tensor]) -> None:
    """theta_u <- theta_u - eta * (m_S * g) is implemented by zeroing non-salient gradients."""
    named = dict(model.named_parameters())
    for name, parameter in named.items():
        if parameter.grad is None:
            continue
        mask = masks.get(name)
        if mask is None:
            parameter.grad.zero_()
        else:
            parameter.grad.mul_(mask.to(parameter.grad.dtype))


def _anchor(model: torch.nn.Module, original: dict[str, torch.Tensor], masks: dict[str, torch.Tensor]) -> torch.Tensor:
    total = None
    named = dict(model.named_parameters())
    for name, mask in masks.items():
        if name not in named or name not in original:
            continue
        delta = (named[name] - original[name]).pow(2) * mask.to(named[name].dtype)
        piece = delta.sum()
        total = piece if total is None else total + piece
    if total is None:
        return torch.zeros((), device=next(model.parameters()).device)
    return total


def run_salun(raw_config: dict[str, Any]) -> dict[str, Any]:
    config = _config(raw_config)
    data_dir = Path(config["data_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(config["seed"]))
    generator = torch.Generator().manual_seed(int(config["seed"]))

    teacher = load_model(data_dir / MODEL_FILENAME, device=device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    images = load_forget_images(data_dir / FORGET_FILENAME)
    if config["train_limit"]:
        images = images[: int(config["train_limit"])]
    teacher_logits = predict_logits(teacher, images, batch_size=int(config["eval_batch_size"]), device=device)
    pseudo_labels = teacher_logits.argmax(dim=-1)
    counts = build_class_counts(pseudo_labels)
    active = counts > 0

    objective = str(config["forget_objective"])
    if objective == "random-label":
        # The paper's own objective (RL.py: `targets = randint(0, num_classes)`). Kept as an
        # ablation arm so the soft runner-up deviation can be shown to help rather than merely
        # differ. Labels are resampled in the training loop each epoch, as the paper does.
        target_probs_all = None
        print(
            f"forget objective: random-label (paper baseline; {int(active.sum())} active classes)",
            flush=True,
        )
    else:
        target_probs_all = runner_up_target(
            teacher_logits,
            counts,
            temperature=float(config["temperature"]),
            delta_min=float(config["delta_min"]),
            delta_max=float(config["delta_max"]),
        )
        print(f"forget objective: {objective}", flush=True)
    print(
        f"forget set: {len(images)} images, {int(active.sum())} active classes "
        f"(counts {int(counts[active].min())}..{int(counts[active].max())})",
        flush=True,
    )

    # ---------------- mask: compute once, or load ----------------
    mask_path = Path(config["mask_path"]) if config["mask_path"] else output_dir / "mask.pt"
    if config["compute_mask"] or not mask_path.is_file():
        started = time.time()
        mask_info = compute_saliency_mask(
            teacher,
            images,
            pseudo_labels,
            keep_ratio=float(config["keep_ratio"]),
            device=device,
            batch_size=int(config["mask_batch_size"]),
        )
        masks = mask_info["masks"]
        if config["class_gate"]:
            masks = class_gate_head(masks, active)
            kept = sum(int(m.sum()) for m in masks.values())
            mask_info["masked_elements_after_gate"] = int(kept)
            mask_info["effective_ratio_after_gate"] = kept / max(mask_info["candidate_elements"], 1)
        mask_info["active_classes"] = int(active.sum())
        mask_info["computed_seconds"] = round(time.time() - started, 2)
        torch.save(
            {"masks": masks, "meta": {k: v for k, v in mask_info.items() if k != "masks"}},
            mask_path,
        )
        print(
            f"mask computed in {mask_info['computed_seconds']}s: gamma={mask_info['gamma']:.3e}, "
            f"kept {mask_info['masked_elements']}/{mask_info['candidate_elements']} "
            f"({mask_info['effective_ratio']:.3%}) -> {mask_path}",
            flush=True,
        )
    else:
        payload = torch.load(mask_path, map_location="cpu", weights_only=False)
        masks = {name: mask.to(device) for name, mask in payload["masks"].items()}
        mask_info = dict(payload.get("meta", {}))
        print(f"mask loaded from {mask_path} (gamma={mask_info.get('gamma')})", flush=True)

    masks = {name: mask.to(device) for name, mask in masks.items()}

    student = copy.deepcopy(teacher).to(device)
    for name, parameter in student.named_parameters():
        parameter.requires_grad_(name in masks)
    original = {
        name: parameter.detach().clone()
        for name, parameter in student.named_parameters()
        if name in masks
    }
    # Proximal projection works on the flat parameter vector in `named_parameters()` order, which
    # is the same order RL_pro.py concatenates.
    parameter_order = [name for name, _ in student.named_parameters()]
    trainable = [p for n, p in student.named_parameters() if n in masks]
    if not trainable:
        raise RuntimeError("no trainable parameters; check the mask")
    optimizer = torch.optim.SGD(
        trainable,
        lr=float(config["lr"]),
        momentum=0.0,
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = None
    if config["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(config["epochs"]), 1))
    print(f"trainable tensors: {len(trainable)}", flush=True)

    # ---------------- fixed evaluation subset ----------------
    if config["eval_limit"] and int(config["eval_limit"]) < len(images):
        pick = torch.randperm(len(images), generator=torch.Generator().manual_seed(int(config["seed"]) + 7))
        eval_index = pick[: int(config["eval_limit"])].sort().values
    else:
        eval_index = torch.arange(len(images))
    eval_images = images[eval_index]
    eval_labels = pseudo_labels[eval_index]
    variants = tuple(config["aug_variants"] or DEFAULT_VARIANTS)
    eval_altered = build_altered_by_variant(eval_images, variants=variants, seed=int(config["seed"]) + 17)
    eval_target = build_target_copy(eval_images, seed=int(config["seed"]) + 31)
    orig_eval_clean = predict_logits(teacher, eval_images, batch_size=int(config["eval_batch_size"]), device=device)
    orig_eval_altered = [
        predict_logits(teacher, item, batch_size=int(config["eval_batch_size"]), device=device)
        for item in eval_altered
    ]
    orig_eval_target = predict_logits(teacher, eval_target, batch_size=int(config["eval_batch_size"]), device=device)

    def evaluate(model: torch.nn.Module) -> dict[str, Any]:
        return evaluate_logits(
            orig_clean=orig_eval_clean,
            sub_clean=predict_logits(model, eval_images, batch_size=int(config["eval_batch_size"]), device=device),
            labels=eval_labels,
            orig_alterations=orig_eval_altered,
            sub_alterations=[
                predict_logits(model, item, batch_size=int(config["eval_batch_size"]), device=device)
                for item in eval_altered
            ],
            orig_target=orig_eval_target,
            labels_source="original_argmax",
        )

    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("")
    # Start each run with a fresh step trace so an old run's lines cannot be mistaken for this one.
    (output_dir / "train.log").write_text("")
    history: list[dict[str, Any]] = []

    baseline_report = evaluate(student)
    baseline_scores = flatten_scores(baseline_report)
    best_state = {n: t.detach().cpu().float().clone() for n, t in student.state_dict().items()}
    best = {
        "score": baseline_scores.get("proxy_score", 0.0),
        "epoch": 0,
        "step": 0,
        "report": baseline_report,
    }
    torch.save(best_state, output_dir / "best.pt")
    (output_dir / "best_eval.json").write_text(json.dumps(baseline_report, indent=2))
    print(f"baseline proxy: {baseline_scores.get('proxy_score', 0.0):.6f}", flush=True)

    def consider(report: dict[str, Any], epoch: int, step: int) -> dict[str, float]:
        nonlocal best
        scores = flatten_scores(report)
        retain = scores.get("retain_score")
        floor = float(config["min_retain_score"])
        if floor and retain is not None and retain < floor:
            print(f"epoch {epoch} step {step}: retain {retain:.4f} < floor {floor}", flush=True)
            return scores
        if scores.get("proxy_score", -math.inf) > best["score"]:
            best = {
                "score": scores["proxy_score"],
                "epoch": epoch,
                "step": step,
                "report": report,
            }
            torch.save(
                {n: t.detach().cpu().float().clone() for n, t in student.state_dict().items()},
                output_dir / "best.pt",
            )
            (output_dir / "best_eval.json").write_text(json.dumps(report, indent=2))
            print(f"new best at epoch {epoch} step {step}: {best['score']:.4f}", flush=True)
        return scores

    temperature = float(config["temperature"])
    stopped = False
    for epoch in range(1, int(config["epochs"]) + 1):
        started = time.time()
        student.train()
        losses: list[float] = []
        steps = 0
        total_batches = max(
            1,
            min(
                math.ceil(len(images) / max(1, int(config["batch_size"]))),
                int(config["max_steps_per_epoch"]) or math.inf,
            ),
        )
        print(
            f"epoch {epoch}/{int(config['epochs'])} starting: "
            f"{len(images)} images, {total_batches} steps, batch {int(config['batch_size'])}",
            flush=True,
        )

        for index, batch in _batches(
            images,
            int(config["batch_size"]),
            generator,
            int(config["max_steps_per_epoch"]),
        ):
            student.zero_grad(set_to_none=True)
            cutmix_value: float | None = None

            inputs = preprocess(batch, device)
            logits = student(inputs)

            if target_probs_all is None:
                # Paper objective: uniform random target, plain CE, resampled every batch so a
                # sample sees many different random labels across the run (as RL.py does).
                target = torch.randint(
                    0, NUM_CLASSES, (len(batch),), generator=generator
                ).to(device)
                forget_loss = F.cross_entropy(logits, target)
            else:
                # 1. Soft runner-up distillation on the forget batch.
                target = target_probs_all[index].to(device)
                log_student = F.log_softmax(logits / temperature, dim=-1)
                forget_loss = F.kl_div(log_student, target, reduction="batchmean") * temperature**2

            # 2. L2-SP anchor as the retain surrogate (D_r is unavailable).
            anchor = _anchor(student, original, masks)
            total_loss = forget_loss + float(config["anchor_weight"]) * anchor

            # 3. CutMix KL: anchor the teacher's T=2 distribution on mixed forget pairs,
            #    which covers the regions a single forget image does not exercise.
            if float(config["cutmix_weight"]) > 0 and config["cutmix_prob"] >= 1.0:
                mixed, lam = cutmix_batch(batch, generator)
                with torch.no_grad():
                    # predict_logits returns CPU tensors by design (it mirrors the scoring
                    # server), so move them to the device before building the loss.
                    teacher_mixed = predict_logits(
                        teacher, mixed, batch_size=len(mixed), device=device
                    ).to(device)
                student_mixed = student(preprocess(mixed, device))
                cutmix_loss = F.kl_div(
                    F.log_softmax(student_mixed / temperature, dim=-1),
                    F.softmax(teacher_mixed / temperature, dim=-1),
                    reduction="batchmean",
                ) * temperature**2
                cutmix_value = float(cutmix_loss.detach())
                total_loss = total_loss + float(config["cutmix_weight"]) * cutmix_loss * lam

            total_loss.backward()

            # 4. theta_u <- theta_u - eta * (m_S * g)
            _apply_mask_grads(student, masks)
            if float(config["grad_clip"]):
                torch.nn.utils.clip_grad_norm_(trainable, float(config["grad_clip"]))
            optimizer.step()
            if str(config["projection"]) == "proximal":
                # RL_pro.py lines 53-62: snap every coordinate whose cumulative drift is below
                # the top-ratio |theta - theta0| back to theta0. Sparsity grows over training
                # instead of being fixed at the start.
                total_steps = max(1, int(config["epochs"])) * max(1, len(images) // int(config["batch_size"]))
                done = (epoch - 1) * max(1, len(images) // int(config["batch_size"])) + steps
                shrink = float(config["projection_ratio"]) * (total_steps - done) / total_steps
                with torch.no_grad():
                    named = dict(student.named_parameters())
                    flat_student = torch.cat([named[n].detach().view(-1) for n in parameter_order])
                    flat_original = torch.cat(
                        [
                            (original[n] if n in original else named[n].detach()).view(-1)
                            for n in parameter_order
                        ]
                    )
                    diff = flat_student - flat_original
                    keep_count = int(max(0.0, shrink) * diff.numel())
                    if keep_count > 0:
                        threshold = torch.topk(diff.abs(), keep_count).values.min()
                        projected = torch.where(
                            diff > threshold,
                            flat_student - threshold,
                            torch.where(
                                diff < -threshold, flat_student + threshold, flat_original
                            ),
                        )
                        offset = 0
                        for parameter in student.parameters():
                            size = parameter.numel()
                            parameter.copy_(projected[offset : offset + size].view(parameter.shape))
                            offset += size

            # Non-salient weights must stay exactly at theta_o.
            with torch.no_grad():
                for name, parameter in student.named_parameters():
                    if name not in masks:
                        parameter.copy_(original.get(name, parameter))
            losses.append(float(total_loss.detach()))
            steps += 1

            # Per-step trace so the run is followable live in the Slurm log (`tail -f`).
            # Logged every step for the first epoch and every `log_every` steps afterwards, so the
            # log stays readable on a long run without hiding the start.
            log_every = int(config["log_every"])
            if epoch == 1 or (log_every and steps % log_every == 0) or steps == total_batches:
                recent = losses[-log_every:] if log_every else losses
                anchor_value = float(anchor.detach()) if anchor.numel() == 1 else float(anchor.detach().mean())
                print(
                    f"  epoch {epoch}/{int(config['epochs'])} "
                    f"step {steps}/{total_batches} "
                    f"loss {float(total_loss.detach()):.4f} "
                    f"(forget {float(forget_loss.detach()):.4f}, "
                    f"anchor {anchor_value:.4f}"
                    + (f", cutmix {float(cutmix_loss.detach()):.4f}" if cutmix_value is not None else "")
                    + f") "
                    f"avg{len(recent)} {sum(recent) / len(recent):.4f} "
                    f"lr {float(optimizer.param_groups[0]['lr']):.2e} "
                    f"elapsed {time.time() - started:.0f}s",
                    flush=True,
                )
                log_path = output_dir / "train.log"
                try:
                    with log_path.open("a") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "epoch": epoch,
                                    "step": steps,
                                    "total_steps": total_batches,
                                    "loss": round(float(total_loss.detach()), 6),
                                    "forget_loss": round(float(forget_loss.detach()), 6),
                                    "anchor": round(anchor_value, 6),
                                    "cutmix": None if cutmix_value is None else round(cutmix_value, 6),
                                    "lr": float(optimizer.param_groups[0]["lr"]),
                                    "seconds": round(time.time() - started, 2),
                                }
                            )
                            + "\n"
                        )
                except OSError:
                    pass

            eval_steps = int(config["eval_steps"])
            if eval_steps and steps % eval_steps == 0:
                student.eval()
                report = evaluate(student)
                consider(report, epoch, steps)
                student.train()

        if scheduler is not None:
            scheduler.step()

        student.eval()
        report = evaluate(student)
        scores = consider(report, epoch, 0)
        row = {
            "epoch": epoch,
            "steps": steps,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "loss_mean": sum(losses) / len(losses) if losses else None,
            "seconds": round(time.time() - started, 2),
            "scores": scores,
        }
        history.append(row)
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(
            f"epoch {epoch}: proxy={scores.get('proxy_score', 0.0):.4f} "
            f"retain={scores.get('retain_score')} loss={row['loss_mean']}",
            flush=True,
        )

        cap = float(config["forget_loss_cap"])
        if cap and scores.get("forget_clean_loss_mean_sub") is not None:
            if scores["forget_clean_loss_mean_sub"] > cap * max(scores.get("forget_clean_loss_mean_target", 1.0), 1e-6):
                print("forget loss cap exceeded; stopping", flush=True)
                stopped = True
                break

    # ---------------- finalise ----------------
    torch.save({n: t.detach().cpu().float().clone() for n, t in student.state_dict().items()}, output_dir / "last.pt")

    # If nothing beat the untrained baseline, `best.pt` is still theta_o, so submission.pt ships an
    # unmodified model that would score exactly 0. That is the correct choice on the proxy, but it
    # must never look like a successful unlearning run, so it is reported explicitly.
    kept_baseline = best["epoch"] == 0

    final_state = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
    torch.save(final_state, output_dir / "submission.pt")
    torch.save(pseudo_labels, output_dir / "pseudo_labels.pt")

    validation = validate_submission(output_dir / "submission.pt")
    original_state = {name: tensor.detach().cpu() for name, tensor in teacher.state_dict().items()}
    identical = all(
        torch.equal(final_state[name].float(), original_state[name].float())
        for name in original_state
        if name in final_state
    )
    submission_mb = round(validation["file_bytes"] / (1024 * 1024), 2)
    summary = {
        "method": f"SalUn + {config['forget_objective']} ({config['projection']} mask)",
        "paper": "arXiv:2310.12508",
        "config": {k: v for k, v in config.items() if k != "aug_variants"} | {"aug_variants": list(variants)},
        "mask": {k: v for k, v in mask_info.items() if k != "masks"},
        "classes": {
            "active": int(active.sum()),
            "counts_min": int(counts[active].min()) if bool(active.any()) else 0,
            "counts_max": int(counts[active].max()) if bool(active.any()) else 0,
        },
        "history": history,
        "best": {"epoch": best["epoch"], "step": best["step"], "proxy_score": best["score"]},
        "kept_untrained_baseline": kept_baseline,
        "submission_identical_to_original": bool(identical),
        "stopped_early": stopped,
        "submission": validation | {"size_mb": submission_mb},
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if kept_baseline:
        print(
            "WARNING: no epoch beat the untrained baseline proxy, so submission.pt is the unmodified "
            "model and would score exactly 0. The proxy is a noisy local stand-in; check history "
            "before reading this as a failure of the method.",
            flush=True,
        )
    return summary


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--anchor-weight", type=float, default=None)
    parser.add_argument("--cutmix-weight", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--delta-min", type=float, default=None)
    parser.add_argument("--delta-max", type=float, default=None)
    parser.add_argument("--keep-ratio", type=float, default=None)
    parser.add_argument(
        "--forget-objective",
        choices=["soft-runner-up", "random-label"],
        default=None,
        help="soft-runner-up (default) or random-label (the paper's own objective)",
    )
    parser.add_argument(
        "--projection",
        choices=["fixed", "proximal"],
        default=None,
        help="fixed mask (default) or RL_pro-style proximal L0 projection",
    )
    parser.add_argument("--projection-ratio", type=float, default=None)
    parser.add_argument("--mask-path", default=None)
    parser.add_argument("--compute-mask", action="store_true", default=None)
    parser.add_argument("--class-gate", dest="class_gate", action="store_true", default=None)
    parser.add_argument("--no-class-gate", dest="class_gate", action="store_false", default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="log every N steps after epoch 1 (epoch 1 always logs every step)",
    )
    parser.add_argument("--forget-loss-cap", type=float, default=None)
    parser.add_argument("--min-retain-score", type=float, default=None)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    args = parser.parse_args()

    config: dict[str, Any] = {}
    for key, value in vars(args).items():
        if value is None:
            continue
        config[key] = value

    summary = run_salun(config)
    print(
        json.dumps(
            {
                "best_epoch": summary["best"]["epoch"],
                "proxy_score": summary["best"]["proxy_score"],
                "submission_ok": summary["submission"]["ok"],
                "mask": summary["mask"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
