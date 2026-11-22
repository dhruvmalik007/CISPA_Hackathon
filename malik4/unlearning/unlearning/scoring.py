"""Gold-free scoring proxies for the unlearning challenge.

The real score needs a retrained "gold" model, a hidden control set, and hidden
altered copies. None of those are available. What *is* available is the original
model itself, so this module uses two proxies:

* **Loss target**: the original model's loss on altered copies of each forget image.
  An altered copy is closer to an image the original never memorised, so its loss
  distribution approximates what a retrained gold model would produce.
* **Prediction target**: the original model's temperature-2 distribution on a
  *different* alteration of the same image. The submission is rewarded when its
  distribution looks like that, not merely when its confidence drops.

These are guardrails for model selection. Only the public leaderboard split is
ground truth for the real forget/retain score.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from unlearning.task import NUM_CLASSES, predict_logits

EPS = 1e-8
SCORE_TEMPERATURE = 2.0


def softmax_temperature(logits: torch.Tensor, temperature: float = SCORE_TEMPERATURE) -> torch.Tensor:
    return torch.softmax(logits.float() / temperature, dim=-1)


def per_sample_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.float(), labels.long(), reduction="none")


def distribution_stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().flatten()
    if values.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    quantiles = torch.quantile(values, torch.tensor([0.10, 0.50, 0.90], dtype=torch.float32))
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "median": float(quantiles[1]),
        "p10": float(quantiles[0]),
        "p90": float(quantiles[2]),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Per-sample KL(p || q) for probability rows."""
    return (p * (p.clamp_min(EPS).log() - q.clamp_min(EPS).log())).sum(dim=-1)


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    middle = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, middle) + 0.5 * kl_divergence(q, middle)


def _clip01(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def _mean_entropy(probabilities: torch.Tensor) -> float:
    return float(-(probabilities.clamp_min(EPS).log() * probabilities).sum(dim=-1).mean())


def _targets(orig_alterations: Sequence[torch.Tensor], labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-image target loss and probabilities from the original model's alterations."""
    losses = torch.stack([per_sample_ce(logits, labels) for logits in orig_alterations], dim=0)
    probabilities = torch.stack([softmax_temperature(logits) for logits in orig_alterations], dim=0)
    return losses.mean(dim=0), probabilities.mean(dim=0)


def _score_group(
    orig_logits: torch.Tensor,
    sub_logits: torch.Tensor,
    labels: torch.Tensor,
    target_loss: torch.Tensor,
    target_probabilities: torch.Tensor,
) -> dict[str, Any]:
    orig_ce = per_sample_ce(orig_logits, labels)
    sub_ce = per_sample_ce(sub_logits, labels)
    stats_orig = distribution_stats(orig_ce)
    stats_sub = distribution_stats(sub_ce)
    stats_target = distribution_stats(target_loss)

    d_sub = abs(stats_sub["mean"] - stats_target["mean"]) + abs(stats_sub["std"] - stats_target["std"])
    d_orig = abs(stats_orig["mean"] - stats_target["mean"]) + abs(stats_orig["std"] - stats_target["std"])
    mean_std_score = 1.0 - _clip01(d_sub / max(d_orig, EPS))

    p_orig = softmax_temperature(orig_logits)
    p_sub = softmax_temperature(sub_logits)
    kl_target_sub = float(kl_divergence(target_probabilities, p_sub).mean())
    kl_target_orig = float(kl_divergence(target_probabilities, p_orig).mean())
    prediction_score = 1.0 - _clip01(kl_target_sub / max(kl_target_orig, EPS))

    agreement = float((orig_logits.argmax(dim=-1) == sub_logits.argmax(dim=-1)).float().mean())
    return {
        "n": int(len(labels)),
        "loss": {
            "orig": stats_orig,
            "sub": stats_sub,
            "target": stats_target,
            "delta_mean": stats_sub["mean"] - stats_orig["mean"],
            "delta_std": stats_sub["std"] - stats_orig["std"],
        },
        "mean_std_score": mean_std_score,
        "prediction_score": prediction_score,
        "group_score": mean_std_score * prediction_score,
        "kl_target_sub": kl_target_sub,
        "kl_target_orig": kl_target_orig,
        "top1_agreement_orig_sub": agreement,
        "top1_change_rate": 1.0 - agreement,
        "confidence": {
            "orig": float(p_orig.max(dim=-1).values.mean()),
            "sub": float(p_sub.max(dim=-1).values.mean()),
        },
        "entropy": {"orig": _mean_entropy(p_orig), "sub": _mean_entropy(p_sub)},
        "js_divergence": float(js_divergence(p_orig, p_sub).mean()),
        "top1_counts": {
            "orig": torch.bincount(orig_logits.argmax(dim=-1), minlength=NUM_CLASSES).tolist(),
            "sub": torch.bincount(sub_logits.argmax(dim=-1), minlength=NUM_CLASSES).tolist(),
        },
    }


def score_forget(
    orig_clean: torch.Tensor,
    sub_clean: torch.Tensor,
    labels: torch.Tensor,
    orig_alterations: Sequence[torch.Tensor],
    sub_alterations: Sequence[torch.Tensor],
    orig_target: torch.Tensor | None = None,
) -> dict[str, Any]:
    if orig_target is not None:
        target_loss = per_sample_ce(orig_target, labels)
        target_probabilities = softmax_temperature(orig_target)
    else:
        target_loss, target_probabilities = _targets(orig_alterations, labels)

    clean = _score_group(orig_clean, sub_clean, labels, target_loss, target_probabilities)

    if not sub_alterations:
        return {"clean": clean, "altered": None, "combined_score": clean["group_score"]}

    repeated_labels = labels.repeat(len(sub_alterations))
    sub_altered = torch.cat(list(sub_alterations), dim=0)
    orig_altered = torch.cat(list(orig_alterations), dim=0)
    altered = _score_group(
        orig_altered,
        sub_altered,
        repeated_labels,
        target_loss.repeat(len(sub_alterations)),
        target_probabilities.repeat(len(sub_alterations), 1),
    )

    return {
        "clean": clean,
        "altered": altered,
        "combined_score": 0.5 * clean["group_score"] + 0.5 * altered["group_score"],
    }


def score_retain(
    orig_logits: torch.Tensor,
    sub_logits: torch.Tensor,
    labels: torch.Tensor | None = None,
    class_map: Sequence[int] | torch.Tensor | None = None,
) -> dict[str, Any]:
    p_orig = softmax_temperature(orig_logits)
    p_sub = softmax_temperature(sub_logits)
    kl = float(kl_divergence(p_orig, p_sub).mean())
    kl_similarity = 1.0 / (1.0 + kl)
    agreement = float((orig_logits.argmax(dim=-1) == sub_logits.argmax(dim=-1)).float().mean())

    report: dict[str, Any] = {
        "n": int(len(orig_logits)),
        "kl_orig_sub": kl,
        "kl_similarity": kl_similarity,
        "top1_agreement_orig_sub": agreement,
        "confidence": {
            "orig": float(p_orig.max(dim=-1).values.mean()),
            "sub": float(p_sub.max(dim=-1).values.mean()),
        },
        "entropy": {"orig": _mean_entropy(p_orig), "sub": _mean_entropy(p_sub)},
        "js_divergence": float(js_divergence(p_orig, p_sub).mean()),
    }

    if labels is not None and class_map is not None:
        mapping = torch.as_tensor(class_map, dtype=torch.long)
        mapped = mapping[labels.long()]
        acc_orig = float((orig_logits.argmax(dim=-1) == mapped).float().mean())
        acc_sub = float((sub_logits.argmax(dim=-1) == mapped).float().mean())
        if acc_orig > 1e-6:
            accuracy_score = _clip01(acc_sub / acc_orig)
            accuracy_source = "mapped_labels"
        else:
            # A near-zero baseline means the class map or the auxiliary set is not usable.
            accuracy_score = agreement
            accuracy_source = "top1_agreement_fallback"
        report.update(
            accuracy_orig=acc_orig,
            accuracy_sub=acc_sub,
            accuracy_score=accuracy_score,
            accuracy_source=accuracy_source,
        )
    else:
        report.update(
            accuracy_orig=None,
            accuracy_sub=None,
            accuracy_score=agreement,
            accuracy_source="top1_agreement_proxy",
        )

    report["retain_score"] = report["accuracy_score"] * kl_similarity
    return report


def evaluate_logits(
    orig_clean: torch.Tensor,
    sub_clean: torch.Tensor,
    labels: torch.Tensor,
    orig_alterations: Sequence[torch.Tensor] | None = None,
    sub_alterations: Sequence[torch.Tensor] | None = None,
    orig_target: torch.Tensor | None = None,
    aux_orig: torch.Tensor | None = None,
    aux_sub: torch.Tensor | None = None,
    aux_labels: torch.Tensor | None = None,
    aux_class_map: Sequence[int] | None = None,
    labels_source: str = "pseudo",
) -> dict[str, Any]:
    original_alterations = list(orig_alterations or [])
    submission_alterations = list(sub_alterations or [])
    if not original_alterations:
        raise ValueError("at least one original alteration is required to build the target")

    forget = score_forget(
        orig_clean,
        sub_clean,
        labels,
        original_alterations,
        submission_alterations,
        orig_target=orig_target,
    )
    report: dict[str, Any] = {
        "labels_source": labels_source,
        "forget": forget,
        "retain": None,
        "warnings": [],
    }

    if aux_orig is not None and aux_sub is not None:
        report["retain"] = score_retain(aux_orig, aux_sub, aux_labels, aux_class_map)
    else:
        report["warnings"].append(
            "No in-distribution auxiliary set was supplied, so retain quality is unmeasured. "
            "The utility gate cannot be verified locally."
        )

    if report["retain"] is not None:
        report["proxy_score"] = forget["combined_score"] * report["retain"]["retain_score"]
        report["proxy_definition"] = "forget_proxy * retain_proxy"
    else:
        report["proxy_score"] = forget["combined_score"]
        report["proxy_definition"] = "forget_proxy only (retain unavailable)"
    report["warnings"].append(
        "Proxy only: the real score uses hidden gold models, hidden altered copies, a hidden "
        "control set, and a hidden validation gate. Use the public leaderboard split to rank."
    )
    return report


@torch.no_grad()
def evaluate_models(
    original: torch.nn.Module,
    submission: torch.nn.Module,
    forget_images: torch.Tensor,
    altered_by_variant: Sequence[torch.Tensor] | None = None,
    target_images: torch.Tensor | None = None,
    aux_images: torch.Tensor | None = None,
    aux_labels: torch.Tensor | None = None,
    aux_class_map: Sequence[int] | None = None,
    labels: torch.Tensor | None = None,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Compute every proxy metric for one submission."""
    orig_clean = predict_logits(original, forget_images, batch_size=batch_size)
    sub_clean = predict_logits(submission, forget_images, batch_size=batch_size)
    label_source = "pseudo"
    if labels is None:
        labels = orig_clean.argmax(dim=-1)
        label_source = "original_argmax"

    orig_alterations = [predict_logits(original, images, batch_size=batch_size) for images in (altered_by_variant or [])]
    sub_alterations = [predict_logits(submission, images, batch_size=batch_size) for images in (altered_by_variant or [])]
    orig_target = (
        predict_logits(original, target_images, batch_size=batch_size) if target_images is not None else None
    )

    aux_orig = aux_sub = None
    if aux_images is not None:
        aux_orig = predict_logits(original, aux_images, batch_size=batch_size)
        aux_sub = predict_logits(submission, aux_images, batch_size=batch_size)

    return evaluate_logits(
        orig_clean=orig_clean,
        sub_clean=sub_clean,
        labels=labels,
        orig_alterations=orig_alterations,
        sub_alterations=sub_alterations,
        orig_target=orig_target,
        aux_orig=aux_orig,
        aux_sub=aux_sub,
        aux_labels=aux_labels,
        aux_class_map=aux_class_map,
        labels_source=label_source,
    )


def flatten_scores(report: dict[str, Any]) -> dict[str, float]:
    """Small flat map for metrics.jsonl rows."""
    forget = report["forget"]
    flat = {
        "proxy_score": report["proxy_score"],
        "forget_score": forget["combined_score"],
        "forget_clean": forget["clean"]["group_score"],
        "forget_clean_mean_std": forget["clean"]["mean_std_score"],
        "forget_clean_prediction": forget["clean"]["prediction_score"],
        "forget_clean_loss_mean_sub": forget["clean"]["loss"]["sub"]["mean"],
        "forget_clean_loss_mean_orig": forget["clean"]["loss"]["orig"]["mean"],
        "forget_clean_loss_mean_target": forget["clean"]["loss"]["target"]["mean"],
        "forget_clean_loss_std_sub": forget["clean"]["loss"]["sub"]["std"],
        "forget_clean_top1_change": forget["clean"]["top1_change_rate"],
    }
    if forget.get("altered"):
        altered = forget["altered"]
        flat.update(
            forget_altered=altered["group_score"],
            forget_altered_mean_std=altered["mean_std_score"],
            forget_altered_prediction=altered["prediction_score"],
            forget_altered_loss_mean_sub=altered["loss"]["sub"]["mean"],
            forget_altered_top1_change=altered["top1_change_rate"],
        )
    if report.get("retain"):
        retain = report["retain"]
        flat.update(
            retain_score=retain["retain_score"],
            retain_acc_orig=retain["accuracy_orig"] if retain["accuracy_orig"] is not None else -1.0,
            retain_acc_sub=retain["accuracy_sub"] if retain["accuracy_sub"] is not None else -1.0,
            retain_kl=retain["kl_orig_sub"],
            retain_agreement=retain["top1_agreement_orig_sub"],
        )
    return {key: float(value) for key, value in flat.items()}
