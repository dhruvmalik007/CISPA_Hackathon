"""Statistics for tiny-sample model-equality testing.

The budget (100 responses per endpoint, one answer per prompt) rules out the approaches in the
paper the organisers cite: those fit output *distributions* and need thousands of samples per
endpoint to estimate them. With one sample per prompt, the effective sample size is the number of
**prompts**, not responses — 100 at the very most, and fewer is safer.

So the design is deliberately paired. Every prompt goes to every endpoint, and each pair of
endpoints is compared on the *same* prompts. That removes prompt difficulty from the comparison
entirely: a hard prompt is hard for both endpoints, and only the residual difference is evidence.
The test is a paired sign-flip permutation test, which is exact for any n and makes no distributional
assumption — the right choice when n is 20-100 and the data are similarity scores in [0, 1].

Two quantities are reported for every pair:

    observed   mean paired difference in similarity (same-prompt pairs only)
    p_value    paired sign-flip permutation test against "these two endpoints agree as often as
               their own independent accuracy would predict"

A pair is called "same model" only when the p-value survives Benjamini-Hochberg across all pairs
in its family, at a false-discovery rate the caller controls. Multiple-testing correction is not
optional here: 528 pairs at alpha=0.05 would produce ~26 false positives by chance alone.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    """Matthews correlation coefficient, floored at 0 exactly as the server does.

    The denominator is zero when a prediction is constant (all same or all different), which the
    competition scores as `max(0, MCC)` of an undefined quantity; returning 0.0 matches the
    practical outcome and avoids a NaN poisoning downstream aggregation.
    """
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0.0
    return max(0.0, (tp * tn - fp * fn) / denominator)


def pair_counts(
    truth: dict[str, tuple[str, str]],
    guess: dict[str, tuple[str, str]],
    endpoints: Sequence[str] | None = None,
) -> tuple[int, int, int, int]:
    """(tp, tn, fp, fn) over unordered pairs, where "same" means same family AND same model."""
    names = sorted(set(endpoints) if endpoints is not None else set(truth) & set(guess))
    tp = tn = fp = fn = 0
    for a, b in itertools.combinations(names, 2):
        actual = truth[a] == truth[b]
        called = guess[a] == guess[b]
        if actual and called:
            tp += 1
        elif actual:
            fn += 1
        elif called:
            fp += 1
        else:
            tn += 1
    return tp, tn, fp, fn


def family_counts(
    truth: dict[str, tuple[str, str]],
    guess: dict[str, tuple[str, str]],
    endpoints: Sequence[str] | None = None,
) -> tuple[int, int, int, int]:
    """(tp, tn, fp, fn) over pairs, where "same" means same family only."""
    names = sorted(set(endpoints) if endpoints is not None else set(truth) & set(guess))
    tp = tn = fp = fn = 0
    for a, b in itertools.combinations(names, 2):
        actual = truth[a][0] == truth[b][0]
        called = guess[a][0] == guess[b][0]
        if actual and called:
            tp += 1
        elif actual:
            fn += 1
        elif called:
            fp += 1
        else:
            tn += 1
    return tp, tn, fp, fn


def competition_score(
    truth: dict[str, tuple[str, str]],
    guess: dict[str, tuple[str, str]],
    endpoints: Sequence[str] | None = None,
) -> dict:
    """The organisers' score: 0.1 * MCC_family + 0.9 * MCC_model.

    MCC_model is computed only over pairs that truly share a family, because pairs from different
    families are never the same model and the server does not count them there.
    """
    names = sorted(set(endpoints) if endpoints is not None else set(truth) & set(guess))
    ftp, ftn, ffp, ffn = family_counts(truth, guess, names)
    mcc_family = mcc(ftp, ftn, ffp, ffn)

    mtp = mtn = mfp = mfn = 0
    for a, b in itertools.combinations(names, 2):
        if truth[a][0] != truth[b][0]:
            continue
        actual = truth[a][1] == truth[b][1]
        called = guess[a][1] == guess[b][1] and guess[a][0] == guess[b][0]
        if actual and called:
            mtp += 1
        elif actual:
            mfn += 1
        elif called:
            mfp += 1
        else:
            mtn += 1
    mcc_model = mcc(mtp, mtn, mfp, mfn)
    return {
        "score": 0.1 * mcc_family + 0.9 * mcc_model,
        "mcc_family": mcc_family,
        "mcc_model": mcc_model,
        "family_pairs": {"tp": ftp, "tn": ftn, "fp": ffp, "fn": ffn},
        "model_pairs": {"tp": mtp, "tn": mtn, "fp": mfp, "fn": mfn},
    }


@dataclass
class PairedTest:
    """Result of the paired permutation test for one endpoint pair."""

    a: str
    b: str
    n: int
    observed: float
    p_value: float
    effect: float
    adjusted_p: float = float("nan")

    def as_dict(self) -> dict:
        return {
            "a": self.a,
            "b": self.b,
            "n": self.n,
            "observed": self.observed,
            "effect": self.effect,
            "p_value": self.p_value,
            "adjusted_p": self.adjusted_p,
        }


def paired_permutation_test(
    similarities: np.ndarray,
    rng: np.random.Generator,
    permutations: int = 2000,
) -> tuple[float, float]:
    """Sign-flip permutation test on the mean of paired similarities.

    Under the null ("these responses carry no more agreement than chance between an independent
    pair"), each paired difference is symmetric about 0, so flipping its sign leaves the
    distribution unchanged. This is exact for any n and needs no normality assumption.

    Returns (observed mean, two-sided p-value). The two-sided form matters because a pair can also
    be *less* similar than chance, which is evidence of different models and must not be ignored.
    """
    values = np.asarray(similarities, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = values.size
    if n == 0:
        return float("nan"), 1.0
    observed = float(values.mean())
    if n < 3:
        # Too few prompts for any permutation evidence; report the effect and no significance.
        return observed, 1.0

    signs = rng.choice(np.array([-1.0, 1.0]), size=(permutations, n))
    null = (signs * values).mean(axis=1)
    # +1 smoothing keeps the p-value from ever being exactly zero, which would make the
    # Benjamini-Hochberg step degenerate.
    extreme = np.sum(np.abs(null) >= abs(observed) - 1e-12)
    return observed, float((extreme + 1) / (permutations + 1))


def benjamini_hochberg(p_values: Sequence[float], fdr: float = 0.05) -> np.ndarray:
    """Step-up FDR control. Returns adjusted p-values in the input order.

    With 528 candidate pairs, uncorrected testing at 0.05 is guaranteed to invent "same" pairs.
    """
    values = np.asarray(p_values, dtype=np.float64)
    n = values.size
    if n == 0:
        return values
    order = np.argsort(values)
    ranked = values[order]
    scaled = ranked * n / np.arange(1, n + 1)
    # Enforce monotonicity from the largest p-value downwards.
    adjusted_ranked = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty(n, dtype=np.float64)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def _normalise(text: str) -> str:
    return " ".join(str(text).lower().split())


def _char_ngrams(text: str, n: int = 4) -> set[str]:
    text = _normalise(text)
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def similarity(a: str, b: str) -> float:
    """Similarity in [0, 1] designed to survive the competition's disguises.

    The disguises are watermark, system prompt, and decoding strategy; the signals that count are
    base model, quantization, and LoRA adapter. Character 4-gram Jaccard tolerates a watermark
    (a few inserted characters cost little) and a reordered system prompt, while still separating
    genuinely different continuations. It is deliberately not exact-match: exact match would call
    every watermarked pair "different", which is precisely the trap.

    Empty responses are treated as maximally uninformative: two empty strings are similar (both
    endpoints failed the same way), but an empty versus a non-empty is not.
    """
    left, right = _normalise(a), _normalise(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return _jaccard(_char_ngrams(left), _char_ngrams(right))


def build_similarity_tensor(
    by_prompt: dict[str, dict[str, str]],
    endpoints: Sequence[str],
) -> dict[tuple[str, str], np.ndarray]:
    """For each endpoint pair, the vector of per-prompt similarities, aligned across prompts.

    Only prompts answered by BOTH endpoints contribute, so the vector is genuinely paired. Prompts
    answered by one endpoint alone are wasted budget and are reported separately.
    """
    prompt_ids = sorted(by_prompt)
    pairs: dict[tuple[str, str], np.ndarray] = {}
    for a, b in itertools.combinations(sorted(endpoints), 2):
        values = []
        for pid in prompt_ids:
            responses = by_prompt[pid]
            if a in responses and b in responses:
                values.append(similarity(responses[a], responses[b]))
        pairs[(a, b)] = np.asarray(values, dtype=np.float64)
    return pairs


def coverage(by_prompt: dict[str, dict[str, str]], endpoints: Sequence[str]) -> dict:
    """How much usable paired data exists, per endpoint and overall.

    A missing answer is not a neutral event: it silently shrinks n for every pair involving that
    endpoint, which weakens exactly the tests that matter.
    """
    per_endpoint = {ep: 0 for ep in endpoints}
    for responses in by_prompt.values():
        for ep in endpoints:
            if ep in responses:
                per_endpoint[ep] += 1
    values = list(per_endpoint.values())
    return {
        "prompts": len(by_prompt),
        "per_endpoint": per_endpoint,
        "min": min(values) if values else 0,
        "max": max(values) if values else 0,
        "complete_prompts": sum(
            1 for responses in by_prompt.values() if all(ep in responses for ep in endpoints)
        ),
    }
