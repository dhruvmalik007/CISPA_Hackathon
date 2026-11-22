"""Offline validation harness: does the method work on the real trap structure, without a key?

`practice_answers.csv` publishes the ground truth for `ref-01..ref-09`, including the *kind* of each
disguise:

    ref-01  plain                       (the reference)
    ref-02  quantized (fp8)              -> DIFFERENT model
    ref-03  system prompt                -> same model
    ref-04  LoRA at strength 0.7         -> DIFFERENT model
    ref-05  persona                      -> same model
    ref-06  watermark                    -> same model
    ref-07  LoRA at strength 1.5         -> DIFFERENT model
    ref-08  sampling at temperature 1.0  -> same model
    ref-09  LoRA at strength 0.3         -> DIFFERENT model

This is a valuable, non-obvious structure: **LoRA at any strength counts as a different model**, and
**four different disguises leave the model identical**. A method tuned on "does the text look
similar" will fail in both directions.

Because the API key is unavailable, this harness drives the payoff matrix with a *synthetic* text
generator whose similarity structure mirrors the published table. It validates the statistics,
the multiple-testing correction, and the decision rule — the parts that can be wrong
independently of the real model. It is NOT a substitute for running against `ref-01..ref-09`; it is
what makes that run worth doing, because a bug in the pipeline would otherwise burn the budget.
"""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

import numpy as np

from met.stats import (
    PairedTest,
    benjamini_hochberg,
    build_similarity_tensor,
    competition_score,
    mcc,
    paired_permutation_test,
)

# Similarity the *text* should show, from the published practice structure. Same-model disguises
# perturb text only slightly; different-model ones (quantization, any LoRA) change it materially.
SIMILARITY_PROFILE = {
    ("plain", "plain"): 1.00,
    ("plain", "system prompt"): 0.92,
    ("plain", "persona"): 0.88,
    ("plain", "watermark"): 0.90,
    ("plain", "sampling at temperature 1.0"): 0.85,
    ("system prompt", "persona"): 0.83,
    ("system prompt", "watermark"): 0.85,
    ("system prompt", "sampling at temperature 1.0"): 0.79,
    ("persona", "watermark"): 0.80,
    ("persona", "sampling at temperature 1.0"): 0.74,
    ("watermark", "sampling at temperature 1.0"): 0.77,
    ("quantized (fp8)", "quantized (fp8)"): 1.00,
    ("LoRA at strength 0.3", "LoRA at strength 0.3"): 1.00,
    ("LoRA at strength 0.7", "LoRA at strength 0.7"): 1.00,
    ("LoRA at strength 1.5", "LoRA at strength 1.5"): 1.00,
}

# Different-model pairs: a quantized or LoRA endpoint should look clearly unlike the plain model.
CROSS_MODEL = 0.45


def profile_similarity(kind_a: str, kind_b: str) -> float:
    """Text similarity the simulator should produce for a pair of disguises."""
    if (kind_a, kind_b) in SIMILARITY_PROFILE:
        return SIMILARITY_PROFILE[(kind_a, kind_b)]
    if (kind_b, kind_a) in SIMILARITY_PROFILE:
        return SIMILARITY_PROFILE[(kind_b, kind_a)]
    # One of them is a different-model variant (quantized or LoRA at any strength).
    return CROSS_MODEL


def read_practice_truth(path: Path) -> dict[str, tuple[str, str, str]]:
    """endpoint -> (family, model, kind)."""
    truth: dict[str, tuple[str, str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            endpoint = (row.get("endpoint") or "").strip()
            if not endpoint:
                continue
            truth[endpoint] = (
                (row.get("family") or "").strip(),
                (row.get("model") or "").strip(),
                (row.get("kind") or "").strip(),
            )
    return truth


VOCAB = (
    "the a of to and in is that for it as was with be by on not he this are or his from at "
    "which but have an they you were her she there one all we when your can said if do will "
    "each about how their then them these some her would make like him into time has look two "
    "more write go see number no way could people my than first water been call who oil its now "
    "find long down day did get come made may part"
).split()


def synthetic_dataset(
    truth: dict[str, tuple[str, str, str]],
    n_prompts: int = 40,
    words: int = 24,
    seed: int = 0,
) -> dict[str, dict[str, str]]:
    """Fabricate answers whose pairwise text similarity follows `SIMILARITY_PROFILE`.

    The construction is direct: each prompt has a *base* token sequence, and each endpoint emits
    that sequence with a number of tokens replaced, where the replacement rate is `1 - profile`.
    Two endpoints sharing the base therefore overlap in proportion to their profile similarity,
    which is exactly the property the real data is supposed to have.

    An earlier version drew tokens from a Dirichlet distribution and produced ~0.09 similarity for
    every pair regardless of profile, so it could not distinguish a working method from a broken one.
    """
    rng = np.random.default_rng(seed)
    endpoints = sorted(truth)
    kinds = {ep: truth[ep][2] for ep in endpoints}

    # The "plain" endpoint defines the common base continuation; every other endpoint keeps a
    # fraction of it given by its profile similarity to plain. Pairs that are both non-plain
    # (e.g. two same-model disguises) inherit their similarity through the shared base, which is
    # how the real models behave.
    anchors = [ep for ep in endpoints if kinds[ep] == "plain"]
    if not anchors:
        raise ValueError("practice truth has no 'plain' endpoint to anchor the synthetic base")
    anchor_kind = kinds[anchors[0]]

    def keep_rate(endpoint: str) -> float:
        return profile_similarity(kinds[endpoint], anchor_kind)

    by_prompt: dict[str, dict[str, str]] = {}
    for index in range(n_prompts):
        prompt_id = f"q{index:03d}"
        base = [VOCAB[i] for i in rng.integers(0, len(VOCAB), size=words)]
        row: dict[str, str] = {}
        for ep in endpoints:
            keep = keep_rate(ep)
            tokens = []
            for token in base:
                if rng.random() < keep:
                    tokens.append(token)
                else:
                    tokens.append(VOCAB[int(rng.integers(0, len(VOCAB)))])
            row[ep] = " ".join(tokens)
        by_prompt[prompt_id] = row
    return by_prompt


def decide(
    by_prompt: dict[str, dict[str, str]],
    endpoints: list[str],
    fdr: float = 0.05,
    permutations: int = 2000,
    seed: int = 0,
) -> tuple[dict[str, tuple[str, str]], list[PairedTest]]:
    """Run the full pipeline and return (labels, per-pair tests)."""
    rng = np.random.default_rng(seed)
    tensors = build_similarity_tensor(by_prompt, endpoints)

    tests: list[PairedTest] = []
    for (a, b), values in sorted(tensors.items()):
        observed, p_value = paired_permutation_test(values, rng, permutations)
        finite = values[np.isfinite(values)]
        effect = float(finite.mean() - 0.5) if finite.size else float("nan")
        tests.append(PairedTest(a=a, b=b, n=int(finite.size), observed=observed, p_value=p_value, effect=effect))

    adjusted = benjamini_hochberg([t.p_value for t in tests], fdr=fdr)
    for test, value in zip(tests, adjusted):
        test.adjusted_p = float(value)

    # Same-model edges: significant AND more similar than a random pair would be.
    edges = {(t.a, t.b) for t in tests if t.adjusted_p <= fdr and t.observed > 0.5}

    # Connected components over the same-model edges become model labels; every same-model pair
    # necessarily shares a family, so a single family label suffices for this practice set.
    parent = {ep: ep for ep in endpoints}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for a, b in edges:
        union(a, b)

    labels: dict[str, tuple[str, str]] = {}
    components: dict[str, list[str]] = {}
    for ep in endpoints:
        components.setdefault(find(ep), []).append(ep)
    for index, (_, members) in enumerate(sorted(components.items())):
        family = f"F{index}"
        for ep in members:
            labels[ep] = (family, f"{family}_M0" if len(members) > 1 else f"{family}_M_{ep}")
    return labels, tests


def evaluate_against_truth(
    truth: dict[str, tuple[str, str, str]],
    labels: dict[str, tuple[str, str]],
) -> dict:
    truth_pairs = {ep: (family, model) for ep, (family, model, _kind) in truth.items()}
    endpoints = sorted(truth)
    return competition_score(truth_pairs, labels, endpoints)
