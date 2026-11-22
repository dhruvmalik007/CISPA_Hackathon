#!/usr/bin/env python3
"""Turn collected answers into a submission: the production entry point.

Pipeline, all of which is exercised by `scripts/validate_offline.py` against the organisers' scorer:

    answers.jsonl  ->  paired similarity per endpoint pair  ->  paired sign-flip permutation test
                   ->  Benjamini-Hochberg FDR  ->  union-find components  ->  endpoint,family,model

Two decisions worth stating, because both are easy to get wrong:

* **Family vs model.** The score is 90% about distinguishing models *within* a family. A method that
  only clusters families caps out at 0.1. Here, connected components of the significant same-model
  edges define the model labels, and families are the components of the weaker family-level edges.
  With a single observed cluster this collapses to one family, which is correct for the practice
  set but would be wrong on 33 endpoints if the family threshold were not looser than the model one.

* **Missing answers are not neutral.** An endpoint short on budget silently shrinks n for every pair
  it belongs to. The coverage report is printed first, so a truncated collection is visible before
  its consequences are read as evidence.

Usage:
    python scripts/solve.py --answers data/ep_answers.jsonl --out out/submission.csv
    python scripts/solve.py --answers data/reference_answers.jsonl --out out/ref_answer.csv --repeats
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from met.client import AnswerStore, responses_by_prompt, write_submission  # noqa: E402
from met.stats import (  # noqa: E402
    PairedTest,
    benjamini_hochberg,
    build_similarity_tensor,
    coverage,
    paired_permutation_test,
)

# A pair is called same-model when its mean paired similarity exceeds this floor AND its
# FDR-adjusted p-value clears the threshold. The floor matters: without it, a pair that is
# uniformly dissimilar can still be "significantly" dissimilar, which is evidence of DIFFERENCE,
# not sameness.
SIMILARITY_FLOOR = 0.5

# Families are a coarser question than models (10% vs 90% of the score), so a looser floor is
# appropriate: two endpoints can share a base model while their outputs diverge under a LoRA
# adapter, and that pair must still be grouped into one family.
FAMILY_FLOOR = 0.35


def collect_tests(
    by_prompt: dict[str, dict[str, str]],
    endpoints: list[str],
    permutations: int = 2000,
    seed: int = 0,
) -> tuple[list[PairedTest], dict[tuple[str, str], np.ndarray]]:
    rng = np.random.default_rng(seed)
    tensors = build_similarity_tensor(by_prompt, endpoints)
    tests: list[PairedTest] = []
    for (a, b), values in sorted(tensors.items()):
        finite = values[np.isfinite(values)]
        observed, p_value = paired_permutation_test(finite, rng, permutations)
        effect = float(finite.mean() - SIMILARITY_FLOOR) if finite.size else float("nan")
        tests.append(
            PairedTest(a=a, b=b, n=int(finite.size), observed=observed, p_value=p_value, effect=effect)
        )
    return tests, tensors


def components(endpoints: list[str], edges: set[tuple[str, str]]) -> dict[str, str]:
    """Union-find over edges; returns endpoint -> component root."""
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
    return {ep: find(ep) for ep in endpoints}


def label(
    tests: list[PairedTest],
    endpoints: list[str],
    fdr: float = 0.05,
    family_fdr: float = 0.25,
    similarity_floor: float = SIMILARITY_FLOOR,
) -> tuple[dict[str, tuple[str, str]], dict]:
    """Return (labels, diagnostics).

    `family_fdr` is deliberately looser than `fdr`. The family question is only 10% of the score and
    is over all 528 pairs, so a missed family link costs little, while over-merging families would
    destroy the 90% model term by making cross-family "same model" claims.
    """
    adjusted = benjamini_hochberg([t.p_value for t in tests], fdr=fdr)
    for test, value in zip(tests, adjusted):
        test.adjusted_p = float(value)

    # Model edges: significant and clearly more similar than an unrelated pair.
    model_edges = {
        (t.a, t.b) for t in tests if t.adjusted_p <= fdr and t.observed > similarity_floor
    }
    # Family edges: same criterion but a looser FDR and floor, because family is the coarser question.
    family_store = []
    for test in tests:
        flat = PairedTest(**{**test.as_dict(), "adjusted_p": float("nan")})
        family_store.append(flat)
    family_adjusted = benjamini_hochberg([t.p_value for t in family_store], fdr=family_fdr)
    family_edges = {
        (t.a, t.b)
        for t, adj in zip(family_store, family_adjusted)
        if adj <= family_fdr and t.observed > FAMILY_FLOOR
    }
    # Every model edge is necessarily also a family edge.
    family_edges |= model_edges

    model_comp = components(endpoints, model_edges)
    family_comp = components(endpoints, family_edges)

    family_index: dict[str, str] = {}
    model_index: dict[str, str] = {}
    labels: dict[str, tuple[str, str]] = {}
    for ep in endpoints:
        froot = family_comp[ep]
        family = family_index.setdefault(froot, f"F{len(family_index)}")
        mroot = model_comp[ep]
        model = model_index.setdefault(mroot, f"{family}_M{len(model_index)}")
        labels[ep] = (family, model)

    diagnostics = {
        "pairs": len(tests),
        "significant_model_edges": len(model_edges),
        "significant_family_edges": len(family_edges),
        "families": len(set(labels[ep][0] for ep in endpoints)),
        "models": len(set(labels[ep][1] for ep in endpoints)),
    }
    return labels, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", required=True, help="answers JSONL produced by query")
    parser.add_argument("--out", required=True, help="submission CSV path")
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--family-fdr", type=float, default=0.25)
    parser.add_argument("--floor", type=float, default=SIMILARITY_FLOOR)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="also write a diagnostics JSON")
    args = parser.parse_args()

    store = AnswerStore(Path(args.answers))
    if not store.rows:
        print(f"no answers in {args.answers}", file=sys.stderr)
        return 2

    endpoints = sorted({row["endpoint"] for row in store.rows})
    by_prompt = responses_by_prompt(store)

    print(f"answers: {len(store.rows)}  endpoints: {len(endpoints)}  prompts: {len(by_prompt)}")
    cov = coverage(by_prompt, endpoints)
    print(
        f"coverage: {cov['complete_prompts']}/{cov['prompts']} prompts answered by every endpoint; "
        f"per-endpoint min {cov['min']} max {cov['max']}"
    )
    short = [ep for ep, n in cov["per_endpoint"].items() if n < cov["max"]]
    if short:
        print(f"  WARNING: uneven coverage on {', '.join(short)} - paired n is reduced there")

    tests, tensors = collect_tests(by_prompt, endpoints, args.permutations, args.seed)
    labels, diagnostics = label(
        tests, endpoints, fdr=args.fdr, family_fdr=args.family_fdr, similarity_floor=args.floor
    )

    rows = [(ep, labels[ep][0], labels[ep][1]) for ep in endpoints]
    out = write_submission(rows, Path(args.out))
    print(f"wrote {out}")
    print(json.dumps(diagnostics, indent=2))

    strongest = sorted(tests, key=lambda t: t.observed, reverse=True)[:5]
    print("most similar pairs:")
    for test in strongest:
        print(
            f"  {test.a} {test.b}  sim={test.observed:.3f} n={test.n} "
            f"p={test.p_value:.4f} adj={test.adjusted_p:.4f}"
        )

    if args.json:
        payload = {
            "answers": args.answers,
            "endpoints": endpoints,
            "coverage": cov,
            "diagnostics": diagnostics,
            "labels": {ep: {"family": f, "model": m} for ep, (f, m) in labels.items()},
            "tests": [t.as_dict() for t in tests],
        }
        path = out.with_suffix(".diagnostics.json")
        path.write_text(json.dumps(payload, indent=2))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
