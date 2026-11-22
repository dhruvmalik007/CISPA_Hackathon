#!/usr/bin/env python3
"""Does per-prompt centering rescue the similarity signal on real data?

Motivation, from the data. Raw pairwise similarity is dominated by a prompt-level offset: some
prompts make every endpoint produce similar text (short answer-style prompts), others make every
endpoint diverge (long continuations). That offset is pure nuisance - it says nothing about which
endpoints share weights - yet it swamps the between-pair differences.

Centering removes it: for each prompt, subtract that prompt's mean similarity across all endpoint
pairs. A pair scores high only if it agrees *more than the other pairs on the same prompt*. Under
the null (no weight sharing) the centered scores are symmetric about zero by construction, which is
also what makes the sign-flip permutation test meaningful.

This is measured, not assumed. Each variant is scored with AUROC on real reference answers, using a
held-out prompt split so the comparison is not flattered by selecting prompts on the scored data.

Usage:
    python scripts/centered_eval.py --answers data/falcon2/ref_p2.jsonl
    python scripts/centered_eval.py --answers data/falcon2/ref_calib.jsonl --all-variants
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_real_answers(path: Path) -> dict[str, dict[str, str]]:
    by_prompt: dict[str, dict[str, str]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            endpoint = str(row.get("endpoint") or "")
            prompt_id = row.get("id")
            response = row.get("response")
            if not endpoint.startswith("ref-") or prompt_id is None or response is None:
                continue
            by_prompt[str(prompt_id)][endpoint] = str(response)
    return dict(by_prompt)


def load_truth(path: Path) -> dict[str, tuple[str, str]]:
    truth: dict[str, tuple[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            truth[row["endpoint"]] = (row["family"], row["model"])
    return truth


def auroc(same: np.ndarray, diff: np.ndarray) -> float:
    if same.size == 0 or diff.size == 0:
        return float("nan")
    wins = sum(float((diff < v).sum()) + 0.5 * float((diff == v).sum()) for v in same)
    return wins / (same.size * diff.size)


def toks(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def sim_multiset(a: str, b: str) -> float:
    ca, cb = Counter(toks(a)), Counter(toks(b))
    if not ca and not cb:
        return 1.0
    if not ca or not cb:
        return 0.0
    return 2 * sum((ca & cb).values()) / (sum(ca.values()) + sum(cb.values()))


def sim_first_tokens(a: str, b: str, k: int = 8) -> float:
    ta, tb = toks(a)[:k], toks(b)[:k]
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return sum(1 for x, y in zip(ta, tb) if x == y) / k


VARIANTS = {
    "raw": lambda grid: grid,
    "centered": lambda grid: grid - grid.mean(axis=0, keepdims=True),
    "zscored": lambda grid: (grid - grid.mean(axis=0, keepdims=True))
    / (grid.std(axis=0, keepdims=True) + 1e-12),
}


def similarity_matrix(by_prompt, endpoints, metric) -> np.ndarray:
    """n_endpoints x n_endpoints mean similarity, averaged over prompts."""
    n = len(endpoints)
    grid = np.zeros((n, n))
    for i, left in enumerate(endpoints):
        for j, right in enumerate(endpoints):
            if i == j:
                continue
            values = [metric(rows[left], rows[right]) for rows in by_prompt.values() if left in rows and right in rows]
            grid[i, j] = float(np.mean(values)) if values else np.nan
    return grid


def pair_scores(grid: np.ndarray, endpoints: list[str], variant: str) -> dict[tuple[str, str], float]:
    """Turn a symmetric matrix into per-pair scores, upper triangle only."""
    transformed = VARIANTS[variant](grid)
    scores: dict[tuple[str, str], float] = {}
    for i, left in enumerate(endpoints):
        for j in range(i + 1, len(endpoints)):
            scores[(left, endpoints[j])] = float(transformed[i, j])
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--truth", default=str(ROOT / "official" / "practice_answers.csv"))
    parser.add_argument("--metric", choices=["multiset", "first8"], default="multiset")
    parser.add_argument("--all-variants", action="store_true")
    args = parser.parse_args()

    answers_path = Path(args.answers)
    if not answers_path.is_absolute():
        answers_path = ROOT / args.answers
    truth = load_truth(Path(args.truth))
    endpoints = sorted(truth)
    data = load_real_answers(answers_path)
    paired = {p: r for p, r in data.items() if all(ep in r for ep in endpoints)}
    if len(paired) < 4:
        print(f"{answers_path.name}: only {len(paired)} fully-paired prompts", file=sys.stderr)
        return 2

    metric = sim_multiset if args.metric == "multiset" else sim_first_tokens
    is_same = {(a, b): truth[a][1] == truth[b][1] for i, a in enumerate(endpoints) for b in endpoints[i + 1 :]}

    # Held-out split: rank variants on even prompts, report on odd prompts.
    even = {p: paired[p] for p in sorted(paired)[0::2]}
    odd = {p: paired[p] for p in sorted(paired)[1::2]}

    print(f"{answers_path.name}: {len(paired)} prompts (split {len(even)}/{len(odd)}), metric={args.metric}\n")

    variants = ["raw", "centered", "zscored"] if args.all_variants else ["raw", "centered"]
    print(f"{'variant':<12}{'AUROC(all)':>12}{'AUROC(even)':>13}{'AUROC(odd)':>12}   direction")
    for variant in variants:
        row = []
        for subset in (paired, even, odd):
            grid = similarity_matrix(subset, endpoints, metric)
            scores = pair_scores(grid, endpoints, variant)
            same = np.array([scores[p] for p in scores if is_same[p]])
            diff = np.array([scores[p] for p in scores if not is_same[p]])
            row.append(auroc(same, diff))
        direction = "higher=same" if row[0] >= 0.5 else "higher=DIFF (inverted)"
        print(f"{variant:<12}{row[0]:>12.3f}{row[1]:>13.3f}{row[2]:>12.3f}   {direction}")

    print(
        "\nNote: `raw` and `centered` are on different scales, but AUROC is rank-based, so the "
        "comparison is fair. A variant only helps if it improves the held-out columns, not just `all`."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
