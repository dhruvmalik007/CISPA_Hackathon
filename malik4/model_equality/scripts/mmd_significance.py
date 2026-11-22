#!/usr/bin/env python3
"""Is the MMD separation statistically significant, and how does it scale with repeats?

The partial run showed MMD reaching AUROC 0.738 on held-out prompts once there were 8 repeats per
prompt, where every earlier test had ~1. That is either a real power effect or noise on 4 held-out
prompts. This script answers it properly:

  * permutation test on the same-vs-different AUROC, so a lucky split cannot pass
  * learning curve in the number of repeats, which is the actionable question: how many repeats does
    the method need before it works? That number is what a submission should spend.

If the curve keeps climbing with repeats, the earlier "no signal" conclusion was a power artefact and
the method is viable - which would be a material correction to my previous answer.

Usage:
    python scripts/mmd_significance.py --answers data/ref_power.jsonl
    python scripts/mmd_significance.py --answers data/ref_power.jsonl --permutations 2000
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_pooled(path: Path) -> dict[str, dict[str, list[str]]]:
    pooled: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
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
            pooled[str(prompt_id)][endpoint].append(str(response))
    return {pid: dict(eps) for pid, eps in pooled.items()}


def load_truth(path: Path) -> dict[str, tuple[str, str]]:
    truth: dict[str, tuple[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            truth[row["endpoint"]] = (row["family"], row["model"])
    return truth


def toks(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def mmd2(a: list[str], b: list[str]) -> float:
    """Unbiased MMD^2 with a Jaccard token-set kernel. Lower means same model."""
    sa = [toks(x) for x in a]
    sb = [toks(x) for x in b]

    def k(u: set[str], v: set[str]) -> float:
        if not u and not v:
            return 1.0
        if not u or not v:
            return 0.0
        return len(u & v) / len(u | v)

    m, n = len(sa), len(sb)
    if m < 2 or n < 2:
        return float("nan")
    wa = sum(k(sa[i], sa[j]) for i in range(m) for j in range(m) if i != j) / (m * (m - 1))
    wb = sum(k(sb[i], sb[j]) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    cr = sum(k(x, y) for x in sa for y in sb) / (m * n)
    return wa + wb - 2 * cr


def auroc_lower_is_same(same: np.ndarray, diff: np.ndarray) -> float:
    same = same[np.isfinite(same)]
    diff = diff[np.isfinite(diff)]
    if same.size == 0 or diff.size == 0:
        return float("nan")
    wins = sum(float((diff > v).sum()) + 0.5 * float((diff == v).sum()) for v in same)
    return wins / (same.size * diff.size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", default="data/ref_power.jsonl")
    parser.add_argument("--truth", default=str(ROOT / "official" / "practice_answers.csv"))
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    answers = Path(args.answers)
    if not answers.is_absolute():
        answers = ROOT / answers
    truth = load_truth(Path(args.truth))
    endpoints = sorted(truth)
    pooled = load_pooled(answers)
    usable = {p: e for p, e in pooled.items() if all(ep in e for ep in endpoints)}
    if len(usable) < 4:
        print(f"need at least 4 complete prompts, have {len(usable)}", file=sys.stderr)
        return 2

    max_repeats = min(min(len(e[ep]) for ep in endpoints) for e in usable.values())
    prompt_ids = sorted(usable)
    pairs = [(a, b) for i, a in enumerate(endpoints) for b in endpoints[i + 1 :]]
    is_same = {p: truth[p[0]][1] == truth[p[1]][1] for p in pairs}
    rng = np.random.default_rng(args.seed)

    print(f"{answers.name}: {len(prompt_ids)} prompts x {len(endpoints)} endpoints x up to {max_repeats} repeats")
    print(f"same-model pairs: {sum(is_same.values())}, different-model pairs: {sum(not v for v in is_same.values())}\n")

    print("Learning curve in the number of repeats per prompt:")
    print(f"{'repeats':>8}{'AUROC':>9}{'same':>10}{'diff':>10}{'perm p':>9}")

    last_auc = float("nan")
    for repeats in [2, 3, 4, 6, 8, 10, 12]:
        if repeats > max_repeats:
            break
        scores: dict[tuple[str, str], float] = {}
        for left, right in pairs:
            values = [
                mmd2(usable[p][left][:repeats], usable[p][right][:repeats]) for p in prompt_ids
            ]
            scores[(left, right)] = float(np.nanmean(values))

        same = np.array([scores[p] for p in pairs if is_same[p]])
        diff = np.array([scores[p] for p in pairs if not is_same[p]])
        auc = auroc_lower_is_same(same, diff)
        last_auc = auc

        # Permutation test: shuffle which endpoint is "same", preserving the same/diff counts.
        null = []
        labels = np.array([is_same[p] for p in pairs])
        values = np.array([scores[p] for p in pairs])
        for _ in range(args.permutations):
            perm = rng.permutation(labels)
            a = auroc_lower_is_same(values[perm], values[~perm])
            if np.isfinite(a):
                null.append(a)
        null_arr = np.array(null)
        p_value = float((np.sum(null_arr >= auc) + 1) / (null_arr.size + 1)) if np.isfinite(auc) else float("nan")

        print(f"{repeats:>8}{auc:>9.3f}{np.nanmean(same):>10.4f}{np.nanmean(diff):>10.4f}{p_value:>9.3f}")

    print(
        "\nReading this: AUROC > 0.5 means same-model pairs are closer, as the method assumes. "
        "A p-value above ~0.05 means the separation is not distinguishable from chance at that repeat count."
    )
    if np.isfinite(last_auc) and last_auc < 0.6:
        print("=> even at the largest repeat count, separation is weak; more repeats are unlikely to rescue it.")
    elif np.isfinite(last_auc):
        print("=> separation is emerging with repeats; the curve below shows the cost of getting it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
