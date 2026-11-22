#!/usr/bin/env python3
"""MET-style kernel MMD over pooled sample sets, tested on real reference answers.

Why this is different from every metric tried so far. Those compared one response from A with one
response from B per prompt, then averaged. That is a *pointwise* comparison, and it is defeated by
any endpoint whose sampling variance is high: `ref-08` is a same-model deployment at temperature 1.0
and its own repeats agree only 0.20 with each other, so any pointwise metric reads it as "different".

The paper's method (Gao et al., arXiv:2410.20247) instead compares *distributions*: pool the k
samples produced by each endpoint for a prompt into a set, and measure the maximum mean discrepancy
between the two sets with an unbiased MMD^2 estimator. High within-endpoint variance then inflates
both sets equally, instead of being read as evidence of difference. This is exactly why the paper
spends budget on repeats rather than on more prompts.

Estimator (unbiased, Gao et al. eq. 4):

    MMD^2_u(A, B) = 1/(m(m-1)) sum_{i!=j} k(a_i,a_j)
                  + 1/(n(n-1)) sum_{i!=j} k(b_i,b_j)
                  - 2/(mn)     sum_{i,j}  k(a_i,b_j)

A same-model pair should give MMD^2_u ~ 0; different models give a clearly positive value. So
"same model" means SMALL here - the opposite direction from the Jaccard metrics, which is the
distinction that broke the earlier attempt.

Usage:
    python scripts/mmd_eval.py --answers data/falcon2/ref_calib.jsonl
    python scripts/mmd_eval.py --answers data/falcon2/ref_calib.jsonl --kernel char4
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


def load_pooled(path: Path) -> dict[str, dict[str, list[str]]]:
    """{prompt_id: {endpoint: [responses]}} - all repeats kept."""
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


def toks(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def kernel_char4(a: str, b: str) -> float:
    na, nb = " ".join(a.lower().split()), " ".join(b.lower().split())
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    ga = {na[i : i + 4] for i in range(max(1, len(na) - 3))}
    gb = {nb[i : i + 4] for i in range(max(1, len(nb) - 3))}
    union = ga | gb
    return len(ga & gb) / len(union) if union else 1.0


def kernel_token_bag(a: str, b: str) -> float:
    ca, cb = Counter(toks(a)), Counter(toks(b))
    if not ca and not cb:
        return 1.0
    if not ca or not cb:
        return 0.0
    return sum((ca & cb).values()) / max(1, min(sum(ca.values()), sum(cb.values())))


KERNELS = {"char4": kernel_char4, "token": kernel_token_bag}


def mmd2_unbiased(a: list[str], b: list[str], kernel) -> float:
    m, n = len(a), len(b)
    if m < 2 or n < 2:
        return float("nan")
    within_a = sum(kernel(a[i], a[j]) for i in range(m) for j in range(m) if i != j) / (m * (m - 1))
    within_b = sum(kernel(b[i], b[j]) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    cross = sum(kernel(x, y) for x in a for y in b) / (m * n)
    return within_a + within_b - 2 * cross


def auroc_lower_is_same(same: np.ndarray, diff: np.ndarray) -> float:
    """P(same-model distance < different-model distance). 1 = perfect, 0.5 = chance."""
    same = same[np.isfinite(same)]
    diff = diff[np.isfinite(diff)]
    if same.size == 0 or diff.size == 0:
        return float("nan")
    wins = sum(float((diff > v).sum()) + 0.5 * float((diff == v).sum()) for v in same)
    return wins / (same.size * diff.size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--truth", default=str(ROOT / "official" / "practice_answers.csv"))
    parser.add_argument("--kernel", choices=sorted(KERNELS), default="token")
    parser.add_argument("--min-repeats", type=int, default=2)
    parser.add_argument("--pool", choices=["prompt", "all"], default="prompt", help="pool per prompt, or all prompts together")
    args = parser.parse_args()

    answers_path = Path(args.answers)
    if not answers_path.is_absolute():
        answers_path = ROOT / args.answers
    truth = load_truth(Path(args.truth))
    endpoints = sorted(truth)
    pooled = load_pooled(answers_path)
    kernel = KERNELS[args.kernel]

    usable = {
        pid: eps
        for pid, eps in pooled.items()
        if all(ep in eps and len(eps[ep]) >= args.min_repeats for ep in endpoints)
    }
    if not usable:
        counts = {
            pid: min(len(eps.get(ep, [])) for ep in endpoints) for pid, eps in pooled.items()
        }
        print(f"no prompt has >= {args.min_repeats} repeats for all endpoints; have {counts}", file=sys.stderr)
        return 2

    print(f"{answers_path.name}: {len(usable)} prompts with >= {args.min_repeats} repeats each")
    repeats = {ep: [len(eps[ep]) for eps in usable.values()] for ep in endpoints}
    print(f"  repeats per endpoint: min {min(min(v) for v in repeats.values())} "
          f"max {max(max(v) for v in repeats.values())}\n")

    pairs = [(a, b) for i, a in enumerate(endpoints) for b in endpoints[i + 1 :]]
    rows = []
    for left, right in pairs:
        if args.pool == "prompt":
            values = [
                mmd2_unbiased(eps[left], eps[right], kernel) for eps in usable.values()
            ]
            dist = float(np.nanmean(values))
        else:
            a_all = [r for eps in usable.values() for r in eps[left]]
            b_all = [r for eps in usable.values() for r in eps[right]]
            dist = mmd2_unbiased(a_all, b_all, kernel)
        rows.append((left, right, dist, truth[left][1] == truth[right][1]))

    same = np.array([d for _a, _b, d, s in rows if s])
    diff = np.array([d for _a, _b, d, s in rows if not s])
    auc = auroc_lower_is_same(same, diff)

    print(f"{'pair':<16}{'truth':<6}{'MMD2_u':>10}")
    for left, right, dist, is_same in sorted(rows, key=lambda r: (not r[3], r[2])):
        print(f"{left + '/' + right:<16}{'SAME' if is_same else 'DIFF':<6}{dist:>10.4f}")

    print(
        f"\nAUROC (same-model distance LOWER) {auc:.3f}   "
        f"same mean {np.nanmean(same):.4f}   different mean {np.nanmean(diff):.4f}"
    )
    if auc < 0.5:
        print("WARNING: inverted even for MMD - same-model pairs are FURTHER apart.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
