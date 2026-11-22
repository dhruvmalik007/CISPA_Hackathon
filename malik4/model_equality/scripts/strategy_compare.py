#!/usr/bin/env python3
"""Choose the submission strategy by MEASURING each candidate on the reference endpoints.

The trivial answers are known dead ends: 'all distinct' and 'all same' both score exactly 0, because
each sets one factor of `tp*tn - fp*fn` to zero. So the only way to score anything is to make some
pairs right and more of them right than wrong. That needs a decision rule whose value has been
measured, not assumed.

This script does the measurement honestly:

  * for each candidate rule, build the labels using ONLY similarity structure,
  * score with the organisers' own `score_practice.py` on `ref-01..ref-09` (all one family there,
    so this measures the 90% model term, which is the part that matters),
  * report the score, so the rule that is submitted is the one that actually won.

A rule that scores 0 on reference data is not submitted. That is the whole point: it turns "pick a
method" into a measured comparison, and it is the check that `1st-try.md` never ran.

Usage:
    python scripts/strategy_compare.py --answers data/ref_power.jsonl
    python scripts/strategy_compare.py --answers data/ref_power.jsonl --apply data/answers.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_pooled(path: Path, prefix: str):
    """{prompt: {endpoint: [responses]}} - repeats kept, because the comparison is per prompt."""
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
            if not endpoint.startswith(prefix) or row.get("id") is None or row.get("response") is None:
                continue
            pooled[str(row["id"])][endpoint].append(str(row["response"]))
    return pooled


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def pair_similarity(pooled, endpoints, left: str, right: str, per_prompt: bool = True) -> float:
    """Mean similarity between two endpoints.

    With repeats, every cross pair within a prompt is compared, so sampling noise is averaged out
    without ever mixing prompts. Mixing prompts is what produced negative MMD^2 earlier: responses to
    different questions are dissimilar regardless of the model, which swamps the weights signal.
    """
    values = []
    for _prompt, rows in pooled.items():
        if left not in rows or right not in rows:
            continue
        if per_prompt:
            for x in rows[left]:
                tx = tokens(x)
                for y in rows[right]:
                    values.append(jaccard(tx, tokens(y)))
        else:
            values.append(jaccard(tokens(rows[left][0]), tokens(rows[right][0])))
    return float(np.mean(values)) if values else float("nan")


def similarity_matrix(pooled, endpoints) -> np.ndarray:
    n = len(endpoints)
    grid = np.full((n, n), np.nan)
    for i, left in enumerate(endpoints):
        grid[i, i] = 1.0
        for j in range(i + 1, n):
            value = pair_similarity(pooled, endpoints, left, endpoints[j])
            grid[i, j] = grid[j, i] = value
    return grid


def cluster(grid: np.ndarray, tau: float) -> list[int]:
    """Union-find over pairs whose similarity is at least tau."""
    n = grid.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if np.isfinite(grid[i, j]) and grid[i, j] >= tau:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    return [find(i) for i in range(n)]


def labels_from(roots: list[int], endpoints: list[str]) -> dict[str, tuple[str, str]]:
    groups: dict[int, list[str]] = defaultdict(list)
    for root, endpoint in zip(roots, endpoints):
        groups[root].append(endpoint)
    ordered = sorted(groups.values(), key=lambda members: sorted(members))
    labels: dict[str, tuple[str, str]] = {}
    for index, members in enumerate(ordered, start=1):
        family = f"F{index}"
        for endpoint in members:
            labels[endpoint] = (family, f"{family}_M1")
    return labels


def score_with_official(labels, endpoints, truth_path: Path) -> float:
    """Run the organisers' scorer unmodified and parse the MCC it prints."""
    rows = [(ep, *labels[ep]) for ep in endpoints]
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "answer.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["endpoint", "family", "model"])
            writer.writerows(rows)
        reply = subprocess.run(
            [sys.executable, str(ROOT / "official" / "score_practice.py"), str(path), str(truth_path)],
            capture_output=True,
            text=True,
        )
    for line in reply.stdout.splitlines():
        if line.startswith("Practice score"):
            try:
                return float(line.rsplit(":", 1)[1].strip())
            except (IndexError, ValueError):
                return float("nan")
    return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", default="data/ref_power.jsonl")
    parser.add_argument("--truth", default=str(ROOT / "official" / "practice_answers.csv"))
    parser.add_argument("--apply", default=None, help="scored answers file to label with the winning rule")
    parser.add_argument("--out", default="out/submission.csv")
    parser.add_argument("--top-k", type=int, default=4, help="for the outlier rule: how many endpoints to set apart")
    args = parser.parse_args()

    answers = Path(args.answers)
    if not answers.is_absolute():
        answers = ROOT / answers
    truth_path = Path(args.truth)
    pooled = load_pooled(answers, "ref-")
    endpoints = sorted({ep for rows in pooled.values() for ep in rows})
    complete = {p: r for p, r in pooled.items() if all(ep in r for ep in endpoints)}
    if not complete or len(endpoints) < 3:
        print(f"need complete prompts across >=3 ref endpoints; have {len(endpoints)} endpoints", file=sys.stderr)
        return 2
    print(f"calibration: {answers.name} - {len(complete)} prompts x {len(endpoints)} endpoints\n")

    grid = similarity_matrix(complete, endpoints)
    off = grid[~np.eye(grid.shape[0], dtype=bool)]
    off = off[np.isfinite(off)]
    print(f"pairwise similarity: min {off.min():.3f} median {np.median(off):.3f} max {off.max():.3f}\n")

    candidates: list[tuple[str, dict[str, tuple[str, str]]]] = []

    # 1. every endpoint its own group
    candidates.append(("all-distinct", labels_from(list(range(len(endpoints))), endpoints)))
    # 2. everything in one group
    candidates.append(("all-same", labels_from([0] * len(endpoints), endpoints)))
    # 3. threshold sweeps - this is the family of rules 1st-try.md proposes
    for tau in np.percentile(off, [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]):
        candidates.append((f"threshold>={tau:.3f}", labels_from(cluster(grid, float(tau)), endpoints)))
    # 4. outlier rule: set apart the endpoints least similar to the crowd
    centrality = np.array([np.nanmean(np.delete(grid[i], i)) for i in range(len(endpoints))])
    order = list(np.argsort(centrality))  # least similar first
    for k in range(1, min(args.top_k, len(endpoints) - 1) + 1):
        roots = [0 if i not in set(order[:k]) else i + 1 for i in range(len(endpoints))]
        candidates.append((f"outlier-split-{k}", labels_from(roots, endpoints)))

    print(f"{'rule':<22}{'groups':>8}{'MCC':>8}")
    results = []
    for name, labels in candidates:
        value = score_with_official(labels, endpoints, truth_path)
        groups = len({v[0] for v in labels.values()})
        results.append((value, name, labels, groups))
        print(f"{name:<22}{groups:>8}{value:>8.3f}")

    results.sort(key=lambda r: (-(r[0] if np.isfinite(r[0]) else -1), r[1]))
    best_value, best_name, best_labels, best_groups = results[0]
    print(f"\nwinner: {best_name}  MCC {best_value:.3f}  ({best_groups} groups)")

    if best_value <= 0:
        print(
            "\nNo rule scores above 0 on reference data. That means every candidate labels the pairs no\n"
            "better than chance, and submitting any of them is expected to score 0. This is a real\n"
            "finding, not a bug: the 90% model term is not recoverable from text comparison here."
        )
    if args.apply and best_value > 0:
        scored = load_pooled(Path(args.apply), "ep-")
        targets = sorted({ep for rows in scored.values() for ep in rows})
        keep = {p: r for p, r in scored.items() if all(ep in r for ep in targets)}
        target_grid = similarity_matrix(keep, targets)
        target_endpoints = targets
        # Re-derive the winning rule on the scored endpoints.
        if best_name == "all-distinct":
            roots = list(range(len(target_endpoints)))
        elif best_name == "all-same":
            roots = [0] * len(target_endpoints)
        elif best_name.startswith("threshold>="):
            tau = float(best_name.split(">=")[1])
            roots = cluster(target_grid, tau)
        else:
            k = int(best_name.rsplit("-", 1)[1])
            centrality_target = np.array(
                [np.nanmean(np.delete(target_grid[i], i)) for i in range(len(target_endpoints))]
            )
            order_t = list(np.argsort(centrality_target))
            roots = [0 if i not in set(order_t[:k]) else i + 1 for i in range(len(target_endpoints))]
        labels = labels_from(roots, target_endpoints)
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["endpoint", "family", "model"])
            for ep in target_endpoints:
                writer.writerow([ep, *labels[ep]])
        print(f"\napplied '{best_name}' to {args.apply} -> {out} ({len(target_endpoints)} rows)")
    elif args.apply:
        print(f"\nnot applying to {args.apply}: no rule beat chance, so any file would be noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
