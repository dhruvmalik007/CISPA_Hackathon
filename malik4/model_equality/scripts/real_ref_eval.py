#!/usr/bin/env python3
"""Evaluate a method against REAL reference responses collected by the team.

The simulator encodes the published trap structure but cannot reproduce real model text. This runs
the production decision path on real API responses, which is the only honest test of the thresholds
before spending the scored budget.

Inputs are collected `ref_*.jsonl` files (one JSON object per answer with `endpoint`, `id`,
`repeat`, `response`). Ground truth comes from `practice_answers.csv`.

Reports AUROC explicitly, because unlike MCC it needs no clustering and directly answers the
question "does this similarity separate same-model from different-model pairs, and in which
direction". AUROC < 0.5 means the metric is inverted.

Usage:
    python scripts/real_ref_eval.py --answers data/falcon2/ref_p2.jsonl
    python scripts/real_ref_eval.py --answers data/falcon2/ref_calib.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from met import stats  # noqa: E402


def load_real_answers(path: Path) -> dict[str, dict[str, str]]:
    """Group real answers as {prompt_id: {endpoint: response}}."""
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
    """P(random same-model pair scores above a random different-model pair).

    1.0 perfect, 0.5 useless, <0.5 inverted.
    """
    if same.size == 0 or diff.size == 0:
        return float("nan")
    wins = 0.0
    for value in same:
        wins += float((diff < value).sum()) + 0.5 * float((diff == value).sum())
    return wins / (same.size * diff.size)


def build_labels(
    tensors: dict[tuple[str, str], np.ndarray],
    endpoints: list[str],
    seed: int,
    fdr: float,
    floor: float,
) -> tuple[dict[str, tuple[str, str]], dict[tuple[str, str], tuple[float, float]]]:
    """Union-find over significant, sufficiently-similar edges.

    This practice set is a single family, so one level suffices: every significant edge joins two
    endpoints belonging to the same model.
    """
    rng = np.random.default_rng(seed)
    p_values: dict[tuple[str, str], float] = {}
    for pair, values in tensors.items():
        _observed, p_value = stats.paired_permutation_test(values, rng)
        p_values[pair] = p_value

    pairs = sorted(p_values)
    adjusted = stats.benjamini_hochberg([p_values[p] for p in pairs], fdr=fdr)

    parent = {ep: ep for ep in endpoints}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges: dict[tuple[str, str], tuple[float, float]] = {}
    for pair, adj_p in zip(pairs, adjusted):
        values = tensors[pair]
        finite = values[np.isfinite(values)]
        mean = float(finite.mean()) if finite.size else 0.0
        edges[pair] = (mean, float(adj_p))
        if adj_p <= fdr and mean >= floor:
            ra, rb = find(pair[0]), find(pair[1])
            if ra != rb:
                parent[rb] = ra

    components: dict[str, list[str]] = defaultdict(list)
    for ep in endpoints:
        components[find(ep)].append(ep)

    labels: dict[str, tuple[str, str]] = {}
    for index, (_root, members) in enumerate(sorted(components.items(), key=lambda kv: sorted(kv[1]))):
        family = f"F{index}"
        model = f"F{index}_M0" if len(members) > 1 else f"F{index}_M_{sorted(members)[0]}"
        for ep in members:
            labels[ep] = (family, model)
    return labels, edges


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--truth", default=str(ROOT / "official" / "practice_answers.csv"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--floor", type=float, default=0.50)
    parser.add_argument("--max-prompts", type=int, default=None, help="cap prompts, to see the effect of n")
    args = parser.parse_args()

    answers_path = Path(args.answers)
    if not answers_path.is_absolute():
        answers_path = ROOT / answers_path
    truth = load_truth(Path(args.truth))
    endpoints = sorted(truth)

    data_all = load_real_answers(answers_path)
    paired = {pid: rows for pid, rows in data_all.items() if all(ep in rows for ep in endpoints)}
    prompt_ids = sorted(paired)
    if args.max_prompts:
        prompt_ids = prompt_ids[: args.max_prompts]
    if not prompt_ids:
        print(
            f"no fully-paired prompts in {answers_path.name}: "
            f"{len(data_all)} ids, endpoints needing {len(endpoints)}",
            file=sys.stderr,
        )
        return 2

    data = {pid: paired[pid] for pid in prompt_ids}
    print(f"{answers_path.name}: {len(prompt_ids)} fully-paired prompts x {len(endpoints)} endpoints")

    tensors = stats.build_similarity_tensor(data, endpoints)

    same_scores = []
    diff_scores = []
    for (left, right), values in sorted(tensors.items()):
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        mean = float(finite.mean())
        (same_scores if truth[left][1] == truth[right][1] else diff_scores).append(mean)

    same = np.asarray(same_scores)
    diff = np.asarray(diff_scores)
    auc = auroc(same, diff)

    print(
        f"\nAUROC {auc:.3f}  (1.0 perfect, 0.5 useless, <0.5 inverted)\n"
        f"  same-model   mean {same.mean():.4f}  n={same.size}\n"
        f"  different    mean {diff.mean():.4f}  n={diff.size}"
    )
    if auc < 0.5:
        print("  WARNING: inverted - higher similarity means MORE likely different.")

    labels, edges = build_labels(tensors, endpoints, args.seed, args.fdr, args.floor)
    truth_pairs = {ep: (family, model) for ep, (family, model) in truth.items()}
    score = stats.competition_score(truth_pairs, labels, endpoints)

    correct = 0
    print(f"\n{'pair':<16}{'truth':<6}{'mean_sim':>9}{'adj_p':>8}{'called':>8}  ok")
    for pair in sorted(edges, key=lambda p: -edges[p][0]):
        mean, adj_p = edges[pair]
        is_same = truth[pair[0]][1] == truth[pair[1]][1]
        called_same = labels[pair[0]][1] == labels[pair[1]][1]
        ok = called_same == is_same
        correct += int(ok)
        print(
            f"{pair[0] + '/' + pair[1]:<16}{'SAME' if is_same else 'DIFF':<6}{mean:>9.3f}"
            f"{adj_p:>8.4f}{'SAME' if called_same else 'DIFF':>8}  {'OK' if ok else 'XX'}"
        )

    print(f"\ncorrect: {correct}/{len(edges)}")
    print(f"MCC_model {score['mcc_model']:.3f}  MCC_family {score['mcc_family']:.3f}  score {score['score']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
