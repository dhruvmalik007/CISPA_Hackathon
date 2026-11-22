#!/usr/bin/env python3
"""Per-prompt constrained-output evaluation for round 3.

Each prompt is scored independently: the distance between two endpoints is computed only from
responses to that prompt, then AUROC is measured over endpoint pairs. Prompts that do not separate
same-model from different-model reference endpoints are discarded.

Usage:
    python scripts/round3_eval.py --answers data/ref_power.jsonl
    python scripts/round3_eval.py --answers data/ref_power.jsonl --top 10 --threshold 0.80
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = [f"ref-{i:02d}" for i in range(1, 10)]
METRICS = ("exact", "edit", "char3", "composite")


def load_pooled(path: Path, prefix: str = "ref-") -> dict[str, dict[str, list[str]]]:
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
            prompt = row.get("id")
            response = row.get("response")
            if endpoint.startswith(prefix) and prompt is not None and response is not None:
                pooled[str(prompt)][endpoint].append(str(response))
    return {k: dict(v) for k, v in pooled.items()}


def load_truth(path: Path) -> dict[str, tuple[str, str]]:
    truth: dict[str, tuple[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            truth[row["endpoint"]] = (row["family"], row["model"])
    return truth


def normalise(text: str) -> str:
    value = str(text).lower().strip()
    value = re.sub(r"\s+", " ", value)
    value = value.strip("`'\" ")
    value = re.sub(r"[.!?,;:]+$", "", value)
    return value


def exact_similarity(a: str, b: str) -> float:
    return 1.0 if normalise(a) == normalise(b) else 0.0


def edit_similarity(a: str, b: str) -> float:
    left, right = normalise(a), normalise(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def char_ngrams(text: str, n: int = 3) -> set[str]:
    value = normalise(text)
    if len(value) < n:
        return {value} if value else set()
    return {value[i : i + n] for i in range(len(value) - n + 1)}


def char_kernel(a: str, b: str) -> float:
    left, right = char_ngrams(a), char_ngrams(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def mmd2(a: list[str], b: list[str]) -> float:
    m, n = len(a), len(b)
    if m < 2 or n < 2:
        return float("nan")
    within_a = sum(char_kernel(a[i], a[j]) for i in range(m) for j in range(m) if i != j) / (m * (m - 1))
    within_b = sum(char_kernel(b[i], b[j]) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    cross = sum(char_kernel(x, y) for x in a for y in b) / (m * n)
    return within_a + within_b - 2 * cross


def pair_distances(per_endpoint: dict[str, list[str]], endpoints: list[str]) -> dict[str, dict[tuple[str, str], float]]:
    out: dict[str, dict[tuple[str, str], float]] = {name: {} for name in METRICS}
    for i, left in enumerate(endpoints):
        for right in endpoints[i + 1 :]:
            a, b = per_endpoint[left], per_endpoint[right]
            exact = 1.0 - float(np.mean([exact_similarity(x, y) for x in a for y in b]))
            edit = 1.0 - float(np.mean([edit_similarity(x, y) for x in a for y in b]))
            char3 = mmd2(a, b)
            out["exact"][(left, right)] = exact
            out["edit"][(left, right)] = edit
            out["char3"][(left, right)] = char3
            values = [v for v in (exact, edit, char3) if np.isfinite(v)]
            out["composite"][(left, right)] = float(np.mean(values)) if values else float("nan")
    return out


def auroc(diff: list[float], same: list[float]) -> float:
    a = np.asarray([v for v in diff if np.isfinite(v)], dtype=float)
    b = np.asarray([v for v in same if np.isfinite(v)], dtype=float)
    if not a.size or not b.size:
        return float("nan")
    wins = sum(float((a > v).sum()) + 0.5 * float((a == v).sum()) for v in b)
    return wins / (a.size * b.size)


def cluster(grid: np.ndarray, tau: float) -> list[int]:
    n = grid.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if np.isfinite(grid[i, j]) and grid[i, j] <= tau:
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
        for endpoint in members:
            labels[endpoint] = ("P", f"M{index}")
    return labels


def score_with_official(labels: dict[str, tuple[str, str]], endpoints: list[str]) -> float:
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "answer.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["endpoint", "family", "model"])
            for endpoint in endpoints:
                writer.writerow([endpoint, *labels[endpoint]])
        reply = subprocess.run(
            [sys.executable, str(ROOT / "official" / "score_practice.py"), str(path)],
            capture_output=True,
            text=True,
        )
    for line in reply.stdout.splitlines():
        if line.startswith("Practice score"):
            return float(line.rsplit(":", 1)[1].strip())
    return float("nan")


def average_grid(
    distances: dict[str, dict[tuple[str, str], float]],
    selected: list[str],
    endpoints: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    n = len(endpoints)
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for prompt in selected:
        for pair, distance in distances[prompt].items():
            if np.isfinite(distance):
                values[pair].append(distance)
    grid = np.eye(n)
    counts = np.zeros((n, n), dtype=int)
    for i, left in enumerate(endpoints):
        for j in range(i + 1, n):
            pair = (left, endpoints[j])
            value = float(np.mean(values[pair])) if values[pair] else float("nan")
            grid[i, j] = grid[j, i] = value
            counts[i, j] = counts[j, i] = len(values[pair])
    return grid, counts


def practice_mcc(labels: dict[str, tuple[str, str]], endpoints: list[str], truth: dict[str, tuple[str, str]]) -> float:
    tp = tn = fp = fn = 0
    for i, left in enumerate(endpoints):
        for right in endpoints[i + 1 :]:
            actual = truth[left] == truth[right]
            called = labels[left] == labels[right]
            if actual and called:
                tp += 1
            elif actual:
                fn += 1
            elif called:
                fp += 1
            else:
                tn += 1
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return max(0.0, (tp * tn - fp * fn) / denominator) if denominator else 0.0


def best_threshold(
    grid: np.ndarray,
    endpoints: list[str],
    truth: dict[str, tuple[str, str]],
) -> tuple[float, float, int]:
    off = grid[~np.eye(grid.shape[0], dtype=bool)]
    finite = off[np.isfinite(off)]
    if finite.size == 0:
        return float("nan"), float("nan"), 0
    candidates = sorted({float(v) for v in finite})
    candidates += [
        float(np.quantile(finite, q))
        for q in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    ]
    best = (float("nan"), -1.0, 0)
    for tau in sorted(set(candidates)):
        roots = cluster(grid, tau)
        groups = len(set(roots))
        if groups < 2 or groups >= len(endpoints):
            continue
        score = practice_mcc(labels_from(roots, endpoints), endpoints, truth)
        if np.isfinite(score) and score > best[1]:
            best = (tau, score, groups)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", default="data/ref_power.jsonl")
    parser.add_argument("--truth", default=str(ROOT / "official" / "practice_answers.csv"))
    parser.add_argument("--min-repeats", type=int, default=2)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--out", default="out/round3_eval.json")
    args = parser.parse_args()

    answers = Path(args.answers)
    if not answers.is_absolute():
        answers = ROOT / answers
    truth = load_truth(Path(args.truth))
    endpoints = sorted(truth)
    pooled = load_pooled(answers)
    usable = {
        pid: per_ep
        for pid, per_ep in pooled.items()
        if all(ep in per_ep and len(per_ep[ep]) >= args.min_repeats for ep in endpoints)
    }
    if not usable:
        print(f"no prompts with >= {args.min_repeats} repeats on all endpoints", file=sys.stderr)
        return 2

    same_pairs = {
        (a, b)
        for i, a in enumerate(endpoints)
        for b in endpoints[i + 1 :]
        if truth[a][1] == truth[b][1]
    }
    all_pairs = [
        (a, b)
        for i, a in enumerate(endpoints)
        for b in endpoints[i + 1 :]
    ]
    prompt_rows: dict[str, dict] = {}
    for pid in sorted(usable):
        distances = pair_distances(usable[pid], endpoints)
        row = {"id": pid, "scores": {}, "mcc": {}, "distances": distances}
        for metric in METRICS:
            same = [distances[metric][pair] for pair in all_pairs if pair in same_pairs]
            diff = [distances[metric][pair] for pair in all_pairs if pair not in same_pairs]
            row["scores"][metric] = auroc(diff, same)
            grid = np.eye(len(endpoints))
            for i, left in enumerate(endpoints):
                for j, right in enumerate(endpoints[i + 1 :], start=i + 1):
                    value = distances[metric].get((left, right), float("nan"))
                    grid[i, j] = grid[j, i] = value
            _tau, mcc, _groups = best_threshold(grid, endpoints, truth)
            row["mcc"][metric] = mcc
        prompt_rows[pid] = row

    print(f"{answers.name}: {len(usable)} complete prompts x {len(endpoints)} endpoints")
    print(f"same-model pairs: {len(same_pairs)}, different-model pairs: {len(all_pairs) - len(same_pairs)}\n")
    print(f"{'prompt':<14}" + "".join(f"{m:>10}" for m in METRICS))
    for pid in sorted(prompt_rows):
        scores = prompt_rows[pid]["scores"]
        print(f"{pid:<14}" + "".join(f"{scores[m]:>10.3f}" if np.isfinite(scores[m]) else f"{'nan':>10}" for m in METRICS))

    print("\nper-prompt best thresholded MCC (top 20):")
    mcc_ranked = sorted(
        prompt_rows,
        key=lambda pid: -max(
            (prompt_rows[pid]["mcc"][m] for m in METRICS if np.isfinite(prompt_rows[pid]["mcc"][m])),
            default=-1,
        ),
    )
    for pid in mcc_ranked[:20]:
        values = prompt_rows[pid]["mcc"]
        print(
            f"  {pid:<14}"
            + "".join(f"{values[m]:>10.3f}" if np.isfinite(values[m]) else f"{'nan':>10}" for m in METRICS)
        )

    selection: dict[str, dict] = {}
    for metric in METRICS:
        ranked = sorted(
            prompt_rows,
            key=lambda pid: -(prompt_rows[pid]["scores"][metric] if np.isfinite(prompt_rows[pid]["scores"][metric]) else -1),
        )
        passing = [pid for pid in ranked if prompt_rows[pid]["scores"][metric] >= args.threshold]
        for k in (5, 8, 10):
            chosen = ranked[: min(k, len(ranked))]
            grid, _counts = average_grid(
                {pid: prompt_rows[pid]["distances"][metric] for pid in chosen},
                chosen,
                endpoints,
            )
            tau, practice, groups = best_threshold(grid, endpoints, truth)
            key = f"{metric}_top{k}"
            selection[key] = {
                "metric": metric,
                "top_k": k,
                "prompts": chosen,
                "auroc": {pid: prompt_rows[pid]["scores"][metric] for pid in chosen},
                "threshold": tau,
                "practice_score": practice,
                "groups": groups,
                "above_threshold": len(passing),
            }
            print(
                f"\n{key}: {len(passing)} prompts >= {args.threshold}; selected {len(chosen)}; "
                f"threshold {tau:.4f}; practice MCC {practice:.3f}; groups {groups}"
            )
            print("  " + ", ".join(chosen))

    best_key = max(
        selection,
        key=lambda key: selection[key]["practice_score"]
        if np.isfinite(selection[key]["practice_score"])
        else -1,
    )
    best = selection[best_key]
    print(
        f"\nbest: {best_key} practice MCC {best['practice_score']:.3f} "
        f"with {len(best['prompts'])} prompts"
    )

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "answers": str(answers),
        "prompts": len(usable),
        "endpoints": endpoints,
        "metrics": selection,
        "best_metric": best["metric"],
        "best_key": best_key,
        "best": best,
        "per_prompt": {
            pid: {"scores": row["scores"], "mcc": row["mcc"]}
            for pid, row in prompt_rows.items()
        },
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
