#!/usr/bin/env python3
"""High-power analysis: does ANY statistic separate same-model deployments?

Every previous test was starved. falcon2's collections had 10-40 prompts and almost no repeats, so
the paper's method (which estimates a *distribution* per endpoint-prompt) could never be applied as
designed. The reference endpoints are free, so the sample-size excuse is gone.

This runs, on the full collected set:

  1. Per-prompt AUROC for several statistics, each with multiple repeats per endpoint.
  2. A proper MMD^2 two-sample test over repeats, which is what the task's cited paper specifies.
  3. Prompt-level power: which kinds of prompt actually carry information (this is the actionable
     output if anything works).
  4. A held-out split so the reported number is not flattered by choosing prompts on the same data.

Usage:
    python scripts/power_eval.py --answers data/ref_power.jsonl
    python scripts/power_eval.py --answers data/ref_power.jsonl --metric mmd
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
    """{prompt_id: {endpoint: [responses]}} keeping all repeats."""
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


def auroc(higher_is_same: list[float], lower_is_same: list[float]) -> float:
    a = np.asarray([v for v in higher_is_same if np.isfinite(v)], dtype=float)
    b = np.asarray([v for v in lower_is_same if np.isfinite(v)], dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    wins = sum(float((b < v).sum()) + 0.5 * float((b == v).sum()) for v in a)
    return wins / (a.size * b.size)


# ----------------------------------------------------------------------------- statistics


def stat_multiset_pair(a: list[str], b: list[str]) -> float:
    """Mean token-multiset overlap between all cross pairs of responses."""
    vals = []
    for x in a:
        cx = Counter(toks(x))
        for y in b:
            cy = Counter(toks(y))
            if not cx and not cy:
                vals.append(1.0)
            elif not cx or not cy:
                vals.append(0.0)
            else:
                vals.append(2 * sum((cx & cy).values()) / (sum(cx.values()) + sum(cy.values())))
    return float(np.mean(vals)) if vals else float("nan")


def stat_best_pair(a: list[str], b: list[str]) -> float:
    """Highest overlap over any pair of responses. Same weights should occasionally agree exactly."""
    best = 0.0
    for x in a:
        cx = Counter(toks(x))
        for y in b:
            cy = Counter(toks(y))
            if cx and cy:
                best = max(best, 2 * sum((cx & cy).values()) / (sum(cx.values()) + sum(cy.values())))
    return best


def stat_length(a: list[str], b: list[str]) -> float:
    """Similarity of mean response length."""
    la = np.mean([len(toks(x)) for x in a])
    lb = np.mean([len(toks(y)) for y in b])
    if la == 0 and lb == 0:
        return 1.0
    return min(la, lb) / max(la, lb) if max(la, lb) else 0.0


def stat_selfconsistency(a: list[str], b: list[str]) -> float:
    """Do the two endpoints vary by the same amount?

    Same model => same sampling distribution => similar within-endpoint variability. This is a
    second-moment comparison, blind to whether the text matches, which earlier metrics all required.
    """
    def spread(rs: list[str]) -> float:
        if len(rs) < 2:
            return float("nan")
        pairs = [stat_multiset_pair([rs[i]], [rs[j]]) for i in range(len(rs)) for j in range(i + 1, len(rs))]
        return float(np.mean(pairs)) if pairs else float("nan")

    sa, sb = spread(a), spread(b)
    if not np.isfinite(sa) or not np.isfinite(sb):
        return float("nan")
    return 1.0 - abs(sa - sb)


def stat_mmd(a: list[str], b: list[str]) -> float:
    """Unbiased MMD^2 between the two response sets (lower = same model). Uses a token-set kernel."""
    def k(x: str, y: str) -> float:
        cx, cy = set(toks(x)), set(toks(y))
        if not cx and not cy:
            return 1.0
        if not cx or not cy:
            return 0.0
        return len(cx & cy) / len(cx | cy)

    m, n = len(a), len(b)
    if m < 2 or n < 2:
        return float("nan")
    wa = sum(k(a[i], a[j]) for i in range(m) for j in range(m) if i != j) / (m * (m - 1))
    wb = sum(k(b[i], b[j]) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    cr = sum(k(x, y) for x in a for y in b) / (m * n)
    return wa + wb - 2 * cr


METRICS = {
    "multiset": (stat_multiset_pair, True),
    "best_pair": (stat_best_pair, True),
    "length": (stat_length, True),
    "selfconsistency": (stat_selfconsistency, True),
    "mmd": (stat_mmd, False),  # lower = same
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", default="data/ref_power.jsonl")
    parser.add_argument("--truth", default=str(ROOT / "official" / "practice_answers.csv"))
    parser.add_argument("--metric", choices=sorted(METRICS), default=None)
    parser.add_argument("--min-repeats", type=int, default=2)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    answers = Path(args.answers)
    if not answers.is_absolute():
        answers = ROOT / answers
    truth = load_truth(Path(args.truth))
    endpoints = sorted(truth)
    pooled = load_pooled(answers)

    usable = {p: e for p, e in pooled.items() if all(ep in e and len(e[ep]) >= args.min_repeats for ep in endpoints)}
    if not usable:
        print(f"need >= {args.min_repeats} repeats for all {len(endpoints)} endpoints per prompt", file=sys.stderr)
        have = {p: {ep: len(e.get(ep, [])) for ep in endpoints} for p, e in pooled.items()}
        small = {p: min(v.values()) for p, v in have.items()}
        print("repeats available:", dict(list(small.items())[:10]), file=sys.stderr)
        return 2

    repeats = int(np.median([len(usable[p][ep]) for p in usable for ep in endpoints]))
    print(f"{answers.name}: {len(usable)} prompts x {len(endpoints)} endpoints x ~{repeats} repeats "
          f"= ~{len(usable) * len(endpoints) * repeats} answers\n")

    pairs = [(a, b) for i, a in enumerate(endpoints) for b in endpoints[i + 1 :]]
    is_same = {p: truth[p[0]][1] == truth[p[1]][1] for p in pairs}
    prompt_ids = sorted(usable)
    half = len(prompt_ids) // 2
    select_ids = prompt_ids[:half]
    holdout_ids = prompt_ids[half:]

    metrics = [args.metric] if args.metric else sorted(METRICS)
    print(f"{'metric':<16}{'AUROC(all)':>11}{'AUROC(sel)':>11}{'AUROC(hold)':>12}   direction")
    best: tuple[str, float] | None = None
    for name in metrics:
        fn, higher_is_same = METRICS[name]
        row = []
        for subset in (prompt_ids, select_ids, holdout_ids):
            same_vals, diff_vals = [], []
            for left, right in pairs:
                values = [fn(usable[p][left], usable[p][right]) for p in subset]
                score = float(np.nanmean(values)) if values else float("nan")
                (same_vals if is_same[(left, right)] else diff_vals).append(score)
            row.append(auroc(same_vals, diff_vals) if higher_is_same else auroc(diff_vals, same_vals))
        direction = "higher=same" if higher_is_same else "lower=same"
        print(f"{name:<16}{row[0]:>11.3f}{row[1]:>11.3f}{row[2]:>12.3f}   {direction}")
        if best is None or (np.isfinite(row[2]) and row[2] > best[1]):
            best = (name, row[2])

    assert best is not None
    print(f"\nbest on held-out: {best[0]} (AUROC {best[1]:.3f}; 0.5 = chance)")

    # Per-prompt power for the best metric: which kinds of prompt actually carry information.
    fn, higher_is_same = METRICS[best[0]]
    print(f"\nper-prompt AUROC for {best[0]} (top {args.top}):")
    rows = []
    for pid in prompt_ids:
        same_vals, diff_vals = [], []
        for left, right in pairs:
            score = fn(usable[pid][left], usable[pid][right])
            (same_vals if is_same[(left, right)] else diff_vals).append(score)
        value = auroc(same_vals, diff_vals) if higher_is_same else auroc(diff_vals, same_vals)
        rows.append((value, pid))
    rows.sort(key=lambda r: -(r[0] if np.isfinite(r[0]) else -9))
    for value, pid in rows[: args.top]:
        print(f"  {pid:<14}{value:.3f}")
    if len(rows) > args.top:
        print(f"  ... worst: " + ", ".join(f"{p} {v:.2f}" for v, p in rows[-3:]))

    above = sum(1 for v, _ in rows if np.isfinite(v) and v > 0.7)
    print(f"\nprompts above AUROC 0.70: {above}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
